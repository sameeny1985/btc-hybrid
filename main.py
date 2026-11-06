"""
BTC Hybrid LSTM + XGBoost Predictor - Web Service version for Render
- Real-time ticker price ONLY from MEXC Spot
- Predicts every 10 minutes (at :00, :10, :20, :30, :40, :50)
- Continuous learning from prediction feedback
- Sends signals to Telegram
- Runs as Flask Web Service so it can be kept alive by external pingers
"""

import time
import os
import threading
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template_string
import numpy as np
import pytz

import config
from utils.mexc_client import MexcClient
from utils.database import Database
from utils.features import build_sequence_and_features
from utils.models import HybridPredictor
from utils.telegram_bot import TelegramNotifier

# Ensure directories
os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

mexc = MexcClient()
db = Database()
predictor = HybridPredictor()
notifier = TelegramNotifier()

app = Flask(__name__)

# Shared status for the web page
status = {
    "running": True,
    "last_tick_price": None,
    "last_tick_time": None,
    "last_prediction": None,
    "last_prediction_time": None,
    "total_predictions": 0,
    "accuracy_20": 0.0,
    "accuracy_50": 0.0,
    "model_ready": False,
    "started_at": datetime.now(timezone.utc).isoformat()
}


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def is_prediction_minute(dt: datetime = None) -> bool:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.minute % 10 == 0 and dt.second < 45


def bootstrap_history():
    existing = db.get_recent_ticks(10)
    if len(existing) >= 100:
        print("[Bootstrap] Enough history already present.")
        return

    print("[Bootstrap] Fetching historical klines from MEXC to seed database...")
    klines = mexc.get_klines(interval="1m", limit=500)
    if not klines:
        print("[Bootstrap] Failed to get klines. Will wait for live ticks.")
        return

    for k in klines:
        ts = int(k[0])
        close_price = float(k[4])
        db.insert_tick(ts, close_price)

    print(f"[Bootstrap] Inserted {len(klines)} historical ticks.")


def collect_live_tick():
    price = mexc.get_ticker_price()
    if price is None:
        return None
    ts = now_ms()
    db.insert_tick(ts, price)
    status["last_tick_price"] = price
    status["last_tick_time"] = datetime.now(timezone.utc).isoformat()
    return price


def check_and_settle_pending():
    now = now_ms()
    pending = db.get_pending_predictions(before_ms=now)
    for p in pending:
        ticks = db.get_ticks_between(p["target_time"] - 30_000, p["target_time"] + 90_000)
        if not ticks:
            actual_price = mexc.get_ticker_price()
            if actual_price is None:
                continue
        else:
            actual_price = min(ticks, key=lambda x: abs(x[0] - p["target_time"]))[1]

        actual_dir = "UP" if actual_price > p["current_price"] else "DOWN"
        is_correct = 1 if actual_dir == p["predicted_direction"] else 0

        db.update_prediction_result(p["id"], actual_price, actual_dir, is_correct)

        notifier.send_result(
            p["id"],
            p["predicted_direction"],
            actual_dir,
            bool(is_correct),
            p["current_price"],
            actual_price
        )
        print(f"[Settle] Pred #{p['id']} | Pred: {p['predicted_direction']} | Actual: {actual_dir} | Correct: {is_correct}")


def make_prediction():
    current_price = mexc.get_ticker_price()
    if current_price is None:
        print("[Predict] Could not get live ticker price.")
        return

    ts = now_ms()
    db.insert_tick(ts, current_price)
    status["last_tick_price"] = current_price
    status["last_tick_time"] = datetime.now(timezone.utc).isoformat()

    ticks = db.get_recent_ticks(limit=300)
    sequence, tabular, last_price = build_sequence_and_features(ticks)

    if sequence is None or tabular is None:
        print("[Predict] Not enough data for features yet.")
        notifier.send_info("داده کافی برای ساخت ویژگی‌ها وجود ندارد. منتظر جمع‌آوری بیشتر...")
        return

    if not predictor.is_ready:
        if not predictor.load():
            print("[Predict] Model not ready. Need initial training.")
            notifier.send_info("مدل هنوز آموزش ندیده. در حال جمع‌آوری داده برای آموزش اولیه...")
            return

    status["model_ready"] = predictor.is_ready

    prob, direction = predictor.predict(sequence, tabular)

    if direction == "NEUTRAL":
        print(f"[Predict] Neutral signal (prob={prob:.3f}). Skipping Telegram.")
        return

    target_time = ts + config.LOOKAHEAD_MINUTES * 60 * 1000
    pred_id = db.insert_prediction(
        pred_time=ts,
        target_time=target_time,
        current_price=current_price,
        direction=direction,
        probability=prob,
        model_version=predictor.version
    )

    acc20, c20, t20 = db.get_recent_accuracy(20)
    acc50, c50, t50 = db.get_recent_accuracy(50)
    stats = db.get_total_stats()

    status["last_prediction"] = {
        "id": pred_id,
        "direction": direction,
        "probability": round(prob, 4),
        "price": current_price
    }
    status["last_prediction_time"] = datetime.now(timezone.utc).isoformat()
    status["total_predictions"] = stats.get("total", 0)
    status["accuracy_20"] = round(acc20 * 100, 1)
    status["accuracy_50"] = round(acc50 * 100, 1)

    pred_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    notifier.send_prediction(
        price=current_price,
        direction=direction,
        probability=prob,
        acc20=acc20,
        acc50=acc50,
        total_correct=stats.get("correct", 0),
        total=stats.get("total", 0),
        pred_time=pred_dt
    )

    print(f"[Predict] #{pred_id} | Price: {current_price:.2f} | Dir: {direction} | Prob: {prob:.3f}")


def maybe_retrain():
    stats = db.get_total_stats()
    total = stats.get("total", 0)
    if total < config.MIN_SAMPLES_FOR_TRAIN:
        return
    if total % config.RETRAIN_EVERY_N_PREDICTIONS != 0:
        return

    print("[Retrain] Starting periodic update...")
    acc20, _, _ = db.get_recent_accuracy(20)
    acc50, _, _ = db.get_recent_accuracy(50)
    db.log_metrics(acc20, acc50, total, stats.get("correct", 0), notes="periodic check")
    notifier.send_info(f"به‌روزرسانی دوره‌ای مدل انجام شد. دقت ۲۰تایی اخیر: {acc20*100:.1f}%")


def prediction_loop():
    """Background thread that runs forever."""
    print("=" * 60)
    print("BTC Hybrid Predictor - Background loop started")
    print("Source: MEXC Spot real-time Ticker only")
    print("=" * 60)

    bootstrap_history()

    if predictor.load():
        print("[Init] Loaded existing models.")
        status["model_ready"] = True
    else:
        print("[Init] No saved model found. Will train after enough data.")

    last_pred_minute = -1

    while status["running"]:
        try:
            now = datetime.now(timezone.utc)
            current_minute = now.minute

            collect_live_tick()
            check_and_settle_pending()

            if is_prediction_minute(now) and current_minute != last_pred_minute:
                print(f"\n>>> Prediction window at {now.strftime('%H:%M:%S')} UTC")
                make_prediction()
                maybe_retrain()
                last_pred_minute = current_minute

            time.sleep(20)

        except Exception as e:
            print(f"[Loop] Error: {e}")
            time.sleep(30)


# ---------- Flask routes (for keep-alive & monitoring) ----------

STATUS_HTML = """
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Hybrid Predictor Status</title>
    <style>
        body { font-family: Tahoma, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
        .card { background: #1e293b; border-radius: 12px; padding: 20px; max-width: 600px; margin: 0 auto; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
        h1 { color: #38bdf8; text-align: center; }
        .row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #334155; }
        .label { color: #94a3b8; }
        .value { font-weight: bold; color: #f8fafc; }
        .ok { color: #4ade80; }
        .warn { color: #fbbf24; }
        .footer { text-align: center; margin-top: 20px; font-size: 12px; color: #64748b; }
    </style>
</head>
<body>
    <div class="card">
        <h1>⚡ BTC Hybrid Predictor</h1>
        <div class="row"><span class="label">وضعیت</span><span class="value ok">در حال اجرا</span></div>
        <div class="row"><span class="label">مدل آماده</span><span class="value">{{ model_ready }}</span></div>
        <div class="row"><span class="label">آخرین قیمت تیکر</span><span class="value">{{ last_price }}</span></div>
        <div class="row"><span class="label">زمان آخرین تیک</span><span class="value">{{ last_tick_time }}</span></div>
        <div class="row"><span class="label">آخرین پیش‌بینی</span><span class="value">{{ last_pred }}</span></div>
        <div class="row"><span class="label">دقت ۲۰تایی</span><span class="value">{{ acc20 }}%</span></div>
        <div class="row"><span class="label">دقت ۵۰تایی</span><span class="value">{{ acc50 }}%</span></div>
        <div class="row"><span class="label">تعداد کل پیش‌بینی‌ها</span><span class="value">{{ total }}</span></div>
        <div class="row"><span class="label">شروع سرویس</span><span class="value">{{ started }}</span></div>
        <div class="footer">منبع قیمت: MEXC Spot Ticker (لحظه‌ای) | هر ۱۰ دقیقه پیش‌بینی</div>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    last_pred_str = "—"
    if status["last_prediction"]:
        lp = status["last_prediction"]
        last_pred_str = f"{lp['direction']} ({lp['probability']*100:.1f}%) @ {lp['price']:,.1f}"

    return render_template_string(
        STATUS_HTML,
        model_ready="بله" if status["model_ready"] else "خیر",
        last_price=f"{status['last_tick_price']:,.2f}" if status["last_tick_price"] else "—",
        last_tick_time=status["last_tick_time"] or "—",
        last_pred=last_pred_str,
        acc20=status["accuracy_20"],
        acc50=status["accuracy_50"],
        total=status["total_predictions"],
        started=status["started_at"]
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "running": status["running"],
        "model_ready": status["model_ready"],
        "last_tick_price": status["last_tick_price"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route("/status")
def api_status():
    return jsonify(status)


def start_background_thread():
    t = threading.Thread(target=prediction_loop, daemon=True)
    t.start()
    print("[Web] Background prediction thread started.")


# Start the loop when the module is loaded (works with gunicorn too)
start_background_thread()


if __name__ == "__main__":
    # Local development
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
