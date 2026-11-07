"""
BTC Hybrid Predictor - resilient version for Render
- LSTM + XGBoost
- Trains on historical data at start
- Signals every 10 minutes
- Also runs prediction logic on every HTTP request (so UptimeRobot keep-alive keeps it working)
"""

import time
import os
import threading
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template_string
import numpy as np

import config
from utils.mexc_client import MexcClient
from utils.database import Database
from utils.features import build_sequence_and_features, compute_features
from utils.models import HybridPredictor
from utils.telegram_bot import TelegramNotifier

os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

mexc = MexcClient()
db = Database()
predictor = HybridPredictor()
notifier = TelegramNotifier()

app = Flask(__name__)

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
    "training_status": "not_started",
    "last_loop_log": None,
    "started_at": datetime.now(timezone.utc).isoformat()
}

_last_pred_minute = -1
_lock = threading.Lock()


def now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def is_prediction_minute(dt=None):
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.minute % 10 == 0


def bootstrap_history():
    existing = db.get_recent_ticks(50)
    if len(existing) >= 300:
        print("[Bootstrap] History OK")
        return
    print("[Bootstrap] Fetching klines...")
    klines = mexc.get_klines(interval="1m", limit=500)
    if not klines:
        return
    for k in klines:
        db.insert_tick(int(k[0]), float(k[4]))
    print(f"[Bootstrap] Inserted {len(klines)} ticks")


def prepare_training_data(ticks, seq_len=60, lookahead=10):
    if len(ticks) < seq_len + lookahead + 20:
        return None, None, None, None
    import pandas as pd
    prices = np.array([p for _, p in ticks], dtype=np.float64)
    timestamps = np.array([t for t, _ in ticks])
    df = pd.DataFrame({"close": prices, "high": prices, "low": prices, "open": prices})
    df.index = pd.to_datetime(timestamps, unit="ms")
    feats = compute_features(df)
    if feats.empty:
        return None, None, None, None
    feature_values = feats.values
    sequences, labels, tabular_list, tab_labels = [], [], [], []
    n = len(prices)
    for i in range(seq_len, n - lookahead):
        window = prices[i - seq_len: i + 1]
        log_rets = np.diff(np.log(window + 1e-12))
        if len(log_rets) != seq_len:
            continue
        label = 1 if prices[i + lookahead] > prices[i] else 0
        feat_row = feature_values[min(i, len(feature_values) - 1)]
        sequences.append(log_rets.reshape(seq_len, 1))
        labels.append(label)
        tabular_list.append(feat_row)
        tab_labels.append(label)
    if len(sequences) < 40:
        return None, None, None, None
    return (np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(tabular_list, dtype=np.float32),
            np.array(tab_labels, dtype=np.float32))


def train_model_on_history():
    status["training_status"] = "training"
    print("[Train] Starting...")
    ticks = db.get_recent_ticks(limit=600)
    if len(ticks) < 150:
        status["training_status"] = "failed_not_enough_data"
        return False
    X_seq, y, X_tab, y_tab = prepare_training_data(ticks)
    if X_seq is None:
        status["training_status"] = "failed_prepare"
        return False
    try:
        predictor.train_initial(X_seq, y, X_tab, y_tab)
        status["model_ready"] = True
        status["training_status"] = "completed"
        print("[Train] DONE")
        try:
            notifier.send_info("✅ آموزش اولیه مدل با داده‌های تاریخی تمام شد. از این لحظه سیگنال‌ها ارسال می‌شوند.")
        except Exception:
            pass
        return True
    except Exception as e:
        status["training_status"] = f"failed: {e}"
        print(f"[Train] Failed: {e}")
        return False


def collect_live_tick():
    try:
        price = mexc.get_ticker_price()
        if price is None:
            return None
        db.insert_tick(now_ms(), price)
        status["last_tick_price"] = price
        status["last_tick_time"] = datetime.now(timezone.utc).isoformat()
        return price
    except Exception as e:
        print(f"[Tick] Error: {e}")
        return None


def make_prediction(force=False):
    """force=True sends even if Neutral."""
    global _last_pred_minute
    current_price = mexc.get_ticker_price()
    if current_price is None:
        print("[Predict] No ticker")
        return
    ts = now_ms()
    db.insert_tick(ts, current_price)
    status["last_tick_price"] = current_price
    status["last_tick_time"] = datetime.now(timezone.utc).isoformat()

    ticks = db.get_recent_ticks(limit=300)
    sequence, tabular, _ = build_sequence_and_features(ticks)
    if sequence is None or tabular is None:
        print("[Predict] Not enough features")
        return

    if not predictor.is_ready:
        print("[Predict] Model not ready")
        return

    prob, direction = predictor.predict(sequence, tabular)
    print(f"[Predict] prob={prob:.4f} dir={direction}")

    # Always send if force, or if not neutral
    if direction == "NEUTRAL" and not force:
        # still send a short info so user knows system is alive
        try:
            notifier.send_info(f"⏳ مدل آماده است ولی سیگنال ضعیف (احتمال {prob*100:.1f}%). قیمت فعلی: {current_price:,.2f}")
        except Exception:
            pass
        return

    if direction == "NEUTRAL":
        direction = "UP" if prob >= 0.5 else "DOWN"

    target_time = ts + config.LOOKAHEAD_MINUTES * 60 * 1000
    pred_id = db.insert_prediction(ts, target_time, current_price, direction, prob, model_version=predictor.version)

    acc20, _, _ = db.get_recent_accuracy(20)
    acc50, _, _ = db.get_recent_accuracy(50)
    stats = db.get_total_stats()

    status["last_prediction"] = {"id": pred_id, "direction": direction, "probability": round(prob, 4), "price": current_price}
    status["last_prediction_time"] = datetime.now(timezone.utc).isoformat()
    status["total_predictions"] = stats.get("total", 0)
    status["accuracy_20"] = round(acc20 * 100, 1)
    status["accuracy_50"] = round(acc50 * 100, 1)

    try:
        notifier.send_prediction(
            price=current_price, direction=direction, probability=prob,
            acc20=acc20, acc50=acc50,
            total_correct=stats.get("correct", 0), total=stats.get("total", 0),
            pred_time=datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        )
        print(f"[Predict] SENT #{pred_id} {direction} @ {current_price}")
    except Exception as e:
        print(f"[Predict] Telegram error: {e}")


def do_work():
    """Called from background thread AND from HTTP requests."""
    global _last_pred_minute
    with _lock:
        try:
            now = datetime.now(timezone.utc)
            status["last_loop_log"] = now.isoformat()
            collect_live_tick()

            if not predictor.is_ready:
                if status["training_status"] not in ("completed", "loaded_from_disk", "training"):
                    bootstrap_history()
                    if predictor.load():
                        status["model_ready"] = True
                        status["training_status"] = "loaded_from_disk"
                    else:
                        train_model_on_history()

            if predictor.is_ready and is_prediction_minute(now) and now.minute != _last_pred_minute:
                print(f">>> WINDOW {now.strftime('%H:%M')} — predicting")
                make_prediction(force=True)
                _last_pred_minute = now.minute
        except Exception as e:
            print(f"[do_work] {e}")
            import traceback
            traceback.print_exc()


def background_loop():
    print("=" * 50)
    print("Background loop started")
    bootstrap_history()
    if predictor.load():
        status["model_ready"] = True
        status["training_status"] = "loaded_from_disk"
        print("Model loaded from disk")
    else:
        train_model_on_history()
    while status["running"]:
        try:
            do_work()
            time.sleep(25)
        except Exception as e:
            print(f"[bg] {e}")
            time.sleep(40)


STATUS_HTML = """
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC Predictor</title>
<style>
body{font-family:Tahoma,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}
.card{background:#1e293b;border-radius:12px;padding:20px;max-width:600px;margin:auto}
h1{color:#38bdf8;text-align:center}
.row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #334155}
.label{color:#94a3b8}.value{font-weight:bold}
.ok{color:#4ade80}
</style>
</head>
<body>
<div class="card">
<h1>⚡ BTC Hybrid Predictor</h1>
<div class="row"><span class="label">مدل</span><span class="value">{{ model }}</span></div>
<div class="row"><span class="label">قیمت تیکر مکسی</span><span class="value">{{ price }}</span></div>
<div class="row"><span class="label">زمان آخرین تیک</span><span class="value">{{ tick_time }}</span></div>
<div class="row"><span class="label">آخرین سیگنال</span><span class="value">{{ last_pred }}</span></div>
<div class="row"><span class="label">دقت ۲۰ / ۵۰</span><span class="value">{{ acc20 }}% / {{ acc50 }}%</span></div>
<div class="row"><span class="label">تعداد سیگنال</span><span class="value">{{ total }}</span></div>
<div class="row"><span class="label">آخرین فعالیت</span><span class="value">{{ last_loop }}</span></div>
</div>
</body></html>
"""


@app.route("/")
def home():
    # Keep-alive also drives work
    try:
        do_work()
    except Exception:
        pass
    lp = status.get("last_prediction")
    last_pred_str = "—"
    if lp:
        last_pred_str = f"{lp['direction']} ({lp['probability']*100:.1f}%) @ {lp['price']:,.1f}"
    return render_template_string(
        STATUS_HTML,
        model="آماده ✅" if status["model_ready"] else status.get("training_status", "—"),
        price=f"{status['last_tick_price']:,.2f}" if status["last_tick_price"] else "—",
        tick_time=status.get("last_tick_time") or "—",
        last_pred=last_pred_str,
        acc20=status["accuracy_20"],
        acc50=status["accuracy_50"],
        total=status["total_predictions"],
        last_loop=status.get("last_loop_log") or "—"
    )


@app.route("/health")
def health():
    try:
        do_work()
    except Exception:
        pass
    return jsonify({
        "status": "ok",
        "model_ready": status["model_ready"],
        "price": status["last_tick_price"],
        "last_loop": status.get("last_loop_log"),
        "training": status.get("training_status")
    })


@app.route("/force")
def force_signal():
    """Manual trigger for testing."""
    if not predictor.is_ready:
        return jsonify({"error": "model not ready"}), 400
    make_prediction(force=True)
    return jsonify({"ok": True, "last": status.get("last_prediction")})


# Start background thread
t = threading.Thread(target=background_loop, daemon=True)
t.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
