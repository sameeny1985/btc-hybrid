"""
BTC Hybrid LSTM + XGBoost Predictor - Web Service version for Render
- Real-time ticker price ONLY from MEXC Spot
- Trains model on historical data at startup, then starts signaling immediately
- Predicts every 10 minutes (at :00, :10, :20, :30, :40, :50)
- Continuous learning from prediction feedback
- Sends signals to Telegram
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
from utils.features import build_sequence_and_features, ticks_to_dataframe, compute_features
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
    "started_at": datetime.now(timezone.utc).isoformat()
}


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def is_prediction_minute(dt: datetime = None) -> bool:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.minute % 10 == 0 and dt.second < 45


def bootstrap_history():
    existing = db.get_recent_ticks(50)
    if len(existing) >= 300:
        print("[Bootstrap] Enough history already present.")
        return

    print("[Bootstrap] Fetching historical klines from MEXC...")
    klines = mexc.get_klines(interval="1m", limit=500)
    if not klines:
        print("[Bootstrap] Failed to get klines.")
        return

    for k in klines:
        ts = int(k[0])
        close_price = float(k[4])
        db.insert_tick(ts, close_price)

    print(f"[Bootstrap] Inserted {len(klines)} historical ticks.")


def prepare_training_data(ticks, seq_len=60, lookahead=10):
    """
    Build sequences and labels from historical ticks.
    Label = 1 if price after `lookahead` minutes is higher than current price.
    """
    if len(ticks) < seq_len + lookahead + 20:
        return None, None, None, None

    # ticks is list of (timestamp_ms, price) sorted ascending
    prices = np.array([p for _, p in ticks], dtype=np.float64)
    timestamps = np.array([t for t, _ in ticks])

    sequences = []
    labels = []
    tabular_list = []
    tab_labels = []

    # We need feature matrix too. Build a simple dataframe of closes.
    import pandas as pd
    df = pd.DataFrame({"close": prices, "high": prices, "low": prices, "open": prices})
    df.index = pd.to_datetime(timestamps, unit="ms")

    feats = compute_features(df)
    if feats.empty:
        return None, None, None, None

    feature_values = feats.values
    feature_index = feats.index

    # Align by using integer positions
    n = len(prices)
    for i in range(seq_len, n - lookahead):
        # sequence of log returns
        window = prices[i - seq_len : i + 1]
        log_rets = np.diff(np.log(window + 1e-12))
        if len(log_rets) != seq_len:
            continue

        future_price = prices[i + lookahead]
        current_price = prices[i]
        label = 1 if future_price > current_price else 0

        # tabular features at position i (approximate)
        # find closest feature row
        try:
            # use the last available feature row up to i
            feat_row = feature_values[min(i, len(feature_values) - 1)]
        except Exception:
            continue

        sequences.append(log_rets.reshape(seq_len, 1))
        labels.append(label)
        tabular_list.append(feat_row)
        tab_labels.append(label)

    if len(sequences) < 50:
        return None, None, None, None

    X_seq = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    X_tab = np.array(tabular_list, dtype=np.float32)
    y_tab = np.array(tab_labels, dtype=np.float32)

    print(f"[Train] Prepared {len(X_seq)} samples for training.")
    return X_seq, y, X_tab, y_tab


def train_model_on_history():
    """Train the hybrid model using historical data already in DB."""
    status["training_status"] = "training"
    print("[Train] Starting initial training on historical data...")

    ticks = db.get_recent_ticks(limit=600)
    if len(ticks) < 200:
        print("[Train] Not enough ticks for training.")
        status["training_status"] = "failed_not_enough_data"
        return False

    # ticks from get_recent_ticks are already sorted ascending (oldest first)
    X_seq, y, X_tab, y_tab = prepare_training_data(ticks)

    if X_seq is None:
        print("[Train] Could not prepare training data.")
        status["training_status"] = "failed_prepare"
        return False

    try:
        predictor.train_initial(X_seq, y, X_tab, y_tab)
        status["model_ready"] = True
        status["training_status"] = "completed"
        print("[Train] Initial training completed successfully. Model is ready.")
        notifier.send_info("✅ آموزش اولیه مدل با داده‌های تاریخی تمام شد. از این لحظه سیگنال‌ها ارسال می‌شوند.")
        return True
    except Exception as e:
        print(f"[Train] Training failed: {e}")
        status["training_status"] = f"failed: {str(e)[:100]}"
        return False


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
        return

    if not predictor.is_ready:
        print("[Predict] Model still not ready.")
        return

    status["model_ready"] = True

    prob, direction = predictor.predict(sequence, tabular)

    if direction == "NEUTRAL":
        print(f"[Predict] Neutral (prob={prob:.3f}). Skipping.")
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

    acc20, _, _ = db.get_recent_accuracy(20)
    acc50, _, _ = db.get_recent_accuracy(50)
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

    print("[Retrain] Periodic update triggered...")
    acc20, _, _ = db.get_recent_accuracy(20)
    acc50, _, _ = db.get_recent_accuracy(50)
    db.log_metrics(acc20, acc50, total, stats.get("correct", 0), notes="periodic check")
    notifier.send_info(f"به‌روزرسانی دوره‌ای مدل. دقت ۲۰تایی اخیر: {acc20*100:.1f}%")


def prediction_loop():
    print("=" * 60)
    print("BTC Hybrid Predictor started")
    print("=" * 60)

    # 1. Bootstrap historical data
    bootstrap_history()

    # 2. Try load existing model, otherwise train on historical data
    if predictor.load():
        print("[Init] Loaded existing model from disk.")
        status["model_ready"] = True
        status["training_status"] = "loaded_from_disk"
    else:
        print("[Init] No saved model. Training on historical data now...")
        success = train_model_on_history()
        if not success:
            print("[Init] Training failed. Will retry later or wait for more data.")
            # We can still try again later if more data arrives
            status["training_status"] = "failed_will_retry"

    # 3. Main loop
    last_pred_minute = -1

    while status["running"]:
        try:
            now = datetime.now(timezone.utc)
            current_minute = now.minute

            collect_live_tick()
            check_and_settle_pending()

            # If model still not ready, try training again after some data collected
            if not predictor.is_ready and len(db.get_recent_ticks(100)) > 250:
                print("[Init] Retrying training...")
                train_model_on_history()

            if predictor.is_ready and is_prediction_minute(now) and current_minute != last_pred_minute:
                print(f"\n>>> Prediction window {now.strftime('%H:%M:%S')} UTC")
                make_prediction()
                maybe_retrain()
                last_pred_minute = current_minute

            time.sleep(15)

        except Exception as e:
            print(f"[Loop] Error: {e}")
            time.sleep(30)


# ---------- Flask routes ----------

STATUS_HTML = """
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Hybrid Predictor</title>
    <style>
        body { font-family: Tahoma, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
        .card { background: #1e293b; border-radius: 12px; padding: 20px; max-width: 620px; margin: 0 auto; }
        h1 { color: #38bdf8; text-align: center; }
        .row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #334155; }
        .label { color: #94a3b8; }
        .value { font-weight: bold; }
        .ok { color: #4ade80; }
        .warn { color: #fbbf24; }
        .footer { text-align: center; margin-top: 18px; font-size: 12px; color: #64748b; }
    </style>
</head>
<body>
    <div class="card">
        <h1>⚡ BTC Hybrid Predictor</h1>
        <div class="row"><span class="label">وضعیت سرویس</span><span class="value ok">در حال اجرا</span></div>
        <div class="row"><span class="label">وضعیت مدل</span><span class="value">{{ model_status }}</span></div>
        <div class="row"><span class="label">آخرین قیمت تیکر</span><span class="value">{{ last_price }}</span></div>
        <div class="row"><span class="label">آخرین پیش‌بینی</span><span class="value">{{ last_pred }}</span></div>
        <div class="row"><span class="label">دقت ۲۰تایی</span><span class="value">{{ acc20 }}%</span></div>
        <div class="row"><span class="label">دقت ۵۰تایی</span><span class="value">{{ acc50 }}%</span></div>
        <div class="row"><span class="label">تعداد کل سیگنال‌ها</span><span class="value">{{ total }}</span></div>
        <div class="footer">منبع: MEXC Spot Ticker | آموزش اولیه روی داده تاریخی</div>
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

    model_status = "آماده ✅" if status["model_ready"] else f"در حال آماده‌سازی ({status.get('training_status', '')})"

    return render_template_string(
        STATUS_HTML,
        model_status=model_status,
        last_price=f"{status['last_tick_price']:,.2f}" if status["last_tick_price"] else "—",
        last_pred=last_pred_str,
        acc20=status["accuracy_20"],
        acc50=status["accuracy_50"],
        total=status["total_predictions"]
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": status["model_ready"],
        "training_status": status.get("training_status"),
        "last_tick_price": status["last_tick_price"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route("/status")
def api_status():
    return jsonify(status)


def start_background_thread():
    t = threading.Thread(target=prediction_loop, daemon=True)
    t.start()
    print("[Web] Background thread started.")


start_background_thread()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
