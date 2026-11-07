"""
Lightweight startup version for Render.
Server starts immediately. Heavy work (TF/train/predict) runs later in background.
"""

import os
import time
import threading
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template_string

# Minimal imports first so gunicorn can bind port quickly
app = Flask(__name__)

status = {
    "ready": False,
    "model_ready": False,
    "training_status": "pending",
    "last_tick_price": None,
    "last_tick_time": None,
    "last_prediction": None,
    "last_error": None,
    "started_at": datetime.now(timezone.utc).isoformat(),
    "last_loop": None,
}

_worker_started = False
_lock = threading.Lock()


def _start_worker_once():
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
        t = threading.Thread(target=_background_worker, daemon=True)
        t.start()


def _background_worker():
    """All heavy imports and work happen here, AFTER server is up."""
    try:
        status["training_status"] = "importing"
        import config
        from utils.mexc_client import MexcClient
        from utils.database import Database
        from utils.features import build_sequence_and_features, compute_features
        from utils.models import HybridPredictor
        from utils.telegram_bot import TelegramNotifier
        import numpy as np

        mexc = MexcClient()
        db = Database()
        predictor = HybridPredictor()
        notifier = TelegramNotifier()

        status["training_status"] = "bootstrap"
        # bootstrap
        ticks = db.get_recent_ticks(200)
        if len(ticks) < 150:
            klines = mexc.get_klines(interval="1m", limit=500)
            if klines:
                for k in klines:
                    db.insert_tick(int(k[0]), float(k[4]))

        status["training_status"] = "loading_or_training"
        if predictor.load():
            status["model_ready"] = True
            status["training_status"] = "loaded"
        else:
            # train
            ticks = db.get_recent_ticks(600)
            if len(ticks) >= 150:
                try:
                    prices = np.array([p for _, p in ticks], dtype=np.float64)
                    timestamps = np.array([t for t, _ in ticks])
                    import pandas as pd
                    df = pd.DataFrame({"close": prices, "high": prices, "low": prices, "open": prices})
                    df.index = pd.to_datetime(timestamps, unit="ms")
                    feats = compute_features(df)
                    seq_len, lookahead = 60, 10
                    sequences, labels = [], []
                    n = len(prices)
                    for i in range(seq_len, n - lookahead):
                        window = prices[i - seq_len:i + 1]
                        log_rets = np.diff(np.log(window + 1e-12))
                        if len(log_rets) != seq_len:
                            continue
                        label = 1 if prices[i + lookahead] > prices[i] else 0
                        sequences.append(log_rets.reshape(seq_len, 1))
                        labels.append(label)
                    if len(sequences) >= 40:
                        X = np.array(sequences, dtype=np.float32)
                        y = np.array(labels, dtype=np.float32)
                        predictor.train_initial(X, y)
                        status["model_ready"] = bool(predictor.is_ready)
                        status["training_status"] = "trained" if predictor.is_ready else "training_failed"
                        if predictor.is_ready:
                            try:
                                notifier.send_info("✅ آموزش LSTM تمام شد. سیگنال‌ها از این لحظه ارسال می‌شوند.")
                                status["telegram_status"] = "connected"
                            except Exception as te:
                                status["telegram_status"] = f"error: {te}"
                                status["last_error"] = f"telegram: {te}"
                    else:
                        status["training_status"] = "not_enough_samples"
                except Exception as e:
                    status["training_status"] = f"train_error: {e}"
                    status["last_error"] = str(e)
            else:
                status["training_status"] = "not_enough_ticks"

        status["ready"] = True
        last_prediction_slot = None
        status["telegram_status"] = "configured" if notifier.configured else "missing_credentials"

        while True:
            try:
                status["last_loop"] = datetime.now(timezone.utc).isoformat()
                price = mexc.get_ticker_price()
                if price:
                    status["last_tick_price"] = price
                    status["last_tick_time"] = datetime.now(timezone.utc).isoformat()
                    db.insert_tick(int(time.time() * 1000), price)
                    # Resolve predictions whose 10-minute target time has passed.
                    try:
                        pending = db.get_pending_predictions(int(time.time() * 1000))
                        for old in pending:
                            actual_direction = "UP" if price > float(old["current_price"]) else "DOWN" if price < float(old["current_price"]) else "NEUTRAL"
                            correct = int(actual_direction == old["predicted_direction"])
                            db.update_prediction_result(old["id"], price, actual_direction, correct)
                            try:
                                notifier.send_result(
                                    old["id"], old["predicted_direction"], actual_direction,
                                    bool(correct), float(old["current_price"]), price
                                )
                            except Exception as te:
                                status["telegram_status"] = f"error: {te}"
                                status["last_error"] = f"telegram: {te}"
                    except Exception as eval_error:
                        status["evaluation_error"] = str(eval_error)

                now = datetime.now(timezone.utc)
                slot = int(now.timestamp() // 600)
                if status["model_ready"] and price is not None and now.minute % 10 == 0 and slot != last_prediction_slot:
                    last_prediction_slot = slot
                    ticks = db.get_recent_ticks(300)
                    seq, tab, _ = build_sequence_and_features(ticks)
                    if seq is not None:
                        prob, direction = predictor.predict(seq, tab)
                        if direction == "NEUTRAL":
                            direction = "UP" if prob >= 0.5 else "DOWN"
                        ts = int(time.time() * 1000)
                        pred_id = db.insert_prediction(
                            ts, ts + 10 * 60 * 1000, price or 0, direction, prob
                        )
                        status["last_prediction"] = {
                            "id": pred_id, "direction": direction,
                            "probability": round(prob, 4), "price": price
                        }
                        try:
                            acc20, _, _ = db.get_recent_accuracy(20)
                            acc50, _, _ = db.get_recent_accuracy(50)
                            stats = db.get_total_stats()
                            notifier.send_prediction(
                                price=price, direction=direction, probability=prob,
                                acc20=acc20, acc50=acc50,
                                total_correct=stats.get("correct", 0),
                                total=stats.get("total", 0),
                                pred_time=now
                            )
                        except Exception as te:
                            status["telegram_status"] = f"error: {te}"
                            status["last_error"] = f"telegram: {te}"
                        else:
                            status["telegram_status"] = "sent"
                time.sleep(20)
            except Exception as e:
                status["last_error"] = str(e)
                time.sleep(30)
    except Exception as e:
        status["last_error"] = f"worker_fatal: {e}"
        status["training_status"] = f"fatal: {e}"
        status["ready"] = False
        # Allow a later request / health check to start a fresh worker.
        with _lock:
            global _worker_started
            _worker_started = False


HTML = """
<!DOCTYPE html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC LSTM</title>
<style>
body{font-family:Tahoma,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}
.card{background:#1e293b;border-radius:12px;padding:20px;max-width:560px;margin:auto}
h1{color:#38bdf8;text-align:center}
.row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #334155}
.label{color:#94a3b8}.ok{color:#4ade80}
</style></head><body><div class="card">
<h1>⚡ BTC LSTM Predictor</h1>
<div class="row"><span class="label">وضعیت</span><span class="ok">{{ state }}</span></div>
<div class="row"><span class="label">مدل</span><span>{{ model }}</span></div>
<div class="row"><span class="label">قیمت تیکر مکسی</span><span>{{ price }}</span></div>
<div class="row"><span class="label">آخرین سیگنال</span><span>{{ pred }}</span></div>
<div class="row"><span class="label">آخرین فعالیت</span><span>{{ loop }}</span></div>
<div class="row"><span class="label">وضعیت تلگرام</span><span>{{ telegram }}</span></div>
<div class="row"><span class="label">خطا</span><span>{{ err }}</span></div>
</div></body></html>
"""


@app.route("/")
def home():
    _start_worker_once()
    lp = status.get("last_prediction")
    pred = "—"
    if lp:
        pred = f"{lp.get('direction')} ({lp.get('probability',0)*100:.1f}%) @ {lp.get('price')}"
    return render_template_string(
        HTML,
        state="در حال اجرا",
        model=("آماده" if status["model_ready"] else status.get("training_status", "—")),
        price=(f"{status['last_tick_price']:,.2f}" if status.get("last_tick_price") else "—"),
        pred=pred,
        loop=status.get("last_loop") or "—",
        telegram=status.get("telegram_status") or "در انتظار بررسی",
        err=status.get("last_error") or "—",
    )


@app.route("/health")
def health():
    _start_worker_once()
    return jsonify({
        "status": "ok",
        "model_ready": status["model_ready"],
        "training_status": status.get("training_status"),
        "price": status.get("last_tick_price"),
        "last_error": status.get("last_error"),
        "last_loop": status.get("last_loop"),
        "telegram_status": status.get("telegram_status"),
        "last_prediction": status.get("last_prediction"),
        "evaluation_error": status.get("evaluation_error"),
    })


@app.route("/force")
def force():
    _start_worker_once()
    return jsonify({"ok": True, "msg": "worker running", "status": status})


# Start worker shortly after import (non-blocking)
threading.Timer(2.0, _start_worker_once).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
