"""BTC signal engine. This process is supervised by start.py, not by Gunicorn."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from utils.database import Database
from utils.mexc_client import MexcClient
from utils.models import HybridPredictor
from utils.telegram_bot import TelegramNotifier
from utils.features import build_sequence_and_features, ticks_to_dataframe


def log(message):
    print(f"[ENGINE {datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def set_state(db, key, value):
    try:
        db.set_state(key, value)
    except Exception as exc:
        log(f"Could not update state '{key}': {exc}")


def train_model(db, predictor):
    """Train the existing 60-return / 10-minute LSTM strategy without changing thresholds."""
    ticks = db.get_recent_ticks(2000)
    minute_bars = ticks_to_dataframe(ticks)
    log(f"Training check: {len(ticks)} stored snapshots, {len(minute_bars)} one-minute bars available")
    minimum_bars = config.SEQUENCE_LENGTH + config.LOOKAHEAD_MINUTES + 40
    if len(minute_bars) < minimum_bars:
        set_state(db, "model_status", f"waiting_for_minute_data ({len(minute_bars)}/{minimum_bars})")
        return False

    # Training labels are measured in one-minute bars, so lookahead=10 means ten minutes,
    # even though live ticker snapshots are collected every 20 seconds.
    prices = minute_bars["close"].to_numpy(dtype=np.float64)
    seq_len = config.SEQUENCE_LENGTH
    lookahead = config.LOOKAHEAD_MINUTES
    sequences, labels = [], []
    for i in range(seq_len, len(prices) - lookahead):
        window = prices[i - seq_len:i + 1]
        if np.any(window <= 0):
            continue
        log_returns = np.diff(np.log(window))
        if len(log_returns) != seq_len:
            continue
        sequences.append(log_returns.reshape(seq_len, 1))
        labels.append(1.0 if prices[i + lookahead] > prices[i] else 0.0)

    if len(sequences) < 40:
        set_state(db, "model_status", f"not_enough_training_samples ({len(sequences)})")
        log(f"Not enough valid sequences to train: {len(sequences)}")
        return False

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    log(f"Starting LSTM training: X={X.shape}, labels={len(y)}")
    set_state(db, "model_status", f"training ({len(y)} samples)")
    predictor.train_initial(X, y)
    if predictor.is_ready:
        log(f"Training completed; model saved to {config.LSTM_MODEL_PATH}")
        set_state(db, "model_status", f"ready (trained {len(y)} samples)")
        return True
    set_state(db, "model_status", "training_failed_tensorflow_unavailable_or_error")
    return False


def run_engine():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    db = Database()
    mexc = MexcClient()
    predictor = HybridPredictor()
    telegram = TelegramNotifier()

    set_state(db, "engine_status", "starting")
    set_state(db, "last_error", "")
    log(f"Engine starting for {config.SYMBOL}; database={'PostgreSQL' if os.getenv('DATABASE_URL') else 'SQLite'}")

    # Test Telegram explicitly. Never report 'sent' unless Telegram confirms success.
    if telegram.configured:
        try:
            bot = telegram.test_connection()
            set_state(db, "telegram_status", f"connected (@{bot.get('username', 'bot')})")
            log(f"Telegram connection OK: @{bot.get('username', 'bot')}")
        except Exception as exc:
            set_state(db, "telegram_status", f"connection_error: {exc}")
            log(f"Telegram connection failed: {exc}")
    else:
        set_state(db, "telegram_status", "missing_credentials")
        log("Telegram credentials missing; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Render")

    # Restore the saved model if available.
    if predictor.load():
        log(f"Saved model loaded from {config.LSTM_MODEL_PATH}")
        set_state(db, "model_status", "ready (loaded from disk)")
    else:
        log("No usable saved model found; model training will be attempted after data bootstrap")
        set_state(db, "model_status", "not_loaded")

    # Bootstrap historical one-minute candles only when database has too little data.
    existing = db.get_recent_ticks(2000)
    existing_minute_bars = ticks_to_dataframe(existing)
    minimum_bars = config.SEQUENCE_LENGTH + config.LOOKAHEAD_MINUTES + 40
    if len(existing_minute_bars) < minimum_bars:
        log(f"Only {len(existing_minute_bars)} one-minute bars found; fetching historical candles from MEXC")
        klines = mexc.get_klines(interval="1m", limit=500)
        inserted = 0
        for candle in klines:
            try:
                db.insert_tick(int(candle[0]), float(candle[4]))
                inserted += 1
            except Exception as exc:
                log(f"Skipping invalid historical candle: {exc}")
        log(f"Historical bootstrap inserted {inserted} rows")
    else:
        log(f"Historical database already contains {len(existing_minute_bars)} one-minute bars; bootstrap skipped")

    # If model is not yet ready, try training. Retry at most every five minutes.
    next_train_attempt = time.time() + 300
    if not predictor.is_ready:
        try:
            train_model(db, predictor)
        except Exception as exc:
            log(f"Initial training error: {exc}\n{traceback.format_exc()}")
            set_state(db, "last_error", f"training: {exc}")

    # Notify once per engine start if Telegram is available.
    if telegram.configured:
        try:
            telegram.send_info(
                "موتور پیش‌بینی BTC فعال شد.\n"
                f"مدل: {'آماده' if predictor.is_ready else 'در انتظار آموزش'}\n"
                "بازه پیش‌بینی: ۱۰ دقیقه"
            )
            set_state(db, "telegram_status", "connected; startup message delivered")
            log("Telegram startup message delivered successfully")
        except Exception as exc:
            set_state(db, "telegram_status", f"send_error: {exc}")
            log(f"Telegram startup message failed: {exc}")

    set_state(db, "engine_status", "running")
    log("Prediction loop started")

    while True:
        cycle_start = time.time()
        try:
            now = datetime.now(timezone.utc)
            set_state(db, "worker_heartbeat", now.isoformat())

            price = mexc.get_ticker_price()
            if price is not None and price > 0:
                now_ms = int(time.time() * 1000)
                db.insert_tick(now_ms, float(price))
                set_state(db, "last_price", f"{price:.8f}")

                # Resolve predictions whose 10-minute target time has passed.
                pending = db.get_pending_predictions(now_ms)
                for old in pending:
                    old_price = float(old["current_price"])
                    actual_direction = "UP" if price > old_price else "DOWN" if price < old_price else "NEUTRAL"
                    correct = actual_direction == old["predicted_direction"]
                    db.update_prediction_result(old["id"], float(price), actual_direction, int(correct))
                    log(f"Prediction #{old['id']} evaluated: predicted={old['predicted_direction']} actual={actual_direction} correct={correct}")
                    if telegram.configured:
                        try:
                            telegram.send_result(old["id"], old["predicted_direction"], actual_direction,
                                                 correct, old_price, float(price))
                            set_state(db, "telegram_status", "connected; result message delivered")
                            log(f"Telegram result for prediction #{old['id']} delivered")
                        except Exception as exc:
                            set_state(db, "telegram_status", f"send_error: {exc}")
                            set_state(db, "last_error", f"Telegram result: {exc}")
                            log(f"Telegram result send failed: {exc}")
            else:
                log("MEXC ticker did not return a valid price")

            # Retry model training later if bootstrap had too little data or TF was unavailable.
            if not predictor.is_ready and time.time() >= next_train_attempt:
                next_train_attempt = time.time() + 300
                try:
                    train_model(db, predictor)
                    if predictor.is_ready and telegram.configured:
                        telegram.send_info("✅ آموزش مدل LSTM تکمیل شد؛ موتور آماده تولید پیش‌بینی است.")
                        set_state(db, "telegram_status", "connected; training message delivered")
                except Exception as exc:
                    log(f"Training retry failed: {exc}")
                    set_state(db, "last_error", f"training: {exc}")

            # One prediction per 10-minute UTC slot, even if the process starts mid-slot.
            if predictor.is_ready and price is not None and price > 0:
                now_epoch = int(time.time())
                slot_start = (now_epoch // 600) * 600
                slot_start_ms = slot_start * 1000
                slot_end_ms = slot_start_ms + 600_000
                if not db.has_prediction_in_slot(slot_start_ms, slot_end_ms):
                    recent_ticks = db.get_recent_ticks(600)
                    sequence, tabular, _ = build_sequence_and_features(recent_ticks)
                    if sequence is None:
                        log(f"Prediction skipped: insufficient recent minute data ({len(recent_ticks)} ticks)")
                    else:
                        probability, direction = predictor.predict(sequence, tabular)
                        if direction == "NEUTRAL":
                            direction = "UP" if probability >= 0.5 else "DOWN"
                        pred_time_ms = int(time.time() * 1000)
                        prediction_id = db.insert_prediction(
                            pred_time_ms,
                            pred_time_ms + config.LOOKAHEAD_MINUTES * 60_000,
                            float(price), direction, float(probability),
                            model_version=predictor.version,
                        )
                        pred_dt = datetime.now(timezone.utc)
                        log(f"PREDICTION CREATED id={prediction_id} direction={direction} probability={probability:.4f} price={price:.2f}")
                        set_state(db, "last_prediction", f"{direction} {probability:.4f} @ {price:.2f}")
                        if telegram.configured:
                            try:
                                acc20, _, _ = db.get_recent_accuracy(20)
                                acc50, _, _ = db.get_recent_accuracy(50)
                                stats = db.get_total_stats()
                                telegram.send_prediction(
                                    price=float(price), direction=direction, probability=float(probability),
                                    acc20=acc20, acc50=acc50,
                                    total_correct=int(stats.get("correct") or 0),
                                    total=int(stats.get("total") or 0), pred_time=pred_dt,
                                )
                                set_state(db, "telegram_status", "connected; prediction message delivered")
                                log(f"Telegram prediction #{prediction_id} delivered successfully")
                            except Exception as exc:
                                set_state(db, "telegram_status", f"send_error: {exc}")
                                set_state(db, "last_error", f"Telegram prediction: {exc}")
                                log(f"Telegram prediction send failed: {exc}")
                        else:
                            set_state(db, "telegram_status", "missing_credentials")
                            log("Prediction saved, but Telegram message skipped: credentials missing")

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            log(f"Loop error: {error_text}\n{traceback.format_exc()}")
            set_state(db, "last_error", error_text)

        elapsed = time.time() - cycle_start
        time.sleep(max(2, config.POLL_SECONDS - elapsed))


if __name__ == "__main__":
    try:
        run_engine()
    except KeyboardInterrupt:
        log("Engine stopped by supervisor")
        sys.exit(0)
    except Exception as exc:
        log(f"FATAL ENGINE ERROR: {exc}\n{traceback.format_exc()}")
        try:
            Database().set_state("engine_status", "fatal")
            Database().set_state("last_error", str(exc))
        except Exception:
            pass
        sys.exit(1)
