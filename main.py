"""Flask health/status UI. The prediction engine runs in worker.py."""
from flask import Flask, jsonify, render_template_string
from utils.database import Database
from pathlib import Path
from datetime import datetime, timezone
import config

app = Flask(__name__)

HTML = """<!doctype html><html lang=\"fa\" dir=\"rtl\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>BTC Hybrid</title>
<style>body{font-family:Tahoma,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}.card{background:#1e293b;border-radius:12px;padding:20px;max-width:650px;margin:auto}h1{color:#38bdf8;text-align:center}.row{display:flex;gap:15px;justify-content:space-between;flex-wrap:wrap;padding:10px 0;border-bottom:1px solid #334155}.label{color:#94a3b8}.value{overflow-wrap:anywhere;text-align:left}</style></head><body><div class=\"card\">
<h1>⚡ BTC Hybrid Predictor</h1>
<div class=\"row\"><span class=\"label\">وضعیت سرویس</span><span class=\"value\">{{ state }}</span></div>
<div class=\"row\"><span class=\"label\">وضعیت مدل</span><span class=\"value\">{{ model }}</span></div>
<div class=\"row\"><span class=\"label\">قیمت فعلی BTC</span><span class=\"value\">{{ price }}</span></div>
<div class=\"row\"><span class=\"label\">آخرین پیش‌بینی</span><span class=\"value\">{{ prediction }}</span></div>
<div class=\"row\"><span class=\"label\">آخرین فعالیت موتور</span><span class=\"value\">{{ heartbeat }}</span></div>
<div class=\"row\"><span class=\"label\">تلگرام</span><span class=\"value\">{{ telegram }}</span></div>
<div class=\"row\"><span class=\"label\">آخرین خطا</span><span class=\"value\">{{ error }}</span></div>
<p style=\"color:#94a3b8\">جزئیات فنی: <a href=\"/health\" style=\"color:#38bdf8\">/health</a></p>
</div></body></html>"""


def snapshot():
    db = Database()
    state = db.get_all_state()
    latest_tick = db.get_latest_tick()
    latest_prediction = db.get_latest_prediction()
    return state, latest_tick, latest_prediction


@app.get("/")
def home():
    try:
        state, tick, pred = snapshot()
        price = f"{float(tick['price']):,.2f} USDT" if tick else "در انتظار داده"
        if pred:
            prediction = f"{pred['predicted_direction']} | احتمال {float(pred['probability']):.1%} | {pred['pred_time']}"
        else:
            prediction = "هنوز ثبت نشده"
        return render_template_string(
            HTML,
            state=state.get("engine_status", "در انتظار شروع"),
            model=state.get("model_status", "در انتظار بررسی"),
            price=price,
            prediction=prediction,
            heartbeat=state.get("worker_heartbeat", "—"),
            telegram=state.get("telegram_status", "بررسی نشده"),
            error=state.get("last_error", "—"),
        )
    except Exception as exc:
        return render_template_string(HTML, state="خطای پایگاه داده", model="—", price="—", prediction="—", heartbeat="—", telegram="—", error=str(exc)), 503


@app.get("/health")
def health():
    try:
        state, tick, pred = snapshot()
        heartbeat = state.get("worker_heartbeat")
        heartbeat_age = None
        if heartbeat:
            try:
                beat_dt = datetime.fromisoformat(heartbeat)
                if beat_dt.tzinfo is None:
                    beat_dt = beat_dt.replace(tzinfo=timezone.utc)
                heartbeat_age = max(0, int((datetime.now(timezone.utc) - beat_dt).total_seconds()))
            except (TypeError, ValueError):
                heartbeat_age = None
        return jsonify({
            "ok": True,
            "engine_alive": heartbeat_age is not None and heartbeat_age <= 120,
            "worker_heartbeat_age_seconds": heartbeat_age,
            "engine_status": state.get("engine_status", "unknown"),
            "model_status": state.get("model_status", "unknown"),
            "telegram_status": state.get("telegram_status", "unknown"),
            "worker_heartbeat": heartbeat,
            "last_error": state.get("last_error"),
            "last_price": tick.get("price") if tick else None,
            "last_prediction": dict(pred) if pred else None,
            "model_file_exists": Path(config.LSTM_MODEL_PATH).exists(),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503


@app.get("/ready")
def ready():
    try:
        state, _, _ = snapshot()
        return jsonify({"ok": True, "engine_status": state.get("engine_status", "unknown")})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
