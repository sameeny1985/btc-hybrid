"""
Reliable synchronous Telegram notifier using the official Bot HTTP API.
No async event-loop lifecycle is required.
"""
import requests
import config
from datetime import datetime
import pytz


class TelegramNotifier:
    def __init__(self):
        token = (config.TELEGRAM_BOT_TOKEN or "").strip()
        chat_id = (config.TELEGRAM_CHAT_ID or "").strip()
        if token.startswith("YOUR_") or "..." in token:
            token = ""
        if chat_id.startswith("YOUR_") or "xxx" in chat_id.lower():
            chat_id = ""
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BTC-LSTM-Predictor/1.1"})

    @property
    def configured(self):
        return bool(self.token and self.chat_id)

    def _send(self, text: str):
        if not self.configured:
            raise RuntimeError(
                "Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
            )
        response = self.session.post(
            f"{self.base_url}/sendMessage",
            json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        try:
            data = response.json()
        except Exception:
            data = {"description": response.text[:500]}
        if not response.ok or not data.get("ok"):
            raise RuntimeError(
                f"Telegram API error ({response.status_code}): "
                f"{data.get('description', 'Unknown Telegram error')}"
            )
        return data.get("result")

    def send_sync(self, text: str):
        return self._send(text)

    def send_prediction(self, price: float, direction: str, probability: float,
                        acc20: float, acc50: float, total_correct: int, total: int,
                        pred_time: datetime):
        tehran = pytz.timezone("Asia/Tehran")
        if pred_time.tzinfo is None:
            pred_time = pred_time.replace(tzinfo=pytz.UTC)
        t_str = pred_time.astimezone(tehran).strftime("%Y-%m-%d %H:%M:%S")

        if direction == "UP":
            dir_text, emoji = "🔼 <b>بالاتر</b> خواهد بود", "🟢"
        elif direction == "DOWN":
            dir_text, emoji = "🔽 <b>پایین‌تر</b> خواهد بود", "🔴"
        else:
            dir_text, emoji = "⚪ خنثی", "⚪"

        # تشخیص احتمال پایین
        confidence_note = ""
        if probability < 0.55 and probability > 0.45:
            confidence_note = "\n⚠️ <b>توجه: درصد احتمال پایین است (سیگنال ضعیف)</b>"
        elif probability < 0.60 and probability > 0.40:
            confidence_note = "\nℹ️ احتمال متوسط"

        msg = f"""
{emoji} <b>سیگنال پیش‌بینی بیت‌کوین</b>

⏰ زمان: <code>{t_str}</code> (تهران)
💰 قیمت فعلی (تیکر مکسی): <b>{price:,.2f}</b> USDT

📊 پیش‌بینی ۱۰ دقیقه آینده:
{dir_text}

🎯 احتمال مدل: <b>{probability*100:.1f}%</b>{confidence_note}

📈 دقت اخیر:
• آخرین ۲۰ پیش‌بینی: <b>{acc20*100:.1f}%</b>
• آخرین ۵۰ پیش‌بینی: <b>{acc50*100:.1f}%</b>
• کل (صحیح/کل): <b>{total_correct}/{total}</b>

🔗 منبع قیمت: MEXC Spot Ticker
"""
        return self.send_sync(msg.strip())

    def send_result(self, pred_id: int, predicted: str, actual: str, correct: bool,
                    pred_price: float, actual_price: float):
        status = "✅ درست" if correct else "❌ غلط"
        msg = f"""
📋 <b>نتیجه پیش‌بینی #{pred_id}</b>

پیش‌بینی: {predicted}
واقعیت: {actual}
وضعیت: {status}

قیمت مبنا: {pred_price:,.2f}
قیمت واقعی: {actual_price:,.2f}
"""
        return self.send_sync(msg.strip())

    def send_info(self, text: str):
        return self.send_sync(f"ℹ️ {text}")

    def test_connection(self):
        if not self.configured:
            raise RuntimeError("Telegram credentials are missing.")
        response = self.session.get(f"{self.base_url}/getMe", timeout=10)
        data = response.json()
        if not response.ok or not data.get("ok"):
            raise RuntimeError(data.get("description", "Telegram getMe failed"))
        return data.get("result", {})
