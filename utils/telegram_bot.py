import asyncio
from telegram import Bot
from telegram.constants import ParseMode
import config
from datetime import datetime
import pytz

class TelegramNotifier:
    def __init__(self):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.bot = None
        if self.token:
            self.bot = Bot(token=self.token)

    async def _send(self, text: str):
        if not self.bot or not self.chat_id:
            print("[Telegram] Not configured. Message:\n", text)
            return
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
        except Exception as e:
            print(f"[Telegram] Send error: {e}")

    def send_sync(self, text: str):
        """Synchronous wrapper."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context
                asyncio.create_task(self._send(text))
            else:
                loop.run_until_complete(self._send(text))
        except RuntimeError:
            # New loop
            asyncio.run(self._send(text))

    def send_prediction(self, price: float, direction: str, probability: float,
                        acc20: float, acc50: float, total_correct: int, total: int,
                        pred_time: datetime):
        tehran = pytz.timezone("Asia/Tehran")
        t_str = pred_time.astimezone(tehran).strftime("%Y-%m-%d %H:%M:%S")

        if direction == "UP":
            dir_text = "🔼 <b>بالاتر</b> خواهد بود"
            emoji = "🟢"
        elif direction == "DOWN":
            dir_text = "🔽 <b>پایین‌تر</b> خواهد بود"
            emoji = "🔴"
        else:
            dir_text = "⚪ خنثی (احتمال ضعیف)"
            emoji = "⚪"

        msg = f"""
{emoji} <b>سیگنال پیش‌بینی بیت‌کوین</b>

⏰ زمان: <code>{t_str}</code> (تهران)
💰 قیمت فعلی (تیکر مکسی): <b>{price:,.2f}</b> USDT

📊 پیش‌بینی ۱۰ دقیقه آینده:
{dir_text}

🎯 احتمال مدل: <b>{probability*100:.1f}%</b>

📈 دقت اخیر:
• آخرین ۲۰ پیش‌بینی: <b>{acc20*100:.1f}%</b>
• آخرین ۵۰ پیش‌بینی: <b>{acc50*100:.1f}%</b>
• کل (صحیح/کل): <b>{total_correct}/{total}</b>

🔗 منبع قیمت: MEXC Spot Ticker (لحظه‌ای)
"""
        self.send_sync(msg.strip())

    def send_result(self, pred_id: int, predicted: str, actual: str, correct: bool,
                    pred_price: float, actual_price: float):
        status = "✅ درست" if correct else "❌ غلط"
        msg = f"""
📋 <b>نتیجه پیش‌بینی #{pred_id}</b>

پیش‌بینی: {predicted}
واقعیت: {actual}
وضعیت: {status}

قیمت پیش‌بینی: {pred_price:,.2f}
قیمت واقعی: {actual_price:,.2f}
"""
        self.send_sync(msg.strip())

    def send_info(self, text: str):
        self.send_sync(f"ℹ️ {text}")
