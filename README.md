# BTC Hybrid Predictor — Render deployment package

این نسخه برای رفع مشکل توقف موتور سیگنال هنگام راه‌اندازی مجدد Gunicorn آماده شده است.

## ساختار اجرا

- `main.py`: فقط وب‌سایت وضعیت و مسیرهای `/health` و `/ready` را اجرا می‌کند.
- `worker.py`: موتور مستقل دریافت قیمت، بارگذاری/آموزش LSTM، تولید پیش‌بینی ۱۰ دقیقه‌ای و ارسال تلگرام است.
- `start.py`: هر دو فرایند را در یک سرویس Render اجرا و موتور را در صورت خروج دوباره راه‌اندازی می‌کند.
- `utils/database.py`: ذخیره قیمت‌ها، پیش‌بینی‌ها و وضعیت موتور در PostgreSQL (یا SQLite برای اجرای محلی).
- مدل در مسیر `models/lstm_model.keras` ذخیره می‌شود؛ در Render این پوشه روی Persistent Disk قرار دارد.

## دیپلوی در Render

1. تمام فایل‌های ZIP را در ریشه همان GitHub repository قرار بده؛ پوشه `utils` باید کنار `main.py` باشد.
2. در Render، سرویس Web موجود را به همان repository متصل نگه دار.
3. در بخش **Build Command** قرار بده:

   `pip install --upgrade pip && pip install -r requirements.txt`

4. در بخش **Start Command** قرار بده:

   `python start.py`

5. مسیر Health Check را `/health` قرار بده.
6. متغیرهای محیطی زیر را در Render تنظیم کن:
   - `TELEGRAM_BOT_TOKEN`: توکن واقعی BotFather
   - `TELEGRAM_CHAT_ID`: شناسه عددی چت یا کانال؛ بات باید در کانال دسترسی ارسال پیام داشته باشد.
   - `DATABASE_URL`: آدرس PostgreSQL متصل به سرویس. اگر از Blueprint همین پروژه استفاده می‌کنی، Render آن را از دیتابیس `btc-predictor-db` تنظیم می‌کند.
   - `PYTHON_VERSION`: مقدار `3.11.10`
7. Persistent Disk را به مسیر `/opt/render/project/src/models` متصل کن. Blueprint این دیسک را تعریف کرده است. اگر سرویس موجودت دیسک دارد، از دیسک فعلی استفاده کن و دیسک دوم نساز.
8. Deploy کن و صفحه اصلی سرویس یا `/health` را باز کن.

> نکته: اگر از `render.yaml` برای ساخت Blueprint استفاده می‌کنی، قبل از تأیید بررسی کن که دیتابیس جدید و سرویس تکراری ناخواسته نسازد. برای سرویس موجود، معمولاً کافی است Build Command، Start Command و Health Check را مطابق بالا در تنظیمات همان سرویس اصلاح کنی.

## تنظیم تلگرام

- `TELEGRAM_BOT_TOKEN` و `TELEGRAM_CHAT_ID` را فقط در Render Environment وارد کن؛ داخل فایل‌ها یا GitHub قرار نده.
- بات باید به کانال اضافه شده و مجوز ارسال پیام داشته باشد.
- موتور ابتدا با `getMe` اتصال توکن را بررسی می‌کند. سپس پیام شروع می‌فرستد.
- پس از هر پیش‌بینی، فقط در صورت پاسخ موفق Telegram API عبارت ارسال موفق در لاگ ثبت می‌شود.

## منطق پیش‌بینی

پارامترهای اصلی استراتژی قبلی حفظ شده‌اند: جفت `BTCUSDT`، دنباله ۶۰ بازده لگاریتمی، افق ۱۰ دقیقه، مدل Bidirectional LSTM و آستانه‌های احتمال ۰٫۵۱ و ۰٫۴۹. پیش‌بینی‌ها بر اساس اسلات‌های ۱۰ دقیقه‌ای ثبت می‌شوند و وجود پیش‌بینی همان اسلات در دیتابیس بررسی می‌شود تا با ری‌استارت سرویس پیام تکراری ساخته نشود.

## بررسی اجرا

- `/`: صفحه وضعیت فارسی
- `/health`: وضعیت موتور، مدل، تلگرام، آخرین قیمت و آخرین پیش‌بینی به‌صورت JSON
- `/ready`: بررسی پاسخ‌گویی دیتابیس و وضعیت موتور

در Render Logs باید خطوطی مانند این‌ها دیده شوند:

- `[SUPERVISOR] Started worker ...`
- `[ENGINE] ... Engine starting ...`
- `Saved model loaded ...` یا `Starting LSTM training ...`
- `PREDICTION CREATED ...`
- `Telegram prediction #... delivered successfully`

اگر خط آخر وجود ندارد، به مقدار `telegram_status` و `last_error` در `/health` و لاگ‌ها نگاه کن. صفحه وب با HTTP 200 به‌تنهایی به معنی فعال بودن موتور سیگنال یا ارسال تلگرام نیست.

## اجرای محلی

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows; در Linux/macOS از cp استفاده کن
python start.py
```

برای اجرای محلی بدون PostgreSQL، متغیر `DATABASE_URL` را از `.env` حذف کن تا SQLite در پوشه `data` استفاده شود.
