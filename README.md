# BTC LSTM Predictor (MEXC + Flask + Telegram)

نسخه **Web Service** + **PostgreSQL** مناسب برای Render.

## ویژگی‌ها

- قیمت لحظه‌ای **فقط از تیکر مکسی**
- مدل فعال: Bidirectional LSTM با TensorFlow (در این نسخه XGBoost وجود ندارد)
- پیش‌بینی در دقایق ۰، ۱۰، ۲۰، ۳۰، ۴۰، ۵۰
- دیتابیس **PostgreSQL** (لینک از طریق `DATABASE_URL`)
- آموزش اولیه LSTM از داده‌های تاریخی؛ مدل ذخیره‌شده در دیسک پایدار
- ارسال سیگنال به تلگرام
- صفحه وضعیت در `/` + `/health` برای keep-alive

## استقرار روی Render

### ۱. ساخت دیتابیس PostgreSQL
- در Render Dashboard → New → PostgreSQL
- نام پیشنهادی: `btc-predictor-db`
- پلن Free کافی است

### ۲. ساخت Web Service
- New → Web Service
- Connect به ریپوی GitHub
- تنظیمات:
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**:  
    `gunicorn main:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`

### ۳. Environment Variables
حتماً این‌ها را اضافه کنید:

| Key                  | Value                          |
|----------------------|--------------------------------|
| `TELEGRAM_BOT_TOKEN` | توکن بات تلگرام                |
| `TELEGRAM_CHAT_ID`   | آیدی کانال یا چت               |
| `DATABASE_URL`       | از دیتابیس Postgres کپی کنید   |

**نکته:** اگر از Blueprint (`render.yaml`) استفاده کنید، `DATABASE_URL` به صورت خودکار لینک می‌شود.

### ۴. Persistent Disk (برای مدل‌ها)
- Name: `predictor-models`
- Mount Path: `/opt/render/project/src/models`
- Size: 1 GB

### ۵. Keep-Alive
آدرس سرویس را در UptimeRobot یا cron-job.org هر ۵ دقیقه پینگ کنید (`/` یا `/health`).

## اجرای محلی

```bash
pip install -r requirements.txt
cp .env.example .env
# TELEGRAM_... و DATABASE_URL را پر کنید
python main.py
```

## ساختار

```
btc_hybrid_predictor/
├── main.py
├── config.py               # DATABASE_URL از محیط خوانده می‌شود
├── requirements.txt
├── render.yaml
├── utils/
│   ├── mexc_client.py
│   ├── database.py         # PostgreSQL
│   ├── features.py
│   ├── models.py
│   └── telegram_bot.py
└── models/                 # مدل‌های ذخیره‌شده (Persistent Disk)
```

قیمت فعلی همیشه از تیکر زنده مکسی گرفته می‌شود.
Python 3.11 forced via runtime.txt and PYTHON_VERSION


## نکات نسخه اصلاح‌شده
- ارسال تلگرام از API رسمی Bot HTTP و به‌صورت synchronous انجام می‌شود؛ خطای واقعی API در `/health` قابل مشاهده است.
- برای ارسال سیگنال باید هر دو متغیر `TELEGRAM_BOT_TOKEN` و `TELEGRAM_CHAT_ID` در Render تنظیم شوند.
- برای ارسال به کانال، بات باید در کانال عضو و مدیر با مجوز ارسال پیام باشد.
- `/health` وضعیت مدل، قیمت، آخرین سیگنال و وضعیت تلگرام را برمی‌گرداند.
- اندیکاتورها در `utils/features.py` محاسبه می‌شوند؛ ورودی مدل پیش‌بینی فعلی فقط دنباله بازده‌های لگاریتمی ۶۰ گام است و ویژگی‌های جدولی به مدل داده نمی‌شوند.
