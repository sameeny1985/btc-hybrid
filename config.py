import os
from dotenv import load_dotenv

load_dotenv()

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Database (PostgreSQL on Render)
# Render automatically provides DATABASE_URL when you link a Postgres database
DATABASE_URL = os.getenv("DATABASE_URL", "")

# MEXC
MEXC_BASE_URL = "https://api.mexc.com"
SYMBOL = "BTCUSDT"

# Timing
PREDICTION_INTERVAL_MINUTES = 10
# Predictions only at minutes: 0, 10, 20, 30, 40, 50

# Model paths (still use local disk / persistent disk for models)
DATA_DIR = "data"
MODEL_DIR = "models"
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Feature & model settings
SEQUENCE_LENGTH = 40          # last 60 minutes of price points
LOOKAHEAD_MINUTES = 10
MIN_SAMPLES_FOR_TRAIN = 200
RETRAIN_EVERY_N_PREDICTIONS = 30
FINE_TUNE_LR = 1e-5

# Decision thresholds (adaptive later)
PROB_THRESHOLD_HIGH = 0.55
PROB_THRESHOLD_LOW = 0.45
