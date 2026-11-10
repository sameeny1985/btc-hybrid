import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MEXC_BASE_URL = os.getenv("MEXC_BASE_URL", "https://api.mexc.com").rstrip("/")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").upper()

DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))
DB_PATH = os.getenv("DB_PATH", str(Path(DATA_DIR) / "predictor.db"))
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", str(Path(MODEL_DIR) / "lstm_model.keras"))

SEQUENCE_LENGTH = 60
LOOKAHEAD_MINUTES = 10
MIN_SAMPLES_FOR_TRAIN = 150
RETRAIN_EVERY_N_PREDICTIONS = 30
FINE_TUNE_LR = 1e-5
PROB_THRESHOLD_HIGH = 0.53
PROB_THRESHOLD_LOW = 0.47
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))
PREDICTION_INTERVAL_SECONDS = 600
