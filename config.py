import os
from dotenv import load_dotenv

load_dotenv()

# ========== این بخش را با اطلاعات خودتان پر کنید ==========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
# ========================================================

# MEXC
MEXC_BASE_URL = "https://api.mexc.com"
SYMBOL = "BTCUSDT"

# Local storage paths
DATA_DIR = "data"
MODEL_DIR = "models"
DB_PATH = os.path.join(DATA_DIR, "predictor.db")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")

# Model settings
SEQUENCE_LENGTH = 60
LOOKAHEAD_MINUTES = 10
MIN_SAMPLES_FOR_TRAIN = 150
RETRAIN_EVERY_N_PREDICTIONS = 30
FINE_TUNE_LR = 1e-5

PROB_THRESHOLD_HIGH = 0.51
PROB_THRESHOLD_LOW = 0.49
