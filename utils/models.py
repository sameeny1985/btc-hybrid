"""
Pure LSTM model (no XGBoost) - lighter version for Render Free tier
"""

import os
import numpy as np
import joblib
from typing import Tuple, Optional
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU

try:
    import tensorflow as tf
    # Limit memory growth
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Limit CPU threads a bit
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception as e:
    print(f"[Models] TensorFlow error: {e}")
    TF_AVAILABLE = False


class HybridPredictor:
    """Name kept for compatibility, but now pure LSTM."""

    def __init__(self):
        self.lstm_model = None
        self.scaler = None
        self.is_ready = False
        self.version = "lstm_v1"
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    def _build_lstm(self, seq_len: int = 60):
        # Smaller model to save memory
        model = Sequential([
            LSTM(32, input_shape=(seq_len, 1), return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train_initial(self, sequences: np.ndarray, labels: np.ndarray,
                      tabular: np.ndarray = None, tabular_labels: np.ndarray = None):
        """Train only LSTM. tabular args kept for compatibility but ignored."""
        if not TF_AVAILABLE:
            print("[Models] TensorFlow not available")
            return

        print(f"[Models] Training LSTM on {len(sequences)} samples...")
        self.lstm_model = self._build_lstm(sequences.shape[1])

        es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0)
        self.lstm_model.fit(
            sequences, labels,
            epochs=25,
            batch_size=32,
            validation_split=0.15,
            callbacks=[es],
            verbose=0
        )
        self.is_ready = True
        self.save()
        print("[Models] LSTM training completed.")

    def predict(self, sequence: np.ndarray, tabular: np.ndarray = None) -> Tuple[float, str]:
        if not self.is_ready or self.lstm_model is None:
            return 0.5, "NEUTRAL"

        prob = float(self.lstm_model.predict(sequence, verbose=0)[0][0])

        if prob >= config.PROB_THRESHOLD_HIGH:
            direction = "UP"
        elif prob <= config.PROB_THRESHOLD_LOW:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        return prob, direction

    def fine_tune_lstm(self, sequences: np.ndarray, labels: np.ndarray, epochs: int = 3):
        if self.lstm_model is None or not TF_AVAILABLE:
            return
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=config.FINE_TUNE_LR),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        self.lstm_model.fit(sequences, labels, epochs=epochs, batch_size=16, verbose=0)
        self.save()

    def save(self):
        if self.lstm_model is not None:
            self.lstm_model.save(config.LSTM_MODEL_PATH)

    def load(self) -> bool:
        try:
            if os.path.exists(config.LSTM_MODEL_PATH) and TF_AVAILABLE:
                self.lstm_model = load_model(config.LSTM_MODEL_PATH)
                self.is_ready = True
                return True
        except Exception as e:
            print(f"[Models] Load failed: {e}")
        return False
