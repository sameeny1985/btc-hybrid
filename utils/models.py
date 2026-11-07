"""
Pure LSTM + TensorFlow only (no XGBoost)
"""

import os
import numpy as np
from typing import Tuple
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception as e:
    print(f"[Models] TensorFlow error: {e}")
    TF_AVAILABLE = False


class HybridPredictor:
    """Name kept for compatibility — pure LSTM only."""

    def __init__(self):
        self.lstm_model = None
        self.is_ready = False
        self.version = "lstm_only_v1"
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    def _build_lstm(self, seq_len: int = 60):
        model = Sequential([
            Input(shape=(seq_len, 1)),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32, return_sequences=False)),
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
                      tabular=None, tabular_labels=None):
        """Train LSTM only. tabular args ignored (kept for call compatibility)."""
        if not TF_AVAILABLE:
            print("[Models] TensorFlow not available")
            return
        print(f"[Models] Training LSTM on {len(sequences)} samples...")
        self.lstm_model = self._build_lstm(sequences.shape[1])
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)
        self.lstm_model.fit(
            sequences, labels,
            epochs=35,
            batch_size=32,
            validation_split=0.15,
            callbacks=[es],
            verbose=0
        )
        self.is_ready = True
        self.save()
        print("[Models] LSTM training completed.")

    def predict(self, sequence: np.ndarray, tabular=None) -> Tuple[float, str]:
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
