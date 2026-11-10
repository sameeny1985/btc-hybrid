"""
Professional Multivariate Bidirectional LSTM for 10-minute BTC direction prediction.
Uses selected technical indicators + returns over the last 60 minutes.
"""

import os
import numpy as np
from typing import Tuple, Optional
import joblib
from sklearn.preprocessing import StandardScaler
import config

TF_AVAILABLE = False
try:
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, Dropout, Bidirectional,
        BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except Exception as e:
    print(f"[Models] TensorFlow error: {e}")


class HybridPredictor:
    """
    Multivariate BiLSTM.
    Input shape: (batch, 60, n_features)
    Output: probability that price will be higher in 10 minutes.
    """

    def __init__(self):
        self.lstm_model = None
        self.scaler = None
        self.is_ready = False
        self.version = "multi_bilstm_v2"
        self.n_features = 16
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        self.scaler_path = os.path.join(config.MODEL_DIR, "feature_scaler.joblib")

    def _build_lstm(self, seq_len: int = 60, n_features: int = 16):
        """
        Stronger architecture:
        - Bidirectional LSTM layers
        - BatchNormalization
        - Dropout for regularization
        """
        inp = Input(shape=(seq_len, n_features), name="sequence")

        x = Bidirectional(LSTM(96, return_sequences=True))(inp)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Bidirectional(LSTM(48, return_sequences=False))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Dense(32, activation="relu")(x)
        x = Dropout(0.15)(x)
        x = Dense(16, activation="relu")(x)

        out = Dense(1, activation="sigmoid", name="prob")(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=Adam(learning_rate=0.0008),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train_initial(self, sequences: np.ndarray, labels: np.ndarray,
                      tabular=None, tabular_labels=None):
        """
        sequences: (n_samples, seq_len, n_features)
        labels: (n_samples,)
        """
        if not TF_AVAILABLE:
            print("[Models] TensorFlow not available")
            return

        if sequences.ndim != 3:
            print(f"[Models] Expected 3D sequences, got shape {sequences.shape}")
            return

        n_samples, seq_len, n_features = sequences.shape
        self.n_features = n_features
        print(f"[Models] Training Multivariate BiLSTM on {n_samples} samples | shape={sequences.shape}")

        # Scale features
        flat = sequences.reshape(-1, n_features)
        self.scaler = StandardScaler()
        flat_scaled = self.scaler.fit_transform(flat)
        sequences_scaled = flat_scaled.reshape(n_samples, seq_len, n_features).astype(np.float32)

        self.lstm_model = self._build_lstm(seq_len, n_features)

        es = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        rlrop = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=0
        )

        self.lstm_model.fit(
            sequences_scaled, labels,
            epochs=40,
            batch_size=32,
            validation_split=0.18,
            callbacks=[es, rlrop],
            verbose=0
        )

        self.is_ready = True
        self.save()
        print("[Models] Multivariate BiLSTM training completed & saved.")

    def predict(self, sequence: np.ndarray, tabular=None) -> Tuple[float, str]:
        if not self.is_ready or self.lstm_model is None or self.scaler is None:
            return 0.5, "NEUTRAL"

        if sequence.ndim == 2:
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

        n_features = sequence.shape[-1]
        flat = sequence.reshape(-1, n_features)
        try:
            flat_scaled = self.scaler.transform(flat)
        except Exception:
            flat_scaled = flat
        seq_scaled = flat_scaled.reshape(sequence.shape).astype(np.float32)

        prob = float(self.lstm_model.predict(seq_scaled, verbose=0)[0][0])

        # آستانه‌های سخت‌گیرانه‌تر برای سیگنال‌های واقعی‌تر
        if prob >= 0.55:
            direction = "UP"
        elif prob <= 0.45:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        return prob, direction

    def save(self):
        if self.lstm_model is not None:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            self.lstm_model.save(config.LSTM_MODEL_PATH)
            if self.scaler is not None:
                joblib.dump(self.scaler, self.scaler_path)

    def load(self) -> bool:
        try:
            if os.path.exists(config.LSTM_MODEL_PATH) and TF_AVAILABLE:
                self.lstm_model = load_model(config.LSTM_MODEL_PATH)
                if os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)
                else:
                    print("[Models] Scaler not found – model may need retrain")
                    return False
                self.is_ready = True
                try:
                    self.n_features = self.lstm_model.input_shape[-1]
                except Exception:
                    pass
                return True
        except Exception as e:
            print(f"[Models] Load failed: {e}")
        return False
