"""
Multivariate Bidirectional LSTM - Robust version for Render
Uses selected technical indicators over last 60 minutes.
"""

import os
import numpy as np
from typing import Tuple
import joblib
from sklearn.preprocessing import StandardScaler
import config

TF_AVAILABLE = False
try:
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    # Limit memory growth to reduce OOM risk on free tier
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except Exception as e:
    print(f"[Models] TensorFlow not available: {e}")


class HybridPredictor:
    def __init__(self):
        self.lstm_model = None
        self.scaler = None
        self.is_ready = False
        self.version = "multi_bilstm_v3_robust"
        self.n_features = 16
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        self.scaler_path = os.path.join(config.MODEL_DIR, "feature_scaler.joblib")

    def _build_lstm(self, seq_len: int = 60, n_features: int = 16):
        # Smaller architecture to reduce memory / crash risk on free Render
        inp = Input(shape=(seq_len, n_features), name="sequence")
        x = Bidirectional(LSTM(64, return_sequences=True))(inp)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
        x = Dropout(0.2)(x)
        x = Dense(24, activation="relu")(x)
        x = Dropout(0.15)(x)
        out = Dense(1, activation="sigmoid", name="prob")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train_initial(self, sequences: np.ndarray, labels: np.ndarray, tabular=None, tabular_labels=None):
        if not TF_AVAILABLE:
            print("[Models] TensorFlow not available - cannot train")
            return
        try:
            if sequences.ndim != 3:
                print(f"[Models] Bad shape: {sequences.shape}")
                return
            n_samples, seq_len, n_features = sequences.shape
            self.n_features = n_features
            print(f"[Models] Training Multivariate BiLSTM: {n_samples} samples, shape={sequences.shape}")

            flat = sequences.reshape(-1, n_features)
            self.scaler = StandardScaler()
            flat_scaled = self.scaler.fit_transform(flat)
            sequences_scaled = flat_scaled.reshape(n_samples, seq_len, n_features).astype(np.float32)

            self.lstm_model = self._build_lstm(seq_len, n_features)

            es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)
            rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0)

            self.lstm_model.fit(
                sequences_scaled, labels,
                epochs=30,
                batch_size=32,
                validation_split=0.15,
                callbacks=[es, rlrop],
                verbose=0
            )
            self.is_ready = True
            self.save()
            print("[Models] Training completed and saved.")
        except Exception as e:
            print(f"[Models] Training failed: {e}")
            self.is_ready = False
            self.lstm_model = None

    def predict(self, sequence: np.ndarray, tabular=None) -> Tuple[float, str]:
        if not self.is_ready or self.lstm_model is None:
            return 0.5, "NEUTRAL"
        try:
            if sequence.ndim == 2:
                sequence = sequence.reshape(1, *sequence.shape)
            n_features = sequence.shape[-1]
            flat = sequence.reshape(-1, n_features)
            if self.scaler is not None:
                flat_scaled = self.scaler.transform(flat)
            else:
                flat_scaled = flat
            seq_scaled = flat_scaled.reshape(sequence.shape).astype(np.float32)
            prob = float(self.lstm_model.predict(seq_scaled, verbose=0)[0][0])

            if prob >= 0.53:
                direction = "UP"
            elif prob <= 0.47:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
            return prob, direction
        except Exception as e:
            print(f"[Models] Predict error: {e}")
            return 0.5, "NEUTRAL"

    def save(self):
        try:
            if self.lstm_model is not None:
                os.makedirs(config.MODEL_DIR, exist_ok=True)
                self.lstm_model.save(config.LSTM_MODEL_PATH)
                if self.scaler is not None:
                    joblib.dump(self.scaler, self.scaler_path)
        except Exception as e:
            print(f"[Models] Save error: {e}")

    def load(self) -> bool:
        try:
            if not TF_AVAILABLE:
                return False
            if not os.path.exists(config.LSTM_MODEL_PATH):
                return False
            self.lstm_model = load_model(config.LSTM_MODEL_PATH)
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            else:
                # Old model without scaler → force retrain
                print("[Models] Scaler missing - will retrain")
                self.lstm_model = None
                return False
            self.is_ready = True
            try:
                self.n_features = self.lstm_model.input_shape[-1]
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[Models] Load failed (will retrain): {e}")
            self.lstm_model = None
            self.scaler = None
            self.is_ready = False
            return False
