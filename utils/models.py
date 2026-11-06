"""
Hybrid LSTM + XGBoost models with continual learning support
"""

import os
import numpy as np
import joblib
from typing import Tuple, Optional, Dict
import config

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception as e:
    print(f"[Models] TensorFlow not available: {e}")
    TF_AVAILABLE = False

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    XGB_AVAILABLE = True
except Exception as e:
    print(f"[Models] XGBoost not available: {e}")
    XGB_AVAILABLE = False


class HybridPredictor:
    def __init__(self):
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = None
        self.is_ready = False
        self.version = "hybrid_v1"
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    def _build_lstm(self, seq_len: int = config.SEQUENCE_LENGTH) -> "tf.keras.Model":
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(seq_len, 1))),
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
                      tabular: np.ndarray, tabular_labels: np.ndarray):
        """
        sequences: (n, seq_len, 1)
        labels: (n,)  1 = UP, 0 = DOWN
        tabular: (n, n_features)
        """
        if not TF_AVAILABLE or not XGB_AVAILABLE:
            print("[Models] Cannot train - missing libraries")
            return

        # Scale tabular
        self.scaler = StandardScaler()
        tab_scaled = self.scaler.fit_transform(tabular)

        # LSTM
        self.lstm_model = self._build_lstm(sequences.shape[1])
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        self.lstm_model.fit(
            sequences, labels,
            epochs=40,
            batch_size=32,
            validation_split=0.15,
            callbacks=[es],
            verbose=0
        )

        # XGBoost on tabular + LSTM probability as extra feature
        lstm_probs = self.lstm_model.predict(sequences, verbose=0).flatten()
        xgb_input = np.column_stack([tab_scaled, lstm_probs])

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
        self.xgb_model.fit(xgb_input, tabular_labels)

        self.is_ready = True
        self.save()
        print("[Models] Initial training completed.")

    def predict(self, sequence: np.ndarray, tabular: np.ndarray) -> Tuple[float, str]:
        """
        Returns (probability_of_UP, direction_string)
        """
        if not self.is_ready or self.lstm_model is None or self.xgb_model is None:
            # Fallback random-ish (should not happen in production after bootstrap)
            return 0.5, "NEUTRAL"

        # LSTM prob
        lstm_prob = float(self.lstm_model.predict(sequence, verbose=0)[0][0])

        # Scale tabular
        tab_scaled = self.scaler.transform(tabular)
        xgb_input = np.column_stack([tab_scaled, [lstm_prob]])

        xgb_prob = float(self.xgb_model.predict_proba(xgb_input)[0][1])

        # Ensemble: weighted average (can be made adaptive later)
        # Give more weight to XGBoost when we have good recent accuracy
        final_prob = 0.4 * lstm_prob + 0.6 * xgb_prob

        if final_prob >= config.PROB_THRESHOLD_HIGH:
            direction = "UP"
        elif final_prob <= config.PROB_THRESHOLD_LOW:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        return final_prob, direction

    def fine_tune_lstm(self, sequences: np.ndarray, labels: np.ndarray, epochs: int = 5):
        """Light continual learning for LSTM."""
        if self.lstm_model is None or not TF_AVAILABLE:
            return
        # Very small learning rate
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=config.FINE_TUNE_LR),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        self.lstm_model.fit(sequences, labels, epochs=epochs, batch_size=16, verbose=0)
        self.save_lstm()

    def retrain_xgb(self, tabular: np.ndarray, labels: np.ndarray, lstm_probs: np.ndarray):
        if self.xgb_model is None or self.scaler is None:
            return
        tab_scaled = self.scaler.transform(tabular)
        xgb_input = np.column_stack([tab_scaled, lstm_probs])
        # Incremental-ish: fit again on recent data (full retrain on window)
        self.xgb_model.fit(xgb_input, labels)
        self.save_xgb()

    def save(self):
        self.save_lstm()
        self.save_xgb()
        if self.scaler is not None:
            joblib.dump(self.scaler, config.SCALER_PATH)

    def save_lstm(self):
        if self.lstm_model is not None:
            self.lstm_model.save(config.LSTM_MODEL_PATH)

    def save_xgb(self):
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, config.XGB_MODEL_PATH)

    def load(self) -> bool:
        try:
            if os.path.exists(config.LSTM_MODEL_PATH) and TF_AVAILABLE:
                self.lstm_model = load_model(config.LSTM_MODEL_PATH)
            if os.path.exists(config.XGB_MODEL_PATH) and XGB_AVAILABLE:
                self.xgb_model = joblib.load(config.XGB_MODEL_PATH)
            if os.path.exists(config.SCALER_PATH):
                self.scaler = joblib.load(config.SCALER_PATH)
            self.is_ready = (self.lstm_model is not None and self.xgb_model is not None and self.scaler is not None)
            return self.is_ready
        except Exception as e:
            print(f"[Models] Load failed: {e}")
            return False
