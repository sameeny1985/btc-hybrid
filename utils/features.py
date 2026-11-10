"""
Feature engineering for Multivariate BiLSTM
Selected high-signal indicators (reduces noise)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import config

SELECTED_FEATURES = [
    "ret_1", "ret_3", "ret_5",
    "rsi_7", "rsi_14",
    "stoch_k",
    "macd_hist",
    "atr_ratio", "bb_pos", "bb_width", "rv_10",
    "dist_ema_9", "dist_ema_21",
    "adx",
    "body_ratio", "range_pct",
]


def ticks_to_dataframe(ticks: List[Tuple[int, float]]) -> pd.DataFrame:
    if not ticks:
        return pd.DataFrame()
    df = pd.DataFrame(ticks, columns=["timestamp", "price"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime").sort_index()
    ohlc = df["price"].resample("1min").ohlc()
    ohlc = ohlc.dropna()
    ohlc["volume"] = 1.0
    return ohlc


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    feats = pd.DataFrame(index=df.index)

    feats["ret_1"] = np.log(close / close.shift(1))
    feats["ret_3"] = np.log(close / close.shift(3))
    feats["ret_5"] = np.log(close / close.shift(5))

    feats["rsi_7"] = RSIIndicator(close, window=7).rsi() / 100.0
    feats["rsi_14"] = RSIIndicator(close, window=14).rsi() / 100.0

    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    feats["stoch_k"] = stoch.stoch() / 100.0

    macd = MACD(close)
    feats["macd_hist"] = macd.macd_diff() / (close + 1e-8)

    atr7 = AverageTrueRange(high, low, close, window=7).average_true_range()
    atr14 = AverageTrueRange(high, low, close, window=14).average_true_range()
    feats["atr_ratio"] = atr7 / (atr14 + 1e-8)

    bb = BollingerBands(close, window=20, window_dev=2)
    feats["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (close + 1e-8)
    feats["bb_pos"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)

    feats["rv_10"] = feats["ret_1"].rolling(10).std()

    ema9 = EMAIndicator(close, window=9).ema_indicator()
    ema21 = EMAIndicator(close, window=21).ema_indicator()
    feats["dist_ema_9"] = (close - ema9) / (close + 1e-8)
    feats["dist_ema_21"] = (close - ema21) / (close + 1e-8)

    try:
        feats["adx"] = ADXIndicator(high, low, close, window=14).adx() / 100.0
    except Exception:
        feats["adx"] = 0.25

    feats["body_ratio"] = (close - open_).abs() / (high - low + 1e-8)
    feats["range_pct"] = (high - low) / (close + 1e-8)

    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return feats[SELECTED_FEATURES]


def build_sequence_and_features(
    ticks: List[Tuple[int, float]],
    seq_len: int = config.SEQUENCE_LENGTH
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Returns:
        sequence: (1, seq_len, n_features) for Multivariate LSTM
        last_feats: (1, n_features)
        last_price: float
    """
    df = ticks_to_dataframe(ticks)
    if len(df) < seq_len + 15:
        return None, None, None

    feats = compute_features(df)
    if feats.empty or len(feats) < seq_len:
        return None, None, None

    seq = feats.iloc[-seq_len:].values.astype(np.float32)
    if seq.shape != (seq_len, len(SELECTED_FEATURES)):
        return None, None, None

    seq = seq.reshape(1, seq_len, len(SELECTED_FEATURES))
    last_feats = feats.iloc[-1].values.astype(np.float32).reshape(1, -1)
    last_price = float(df["close"].iloc[-1])
    return seq, last_feats, last_price


def get_feature_names() -> List[str]:
    return SELECTED_FEATURES.copy()
