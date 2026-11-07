"""
Feature engineering for Hybrid LSTM + XGBoost
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import ta
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import config

def ticks_to_dataframe(ticks: List[Tuple[int, float]]) -> pd.DataFrame:
    """Convert list of (timestamp_ms, price) to DataFrame with 1-min resampling."""
    if not ticks:
        return pd.DataFrame()
    df = pd.DataFrame(ticks, columns=["timestamp", "price"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime").sort_index()
    # Resample to 1-minute OHLC-like (we only have last price, so close = price)
    ohlc = df["price"].resample("1min").ohlc()
    ohlc = ohlc.dropna()
    ohlc["volume"] = 1.0  # placeholder since we poll ticker, not real volume
    return ohlc


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-layer features:
    - Returns & price structure
    - Volatility regime
    - Momentum multi-period
    - Trend structure
    - Statistical moments
    """
    if len(df) < 50:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    feats = pd.DataFrame(index=df.index)

    # 1. Returns
    for lag in [1, 3, 5, 10, 15]:
        feats[f"ret_{lag}"] = np.log(close / close.shift(lag))

    # 2. Price structure
    feats["body_ratio"] = (close - open_).abs() / (high - low + 1e-8)
    feats["upper_shadow"] = (high - np.maximum(close, open_)) / (high - low + 1e-8)
    feats["lower_shadow"] = (np.minimum(close, open_) - low) / (high - low + 1e-8)
    feats["range_pct"] = (high - low) / close

    # Distance from recent extremes
    feats["dist_from_high_30"] = (close - high.rolling(30).max()) / close
    feats["dist_from_low_30"] = (close - low.rolling(30).min()) / close

    # 3. Volatility
    atr7 = AverageTrueRange(high, low, close, window=7)
    atr14 = AverageTrueRange(high, low, close, window=14)
    feats["atr_7"] = atr7.average_true_range() / close
    feats["atr_14"] = atr14.average_true_range() / close
    feats["atr_ratio"] = feats["atr_7"] / (feats["atr_14"] + 1e-8)

    bb = BollingerBands(close, window=20, window_dev=2)
    feats["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
    feats["bb_pos"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)

    # Realized vol
    feats["rv_10"] = feats["ret_1"].rolling(10).std()
    feats["rv_30"] = feats["ret_1"].rolling(30).std()

    # 4. Momentum multi-layer
    for period in [7, 14, 21]:
        rsi = RSIIndicator(close, window=period)
        feats[f"rsi_{period}"] = rsi.rsi() / 100.0

    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    feats["stoch_k"] = stoch.stoch() / 100.0
    feats["stoch_d"] = stoch.stoch_signal() / 100.0

    for period in [5, 10]:
        roc = ROCIndicator(close, window=period)
        feats[f"roc_{period}"] = roc.roc() / 100.0

    # 5. Trend
    for span in [9, 21, 50]:
        ema = EMAIndicator(close, window=span)
        feats[f"ema_{span}_slope"] = ema.ema_indicator().pct_change(3)
        feats[f"dist_ema_{span}"] = (close - ema.ema_indicator()) / close

    macd = MACD(close)
    feats["macd_hist"] = macd.macd_diff() / close
    feats["macd_hist_change"] = feats["macd_hist"].diff()

    try:
        adx = ADXIndicator(high, low, close, window=14)
        feats["adx"] = adx.adx() / 100.0
    except Exception:
        feats["adx"] = 0.25

    # 6. Statistical
    feats["skew_20"] = feats["ret_1"].rolling(20).skew()
    feats["kurt_20"] = feats["ret_1"].rolling(20).kurt()

    # Consecutive direction
    direction = np.sign(feats["ret_1"])
    feats["consec_up"] = direction.groupby((direction != direction.shift()).cumsum()).cumsum().clip(lower=0)
    feats["consec_down"] = (-direction).groupby((direction != -direction.shift()).cumsum()).cumsum().clip(lower=0)

    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return feats


def build_sequence_and_features(
    ticks: List[Tuple[int, float]],
    seq_len: int = config.SEQUENCE_LENGTH
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Returns:
        sequence: (1, seq_len, 1) for LSTM  - normalized log returns or prices
        tabular_features: (1, n_features) for XGBoost
        last_price: float
    """
    df = ticks_to_dataframe(ticks)
    if len(df) < seq_len + 10:
        return None, None, None

    feats = compute_features(df)
    if feats.empty or len(feats) < seq_len:
        return None, None, None

    # Sequence for LSTM: last seq_len log-returns
    closes = df["close"].values
    log_rets = np.diff(np.log(closes + 1e-12))
    if len(log_rets) < seq_len:
        return None, None, None

    seq = log_rets[-seq_len:]
    seq = seq.reshape(1, seq_len, 1).astype(np.float32)

    # Tabular: last row of features
    last_feats = feats.iloc[-1].values.astype(np.float32).reshape(1, -1)

    last_price = float(closes[-1])
    return seq, last_feats, last_price


def get_feature_names(df_sample: pd.DataFrame = None) -> List[str]:
    """Helper to get feature column names."""
    # Create a dummy to extract names if needed
    return [
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_15",
        "body_ratio", "upper_shadow", "lower_shadow", "range_pct",
        "dist_from_high_30", "dist_from_low_30",
        "atr_7", "atr_14", "atr_ratio", "bb_width", "bb_pos",
        "rv_10", "rv_30",
        "rsi_7", "rsi_14", "rsi_21",
        "stoch_k", "stoch_d",
        "roc_5", "roc_10",
        "ema_9_slope", "dist_ema_9", "ema_21_slope", "dist_ema_21", "ema_50_slope", "dist_ema_50",
        "macd_hist", "macd_hist_change", "adx",
        "skew_20", "kurt_20", "consec_up", "consec_down"
    ]
