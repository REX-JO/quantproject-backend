from __future__ import annotations

import numpy as np
import pandas as pd


LAGS = (1, 2, 3, 6, 12, 24)
ROLLING_WINDOWS = (6, 12, 24)
RAW_NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
)
EXPECTED_FEATURE_NAMES = (
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "log_return",
    "range_pct",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "atr_14_pct",
    "log_volume",
    "volume_change",
    "trade_intensity",
    "taker_buy_ratio",
    "quote_volume_change",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "log_return_lag_1",
    "volume_change_lag_1",
    "taker_buy_ratio_lag_1",
    "log_return_lag_2",
    "volume_change_lag_2",
    "taker_buy_ratio_lag_2",
    "log_return_lag_3",
    "volume_change_lag_3",
    "taker_buy_ratio_lag_3",
    "log_return_lag_6",
    "volume_change_lag_6",
    "taker_buy_ratio_lag_6",
    "log_return_lag_12",
    "volume_change_lag_12",
    "taker_buy_ratio_lag_12",
    "log_return_lag_24",
    "volume_change_lag_24",
    "taker_buy_ratio_lag_24",
    "return_mean_6",
    "return_std_6",
    "volume_mean_6",
    "range_mean_6",
    "return_mean_12",
    "return_std_12",
    "volume_mean_12",
    "range_mean_12",
    "return_mean_24",
    "return_std_24",
    "volume_mean_24",
    "range_mean_24",
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "bollinger_z_20",
)


def _prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"open_time", *RAW_NUMERIC_COLUMNS}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"missing raw columns: {', '.join(missing)}")

    frame = raw.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    if frame["open_time"].isna().any():
        raise ValueError("open_time contains invalid timestamps")
    frame = frame.sort_values("open_time").drop_duplicates("open_time").set_index("open_time")
    for column in RAW_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_inference_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the exact 53-feature contract used to train the 4h models."""
    frame = _prepare_raw(raw)
    close = frame["close"].clip(lower=1e-12)
    volume = frame["volume"].clip(lower=0)

    frame["log_return"] = np.log(close / close.shift(1))
    frame["range_pct"] = (frame["high"] - frame["low"]) / close
    frame["body_pct"] = (frame["close"] - frame["open"]) / frame["open"].replace(0, np.nan)
    frame["upper_shadow_pct"] = (
        frame["high"] - frame[["open", "close"]].max(axis=1)
    ) / close
    frame["lower_shadow_pct"] = (
        frame[["open", "close"]].min(axis=1) - frame["low"]
    ) / close

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14_pct"] = true_range.rolling(14).mean() / close

    frame["log_volume"] = np.log1p(volume)
    frame["volume_change"] = np.log1p(volume) - np.log1p(volume.shift(1))
    positive_volume = volume.gt(0)
    frame["trade_intensity"] = (
        frame["number_of_trades"] / volume
    ).where(positive_volume, 0.0)
    frame["taker_buy_ratio"] = (
        frame["taker_buy_base"] / volume
    ).where(positive_volume, 0.0)
    quote_volume = frame["quote_asset_volume"].clip(lower=0)
    frame["quote_volume_change"] = np.log1p(quote_volume) - np.log1p(quote_volume.shift(1))

    hour = frame.index.hour
    day = frame.index.dayofweek
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * day / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * day / 7)

    for lag in LAGS:
        frame[f"log_return_lag_{lag}"] = frame["log_return"].shift(lag)
        frame[f"volume_change_lag_{lag}"] = frame["volume_change"].shift(lag)
        frame[f"taker_buy_ratio_lag_{lag}"] = frame["taker_buy_ratio"].shift(lag)

    for window in ROLLING_WINDOWS:
        frame[f"return_mean_{window}"] = frame["log_return"].rolling(window).mean()
        frame[f"return_std_{window}"] = frame["log_return"].rolling(window).std()
        frame[f"volume_mean_{window}"] = frame["log_volume"].rolling(window).mean()
        frame[f"range_mean_{window}"] = frame["range_pct"].rolling(window).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    frame["rsi_14"] = 100 - (100 / (1 + relative_strength))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    frame["macd_pct"] = macd / close
    frame["macd_signal_pct"] = macd.ewm(span=9, adjust=False).mean() / close

    rolling_close_mean = close.rolling(20).mean()
    rolling_close_std = close.rolling(20).std()
    frame["bollinger_z_20"] = (
        close - rolling_close_mean
    ) / rolling_close_std.replace(0, np.nan)

    features = frame.loc[:, EXPECTED_FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    return features.dropna()
