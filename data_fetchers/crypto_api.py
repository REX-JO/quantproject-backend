from __future__ import annotations

import time

import pandas as pd
import requests

from models.cryptos.crypto_inference_features import (
    RAW_NUMERIC_COLUMNS,
    build_inference_features,
)


BINANCE_KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)
SUPPORTED_SYMBOLS = frozenset({"BTC", "ETH", "SOL", "XRP"})


def _fill_missing_hours(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("open_time").drop_duplicates("open_time").set_index("open_time")
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="1h", tz="UTC")
    frame = frame.reindex(full_index)
    missing = frame["close"].isna()
    previous_close = frame["close"].ffill()

    for column in ("open", "high", "low", "close"):
        frame.loc[missing, column] = previous_close.loc[missing]
    for column in (
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
    ):
        frame.loc[missing, column] = 0.0

    if frame.loc[:, RAW_NUMERIC_COLUMNS].isna().any().any():
        raise ValueError("crypto K-line data contains missing numeric values")
    return frame.rename_axis("open_time").reset_index()


def build_crypto_features_from_klines(
    symbol: str,
    payload: list[list[object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build crypto model features from Binance K-lines supplied by the frontend."""
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"unsupported crypto symbol: {symbol}")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Binance K-line data for {symbol} must be a non-empty array")
    if any(not isinstance(row, list) or len(row) != 12 for row in payload):
        raise ValueError(f"each Binance K-line row for {symbol} must contain 12 fields")

    frame = pd.DataFrame(payload, columns=BINANCE_KLINE_COLUMNS)
    for column in RAW_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["close_time"] = pd.to_numeric(frame["close_time"], errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)

    display_frame = _fill_missing_hours(frame.copy())

    now_ms = time.time_ns() // 1_000_000
    closed_frame = frame.loc[frame["close_time"] <= now_ms].copy()
    if closed_frame.empty:
        raise RuntimeError(f"Binance returned no closed K-line rows for {symbol}")

    closed_frame = _fill_missing_hours(closed_frame)
    features = build_inference_features(closed_frame)
    if len(features) < 24:
        raise ValueError(f"{symbol} has only {len(features)} complete feature rows; expected at least 24")
    return features, display_frame.tail(100).copy()


def fetch_crypto_features(
    symbol: str,
    *,
    limit: int = 300,
    timeout_seconds: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch closed Binance Spot 1h candles and build live inference features."""
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"unsupported crypto symbol: {symbol}")

    response = requests.get(
        "https://data-api.binance.vision/api/v3/klines",
        params={"symbol": f"{symbol}USDT", "interval": "1h", "limit": limit},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Binance returned no K-line rows for {symbol}")
    return build_crypto_features_from_klines(symbol, payload)
