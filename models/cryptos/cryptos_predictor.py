from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.jit.mobile import _load_for_lite_interpreter


SUPPORTED_SYMBOLS = ("BTC", "ETH", "SOL", "XRP")


@dataclass(frozen=True)
class CryptoModelBundle:
    symbol: str
    model: Any
    feature_columns: tuple[str, ...]
    sequence_length: int
    display_threshold: float
    target_definition: str
    validation_auc: float
    training_coverage: dict[str, str]
    source_checkpoint: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_crypto_model(weights_dir: Path, symbol: str) -> CryptoModelBundle:
    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(f"unsupported crypto symbol: {symbol}")

    stem = f"{symbol.lower()}_4h_transformer"
    spec_path = Path(weights_dir) / f"{stem}.ptl.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    if spec.get("symbol") != symbol:
        raise ValueError(f"{spec_path.name} symbol does not match {symbol}")
    feature_columns = tuple(spec.get("feature_columns", ()))
    input_shape = spec.get("input", {}).get("shape", ())
    if len(feature_columns) != 53 or input_shape != ["batch", 24, 53]:
        raise ValueError(f"{spec_path.name} has an unsupported input contract")

    checkpoint_path = Path(weights_dir) / str(spec["source_checkpoint"])
    expected_checkpoint_hash = str(spec["source_checkpoint_sha256"]).lower()
    if _sha256(checkpoint_path).lower() != expected_checkpoint_hash:
        raise ValueError(f"{checkpoint_path.name} SHA-256 does not match its model specification")

    model_path = Path(weights_dir) / str(spec["model_file"])
    model = _load_for_lite_interpreter(str(model_path))
    return CryptoModelBundle(
        symbol=symbol,
        model=model,
        feature_columns=feature_columns,
        sequence_length=int(input_shape[1]),
        display_threshold=float(spec["display_threshold"]),
        target_definition=str(spec["target_definition"]),
        validation_auc=float(spec["validation_auc"]),
        training_coverage=dict(spec["training_coverage"]),
        source_checkpoint=checkpoint_path.name,
    )


def load_crypto_models(weights_dir: Path) -> dict[str, CryptoModelBundle]:
    return {
        symbol: load_crypto_model(weights_dir, symbol)
        for symbol in SUPPORTED_SYMBOLS
    }


def predict_latest_score(
    bundle: CryptoModelBundle,
    rows: pd.DataFrame,
) -> dict[str, float | str]:
    if list(rows.columns) != list(bundle.feature_columns):
        raise ValueError(f"{bundle.symbol} feature columns do not match the model specification")
    if len(rows) < bundle.sequence_length:
        raise ValueError(
            f"{bundle.symbol} needs {bundle.sequence_length} feature rows; got {len(rows)}"
        )

    values = rows.tail(bundle.sequence_length).to_numpy(dtype=np.float32).copy()
    if not np.isfinite(values).all():
        raise ValueError(f"{bundle.symbol} feature rows contain NaN or infinite values")

    with torch.no_grad():
        output = bundle.model(torch.from_numpy(values).unsqueeze(0))
    score = float(output[0].item())
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{bundle.symbol} model returned an invalid score: {score}")

    return {
        "up_score": score,
        "trend_label": "偏多" if score >= bundle.display_threshold else "偏空",
    }
