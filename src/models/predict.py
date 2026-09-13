"""Load the exported pipeline and turn feature rows into price predictions.

The pipeline is trained on ``log1p(price)``; :func:`predict` inverts that so
callers always get price in crore.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .train import DEFAULT_MODEL_OUT, FEATURE_COLS


def load_model(path: str | Path = DEFAULT_MODEL_OUT):
    """Load the pickled pipeline produced by :func:`src.models.train.train_and_export`."""
    return joblib.load(path)


def predict(model, X) -> np.ndarray:
    """Predict price in crore for a DataFrame (or list of dicts) of raw rows.

    Columns are reindexed to the model's expected feature order; unknown extra
    columns are dropped and any missing feature column raises inside the model.
    """
    X = pd.DataFrame(X).reindex(columns=FEATURE_COLS)
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"missing / unmapped feature values for columns: {bad}")
    return np.expm1(model.predict(X))


def predict_one(model, **features) -> float:
    """Convenience wrapper: keyword feature values -> a single price (crore)."""
    return float(predict(model, [features])[0])
