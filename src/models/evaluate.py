"""Evaluation metrics for the price model.

The model is trained on ``log1p(price)``; every helper here takes the
log-scale truth and prediction, and reports error metrics back on the
natural crore scale (what a reader actually cares about) alongside the
log-scale R2 that the hyper-parameter search optimised.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Crore price bands used for the held-out error breakdown. Upper edge ``inf``
# is serialised as ``None`` (JSON has no infinity).
DEFAULT_PRICE_BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, float("inf"))]


def regression_metrics(y_true_log, y_pred_log) -> dict:
    """Compute R2 (log and crore scale) plus MAE / RMSE (crore).

    Parameters
    ----------
    y_true_log, y_pred_log
        Ground-truth and predicted values on the ``log1p(price)`` scale.
    """
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "r2_crore": float(r2_score(y_true, y_pred)),
        "mae_crore": float(mean_absolute_error(y_true, y_pred)),
        "rmse_crore": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": int(y_true_log.shape[0]),
    }


def evaluate_pipeline(pipeline, X, y_log) -> dict:
    """Run ``pipeline.predict`` on ``X`` and score against ``y_log``."""
    return regression_metrics(y_log, pipeline.predict(X))


def cv_summary(search) -> dict:
    """Extract the cross-validation result for the *selected* candidate.

    This is the number that is comparable to the original notebook's
    ``search.best_score_`` — it is **not** the exported model's test score.
    """
    idx = int(search.best_index_)
    results = search.cv_results_
    n_splits = search.cv.get_n_splits() if hasattr(search.cv, "get_n_splits") else None
    return {
        "cv_folds": int(n_splits) if n_splits is not None else None,
        "scoring": "r2",
        "target": "log1p(price)",
        "best_cv_r2_mean": float(results["mean_test_score"][idx]),
        "best_cv_r2_std": float(results["std_test_score"][idx]),
        "n_candidates_evaluated": int(len(results["mean_test_score"])),
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
    }


def error_by_price_band(y_true_log, y_pred_log, bands=DEFAULT_PRICE_BANDS) -> list[dict]:
    """Held-out MAE / RMSE within crore price bands (natural scale).

    Per-band R2 is deliberately omitted: within a narrow band the target
    variance is tiny, so R2 becomes unstable and routinely goes negative even
    for good predictions. MAE is the honest readout. Absolute error is expected
    to grow with price -- this table quantifies that as a standing model
    limitation, independent of any train/test-split question.
    """
    y_true = np.expm1(np.asarray(y_true_log, dtype=float))
    y_pred = np.expm1(np.asarray(y_pred_log, dtype=float))
    n_total = int(y_true.shape[0])
    rows = []
    for lo, hi in bands:
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        rows.append(
            {
                "band_cr": [lo, None if hi == float("inf") else hi],
                "n": n,
                "share": round(n / n_total, 4) if n_total else 0.0,
                "mae_crore": float(mean_absolute_error(y_true[mask], y_pred[mask])) if n else None,
                "rmse_crore": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))) if n else None,
            }
        )
    return rows


def _sample_summary(x) -> dict:
    x = np.asarray(x, dtype=float)
    q = np.percentile(x, [25, 50, 75, 95, 99])
    return {
        "n": int(x.shape[0]),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "p25": float(q[0]),
        "p50": float(q[1]),
        "p75": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "max": float(x.max()),
    }


def distribution_comparison(train_values, test_values) -> dict:
    """KS 2-sample + Mann-Whitney U between two 1-D samples, plus summaries.

    Used to check whether the held-out split is representative of the training
    data on a given variable (here: price).
    """
    from scipy import stats

    a = np.asarray(train_values, dtype=float)
    b = np.asarray(test_values, dtype=float)
    ks = stats.ks_2samp(a, b)
    mw = stats.mannwhitneyu(a, b)
    return {
        "train": _sample_summary(a),
        "test": _sample_summary(b),
        "ks_2samp": {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)},
        "mann_whitney_u": {"pvalue": float(mw.pvalue)},
    }


def bootstrap_r2_ci(y_true_log, y_pred_log, n_boot=2000, alpha=0.05, random_state=42) -> dict:
    """Percentile bootstrap CI for R2 on the log target (paired resampling)."""
    yt = np.asarray(y_true_log, dtype=float)
    yp = np.asarray(y_pred_log, dtype=float)
    n = yt.shape[0]
    rng = np.random.default_rng(random_state)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = r2_score(yt[idx], yp[idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(r2_score(yt, yp)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_boot": int(n_boot),
        "alpha": alpha,
    }


def _jsonable(value):
    """np scalars -> plain Python so the summary can be json.dumps'd."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
