"""Explainability for the exported price model (``PROJECT_PLAN.md``, the *Modelling* section).

The shipped pipeline (``models/price_pipeline.pkl``) is a
``RandomForestRegressor`` on top of a ``ColumnTransformer``, trained on
``log1p(price)``. **Nothing here refits it** -- ``shap.TreeExplainer`` reads the
fitted trees directly, and the training split for the global summary comes from
:func:`src.models.train.split_dataset` (same ``test_size`` / ``random_state`` as
training), so the summary is computed on exactly the rows the model saw.

The insights notebook's standardised-linear-coefficient method is deliberately
**not** ported: it assumes a linear model and is meaningless for a forest. Global
importance here is ``feature_importances_`` (impurity) plus mean ``|SHAP|``;
per-prediction explanation is SHAP ``TreeExplainer``.

SHAP units
----------
SHAP values are in the model's output space, ``log1p(price)``. They are additive
*there* (``sum(shap) + base_value == model log-output``), and
``expm1(model log-output)`` is the crore prediction. They do **not** pass through
``expm1`` linearly, so this module reports per-feature contributions in log units
and the crore price separately. Sign and rank carry over to price; the
magnitudes are log-space.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from .predict import load_model
from .train import DEFAULT_MODEL_OUT, FEATURE_COLS, REPO_ROOT, _rel, load_dataset, split_dataset

DEFAULT_IMPORTANCE_JSON = REPO_ROOT / "reports" / "model" / "feature_importance.json"
DEFAULT_IMPORTANCE_MD = REPO_ROOT / "reports" / "model" / "explainability_summary.md"

# transformed-feature name -> friendly label (only one column is renamed by the
# preprocessor: property_type -> property_type_house via drop="if_binary")
_DISPLAY = {"property_type_house": "property_type (house=1)"}


def _feature_label(name: str) -> str:
    return _DISPLAY.get(name, name)


def _raw_col(feature_name: str) -> str:
    """Transformed feature name -> the raw input column it came from."""
    return "property_type" if feature_name == "property_type_house" else feature_name


def _scalar(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


class PriceExplainer:
    """The exported pipeline + a ``shap.TreeExplainer`` over its forest step.

    Build once and reuse (the app's prediction page should hold a single
    instance). ``model`` defaults to :func:`src.models.predict.load_model`.
    """

    def __init__(self, model=None):
        self.pipeline = load_model() if model is None else model
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.regressor = self.pipeline.named_steps["regressor"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        # no data passed -> feature_perturbation="tree_path_dependent": exact,
        # additive, uses only the fitted trees (no background set, no refit).
        self.explainer = shap.TreeExplainer(self.regressor)
        self.base_value_log = float(np.ravel(self.explainer.expected_value)[0])

    # ------------------------------------------------------------------ #
    def _transform(self, raw: pd.DataFrame) -> np.ndarray:
        missing = [c for c in FEATURE_COLS if c not in raw.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        return self.preprocessor.transform(raw[FEATURE_COLS])

    # ------------------------------------------------------------------ #
    def explain_property(self, raw_row, top_n: int = 5) -> dict:
        """Explain one property.

        ``raw_row``: dict or 1-row DataFrame with the 12 ``FEATURE_COLS`` (the
        same shape :func:`src.models.predict.predict` takes). Returns predicted
        price (crore), the model's base value, and every feature's signed SHAP
        contribution (log units), plus the ``top_n`` pushing the price up and the
        ``top_n`` pushing it down.
        """
        raw = pd.DataFrame([raw_row] if isinstance(raw_row, dict) else raw_row).reset_index(drop=True)
        Xt = self._transform(raw)
        shap_row = np.ravel(self.explainer.shap_values(Xt))
        pred_log = float(self.regressor.predict(Xt)[0])
        residual = pred_log - self.base_value_log - float(shap_row.sum())  # ~0 (exact SHAP)

        contribs = [
            {
                "feature": name,
                "label": _feature_label(name),
                "input_value": _scalar(raw.iloc[0][_raw_col(name)]),
                "shap_log": float(shap_row[i]),
                "direction": "up" if shap_row[i] > 0 else "down" if shap_row[i] < 0 else "flat",
            }
            for i, name in enumerate(self.feature_names)
        ]
        contribs.sort(key=lambda d: abs(d["shap_log"]), reverse=True)

        return {
            "predicted_price_cr": float(np.expm1(pred_log)),
            "base_value_cr": float(np.expm1(self.base_value_log)),
            "prediction_log": pred_log,
            "base_value_log": self.base_value_log,
            "additivity_residual_log": residual,
            "shap_units": "log1p(price) -- additive in log space, not in crore",
            "contributions": contribs,
            "pushed_up": [c for c in contribs if c["direction"] == "up"][:top_n],
            "pushed_down": [c for c in contribs if c["direction"] == "down"][:top_n],
        }

    # ------------------------------------------------------------------ #
    def global_importance(
        self,
        X: pd.DataFrame | None = None,
        sample_size: int | None = 400,
        random_state: int = 42,
    ) -> dict:
        """Impurity importance + mean ``|SHAP|`` per feature.

        ``X`` defaults to the training split (the rows the shipped model was fit
        on), via :func:`src.models.train.split_dataset`. Path-dependent SHAP on
        this 500-tree forest costs ~0.25 s/row, so by default a deterministic
        ``sample_size`` of rows is used for the mean ``|SHAP|`` estimate; pass
        ``sample_size=None`` to use every row. The RF impurity importances are
        exact regardless.
        """
        if X is None:
            X_all, y_all = load_dataset()
            X, _, _, _ = split_dataset(X_all, np.log1p(y_all))
        n_available = len(X)
        if sample_size is not None and sample_size < n_available:
            X_shap = X.sample(n=sample_size, random_state=random_state)
        else:
            X_shap = X
        Xt = self.preprocessor.transform(X_shap[FEATURE_COLS])
        shap_vals = self.explainer.shap_values(Xt)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        rf_imp = self.regressor.feature_importances_

        rows = [
            {
                "feature": name,
                "label": _feature_label(name),
                "mean_abs_shap_log": float(mean_abs[i]),
                "rf_impurity_importance": float(rf_imp[i]),
            }
            for i, name in enumerate(self.feature_names)
        ]
        by_shap = sorted(rows, key=lambda r: r["mean_abs_shap_log"], reverse=True)
        by_rf = sorted(rows, key=lambda r: r["rf_impurity_importance"], reverse=True)
        return {
            "n_rows_explained": int(X_shap.shape[0]),
            "n_rows_available": int(n_available),
            "basis": (
                f"random {X_shap.shape[0]}-row sample (random_state={random_state}) of the "
                "training split (test_size=0.2, random_state=42) -- the rows the shipped model was fit on"
                if X_shap.shape[0] < n_available
                else "full training split (test_size=0.2, random_state=42) -- the rows the shipped model was fit on"
            ),
            "shap_units": "log1p(price)",
            "base_value_log": self.base_value_log,
            "base_value_cr": float(np.expm1(self.base_value_log)),
            "features": by_shap,
            "shap_rank": [r["feature"] for r in by_shap],
            "rf_impurity_rank": [r["feature"] for r in by_rf],
        }

    # ------------------------------------------------------------------ #
    def write_global_reports(
        self,
        X: pd.DataFrame | None = None,
        json_path: str | Path = DEFAULT_IMPORTANCE_JSON,
        md_path: str | Path = DEFAULT_IMPORTANCE_MD,
    ) -> dict:
        """Compute the global summary and write the JSON + Markdown to ``reports/``."""
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_file": _rel(DEFAULT_MODEL_OUT),
            **self.global_importance(X),
        }
        json_path, md_path = Path(json_path), Path(md_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(_render_md(report), encoding="utf-8")
        return report


def _render_md(r: dict) -> str:
    lines = [
        "# Model explainability summary",
        "",
        f"_Generated by `src/models/explain.py` on {r['created_utc']} -- do not hand-edit._",
        "",
        "Global feature importance for the exported model (`models/price_pipeline.pkl`),",
        f"over the {r['basis']}.",
        "",
        "- **`mean_abs_shap_log`** -- mean |SHAP value| per feature, in `log1p(price)` units.",
        "  Additive in log space; **not** convertible to a fixed number of crore.",
        "- **`rf_impurity_importance`** -- the forest's built-in `feature_importances_`",
        "  (mean impurity decrease). Fast but biased toward high-cardinality features.",
        f"- Model base value (mean output): {r['base_value_log']:.4f} log = ~Rs {r['base_value_cr']:.3f} Cr.",
        "",
        "| Rank | Feature | mean \\|SHAP\\| (log) | RF impurity |",
        "|---|---|---|---|",
    ]
    for i, feat in enumerate(r["features"], 1):
        lines.append(
            f"| {i} | `{feat['label']}` | {feat['mean_abs_shap_log']:.4f} | {feat['rf_impurity_importance']:.4f} |"
        )
    lines += [
        "",
        f"SHAP rank: {' > '.join(f'`{f}`' for f in r['shap_rank'])}",
        "",
        f"RF impurity rank: {' > '.join(f'`{f}`' for f in r['rf_impurity_rank'])}",
        "",
        _MANUAL_CHECKS,
        "Feeds `docs/model_documentation.md` (not yet written).",
        "",
    ]
    return "\n".join(lines)


# Static: findings from manual review of specific rows (do not depend on a run).
_MANUAL_CHECKS = """## Data-quality checks (manual)

**Row 33 — replaced in the demo (checked during review).** An earlier draft used
row 33 (`society` "greenopolis", sector 89, Rs 0.70 Cr, 1297 sqft, 2 BHK) as the
sub-Rs 1 Cr example; its predicted Rs 1.10 Cr missed by Rs 0.40 Cr, about 2.4x
the 0-1 Cr band's average MAE. Tracing it to source:

- The **price is internally consistent and plausible if low** --
  Rs 0.70 Cr / 1297 sqft = Rs 5,397/sqft, matching the stored `price_per_sqft`;
  no lakh/crore or area-unit parsing error. A bare 2 BHK resale in Greenopolis
  (a delayed-delivery project) can sit around this figure.
- But the row is **mislabelled `property_type = "house"`** while being a
  **14th-floor unit** (`floorNum = 14`) in Greenopolis, a high-rise apartment
  complex. SHAP shows `property_type = house` adding +0.115 log (~+12 %) to the
  prediction -- a spurious independent-house premium applied to a flat, which
  accounts for most of the miss. `agePossession` was also imputed
  ("Undefined" -> "New Property").

Replaced by row 2757 (Shree Vardhman Flora, sector 90, 2 BHK flat, Rs 0.70 Cr ->
predicted Rs 0.84 Cr, a typical band miss). **Flag for the cleaning rebuild:**
high-floor rows in apartment-society projects labelled `property_type = "house"`
corrupt the feature vector, not just the prediction, and should be relabelled.
"""


# --- thin module-level wrappers (one-off use; prefer holding a PriceExplainer) --
def explain_property(raw_row, model=None, top_n: int = 5) -> dict:
    return PriceExplainer(model).explain_property(raw_row, top_n=top_n)


def global_importance(model=None, X: pd.DataFrame | None = None) -> dict:
    return PriceExplainer(model).global_importance(X)


if __name__ == "__main__":
    import pprint

    rep = PriceExplainer().write_global_reports()
    print("wrote reports/model/feature_importance.json + explainability_summary.md\n")
    print(f"{'feature':<26}{'mean|SHAP|(log)':>18}{'RF impurity':>14}")
    for f in rep["features"]:
        print(f"{f['label']:<26}{f['mean_abs_shap_log']:>18.4f}{f['rf_impurity_importance']:>14.4f}")
