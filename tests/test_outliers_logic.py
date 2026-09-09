"""IQR / threshold logic in src/preprocessing/outliers.py.

Toy frames use an index far from any hard-coded row in _MANUAL_AREA_DROP /
_MANUAL_AREA_FIXES / _MANUAL_CARPET_FIXES so those guarded loops never fire.
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.outliers import (
    _iqr_bounds,
    _apply_threshold_filters,
    PRICE_PER_SQFT_MAX,
    AREA_MAX,
    BEDROOM_MAX,
)


def test_iqr_bounds_are_tukey_fences():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    q1, q3 = s.quantile(0.25), s.quantile(0.75)   # 3.0, 7.0
    iqr = q3 - q1
    lo, hi = _iqr_bounds(s)
    assert (lo, hi) == pytest.approx((q1 - 1.5 * iqr, q3 + 1.5 * iqr))
    assert (lo, hi) == pytest.approx((-3.0, 13.0))


def _toy_frame():
    # each row is tagged by `survives` so the assertion is self-describing.
    rows = [
        # price_per_sqft, area, bedRoom, survives, why
        (40_000, 90_000, 8, True, "all within caps"),
        (60_000, 90_000, 8, False, "price_per_sqft > 50000"),
        (40_000, 150_000, 8, False, "area >= 100000"),
        (40_000, AREA_MAX, 8, False, "area == 100000 (strict <)"),
        (40_000, 90_000, 12, False, "bedRoom > 10"),
        (np.nan, 90_000, 8, False, "null price_per_sqft fails NaN <= cap"),
        (49_999, 99_999, 10, True, "just inside every cap"),
    ]
    df = pd.DataFrame(
        rows, columns=["price_per_sqft", "area", "bedRoom", "survives", "why"],
        index=range(9000, 9000 + len(rows)),
    )
    df["price"] = 1.0        # unused by the filter, present for realism
    return df


def test_threshold_filters_keep_exactly_the_expected_rows():
    df = _toy_frame()
    kept = _apply_threshold_filters(df)
    assert list(kept.index) == list(df.index[df["survives"]])
    assert set(kept["why"]) == {"all within caps", "just inside every cap"}


def test_caps_match_documented_values():
    assert (PRICE_PER_SQFT_MAX, AREA_MAX, BEDROOM_MAX) == (50_000, 100_000, 10)
