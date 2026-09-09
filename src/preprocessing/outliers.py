"""
Outlier treatment for the Gurgaon properties pipeline.

``treat_outliers()`` is reconstructed from ``outlier-treatment.ipynb`` and
value-checked against the real ``gurgaon_properties_cleaned_v2.csv`` ->
``gurgaon_properties_outlier_treated.csv`` pair: **24/24 columns,
3555/3555 rows, 85,320/85,320 cells exact match** (see ``PROJECT_PLAN.md``
Section 9).

**The uploaded notebook does NOT fully produce its output.** Three
transformations present in the output have no code in the notebook and were
recovered from the input->output diff:

1. the ``area_room_ratio`` column (``area / bedRoom``) -- no cell creates it;
2. a drop of 33 hand-picked low ``area / bedRoom`` rows (garbage listings such
   as a 145 sqft "2 BHK") -- a subset of the ``area / bedRoom < 183`` rows, but
   not reproducible by any threshold;
3. a downward correction of ``bedRoom`` on 70 ``property_type == 'house'`` rows
   (from 4-10 down to 1-4). Not reproducible by any formula, and **not** a join
   to ``houses.csv`` / ``independent-house.csv`` -- those carry the same
   inflated counts (checked: raw ``bedRoom`` matches the *original* value 56/57
   times, the corrected value 0/57).

Items 2 and 3, plus the notebook's own hand-typed index edits (cells 33/35/67),
are quarantined below as ``_RECONSTRUCTED_*`` / ``_MANUAL_*`` constant tables.
**They are replay-only**: every key is a positional row index into
``gurgaon_properties_cleaned_v2.csv`` (labels survive ``drop_duplicates`` and
are kept until the final ``reset_index``). If that file's row order ever
changes, these edits silently land on the wrong rows -- regenerate them from a
fresh diff.

Weak spots in the *derivable* logic, flagged rather than silently kept/fixed:

- **``drop_duplicates()`` on the raw load** -- no ``subset``, no logging, and it
  also collapses rows that differ only in a column dropped later.
- **``area = area * 9`` for ``area < 1000``** on the price_per_sqft-IQR outliers
  -- a blanket "this was recorded in square yards" assumption; a genuinely
  small flat in that set gets inflated 9x.
- **The price-column IQR analysis (notebook cells 6-10) is dead code** -- it
  computes bounds and displays outliers but changes nothing. Not ported.
- **Null ``price`` / ``area`` / ``price_per_sqft`` rows (18 each) are removed
  only as a side effect** of ``NaN <= 50000`` being ``False`` in the threshold
  filters, not by an explicit ``dropna``. Reproduced as-is.
- **``price_per_sqft`` is recomputed twice** (once for the IQR outliers, once
  for every row at the end); the first is redundant. Kept for fidelity.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRICE_PER_SQFT_MAX = 50_000
AREA_MAX = 100_000
BEDROOM_MAX = 10

# price_per_sqft-IQR outlier rows with area below this are assumed to be in
# square yards and converted to square feet (1 sq yd = 9 sq ft).
SQYD_AREA_THRESHOLD = 1_000
SQYD_TO_SQFT = 9

# integer columns that pandas < 2.0 DataFrame.update upcast to float64 as a
# side effect (see _fix_price_per_sqft_outliers); the committed CSVs carry them
# as float, so the port matches.
_UPDATE_UPCASTS_TO_FLOAT = [
    "bedRoom", "bathroom", "study room", "servant room", "store room",
    "pooja room", "others", "furnishing_type", "luxury_score",
]

# Crore -> rupees, for price_per_sqft = price / area.
_CRORE = 10_000_000

# --- hand-typed edits from the notebook (cells 33 / 35 / 67) ---------------- #
# All keys are positional row indices in gurgaon_properties_cleaned_v2.csv.
_MANUAL_AREA_DROP = [818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471]
_MANUAL_AREA_FIXES = {
    48: 115 * 9, 300: 7250, 2666: 5800, 1358: 2660,
    3195: 2850, 2131: 1812, 3088: 2160, 3444: 1175,
}
_MANUAL_CARPET_FIXES = {2131: 1812}

# --- edits with NO notebook cell, recovered from the input->output diff ----- #
# 33 low area/bedRoom "garbage" rows the author dropped by hand.
_RECONSTRUCTED_ROW_DROP = [
    37, 93, 229, 247, 387, 751, 753, 1106, 1206, 1429, 1562, 1580, 1696, 1737,
    1747, 1773, 1936, 1939, 1953, 1997, 2047, 2300, 2360, 2721, 2784, 2806,
    3148, 3246, 3268, 3306, 3329, 3633, 3774,
]
# 70 property_type == 'house' rows whose bedRoom was corrected downward.
_RECONSTRUCTED_HOUSE_BEDROOM = {
    9: 3, 15: 2, 48: 3, 74: 2, 99: 3, 140: 2, 186: 2, 255: 2, 293: 2, 343: 3,
    393: 3, 530: 2, 540: 3, 565: 2, 668: 3, 837: 4, 848: 2, 852: 3, 880: 3,
    886: 1, 935: 2, 1033: 2, 1049: 3, 1087: 3, 1167: 2, 1187: 3, 1213: 2,
    1224: 2, 1384: 2, 1407: 3, 1480: 4, 1500: 2, 1509: 2, 1524: 2, 1532: 1,
    1537: 2, 1627: 2, 1798: 3, 1851: 1, 1951: 2, 2004: 2, 2063: 3, 2173: 2,
    2181: 3, 2244: 2, 2261: 4, 2266: 4, 2277: 1, 2313: 2, 2333: 3, 2343: 2,
    2386: 2, 2461: 4, 2660: 2, 2722: 2, 2984: 2, 3045: 3, 3101: 3, 3198: 2,
    3252: 3, 3254: 2, 3282: 2, 3309: 2, 3328: 2, 3347: 2, 3364: 1, 3546: 2,
    3686: 2, 3713: 2, 3738: 2,
}

OUTPUT_COLUMNS = [
    "property_type", "society", "sector", "price", "price_per_sqft", "area",
    "areaWithType", "bedRoom", "bathroom", "balcony", "floorNum", "facing",
    "agePossession", "super_built_up_area", "built_up_area", "carpet_area",
    "study room", "servant room", "store room", "pooja room", "others",
    "furnishing_type", "luxury_score", "area_room_ratio",
]


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Tukey fences: (Q1 - 1.5 IQR, Q3 + 1.5 IQR)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _price_per_sqft(price_cr: pd.Series, area_sqft: pd.Series) -> pd.Series:
    return np.round((price_cr * _CRORE) / area_sqft)


def _fix_price_per_sqft_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """For the ``price_per_sqft`` IQR outliers: treat a sub-1000 ``area`` as
    square yards (``* 9``), then recompute ``price_per_sqft`` from
    ``price / area`` and write both columns back (notebook cells 14-18)."""
    df = df.copy()
    lo, hi = _iqr_bounds(df["price_per_sqft"])
    outliers = df[(df["price_per_sqft"] < lo) | (df["price_per_sqft"] > hi)].copy()

    outliers["area"] = outliers["area"].apply(
        lambda x: x * SQYD_TO_SQFT if x < SQYD_AREA_THRESHOLD else x
    )
    outliers["price_per_sqft"] = _price_per_sqft(outliers["price"], outliers["area"])
    df.update(outliers)  # aligns on index; only non-NaN values overwrite

    # pandas < 2.0 DataFrame.update upcast every column present in `other` to
    # float64 (its internal where() fills the unmatched rows with NaN). pandas
    # >= 2.0 preserves dtype, so the committed CSVs -- built on the old
    # behaviour -- have these integer columns as float. Reproduce that here.
    for col in _UPDATE_UPCASTS_TO_FLOAT:
        df[col] = df[col].astype(float)

    logger.info("price_per_sqft IQR outliers rescaled: %d row(s)", len(outliers))
    return df


def _apply_threshold_filters(df: pd.DataFrame) -> pd.DataFrame:
    """``price_per_sqft <= 50000``, ``area < 100000``, ``bedRoom <= 10`` --
    interleaved with the manual index edits in the notebook's order.

    Rows with a null ``price_per_sqft`` / ``area`` are dropped here as a side
    effect (``NaN`` fails every comparison); the notebook relies on this rather
    than a ``dropna``.
    """
    df = df.copy()

    before = len(df)
    df = df[df["price_per_sqft"] <= PRICE_PER_SQFT_MAX]
    df = df[df["area"] < AREA_MAX]
    logger.info("price_per_sqft/area threshold filters: %d -> %d row(s)", before, len(df))

    df = df.drop(index=[i for i in _MANUAL_AREA_DROP if i in df.index])
    for idx, value in _MANUAL_AREA_FIXES.items():
        if idx in df.index:
            df.loc[idx, "area"] = value

    before = len(df)
    df = df[df["bedRoom"] <= BEDROOM_MAX]
    logger.info("bedRoom <= %d filter: %d -> %d row(s)", BEDROOM_MAX, before, len(df))

    for idx, value in _MANUAL_CARPET_FIXES.items():
        if idx in df.index:
            df.loc[idx, "carpet_area"] = value
    return df


def _apply_reconstructed_edits(df: pd.DataFrame) -> pd.DataFrame:
    """The two edits with no notebook cell (see the module docstring): drop the
    33 hand-picked garbage rows, and correct ``bedRoom`` on 70 ``house`` rows."""
    df = df.copy()

    present = [i for i in _RECONSTRUCTED_ROW_DROP if i in df.index]
    df = df.drop(index=present)
    logger.info("reconstructed garbage-row drop: %d row(s)", len(present))

    fixed = 0
    for idx, value in _RECONSTRUCTED_HOUSE_BEDROOM.items():
        if idx in df.index:
            df.loc[idx, "bedRoom"] = value
            fixed += 1
    logger.info("reconstructed house bedRoom corrections: %d row(s)", fixed)
    return df


def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the outlier-treatment stage to a cleaned-v2 properties frame.

    Steps, in notebook order: dedupe -> rescale price_per_sqft-IQR outlier
    areas -> threshold filters interleaved with the manual area/carpet edits ->
    recompute ``price_per_sqft`` for every row -> the two reconstructed edits ->
    add ``area_room_ratio``.

    Parameters
    ----------
    df
        Freshly read ``gurgaon_properties_cleaned_v2.csv`` (23 columns, a plain
        ``RangeIndex`` -- the ``_MANUAL_*`` / ``_RECONSTRUCTED_*`` tables key
        off these positions).

    Returns
    -------
    DataFrame with :data:`OUTPUT_COLUMNS` (24), index reset.
    """
    n_in = len(df)
    df = df.drop_duplicates()
    logger.info("drop_duplicates: %d -> %d row(s)", n_in, len(df))

    df = _fix_price_per_sqft_outliers(df)
    df = _apply_threshold_filters(df)

    df["price_per_sqft"] = _price_per_sqft(df["price"], df["area"])

    df = _apply_reconstructed_edits(df)

    df["area_room_ratio"] = df["area"] / df["bedRoom"]

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"expected columns absent: {missing}")
    return df[OUTPUT_COLUMNS].reset_index(drop=True)
