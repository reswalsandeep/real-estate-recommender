"""Outlier treatment: drop duplicates and extreme rows, rescale square-yard
areas, recompute price_per_sqft.

The area_room_ratio column, a drop of 33 low area/bedRoom rows, and a downward
bedRoom fix on 70 house rows have no code in the source notebook. They're applied
from the _RECONSTRUCTED_* / _MANUAL_* tables below, keyed by positional row index
into cleaned_v2.csv - regenerate from a fresh input/output diff if row order
changes. The 33 drops are a subset of the area/bedRoom < 183 rows; the 70 bedRoom
fixes aren't a formula or a join to houses.csv.
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

# int columns that pandas < 2.0 DataFrame.update upcast to float; the committed CSVs carry that
_UPDATE_UPCASTS_TO_FLOAT = [
    "bedRoom", "bathroom", "study room", "servant room", "store room",
    "pooja room", "others", "furnishing_type", "luxury_score",
]

# Crore -> rupees, for price_per_sqft = price / area.
_CRORE = 10_000_000

# hand-typed edits from notebook cells 33/35/67; keys are positional row indices
_MANUAL_AREA_DROP = [818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471]
_MANUAL_AREA_FIXES = {
    48: 115 * 9, 300: 7250, 2666: 5800, 1358: 2660,
    3195: 2850, 2131: 1812, 3088: 2160, 3444: 1175,
}
_MANUAL_CARPET_FIXES = {2131: 1812}

# recovered from the input->output diff (no notebook cell); positional row indices
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
    """For the price_per_sqft IQR outliers: sub-1000 area treated as square
    yards (*9), then price_per_sqft recomputed (notebook cells 14-18)."""
    df = df.copy()
    lo, hi = _iqr_bounds(df["price_per_sqft"])
    outliers = df[(df["price_per_sqft"] < lo) | (df["price_per_sqft"] > hi)].copy()

    outliers["area"] = outliers["area"].apply(
        lambda x: x * SQYD_TO_SQFT if x < SQYD_AREA_THRESHOLD else x
    )
    outliers["price_per_sqft"] = _price_per_sqft(outliers["price"], outliers["area"])
    df.update(outliers)  # aligns on index; only non-NaN values overwrite

    # pandas < 2.0 update upcast these to float; the committed CSVs carry that
    for col in _UPDATE_UPCASTS_TO_FLOAT:
        df[col] = df[col].astype(float)

    logger.info("price_per_sqft IQR outliers rescaled: %d row(s)", len(outliers))
    return df


def _apply_threshold_filters(df: pd.DataFrame) -> pd.DataFrame:
    """price_per_sqft <= 50000, area < 100000, bedRoom <= 10, interleaved with
    the manual index edits in the notebook's order. Null price_per_sqft / area
    rows fall out here (NaN fails every comparison), not via dropna.
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
    """Drop the 33 hand-picked rows and correct bedRoom on 70 house rows (see module docstring)."""
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
    """Apply outlier treatment to a cleaned-v2 frame.

    Steps in notebook order: dedupe -> rescale price_per_sqft-IQR outlier areas
    -> threshold filters + manual edits -> recompute price_per_sqft ->
    reconstructed edits -> add area_room_ratio. Input needs a plain RangeIndex
    (the _MANUAL_* / _RECONSTRUCTED_* tables key off positions). Returns
    OUTPUT_COLUMNS (24), index reset.
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
