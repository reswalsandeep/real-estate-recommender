"""The agePossession group-mode fallback chain in
src/preprocessing/missing_values._impute_age_possession.

One toy frame forces a fill at each of the three cascade stages:
  (sector, property_type) mode  ->  (sector,) mode  ->  (property_type,) mode
"""
import pandas as pd
import pytest

from src.preprocessing.missing_values import _impute_age_possession, _AGE_SENTINEL

U = _AGE_SENTINEL  # "Undefined"


@pytest.fixture
def frame():
    rows = [
        # sector, property_type, agePossession
        ("S1", "flat", "New Property"),
        ("S1", "flat", "New Property"),
        ("S1", "flat", "New Property"),
        ("S1", "flat", U),               # idx 3 -> filled at STAGE 1 (sector+type mode)
        ("S2", "flat", "Old Property"),
        ("S2", "flat", "Old Property"),
        ("S2", "flat", "Old Property"),
        ("S2", "house", U),              # idx 7 -> (S2,house) is all-U, so STAGE 2 (sector mode)
        ("S3", "house", U),              # idx 8 -> S3 is all-U, so STAGE 3 (property_type mode)
        ("S4", "house", "Moderately Old"),
        ("S4", "house", "Moderately Old"),
        ("S4", "house", "Moderately Old"),
    ]
    return pd.DataFrame(rows, columns=["sector", "property_type", "agePossession"])


def test_no_undefined_remains(frame):
    out = _impute_age_possession(frame)
    assert (out["agePossession"] == U).sum() == 0


def test_stage_1_sector_and_type_mode(frame):
    out = _impute_age_possession(frame)
    assert out.loc[3, "agePossession"] == "New Property"


def test_stage_2_sector_mode(frame):
    # (S2, house) group is entirely Undefined -> falls through to the S2 mode
    out = _impute_age_possession(frame)
    assert out.loc[7, "agePossession"] == "Old Property"


def test_stage_3_property_type_mode(frame):
    # S3 has no non-Undefined row at all -> falls through to the "house" mode
    out = _impute_age_possession(frame)
    assert out.loc[8, "agePossession"] == "Moderately Old"


def test_known_values_are_untouched(frame):
    out = _impute_age_possession(frame)
    assert out.loc[0, "agePossession"] == "New Property"
    assert out.loc[9, "agePossession"] == "Moderately Old"
