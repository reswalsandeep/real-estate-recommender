"""Price and floor-number parsers in src/preprocessing/cleaning.py.

Cases are the ones established while validating the module (see the module
docstring and notebooks/01_data_preprocessing_flats.ipynb) — notably the basement
sign-fix: `_parse_floor_num` must keep `Basement -> -1`, not the original
notebook's `-> 1`.
"""
import math

import pandas as pd
import pytest

from src.preprocessing.cleaning import _parse_price, _parse_floor_num


@pytest.mark.parametrize(
    "parts, expected",
    [
        (["1.55", "Crore"], 1.55),
        (["80", "Lac"], 0.80),        # Lac -> Crore is a /100 divide
        (["1.556", "Crore"], 1.56),   # rounds to 2 dp
        (["3", "Crore"], 3.0),
    ],
)
def test_parse_price_values(parts, expected):
    assert _parse_price(parts) == pytest.approx(expected)


def test_parse_price_nan_passthrough():
    # a missing price survives str.split() as a float NaN, not a list
    assert math.isnan(_parse_price(float("nan")))


def test_parse_floor_num_mapping_and_basement_sign():
    s = pd.Series(
        ["Ground", "Basement", "Lower", "4 out of 12 Floors", "14th of 14 Floors"]
    )
    out = list(_parse_floor_num(s))
    assert out == ["0", "-1", "0", "4", "14"]


def test_parse_floor_num_basement_is_negative_not_first_floor():
    # the specific regression the module fixed: digit-only extraction turned
    # a basement ("-1") into "1", indistinguishable from a 1st-floor unit.
    out = list(_parse_floor_num(pd.Series(["Basement"])))
    assert out == ["-1"]
    assert out != ["1"]
