"""Parsers in src/recommender/similarity.py.

`price_range_to_crore` cases are verbatim from the module docstring and the
fix-price-parsing validation ('99 L - 1.37 Cr' -> [0.99, 1.37], single-value
strings no longer dropped). `distance_to_metres` cases are from its docstring.
"""
import pytest

from src.recommender.similarity import price_range_to_crore, distance_to_metres


@pytest.mark.parametrize(
    "text, expected",
    [
        ("₹ 99 L - 1.37 Cr", [0.99, 1.37]),      # unit read per number, not per string
        ("₹ 1.35 - 6.67 Cr", [1.35, 6.67]),      # bare 1.35 inherits Cr
        ("₹ 26.62 - 44.93 L", [0.2662, 0.4493]),  # both lakh
        ("₹ 17 L", [0.17]),                       # single value (old parser dropped it)
        ("₹ 2.75 Cr", [2.75]),
        ("Price on Request", []),
        ("", []),
    ],
)
def test_price_range_to_crore(text, expected):
    got = price_range_to_crore(text)
    assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2.5 KM", 2500.0),
        ("2.5km", 2500.0),
        ("1.5 kms", 1500.0),
        ("800 Meter", 800.0),
        ("800 M", 800.0),
        ("450 m", 450.0),
    ],
)
def test_distance_to_metres_parses(text, expected):
    assert distance_to_metres(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["10 mins", "Close Proximity", "5 minutes drive"])
def test_distance_to_metres_returns_none_for_non_distances(text):
    assert distance_to_metres(text) is None
