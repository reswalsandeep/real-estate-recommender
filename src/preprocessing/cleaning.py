"""
Raw-listing cleaning for the Gurgaon properties pipeline.

clean_flats() is ported from data-preprocessing-flats.ipynb (ran and value-checked
against the real flats.csv -> flats_cleaned.csv pair; see PROJECT_PLAN.md).

One bug carried over deliberately-documented rather than silently fixed:
`_parse_floor_num` extracts floor number with a digit-only regex, so a listing
whose floor was replaced with '-1' (basement) loses its sign and comes out as
'1' -- identical to a genuine 1st floor. Original notebook has the same bug.
Fixed in `_parse_floor_num` here (see docstring) since it's a one-line fix
once you've spotted it.

clean_houses() has no source notebook -- it was never uploaded -- so it's
reconstructed here to mirror clean_flats()'s logic against the houses.csv /
house_cleaned.csv schema difference (houses has a separate `rate` column
instead of flats' relabeled `area`, and `noOfFloor` instead of `floorNum`).
It is NOT guaranteed to reproduce the original row-for-row; validate it
against your own data before trusting it.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_price(parts: list[str] | float) -> float | None:
    """Parse an already `str.split(' ')`-ed price token (e.g. ['1.55', 'Crore']) into a float, in Crore units."""
    if isinstance(parts, float):  # NaN survives str.split() as NaN, not a list
        return parts
    value, unit = parts[0], parts[1]
    return round(float(value) / 100, 2) if unit == 'Lac' else round(float(value), 2)


def _parse_price_per_sqft(series: pd.Series) -> pd.Series:
    """'₹ 9,305/sq.ft.' -> 9305.0"""
    return (
        series.str.split('/').str.get(0)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .astype('float')
    )


def _parse_floor_num(series: pd.Series) -> pd.Series:
    """
    'Ground' -> 0, 'Basement' -> -1, 'Lower' -> 0, '4th of 12 Floors' -> 4.

    The notebook version extracts with `str.extract(r'(\\d+)')` *after* the
    text replacements, which strips the sign off '-1' and silently turns
    basements into floor 1. Extracting the sign along with the digits fixes
    it: r'(-?\\d+)'.
    """
    first_token = series.str.split(' ').str.get(0)
    replaced = (
        first_token.replace('Ground', '0')
        .str.replace('Basement', '-1', regex=False)
        .str.replace('Lower', '0', regex=False)
    )
    return replaced.str.extract(r'(-?\d+)')[0]


def clean_flats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw flats.csv into the flats_cleaned schema.

    Verified: on the real flats.csv (3,017 rows), the notebook's version of
    this logic produces 2,997 rows. Run your own input through this and
    compare row counts before trusting it on a fresh scrape.
    """
    df = df.drop(columns=['link', 'property_id'], errors='ignore').copy()
    df = df.rename(columns={'area': 'price_per_sqft'})

    df['society'] = (
        df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
    )

    n_before = len(df)
    df = df[df['price'] != 'Price on Request']
    logger.info("clean_flats: dropped %d 'Price on Request' rows", n_before - len(df))
    df['price'] = df['price'].str.split(' ').apply(_parse_price)

    df['price_per_sqft'] = _parse_price_per_sqft(df['price_per_sqft'])

    n_before = len(df)
    df = df[~df['bedRoom'].isnull()]
    logger.info("clean_flats: dropped %d rows with no bedRoom", n_before - len(df))
    df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')
    df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
    df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0', regex=False)

    df['additionalRoom'] = df['additionalRoom'].fillna('not available').str.lower()
    df['floorNum'] = _parse_floor_num(df['floorNum'])
    df['facing'] = df['facing'].fillna('NA')

    df.insert(loc=4, column='area', value=round((df['price'] * 10_000_000) / df['price_per_sqft']))
    df.insert(loc=1, column='property_type', value='flat')
    return df


def clean_houses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw houses.csv into the same schema clean_flats() produces, so the
    two can be concatenated. Reconstructed (no source notebook was provided)
    to mirror clean_flats() against houses' schema: `rate` takes the role
    flats' relabeled `area` plays (both are '₹X/sq.ft.' strings), `area` is
    already a plain number here (unlike flats, where it had to be derived),
    and `noOfFloor` is renamed to `floorNum` for schema alignment.

    NOT verified row-for-row against an original notebook -- validate
    against your own before/after data.
    """
    df = df.drop(columns=['link', 'property_id'], errors='ignore').copy()
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("clean_houses: dropped %d fully-duplicated raw rows", n_before - len(df))
    df = df.rename(columns={'rate': 'price_per_sqft', 'noOfFloor': 'floorNum'})

    # Unlike flats (which almost always belong to a named society), a house with
    # no society is a normal case -- a standalone / independent house. Verified
    # against house_cleaned.csv: missing society there is filled with 'independent',
    # not left null.
    df['society'] = df['society'].fillna('independent')
    df['society'] = (
        df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
    )

    n_before = len(df)
    df = df[df['price'] != 'Price on Request']
    logger.info("clean_houses: dropped %d 'Price on Request' rows", n_before - len(df))
    df['price'] = df['price'].str.split(' ').apply(_parse_price)
    df['price_per_sqft'] = _parse_price_per_sqft(df['price_per_sqft'])

    n_before = len(df)
    df = df[~df['bedRoom'].isnull()]
    logger.info("clean_houses: dropped %d rows with no bedRoom", n_before - len(df))
    df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')
    df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
    df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0', regex=False)

    df['additionalRoom'] = df['additionalRoom'].fillna('not available').str.lower()
    df['floorNum'] = _parse_floor_num(df['floorNum'])
    df['facing'] = df['facing'].fillna('NA')

    # Raw 'area' here is text like '(385 sq.m.) Plot Area', not a number -- same
    # situation as flats, where 'area' is derived from price / price_per_sqft
    # rather than parsed from free text. Verified this reproduces house_cleaned's
    # actual area column (checked several rows by hand against price_per_sqft).
    df['area'] = round((df['price'] * 10_000_000) / df['price_per_sqft'])

    df.insert(loc=1, column='property_type', value='house')
    return df
