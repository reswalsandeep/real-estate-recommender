"""Clean the raw flats.csv / houses.csv scrapes into a common schema.

_parse_floor_num keeps the sign so a basement stays -1 (a plain \\d+ regex
reads it as 1). clean_houses has no source notebook and isn't verified
row-for-row against the original.
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
    """'Ground' -> 0, 'Basement' -> -1, 'Lower' -> 0, '4th of 12 Floors' -> 4.

    r'(-?\\d+)' keeps the sign; a plain \\d+ turns a basement into 1.
    """
    first_token = series.str.split(' ').str.get(0)
    replaced = (
        first_token.replace('Ground', '0')
        .str.replace('Basement', '-1', regex=False)
        .str.replace('Lower', '0', regex=False)
    )
    return replaced.str.extract(r'(-?\d+)')[0]


def clean_flats(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw flats.csv into the flats_cleaned schema (3017 rows -> 2997)."""
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
    """Clean raw houses.csv into the same schema as clean_flats, so the two can
    be concatenated. Reconstructed - no source notebook - and not verified
    row-for-row. houses uses `rate` for the price-per-sqft string and
    `noOfFloor` for the floor column.
    """
    df = df.drop(columns=['link', 'property_id'], errors='ignore').copy()
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("clean_houses: dropped %d fully-duplicated raw rows", n_before - len(df))
    df = df.rename(columns={'rate': 'price_per_sqft', 'noOfFloor': 'floorNum'})

    # a house with no society is a normal case (a standalone house); fill 'independent'
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

    # raw 'area' here is free text, so derive it from price / price_per_sqft like flats
    df['area'] = round((df['price'] * 10_000_000) / df['price_per_sqft'])

    df.insert(loc=1, column='property_type', value='house')
    return df
