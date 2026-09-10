"""
Feature engineering for the Gurgaon properties pipeline.

Ported from feature-engineering.ipynb, ran against gurgaon_properties_cleaned_v1.csv
in the original project. Two changes from the notebook version, both noted inline:

1. `_drop_temp_furnishing_columns` drops by column name instead of the notebook's
   `df.iloc[:, :-18]`. The notebook's version assumes exactly 18 unique furnishing
   items are found in the data being processed; on a fresh scrape with even one
   different amenity name, that positional slice silently drops the wrong columns.
2. `_label_furnishing_clusters` orders the KMeans cluster labels by mean furnishing
   count instead of trusting them to come out as 0=unfurnished/1=semi/2=furnished.
   KMeans doesn't guarantee cluster index order; the notebook's mapping was correct
   for that one run but isn't guaranteed to hold if the input data, sklearn version,
   or furnishing vocabulary changes.
"""

from __future__ import annotations

import ast
import logging
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

logger = logging.getLogger(__name__)

# Perceived luxury contribution per amenity, as used in the original pipeline.
# Verbatim from feature-engineering.ipynb — this is curated domain judgment,
# not something to regenerate from scratch.
LUXURY_WEIGHTS: dict[str, int] = {
    '24/7 Power Backup': 8, '24/7 Water Supply': 4, '24x7 Security': 7, 'ATM': 4,
    'Aerobics Centre': 6, 'Airy Rooms': 8, 'Amphitheatre': 7, 'Badminton Court': 7,
    'Banquet Hall': 8, 'Bar/Chill-Out Lounge': 9, 'Barbecue': 7, 'Basketball Court': 7,
    'Billiards': 7, 'Bowling Alley': 8, 'Business Lounge': 9, 'CCTV Camera Security': 8,
    'Cafeteria': 6, 'Car Parking': 6, 'Card Room': 6, 'Centrally Air Conditioned': 9,
    'Changing Area': 6, "Children's Play Area": 7, 'Cigar Lounge': 9, 'Clinic': 5,
    'Club House': 9, 'Concierge Service': 9, 'Conference room': 8, 'Creche/Day care': 7,
    'Cricket Pitch': 7, 'Doctor on Call': 6, 'Earthquake Resistant': 5, 'Entrance Lobby': 7,
    'False Ceiling Lighting': 6, 'Feng Shui / Vaastu Compliant': 5, 'Fire Fighting Systems': 8,
    'Fitness Centre / GYM': 8, 'Flower Garden': 7, 'Food Court': 6, 'Foosball': 5,
    'Football': 7, 'Fountain': 7, 'Gated Community': 7, 'Golf Course': 10, 'Grocery Shop': 6,
    'Gymnasium': 8, 'High Ceiling Height': 8, 'High Speed Elevators': 8, 'Infinity Pool': 9,
    'Intercom Facility': 7, 'Internal Street Lights': 6, 'Internet/wi-fi connectivity': 7,
    'Jacuzzi': 9, 'Jogging Track': 7, 'Landscape Garden': 8, 'Laundry': 6,
    'Lawn Tennis Court': 8, 'Library': 8, 'Lounge': 8, 'Low Density Society': 7,
    'Maintenance Staff': 6, 'Manicured Garden': 7, 'Medical Centre': 5, 'Milk Booth': 4,
    'Mini Theatre': 9, 'Multipurpose Court': 7, 'Multipurpose Hall': 7, 'Natural Light': 8,
    'Natural Pond': 7, 'Park': 8, 'Party Lawn': 8, 'Piped Gas': 7, 'Pool Table': 7,
    'Power Back up Lift': 8, 'Private Garden / Terrace': 9, 'Property Staff': 7,
    'RO System': 7, 'Rain Water Harvesting': 7, 'Reading Lounge': 8, 'Restaurant': 8,
    'Salon': 8, 'Sauna': 9, 'Security / Fire Alarm': 9, 'Security Personnel': 9,
    'Separate entry for servant room': 8, 'Sewage Treatment Plant': 6, 'Shopping Centre': 7,
    'Skating Rink': 7, 'Solar Lighting': 6, 'Solar Water Heating': 7, 'Spa': 9,
    'Spacious Interiors': 9, 'Squash Court': 8, 'Steam Room': 9, 'Sun Deck': 8,
    'Swimming Pool': 8, 'Temple': 5, 'Theatre': 9, 'Toddler Pool': 7, 'Valet Parking': 9,
    'Video Door Security': 9, 'Visitor Parking': 7, 'Water Softener Plant': 7,
    'Water Storage': 7, 'Water purifier': 7, 'Yoga/Meditation Area': 7,
}

ADDITIONAL_ROOM_TYPES = ['study room', 'servant room', 'store room', 'pooja room', 'others']


def get_super_built_up_area(text: str) -> float | None:
    """Extract the super built-up area (sqft) from an `areaWithType` string."""
    match = re.search(r'Super Built up area (\d+\.?\d*)', text)
    return float(match.group(1)) if match else None


def get_area(text: str, area_type: str) -> float | None:
    """Extract a named area figure (e.g. 'Built Up area', 'Carpet area') from `areaWithType`."""
    match = re.search(area_type + r'\s*:\s*(\d+\.?\d*)', text)
    return float(match.group(1)) if match else None


def extract_plot_area(area_with_type: str) -> float | None:
    """Extract plot area (sqft) from `areaWithType`, for land/plot-style listings."""
    match = re.search(r'Plot area (\d+\.?\d*)', area_with_type)
    return float(match.group(1)) if match else None


def convert_to_sqft(text: str, area_value: float | None) -> float | None:
    """Convert an extracted area value to sqft if the source text tagged it as sq.m."""
    if area_value is None:
        return None
    match = re.search(r'{} \((\d+\.?\d*) sq.m.\)'.format(area_value), text)
    if match:
        return float(match.group(1)) * 10.7639
    return area_value


def _fix_unit_scale(row: pd.Series) -> float:
    """
    Correct built_up_area for rows where it was derived from plot area and the
    area/built_up_area ratio implies a unit mismatch (sq.yard or sq.m. mistaken
    for sqft): ratio ~9 -> sq.yard (x9), ratio ~11 -> sq.m. (x10.7).
    """
    if np.isnan(row['area']) or np.isnan(row['built_up_area']):
        return row['built_up_area']
    ratio = round(row['area'] / row['built_up_area'])
    if ratio == 9.0:
        return row['built_up_area'] * 9
    if ratio == 11.0:
        return row['built_up_area'] * 10.7
    return row['built_up_area']


def engineer_area_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive super_built_up_area, built_up_area, carpet_area (all sqft) from areaWithType."""
    df = df.copy()
    df['super_built_up_area'] = df['areaWithType'].apply(get_super_built_up_area)
    df['super_built_up_area'] = df.apply(
        lambda x: convert_to_sqft(x['areaWithType'], x['super_built_up_area']), axis=1
    )
    df['built_up_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Built Up area'))
    df['built_up_area'] = df.apply(
        lambda x: convert_to_sqft(x['areaWithType'], x['built_up_area']), axis=1
    )
    df['carpet_area'] = df['areaWithType'].apply(lambda x: get_area(x, 'Carpet area'))
    df['carpet_area'] = df.apply(
        lambda x: convert_to_sqft(x['areaWithType'], x['carpet_area']), axis=1
    )

    # Rows where none of the three area types were found (typically plots): fall
    # back to plot area, then fix the unit scale if it implies sq.yard/sq.m.
    all_null = df[
        df['super_built_up_area'].isnull()
        & df['built_up_area'].isnull()
        & df['carpet_area'].isnull()
    ].copy()
    n_recovered = len(all_null)
    if n_recovered:
        all_null['built_up_area'] = all_null['areaWithType'].apply(extract_plot_area)
        all_null['built_up_area'] = all_null.apply(_fix_unit_scale, axis=1)
        df.update(all_null)
        logger.info("engineer_area_columns: recovered built_up_area for %d plot-style rows", n_recovered)
    return df


def engineer_additional_room_flags(df: pd.DataFrame) -> pd.DataFrame:
    """One binary column per room type in ADDITIONAL_ROOM_TYPES, from `additionalRoom` text."""
    df = df.copy()
    for room in ADDITIONAL_ROOM_TYPES:
        df[room] = df['additionalRoom'].str.contains(room).astype(int)
    return df


def categorize_age_possession(value: str | float) -> str:
    """Bucket a raw agePossession string into one of 6 categories."""
    if pd.isna(value):
        return "Undefined"
    if "0 to 1 Year Old" in value or "Within 6 months" in value or "Within 3 months" in value:
        return "New Property"
    if "1 to 5 Year Old" in value:
        return "Relatively New"
    if "5 to 10 Year Old" in value:
        return "Moderately Old"
    if "10+ Year Old" in value:
        return "Old Property"
    if "Under Construction" in value or "By" in value:
        return "Under Construction"
    try:
        int(value.split(" ")[-1])  # entries like 'May 2024'
        return "Under Construction"
    except ValueError:
        return "Undefined"


def get_furnishing_count(details, furnishing: str) -> int:
    """Extract the count of a named furnishing item from a furnishDetails string."""
    if not isinstance(details, str):
        return 0
    if f"No {furnishing}" in details:
        return 0
    match = re.search(rf"(\d+) {re.escape(furnishing)}", details)
    if match:
        return int(match.group(1))
    if furnishing in details:
        return 1
    return 0


def _label_furnishing_clusters(scaled_matrix: np.ndarray, raw_counts: pd.DataFrame, random_state: int = 42) -> np.ndarray:
    """
    KMeans(k=3) on the furnishing-count matrix, with cluster labels reordered by
    mean total furnishing count so 0=least-furnished, 2=most-furnished always
    holds — the notebook assumed this ordering rather than enforcing it.
    """
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=random_state, n_init=10)
    raw_labels = kmeans.fit_predict(scaled_matrix)
    # mean total furnishing count per raw cluster label, used to reorder labels below
    totals = raw_counts.sum(axis=1)
    order = totals.groupby(raw_labels).mean().sort_values().index  # ascending mean count
    remap = {old_label: new_label for new_label, old_label in enumerate(order)}
    return np.array([remap[label] for label in raw_labels])


def engineer_furnishing_type(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster properties into unfurnished(0)/semifurnished(1)/furnished(2) from furnishDetails."""
    df = df.copy()
    all_furnishings: list[str] = []
    for detail in df['furnishDetails'].dropna():
        items = detail.replace('[', '').replace(']', '').replace("'", "").split(', ')
        all_furnishings.extend(items)
    unique_furnishings = list(set(all_furnishings))

    columns_to_include = list({re.sub(r'No |\d+', '', f).strip() for f in unique_furnishings} - {''})

    for furnishing in columns_to_include:
        df[furnishing] = df['furnishDetails'].apply(lambda x, f=furnishing: get_furnishing_count(x, f))

    raw_counts = df[columns_to_include].fillna(0)
    scaled = StandardScaler().fit_transform(raw_counts)
    df['furnishing_type'] = _label_furnishing_clusters(scaled, raw_counts)

    # Drop the temporary per-item columns by NAME (see module docstring, point 1).
    df = df.drop(columns=columns_to_include)
    return df


def engineer_luxury_score(df: pd.DataFrame, appartments_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build a luxury_score from the `features` amenity list, using LUXURY_WEIGHTS.
    If appartments_df (project-level data with PropertyName/TopFacilities columns)
    is provided, it's used to backfill `features` for rows where it's missing,
    matched on society name — the `appartments.csv` amenity join
    (see `notebooks/04_feature_engineering.ipynb`).
    """
    df = df.copy()

    if appartments_df is not None:
        app_df = appartments_df.copy()
        app_df['PropertyName'] = app_df['PropertyName'].str.lower()
        missing = df[df['features'].isnull()]
        if len(missing):
            filled = missing.merge(
                app_df, left_on='society', right_on='PropertyName', how='left'
            )['TopFacilities']
            filled.index = missing.index
            df.loc[missing.index, 'features'] = filled
            logger.info(
                "engineer_luxury_score: backfilled features for %d/%d rows via appartments join",
                filled.notna().sum(), len(missing),
            )

    df['features_list'] = df['features'].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) and str(x).startswith('[') else []
    )
    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(df['features_list'])
    features_binary_df = pd.DataFrame(binary_matrix, columns=mlb.classes_, index=df.index)

    known_amenities = [a for a in LUXURY_WEIGHTS if a in features_binary_df.columns]
    df['luxury_score'] = (
        features_binary_df[known_amenities].multiply(list(LUXURY_WEIGHTS[a] for a in known_amenities)).sum(axis=1)
    )
    df = df.drop(columns=['features_list'])
    return df


def engineer_features(
    df: pd.DataFrame,
    appartments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline: areas, additional-room flags, age/possession
    buckets, furnishing type, luxury score. Input should be the output of the
    cleaning stage (gurgaon_properties_cleaned_v1 equivalent); output matches
    gurgaon_properties_cleaned_v2's shape (drops the raw text columns that get
    parsed into the new features).

    Parameters
    ----------
    df : cleaned properties dataframe, must contain areaWithType, additionalRoom,
        agePossession, furnishDetails, features, society.
    appartments_df : optional project-level table (PropertyName, TopFacilities, ...)
        used to backfill missing `features` via a society-name join.
    """
    df = engineer_area_columns(df)
    df = engineer_additional_room_flags(df)
    df['agePossession'] = df['agePossession'].apply(categorize_age_possession)
    df = engineer_furnishing_type(df)
    df = engineer_luxury_score(df, appartments_df=appartments_df)
    df = df.drop(columns=['nearbyLocations', 'furnishDetails', 'features', 'additionalRoom'])
    return df
