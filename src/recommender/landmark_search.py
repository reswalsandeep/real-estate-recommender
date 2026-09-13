"""Landmark-proximity search: pick a landmark, get every project within X km.

A stateless lookup over a precomputed projects x landmarks distance matrix
(metres), from the same LocationAdvantages data as similarity.py's location axis.

Limitations - surface these in the app UI:

* Results are "projects whose listing mentions this landmark within the radius",
  not a true geospatial nearest - there are no project coordinates, so a project
  that didn't list a landmark is absent even if it is physically close.
* Landmark names are fragmented ("IGI Airport" / "Indira Gandhi Intl Airport" /
  "Indira Gandhi International Airport"). _LANDMARK_ALIASES folds the common ones.
* ~184 distance strings are time ("10 mins") or vague ("Close Proximity") and
  carry no distance.
"""

from __future__ import annotations

import ast
import difflib
from pathlib import Path

import pandas as pd

from .similarity import (
    DEFAULT_APPARTMENTS_CSV,
    REPO_ROOT,
    distance_to_metres,
    load_appartments,
)

DEFAULT_DISTANCE_MATRIX_CSV = REPO_ROOT / "data" / "processed" / "landmark_distance_matrix.csv"

# Small, explicit fold for the highest-traffic name variants (keyed on the
# whitespace-normalised, lower-cased raw name). Not a general resolver.
_LANDMARK_ALIASES = {
    "indira gandhi intl airport": "Indira Gandhi International Airport",
    "igi airport": "Indira Gandhi International Airport",
    "dwarka expy": "Dwarka Expressway",
}

_RESULT_COLUMNS = ["project", "distance_m", "distance_km"]


def _canon(name: str) -> str:
    """Whitespace-normalise a landmark name, then apply the alias fold."""
    cleaned = " ".join(str(name).split()).strip()
    return _LANDMARK_ALIASES.get(cleaned.lower(), cleaned)


# --------------------------------------------------------------------------- #
def build_distance_matrix(
    df: pd.DataFrame | None = None,
    save: bool | str | Path = True,
) -> pd.DataFrame:
    """Projects (rows) x landmarks (columns), values = distance in **metres**.

    ``NaN`` where a project does not list a landmark (a real absence -- callers
    must exclude it, not treat it as "far"). When name variants fold together
    (via :func:`_canon`) the **minimum** distance for that project is kept.
    If ``save`` is truthy the matrix is written to
    ``data/processed/landmark_distance_matrix.csv`` (like ``sector_price_map.csv``).
    """
    df = load_appartments() if df is None else df

    records: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        project = row["PropertyName"]
        cells = records.setdefault(project, {})
        try:
            items = ast.literal_eval(row["LocationAdvantages"]).items()
        except (ValueError, SyntaxError):
            continue
        for raw_name, raw_distance in items:
            metres = distance_to_metres(raw_distance)
            if metres is None:
                continue
            landmark = _canon(raw_name)
            if landmark not in cells or metres < cells[landmark]:
                cells[landmark] = float(metres)

    # from_dict drops keys whose value is an empty dict, so projects with no
    # parseable landmark distance would vanish silently -- reindex them back as
    # all-NaN rows so the matrix faithfully covers every project.
    matrix = pd.DataFrame.from_dict(records, orient="index")
    matrix = matrix.reindex(sorted(records)).reindex(sorted(matrix.columns), axis=1)
    matrix.index.name = "project"

    if save:
        path = DEFAULT_DISTANCE_MATRIX_CSV if save is True else Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(path)
    return matrix


def load_distance_matrix(path: str | Path = DEFAULT_DISTANCE_MATRIX_CSV) -> pd.DataFrame:
    """Read the cached matrix; build (and cache) it if the file is absent."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, index_col="project")
    return build_distance_matrix(save=path)


def _resolve_landmark(query: str, columns) -> str:
    """Map a user-supplied landmark name to a real matrix column, or raise."""
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for candidate in (query, _canon(query)):
        if candidate in cols:
            return candidate
        if candidate.strip().lower() in lower:
            return lower[candidate.strip().lower()]
    hints = difflib.get_close_matches(query, cols, n=5, cutoff=0.4)
    suffix = f" Did you mean: {hints}?" if hints else ""
    raise KeyError(f"landmark not found: {query!r}.{suffix}")


def search_by_landmark(
    landmark: str,
    radius_km: float,
    matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Projects within ``radius_km`` of ``landmark``, nearest first.

    ``radius_km`` is kilometres (converted to metres internally -- the caller
    never handles metres). ``radius_km <= 0`` raises ``ValueError``. No matches
    returns an **empty** DataFrame with the normal columns
    (``project``, ``distance_m``, ``distance_km``) -- check ``.empty``.
    """
    if radius_km <= 0:
        raise ValueError(f"radius_km must be > 0, got {radius_km!r}")

    matrix = load_distance_matrix() if matrix is None else matrix
    column = _resolve_landmark(landmark, matrix.columns)
    radius_m = radius_km * 1000.0

    hits = matrix[column].dropna()
    hits = hits[hits <= radius_m].sort_values()
    return pd.DataFrame(
        {
            "project": list(hits.index),
            "distance_m": [round(v) for v in hits.values],
            "distance_km": [round(v / 1000, 2) for v in hits.values],
        },
        columns=_RESULT_COLUMNS,
    )


def list_landmarks(
    matrix: pd.DataFrame | None = None,
    min_projects: int = 1,
) -> list[str]:
    """Landmark names, sorted, optionally restricted to those listed by at least
    ``min_projects`` projects (drops the ~700-name long tail for an app dropdown)."""
    matrix = load_distance_matrix() if matrix is None else matrix
    counts = matrix.notna().sum()
    return sorted(counts[counts >= min_projects].index)


if __name__ == "__main__":
    m = build_distance_matrix()
    no_data = int((~m.notna().any(axis=1)).sum())
    print(f"{m.shape[0]} projects x {m.shape[1]} landmarks; "
          f"{int(m.notna().sum().sum())} known distances; "
          f"{no_data} projects with no parseable landmark distance (all-NaN rows)")
    print(f"landmarks listed by >= 5 projects: {len(list_landmarks(m, min_projects=5))}")
