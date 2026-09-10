"""Project-level "similar developments" recommender for Gurgaon apartments.

Operates on ``data/raw/appartments.csv`` (246 projects after the embedded
header row is dropped) and answers *"which developments are most like this
one"* — item-to-item similarity, not preference/budget matching over
individual listings. See ``PROJECT_PLAN.md`` (the *Recommenders* section) for
that scope split.

Import from the submodules directly, e.g.::

    from src.recommender.recommender import SimilarProjectsRecommender
    from src.recommender.similarity import DEFAULT_WEIGHTS, blend
"""
