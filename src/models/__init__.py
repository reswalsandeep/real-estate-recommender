"""Model training, evaluation, and inference for the Gurgaon price predictor.

This layer exists because the original ``model-selection.ipynb`` exported a
pipeline that did not match the configuration it validated.

Import from the submodules directly, e.g.::

    from src.models.train import train_and_export, build_pipeline
    from src.models.evaluate import regression_metrics
    from src.models.predict import load_model, predict_one
"""
