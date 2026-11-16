"""Classification heads (one per target) with small hyperparameter grids.

Grids are deliberately small: they're tuned on the inner train holdout, once per target.
"""
from __future__ import annotations

import warnings

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from .trainer import HeadSpec

try:
    import xgboost as xgb
except ImportError:  # pip install xgboost  (proposal baseline); sklearn HGB is the fallback
    xgb = None


def logistic(sparse: bool = False, Cs=(0.01, 0.1, 1.0, 10.0)) -> HeadSpec:
    """L2 logistic regression. Dense inputs are standardized; sparse (TF-IDF) inputs only max-abs
    scaled so they stay sparse."""
    def make(p):
        scaler = MaxAbsScaler() if sparse else StandardScaler()
        return make_pipeline(scaler, LogisticRegression(C=p["C"], max_iter=2000,
                                                        solver="liblinear" if sparse else "lbfgs"))
    return HeadSpec("logreg", make, [{"C": c} for c in Cs])


def gbm(n_jobs: int = -1) -> HeadSpec:
    """XGBoost if installed (matches the proposal), else sklearn HistGradientBoosting.
    Both handle NaN natively, so no imputation is needed for missing vitals."""
    if xgb is not None:
        def make(p):
            return xgb.XGBClassifier(n_estimators=p["n_estimators"], learning_rate=0.05,
                                     max_depth=p["max_depth"], subsample=0.8, colsample_bytree=0.8,
                                     min_child_weight=5, tree_method="hist", n_jobs=n_jobs,
                                     eval_metric="logloss")
        grid = [{"max_depth": d, "n_estimators": n} for d in (4, 6) for n in (300, 600)]
        return HeadSpec("xgboost", make, grid)

    warnings.warn("xgboost not installed; using sklearn HistGradientBoostingClassifier. "
                  "Install with: ~/miniconda3/envs/torch/bin/pip install xgboost", stacklevel=2)

    def make(p):
        # early_stopping uses a random 10% of the training rows internally to pick the iteration count
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, max_depth=p["max_depth"],
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=30, random_state=0)
    return HeadSpec("hgb", make, [{"max_depth": d} for d in (4, 8)])
