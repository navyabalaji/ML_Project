"""LightGBM: leaf-wise gradient boosting, very fast on large tabular data."""
from lightgbm import LGBMClassifier

from common import RANDOM_STATE, run_experiment


def make_model(y_train):
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.1,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


if __name__ == "__main__":
    run_experiment("lightgbm", make_model)
