"""Random Forest: many decorrelated trees (bootstrap rows + random feature subsets)
whose probabilities are averaged. Depth/leaf caps keep the saved model reasonably small."""
from sklearn.ensemble import RandomForestClassifier

from common import RANDOM_STATE, run_experiment


def make_model(y_train):
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    run_experiment("random_forest", make_model)
