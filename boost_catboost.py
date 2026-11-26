"""CatBoost: ordered boosting with symmetric (oblivious) trees."""
from catboost import CatBoostClassifier

from common import RANDOM_STATE, run_experiment


def make_model(y_train):
    return CatBoostClassifier(
        iterations=400,
        learning_rate=0.1,
        depth=6,
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=0,
    )


if __name__ == "__main__":
    run_experiment("catboost", make_model)
