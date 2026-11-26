"""Decision Tree: a single tree of if/else splits. Fast, interpretable baseline.
Depth and leaf size are capped to stop it memorising the training data."""
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE, run_experiment


def make_model(y_train):
    return DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    run_experiment("decision_tree", make_model)
