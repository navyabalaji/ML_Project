"""AdaBoost: sequentially re-weights misclassified flights so each new weak
learner (a shallow tree) focuses on the hard cases."""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE, run_experiment


def make_model(y_train):
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                         random_state=RANDOM_STATE),
        n_estimators=200,
        learning_rate=0.5,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    run_experiment("adaboost", make_model)
