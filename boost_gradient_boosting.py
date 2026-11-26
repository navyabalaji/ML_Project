"""Gradient Boosting (scikit-learn): each tree fits the errors (gradients) of the
ensemble so far. Slow on large data, so it uses row subsampling (stochastic GB)."""
from sklearn.ensemble import GradientBoostingClassifier

from common import RANDOM_STATE, run_experiment


def make_model(y_train):
    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.5,          # trains each tree on 50% of rows -> ~2x faster
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    # GradientBoostingClassifier has no class_weight option -> use sample weights
    run_experiment("gradient_boosting", make_model, use_sample_weight=True)
