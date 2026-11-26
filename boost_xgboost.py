"""XGBoost: regularised gradient boosting with histogram-based split finding."""
from xgboost import XGBClassifier

from common import RANDOM_STATE, pos_weight, run_experiment


def make_model(y_train):
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight(y_train),   # handles class imbalance
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    run_experiment("xgboost", make_model)
