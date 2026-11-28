"""
xgboost_model.py - P4: XGBoost on the frozen feature table.

Categoricals are handled natively (enable_categorical) so no one-hot is needed.
NaN (e.g. "no previous leg") is handled natively too: each split learns which
side missing values go to.
Run: python p4_boosting/xgboost_model.py
"""
from itertools import product

import joblib
import xgboost as xgb

from common import run_benchmark


def fit_predict(params, X_tr, y_tr, X_ev, seed):
    model = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=50, reg_lambda=5.0, eval_metric="aucpr",
        n_jobs=-1, random_state=seed, **params)
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_ev)[:, 1]


# tree depth controls interaction order; more trees at a fixed small learning rate
GRID = [dict(max_depth=d, n_estimators=n) for d, n in product([4, 6, 8], [300, 600])]

if __name__ == "__main__":
    run_benchmark("XGBoost", fit_predict, GRID,
                  save_model=lambda m, stem: joblib.dump(m, f"{stem}.joblib"))
