"""
lightgbm_model.py - P4: LightGBM on the frozen feature table.

LightGBM grows trees leaf-wise (num_leaves) instead of level-wise (depth),
and reads pandas 'category' columns as native categoricals.
Run: python p4_boosting/lightgbm_model.py
"""
from itertools import product

import joblib
import lightgbm as lgb

from common import run_benchmark


def fit_predict(params, X_tr, y_tr, X_ev, seed):
    model = lgb.LGBMClassifier(
        learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=100, reg_lambda=5.0, cat_smooth=20,
        n_jobs=-1, random_state=seed, verbose=-1, **params)
    model.fit(X_tr, y_tr)
    return model, model.predict_proba(X_ev)[:, 1]


GRID = [dict(num_leaves=l, n_estimators=n) for l, n in product([31, 63, 127], [300, 600])]

if __name__ == "__main__":
    run_benchmark("LightGBM", fit_predict, GRID,
                  save_model=lambda m, stem: joblib.dump(m, f"{stem}.joblib"))
