"""
catboost_model.py - P4: CatBoost on the frozen feature table.

CatBoost encodes categoricals with "ordered target statistics": each row's
encoding uses only rows before it in a random order, which avoids target
leakage inside the training set. It builds symmetric (oblivious) trees.
Run: python p4_boosting/catboost_model.py
"""
from itertools import product

import joblib
from catboost import CatBoostClassifier

from common import run_benchmark, CATEGORICAL, to_cat   # to_cat: categoricals -> str


def fit_predict(params, X_tr, y_tr, X_ev, seed):
    model = CatBoostClassifier(
        learning_rate=0.08, l2_leaf_reg=5, rsm=0.8, eval_metric="PRAUC",
        cat_features=CATEGORICAL, thread_count=-1, random_seed=seed,
        verbose=0, allow_writing_files=False, **params)
    model.fit(to_cat(X_tr), y_tr)
    return model, model.predict_proba(to_cat(X_ev))[:, 1]


GRID = [dict(depth=d, iterations=n) for d, n in product([6, 8], [400, 800])]

if __name__ == "__main__":
    run_benchmark("CatBoost", fit_predict, GRID,
                  save_model=lambda m, stem: joblib.dump(m, f"{stem}.joblib"))
