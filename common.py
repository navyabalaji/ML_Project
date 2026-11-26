"""
common.py - shared helpers for every model script (boosting, tree, forest, ...).

Each model script calls run_experiment(), which:
  * trains the model on the FULL feature set and on the PCA feature set
  * measures training time, prediction time and model file size
  * computes accuracy, precision, recall, F1, ROC-AUC, PR-AUC on the test set
  * saves metrics (JSON), test-set probabilities (.npy) and the model (.joblib)
    into ./results/ so later analyses (efficiency, error, interpretability,
    ablation) can reuse them without retraining.
"""

import json
import os
import time

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

FEAT_DIR = "features_out"      # produced by features_pca.py
RES_DIR = "results"
RANDOM_STATE = 42
THRESHOLD = 0.5

# Set to e.g. 200_000 to train every model on a random subset (faster runs).
# Keep it the same for ALL models so the comparison stays fair.
MAX_TRAIN_ROWS = None

for _sub in ("models", "preds"):
    os.makedirs(f"{RES_DIR}/{_sub}", exist_ok=True)


def load_data(feature_set):
    """feature_set: 'full' (all filtered features) or 'pca' (PCA components)."""
    suffix = "" if feature_set == "full" else "_pca"
    X_train = np.load(f"{FEAT_DIR}/X_train{suffix}.npy")
    X_test = np.load(f"{FEAT_DIR}/X_test{suffix}.npy")
    y_train = np.load(f"{FEAT_DIR}/y_train.npy")
    y_test = np.load(f"{FEAT_DIR}/y_test.npy")
    if MAX_TRAIN_ROWS and len(X_train) > MAX_TRAIN_ROWS:
        idx = np.sort(np.random.RandomState(RANDOM_STATE)
                      .choice(len(X_train), MAX_TRAIN_ROWS, replace=False))
        X_train, y_train = X_train[idx], y_train[idx]
    return X_train, X_test, y_train, y_test


def pos_weight(y):
    """Ratio of on-time to delayed flights (for imbalance handling)."""
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def balanced_sample_weight(y):
    """Per-row weights that make delayed flights count as much as on-time ones."""
    w = np.ones(len(y), dtype=float)
    w[y == 1] = pos_weight(y)
    return w


def evaluate(y_true, proba):
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def run_experiment(model_name, make_model, use_sample_weight=False,
                   feature_sets=("full", "pca")):
    """
    model_name        : short name, e.g. 'xgboost'
    make_model        : function(y_train) -> unfitted estimator
    use_sample_weight : True for models without a class_weight option
    """
    for fs in feature_sets:
        X_train, X_test, y_train, y_test = load_data(fs)
        model = make_model(y_train)
        tag = f"{model_name}_{fs}"
        print(f"\n=== {tag}: {X_train.shape[0]} train rows, {X_train.shape[1]} features ===")

        t0 = time.perf_counter()
        if use_sample_weight:
            model.fit(X_train, y_train, sample_weight=balanced_sample_weight(y_train))
        else:
            model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = model.predict_proba(X_test)[:, 1]
        pred_time = time.perf_counter() - t0

        model_path = f"{RES_DIR}/models/{tag}.joblib"
        joblib.dump(model, model_path)

        metrics = evaluate(y_test, proba)
        metrics.update({
            "model": model_name,
            "feature_set": fs,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "train_time_s": round(fit_time, 2),
            "predict_time_s": round(pred_time, 3),
            "predict_us_per_row": round(pred_time / len(X_test) * 1e6, 2),
            "model_size_mb": round(os.path.getsize(model_path) / 1e6, 3),
            "threshold": THRESHOLD,
        })

        np.save(f"{RES_DIR}/preds/{tag}_proba.npy", proba.astype(np.float32))
        with open(f"{RES_DIR}/{tag}.json", "w") as f:
            json.dump({k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in metrics.items()}, f, indent=2)

        print(f"  accuracy {metrics['accuracy']:.4f} | precision {metrics['precision']:.4f} | "
              f"recall {metrics['recall']:.4f} | F1 {metrics['f1']:.4f} | "
              f"ROC-AUC {metrics['roc_auc']:.4f}")
        print(f"  train {fit_time:.1f}s | predict {pred_time:.2f}s | "
              f"size {metrics['model_size_mb']:.2f} MB")
