"""
common.py - shared data loading, time splits and evaluation for P4's models.

NOTE: P1 owns the official harness. This file follows the same protocol
(dev Jan 1-23, gap Jan 24, test Jan 25-31, forward-chaining folds) so the
boosting models can run now; swap in P1's functions if they differ.

Protocol per model:
  1. small grid search -> mean PR-AUC over the 3 time folds (seed 0)
  2. best params       -> 3 folds x 3 seeds  -> CV mean +- std
  3. refit on ALL dev  -> 3 seeds -> test ONCE -> test mean +- std
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (FEATURES_CSV, DEV_END, TEST_START, FOLDS, SEEDS, RESULT_DIR,
                    PRED_DIR, MODEL_DIR, REPO_DIR, CHECKPOINT_MIN)

META = json.loads((REPO_DIR / "p2_eda_features" / "feature_groups.json").read_text())
FEATURES, CATEGORICAL, TARGET = META["features"], META["categorical"], META["target"]
GROUPS = META["groups"]


def load_data():
    """Return dev and test DataFrames. Category levels come from DEV only
    (a carrier/airport first seen in test becomes NaN = 'unknown')."""
    df = pd.read_csv(FEATURES_CSV, parse_dates=["FL_DATE"], low_memory=False)
    dev = df[df["FL_DATE"] <= DEV_END].reset_index(drop=True)
    test = df[df["FL_DATE"] >= TEST_START].reset_index(drop=True)
    for c in CATEGORICAL:
        levels = sorted(dev[c].astype(str).unique())
        dev[c] = pd.Categorical(dev[c].astype(str), categories=levels)
        test[c] = pd.Categorical(test[c].astype(str), categories=levels)
    return dev, test


def time_folds(dev):
    """Yield (fold_no, train_idx, val_idx) - forward chaining with a 1-day gap."""
    for i, (tr_end, va_start, va_end) in enumerate(FOLDS):
        tr = np.where(dev["FL_DATE"] <= tr_end)[0]
        va = np.where((dev["FL_DATE"] >= va_start) & (dev["FL_DATE"] <= va_end))[0]
        yield i, tr, va


def scores(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}


def baseline(y):
    """Majority-class predictor: always 'not late' -> ROC-AUC 0.5, PR-AUC = prevalence."""
    p = np.zeros(len(y))
    return {"roc_auc": 0.5, "pr_auc": float(np.mean(y)), "brier": brier_score_loss(y, p)}


def summarise(rows):
    d = pd.DataFrame(rows)[["roc_auc", "pr_auc", "brier", "log_loss"]]
    return {f"{k}_{s}": float(getattr(d[k], s)()) for k in d.columns for s in ("mean", "std")}


def to_cat(X):
    """CatBoost wants categoricals as strings (NaN -> 'nan' = unknown level)."""
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    return X


# ---------------------------------------------------------------- helpers for Analysis F
MODEL_NAMES = ["XGBoost", "LightGBM", "CatBoost"]


def load_final_models():
    """The seed-0 models refit on all dev data (saved by run_benchmark)."""
    import joblib
    return {n: joblib.load(MODEL_DIR / f"{n.lower()}.joblib")
            for n in MODEL_NAMES if (MODEL_DIR / f"{n.lower()}.joblib").exists()}


def predict_proba(name, model, X):
    return model.predict_proba(to_cat(X) if name == "CatBoost" else X)[:, 1]


def native_shap(name, model, X):
    """Exact TreeSHAP values (log-odds units) from each library's own
    implementation. Returns (values [n, n_features], base value [n])."""
    if name == "XGBoost":
        import xgboost as xgb
        v = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    elif name == "LightGBM":
        v = model.predict(X, pred_contrib=True)
    else:
        from catboost import Pool
        v = model.get_feature_importance(Pool(to_cat(X), cat_features=CATEGORICAL), type="ShapValues")
    v = np.asarray(v)
    return v[:, :-1], v[:, -1]


def best_model_name():
    """Model with the highest mean CV PR-AUC (chosen on dev, not test)."""
    res = {n: json.loads((RESULT_DIR / f"{n.lower()}.json").read_text())["cv"]["pr_auc_mean"]
           for n in MODEL_NAMES if (RESULT_DIR / f"{n.lower()}.json").exists()}
    return max(res, key=res.get)


# feature -> "known in advance" or "observed by T"
OBSERVED_GROUPS = {"aircraft_observed", "congestion_observed", "historical_rates"}
FEATURE_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}
FEATURE_KIND = {f: ("observed by T" if g in OBSERVED_GROUPS else "known in advance")
                for f, g in FEATURE_GROUP.items()}


def run_benchmark(name, fit_predict, grid, save_model=None):
    """
    fit_predict(params, X_tr, y_tr, X_ev, seed) -> (model, proba on X_ev)
    grid: list of param dicts to try.
    save_model(model, path_stem): optional, saves the seed-0 final model.
    """
    dev, test = load_data()
    Xd, yd = dev[FEATURES], dev[TARGET].to_numpy()
    Xt, yt = test[FEATURES], test[TARGET].to_numpy()
    print(f"[{name}] dev {len(dev):,} rows, test {len(test):,} rows from {TEST_START}")

    # ---- 1. tuning (seed 0, 3 folds) -----------------------------------
    tuning = []
    for params in grid:
        t0 = time.time(); rows = []
        for f, tr, va in time_folds(dev):
            _, p = fit_predict(params, Xd.iloc[tr], yd[tr], Xd.iloc[va], seed=0)
            rows.append(scores(yd[va], p))
        s = summarise(rows)
        tuning.append({"params": params, **s, "sec": round(time.time() - t0, 1)})
        print(f"  {params}  PR-AUC {s['pr_auc_mean']:.4f}  ROC-AUC {s['roc_auc_mean']:.4f}")
    best = max(tuning, key=lambda r: r["pr_auc_mean"])["params"]
    print(f"  -> best {best}")

    # ---- 2. CV with best params: folds x seeds ---------------------------
    cv_rows = []
    for seed in SEEDS:
        for f, tr, va in time_folds(dev):
            _, p = fit_predict(best, Xd.iloc[tr], yd[tr], Xd.iloc[va], seed=seed)
            cv_rows.append({"fold": f, "seed": seed, **scores(yd[va], p)})

    # ---- 3. refit on all dev, test once ---------------------------------
    test_rows, probs, fit_secs = [], [], []
    for seed in SEEDS:
        t0 = time.time()
        model, p = fit_predict(best, Xd, yd, Xt, seed=seed)
        fit_secs.append(time.time() - t0)
        probs.append(p); test_rows.append({"seed": seed, **scores(yt, p)})
        if seed == SEEDS[0] and save_model:
            save_model(model, MODEL_DIR / name.lower())
    p_mean = np.mean(probs, axis=0)

    pred = test[["FL_DATE", "TAIL_NUM", "OP_CARRIER_FL_NUM", "ORIGIN_CITY_NAME", "DEST_CITY_NAME",
                 "CRS_DEP_TIME", "DEP_DELAY", TARGET]].copy()
    pred["proba"] = p_mean
    pred.to_csv(PRED_DIR / f"{name.lower()}_test.csv", index=False)

    result = {
        "model": name, "checkpoint_min": CHECKPOINT_MIN, "test_start": TEST_START,
        "n_test_flights": int(len(test)), "test_late_rate": float(yt.mean()),
        "best_params": best, "tuning": tuning,
        "cv": summarise(cv_rows), "cv_rows": cv_rows,
        "test": summarise(test_rows), "test_rows": test_rows,
        "test_seed_avg_ensemble": scores(yt, p_mean),
        "baseline_test": baseline(yt), "fit_seconds_full_dev": float(np.mean(fit_secs)),
    }
    (RESULT_DIR / f"{name.lower()}.json").write_text(json.dumps(result, indent=2, default=float))
    c, t = result["cv"], result["test"]
    print(f"[{name}] CV   ROC-AUC {c['roc_auc_mean']:.4f} ± {c['roc_auc_std']:.4f} | "
          f"PR-AUC {c['pr_auc_mean']:.4f} ± {c['pr_auc_std']:.4f}")
    print(f"[{name}] TEST ROC-AUC {t['roc_auc_mean']:.4f} ± {t['roc_auc_std']:.4f} | "
          f"PR-AUC {t['pr_auc_mean']:.4f} ± {t['pr_auc_std']:.4f}  "
          f"(baseline PR-AUC {result['baseline_test']['pr_auc']:.4f})")
    return result
