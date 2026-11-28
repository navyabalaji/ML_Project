"""
analysis_f_importance.py - P4 Analysis F (part 2): feature importance three ways.

  01 built-in importance (gain / prediction-value-change) per model
  02 permutation importance: drop in PR-AUC when one feature is shuffled
  03 GROUP permutation: shuffle a whole feature group at once
     (fairer than single features when features are correlated)
  04 PDP + ICE: how P(LATE) changes as one input changes, all else fixed
  05 do the importance methods agree? (Spearman rank correlation)

Uses the final models on a 20k test sample (post-hoc, nothing is tuned here).
Run: python p4_boosting/analysis_f_importance.py   (after the 3 model scripts)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (load_data, load_final_models, predict_proba, best_model_name,
                    FEATURES, CATEGORICAL, TARGET, GROUPS, FEATURE_KIND)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (FIG_DIR, TABLE_DIR, MODEL_COLORS, C_BLUE, C_ORANGE, C_MUTED, style_plots)

plt = style_plots()
KIND_COLOR = {"known in advance": C_BLUE, "observed by T": C_ORANGE}
N_SAMPLE, N_REPEATS, TOP = 20_000, 3, 15
rng = np.random.default_rng(0)

_, test = load_data()
sample = test.sample(N_SAMPLE, random_state=1).reset_index(drop=True)
X, y = sample[FEATURES], sample[TARGET].to_numpy()
models = load_final_models()
best = best_model_name()
print("models:", list(models), "| best on CV PR-AUC:", best)


def builtin_importance(name, m):
    if name == "XGBoost":
        d = m.get_booster().get_score(importance_type="total_gain")
        s = pd.Series(d).reindex(FEATURES).fillna(0)
    elif name == "LightGBM":
        s = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
    else:  # CatBoost default = PredictionValuesChange
        s = pd.Series(m.get_feature_importance(), index=FEATURES)
    return s / s.sum()


def permute(Xp, cols):
    """Shuffle the given columns together (same row permutation for all)."""
    Xp = Xp.copy(); idx = rng.permutation(len(Xp))
    for c in cols:
        Xp[c] = Xp[c].to_numpy()[idx] if c not in CATEGORICAL else \
            pd.Categorical.from_codes(Xp[c].cat.codes.to_numpy()[idx], Xp[c].cat.categories)
    return Xp


def perm_importance(name, m, col_sets):
    base_p = predict_proba(name, m, X)
    base = {"pr": average_precision_score(y, base_p), "roc": roc_auc_score(y, base_p)}
    rows = []
    for key, cols in col_sets.items():
        for r in range(N_REPEATS):
            p = predict_proba(name, m, permute(X, cols))
            rows.append({"feature": key, "repeat": r,
                         "pr_drop": base["pr"] - average_precision_score(y, p),
                         "roc_drop": base["roc"] - roc_auc_score(y, p)})
    return pd.DataFrame(rows).groupby("feature")[["pr_drop", "roc_drop"]].agg(["mean", "std"])


# ---------------------------------------------------------------- 01 built-in
builtin = pd.DataFrame({n: builtin_importance(n, m) for n, m in models.items()})
builtin.to_csv(TABLE_DIR / "importance_builtin.csv")
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5.5))
for ax, name in zip(np.atleast_1d(axes), models):
    s = builtin[name].sort_values().tail(TOP)
    ax.barh(s.index, s.values, color=[KIND_COLOR[FEATURE_KIND[f]] for f in s.index], height=0.7)
    ax.set_title(name); ax.set_xlabel("normalised built-in importance")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KIND_COLOR.values()]
fig.legend(handles, KIND_COLOR.keys(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Built-in importance (XGB/LGBM: total gain, CatBoost: prediction-value change)", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "f_imp_01_builtin.png"); plt.close(fig)

# ---------------------------------------------------------------- 02 single-feature permutation
perm = {}
for name, m in models.items():
    print(f"  permutation importance: {name}")
    perm[name] = perm_importance(name, m, {f: [f] for f in FEATURES})
perm_pr = pd.DataFrame({n: perm[n][("pr_drop", "mean")] for n in models})
perm_pr.to_csv(TABLE_DIR / "importance_permutation_pr_auc.csv")
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5.5))
for ax, name in zip(np.atleast_1d(axes), models):
    d = perm[name].sort_values(("pr_drop", "mean")).tail(TOP)
    ax.barh(d.index, d[("pr_drop", "mean")], xerr=d[("pr_drop", "std")], height=0.7,
            color=[KIND_COLOR[FEATURE_KIND[f]] for f in d.index], ecolor=C_MUTED)
    ax.set_title(name); ax.set_xlabel("drop in PR-AUC when shuffled")
fig.legend(handles, KIND_COLOR.keys(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Permutation importance (test sample, 3 repeats)", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "f_imp_02_permutation.png"); plt.close(fig)

# ---------------------------------------------------------------- 03 group permutation
gperm = {n: perm_importance(n, m, GROUPS) for n, m in models.items()}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
order = pd.DataFrame({n: gperm[n][("pr_drop", "mean")] for n in models}).mean(1).sort_values().index
h = 0.8 / len(models)
for ax, metric, lab in [(axes[0], "pr_drop", "drop in PR-AUC"), (axes[1], "roc_drop", "drop in ROC-AUC")]:
    for i, name in enumerate(models):
        d = gperm[name].loc[order]
        ax.barh(np.arange(len(order)) + i * h, d[(metric, "mean")], xerr=d[(metric, "std")],
                height=h * 0.9, color=MODEL_COLORS[name], ecolor=C_MUTED, label=name)
    ax.set_xlabel(lab)
axes[0].set_yticks(np.arange(len(order)) + h * (len(models) - 1) / 2, order)
axes[0].legend(loc="lower right")
fig.suptitle("Group permutation: shuffle a whole kind of information at once", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "f_imp_03_group_permutation.png"); plt.close(fig)
pd.concat(gperm, axis=1).round(4).to_csv(TABLE_DIR / "importance_group_permutation.csv")

# ---------------------------------------------------------------- 04 PDP + ICE
# (not dep_hour: it is copied in dep_minute_of_day/hour_sin/hour_cos, so moving it alone is unrealistic)
pdp_feats = ["prev_leg_delay_lb", "origin_recent_late_rate", "planned_turn_slack", "carrier_recent_late_rate"]
sub = X.sample(1000, random_state=2).reset_index(drop=True)
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
for ax, f in zip(axes, pdp_feats):
    grid = np.unique(np.nanpercentile(X[f], np.linspace(2, 98, 25)))
    for name, m in models.items():
        curves = np.empty((len(sub), len(grid)))
        for j, v in enumerate(grid):
            Xg = sub.copy(); Xg[f] = v
            curves[:, j] = predict_proba(name, m, Xg)
        if name == best:   # ICE lines for the best model only
            for k in range(0, len(sub), 20):
                ax.plot(grid, curves[k], color="#bbbbbb", lw=0.5, alpha=0.5)
        ax.plot(grid, curves.mean(0), color=MODEL_COLORS[name], lw=2.5, label=f"{name} PDP")
    ax.set_xlabel(f)
axes[0].set_ylabel("P(LATE)"); axes[0].legend(loc="upper left")
fig.suptitle(f"Partial dependence (coloured) + ICE lines for {best} (grey)", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "f_imp_04_pdp_ice.png"); plt.close(fig)

# ---------------------------------------------------------------- 05 method agreement
shap_file = TABLE_DIR / "shap_importance.csv"
shap_imp = pd.read_csv(shap_file, index_col=0) if shap_file.exists() else None
rows = []
for name in models:
    methods = {"built-in": builtin[name], "permutation": perm_pr[name]}
    if shap_imp is not None and name in shap_imp:
        methods["SHAP"] = shap_imp[name]
    m = pd.DataFrame(methods).corr(method="spearman")
    for a in m.index:
        for b in m.columns:
            if a < b:
                rows.append({"model": name, "pair": f"{a} vs {b}", "spearman": round(m.loc[a, b], 3)})
agree = pd.DataFrame(rows)
agree.to_csv(TABLE_DIR / "importance_method_agreement.csv", index=False)
print("\nAgreement between importance methods (Spearman over all features):")
print(agree.pivot(index="pair", columns="model", values="spearman"))
print("\nGroup permutation, drop in PR-AUC:")
print(pd.DataFrame({n: gperm[n][("pr_drop", "mean")] for n in models}).round(4))
