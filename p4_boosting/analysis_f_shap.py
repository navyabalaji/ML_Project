"""
analysis_f_shap.py - P4 Analysis F (part 1): SHAP explanations of the boosting models.

Uses the final models (refit on dev) on a 20k sample of the test period.
This is post-hoc explanation only - nothing here is used to tune anything.

Outputs (outputs/figures/f_shap_*.png, outputs/tables/shap_*.csv):
  01 global importance (mean |SHAP|) per model, coloured by checkpoint type
  02 beeswarm per model (direction + spread of each feature's effect)
  03 share of explanation per feature group, per model
  04 dependence plots for the 4 strongest numeric features
  05 how much the models lean on "observed by T" info, by hour of day
  06 local explanations: one confident hit, one confident miss
Run: python p4_boosting/analysis_f_shap.py   (after the 3 model scripts)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from common import (load_data, load_final_models, native_shap, predict_proba, best_model_name,
                    FEATURES, CATEGORICAL, TARGET, GROUPS, FEATURE_KIND)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIG_DIR, TABLE_DIR, MODEL_COLORS, C_BLUE, C_ORANGE, C_MUTED, style_plots

plt = style_plots()
KIND_COLOR = {"known in advance": C_BLUE, "observed by T": C_ORANGE}
N_SAMPLE, TOP = 20_000, 15

_, test = load_data()
sample = test.sample(N_SAMPLE, random_state=0).reset_index(drop=True)
X, y = sample[FEATURES], sample[TARGET].to_numpy()
X_num = X.copy()                                          # numeric view for colouring plots
for c in CATEGORICAL:
    X_num[c] = X_num[c].cat.codes.replace(-1, np.nan)

models = load_final_models()
best = best_model_name()
print("models:", list(models), "| best on CV PR-AUC:", best)

shap_vals, base_vals, imp = {}, {}, {}
for name, m in models.items():
    sv, base = native_shap(name, m, X)
    # sanity: SHAP values + base must add up to the model's log-odds output
    p = predict_proba(name, m, X)
    recon = 1 / (1 + np.exp(-(sv.sum(1) + base)))
    print(f"  {name}: max |SHAP reconstruction - proba| = {np.abs(recon - p).max():.2e}")
    shap_vals[name], base_vals[name] = sv, base
    imp[name] = pd.Series(np.abs(sv).mean(0), index=FEATURES)

imp_df = pd.DataFrame(imp)
imp_df.sort_values(best, ascending=False).to_csv(TABLE_DIR / "shap_importance.csv")

# ---------------------------------------------------------------- 01 global bars
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5.5))
for ax, name in zip(np.atleast_1d(axes), models):
    s = imp[name].sort_values().tail(TOP)
    ax.barh(s.index, s.values, color=[KIND_COLOR[FEATURE_KIND[f]] for f in s.index], height=0.7)
    ax.set_title(name); ax.set_xlabel("mean |SHAP| (log-odds)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KIND_COLOR.values()]
fig.legend(handles, KIND_COLOR.keys(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Top features by mean |SHAP| - test sample", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "f_shap_01_global_bar.png"); plt.close(fig)

# ---------------------------------------------------------------- 02 beeswarms
for name in models:
    plt.figure()
    shap.summary_plot(shap_vals[name], X_num, max_display=TOP, show=False, plot_size=(8, 6))
    plt.title(f"{name}: SHAP beeswarm (red = high feature value)", fontweight="bold")
    plt.savefig(FIG_DIR / f"f_shap_02_beeswarm_{name.lower()}.png"); plt.close("all")

# ---------------------------------------------------------------- 03 group share
grp = pd.DataFrame({name: {g: imp[name][fs].sum() for g, fs in GROUPS.items()} for name in models})
grp = grp / grp.sum()
grp.round(4).to_csv(TABLE_DIR / "shap_group_share.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
order = grp.mean(1).sort_values().index
h = 0.8 / len(models)
for i, name in enumerate(models):
    ax.barh(np.arange(len(order)) + i * h, grp.loc[order, name], height=h * 0.9,
            color=MODEL_COLORS[name], label=name)
ax.set_yticks(np.arange(len(order)) + h * (len(models) - 1) / 2, order)
ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("share of total mean |SHAP|"); ax.legend(loc="lower right")
ax.set_title("Which kind of information drives the predictions?")
fig.savefig(FIG_DIR / "f_shap_03_group_share.png"); plt.close(fig)

# ---------------------------------------------------------------- 04 dependence
num_top = [f for f in imp[best].sort_values(ascending=False).index if f not in CATEGORICAL][:4]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, f in zip(axes, num_top):
    x = X_num[f].to_numpy(); s = shap_vals[best][:, FEATURES.index(f)]
    ok = ~np.isnan(x)
    lo, hi = np.nanpercentile(x, [1, 99]); keep = ok & (x >= lo) & (x <= hi)
    ax.scatter(x[keep], s[keep], s=3, alpha=0.15, color=C_BLUE, rasterized=True)
    bins = np.unique(np.nanpercentile(x[keep], np.linspace(0, 100, 21)))
    if len(bins) > 2:
        b = pd.cut(x[keep], bins, include_lowest=True)
        mean = pd.Series(s[keep]).groupby(b, observed=True).mean()
        mids = [iv.mid for iv in mean.index]
        ax.plot(mids, mean.values, color=C_ORANGE, label="binned mean")
    ax.axhline(0, color=C_MUTED, lw=1)
    ax.set_xlabel(f); ax.set_ylabel("SHAP (log-odds)")
    if not ok.all():
        ax.set_title(f"{(~ok).mean():.0%} missing (not shown)", fontsize=8, fontweight="normal")
axes[0].legend()
fig.suptitle(f"{best}: how the top features push the prediction", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG_DIR / "f_shap_04_dependence.png"); plt.close(fig)

# ---------------------------------------------------------------- 05 observed-info reliance by hour
obs_cols = [i for i, f in enumerate(FEATURES) if FEATURE_KIND[f] == "observed by T"]
hours = sample["dep_hour"].to_numpy()
fig, ax = plt.subplots(figsize=(8, 3.8))
rows = {}
for name in models:
    a = np.abs(shap_vals[name])
    share = a[:, obs_cols].sum(1) / a.sum(1)
    rows[name] = pd.Series(share).groupby(hours).mean()
    ax.plot(rows[name].index, rows[name].values, marker="o", ms=4, color=MODEL_COLORS[name], label=name)
pd.DataFrame(rows).round(4).to_csv(TABLE_DIR / "shap_observed_share_by_hour.csv")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Scheduled departure hour (local)"); ax.set_ylabel("share of |SHAP| from observed-by-T features")
ax.set_title("Early flights must rely on the schedule; later flights use what has happened")
ax.legend(); fig.savefig(FIG_DIR / "f_shap_05_observed_share_by_hour.png"); plt.close(fig)

# ---------------------------------------------------------------- 06 local explanations
p_best = predict_proba(best, models[best], X)
hit = int(np.argmax(np.where(y == 1, p_best, -1)))            # most confident correct LATE
miss = int(np.argmin(np.where(y == 1, p_best, 2)))            # LATE flight the model was surest was fine
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, i, title in [(axes[0], hit, "Confident hit"), (axes[1], miss, "Confident miss")]:
    s = pd.Series(shap_vals[best][i], index=FEATURES)
    s = s.reindex(s.abs().sort_values().index[-10:])
    ax.barh(s.index, s.values, color=[C_ORANGE if v > 0 else C_BLUE for v in s.values], height=0.7)
    ax.axvline(0, color=C_MUTED, lw=1)
    r = sample.iloc[i]
    ax.set_title(f"{title}: {r.ORIGIN_CITY_NAME} -> {r.DEST_CITY_NAME}, {r.FL_DATE.date()} "
                 f"{int(r.CRS_DEP_TIME):04d}\np(LATE)={p_best[i]:.2f}, actual delay {r.DEP_DELAY:.0f} min",
                 fontsize=9)
    ax.set_xlabel("SHAP (orange pushes towards LATE)")
fig.tight_layout(); fig.savefig(FIG_DIR / "f_shap_06_local.png"); plt.close(fig)

# ---------------------------------------------------------------- agreement between models
rank_corr = imp_df.corr(method="spearman").round(3)
rank_corr.to_csv(TABLE_DIR / "shap_rank_agreement.csv")
top10 = {n: set(imp[n].sort_values().tail(10).index) for n in models}
print("\nSpearman correlation of SHAP importance between models:\n", rank_corr)
print("\nTop-10 features shared by all models:", sorted(set.intersection(*top10.values())))
print("\nShare of SHAP by feature group:\n", grp.round(3))
print("\nTop 10 features of", best, ":\n", imp[best].sort_values(ascending=False).head(10).round(4))
