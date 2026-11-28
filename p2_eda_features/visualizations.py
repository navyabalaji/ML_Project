"""
visualizations.py - P2 report figures built on the frozen feature table,
plus the final results table/figure (reads every outputs/results/*.json,
so teammates' models appear automatically if they save the same format).

Run: python p2_eda_features/visualizations.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (FEATURES_CSV, FIG_DIR, TABLE_DIR, RESULT_DIR, PRED_DIR, DEV_END, TEST_START,
                    C_BLUE, C_ORANGE, C_AQUA, C_MUTED, MODEL_COLORS, style_plots)

plt = style_plots()
META = json.loads((Path(__file__).parent / "feature_groups.json").read_text())
df = pd.read_csv(FEATURES_CSV, parse_dates=["FL_DATE"], low_memory=False)
rate = df["LATE"].mean()
pct = plt.matplotlib.ticker.PercentFormatter(1.0)


def save(fig, name):
    fig.savefig(FIG_DIR / f"viz_{name}.png"); plt.close(fig); print("  saved viz_" + name)


def binned_rate(ax, col, bins, xlabel, labels=None):
    b = pd.cut(df[col], bins, include_lowest=True)
    s = df.groupby(b, observed=True)["LATE"].agg(["mean", "size"])
    lab = labels or [str(i) for i in s.index]
    ax.bar(lab, s["mean"], color=C_BLUE, width=0.75)
    ax.axhline(rate, color=C_ORANGE, ls="--", lw=1.5, label=f"overall {rate:.1%}")
    ax.set_xlabel(xlabel); ax.yaxis.set_major_formatter(pct)
    ax.tick_params(axis="x", rotation=40)
    if df[col].isna().mean() >= 0.005:
        ax.set_title(f"missing: {df[col].isna().mean():.0%} (LATE {df.loc[df[col].isna(), 'LATE'].mean():.0%})",
                     fontsize=8, fontweight="normal")


# ---------------------------------------------------------------- 1. the engineered signals
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
binned_rate(axes[0, 0], "prev_leg_delay_lb", [-60, 0, 15, 30, 60, 120, 5000],
            "Previous leg delay known at T (min, lower bound)")
binned_rate(axes[0, 1], "planned_turn_slack", [-500, 0, 30, 60, 120, 240, 2000],
            "Planned turnaround slack (min)")
binned_rate(axes[0, 2], "prev_leg_overdue", [-0.5, 0.5, 1.5], "Previous leg overdue at T",
            labels=["no", "yes"])
binned_rate(axes[1, 0], "origin_recent_late_rate", [0, .1, .2, .3, .5, .7, 1],
            "Origin LATE rate in the 2 h before T")
binned_rate(axes[1, 1], "origin_backlog_at_T", [0, 2, 5, 10, 20, 50, 1000],
            "Flights stuck at origin at T (backlog)")
binned_rate(axes[1, 2], "national_recent_late_rate", [0, .1, .15, .2, .3, .4, 1],
            "National LATE rate in the 2 h before T")
for ax in axes[:, 0]:
    ax.set_ylabel("LATE rate")
axes[0, 0].legend(loc="upper left")
fig.suptitle("Engineered features: aircraft history (top) and observed congestion (bottom)", fontweight="bold")
fig.tight_layout(); save(fig, "01_engineered_signals")

# ---------------------------------------------------------------- 2. class separation
cols = ["prev_leg_delay_lb", "origin_recent_mean_delay", "carrier_recent_late_rate", "dep_minute_of_day"]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, c in zip(axes, cols):
    lo, hi = df[c].quantile([0.01, 0.99])
    bins = np.linspace(lo, hi, 40)
    for lab, color, name in [(0, C_BLUE, "on time"), (1, C_ORANGE, "LATE")]:
        ax.hist(df.loc[df.LATE == lab, c].clip(lo, hi).dropna(), bins=bins, density=True,
                histtype="step", lw=2, color=color, label=name)
    ax.set_xlabel(c); ax.set_yticks([])
axes[0].legend(); fig.suptitle("Class separation: distributions overlap heavily -> limited signal", fontweight="bold")
fig.tight_layout(); save(fig, "02_class_separation")

# ---------------------------------------------------------------- 3. correlation of numeric features
num = [f for f in META["features"] if f not in META["categorical"]]
corr = df[num].sample(100_000, random_state=0).corr(method="spearman")
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num)), num, rotation=90, fontsize=7); ax.set_yticks(range(len(num)), num, fontsize=7)
ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman correlation")
ax.set_title("Feature correlations (why single-feature importance can mislead)")
save(fig, "03_feature_correlation")

# ---------------------------------------------------------------- 4. missingness of engineered features
miss = df[META["features"]].isna().mean().loc[lambda s: s > 0].sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(miss.index, miss.values, color=C_BLUE, height=0.7); ax.xaxis.set_major_formatter(pct)
ax.set_title("Missing values are structural (first leg, quiet airports, day 1)")
ax.set_xlabel("share missing - kept as NaN, handled natively by tree models")
save(fig, "04_missingness")

# ---------------------------------------------------------------- 5. observed info over the day
df["_prev_known"] = df["prev_leg_departed_by_T"].fillna(0)   # share of ALL flights
hour = df.groupby("dep_hour").agg(prev_known=("_prev_known", "mean"), has_prev=("has_prev_leg", "mean"))
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(hour.index, hour["has_prev"], marker="o", ms=4, color=C_BLUE, label="aircraft has an earlier leg (24 h)")
ax.plot(hour.index, hour["prev_known"], marker="o", ms=4, color=C_ORANGE, label="previous leg had already departed by T")
ax.yaxis.set_major_formatter(pct); ax.set_ylim(0, 1.02); ax.set_xlabel("Scheduled departure hour (local)")
ax.set_title("How much aircraft history is visible at T = 2 h before departure"); ax.legend()
save(fig, "05_info_available_by_hour")

# ---------------------------------------------------------------- 6. results table + chart
res = []
for f in sorted(RESULT_DIR.glob("*.json")):
    r = json.loads(f.read_text())
    if "cv" not in r:
        continue
    res.append({"model": r["model"],
                **{f"cv_{k}": r["cv"][k] for k in ("roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std")},
                **{f"test_{k}": r["test"][k] for k in ("roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std")},
                "baseline_test_pr_auc": r["baseline_test"]["pr_auc"], "n_test": r["n_test_flights"]})
if res:
    res = pd.DataFrame(res).sort_values("cv_pr_auc_mean", ascending=False)
    res.round(4).to_csv(TABLE_DIR / "results_table.csv", index=False)
    md = ["| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC |", "|---|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append(f"| {r.model} | {r.cv_roc_auc_mean:.4f} ± {r.cv_roc_auc_std:.4f} | "
                  f"{r.cv_pr_auc_mean:.4f} ± {r.cv_pr_auc_std:.4f} | {r.test_roc_auc_mean:.4f} ± {r.test_roc_auc_std:.4f} | "
                  f"{r.test_pr_auc_mean:.4f} ± {r.test_pr_auc_std:.4f} |")
    md.append(f"| Majority class | 0.5000 | (prevalence) | 0.5000 | {res.baseline_test_pr_auc.iloc[0]:.4f} |")
    (TABLE_DIR / "results_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 * len(res) + 2))
    y = np.arange(len(res))
    for ax, m, lab in [(axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC")]:
        ax.errorbar(res[f"cv_{m}_mean"], y + 0.15, xerr=res[f"cv_{m}_std"], fmt="o", ms=8,
                    color=C_BLUE, label="CV (3 folds x 3 seeds)")
        ax.errorbar(res[f"test_{m}_mean"], y - 0.15, xerr=res[f"test_{m}_std"], fmt="s", ms=8,
                    color=C_ORANGE, label="test (3 seeds)")
        base = 0.5 if m == "roc_auc" else res.baseline_test_pr_auc.iloc[0]
        ax.axvline(base, color=C_MUTED, ls=":", label="majority baseline (test)")
        ax.set_yticks(y, res.model); ax.set_xlabel(lab); ax.invert_yaxis()
    axes[0].legend(loc="lower right"); fig.suptitle("Model comparison (mean ± std)", fontweight="bold")
    fig.tight_layout(); save(fig, "06_results_comparison")

# ---------------------------------------------------------------- 7. calibration of saved test predictions
preds = {f.stem.replace("_test", ""): pd.read_csv(f) for f in sorted(PRED_DIR.glob("*_test.csv"))}
if preds:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=C_MUTED, ls=":", label="perfect calibration")
    for i, (name, p) in enumerate(preds.items()):
        b = pd.cut(p["proba"], np.linspace(0, 1, 11), include_lowest=True)
        cal = p.groupby(b, observed=True).agg(pred=("proba", "mean"), obs=("LATE", "mean"))
        color = {k.lower(): v for k, v in MODEL_COLORS.items()}.get(name, [C_BLUE, C_ORANGE, C_AQUA][i % 3])
        ax.plot(cal["pred"], cal["obs"], marker="o", ms=5, color=color, label=name)
    ax.set_xlabel("predicted P(LATE)"); ax.set_ylabel("observed LATE rate (test)")
    ax.set_title("Calibration on the test period"); ax.legend()
    save(fig, "07_calibration")

print("Done. Figures in", FIG_DIR)
