"""
features_pca.py - Feature encoding, feature filtering and PCA.

Input : train.csv, test.csv   (produced by cleanup.py)
Output: folder `features_out/` containing
          - X_train.npy / X_test.npy          full filtered + scaled features
          - X_train_pca.npy / X_test_pca.npy  PCA-reduced features
          - y_train.npy / y_test.npy          labels
          - feature_names.json, feature_groups.json
          - preprocessor.joblib               everything the UI needs at predict time
          - plots (PNG) and tables (CSV) for your report

Everything is FIT ON TRAIN ONLY and then applied to test (no leakage).
"""

import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                      # save plots to files, no window needed
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
OUT = "features_out"
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 42
TARGET = "delayed"

# Order matters for the correlation filter: when two features are highly
# correlated, the LATER one is dropped, so list the preferred one first.
NUMERIC = ["DISTANCE", "dep_hour", "day_of_week", "day_of_month", "is_weekend",
           "origin_hourly_flights", "origin_daily_flights", "is_codeshare",
           "CRS_DEP_TIME"]
ONEHOT = ["OP_UNIQUE_CARRIER", "MKT_UNIQUE_CARRIER", "dep_period"]
TARGET_ENC = ["ORIGIN_CITY_MARKET_ID", "ORIGIN_STATE_ABR",
              "DEST_CITY_MARKET_ID", "DEST_STATE_ABR", "route"]

CORR_THRESHOLD = 0.90        # drop one feature of any pair with |r| above this
RARE_ONEHOT_VAR = 1e-3       # drop one-hot columns active in < ~0.1% of rows
TE_SMOOTHING = 20            # target-encoding smoothing strength
PCA_VARIANCE = 0.95          # keep enough components for 95% of the variance
MI_SAMPLE = 50_000           # rows used for the mutual-information ranking
PLOT_SAMPLE = 20_000         # rows used for the PCA scatter plot

# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(f"Train {train.shape}, Test {test.shape}")

y_train = train[TARGET].to_numpy()
y_test = test[TARGET].to_numpy()
for c in ONEHOT + TARGET_ENC:
    train[c] = train[c].astype(str)
    test[c] = test[c].astype(str)


# ------------------------------------------------------------------
# 2. Target encoding for high-cardinality categoricals
#    Replaces a category by its (smoothed) delay rate. Train values are
#    computed out-of-fold so a row never sees its own label.
# ------------------------------------------------------------------
def te_map(cat, y, prior, m):
    stats = pd.DataFrame({"cat": cat.to_numpy(), "y": y}).groupby("cat")["y"].agg(["sum", "count"])
    return (stats["sum"] + m * prior) / (stats["count"] + m)


def target_encode(train_df, test_df, cols, y, m=TE_SMOOTHING, n_splits=5):
    prior = float(y.mean())
    tr_out = pd.DataFrame(index=train_df.index)
    te_out = pd.DataFrame(index=test_df.index)
    maps = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for c in cols:
        oof = np.full(len(train_df), prior, dtype=float)
        for fit_idx, val_idx in kf.split(train_df):
            mp = te_map(train_df[c].iloc[fit_idx], y[fit_idx], prior, m)
            oof[val_idx] = train_df[c].iloc[val_idx].map(mp).fillna(prior).to_numpy()
        tr_out[c + "_te"] = oof
        full = te_map(train_df[c], y, prior, m)
        maps[c] = full
        te_out[c + "_te"] = test_df[c].map(full).fillna(prior).to_numpy()
    return tr_out, te_out, maps, prior


te_train, te_test, te_maps, te_prior = target_encode(train, test, TARGET_ENC, y_train)

# ------------------------------------------------------------------
# 3. One-hot encoding for low-cardinality categoricals
# ------------------------------------------------------------------
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
oh_train = pd.DataFrame(ohe.fit_transform(train[ONEHOT]),
                        columns=ohe.get_feature_names_out(ONEHOT), index=train.index)
oh_test = pd.DataFrame(ohe.transform(test[ONEHOT]),
                       columns=ohe.get_feature_names_out(ONEHOT), index=test.index)

# ------------------------------------------------------------------
# 4. Assemble full feature table
# ------------------------------------------------------------------
Xtr = pd.concat([train[NUMERIC], te_train, oh_train], axis=1)
Xte = pd.concat([test[NUMERIC], te_test, oh_test], axis=1)
print(f"Features before filtering: {Xtr.shape[1]}")

# ------------------------------------------------------------------
# 5. FEATURE FILTERING
# ------------------------------------------------------------------
dropped = {}

# 5a. Near-constant numeric features
const = [c for c in NUMERIC + list(te_train.columns) if Xtr[c].nunique() <= 1]
for c in const:
    dropped[c] = "constant"

# 5b. Rare one-hot categories (almost never active -> no signal, noisy for PCA)
for c in oh_train.columns:
    if oh_train[c].var() < RARE_ONEHOT_VAR:
        dropped[c] = f"rare category (var<{RARE_ONEHOT_VAR})"

# 5c. Highly correlated pairs among numeric + target-encoded features
cont_cols = [c for c in NUMERIC + list(te_train.columns) if c not in dropped]
corr = Xtr[cont_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
for c in upper.columns:
    partners = upper.index[upper[c] > CORR_THRESHOLD].tolist()
    if partners:
        dropped[c] = f"|corr|={upper[c].max():.2f} with {partners[0]}"

# correlation heatmap (before dropping)
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr.where(~np.eye(len(corr), dtype=bool), 1.0), cmap="coolwarm", vmin=0, vmax=1)
ax.set_xticks(range(len(cont_cols)))
ax.set_yticks(range(len(cont_cols)))
ax.set_xticklabels(cont_cols, rotation=90, fontsize=8)
ax.set_yticklabels(cont_cols, fontsize=8)
fig.colorbar(im, label="|correlation|")
ax.set_title("Absolute correlation: numeric and target-encoded features")
fig.tight_layout()
fig.savefig(f"{OUT}/correlation_heatmap.png", dpi=150)
plt.close(fig)

keep = [c for c in Xtr.columns if c not in dropped]
Xtr, Xte = Xtr[keep], Xte[keep]
print(f"Features after filtering: {len(keep)}")
print("Dropped:")
for k, v in dropped.items():
    print(f"  {k:40s} {v}")
pd.Series(dropped, name="reason").to_csv(f"{OUT}/dropped_features.csv")

# 5d. Mutual information ranking (reported, used in your feature analysis)
rng = np.random.RandomState(RANDOM_STATE)
idx = rng.choice(len(Xtr), size=min(MI_SAMPLE, len(Xtr)), replace=False)
discrete = np.array([(c in oh_train.columns) or c in ("is_weekend", "is_codeshare") for c in keep])
mi = mutual_info_classif(Xtr.iloc[idx], y_train[idx], discrete_features=discrete,
                         random_state=RANDOM_STATE)
mi_s = pd.Series(mi, index=keep).sort_values(ascending=False)
mi_s.to_csv(f"{OUT}/mutual_information.csv", header=["mutual_info"])
print("\nTop 15 features by mutual information:")
print(mi_s.head(15).round(4).to_string())

top = mi_s.head(20)[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top.index, top.values)
ax.set_xlabel("Mutual information with `delayed`")
ax.set_title("Top 20 features by mutual information")
fig.tight_layout()
fig.savefig(f"{OUT}/mutual_information.png", dpi=150)
plt.close(fig)


# 5e. Feature groups (used later for ablation studies)
def group_of(name):
    if name in ("dep_hour", "CRS_DEP_TIME", "day_of_week", "day_of_month", "is_weekend") \
            or name.startswith("dep_period"):
        return "time"
    if name.startswith(("OP_UNIQUE_CARRIER", "MKT_UNIQUE_CARRIER")) or name == "is_codeshare":
        return "carrier"
    if name.startswith("ORIGIN_"):
        return "origin"
    if name.startswith(("DEST_", "route")) or name == "DISTANCE":
        return "route_destination"
    if name in ("origin_hourly_flights", "origin_daily_flights"):
        return "congestion"
    return "other"


groups = {}
for c in keep:
    groups.setdefault(group_of(c), []).append(c)
with open(f"{OUT}/feature_groups.json", "w") as f:
    json.dump(groups, f, indent=2)
print("\nFeature groups:", {g: len(v) for g, v in groups.items()})

# ------------------------------------------------------------------
# 6. Scale (fit on train only) - required for PCA
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(Xtr).astype(np.float32)
X_test = scaler.transform(Xte).astype(np.float32)

# ------------------------------------------------------------------
# 7. PCA
# ------------------------------------------------------------------
pca_full = PCA(random_state=RANDOM_STATE).fit(X_train)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)
n_comp = int(np.searchsorted(cum, PCA_VARIANCE) + 1)
print(f"\nPCA: {n_comp} of {X_train.shape[1]} components explain >= {PCA_VARIANCE:.0%} of variance")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(range(1, len(evr) + 1), evr)
ax[0].set_xlabel("Principal component")
ax[0].set_ylabel("Explained variance ratio")
ax[0].set_title("Scree plot")
ax[1].plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
ax[1].axhline(PCA_VARIANCE, color="r", ls="--", label=f"{PCA_VARIANCE:.0%}")
ax[1].axvline(n_comp, color="g", ls="--", label=f"{n_comp} components")
ax[1].set_xlabel("Number of components")
ax[1].set_ylabel("Cumulative explained variance")
ax[1].set_title("Cumulative variance")
ax[1].legend()
fig.tight_layout()
fig.savefig(f"{OUT}/pca_variance.png", dpi=150)
plt.close(fig)

pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(X_train)
X_train_pca = pca.transform(X_train).astype(np.float32)
X_test_pca = pca.transform(X_test).astype(np.float32)

# 2D projection coloured by delay (class separation visual)
pidx = rng.choice(len(X_train_pca), size=min(PLOT_SAMPLE, len(X_train_pca)), replace=False)
fig, ax = plt.subplots(figsize=(7, 6))
for label, colour, name in [(0, "tab:blue", "On time"), (1, "tab:red", "Delayed")]:
    m = y_train[pidx] == label
    ax.scatter(X_train_pca[pidx][m, 0], X_train_pca[pidx][m, 1], s=4, alpha=0.35, c=colour, label=name)
ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
ax.set_title("Training data projected onto the first two PCs")
ax.legend(markerscale=4)
fig.tight_layout()
fig.savefig(f"{OUT}/pca_2d_scatter.png", dpi=150)
plt.close(fig)

# Loadings: which original features drive each PC
loadings = pd.DataFrame(pca.components_[:5].T, index=keep,
                        columns=[f"PC{i + 1}" for i in range(min(5, n_comp))])
loadings.to_csv(f"{OUT}/pca_loadings_top5.csv")
for pc in loadings.columns[:3]:
    print(f"\nTop features in {pc}:")
    print(loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index).head(6).round(3).to_string())

# ------------------------------------------------------------------
# 8. Save everything
# ------------------------------------------------------------------
np.save(f"{OUT}/X_train.npy", X_train)
np.save(f"{OUT}/X_test.npy", X_test)
np.save(f"{OUT}/X_train_pca.npy", X_train_pca)
np.save(f"{OUT}/X_test_pca.npy", X_test_pca)
np.save(f"{OUT}/y_train.npy", y_train)
np.save(f"{OUT}/y_test.npy", y_test)
with open(f"{OUT}/feature_names.json", "w") as f:
    json.dump(keep, f, indent=2)

joblib.dump({
    "numeric": NUMERIC, "onehot": ONEHOT, "target_enc": TARGET_ENC,
    "te_maps": te_maps, "te_prior": te_prior, "ohe": ohe,
    "kept_columns": keep, "scaler": scaler, "pca": pca,
}, f"{OUT}/preprocessor.joblib")

print(f"\nSaved arrays, plots and preprocessor to ./{OUT}/")
