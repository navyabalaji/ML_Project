"""Collects every results/*.json into one comparison table (results/summary.csv)."""
import glob
import json

import pandas as pd

rows = [json.load(open(p)) for p in glob.glob("results/*.json")]
df = pd.DataFrame(rows)
cols = ["model", "feature_set", "accuracy", "precision", "recall", "f1", "roc_auc",
        "pr_auc", "train_time_s", "predict_time_s", "model_size_mb", "n_features"]
df = df[cols].sort_values(["feature_set", "roc_auc"], ascending=[True, False])
df.to_csv("results/summary.csv", index=False)
print(df.to_string(index=False))
