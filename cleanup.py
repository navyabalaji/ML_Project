"""
cleanup.py - Flight departure delay prediction: loading, cleaning, labelling,
feature engineering and date-based train/test split.

Input : T_ONTIME_MARKETING.csv  (BTS January 2024)
Output: cleaned_flights.csv, train.csv, test.csv
"""

import pandas as pd

CSV_PATH = "T_ONTIME_MARKETING.csv"
DELAY_THRESHOLD = 15            # minutes: delayed = departed more than this late
SPLIT_DATE = "2024-01-24"       # train: <= this date, test: after it

# ------------------------------------------------------------------
# 1. Load only the columns we need (also avoids the DtypeWarning)
# ------------------------------------------------------------------
use = [
    "FL_DATE", "MKT_UNIQUE_CARRIER", "OP_UNIQUE_CARRIER", "TAIL_NUM",
    "OP_CARRIER_FL_NUM", "ORIGIN_CITY_MARKET_ID", "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR", "DEST_CITY_MARKET_ID", "DEST_CITY_NAME",
    "DEST_STATE_ABR", "CRS_DEP_TIME", "DISTANCE",
    "DEP_DELAY", "CANCELLED", "DIVERTED",
]
df = pd.read_csv(CSV_PATH, usecols=use, dtype={"FL_DATE": str})
print("Raw shape:", df.shape)
print("Sample raw dates:", df["FL_DATE"].head(3).tolist())


# ------------------------------------------------------------------
# 2. Parse the date robustly
#    Tries several formats and keeps the one that gives dates inside
#    January 2024 (the dataset's range).
# ------------------------------------------------------------------
def parse_dates(series):
    formats = [
        "%m/%d/%Y %I:%M:%S %p",   # 1/1/2024 12:00:00 AM  (BTS default)
        "%Y-%m-%d",               # 2024-01-01
        "%d-%m-%Y",               # 01-01-2024 (day first)
        "%d/%m/%Y",               # 01/01/2024 (day first)
        "%m/%d/%Y",               # 1/1/2024
    ]
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.99:
            if parsed.min() >= pd.Timestamp("2024-01-01") and \
               parsed.max() <= pd.Timestamp("2024-01-31"):
                print(f"Date format detected: {fmt}")
                return parsed
    raise ValueError(
        "Could not parse FL_DATE into January 2024. "
        "Check the raw values printed above and add the right format."
    )


df["FL_DATE"] = parse_dates(df["FL_DATE"].str.strip())
print("Date range:", df["FL_DATE"].min().date(), "to", df["FL_DATE"].max().date())

# ------------------------------------------------------------------
# 3. Clean: remove cancelled/diverted flights and missing delays
# ------------------------------------------------------------------
df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
df = df.dropna(subset=["DEP_DELAY", "CRS_DEP_TIME", "DISTANCE"])
df = df.drop(columns=["CANCELLED", "DIVERTED"]).reset_index(drop=True)

# ------------------------------------------------------------------
# 4. Label  (DEP_DELAY is used ONLY here - never as a feature)
# ------------------------------------------------------------------
df["delayed"] = (df["DEP_DELAY"] > DELAY_THRESHOLD).astype(int)

# ------------------------------------------------------------------
# 5. Feature engineering (all known before departure -> no leakage)
# ------------------------------------------------------------------
df["day_of_week"] = df["FL_DATE"].dt.dayofweek          # 0 = Monday
df["day_of_month"] = df["FL_DATE"].dt.day
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df["dep_hour"] = (df["CRS_DEP_TIME"] // 100).clip(upper=23).astype(int)
df["dep_period"] = pd.cut(
    df["dep_hour"], bins=[-1, 5, 11, 17, 23],
    labels=["night", "morning", "afternoon", "evening"],
).astype(str)

# Congestion: scheduled departures from the same city market in the same hour
df["origin_hourly_flights"] = (
    df.groupby(["FL_DATE", "ORIGIN_CITY_MARKET_ID", "dep_hour"])["FL_DATE"]
    .transform("size")
)

# Flights scheduled that day from the same market (daily load)
df["origin_daily_flights"] = (
    df.groupby(["FL_DATE", "ORIGIN_CITY_MARKET_ID"])["FL_DATE"].transform("size")
)

# Route identifier (origin -> destination market)
df["route"] = (
    df["ORIGIN_CITY_MARKET_ID"].astype(str) + "_" + df["DEST_CITY_MARKET_ID"].astype(str)
)

# Whether the marketing carrier differs from the operating carrier (codeshare)
df["is_codeshare"] = (df["MKT_UNIQUE_CARRIER"] != df["OP_UNIQUE_CARRIER"]).astype(int)

# ------------------------------------------------------------------
# 6. Quick checks / EDA numbers
# ------------------------------------------------------------------
print("\nCleaned shape:", df.shape)
print("Overall delay rate:", round(df["delayed"].mean(), 4))
print("\nDelay rate by day:")
print(df.groupby("FL_DATE")["delayed"].mean().round(3).to_string())
print("\nDelay rate by departure hour:")
print(df.groupby("dep_hour")["delayed"].mean().round(3).to_string())
print("\nDelay rate by operating carrier:")
print(df.groupby("OP_UNIQUE_CARRIER")["delayed"].mean().round(3).sort_values().to_string())
print("\nMissing values (top 5):")
print(df.isna().mean().sort_values(ascending=False).head(5))

# ------------------------------------------------------------------
# 7. Time-based split: train on earlier days, test on later days
# ------------------------------------------------------------------
train = df[df["FL_DATE"] <= SPLIT_DATE]
test = df[df["FL_DATE"] > SPLIT_DATE]
print(f"\nTrain: {len(train)} rows ({train['delayed'].mean():.3f} delayed)")
print(f"Test : {len(test)} rows ({test['delayed'].mean():.3f} delayed)")

# ------------------------------------------------------------------
# 8. Save
#    Keep DEP_DELAY in the files for optional regression, but it must be
#    excluded from the feature matrix X when training classifiers.
# ------------------------------------------------------------------
df.to_csv("cleaned_flights.csv", index=False)
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
print("\nSaved cleaned_flights.csv, train.csv, test.csv")

# Feature lists for the next step (modelling)
NUMERIC = ["DISTANCE", "CRS_DEP_TIME", "dep_hour", "day_of_week", "day_of_month",
           "is_weekend", "origin_hourly_flights", "origin_daily_flights", "is_codeshare"]
CATEGORICAL = ["OP_UNIQUE_CARRIER", "MKT_UNIQUE_CARRIER", "ORIGIN_CITY_MARKET_ID",
               "ORIGIN_STATE_ABR", "DEST_CITY_MARKET_ID", "DEST_STATE_ABR", "dep_period"]
TARGET = "delayed"