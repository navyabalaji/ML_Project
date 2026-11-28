"""
eda.py - P2 exploratory data analysis on the cleaned BTS data.

Input : ../cleaned_flights.csv
Output: outputs/figures/eda_*.png  and  outputs/tables/eda_*.csv
Run   : python p2_eda_features/eda.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (CLEANED_CSV, FIG_DIR, TABLE_DIR, LATE_THRESHOLD, DEV_END, GAP_DAY,
                    TEST_START, C_BLUE, C_ORANGE, C_MUTED, C_TEXT, style_plots)

plt = style_plots()
df = pd.read_csv(CLEANED_CSV, parse_dates=["FL_DATE"], low_memory=False)
df["LATE"] = (df["DEP_DELAY"] >= LATE_THRESHOLD).astype(int)
rate = df["LATE"].mean()


def save(fig, name):
    fig.savefig(FIG_DIR / f"eda_{name}.png"); plt.close(fig); print("  saved eda_" + name)


def rate_bar(ax, s, horizontal=False):
    """Bar chart of LATE rate with the overall rate as a reference line."""
    labels = s.index.astype(str)
    if horizontal:
        ax.barh(labels, s.values, color=C_BLUE, height=0.7)
        ax.axvline(rate, color=C_ORANGE, ls="--", lw=1.5, label=f"overall {rate:.1%}")
        ax.legend(loc="lower right")
    else:
        ax.bar(labels, s.values, color=C_BLUE, width=0.75)
        ax.axhline(rate, color=C_ORANGE, ls="--", lw=1.5, label=f"overall {rate:.1%}")
        ax.legend(loc="upper left")


# ---------------------------------------------------------------- 1. basic facts
print("=== Basic facts ===")
facts = {
    "flights": len(df), "days": df["FL_DATE"].nunique(),
    "date_range": f"{df.FL_DATE.min().date()} to {df.FL_DATE.max().date()}",
    "operating_carriers": df["OP_UNIQUE_CARRIER"].nunique(),
    "origin_markets": df["ORIGIN_CITY_MARKET_ID"].nunique(), "routes": df["route"].nunique(),
    "aircraft (tails)": df["TAIL_NUM"].nunique(),
    "LATE rate (>=15 min)": round(rate, 4),
    "median DEP_DELAY": df["DEP_DELAY"].median(), "mean DEP_DELAY": round(df["DEP_DELAY"].mean(), 2),
    "share departed early (<0)": round((df["DEP_DELAY"] < 0).mean(), 4),
    "share delayed > 3 h": round((df["DEP_DELAY"] > 180).mean(), 4),
    "missing values (any column)": int(df.isna().sum().sum()),
}
for k, v in facts.items():
    print(f"  {k:30s} {v}")
pd.Series(facts).to_csv(TABLE_DIR / "eda_basic_facts.csv", header=["value"])

# split sizes / class balance per split
split = np.where(df.FL_DATE <= DEV_END, "dev (Jan 1-23)",
                 np.where(df.FL_DATE == GAP_DAY, "gap (Jan 24)", "test (Jan 25-31)"))
bal = df.groupby(split)["LATE"].agg(flights="size", late_rate="mean").round(4)
print("\n=== Class balance by split ===\n", bal)
bal.to_csv(TABLE_DIR / "eda_split_balance.csv")

# describe numeric columns
desc = df[["DEP_DELAY", "DISTANCE", "CRS_DEP_TIME", "origin_hourly_flights",
           "origin_daily_flights"]].describe().T.round(2)
desc.to_csv(TABLE_DIR / "eda_numeric_summary.csv"); print("\n", desc)

# ---------------------------------------------------------------- 2. delay distribution
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(df["DEP_DELAY"].clip(-30, 180), bins=np.arange(-30, 182, 3), color=C_BLUE)
ax.axvline(LATE_THRESHOLD, color=C_ORANGE, ls="--", label="LATE threshold (15 min)")
ax.set_yscale("log"); ax.set_xlabel("Departure delay (min, clipped to [-30, 180])")
ax.set_ylabel("Flights (log scale)"); ax.set_title("Departure delay is heavily right-skewed")
ax.legend(); save(fig, "01_delay_distribution")

# ---------------------------------------------------------------- 3. late rate over time
daily = df.groupby("FL_DATE")["LATE"].agg(["mean", "size"])
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axvspan(pd.Timestamp(GAP_DAY) - pd.Timedelta(hours=12), pd.Timestamp(GAP_DAY) + pd.Timedelta(hours=12),
           color="#dddddd", label="gap day")
ax.axvspan(pd.Timestamp(TEST_START) - pd.Timedelta(hours=12), daily.index.max() + pd.Timedelta(hours=12),
           color="#f9c4a6", alpha=0.6, label="test period")
ax.plot(daily.index, daily["mean"], marker="o", ms=4, color=C_BLUE, label="LATE rate")
ax.set_ylabel("Share of flights LATE"); ax.set_title("Daily LATE rate - winter storms Jan 9 and Jan 12-16")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); ax.legend(loc="upper right")
fig.autofmt_xdate(); save(fig, "02_daily_late_rate")

# ---------------------------------------------------------------- 4. by hour / weekday
df["dep_hour"] = (df["CRS_DEP_TIME"] // 100).clip(upper=23)
fig, ax = plt.subplots(figsize=(8, 3.5))
by_hour = df.groupby("dep_hour")["LATE"].mean()
rate_bar(ax, by_hour); ax.set_xlabel("Scheduled departure hour (local)")
ax.set_ylabel("LATE rate"); ax.set_title("Delays build up through the day")
save(fig, "03_late_by_hour")

piv = df.pivot_table(index="day_of_week", columns="dep_hour", values="LATE", aggfunc="mean")
fig, ax = plt.subplots(figsize=(9, 3))
im = ax.imshow(piv.values, aspect="auto", cmap="Blues")
ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_xticks(range(len(piv.columns)), piv.columns); ax.grid(False)
ax.set_xlabel("Scheduled departure hour"); ax.set_title("LATE rate by weekday x hour")
fig.colorbar(im, ax=ax, label="LATE rate"); save(fig, "04_heatmap_weekday_hour")

# ---------------------------------------------------------------- 5. carriers and airports
car = df.groupby("OP_UNIQUE_CARRIER")["LATE"].agg(["mean", "size"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(6, 5))
rate_bar(ax, car["mean"], horizontal=True)
for i, (m, n) in enumerate(zip(car["mean"], car["size"])):
    ax.text(m + 0.005, i, f"n={n:,}", va="center", color=C_MUTED, fontsize=7)
ax.set_xlabel("LATE rate"); ax.set_title("LATE rate by operating carrier"); save(fig, "05_late_by_carrier")
car.round(4).to_csv(TABLE_DIR / "eda_carrier_rates.csv")

top = df["ORIGIN_CITY_NAME"].value_counts().head(20).index
org = df[df.ORIGIN_CITY_NAME.isin(top)].groupby("ORIGIN_CITY_NAME")["LATE"].mean().sort_values()
fig, ax = plt.subplots(figsize=(6, 5))
rate_bar(ax, org, horizontal=True); ax.set_xlabel("LATE rate")
ax.set_title("20 busiest origin markets"); save(fig, "06_late_by_origin")

# ---------------------------------------------------------------- 6. distance, congestion, rotation
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, col, bins, lab in [
    (axes[0], "DISTANCE", [0, 250, 500, 750, 1000, 1500, 2000, 3000, 6000], "Distance (miles)"),
    (axes[1], "origin_hourly_flights", [0, 5, 10, 20, 40, 60, 80, 120, 300], "Scheduled departures in that hour at origin"),
]:
    s = df.groupby(pd.cut(df[col], bins), observed=True)["LATE"].mean()
    ax.bar(range(len(s)), s.values, color=C_BLUE); ax.set_xticks(range(len(s)), [str(i) for i in s.index], rotation=45, ha="right")
    ax.axhline(rate, color=C_ORANGE, ls="--", lw=1.5); ax.set_xlabel(lab)
df = df.sort_values(["TAIL_NUM", "FL_DATE", "CRS_DEP_TIME"])
df["leg"] = df.groupby(["TAIL_NUM", "FL_DATE"]).cumcount() + 1
s = df[df.leg <= 8].groupby("leg")["LATE"].mean()
axes[2].bar(s.index.astype(str), s.values, color=C_BLUE); axes[2].axhline(rate, color=C_ORANGE, ls="--", lw=1.5)
axes[2].set_xlabel("Leg number of the aircraft that day")
axes[0].set_ylabel("LATE rate"); fig.suptitle("Distance and scheduled load matter little; the aircraft rotation matters a lot", fontweight="bold")
save(fig, "07_distance_congestion_rotation")

# ---------------------------------------------------------------- 7. delay propagation (raw)
df["prev_delay"] = df.groupby(["TAIL_NUM", "FL_DATE"])["DEP_DELAY"].shift(1)
bins = [-60, 0, 15, 30, 60, 120, 4000]
s = df.groupby(pd.cut(df["prev_delay"], bins), observed=True)["LATE"].mean()
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar([str(i) for i in s.index], s.values, color=C_BLUE); ax.axhline(rate, color=C_ORANGE, ls="--", lw=1.5, label="overall")
ax.set_xlabel("Delay of the same aircraft's previous leg (min)"); ax.set_ylabel("LATE rate")
ax.set_title("Delay propagates along the aircraft rotation"); ax.legend(); save(fig, "08_delay_propagation")

print("\nEDA done. Figures in", FIG_DIR)
