"""
features.py - P2 feature engineering. Builds and FREEZES the feature table.

Input : ../cleaned_flights.csv   (from P1's cleanup.py)
Output: ../features_v1.csv       (the frozen feature table)
        p2_eda_features/feature_groups.json   (feature lists, used by all models)
        p2_eda_features/feature_audit.md      (audit rows for every new feature)
        ../features_v1_info.json              (row count + checksum = "frozen")

Checkpoint T = scheduled departure - 2 h (config.CHECKPOINT_MIN).
Every feature is either known in advance (schedule/calendar/static) or was
observed at or before T. Nothing about the flight's own departure is used.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (CLEANED_CSV, FEATURES_CSV, LATE_THRESHOLD, CHECKPOINT_MIN,
                    WINDOW_MIN, DEV_START, DEV_END, HOLIDAYS)

HERE = Path(__file__).resolve().parent
DAY0 = pd.Timestamp(DEV_START)

# Standard-time UTC offset (hours) by state. January = no daylight saving.
# A few states span two zones; we use the zone of their main airports.
TZ = {**dict.fromkeys("CT DE FL GA IN KY MA MD ME MI NC NH NJ NY OH PA RI SC VA VT WV DC".split(), -5),
      **dict.fromkeys("AL AR IA IL KS LA MN MO MS ND NE OK SD TN TX WI".split(), -6),
      **dict.fromkeys("AZ CO ID MT NM UT WY".split(), -7),
      **dict.fromkeys("CA NV OR WA".split(), -8),
      "AK": -9, "HI": -10, "PR": -4, "VI": -4, "TT": 10}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def window_stats(ev_key, ev_time, ev_vals, q_key, q_time, width):
    """For each query (key, t): count and sums of events with the same key
    and event time in (t - width, t]. Fully vectorised (sort + searchsorted)."""
    big = 1e6  # times are minutes in [-1e4, 6e4] -> safely below 1e6 after shift
    ev = ev_key * big + (ev_time + 2e4)
    order = np.argsort(ev, kind="stable")
    ev = ev[order]
    cums = [np.concatenate([[0.0], np.cumsum(v[order])]) for v in ev_vals]
    hi = np.searchsorted(ev, q_key * big + (q_time + 2e4), side="right")
    lo = np.searchsorted(ev, q_key * big + (q_time - width + 2e4), side="right")
    return hi - lo, [c[hi] - c[lo] for c in cums]


def count_upto(ev_key, ev_time, q_key, q_time):
    """Number of events with the same key and time <= t (all history)."""
    big = 1e6
    ev = np.sort(ev_key * big + (ev_time + 2e4))
    q = q_key * big + (q_time + 2e4)
    return np.searchsorted(ev, q, side="right") - np.searchsorted(ev, q_key * big, side="left")


def prior_day_rate(df, key, dev_end_idx, m=20):
    """Smoothed LATE rate of `key` over earlier days only, and only using DEV
    days (so the test period never feeds a statistic)."""
    k = df[key].astype("category").cat.codes.to_numpy()
    d = df["day_idx"].to_numpy()
    n_days = d.max() + 1
    use = d <= dev_end_idx
    cnt = np.zeros((k.max() + 1, n_days)); late = np.zeros_like(cnt)
    np.add.at(cnt, (k[use], d[use]), 1)
    np.add.at(late, (k[use], d[use]), df["LATE"].to_numpy()[use])
    # cumulative up to and including day-1  -> strictly earlier days
    c_cnt = np.cumsum(cnt, axis=1); c_late = np.cumsum(late, axis=1)
    prev_cnt = np.where(d > 0, c_cnt[k, np.maximum(d - 1, 0)], 0)
    prev_late = np.where(d > 0, c_late[k, np.maximum(d - 1, 0)], 0)
    g_cnt = cnt.sum(0).cumsum(); g_late = late.sum(0).cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        g_rate = np.where(d > 0, g_late[np.maximum(d - 1, 0)] / g_cnt[np.maximum(d - 1, 0)], np.nan)
        rate = (prev_late + m * g_rate) / (prev_cnt + m)
    return rate, prev_cnt


# ----------------------------------------------------------------------
# 0. load + label
# ----------------------------------------------------------------------
df = pd.read_csv(CLEANED_CSV, parse_dates=["FL_DATE"], low_memory=False)
print("Loaded", df.shape)
df["LATE"] = (df["DEP_DELAY"] >= LATE_THRESHOLD).astype(int)   # brief says ">= 15"
df = df.drop(columns=["delayed"], errors="ignore")

# ----------------------------------------------------------------------
# 1. time axis: everything in absolute UTC minutes since 2024-01-01 00:00
# ----------------------------------------------------------------------
df["day_idx"] = (df["FL_DATE"] - DAY0).dt.days
hhmm = df["CRS_DEP_TIME"].astype(int)
df["dep_min_local"] = (hhmm // 100) * 60 + hhmm % 100
offset = df["ORIGIN_STATE_ABR"].map(TZ).fillna(-6).astype(float)
df["sched_utc"] = df["day_idx"] * 1440 + df["dep_min_local"] - offset * 60
df["actual_utc"] = df["sched_utc"] + df["DEP_DELAY"]           # used ONLY for "observed by T" logic
df["T_utc"] = df["sched_utc"] - CHECKPOINT_MIN
# sanity: a flight can never have departed by its own checkpoint
assert (df["actual_utc"] > df["T_utc"]).all()

# ----------------------------------------------------------------------
# 2. CALENDAR features (known in advance)
# ----------------------------------------------------------------------
df["day_of_week"] = df["FL_DATE"].dt.dayofweek
df["day_of_month"] = df["FL_DATE"].dt.day
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["dep_hour"] = (df["dep_min_local"] // 60).clip(upper=23)
df["dep_minute_of_day"] = df["dep_min_local"]
df["hour_sin"] = np.sin(2 * np.pi * df["dep_min_local"] / 1440)
df["hour_cos"] = np.cos(2 * np.pi * df["dep_min_local"] / 1440)
hol = pd.to_datetime(HOLIDAYS).to_numpy()
dist = np.abs((df["FL_DATE"].to_numpy()[:, None] - hol[None, :]) / np.timedelta64(1, "D")).min(axis=1)
df["days_to_holiday"] = dist.astype(int)
df["is_holiday"] = (dist == 0).astype(int)

# ----------------------------------------------------------------------
# 3. AIRCRAFT HISTORY (same TAIL_NUM, earlier legs)
# ----------------------------------------------------------------------
df = df.sort_values(["TAIL_NUM", "sched_utc"]).reset_index(drop=True)
same_day = df.groupby(["TAIL_NUM", "FL_DATE"], sort=False)
df["tail_leg_number"] = same_day.cumcount() + 1                         # known in advance
df["tail_legs_today"] = same_day["FL_DATE"].transform("size")           # known in advance

g = df.groupby("TAIL_NUM", sort=False)
prev = {}
for k in (1, 2, 3):
    p_sched = g["sched_utc"].shift(k)
    ok = (df["sched_utc"] - p_sched) <= 24 * 60          # only legs within the last 24 h
    prev[k] = dict(
        sched=p_sched.where(ok),
        actual=g["actual_utc"].shift(k).where(ok),
        delay=g["DEP_DELAY"].shift(k).where(ok),
        dist=g["DISTANCE"].shift(k).where(ok),
    )
    prev[k]["departed"] = prev[k]["actual"] <= df["T_utc"]   # had it left by OUR checkpoint?

p1 = prev[1]
has_prev = p1["sched"].notna()
df["has_prev_leg"] = has_prev.astype(int)
df["sched_gap_prev_leg"] = df["sched_utc"] - p1["sched"]                  # known in advance
# rough block time of the previous leg (~480 mph + 30 min taxi) -> planned slack
est_block = p1["dist"] / 8.0 + 30
df["planned_turn_slack"] = df["sched_gap_prev_leg"] - est_block          # known in advance

df["prev_leg_departed_by_T"] = np.where(has_prev, p1["departed"].astype(float), np.nan)
df["prev_leg_delay_obs"] = p1["delay"].where(p1["departed"])            # NaN if not yet observed
not_dep_lb = (df["T_utc"] - p1["sched"]).clip(lower=0)
# what we know at T: real delay if it left, otherwise "at least this late"
df["prev_leg_delay_lb"] = np.where(p1["departed"], p1["delay"], not_dep_lb).astype(float)
df.loc[~has_prev, "prev_leg_delay_lb"] = np.nan
df["prev_leg_overdue"] = (has_prev & ~p1["departed"] & (p1["sched"] <= df["T_utc"])).astype(int)
# delay we expect to be pushed onto this flight = known lateness - planned slack
df["propagated_delay_est"] = df["prev_leg_delay_lb"] - df["planned_turn_slack"].clip(lower=0)

# most recent leg (within 3 back) whose departure was observed by T
last_known = pd.Series(np.nan, index=df.index); legs_back = pd.Series(np.nan, index=df.index)
for k in (3, 2, 1):  # overwrite so the nearest observed leg wins
    m = prev[k]["departed"]
    last_known[m] = prev[k]["delay"][m]; legs_back[m] = k
df["last_observed_leg_delay"] = last_known
df["legs_since_observed"] = legs_back
df["tail_late_legs_last3_obs"] = sum(
    (prev[k]["departed"] & (prev[k]["delay"] >= LATE_THRESHOLD)).astype(int) for k in (1, 2, 3))

# ----------------------------------------------------------------------
# 4. AIRPORT CONGESTION
# 4a. scheduled load (known in advance from the timetable)
# ----------------------------------------------------------------------
df["origin_hourly_flights"] = df.groupby(["FL_DATE", "ORIGIN_CITY_MARKET_ID", "dep_hour"])["FL_DATE"].transform("size")
df["origin_daily_flights"] = df.groupby(["FL_DATE", "ORIGIN_CITY_MARKET_ID"])["FL_DATE"].transform("size")
df["dest_daily_flights"] = df.groupby(["FL_DATE", "DEST_CITY_MARKET_ID"])["FL_DATE"].transform("size")
df["carrier_origin_daily_flights"] = df.groupby(["FL_DATE", "ORIGIN_CITY_MARKET_ID", "OP_UNIQUE_CARRIER"])["FL_DATE"].transform("size")
# share of the day's departures in this hour (peakiness)
df["origin_hour_share"] = df["origin_hourly_flights"] / df["origin_daily_flights"]

# ----------------------------------------------------------------------
# 4b. observed congestion: what other flights did in the 2 h before T
# ----------------------------------------------------------------------
markets = pd.Index(pd.unique(pd.concat([df["ORIGIN_CITY_MARKET_ID"], df["DEST_CITY_MARKET_ID"]])))
o_code = markets.get_indexer(df["ORIGIN_CITY_MARKET_ID"]).astype(float)
d_code = markets.get_indexer(df["DEST_CITY_MARKET_ID"]).astype(float)
c_code = df["OP_UNIQUE_CARRIER"].astype("category").cat.codes.to_numpy().astype(float)
act = df["actual_utc"].to_numpy(); sch = df["sched_utc"].to_numpy(); T = df["T_utc"].to_numpy()
delay = df["DEP_DELAY"].to_numpy().astype(float); late = df["LATE"].to_numpy().astype(float)


def recent(prefix, ev_key, q_key):
    n, (s_delay, s_late) = window_stats(ev_key, act, [delay, late], q_key, T, WINDOW_MIN)
    with np.errstate(invalid="ignore", divide="ignore"):
        df[f"{prefix}_recent_n_dep"] = n
        df[f"{prefix}_recent_mean_delay"] = np.where(n > 0, s_delay / n, np.nan)
        df[f"{prefix}_recent_late_rate"] = np.where(n > 0, s_late / n, np.nan)


recent("origin", o_code, o_code)                 # departures from our airport
recent("dest", o_code, d_code)                   # departures FROM our destination (its weather/ops)
recent("carrier", c_code, c_code)                # our carrier, nationwide
recent("national", np.zeros_like(o_code), np.zeros_like(o_code))
df = df.drop(columns=["carrier_recent_n_dep", "national_recent_n_dep"])
# backlog at origin: flights scheduled to leave by T that have not left by T
df["origin_backlog_at_T"] = np.clip(count_upto(o_code, sch, o_code, T) - count_upto(o_code, act, o_code, T), 0, None)

# ----------------------------------------------------------------------
# 5. HISTORICAL RATES: smoothed LATE rate on earlier DEV days only
# ----------------------------------------------------------------------
dev_end_idx = (pd.Timestamp(DEV_END) - DAY0).days
for name, key in [("origin", "ORIGIN_CITY_MARKET_ID"), ("dest", "DEST_CITY_MARKET_ID"),
                  ("carrier", "OP_UNIQUE_CARRIER"), ("route", "route"),
                  ("tail", "TAIL_NUM"), ("hour", "dep_hour")]:
    df[f"{name}_hist_late_rate"], cnt = prior_day_rate(df, key, dev_end_idx)
    if name in ("route", "tail"):
        df[f"{name}_hist_n"] = cnt

# ----------------------------------------------------------------------
# 6. feature lists, audit, save + freeze
# ----------------------------------------------------------------------
GROUPS = {
    "calendar": ["day_of_week", "day_of_month", "is_weekend", "dep_hour", "dep_minute_of_day",
                 "hour_sin", "hour_cos", "days_to_holiday", "is_holiday"],
    "schedule_static": ["OP_UNIQUE_CARRIER", "MKT_UNIQUE_CARRIER", "is_codeshare",
                        "ORIGIN_CITY_MARKET_ID", "DEST_CITY_MARKET_ID", "ORIGIN_STATE_ABR",
                        "DEST_STATE_ABR", "DISTANCE", "dep_period"],
    "aircraft_schedule": ["tail_leg_number", "tail_legs_today", "has_prev_leg",
                          "sched_gap_prev_leg", "planned_turn_slack"],
    "aircraft_observed": ["prev_leg_departed_by_T", "prev_leg_delay_obs", "prev_leg_delay_lb",
                          "prev_leg_overdue", "propagated_delay_est", "last_observed_leg_delay",
                          "legs_since_observed", "tail_late_legs_last3_obs"],
    "congestion_scheduled": ["origin_hourly_flights", "origin_daily_flights", "dest_daily_flights",
                             "carrier_origin_daily_flights", "origin_hour_share"],
    "congestion_observed": ["origin_recent_n_dep", "origin_recent_mean_delay", "origin_recent_late_rate",
                            "dest_recent_n_dep", "dest_recent_mean_delay", "dest_recent_late_rate",
                            "carrier_recent_mean_delay", "carrier_recent_late_rate",
                            "national_recent_mean_delay", "national_recent_late_rate",
                            "origin_backlog_at_T"],
    "historical_rates": ["origin_hist_late_rate", "dest_hist_late_rate", "carrier_hist_late_rate",
                         "route_hist_late_rate", "route_hist_n", "tail_hist_late_rate",
                         "tail_hist_n", "hour_hist_late_rate"],
}
CATEGORICAL = ["OP_UNIQUE_CARRIER", "MKT_UNIQUE_CARRIER", "ORIGIN_CITY_MARKET_ID",
               "DEST_CITY_MARKET_ID", "ORIGIN_STATE_ABR", "DEST_STATE_ABR", "dep_period"]
FEATURES = [f for g_ in GROUPS.values() for f in g_]
ID_COLS = ["FL_DATE", "TAIL_NUM", "OP_CARRIER_FL_NUM", "ORIGIN_CITY_NAME", "DEST_CITY_NAME",
           "CRS_DEP_TIME", "route"]
TARGET = "LATE"

out = df[ID_COLS + FEATURES + ["DEP_DELAY", TARGET]].sort_values(["FL_DATE", "CRS_DEP_TIME", "TAIL_NUM"])
out.to_csv(FEATURES_CSV, index=False, float_format="%.5g")

meta = {"groups": GROUPS, "categorical": CATEGORICAL, "features": FEATURES,
        "id_cols": ID_COLS, "target": TARGET, "leak_cols_never_use": ["DEP_DELAY"],
        "checkpoint_min": CHECKPOINT_MIN, "window_min": WINDOW_MIN}
(HERE / "feature_groups.json").write_text(json.dumps(meta, indent=2))

md5 = hashlib.md5(FEATURES_CSV.read_bytes()).hexdigest()
info = {"file": FEATURES_CSV.name, "rows": len(out), "n_features": len(FEATURES),
        "late_rate": round(out[TARGET].mean(), 4), "md5": md5, "version": "v1"}
(FEATURES_CSV.parent / "features_v1_info.json").write_text(json.dumps(info, indent=2))

# ---- column audit rows for the new features (P1 merges into the main audit)
AUDIT = {
    "calendar": ("known-in-advance", "derived from the scheduled date/time only"),
    "schedule_static": ("known-in-advance / static", "timetable and identity of carrier, airports, route"),
    "aircraft_schedule": ("known-in-advance", "aircraft rotation is in the published schedule"),
    "aircraft_observed": ("observed-by-T", "uses earlier legs only if they had departed by T (else 'still on ground')"),
    "congestion_scheduled": ("known-in-advance", "counts of scheduled flights, no outcomes"),
    "congestion_observed": ("observed-by-T", "other flights whose actual departure was in (T-2h, T]"),
    "historical_rates": ("observed-by-T", "LATE rate on strictly earlier days, DEV days only (no test data)"),
}
lines = ["| Feature | Group | Type | Justification |", "|---|---|---|---|"]
for grp, feats in GROUPS.items():
    for f in feats:
        lines.append(f"| {f} | {grp} | {AUDIT[grp][0]} | {AUDIT[grp][1]} |")
lines += ["| DEP_DELAY | label source | outcome-like | used ONLY to make LATE; never a feature |",
          "| LATE | target | outcome | DEP_DELAY >= 15 min |"]
(HERE / "feature_audit.md").write_text("\n".join(lines) + "\n")

print(f"Saved {FEATURES_CSV.name}: {out.shape}, {len(FEATURES)} features, LATE rate {info['late_rate']}")
print("md5", md5)
print("\nMissing share of engineered features (NaN is meaningful, e.g. first leg of the day):")
print(out[FEATURES].isna().mean().loc[lambda s: s > 0].round(3).to_string())
