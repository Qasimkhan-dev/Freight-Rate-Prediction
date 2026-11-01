"""Feature engineering. City names are deliberately NOT used (cold-start)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIP = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}


def haversine(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# feature groups, used by the ablation study
BASE = ["distance", "log_distance", "equip"]
WEIGHT = ["weight", "weight_missing"]
MARKET = ["market_index", "mi_dev"]
QUOTE = ["quote_signal", "qs_day", "qs_dev"]
QUOTE_DAY_ONLY = ["qs_day"]
CAL = ["dow", "dom"]
GEO = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "hav", "circuity",
       "dlat", "dlon"]
TREND = ["t"]  # days since 2025-01-01 (extrapolation risk; ablation only)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"]
    X["log_distance"] = np.log(df["distance"])
    X["equip"] = df["equipment"].map(EQUIP).astype(int)
    X["weight"] = df["weight"]
    X["weight_missing"] = df["weight_missing"]
    X["market_index"] = df["mi_day"]           # day-level market level
    X["mi_dev"] = df["market_index"] - df["mi_day"]   # load-level deviation
    X["quote_signal"] = df["quote_signal"]
    X["qs_day"] = df["qs_day"]                  # day-level quote level
    X["qs_dev"] = df["quote_signal"] - df["qs_day"]
    X["dow"] = df["date"].dt.dayofweek
    X["dom"] = df["date"].dt.day
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    X["hav"] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["circuity"] = df["distance"] / X["hav"].clip(lower=1)
    X["dlat"] = df["delivery_lat"] - df["pickup_lat"]
    X["dlon"] = df["delivery_lon"] - df["pickup_lon"]
    X["t"] = (df["date"] - pd.Timestamp("2025-01-01")).dt.days
    return X
