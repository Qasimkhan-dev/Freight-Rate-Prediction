"""Shared data preparation used by every script."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import cleaning as C, features as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

STAGE1_COLS = F.BASE + F.WEIGHT + F.MARKET + F.QUOTE


def prepare_train_validation():
    """Load + clean train and validation, build features, flag label outliers."""
    train = C.load_frame(DATA / "train_test.csv")
    valid = C.load_frame(DATA / "validation.csv")
    daily = C.daily_market_table(train, valid)      # features only, no labels
    train_c = C.clean_features(train, daily)
    valid_c = C.clean_features(valid, daily)
    X_train = F.build_features(train_c)
    X_valid = F.build_features(valid_c)
    y = train_c["posted_rate"]
    outlier, resid = C.flag_label_outliers(X_train[STAGE1_COLS], y)
    return dict(train=train_c, valid=valid_c, X_train=X_train, X_valid=X_valid,
                y=y, outlier=outlier, resid=resid, daily=daily)


def december_features(dec: pd.DataFrame, train_c: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Features for the 31 fixed December chart rows.

    The file only carries lane/equipment/weight/date, so we attach:
    * city coordinates from the training data (same lookup for any city),
    * the day-level market_index / quote_signal medians for each December date
      taken from validation.csv (features only; no labels involved).
    Load-level deviations are set to zero (a "typical" load that day).
    """
    pick = train_c.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    deli = train_c.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    d = dec.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.join(pick, on="pickup").join(deli, on="delivery")
    d = d.join(daily, on="date")
    if d[["mi_day", "qs_day", "pickup_lat", "delivery_lat"]].isna().any().any():
        raise ValueError("December rows could not be fully featurised")
    d["market_index"] = d["mi_day"]
    d["quote_signal"] = d["qs_day"]
    d["weight_missing"] = 0
    d["distance"] = d["distance"].astype(float)
    d["weight"] = d["weight"].astype(float)
    return F.build_features(d)
