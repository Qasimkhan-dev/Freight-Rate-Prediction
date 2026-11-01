"""Data cleaning helpers.

Data-quality issues found during exploration (see reports/ for details):

1. ``weight`` has sign-flipped values (negative, magnitude still in 5,000-47,500).
   -> take the absolute value.
2. ``weight`` (~0.6%) and ``market_index`` (~0.8%) have missing values.
   -> ``weight`` stays NaN (+ missing flag, LightGBM handles NaN natively);
      ``market_index`` is filled with the same-day median of the other loads.
3. ~1.4% of training labels are corrupted: posted_rate is either ~2.4-5x too
   high or ~0.2-0.4x too low relative to what the load features imply.
   -> flagged with an out-of-fold model and excluded from *training*.
4. 8 cities appear only in validation (12% of rows) -> city IDs are never used
   as features; only distance / coordinates are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_FEATURES = ["distance", "weight", "market_index", "quote_signal"]


def load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def daily_market_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-date median market_index / quote_signal using *features only*.

    No labels are involved, so it is safe to build from train + validation
    rows. Used to (a) fill missing market_index and (b) supply market inputs
    for the fixed December chart rows, which ship without them.
    """
    allrows = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    table = allrows.groupby("date").agg(
        mi_day=("market_index", "median"), qs_day=("quote_signal", "median")
    )
    return table


def clean_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out = out.merge(daily, left_on="date", right_index=True, how="left")
    out["market_index"] = out["market_index"].fillna(out["mi_day"])
    out["quote_signal"] = out["quote_signal"].fillna(out["qs_day"])
    return out


def flag_label_outliers(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.4, folds: int = 5, seed: int = 0
) -> tuple[pd.Series, pd.Series]:
    """Flag rows whose log(rate) is far from an out-of-fold prediction.

    Clean rows have |log residual| well under 0.2 whereas corrupted rows are
    >= 0.88 (x2.4 or x0.41), so the threshold sits in a wide empty gap.
    """
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    ly = np.log(y.values)
    oof = np.zeros(len(ly))
    kf = KFold(folds, shuffle=True, random_state=seed)
    params = dict(objective="l1", n_estimators=300, learning_rate=0.08,
                  num_leaves=31, min_child_samples=30, verbose=-1, n_jobs=1)
    for tr_idx, te_idx in kf.split(X):
        m = lgb.LGBMRegressor(**params).fit(X.iloc[tr_idx], ly[tr_idx])
        oof[te_idx] = m.predict(X.iloc[te_idx])
    res = ly - oof
    return pd.Series(np.abs(res) > threshold, index=X.index, name="label_outlier"), pd.Series(res, index=X.index)
