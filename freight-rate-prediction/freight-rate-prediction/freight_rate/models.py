"""Final model: ridge linear base on log-rate + LightGBM residual ensemble.

Why this structure (details in the report):
* The linear base captures the smooth, global dependence on distance, equipment,
  weight and market level, so the model extrapolates sensibly when a feature
  drifts outside the range of the training window.
* LightGBM models the remaining interactions, geography and slow time drift.
  The time index ``t`` lets it carry the most recent rate level forward.
* Three differently-configured members are averaged in log space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

from . import features as F

GBM_FAST = dict(objective="l2", n_estimators=350, learning_rate=0.08, num_leaves=31,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, verbose=-1, n_jobs=1)
GBM_REG = dict(GBM_FAST, num_leaves=15, n_estimators=500, learning_rate=0.05,
               min_child_samples=80, reg_lambda=5.0)

ALL_COLS = F.BASE + F.WEIGHT + F.MARKET + F.QUOTE + F.GEO + F.TREND
DAY_ONLY_COLS = F.BASE + F.WEIGHT + F.MARKET + F.QUOTE_DAY_ONLY + F.GEO + F.TREND

MEMBERS = {
    "all_features": (ALL_COLS, GBM_FAST),
    "quote_day_only": (DAY_ONLY_COLS, GBM_FAST),
    "all_features_regularised": (ALL_COLS, GBM_REG),
}


def linear_design(X: pd.DataFrame, weight_median: float) -> pd.DataFrame:
    D = pd.DataFrame(index=X.index)
    D["ld"] = X.log_distance
    D["ld2"] = X.log_distance ** 2
    D["ld3"] = X.log_distance ** 3
    D["reefer"] = (X.equip == 1).astype(float)
    D["flatbed"] = (X.equip == 2).astype(float)
    D["w"] = X.weight.fillna(weight_median) / 1e4
    D["w_missing"] = X.weight_missing
    D["mi"] = X.market_index
    D["mi2"] = X.market_index ** 2
    D["qs"] = X.quote_signal
    D["qs2"] = (X.quote_signal - 2.05) ** 2
    D["qs_day"] = X.qs_day
    return D


class FreightRateModel:
    def __init__(self, members: dict | None = None, seeds: int = 1, ridge_alpha: float = 1.0):
        self.members = members or MEMBERS
        self.seeds = seeds
        self.ridge_alpha = ridge_alpha

    def fit(self, X: pd.DataFrame, rate: pd.Series) -> "FreightRateModel":
        y = np.log(rate.values)
        self.w_median_ = float(X.weight.median())
        D = linear_design(X, self.w_median_)
        self.mu_, self.sd_ = D.mean(), D.std()
        self.ridge_ = Ridge(alpha=self.ridge_alpha).fit((D - self.mu_) / self.sd_, y)
        base = self.ridge_.predict((D - self.mu_) / self.sd_)
        self.gbms_ = {}
        for name, (cols, params) in self.members.items():
            self.gbms_[name] = [
                lgb.LGBMRegressor(**dict(params, random_state=s)).fit(X[cols], y - base)
                for s in range(self.seeds)
            ]
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        D = linear_design(X, self.w_median_)
        base = self.ridge_.predict((D - self.mu_) / self.sd_)
        member_preds = []
        for name, (cols, _) in self.members.items():
            member_preds.append(np.mean([m.predict(X[cols]) for m in self.gbms_[name]], axis=0))
        return base + np.mean(member_preds, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X))
