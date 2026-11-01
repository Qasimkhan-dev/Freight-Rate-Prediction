"""Forward-chaining validation, baselines, ablation and robustness checks.

    python scripts/run_cv.py            # writes reports/cv_*.csv

Validation design
-----------------
The final test (validation.csv) is the *two months after* the last training day
(Nov-Dec 2025). Each fold therefore trains on every labelled day up to a cut-off
and predicts the next two calendar months. Folds never see the future, and
weekly/monthly neighbours of a test day are never in the training set.

Primary folds: test windows start May, Jun, Jul, Aug, Sep (>= 4 months of
history, comparable to the 10 months available for the real run).
Stress folds: windows starting Mar and Apr (only 2-3 months of history).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from freight_rate import features as F, pipeline as PL
from freight_rate.models import FreightRateModel, MEMBERS, GBM_FAST

OUT = PL.ROOT / "reports"; OUT.mkdir(exist_ok=True)
PRIMARY = [5, 6, 7, 8, 9]
STRESS = [3, 4]

def window(month):
    a = pd.Timestamp(2025, month, 1); b = a + pd.offsets.MonthEnd(2)
    return a - pd.Timedelta(days=1), a, b

def metrics(y, p):
    e = p - y
    return dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)),
                MAPE=100 * np.mean(np.abs(e) / y), MedAPE=100 * np.median(np.abs(e) / y))

def run_fold(D, month, fit_predict):
    tend, a, b = window(month)
    trm = (D["train"].date <= tend) & ~D["outlier"]
    tem = (D["train"].date >= a) & (D["train"].date <= b)
    p = fit_predict(D["X_train"].loc[trm], D["y"][trm], D["X_train"].loc[tem], D["train"].loc[trm], D["train"].loc[tem])
    y = D["y"][tem]; fl = D["outlier"][tem]
    clean = metrics(y[~fl].values, p[~fl.values]); raw = metrics(y.values, p)
    return clean, raw, pd.DataFrame({"load_id": D["train"].loc[tem, "load_id"].values, "date": D["train"].loc[tem, "date"].values,
                                     "y": y.values, "pred": p, "outlier": fl.values, "fold": month})

def summarise(name, per_fold):
    row = {"model": name}
    for kind, idx in (("clean", 0), ("raw", 1)):
        for m in ("MAE", "RMSE", "MAPE", "MedAPE"):
            row[f"{kind}_{m}"] = np.mean([f[idx][m] for f in per_fold])
    return row

def main():
    t0 = time.time()
    D = PL.prepare_train_validation()
    print(f"outliers flagged: {int(D['outlier'].sum())} | max |res| kept {D['resid'][~D['outlier']].abs().max():.2f} "
          f"| min |res| flagged {D['resid'][D['outlier']].abs().min():.2f}", flush=True)
    rows, preds = [], []

    # ---------- baselines ----------
    def naive(Xtr, ytr, Xte, dtr, dte):
        rpm = (ytr / Xtr.distance).groupby(Xtr.equip).median()
        return (Xte.equip.map(rpm) * Xte.distance).values
    def loglin(Xtr, ytr, Xte, dtr, dte):
        f = lambda X: np.c_[np.ones(len(X)), X.log_distance, X.equip == 1, X.equip == 2].astype(float)
        b = np.linalg.lstsq(f(Xtr), np.log(ytr.values), rcond=None)[0]
        return np.exp(f(Xte) @ b)
    for name, fn in (("Baseline 1: median $/mile x distance (by equipment)", naive),
                     ("Baseline 2: log-linear (distance, equipment)", loglin)):
        rows.append(summarise(name, [run_fold(D, m, fn)[:2] for m in PRIMARY])); print(rows[-1]["model"], round(rows[-1]["clean_MAPE"], 2), flush=True)

    # ---------- feature ablation (single plain LightGBM on log-rate) ----------
    sets = [("GBM: distance + equipment", F.BASE),
            ("GBM: + weight", F.BASE + F.WEIGHT),
            ("GBM: + market_index (day level + deviation)", F.BASE + F.WEIGHT + F.MARKET),
            ("GBM: + quote_signal (raw, day level, deviation)", F.BASE + F.WEIGHT + F.MARKET + F.QUOTE),
            ("GBM: + calendar (dow, day-of-month)", F.BASE + F.WEIGHT + F.MARKET + F.QUOTE + F.CAL),
            ("GBM: + geometry (coords, haversine, circuity)", F.BASE + F.WEIGHT + F.MARKET + F.QUOTE + F.GEO),
            ("GBM: + time index t", F.BASE + F.WEIGHT + F.MARKET + F.QUOTE + F.GEO + F.TREND)]
    for name, cols in sets:
        def fn(Xtr, ytr, Xte, dtr, dte, cols=cols):
            return np.exp(lgb.LGBMRegressor(**dict(GBM_FAST, random_state=0)).fit(Xtr[cols], np.log(ytr)).predict(Xte[cols]))
        rows.append(summarise(name, [run_fold(D, m, fn)[:2] for m in PRIMARY])); print(name, round(rows[-1]["clean_MAPE"], 2), flush=True)

    # ---------- final model ----------
    def final(Xtr, ytr, Xte, dtr, dte, members=None):
        return FreightRateModel(members=members, seeds=1).fit(Xtr, ytr).predict(Xte)
    per = [run_fold(D, m, final) for m in PRIMARY]
    rows.append(summarise("FINAL: linear base + LightGBM residual ensemble", per))
    print("FINAL", round(rows[-1]["clean_MAPE"], 2), flush=True)
    pf = pd.DataFrame([{"fold_start_month": m, **{f"clean_{k}": v for k, v in c.items()}, **{f"raw_{k}": v for k, v in r.items()}}
                       for m, (c, r, _) in zip(PRIMARY, per)])
    pf.to_csv(OUT / "cv_final_per_fold.csv", index=False)
    preds.append(pd.concat([p for _, _, p in per]))
    pd.concat(preds).to_csv(OUT / "cv_predictions.csv", index=False)

    # ---------- stress folds (short history) ----------
    sf = [run_fold(D, m, final) for m in STRESS]
    pd.DataFrame([{"fold_start_month": m, **{f"clean_{k}": v for k, v in c.items()}} for m, (c, _, _) in zip(STRESS, sf)]).to_csv(OUT / "cv_stress_folds.csv", index=False)
    print("stress folds clean MAPE:", [round(c["MAPE"], 2) for c, _, _ in sf], flush=True)

    # ---------- random K-fold (WRONG for this problem) vs forward chaining ----------
    X, y, keep = D["X_train"], D["y"], ~D["outlier"]
    Xk, yk = X[keep], y[keep]; mape = []
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xk):
        p = FreightRateModel(seeds=1).fit(Xk.iloc[a], yk.iloc[a]).predict(Xk.iloc[b])
        mape.append(100 * np.mean(np.abs(p - yk.iloc[b].values) / yk.iloc[b].values))
    pd.DataFrame([{"scheme": "Random 5-fold (leaks neighbouring days)", "clean_MAPE": np.mean(mape)},
                  {"scheme": "Forward chaining, 2-month horizon (used)", "clean_MAPE": rows[-1]["clean_MAPE"]}]).to_csv(OUT / "cv_random_vs_forward.csv", index=False)
    print("random KFold MAPE", round(np.mean(mape), 2), flush=True)

    # ---------- cold-start: remove 8 cities from training ----------
    cities = sorted(set(D["train"].pickup) | set(D["train"].delivery)); rng = np.random.default_rng(7)
    no_geo = {k: ([c for c in cols if c not in F.GEO], p) for k, (cols, p) in MEMBERS.items()}
    no_ll = {k: ([c for c in cols if c not in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]], p) for k, (cols, p) in MEMBERS.items()}
    res = {"with coordinates (final)": [], "without raw lat/lon": [], "without any geometry": []}
    for rep in range(3):
        hold = set(rng.choice(cities, 8, replace=False)); tr_df = D["train"]
        cold = (tr_df.pickup.isin(hold) | tr_df.delivery.isin(hold))
        a, b = pd.Timestamp(2025, 9, 1), pd.Timestamp(2025, 10, 31)
        trm = (tr_df.date < a) & ~D["outlier"] & ~cold; tem = (tr_df.date >= a) & (tr_df.date <= b) & ~D["outlier"]
        for k, mem in zip(res, (None, no_ll, no_geo)):
            p = FreightRateModel(members=mem).fit(X.loc[trm], y[trm]).predict(X.loc[tem])
            e = 100 * np.abs(p - y[tem].values) / y[tem].values
            res[k].append((e.mean(), e[cold[tem].values].mean()))
    pd.DataFrame([{"variant": k, "MAPE_all_rows": np.mean([a for a, _ in v]), "MAPE_rows_with_unseen_cities": np.mean([c for _, c in v])} for k, v in res.items()]).to_csv(OUT / "cv_cold_start.csv", index=False)

    pd.DataFrame(rows).to_csv(OUT / "cv_results.csv", index=False)
    print(pd.DataFrame(rows)[["model", "clean_MAE", "clean_MAPE", "raw_MAE", "raw_RMSE"]].round(2).to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
