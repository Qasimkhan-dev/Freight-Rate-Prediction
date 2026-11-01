"""Train the final model on all clean labelled data and write the deliverables.

    python scripts/train_predict.py

Outputs
  validation_predictions.csv        (load_id,predicted_rate - 12,000 rows)
  data/december_chart_inputs.csv    (completed in place; original kept as *.orig)
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from freight_rate import pipeline as PL
from freight_rate.models import FreightRateModel

ROOT = PL.ROOT

def main():
    D = PL.prepare_train_validation()
    keep = ~D["outlier"]
    print(f"training rows: {int(keep.sum())} of {len(keep)} (excluded {int((~keep).sum())} corrupted labels)")
    model = FreightRateModel(seeds=3).fit(D["X_train"][keep], D["y"][keep])

    # ---- validation predictions ----
    pred = model.predict(D["X_valid"])
    template = pd.read_csv(PL.DATA / "validation_predictions_template.csv")
    out = pd.DataFrame({"load_id": D["valid"]["load_id"].values, "predicted_rate": np.round(pred, 2)})
    assert set(out.load_id) == set(template.load_id) and len(out) == len(template)
    out = template[["load_id"]].merge(out, on="load_id", how="left")      # keep template order
    assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print("wrote validation_predictions.csv", out.shape, "| rate summary:", out.predicted_rate.describe().round(0).to_dict())

    # ---- December chart rows ----
    dec_path = PL.DATA / "december_chart_inputs.csv"
    orig = dec_path.with_suffix(".csv.orig")
    if not orig.exists():
        shutil.copy(dec_path, orig)                     # keep the untouched template
    dec = pd.read_csv(orig)
    Xd = PL.december_features(dec, D["train"], D["daily"])
    dec["predicted_rate"] = np.round(model.predict(Xd), 2)
    dec.to_csv(dec_path, index=False)
    print("wrote", dec_path.name); print(dec[["date", "predicted_rate"]].to_string(index=False))

if __name__ == "__main__":
    main()
