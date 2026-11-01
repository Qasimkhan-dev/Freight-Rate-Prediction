# Freight Rate Prediction

Predicts `posted_rate` for 12,000 loads (Nov-Dec 2025) from 48,000 labelled loads (Jan-Oct 2025).

**Model:** ridge linear base on log(rate) + a 3-member LightGBM residual ensemble.
**Validation:** forward chaining, 2-month horizon (never random k-fold). Forward-fold MAPE 1.87% / MAE $44 on clean rows.
Full write-up: `reports/Freight_Rate_Report.docx` (PDF copy alongside).

## Setup

```bash
python -m pip install -r requirements.txt
```

Put the provided files in `data/` as `train_test.csv`, `validation.csv`,
`validation_predictions_template.csv`, `december_chart_inputs.csv`
(the assessment uses underscores; rename if your copies use hyphens).

## Run

```bash
python scripts/train_predict.py      # trains, writes validation_predictions.csv, fills data/december_chart_inputs.csv (~2 min)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python scripts/run_cv.py             # optional: forward-chaining validation, baselines, ablation (~2-3 min)
python scripts/make_figures.py       # optional: report figures
node reports/build_report.js         # optional: rebuild the DOCX (needs the `docx` npm package)
```

`train_predict.py` keeps an untouched copy of the December template as `data/december_chart_inputs.csv.orig` and re-reads it on every run.
The scorer writes `scorer_results/candidate_december.png`.

## Layout

```
freight_rate/cleaning.py   weight sign fix, missing values, label-outlier detection
freight_rate/features.py   feature engineering (city names deliberately not used)
freight_rate/models.py     FreightRateModel (linear base + LightGBM ensemble)
freight_rate/pipeline.py   shared data prep, December feature construction
scripts/                   train_predict.py, run_cv.py, make_figures.py
reports/                   report, figures, validation CSVs
```

## Key decisions

- 677 corrupted labels (rate 2.4-5x too high or 0.2-0.4x too low) are excluded from training and reported separately in evaluation.
- Negative weights are sign flips -> `abs()`. Missing `market_index` -> same-day median.
- 8 cities appear only in validation, so only distance/coordinates are used, not city names.
- December rows have no `market_index`/`quote_signal`; the per-date medians from `validation.csv` features are used (no labels involved).
- Short-history folds (2-3 months of training) fail badly (8-16% MAPE); the final run has 10 months and in-range features. See report section 3.5.
