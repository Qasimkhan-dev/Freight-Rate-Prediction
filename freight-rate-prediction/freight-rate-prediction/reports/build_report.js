// Builds reports/Freight_Rate_Report.docx  (run:  node reports/build_report.js)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, AlignmentType, HeadingLevel, LevelFormat, PageBreak,
} = require("docx");

const R = __dirname;
const png = (p) => { const b = fs.readFileSync(p); return { data: b, w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
const TEAL = "064A56", LIGHT = "E8F0F1", GREY = "455A60";
const font = "Arial";

const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...opts,
  children: Array.isArray(text) ? text : [new TextRun({ text, font, size: 21 })] });
const r = (text, o = {}) => new TextRun({ text, font, size: 21, ...o });
const b = (text) => r(text, { bold: true });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 260, after: 120 },
  children: [new TextRun({ text: t, font, size: 28, bold: true, color: TEAL })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 80 },
  children: [new TextRun({ text: t, font, size: 23, bold: true, color: TEAL })] });
const bullet = (children) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 70, line: 264 },
  children: Array.isArray(children) ? children : [r(children)] });
const caption = (t) => new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: t, font, size: 18, italics: true, color: GREY })] });
const img = (file, widthPx) => { const i = png(file); return new Paragraph({ keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 60, after: 40 },
  children: [new ImageRun({ type: "png", data: i.data, transformation: { width: widthPx, height: Math.round(widthPx * i.h / i.w) },
    altText: { title: path.basename(file), description: path.basename(file), name: path.basename(file) } })] }); };

const border = { style: BorderStyle.SINGLE, size: 4, color: "C5D3D6" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(widths, rows, headerFill = TEAL) {
  const total = widths.reduce((a, c) => a + c, 0);
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((row, ri) => new TableRow({ tableHeader: ri === 0, children: row.map((cell, ci) => new TableCell({
      width: { size: widths[ci], type: WidthType.DXA }, borders,
      shading: { fill: ri === 0 ? headerFill : (cell.hl ? LIGHT : "FFFFFF"), type: ShadingType.CLEAR, color: "auto" },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ keepNext: ri < rows.length - 1, children: [new TextRun({ text: String(cell.t ?? cell), font, size: 18,
        bold: ri === 0 || !!cell.b, color: ri === 0 ? "FFFFFF" : "000000" })] })] })) })) });
}

const content = [];
content.push(new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: "Freight Rate Prediction", font, size: 44, bold: true, color: TEAL })] }));
content.push(new Paragraph({ spacing: { after: 200 }, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: TEAL, space: 6 } },
  children: [new TextRun({ text: "Validation approach, data-quality handling and the fixed December forecast", font, size: 24, color: GREY })] }));

// ---- 1. Summary
content.push(h1("1. Summary"));
content.push(bullet([b("Task. "), r("Predict posted_rate for 12,000 loads (Nov–Dec 2025) from 48,000 labelled loads (Jan–Oct 2025).")]));
content.push(bullet([b("Model. "), r("Ridge linear base on log(rate) plus a three-member LightGBM residual ensemble. Trained on 47,323 rows after excluding 677 corrupted labels.")]));
content.push(bullet([b("Validation. "), r("Forward chaining with a two-month horizon (train on everything before the window, predict the next two months), which is the same shape as the real task. Random k-fold was rejected because it leaks neighbouring days.")]));
content.push(bullet([b("Result. "), r("1.87% MAPE and $44 MAE on the five forward folds, against 4.39% / $104 for a log-linear baseline and 8.33% / $168 for a median $/mile baseline (clean test rows).")]));
content.push(bullet([b("Caveat. "), r("The official metric is computed by Spotter and is not visible to me. All figures here are my own forward-validation numbers.")]));

// ---- 2. Data quality
content.push(h1("2. Data-quality findings and handling"));
content.push(table([2050, 3910, 3400], [
  ["Issue", "Evidence", "Handling"],
  ["Corrupted labels", "About 670 rates are 2.4–5× too high or 0.2–0.4× too low for what the load implies (677 flagged by an out-of-fold model). Kept rows stay within ±0.20 log; flagged rows are all beyond ±0.80.", "Excluded from training. Reported separately in evaluation (see 3.3)."],
  ["Negative weights", "292 train and 145 validation rows. After taking the absolute value the range is 5,000–47,500, so these are sign flips.", "abs(weight)."],
  ["Missing values", "weight: 300 train / 165 validation. market_index: 374 / 249. No day is entirely missing.", "weight left as NaN plus a flag (LightGBM handles it). market_index filled with that day’s median."],
  ["Unseen cities", "8 cities appear only in validation and touch 12.1% of validation rows.", "City names are never used as features; only distance and coordinates."],
  ["Distance floor", "Distance is floored at 70 miles (e.g. New Orleans–Shreveport is 7 miles apart by coordinates).", "Kept; geometry features give the model both views."],
  ["Drifting market_index", "Validation mean 0.93 vs training 1.08; strong weekly cycle (Thursday high, Sunday low).", "Day-level median used as the market level plus the load’s deviation from it."],
]));
content.push(img(path.join(R, "fig_outliers.png"), 540));
content.push(caption("Figure 1. Out-of-fold log residuals. The orange cluster is separated from the rest by a wide empty gap, so the threshold is not a judgement call."));

// ---- 3. Validation
content.push(h1("3. Train / test split and validation approach"));
content.push(h2("3.1 Why not a random split"));
content.push(p("Rates depend on a slowly moving market level. The average residual of a model that ignores time has a lag-1 autocorrelation of 0.82 across days, so test rows from a random split have training rows from the same week sitting right next to them. Measured directly, random 5-fold gave 1.17% MAPE while forward chaining gave 1.87%. A random split would have understated the error by about 37%."));
content.push(h2("3.2 Forward-chaining design"));
content.push(p("The real task is to train on January–October and predict November–December. Each fold repeats that: train on all labelled days before a cut-off, predict the following two months. The primary folds start in May, June, July, August and September, so every fold has at least four months of history."));
content.push(img(path.join(R, "fig_folds.png"), 540));
content.push(caption("Figure 2. Five forward folds plus the final fit. A test window never overlaps its training data."));
content.push(h2("3.3 How the corrupted labels are treated in evaluation"));
content.push(p("Corrupted rows are excluded from training in every fold. Test metrics are reported two ways: on clean test rows (the model’s real accuracy) and on all test rows including corrupted ones. Corrupted labels dominate the raw RMSE (roughly $630–$670 for every model), so differences between models only show up in the clean numbers and in raw MAE and MAPE."));
content.push(h2("3.4 Results"));
content.push(table([4560, 1200, 1200, 1200, 1200], [
  ["Model (mean of 5 forward folds)", "MAE $", "MAPE %", "Raw MAE $", "Raw MAPE %"],
  ["Median $/mile × distance (by equipment)", "168.2", "8.33", "220.7", "10.46"],
  ["Log-linear (distance, equipment)", "104.3", "4.39", "157.9", "6.60"],
  ["LightGBM: distance + equipment", "102.9", "4.37", "156.5", "6.59"],
  ["  + weight", "91.5", "3.92", "145.2", "6.13"],
  ["  + market index", "89.8", "3.88", "143.4", "6.07"],
  ["  + quote signal", "60.2", "2.81", "114.1", "5.02"],
  ["  + calendar (day of week, day of month)", "62.9", "2.91", "116.7", "5.12"],
  ["  + geometry (coordinates, haversine, circuity)", "56.2", "2.48", "110.1", "4.70"],
  ["  + time index", "51.3", "2.16", "105.2", "4.43"],
  [{ t: "Final: linear base + LightGBM ensemble", b: true, hl: true }, { t: "44.0", b: true, hl: true }, { t: "1.87", b: true, hl: true }, { t: "98.1", b: true, hl: true }, { t: "4.14", b: true, hl: true }],
]));
content.push(p([r("Final model by fold (MAPE %): May–Jun 1.55, Jun–Jul 1.51, Jul–Aug 1.98, Aug–Sep 2.34, Sep–Oct 1.99.")], { spacing: { before: 120, after: 120 } }));
content.push(img(path.join(R, "fig_progress.png"), 540));
content.push(caption("Figure 3. Contribution of each feature group. Calendar features made things worse (market_index already carries the weekly cycle), so they are not in the final model."));
content.push(h2("3.5 Robustness checks, including where it fails"));
content.push(bullet([b("Short-history stress folds. "), r("With only two to three months of training (test windows starting March and April) error rose to 8.0% and 16.2% MAPE. In those windows quote_signal falls well outside the range seen in training and the model misreads it. The real run is not exposed to this: validation market_index (0.72–1.10) and quote_signal (1.23–2.90) both sit inside the training ranges (0.68–1.47 and 0.69–3.61), and the model trains on ten months.")]));
content.push(bullet([b("Unseen cities. "), r("I removed 8 random cities from training and tested on September–October. Rows touching those cities scored 1.98% MAPE with raw coordinates, versus 2.69% without raw coordinates and 2.80% with no geometry. Coordinates generalise to new cities, so they stay in.")]));
content.push(bullet([b("quote_signal. "), r("Its distribution shifts month to month and its within-month effect is small and inconsistent, so it partly works as a period fingerprint. I kept it because forward validation shows a real gain (3.88% to 2.81%), and the ensemble includes a member that sees only its day-level value. This is the main risk if November–December behaves differently from the months seen.")]));

// ---- 4. Model
content.push(h1("4. Why this model"));
content.push(bullet([b("Linear base on log(rate). "), r("Distance, equipment, weight and market level act smoothly and multiplicatively, so a linear model extrapolates sensibly when the market level drifts.")]));
content.push(bullet([b("LightGBM on the residual. "), r("Learns interactions, regional effects and the slow time drift. A time index lets it carry the latest rate level forward, which cut forward error from 2.48% to 2.16%.")]));
content.push(bullet([b("Three-member average. "), r("Different feature views and regularisation (all features, day-level quote only, all features with stronger regularisation). In an earlier comparison the average scored 1.89% against 1.94–2.03% for the individual members.")]));
content.push(bullet([b("Log target. "), r("Rates span $57 to roughly $6,000 and errors are multiplicative. The prediction is the conditional median, which is the right target for MAE and robust to the label noise.")]));

// ---- 5. December
content.push(h1("5. Fixed December prediction chart"));
content.push(p("The chart below was produced by the provided score.py from the completed december_chart_inputs.csv. Lane, equipment, distance and weight are fixed, so only the date varies."));
content.push(img(path.join(R, "..", "scorer_results", "candidate_december.png"), 540));
content.push(caption("Figure 4. candidate_december.png, unmodified scorer output. Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb."));
content.push(bullet([b("Inputs not in the file. "), r("The December file has no market_index or quote_signal. I used the daily median of each across all loads on that date in validation.csv. These are features only, with no labels involved, and the load-level deviation is set to zero (a typical load that day).")]));
content.push(bullet([b("What the chart shows. "), r("Predicted rate sits between about $820 and $839 (about $2.30 per mile), with a weekly cycle of roughly 2% peak to trough that peaks around Thursday. The cycle comes from the weekly pattern in market_index, which is real in the data. The model has no information about December-specific demand beyond what market_index carries.")]));
content.push(bullet([b("Level assumption. "), r("The time index is flat beyond the last training day, so the model assumes the October rate level persists through December.")]));

// ---- 6. Limits
content.push(h1("6. Limitations"));
content.push(bullet("The scoring metric is hidden. If the hidden labels also contain about 1.4% corrupted rows, those rows will inflate any error metric regardless of model quality; I optimised for the conditional median."));
content.push(bullet("Forward validation covers May to October. I cannot test how the model behaves in a regime that never appears in the training year."));
content.push(bullet("Day-level market and quote medians for validation days are computed from validation features, which assumes those features are available at prediction time."));

const doc = new Document({
  creator: "Candidate", title: "Freight Rate Prediction Report",
  styles: { default: { document: { run: { font, size: 21 } } } },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1000, right: 1440, bottom: 1000, left: 1440 } } }, children: content }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(path.join(R, "Freight_Rate_Report.docx"), buf); console.log("wrote Freight_Rate_Report.docx"); });
