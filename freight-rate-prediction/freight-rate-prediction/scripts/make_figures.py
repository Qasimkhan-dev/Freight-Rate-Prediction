"""Figures for the report (reads reports/*.csv, re-derives outlier residuals)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from freight_rate import pipeline as PL

OUT = PL.ROOT / "reports"; TEAL = "#064A56"; ORANGE = "#E07B39"; GREY = "#9DAFB3"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. fold schematic
fig, ax = plt.subplots(figsize=(8.6, 3.3), dpi=200)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for i, s in enumerate([5, 6, 7, 8, 9]):
    y = 4 - i
    ax.barh(y, s - 1, left=0, color=TEAL, height=0.55, label="train" if i == 0 else None)
    ax.barh(y, 2, left=s - 1, color=ORANGE, height=0.55, label="test (next 2 months)" if i == 0 else None)
    ax.text(s - 1 + 1, y, f"fold {i+1}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
ax.barh(-1, 10, left=0, color=TEAL, height=0.55); ax.barh(-1, 2, left=10, color="#B5342B", height=0.55, label="final: validation.csv (no labels)")
ax.text(11, -1, "predict", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
ax.set_yticks(list(range(4, -1, -1)) + [-1]); ax.set_yticklabels([f"Fold {i}" for i in range(1, 6)] + ["Final"])
ax.set_xticks(np.arange(12) + 0.5); ax.set_xticklabels(months); ax.set_xlim(0, 12)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
ax.set_title("Forward-chaining validation mirrors the real task (train through Oct, predict Nov-Dec)", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_folds.png"); plt.close(fig)

# 2. model progression
r = pd.read_csv(OUT / "cv_results.csv")
labels = ["Median $/mile\n× distance", "Log-linear\n(dist, equip)", "GBM\ndist+equip", "+ weight", "+ market\nindex", "+ quote\nsignal", "+ calendar", "+ geometry", "+ time\nindex", "FINAL\nensemble"]
fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
cols = [GREY] * 2 + [TEAL] * 7 + [ORANGE]
b = ax.bar(range(len(r)), r.clean_MAPE, color=cols)
for i, v in enumerate(r.clean_MAPE): ax.text(i, v + 0.12, f"{v:.2f}%", ha="center", fontsize=8)
ax.set_xticks(range(len(r))); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylabel("MAPE % (forward folds, clean rows)")
ax.set_title("Each step on the 5 forward folds (calendar features hurt, so they are excluded)", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_progress.png"); plt.close(fig)

# 3. outlier gap
D = PL.prepare_train_validation(); res = D["resid"]
fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=200)
bins = np.linspace(-1.6, 1.6, 129)
ax.hist(res[~D["outlier"]], bins=bins, color=TEAL, label=f"kept ({int((~D['outlier']).sum()):,})")
ax.hist(res[D["outlier"]], bins=bins, color=ORANGE, label=f"flagged ({int(D['outlier'].sum())})")
ax.set_yscale("log"); ax.set_xlabel("out-of-fold residual of log(posted_rate)"); ax.set_ylabel("rows (log scale)"); ax.legend(frameon=False)
ax.set_title("Corrupted labels sit far from everything else, so the cut-off is unambiguous", loc="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_outliers.png"); plt.close(fig)
print("figures written")
