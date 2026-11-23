"""
fig_si.py

SI figure: grid of first-passage-time histograms. Full page width (17.8 cm).

    rows     b = 1, 2, 3
    columns  Pe = Pe_c * {1/4, 1/2, 1, 2, 4}

Linear time axis. Binning is set per column (per Pe): each column spans
0 to the largest 99th percentile among its b rows, cut into N_BINS equal
bins. All rows in a column share those exact bin edges and the same x- and
y-axes, so the b rows compare directly at each Pe. Different columns have
different ranges and bin widths. Solid line:
mean. Dashed lines: 2.5 and 97.5 percentiles (central 95%).

Usage:  python fig_si.py equal    ->  figS1.pdf
        python fig_si.py murray   ->  figS2.pdf
        python fig_si.py          ->  both
Reads data/flow_<model>.npz written by sim_flow.py.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import style

style.setup()
HERE = Path(__file__).parent              # folder this script lives in

N_BINS = 50                                 # bins per panel (edges differ by column)
COL_LABELS = [r"$\mathrm{Pe}_c/4$", r"$\mathrm{Pe}_c/2$", r"$\mathrm{Pe}_c$",
              r"$2\,\mathrm{Pe}_c$", r"$4\,\mathrm{Pe}_c$"]
FIG_NAME = {"equal": "figS1", "murray": "figS2"}
TITLE = {"equal": "equal areas", "murray": "Murray's law"}


def make_figure(model):
    data = np.load(HERE / "data" / f"flow_{model}.npz")
    pe_c = float(data["pe_c"])
    b_values = [int(b) for b in data["b_values"]]
    factors = [float(f) for f in data["pe_factors"]]

    n_rows, n_cols = len(b_values), len(factors)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(style.DOUBLE, 4.6),
                             sharex="col", sharey="col")

    # one set of bin edges per column (per Pe), shared by all b rows:
    # 0 to the column's largest 99th percentile, cut into N_BINS equal bins
    col_bins = {}
    for f in factors:
        col_max = max(np.percentile(data[f"T_b{b}_f{f}"], 99) for b in b_values)
        col_bins[f] = np.linspace(0, col_max, N_BINS + 1)

    for i, b in enumerate(b_values):
        colour = style.B_COLOURS[b]
        for j, f in enumerate(factors):
            ax = axes[i, j]
            T = data[f"T_b{b}_f{f}"]
            bins = col_bins[f]                  # same edges for every b in this column
            counts, _ = np.histogram(T, bins=bins)
            ax.stairs(counts / T.size, bins, fill=True,
                      color=colour, alpha=0.35)
            ax.stairs(counts / T.size, bins, color=colour, lw=0.8)

            lo, hi = np.percentile(T, [2.5, 97.5])
            ax.axvline(T.mean(), color=colour, lw=1.2)
            ax.axvline(lo, color=colour, lw=0.8, ls="--")
            ax.axvline(hi, color=colour, lw=0.8, ls="--")

            ax.set_xlim(bins[0], bins[-1])
            if i == 0:
                ax.set_title(COL_LABELS[j], fontsize=12)
            if j == 0:
                ax.set_ylabel(f"$b={b}$")

    for ax in axes.flat:
        ax.set_yticks([])
    fig.supxlabel(r"$T/\tau_D$", fontsize=12)
    fig.suptitle(f"{TITLE[model]},  " + r"$\mathrm{Pe}_c = $" + f"{pe_c:.2f}",
                 fontsize=12, y=0.99)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    fig.savefig(HERE / f"{FIG_NAME[model]}.pdf")
    fig.savefig(HERE / f"{FIG_NAME[model]}.png", dpi=300)


if len(sys.argv) > 1:
    models = [sys.argv[1]]
else:
    models = ["equal", "murray"]            # default: make both
for model in models:
    make_figure(model)