"""
Figures 3 (equal areas) and 4 (Murray's law). Full page width (17.8 cm).

    (a)-(c) fraction absorbed against t/tau at Pe = Pe_c/2, Pe_c, 2 Pe_c,
            for b = 1..5
    (d)     <T>/tau (linear) against Pe/Pe_c at 1/4, 1/2, 1, 2, 4 (evenly spaced). Lines: exact formula.
            At each Pe, histograms of T/tau for b = 1 (left) and b = 2
            (right), shared linear bins, whiskers at the 2.5 and 97.5
            percentiles. The two curves cross at Pe_c.

Usage:  python fig_flow.py equal    ->  fig3.pdf
        python fig_flow.py murray   ->  fig4.pdf
Reads data/flow_<model>.npz written by sim_flow.py.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import style
from analytic import beta_of, mean_T

style.setup()
HERE = Path(__file__).parent              # folder this script lives in

if len(sys.argv) > 1:
    model = sys.argv[1]
else:
    model = "equal"                       # default when run with the play button
fig_name = {"equal": "fig3", "murray": "fig4"}[model]

data = np.load(HERE / "data" / f"flow_{model}.npz")
pe_c = float(data["pe_c"])
b_values = [1, 2]                           # panel (d): b = 1 and 2 only
cdf_b = [1, 2, 3, 4, 5]                     # panels (a)-(c): b = 1..5
factors = [float(f) for f in data["pe_factors"]]
T = {(b, f): data[f"T_b{b}_f{f}"] for b in cdf_b for f in factors}

CDF_FACTORS = [0.25, 1.0, 4.0]
CDF_TITLES = [r"$\mathrm{Pe}_c/4$", r"$\mathrm{Pe}_c$", r"$4\,\mathrm{Pe}_c$"]
T_MAX = 5.5
BINS = np.linspace(0, T_MAX, 111)           # shared by every histogram
HIST_WIDTH = 0.38                           # max width, in units of log2(Pe/Pe_c)
WHISKER_OFFSET = 0.05
CAP = 0.04                                  # half-width of whisker caps

fig, axes = plt.subplot_mosaic([["a", "b", "c"], ["d", "d", "d"]],
                               figsize=(style.DOUBLE, 5.2),
                               height_ratios=[1, 1.5])

# ---------- (a)-(c): fraction absorbed ----------
for key, f, title in zip(["a", "b", "c"], CDF_FACTORS, CDF_TITLES):
    ax = axes[key]
    for b in cdf_b:
        t_sorted = np.sort(T[(b, f)])
        frac = np.arange(1, t_sorted.size + 1) / t_sorted.size
        ax.plot(t_sorted, frac, color=style.B_COLOURS[b], lw=1.5,
                label=f"${b}$")
    t_max = max(np.percentile(T[(b, f)], 99) for b in cdf_b)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.02)
    ax.set_title(r"$\mathrm{Pe} = $" + title, fontsize=12)
    ax.set_xlabel(r"$t/\tau_D$")
    if key == "a":
        ax.set_ylabel("probability of\nabsorption")
    else:
        ax.set_yticklabels([])
    if key == "b":
        # narrow column "b: 1..5" tucked into the empty lower right of (b)
        ax.legend(title=r"$b$", frameon=False, loc="lower right", ncol=1,
                  handlelength=0.8, handletextpad=0.3, labelspacing=0.05,
                  borderaxespad=0.15)

# ---------- (d): mean time against Pe, with histograms ----------
ax = axes["d"]
centres = 0.5 * (BINS[:-1] + BINS[1:])
side = {1: -1, 2: +1}                      # b = 1 to the left, b = 2 to the right

for f in factors:
    x0 = np.log2(f)                        # evenly spaced: 1/4, 1/2, 1, 2, 4
    for b in b_values:
        colour = style.B_COLOURS[b]
        sample = T[(b, f)]
        counts, _ = np.histogram(sample, bins=BINS)
        width = HIST_WIDTH * counts / counts.max()
        if side[b] < 0:
            ax.barh(centres, width, height=np.diff(BINS), left=x0 - width,
                    color=colour, alpha=0.3, linewidth=0)
        else:
            ax.barh(centres, width, height=np.diff(BINS), left=x0,
                    color=colour, alpha=0.3, linewidth=0)

        xw = x0 + side[b] * WHISKER_OFFSET
        lo, hi = np.percentile(sample, [2.5, 97.5])
        ax.plot([xw, xw], [lo, hi], color=colour, lw=1.2)
        ax.plot([xw - CAP, xw + CAP], [lo, lo], color=colour, lw=1.2)
        ax.plot([xw - CAP, xw + CAP], [hi, hi], color=colour, lw=1.2)
        ax.plot(xw, sample.mean(), "o", color=colour, ms=4, zorder=4)

# exact curves
x_line = np.linspace(-2.6, 2.6, 300)        # in log2(Pe/Pe_c)
for b in b_values:
    beta = beta_of(b, model)
    y_line = [mean_T(pe_c * 2**x, beta) for x in x_line]
    ax.plot(x_line, y_line, color=style.B_COLOURS[b], lw=1.2,
            label=f"$b={b}$")
ax.axvline(0, color="#777777", lw=0.8, ls=":")      # Pe = Pe_c

ax.set_ylim(0, T_MAX)
ax.set_xlim(-2.6, 2.6)
ax.set_xticks([np.log2(f) for f in factors])
ax.set_xticklabels([r"$\frac{1}{4}$", r"$\frac{1}{2}$", "$1$", "$2$", "$4$"])
ax.set_xlabel(r"$\mathrm{Pe}/\mathrm{Pe}_c$"
              + f"     ($\\mathrm{{Pe}}_c = {pe_c:.2f}$)")
ax.set_ylabel(r"$T/\tau_D$")
ax.legend(frameon=False, loc="upper right", handlelength=1.2)

for key, ax in axes.items():
    x_lab = -0.09 if key == "d" else -0.30
    ax.text(x_lab, 1.16 if key != "d" else 1.02, f"({key})",
            transform=ax.transAxes, fontweight="bold", va="top")

fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
fig.savefig(HERE / f"{fig_name}.pdf")
fig.savefig(HERE / f"{fig_name}.png", dpi=300)