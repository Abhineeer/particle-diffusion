"""
Figure 2: no flow, equal areas. Single column (8.5 cm).

    (a) cumulative probability of absorption by time t, for b = 1, 2, 3
    (b) <T>/tau against b: exact curve 1 + 1/b, limit 1 (dashed), and at
        each b a sideways histogram of T/tau with whiskers at the 2.5 and
        97.5 percentiles. All histograms share the same bins.

Reads data/fig2.npz written by sim_fig2.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import style

style.setup()

data = np.load("data/fig2.npz")
b_values = list(data["b_values"])
T = {b: data[f"T_b{b}"] for b in b_values}

CDF_B = [1, 2, 3, 4, 5]            # curves shown in panel (a)
HIST_WIDTH = 0.42                  # max histogram width, in units of b
T_MAX = 7.0                        # upper edge of the shared bins
BINS = np.linspace(0, T_MAX, 57)

fig, axes = plt.subplot_mosaic([["a"], ["b"]],
                               figsize=(style.SINGLE, 5.0),
                               height_ratios=[1, 1.35])

# ---------- (a) cumulative absorbed fraction ----------
ax = axes["a"]
for b in CDF_B:
    t_sorted = np.sort(T[b])
    frac = np.arange(1, t_sorted.size + 1) / t_sorted.size
    ax.plot(t_sorted, frac, color=style.B_COLOURS[b], lw=1.5, label=f"$b={b}$")
ax.set_xlim(0, 4)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"$t/\tau_D$")
ax.set_ylabel("probability of\nabsorption")
ax.legend(frameon=False, loc="lower right", ncol=2, handlelength=0.9,
          labelspacing=0.1, handletextpad=0.3, columnspacing=0.6,
          borderaxespad=0.1)

# ---------- (b) mean time vs b, with histograms ----------
ax = axes["b"]

# shared-bin histograms, all scaled by the same factor
counts = {b: np.histogram(T[b], bins=BINS, density=True)[0] for b in b_values}
scale = HIST_WIDTH / max(c.max() for c in counts.values())
centres = 0.5 * (BINS[:-1] + BINS[1:])
height = BINS[1] - BINS[0]

for b in b_values:
    colour = style.B_COLOURS[b]
    ax.barh(centres, counts[b] * scale, height=height, left=b,
            color=colour, alpha=0.35, linewidth=0)

    lo, hi = np.percentile(T[b], [2.5, 97.5])
    ax.plot([b, b], [lo, hi], color=colour, lw=1.2)
    ax.plot([b - 0.08, b + 0.08], [lo, lo], color=colour, lw=1.2)
    ax.plot([b - 0.08, b + 0.08], [hi, hi], color=colour, lw=1.2)

    mean = T[b].mean()
    err = T[b].std() / np.sqrt(T[b].size)
    ax.errorbar(b, mean, yerr=err, fmt="o", color=colour, ms=4, zorder=4)

b_line = np.linspace(0.8, b_values[-1] + 0.6, 200)
ax.plot(b_line, 1 + 1 / b_line, color="#555555", lw=1, zorder=3,
        label=r"$1 + 1/b$")
ax.axhline(1, color="#555555", lw=1, ls="--", zorder=3)

ax.set_xlim(0.6, b_values[-1] + 0.6)
ax.set_ylim(0, T_MAX)
ax.set_xticks(b_values)
ax.set_xlabel(r"branches $b$")
ax.set_ylabel(r"$T/\tau_D$")
ax.legend(frameon=False, loc="upper right", handlelength=1.2)

for key, ax in axes.items():
    ax.text(-0.30, 1.04, f"({key})", transform=ax.transAxes,
            fontweight="bold", va="top")

fig.tight_layout(pad=0.3, h_pad=0.8)
fig.savefig("fig2.pdf")
fig.savefig("fig2.png", dpi=300)