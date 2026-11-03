"""
survival_flow.py

Survival across branch count, at three flow speeds. Simulated, not drawn
from a formula.

Units: L = 1, D = 1, so Pe = v exactly.

THE VERTEX RULE
---------------
Flux conservation at a junction gives

    v_branch = v_trunk / beta,      beta = b * A_1 / A_t

We take beta = b: each branch has the same cross-section as the trunk, so
the current is divided b ways. More branches means each one flows more
slowly, which is the cost that works against the extra exits.

(Murray's law would set beta from the real daughter-to-parent area
relationship instead of by hand. That is the next step.)
"""

import numpy as np
import matplotlib.pyplot as plt

import lattice
import star_drift as sd

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 19,
    "legend.title_fontsize": 20,
})


def survival_from_times(times, tgrid):
    s = np.sort(times)
    return 1.0 - np.searchsorted(s, tgrid, side="right") / times.size


if __name__ == "__main__":
    rng = np.random.default_rng(1234)

    N = 60
    n_walkers = 20000
    b_values = [1, 2, 3, 5, 10]
    Pe_values = [0.0, 5.0, 20.0]

    # collect the times first so each panel can set its own time grid
    times = {}
    for Pe in Pe_values:
        for b in b_values:
            _, st = sd.run_many(b, N, n_walkers, rng, Pe=Pe, beta=float(b))
            times[(Pe, b)] = lattice.to_time(st, N)

    print(f"N = {N},  L = {lattice.L_DEFAULT},  D = {lattice.D},  "
          f"{n_walkers} walkers,  beta = b\n")
    print(f"  {'Pe':>5} " + " ".join(f"{('b='+str(b)):>9}" for b in b_values))
    for Pe in Pe_values:
        row = [times[(Pe, b)].mean() for b in b_values]
        print(f"  {Pe:>5.0f} " + " ".join(f"{v:>9.3f}" for v in row))

    # --- figure: one row, three panels --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.0), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(b_values)))

    for ax, Pe in zip(axes, Pe_values):
        # run the grid out to where the slowest curve has essentially died
        tmax = 2.6 * max(times[(Pe, b)].mean() for b in b_values)
        tgrid = np.linspace(0.0, tmax, 500)
        for b, c in zip(b_values, colors):
            ax.plot(tgrid, survival_from_times(times[(Pe, b)], tgrid),
                    color=c, lw=3.0, label=f"{b}")
        ax.set_xlim(0, tmax)
        ax.set_ylim(0, 1)
        ax.set_title(f"Pe = {Pe:.0f}")
        ax.set_xlabel("time  $t$   (units of $L^2/D$)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("survival  $S(t)$")
    axes[0].legend(title="branches  $b$", loc="upper right")

    fig.suptitle("The current is divided $b$ ways, so more branches flow "
                 "more slowly", fontsize=25, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("survival_flow.png", dpi=150)
    print("\nwrote survival_flow.png")