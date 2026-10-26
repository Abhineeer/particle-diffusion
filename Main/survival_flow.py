"""
survival_flow.py

The same survival picture as survival_b.py, but with a current, and
SIMULATED rather than drawn from a formula.

Units: L = 1, D = 1, so Pe = v exactly. The only knob is the flow speed.

THE VERTEX RULE, AS ONE PARAMETER
---------------------------------
Flux conservation at the junction gives

    v_branch = v_trunk / beta,      beta = b * A_1 / A_t

    beta = b   each branch has the same cross-section as the trunk
    beta = 1   the branch areas sum to the trunk area

Both conserve flux. beta is a geometric property of the vessels, so this
is a measurement, not a choice between two rules.

WHAT THE TWO COLUMNS SHOW
-------------------------
beta = b: adding branches divides the current b ways, so each branch flows
          more slowly. The diffusive speed-up from extra exits is more
          than cancelled and MORE BRANCHES BECOMES SLOWER.

beta = 1: every branch carries the same v, so which branch the particle
          takes makes no difference to the ride out, and the survival
          curves collapse onto each other. Branch count stops mattering.

Either way the pure-diffusion lesson survives in a stronger form: beyond
a handful of branches, extra branches do not help.
"""

import numpy as np
import matplotlib.pyplot as plt

import lattice
import star_drift as sd
from survival_b import survival_from_times, gamma0, GAMMA_MAX


if __name__ == "__main__":
    rng = np.random.default_rng(1234)

    N = 60
    n_walkers = 20000
    b_values = [1, 2, 3, 5, 10]
    Pe_values = [0.0, 5.0, 20.0]

    tmax = 2.0
    tgrid = np.linspace(0.0, tmax, 500)

    betas = [("$\\beta = b$:  each branch as wide as the trunk",
              lambda b: float(b)),
             ("$\\beta = 1$:  branch areas sum to the trunk",
              lambda b: 1.0)]

    results = {}
    for bi, (name, bfn) in enumerate(betas):
        for Pe in Pe_values:
            for b in b_values:
                _, st = sd.run_many(b, N, n_walkers, rng, Pe=Pe,
                                    beta=bfn(b))
                t = lattice.to_time(st, N)
                results[(bi, Pe, b)] = (survival_from_times(t, tgrid),
                                        t.mean())

    # ---- numbers ----
    for bi, (name, _) in enumerate(betas):
        print(f"\n{name.replace('$','').replace(chr(92)+'beta','beta')}")
        print(f"  mean first-passage time,  N = {N}\n")
        print(f"  {'Pe':>5} " + " ".join(f"{('b='+str(b)):>8}"
                                         for b in b_values))
        for Pe in Pe_values:
            row = [results[(bi, Pe, b)][1] for b in b_values]
            print(f"  {Pe:>5.0f} " + " ".join(f"{v:>8.3f}" for v in row))

    # ---- figure ----
    # each column gets its own time axis: with flow everything happens
    # sooner, and a shared axis would squash the fast columns into a line
    xlims = {}
    for Pe in Pe_values:
        worst = max(results[(bi, Pe, b)][1]
                    for bi in range(len(betas)) for b in b_values)
        xlims[Pe] = 3.2 * worst

    fig, axes = plt.subplots(2, len(Pe_values), figsize=(14, 7.6),
                             sharey=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(b_values)))

    for bi, (name, _) in enumerate(betas):
        for j, Pe in enumerate(Pe_values):
            ax = axes[bi, j]
            for b, c in zip(b_values, colors):
                S, _ = results[(bi, Pe, b)]
                ax.plot(tgrid, S, color=c, lw=2.1, label=f"b = {b}")
            ax.set_xlim(0, xlims[Pe])
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if bi == 0:
                ax.set_title(f"Pe = $v$ = {Pe:.0f}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{name}\n\nsurvival  $S(t)$", fontsize=9.5)
            ax.set_xlabel("time  $t$   (units of $L^2/D$)")
    axes[0, 0].legend(fontsize=8.5, title="branches")

    fig.suptitle("With flow, whether extra branches help depends on how the "
                 "current divides\n"
                 r"top: current split $b$ ways, more branches is SLOWER   "
                 r"|   bottom: current undivided, branch count stops "
                 r"mattering", fontsize=11.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("survival_flow.png", dpi=150)
    print("\nwrote survival_flow.png")
