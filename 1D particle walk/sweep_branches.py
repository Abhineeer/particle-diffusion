"""
sweep_branches.py

The six-branch question. Loop b = 1..6, measure the trunk-return probability
h_V for each, compare to 1/(1+b), make one figure.

Uses the fast engine (walker_fast.py). Releases at the VERTEX, because
h_V = 1/(1+b) is defined from the vertex -- releasing at the trunk tip would
give ~1 for every b and the curve would vanish.
"""

import numpy as np
import matplotlib.pyplot as plt
from walker_fast import run_ensemble_fast

rng = np.random.default_rng(1)

N = 100
n_walkers = 20_000
b_values = [1, 2, 3, 4, 5, 6]

measured = []
errbar = []

print(f"branch-count sweep (N={N}, walkers={n_walkers}, start=vertex)\n")
print(f"  {'b':>2} {'measured hV':>12} {'1/(1+b)':>10} {'tip split':>28}")
for b in b_values:
    branch, reached_trunk, steps = run_ensemble_fast(b, N, n_walkers, rng, start="vertex")

    hV = reached_trunk.mean()
    measured.append(hV)
    errbar.append(np.sqrt(hV * (1 - hV) / n_walkers))

    tip_frac = [(branch == i).mean() for i in range(1, b + 1)]
    tip_str = " ".join(f"{f:.3f}" for f in tip_frac)
    print(f"  {b:>2} {hV:>12.4f} {1/(1+b):>10.4f}   {tip_str}")

measured = np.array(measured)
errbar = np.array(errbar)

fig, ax = plt.subplots(figsize=(7, 5))
bfine = np.linspace(1, 6, 200)
ax.plot(bfine, 1 / (1 + bfine), color="0.5", lw=1.5, label=r"theory  $1/(1+b)$")
ax.errorbar(b_values, measured, yerr=errbar, fmt="o", color="teal",
            capsize=3, ms=7, label="simulation")
ax.set_xlabel("number of branches, b")
ax.set_ylabel(r"trunk-return probability  $h_V$")
ax.set_title(f"Trunk-return probability vs branch count\n(N={N}, {n_walkers} walkers per point)")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig("sweep_branches.png", dpi=150)
print("\nwrote sweep_branches.png")