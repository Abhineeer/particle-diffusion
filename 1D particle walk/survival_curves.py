"""
Survival probability S(t) vs time, one curve per branch count b.

S(t) = probability the particle is still on the network at time t
     = fraction of walkers not yet absorbed.

Release at the vertex. More branches -> more exits -> faster drain, so the
curves fall off more steeply as b increases. The late-time slope of each is
the fundamental decay rate D*arctan^2(sqrt b)/L^2 from the decay-rate study.
"""

import numpy as np
import matplotlib.pyplot as plt
from walker_fast import run_ensemble_fast

rng = np.random.default_rng(31)

L, D = 1.0, 1.0
N = 100
n_walkers = 40_000
dt = (L / N) ** 2 / (2 * D)

b_values = [1, 2, 3, 4, 5, 6, 10, 12, 14, 16, 18]
tmax = 4.0
tgrid = np.linspace(0, tmax, 400)
kgrid = tgrid / dt                      # same grid in lattice steps

def survival_from_steps(steps, kgrid):
    s = np.sort(steps)
    return 1.0 - np.searchsorted(s, kgrid, side="right") / steps.size

colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(b_values)))

fig, (axlin, axlog) = plt.subplots(1, 2, figsize=(13, 5))

for b, c in zip(b_values, colors):
    _, _, steps = run_ensemble_fast(b, N, n_walkers, rng, start="vertex")
    S = survival_from_steps(steps, kgrid)
    axlin.plot(tgrid, S, color=c, lw=2, label=f"b = {b}")
    axlog.plot(tgrid, S, color=c, lw=2, label=f"b = {b}")

axlin.set_xlabel("time  t")
axlin.set_ylabel("survival probability  S(t)")
axlin.set_title("linear")
axlin.set_ylim(0, 1)
axlin.grid(alpha=0.25)
axlin.legend(title="branches")

axlog.set_xlabel("time  t")
axlog.set_ylabel("survival probability  S(t)")
axlog.set_title("log scale (straight tail = single exponential; steeper = faster decay)")
axlog.set_yscale("log")
axlog.set_ylim(1e-3, 1)
axlog.grid(alpha=0.25, which="both")
axlog.legend(title="branches")

fig.suptitle(f"Survival vs time across branch count  (N={N}, {n_walkers} walkers/curve)",
             fontsize=13)
fig.tight_layout()
fig.savefig("survival_curves.png", dpi=150)
print("wrote survival_curves.png")