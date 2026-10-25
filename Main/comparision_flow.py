"""
comparison_flow.py

The same two-length comparison as comparison.py, but with flow switched on.

WHY THE COLLAPSE MUST BREAK
---------------------------
With pure diffusion there is only one timescale, L^2/D, so rescaling by
L^2 collapses every length onto one curve. That is comparison.py.

Switch on a current and there are two:

    tau_adv  = L / v        linear in L
    tau_diff = L^2 / D      quadratic in L

They scale differently, so no single power of L can collapse both. The
self-similarity that makes the composition idea work is a property of
PURE DIFFUSION, and flow destroys it.

IMPORTANT: the two runs share the same physical velocity v, not the same
Peclet number. Same fluid, two different vessel lengths. Since
Pe = L*v/D, the longer segment automatically sits at twice the Peclet
number of the shorter one. That is exactly the situation inside a tree,
where the vessels shorten with depth while the fluid is the same.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import lattice
import walker_drift as wd


def sample_at_v(N, eps, n_walkers, rng):
    """Times for a segment of N spacings at a fixed coin bias eps (fixed v)."""
    Pe = 2.0 * N * eps          # Pe = 2*N*eps, so fixing eps fixes v
    return lattice.to_time(wd.run_many(N, n_walkers, rng, Pe=Pe))


if __name__ == "__main__":
    rng = np.random.default_rng(808)

    n_walkers = 40000
    N1, N2 = 100, 200
    L1, L2 = lattice.length(N1), lattice.length(N2)
    factor = (L2 / L1)**2

    # three flow strengths, set by the coin bias so that v is the same for
    # both lengths within each case
    cases = [("no flow", 0.0),
             ("weak flow", 0.01),
             ("strong flow", 0.05)]

    print(f"L1 = {L1:.0f}, L2 = {L2:.0f}, {n_walkers} walkers each\n")
    print(f"  {'case':>12} {'eps':>7} {'Pe(L1)':>8} {'Pe(L2)':>8} "
          f"{'mean ratio':>12} {'KS p':>8}")

    store = {}
    for name, eps in cases:
        t1 = sample_at_v(N1, eps, n_walkers, rng)
        t2 = sample_at_v(N2, eps, n_walkers, rng)
        store[name] = (t1, t2)
        ks = stats.ks_2samp(t1, t2 / factor)
        print(f"  {name:>12} {eps:>7.3f} {2*N1*eps:>8.1f} {2*N2*eps:>8.1f} "
              f"{t2.mean()/t1.mean():>12.3f} {ks.pvalue:>8.3f}")

    print("\n  mean ratio 4 means pure L^2 scaling; 2 would be pure L/v.")
    print("  KS p near 1 means the rescaled samples match; near 0 means")
    print("  the collapse has failed.")

    # --- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (name, eps) in zip(axes, cases):
        t1, t2 = store[name]
        hi = np.percentile(t1, 99)
        ax.hist(t1, bins=80, range=(0, hi), density=True, histtype="step",
                lw=2, color="#2b857f", label=f"L = {L1:.0f}")
        ax.hist(t2 / factor, bins=80, range=(0, hi), density=True,
                histtype="step", lw=2, color="#e08214",
                label=f"L = {L2:.0f}, rescaled by $1/4$")
        ks = stats.ks_2samp(t1, t2 / factor)
        ax.set_title(f"{name}   (Pe = {2*N1*eps:.0f} and {2*N2*eps:.0f})\n"
                     f"KS p = {ks.pvalue:.3f}")
        ax.set_xlabel("first-passage time  t")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")

    fig.suptitle(r"The $L^2$ collapse is a property of pure diffusion. "
                 r"Flow breaks it.")
    fig.tight_layout()
    fig.savefig("comparison_flow.png", dpi=150)
    print("\nwrote comparison_flow.png")