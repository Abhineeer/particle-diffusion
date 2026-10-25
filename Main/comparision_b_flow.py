"""
comparison_b_flow.py

Diminishing returns from extra branches, with and without flow.

PURE DIFFUSION
--------------
Adding branches speeds absorption up, but with a ceiling:

    <t> = (L^2/D) * (1/2 + 1/b)   ->   L^2/(2D)

The 1/b term is the whole b-dependence, and it dies. Going from b = 1 to
b = 2 is worth a lot; going from 10 to 20 is worth almost nothing. At
large b the vertex stops being a junction and acts like an absorber.

WITH FLOW
---------
This depends entirely on how the current divides at the vertex, which is
the open question:

  "flux"  trunk carries v, each branch carries v/b. More branches means
          each one flows more slowly, so the speed-up from extra exits is
          cancelled, and can reverse.

  "equal" every branch carries v. More branches then changes almost
          nothing, because the ride out is the same whichever branch is
          taken.

The two rules give qualitatively different answers, which is why the
vertex rule has to be settled before any of this means anything.
"""

import numpy as np
import matplotlib.pyplot as plt

import lattice
import star_drift as sd


if __name__ == "__main__":
    rng = np.random.default_rng(2026)

    N = 60
    n_walkers = 4000
    b_values = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    Pe_values = [0.0, 5.0, 10.0, 40.0]
    L = lattice.length(N)

    results = {}
    for split in ("flux", "equal"):
        for Pe in Pe_values:
            means = []
            for b in b_values:
                _, st = sd.run_many(b, N, n_walkers, rng, Pe=Pe, split=split)
                means.append(lattice.to_time(st).mean())
            results[(split, Pe)] = np.array(means)

    # ---- the diminishing-returns table, pure diffusion ----
    m0 = results[("flux", 0.0)]
    print(f"PURE DIFFUSION   N = {N}, L = {L:.0f}\n")
    print(f"  {'b':>4} {'mean t':>10} {'exact':>10} "
          f"{'gain over previous b':>22}")
    for i, b in enumerate(b_values):
        exact = lattice.to_time(sd.__dict__.get('_x', 0)) if False else \
            L**2 * (0.5 + 1.0/b) / lattice.D
        gain = "" if i == 0 else f"{m0[i-1] - m0[i]:.0f}"
        print(f"  {b:>4} {m0[i]:>10.0f} {exact:>10.0f} {gain:>22}")
    print(f"\n  floor at L^2/2D = {L**2/(2*lattice.D):.0f}")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(Pe_values)))

    bfine = np.linspace(1, 20, 200)
    exact_curve = L**2 * (0.5 + 1.0/bfine) / lattice.D

    for ax, split, title in (
            (ax1, "flux", "flux-conserving split:  branch carries $v/b$"),
            (ax2, "equal", "equal-speed split:  branch carries $v$")):
        ax.plot(bfine, exact_curve, color="0.6", lw=1.4, zorder=1,
                label=r"exact, no flow: $L^2(1/2+1/b)/D$")
        ax.axhline(L**2/(2*lattice.D), color="k", ls=":", lw=1.1,
                   label=r"$L^2/2D$ floor")
        for Pe, c in zip(Pe_values, colors):
            ax.plot(b_values, results[(split, Pe)], "o-", color=c, ms=6,
                    lw=1.6, label=f"Pe = {Pe:.0f}")
        ax.set_yscale("log")
        ax.set_xlabel("number of branches  b")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7.5)
    ax1.set_ylabel("mean first-passage time")

    fig.suptitle("Extra branches help only under pure diffusion, and the "
                 "help saturates.\nUnder flow the answer depends on the "
                 "vertex rule.", fontsize=11)
    fig.tight_layout()
    fig.savefig("comparison_b_flow.png", dpi=150)
    print("\nwrote comparison_b_flow.png")