"""
test_scaling.py

THE DELIVERABLE.

Question: does the first-passage time on a segment scale like L^2?

If it does, then a distribution of times measured at one length can be
rescaled to any other length, and we never have to simulate a long
segment again. That is what makes the composition idea work: a tree is
built by stitching rescaled copies of one measured distribution.

Why this is a real test and not a tautology. The lattice spacing a is
fixed, so a longer segment is MORE SITES. The walker genuinely has
further to travel. Nothing in the code multiplies the clock by 4.

What to expect. The exact lattice mean from the reflecting end is
N*(N+1) steps, so the ratio of means between 2N and N is

    2N(2N+1) / (N(N+1))  =  2(2N+1)/(N+1)  ~  4 - 2/N

NOT exactly 4. The gap is the lattice being coarse and it shrinks as N
grows. At N = 100 expect about 3.98. Chasing that 2% as if it were a
bug would waste an afternoon.

The collapse plot: histogram t / L^2 for several N. If the curves lie
on top of each other, the scaling holds.
"""

import numpy as np
import matplotlib.pyplot as plt

import lattice
import walker
import test_segment as seg


def run_at(N, n_walkers, rng):
    """Return the physical first-passage times for a segment of N spacings."""
    steps = walker.run_many(N, n_walkers, rng)
    return lattice.to_time(steps)


if __name__ == "__main__":
    rng = np.random.default_rng(11)

    n_walkers = 40000
    N_values = [50, 100, 200]

    results = {}
    print(f"{n_walkers} walkers per length\n")
    print(f"  {'N':>5} {'L':>7} {'mean t':>12} {'exact t':>12} "
          f"{'mean t / L^2':>14}")

    for N in N_values:
        t = run_at(N, n_walkers, rng)
        results[N] = t
        L = lattice.length(N)
        exact = lattice.to_time(N * (N + 1))
        print(f"  {N:>5} {L:>7.0f} {t.mean():>12.1f} {exact:>12.1f} "
              f"{t.mean()/L**2:>14.5f}")

    # --- the ratio test ----------------------------------------------------
    print("\nratio of means when the length doubles")
    print(f"  {'N -> 2N':>12} {'measured':>10} {'exact lattice':>15} "
          f"{'continuum':>11}")
    for N in N_values[:-1]:
        measured = results[2*N].mean() / results[N].mean()
        exact = 2*(2*N + 1) / (N + 1)
        print(f"  {N:>5} -> {2*N:<4} {measured:>10.4f} {exact:>15.4f} "
              f"{4.0:>11.4f}")

    # --- the collapse ------------------------------------------------------
    # dividing t by L^2 should make every distribution the same curve
    print("\ncollapse check: mean of t/L^2 should be identical across N")
    for N in N_values:
        L = lattice.length(N)
        print(f"   N = {N:>4}   <t>/L^2 = {results[N].mean()/L**2:.5f}")

    # --- figure ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = ["#2b857f", "#e08214", "#7a4fa3"]

    # left: raw, the curves are nowhere near each other
    for N, c in zip(N_values, colors):
        ax1.hist(results[N], bins=100, density=True, histtype="step",
                 lw=2, color=c, label=f"N = {N}")
    ax1.set_xlabel("first-passage time  t")
    ax1.set_ylabel("density")
    ax1.set_title("raw times: different lengths, different scales")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # right: rescaled by L^2, they collapse onto the analytic curve
    for N, c in zip(N_values, colors):
        L = lattice.length(N)
        ax2.hist(results[N]/L**2, bins=100, range=(0, 2.5), density=True,
                 histtype="step", lw=2, color=c, label=f"N = {N}")

    # the analytic curve, written in rescaled units (L = 1)
    u = np.linspace(0.002, 2.5, 400)
    ax2.plot(u, seg.fpt_density(u, 1.0), "k--", lw=1.5,
             label="analytic, $L=1$")
    ax2.set_xlabel(r"rescaled time  $t / L^2$")
    ax2.set_ylabel("density")
    ax2.set_title(r"rescaled by $L^2$: the curves collapse")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("test_scaling.png", dpi=150)
    print("\nwrote test_scaling.png")