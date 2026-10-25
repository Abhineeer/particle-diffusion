"""
comparison.py

Sample first-passage times at one length. Sample at another length.
Compare the two.

THE CLAIM BEING TESTED
----------------------
Pure diffusion has no intrinsic length, so the only timescale available
is L^2/D. If that is the whole story, then the first-passage time at
length L_2 is just the time at L_1 stretched by (L_2/L_1)^2:

    t(L_2)  =  (L_2/L_1)^2 * t(L_1)     as distributions, not just means

If true, one measured distribution covers every length, which is what
makes it possible to build a deep tree out of rescaled copies instead of
simulating the whole thing.

WHY THIS IS NOT CIRCULAR
------------------------
The lattice spacing a is fixed. A longer segment is MORE SITES, so the
walker genuinely has further to go. No line of code multiplies the clock.

WHAT TO EXPECT
--------------
Exactly on the lattice the mean from the reflecting end is N(N+1) steps,
so doubling N gives

    2N(2N+1) / (N(N+1))  ~  4 - 2/N

slightly under 4, by an amount that shrinks as N grows. That gap is the
lattice being coarse, not a failure of the scaling.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import lattice
import walker
import test_segment as seg


def sample(N, n_walkers, rng):
    """First-passage times for a segment of N spacings, released at the wall."""
    return lattice.to_time(walker.run_many(N, n_walkers, rng))


if __name__ == "__main__":
    rng = np.random.default_rng(2024)

    n_walkers = 60000
    N1, N2 = 100, 200          # the two lengths being compared

    t1 = sample(N1, n_walkers, rng)
    t2 = sample(N2, n_walkers, rng)

    L1, L2 = lattice.length(N1), lattice.length(N2)
    factor = (L2 / L1)**2

    print("SAMPLE AT ONE LENGTH, SAMPLE AT ANOTHER, COMPARE\n")
    print(f"  spacing a = {lattice.a},  D = {lattice.D},  "
          f"{n_walkers} walkers each\n")
    print(f"  {'':>14} {'L':>8} {'sites':>7} {'mean t':>12} {'exact':>12}")
    for N, t in ((N1, t1), (N2, t2)):
        print(f"  {'run':>14} {lattice.length(N):>8.0f} {N:>7} "
              f"{t.mean():>12.1f} {lattice.to_time(N*(N+1)):>12.1f}")

    # --- the mean -----------------------------------------------------
    measured = t2.mean() / t1.mean()
    exact = 2*(2*N1 + 1) / (N1 + 1)
    print(f"\n  ratio of means   measured {measured:.4f}   "
          f"exact lattice {exact:.4f}   continuum {factor:.4f}")

    # --- the whole distribution, not just the mean ---------------------
    # rescale the longer run down onto the shorter one and ask whether the
    # two samples are the same distribution
    t2_rescaled = t2 / factor
    ks = stats.ks_2samp(t1, t2_rescaled)
    print(f"\n  two-sample KS on t(L1) vs t(L2)/{factor:.0f}")
    print(f"     statistic = {ks.statistic:.4f}")
    print(f"     p-value   = {ks.pvalue:.3f}")
    print(f"     (a large p means the rescaled samples are "
          f"indistinguishable)")

    # --- figure -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    hi = np.percentile(t2, 99.5)
    ax1.hist(t1, bins=90, range=(0, hi), density=True, histtype="step",
             lw=2, color="#2b857f", label=f"L = {L1:.0f}")
    ax1.hist(t2, bins=90, range=(0, hi), density=True, histtype="step",
             lw=2, color="#e08214", label=f"L = {L2:.0f}")
    ax1.set_xlabel("first-passage time  t")
    ax1.set_ylabel("density")
    ax1.set_title("the two samples, as measured")
    ax1.legend()
    ax1.grid(alpha=0.25)

    hi2 = np.percentile(t1, 99.5)
    ax2.hist(t1, bins=90, range=(0, hi2), density=True, histtype="step",
             lw=2, color="#2b857f", label=f"L = {L1:.0f}")
    ax2.hist(t2_rescaled, bins=90, range=(0, hi2), density=True,
             histtype="step", lw=2, color="#e08214",
             label=f"L = {L2:.0f},  rescaled by $(L_1/L_2)^2$")
    u = np.linspace(hi2/400, hi2, 400)
    ax2.plot(u, seg.fpt_density(u, L1), "k--", lw=1.4, label="analytic")
    ax2.set_xlabel("first-passage time  t")
    ax2.set_title(f"after rescaling:  KS p = {ks.pvalue:.2f}")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.suptitle(r"First-passage times scale as $L^2$: one distribution "
                 r"covers every length")
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=150)
    print("\nwrote comparison.png")