"""
test_segment.py

Does the simulated walk on one segment agree with the exact answer?

Setup: reflecting at x = 0, absorbing at x = L, released at x = 0.

The analytic solution. Separating variables with p_x(0) = 0 and p(L) = 0
gives cosine modes that vanish at the far end:

    mu_n = (2n+1) * pi / (2L)        n = 0, 1, 2, ...

A point release at x = 0 projects onto every mode with the same weight,
and integrating the density over the segment gives the survival
probability

    S(t) = sum_n  4*(-1)^n / ((2n+1)*pi)  * exp(-D * mu_n^2 * t)

and the first-passage density is f(t) = -dS/dt, so

    f(t) = sum_n  4*(-1)^n / ((2n+1)*pi) * D * mu_n^2 * exp(-D * mu_n^2 * t)

Three things get checked here:
    1. the mean time matches the exact lattice value N*(N+1) steps
    2. the simulated S(t) lies on the analytic S(t)
    3. the histogram of times lies on the analytic f(t)
"""

import numpy as np
import matplotlib.pyplot as plt

import lattice
import walker


def mu(n, L):
    """The n-th eigenvalue."""
    return (2*n + 1) * np.pi / (2*L)


def survival(t, L, n_modes=200):
    """S(t), the probability the walker is still alive at time t."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros(t.size)
    for n in range(n_modes):
        m = mu(n, L)
        weight = 4 * (-1)**n / ((2*n + 1) * np.pi)
        out = out + weight * np.exp(-lattice.D * m**2 * t)
    return out


def fpt_density(t, L, n_modes=200):
    """f(t) = -dS/dt, the first-passage time density."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros(t.size)
    for n in range(n_modes):
        m = mu(n, L)
        weight = 4 * (-1)**n / ((2*n + 1) * np.pi)
        out = out + weight * lattice.D * m**2 * np.exp(-lattice.D * m**2 * t)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    N = 200
    n_walkers = 100000
    L = lattice.length(N)

    steps = walker.run_many(N, n_walkers, rng)
    times = lattice.to_time(steps)

    # --- check 1: the mean -------------------------------------------------
    exact_steps = N * (N + 1)
    print(f"N = {N},  L = {L},  {n_walkers} walkers\n")
    print("mean first-passage time")
    print(f"   simulated          = {times.mean():12.2f}")
    print(f"   exact lattice      = {lattice.to_time(exact_steps):12.2f}"
          f"    (N*(N+1) steps)")
    print(f"   continuum L^2/(2D) = {lattice.mfpt_segment(N):12.2f}")
    print(f"   sim / lattice      = {times.mean()/lattice.to_time(exact_steps):12.4f}")

    # --- check 2 and 3: the whole distribution -----------------------------
    tmax = 5 * lattice.mfpt_segment(N)
    tgrid = np.linspace(tmax/400, tmax, 400)

    sorted_times = np.sort(times)
    S_sim = 1.0 - np.searchsorted(sorted_times, tgrid, side="right") / n_walkers
    S_exact = survival(tgrid, L)

    print(f"\nsurvival curve")
    print(f"   max |S_sim - S_exact| = {np.max(np.abs(S_sim - S_exact)):.4f}")

    # --- figure ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(times, bins=120, range=(0, tmax), density=True,
             color="#8fc7c2", label=f"simulation ({n_walkers} walkers)")
    ax1.plot(tgrid, fpt_density(tgrid, L), color="#e08214", lw=2,
             label="analytic  f(t)")
    ax1.axvline(lattice.mfpt_segment(N), color="k", ls="--", lw=1,
                label=r"$L^2/2D$")
    ax1.set_xlabel("first-passage time  t")
    ax1.set_ylabel("probability density  f(t)")
    ax1.set_title("First-passage time distribution")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.plot(tgrid, S_sim, color="#2b857f", lw=2, label="simulation")
    ax2.plot(tgrid, S_exact, color="#e08214", lw=2, ls="--", label="analytic  S(t)")
    ax2.set_xlabel("time  t")
    ax2.set_ylabel("survival  S(t)")
    ax2.set_title("Survival probability")
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("test_segment.png", dpi=150)
    print("\nwrote test_segment.png")
