"""
walker.py

A random walk on ONE segment.

    sites      n = 0, 1, ..., N
    n = 0      reflecting end  (the walker is released here)
    n = N      absorbing end   (the walker dies here)

The two boundary rules are the ones that come out of the master equation:

    reflecting: a step off the end is rejected and the walker stays put.
                It still costs one step of time.
    absorbing:  the walker is removed and we record the step count.

The walk itself counts STEPS and knows nothing about units. N is the
resolution of the lattice, not the length of the segment: converting to
physical time is lattice.to_time(steps, N), which needs N because
dt = L^2/(2*D*N^2).

Exact lattice answer, for checking: from the reflecting end the mean is
N*(N+1) steps. Writing T(n) for the mean steps left from site n,

    interior:   T(n) = 1 + (T(n-1) + T(n+1))/2
    reflecting: T(0) = 1 + (T(0) + T(1))/2   ->   T(0) = 2 + T(1)
    absorbing:  T(N) = 0

solves to T(n) = N(N+1) - n(n+1), so T(0) = N*(N+1). In time that is
(L^2/2D)*(1 + 1/N), and the 1/N is the lattice being coarse.
"""

import numpy as np
import lattice


def one_walk(N, rng):
    """
    Walk a single particle from n = 0 until it is absorbed at n = N.
    Returns the number of steps taken.

    Slow, but readable. Use it to check that run_many does the same thing.
    """
    n = 0
    steps = 0

    while n < N:
        if rng.random() < 0.5:
            n = n - 1
        else:
            n = n + 1

        if n < 0:        # reflecting end: the step is rejected
            n = 0

        steps = steps + 1

    return steps


def run_many(N, n_walkers, rng):
    """
    Same walk, but all walkers are stepped together with numpy.
    Returns an array of step counts, one per walker.
    """
    n = np.zeros(n_walkers, dtype=int)       # everyone starts at the wall
    steps = np.zeros(n_walkers, dtype=int)
    alive = np.ones(n_walkers, dtype=bool)

    while alive.any():
        live = np.where(alive)[0]

        coin = rng.integers(0, 2, size=live.size) * 2 - 1    # -1 or +1
        n[live] = n[live] + coin

        n[n < 0] = 0                 # reflecting end

        steps[live] = steps[live] + 1

        alive[n >= N] = False        # absorbing end

    return steps


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_walkers = 20000

    print(f"{n_walkers} walkers per row,  L = {lattice.L_DEFAULT}, "
          f"D = {lattice.D}\n")
    print(f"  {'N':>5} {'mean steps':>12} {'exact N(N+1)':>14} "
          f"{'mean time':>11} {'lattice <t>':>12} {'continuum':>11}")

    for N in (25, 50, 100, 200):
        steps = run_many(N, n_walkers, rng)
        mean_time = lattice.to_time(steps.mean(), N)
        print(f"  {N:>5} {steps.mean():>12.1f} {N*(N+1):>14d} "
              f"{mean_time:>11.5f} {lattice.mfpt_segment_lattice(N):>12.5f} "
              f"{lattice.mfpt_segment(N):>11.5f}")

    print("\n  compare the simulation against the LATTICE column, not the")
    print("  continuum one: the 1/N gap is real and expected.")

    slow = np.array([one_walk(50, rng) for _ in range(2000)])
    print(f"\n  N=50 slow version: {slow.mean():.1f} steps  (exact {50*51})")