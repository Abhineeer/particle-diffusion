"""
walker_drift.py

The same single-segment walk as walker.py, but with a drift.

    n = 0   reflecting, release point
    n = N   absorbing
    flow pushes TOWARD the absorbing end

THE BIAS
--------
The only thing a lattice walk can change is the coin:

    p(+1) = (1 + eps)/2        p(-1) = (1 - eps)/2

Mean displacement per step is a*eps, and one step takes dt = a^2/(2D), so

    v = a*eps / dt = 2*D*eps / a      ->      eps = v*a / (2D)

and in terms of the Peclet number Pe = L*v/D with L = N*a,

    Pe = 2*N*eps        ->      eps = Pe / (2N)

TWO WARNINGS
------------
1. eps <= 1 is a hard ceiling: you cannot bias a coin past certainty.
   Since eps = Pe/(2N), a target Pe needs N >= Pe/2. High Peclet is
   bought with MORE SITES, not a stronger bias.

2. The variance per step is a^2 * (1 - eps^2), not a^2, so the effective
   diffusion coefficient is D*(1 - eps^2). At eps = 0.1 that is a 1%
   correction; at eps = 0.5 it is 25% and the Pe you asked for is not the
   Pe you got. Keep eps small and raise N instead.
"""

import numpy as np
import lattice


def eps_from_pe(Pe, N):
    """Coin bias for a target Peclet number on an N-site segment."""
    eps = Pe / (2.0 * N)
    if eps >= 1.0:
        raise ValueError(f"Pe = {Pe} needs eps = {eps:.2f} >= 1 at N = {N}. "
                         f"Use N >= {int(np.ceil(Pe/2))}.")
    return eps


def run_many(N, n_walkers, rng, Pe=0.0):
    """
    Release n_walkers at the reflecting end n = 0, walk until absorbed at
    n = N, with flow pushing toward the absorbing end.

    Pe = 0 reproduces walker.run_many exactly.
    Returns an array of step counts.
    """
    eps = eps_from_pe(Pe, N)
    p_right = 0.5 * (1.0 + eps)

    n = np.zeros(n_walkers, dtype=int)
    steps = np.zeros(n_walkers, dtype=int)
    alive = np.ones(n_walkers, dtype=bool)

    while alive.any():
        live = np.where(alive)[0]

        coin = np.where(rng.random(live.size) < p_right, 1, -1)
        n[live] = n[live] + coin

        n[n < 0] = 0                 # reflecting end, the step is rejected

        steps[live] = steps[live] + 1
        alive[n >= N] = False        # absorbing end

    return steps


def tau_advection(N, Pe):
    """L / v, the time to be carried the length of the segment."""
    L = lattice.length(N)
    v = Pe * lattice.D / L
    return L / v


def tau_diffusion(N):
    """L^2 / D."""
    return lattice.length(N)**2 / lattice.D


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    N = 200
    n_walkers = 20000

    print(f"N = {N},  L = {lattice.length(N):.0f},  {n_walkers} walkers\n")
    print(f"  {'Pe':>6} {'eps':>8} {'mean t':>11} {'tau_adv':>10} "
          f"{'tau_diff':>10} {'spread/mean':>12}")

    for Pe in (0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0):
        t = lattice.to_time(run_many(N, n_walkers, rng, Pe=Pe))
        eps = eps_from_pe(Pe, N)
        ta = tau_advection(N, Pe) if Pe > 0 else float("inf")
        print(f"  {Pe:>6.0f} {eps:>8.4f} {t.mean():>11.1f} "
              f"{ta:>10.1f} {tau_diffusion(N):>10.1f} "
              f"{t.std()/t.mean():>12.3f}")

    print("\n  as Pe grows the mean tracks tau_adv = L/v and the relative")
    print("  spread collapses: the walk stops being a random search and")
    print("  becomes a ride.")