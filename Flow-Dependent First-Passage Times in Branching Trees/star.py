"""
Lattice walk on a star network: one trunk plus b identical branches.
Every edge has N spacings (a = 1/N, L = D = 1).

    trunk     site 0 = root, REFLECTING (bounce to site 1)
              site N = junction
    branch k  site 0 = junction (never stored, see below)
              site N = tip, ABSORBING

A walker AT the junction is always stored as (edge 0, site N).

Junction rule (derived in the junction note): step onto edge k with
probability A_k / sum_j A_j, i.e.

    P(trunk)       = 1 / (1 + beta)
    P(each branch) = (beta / b) / (1 + beta),     beta = b A_1 / A_0

Flow points from the root toward the tips. Trunk velocity Pe, branch
velocity Pe / beta (flux conservation A_0 v_0 = b A_1 v_1).

    equal areas    beta = b
    Murray's law   beta = b**(1/3)
"""

import numpy as np
from units import eps, steps_to_time

TRUNK = 0


def run_star(b, N, n_walkers, rng, Pe=0.0, beta=None):
    """
    Release n_walkers at the root; walk until each is absorbed at a tip.

    Returns (T, branch):
        T       absorption time of each walker, in units of tau = L^2/D
        branch  which branch absorbed it, 1..b
    """
    if beta is None:
        beta = float(b) # default: equal areas
    e_trunk = eps(Pe, N)
    e_branch = e_trunk / beta
    p_plus_trunk = 0.5 * (1 + e_trunk)
    p_plus_branch = 0.5 * (1 + e_branch)

    p_edge = np.full(b + 1, (beta / b) / (1 + beta))
    p_edge[TRUNK] = 1 / (1 + beta)

    edge = np.zeros(n_walkers, dtype=int)       # everyone starts on the trunk
    site = np.zeros(n_walkers, dtype=int)       # ... at the root
    steps = np.zeros(n_walkers, dtype=int)
    alive = np.ones(n_walkers, dtype=bool)

    while alive.any():
        live = np.where(alive)[0]
        steps[live] = steps[live] + 1

        # sort live walkers by where they are BEFORE anyone moves
        on_trunk = edge[live] == TRUNK
        at_root = live[on_trunk & (site[live] == 0)]
        at_junction = live[on_trunk & (site[live] == N)]
        moving = live[~np.isin(live, at_root) & ~np.isin(live, at_junction)]

        # reflecting root: bounce
        site[at_root] = 1

        # junction: area-weighted choice of edge
        if at_junction.size:
            k = rng.choice(b + 1, size=at_junction.size, p=p_edge)
            back = at_junction[k == TRUNK]
            out = at_junction[k != TRUNK]
            site[back] = N - 1
            edge[out] = k[k != TRUNK]
            site[out] = 1

        # interior sites: biased coin, bias depends on the edge
        if moving.size:
            p_plus = np.where(edge[moving] == TRUNK,
                              p_plus_trunk, p_plus_branch)
            u = rng.random(moving.size)
            site[moving] = site[moving] + np.where(u < p_plus, 1, -1)

            # a branch walker that stepped back to site 0 is at the junction
            returned = moving[(edge[moving] != TRUNK) & (site[moving] == 0)]
            edge[returned] = TRUNK
            site[returned] = N

            # absorbed at a tip
            absorbed = moving[(edge[moving] != TRUNK) & (site[moving] == N)]
            alive[absorbed] = False

    return steps_to_time(steps, N), edge