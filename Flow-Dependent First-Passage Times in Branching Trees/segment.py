"""

One edge on a lattice of sites 0, 1, ..., N  (spacing a = 1/N).

    site 0   the root, REFLECTING: a walker here always steps to site 1
    site N   the tip, ABSORBING: a walker is removed on arrival

Flow points from the root toward the tip. Walkers start at the root.

Reflection is a bounce, not a stay. Staying put wastes a step at the wall
and biases <T> by 1/N (2% at N = 50). Bouncing gives exactly N^2 steps,
i.e. <T> = 1/2, at Pe = 0.

"""

import numpy as np
from units import eps, steps_to_time

def run_segment(N, n_walkers, rng, Pe=0.0):
    # Release n_walkers at the root; return absorption times in units of tau.
    e = eps(Pe, N)
    p_plus = 0.5 * (1 + e)# probability of stepping toward the tip

    site = np.zeros(n_walkers, dtype=int)  # every walker starts at 0
    steps = np.zeros(n_walkers, dtype=int) # steps taken so far
    alive = np.ones(n_walkers, dtype=int) # True until absorbed

    while alive.any():
        live = np.where(alive)[0] # indices of walers still moving

        u = rng.random(live.size)
        coin = np.where(u < p_plus, 1, -1)
        coin[site[live] == 0] = 1 # reflecting root: always step

        site[live] = site[live] + coin
        steps[live] = steps[live] + 1

        absorbed = live[site[live] == N] # walkers that just reached the tip
        alive[absorbed] = False

    return steps_to_time(steps, N)

