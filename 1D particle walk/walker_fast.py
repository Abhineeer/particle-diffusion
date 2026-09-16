"""
walker_fast.py

Same walk as walker_core_logic.py, same conventions (edge 0 = trunk reflecting,
edges 1..b = branches absorbing; sites n = 0..N, n=0 vertex, n=N far end).
The only difference: every walker is stepped at once with numpy, so an
ensemble of tens of thousands finishes in a second or two instead of a minute.

The release point is a parameter, matching walker_core_logic.py:
    start="vertex"    -> n=0    (needed to measure h_V = 1/(1+b))
    start="trunk_tip" -> n=N-1  (long trajectories)

Read walker_core_logic.py to understand the rules; read this to run them fast.
"""

import numpy as np

TRUNK = 0


def start_n(start, N, n_walkers):
    """Return the initial n array for every walker."""
    if start == "vertex":
        return np.zeros(n_walkers, dtype=int)
    elif start == "trunk_tip":
        return np.full(n_walkers, N - 1, dtype=int)
    else:
        raise ValueError("start must be 'vertex' or 'trunk_tip'")


def run_ensemble_fast(b, N, n_walkers, rng, start="vertex"):
    """
    Release n_walkers from `start`, walk them all until absorbed.

    Returns:
        branch          array (n_walkers,) of which branch caught each walker
        reached_trunk   bool array, True if the walker ever hit the trunk tip
        steps           array of absorption time (in steps) per walker
    """
    edge = np.zeros(n_walkers, dtype=int)          # all start on the trunk
    n = start_n(start, N, n_walkers)
    reached_trunk = np.zeros(n_walkers, dtype=bool)
    branch = np.full(n_walkers, -1, dtype=int)
    steps = np.zeros(n_walkers, dtype=int)
    alive = np.ones(n_walkers, dtype=bool)

    max_steps = 500 * N * N
    for step in range(1, max_steps + 1):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        steps[idx] = step

        at_vertex = n[idx] == 0
        at_tip = n[idx] == N

        # --- vertex: pick an edge uniformly, land on site 1 ---
        v = idx[at_vertex]
        if v.size:
            edge[v] = rng.integers(0, b + 1, size=v.size)
            n[v] = 1

        # --- far ends ---
        tip = idx[at_tip]
        if tip.size:
            trunk_tip = tip[edge[tip] == TRUNK]
            branch_tip = tip[edge[tip] != TRUNK]
            # trunk tip: record the touch, reflect back
            reached_trunk[trunk_tip] = True
            n[trunk_tip] = N - 1
            # branch tip: die here
            branch[branch_tip] = edge[branch_tip]
            alive[branch_tip] = False

        # --- interior: fair coin, +/- 1 ---
        interior = idx[~at_vertex & ~at_tip]
        if interior.size:
            coin = rng.integers(0, 2, size=interior.size) * 2 - 1  # -1 or +1
            n[interior] += coin

    return branch, reached_trunk, steps