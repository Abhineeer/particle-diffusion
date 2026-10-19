"""
star.py

A random walk on the star graph: one trunk plus b branches.

    edge 0        trunk,    far end REFLECTING
    edge 1..b     branches, far end ABSORBING

Each edge is a chain of sites n = 0, 1, ..., N.
    n = 0   is the vertex, SHARED by all b+1 edges
    n = N   is the far end of that edge

The vertex rule is the one forced by Kirchhoff: a walker standing on
the vertex steps onto a uniformly chosen edge, each with probability
1/(b+1). Note the trunk is one of the choices, so it is 1/(b+1), not
1/b.

Release point is a parameter because it changes what you measure:
    "vertex"    n = 0 on the trunk. This is the start for which
                h_V = 1/(1+b) is defined.
    "trunk_tip" n = N on the trunk, i.e. at the reflecting wall.
"""

import numpy as np

TRUNK = 0


def run_many(b, N, n_walkers, rng, start="vertex"):
    """
    Release n_walkers and walk them until each is absorbed on a branch tip.

    Returns three arrays, one entry per walker:
        branch         which branch caught it, 1..b
        touched_trunk  True if it ever reached the reflecting trunk tip
        steps          how many steps it took
    """
    edge = np.zeros(n_walkers, dtype=int)
    if start == "vertex":
        n = np.zeros(n_walkers, dtype=int)
    elif start == "trunk_tip":
        n = np.full(n_walkers, N, dtype=int)
    else:
        raise ValueError("start must be 'vertex' or 'trunk_tip'")

    branch = np.zeros(n_walkers, dtype=int)
    touched_trunk = (n == N) & (edge == TRUNK)
    steps = np.zeros(n_walkers, dtype=int)
    alive = np.ones(n_walkers, dtype=bool)

    while alive.any():
        live = np.where(alive)[0]
        steps[live] = steps[live] + 1

        # Split the live walkers BEFORE moving any of them. If we picked out
        # the vertex walkers first and moved them to n = 1, they would then
        # look like ordinary walkers and get a second move in the same
        # iteration, while only one step was counted.
        at_vertex = live[n[live] == 0]
        moving = live[n[live] != 0]

        # --- walkers standing on the vertex pick an edge --------------------
        if at_vertex.size:
            edge[at_vertex] = rng.integers(0, b + 1, size=at_vertex.size)
            n[at_vertex] = 1

        # --- everyone else flips a coin ------------------------------------
        if moving.size:
            coin = rng.integers(0, 2, size=moving.size) * 2 - 1
            n[moving] = n[moving] + coin

            # trunk tip reflects: a step past the end is rejected
            off_end = moving[(edge[moving] == TRUNK) & (n[moving] > N)]
            n[off_end] = N

            # record the touch and kill the absorbed
            on_trunk_tip = moving[(edge[moving] == TRUNK) & (n[moving] == N)]
            touched_trunk[on_trunk_tip] = True

            absorbed = moving[(edge[moving] != TRUNK) & (n[moving] == N)]
            branch[absorbed] = edge[absorbed]
            alive[absorbed] = False

    return branch, touched_trunk, steps


# --------------------------------------------------------------------------
# exact lattice answers, for the test to compare against
# --------------------------------------------------------------------------
def h_vertex_exact(b):
    """
    Probability of reaching the trunk tip before absorption, from the vertex.

    h is harmonic, so linear along each edge:
        trunk    h(n) = h_V + (1 - h_V) * n/N
        branch   h(n) = h_V * (1 - n/N)
    and at the vertex h_V is the average over the b+1 neighbours:
        h_V = [ h_trunk(1) + b * h_branch(1) ] / (b+1)
    Substituting and simplifying, every N cancels:

        h_V = 1 / (1 + b)

    exactly, with no lattice correction at all.
    """
    return 1.0 / (1.0 + b)


def mfpt_vertex_exact(b, N):
    """
    Mean steps to absorption starting from the vertex.

    T(n) = -n^2 + A n + T_V on each edge. The reflecting trunk tip gives
    A = 2N + 1; the absorbing branch tip pins the branch constant. Averaging
    over the b+1 neighbours at the vertex and solving:

        T_V = N * (1 + 2N + b*N) / b   steps

    For b = 1 this is N(3N + 1), which is the exact answer for a plain
    segment of length 2N started at its midpoint. Good consistency check.

    In physical time this is L^2 * (1/2 + 1/b) / D to leading order.
    """
    return N * (1 + 2*N + b*N) / b


def mfpt_trunktip_exact(b, N):
    """Mean steps starting from the reflecting trunk tip: N^2 + N + T_V."""
    return N**2 + N + mfpt_vertex_exact(b, N)