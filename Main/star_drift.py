"""
star_drift.py

The star walk with a current flowing from the trunk out through the
branches.

    edge 0      trunk,    reflecting at n = N, flow points toward n = 0
    edge 1..b   branches, absorbing  at n = N, flow points toward n = N

so the current carries a particle down the trunk, through the vertex, and
out a branch.

THE VERTEX RULE IS AN OPEN QUESTION
-----------------------------------
How a current divides at a junction is exactly what has not been settled
yet, so it is a parameter here, not a hardcoded choice:

Volumetric flux conservation at the vertex says

    A_t * v_t  =  b * A_1 * v_1      ->      v_1 = v_t / beta

with

    beta = b * A_1 / A_t

the ratio of total branch cross-section to trunk cross-section. beta is
the ONE parameter, and it is a geometric fact about the vessels, not a
modelling choice between two rules:

    beta = b   every branch has the SAME cross-section as the trunk, so
               the total area grows b-fold and each branch flows at v/b.
    beta = 1   the branch areas SUM to the trunk area, so each branch
               flows at the same v as the trunk. This is what 3D flow
               work finds outside special geometries.

Both conserve flux. They differ only in how the cross-section is divided,
which is measurable from vessel data.

The branching choice AT the vertex is left uniform over the b+1 edges,
i.e. the Kirchhoff rule from pure diffusion. Whether a current should also
bias that choice is a separate open question.

UNITS
-----
L = 1 and D = 1, so Pe = L*v/D = v exactly: the Peclet number IS the flow
speed. N is the resolution of each edge, and the coin bias is

    eps = v * a / (2D) = Pe / (2N)
"""

import numpy as np
import lattice

TRUNK = 0


def run_many(b, N, n_walkers, rng, Pe=0.0, beta=None):
    """
    Release at the vertex, walk until absorbed on a branch tip.

    Pe   the trunk Peclet number, which with L = D = 1 is just v.
    beta total branch area over trunk area. Defaults to b, meaning every
         branch has the same cross-section as the trunk. beta = 1 means
         the branch areas sum to the trunk area.

    Returns (touched_trunk, steps).
    """
    if beta is None:
        beta = float(b)
    eps_t = Pe / (2.0 * N)
    eps_b = eps_t / beta
    if max(eps_t, eps_b) >= 1.0:
        raise ValueError(f"bias too large at N = {N}; raise N")

    # on the trunk the flow points toward n = 0, so the favoured step is -1
    p_down_trunk = 0.5 * (1.0 + eps_t)
    # on a branch the flow points toward n = N, so the favoured step is +1
    p_out_branch = 0.5 * (1.0 + eps_b)

    edge = np.zeros(n_walkers, dtype=int)
    n = np.zeros(n_walkers, dtype=int)
    touched_trunk = np.zeros(n_walkers, dtype=bool)
    steps = np.zeros(n_walkers, dtype=int)
    alive = np.ones(n_walkers, dtype=bool)

    while alive.any():
        live = np.where(alive)[0]
        steps[live] += 1

        # split BEFORE moving anyone, or the vertex walkers get two moves
        at_vertex = live[n[live] == 0]
        moving = live[n[live] != 0]

        if at_vertex.size:
            edge[at_vertex] = rng.integers(0, b + 1, size=at_vertex.size)
            n[at_vertex] = 1

        if moving.size:
            on_trunk = edge[moving] == TRUNK
            u = rng.random(moving.size)
            # trunk: favoured direction is -1; branch: favoured is +1
            coin = np.where(on_trunk,
                            np.where(u < p_down_trunk, -1, 1),
                            np.where(u < p_out_branch, 1, -1))
            n[moving] = n[moving] + coin

            off_end = moving[on_trunk & (n[moving] > N)]
            n[off_end] = N

            hit_wall = moving[on_trunk & (n[moving] == N)]
            touched_trunk[hit_wall] = True

            absorbed = moving[~on_trunk & (n[moving] == N)]
            alive[absorbed] = False

    return touched_trunk, steps


if __name__ == "__main__":
    rng = np.random.default_rng(404)

    N, n_walkers = 50, 3000
    b_values = [1, 2, 3, 5, 10]

    for beta_name, beta_fn in (("beta = b  (equal cross-sections)",
                                lambda b: float(b)),
                               ("beta = 1  (areas sum to the trunk)",
                                lambda b: 1.0)):
        print(f"\n{beta_name}   N = {N}, {n_walkers} walkers")
        print("mean first-passage time\n")
        print(f"  {'Pe':>5} " + " ".join(f"{('b='+str(b)):>8}"
                                         for b in b_values))
        for Pe in (0.0, 10.0, 40.0):
            row = []
            for b in b_values:
                _, st = run_many(b, N, n_walkers, rng, Pe=Pe,
                                 beta=beta_fn(b))
                row.append(lattice.to_time(st, N).mean())
            print(f"  {Pe:>5.0f} " + " ".join(f"{v:>8.3f}" for v in row))