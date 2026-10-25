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

  split = "flux"   the trunk carries v, each branch carries v/b.
                   Volumetric flux is conserved if every edge has the same
                   cross-section. This is the incompressible guess.

  split = "equal"  every branch carries the same v as the trunk. Simpler,
                   but b branches each carrying v cannot be fed by one
                   trunk carrying v, so flux is NOT conserved.

The branching choice AT the vertex is left uniform over the b+1 edges,
i.e. the Kirchhoff rule from pure diffusion. Whether a current should also
bias that choice is part of the same open question.

Pe is quoted for the TRUNK: Pe = L*v/D with L = N*a, so eps_trunk = Pe/(2N).
"""

import numpy as np
import lattice

TRUNK = 0


def branch_eps(eps_trunk, b, split):
    if split == "flux":
        return eps_trunk / b
    elif split == "equal":
        return eps_trunk
    raise ValueError("split must be 'flux' or 'equal'")


def run_many(b, N, n_walkers, rng, Pe=0.0, split="flux"):
    """
    Release at the vertex, walk until absorbed on a branch tip.
    Returns (touched_trunk, steps).
    """
    eps_t = Pe / (2.0 * N)
    eps_b = branch_eps(eps_t, b, split)
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

    N = 60
    n_walkers = 4000
    b_values = [1, 2, 3, 5, 10, 20]

    for split in ("flux", "equal"):
        print(f"\nsplit = {split!r},  N = {N},  {n_walkers} walkers")
        print("mean first-passage time\n")
        print(f"  {'Pe':>5} " + " ".join(f"{('b='+str(b)):>9}"
                                         for b in b_values))
        for Pe in (0.0, 10.0, 40.0):
            row = []
            for b in b_values:
                _, st = run_many(b, N, n_walkers, rng, Pe=Pe, split=split)
                row.append(lattice.to_time(st).mean())
            print(f"  {Pe:>5.0f} " + " ".join(f"{v:>9.0f}" for v in row))