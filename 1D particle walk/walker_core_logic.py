"""
walker_core_logic.py

One particle doing a random walk on the star graph.
Same convention as network_2branches_fd.py:
    edge 0        = trunk    (far end reflecting)
    edge 1..b     = branches (far end absorbing)
Each edge is a chain of sites n = 0, 1, ..., N.
    n = 0  is the vertex (shared by all edges)
    n = N  is the far end

The release point is a parameter, because it changes what you can measure:
    start="vertex"    -> release at the vertex (n=0). This is the start you
                         need to measure h_V = 1/(1+b).
    start="trunk_tip" -> release one site below the reflecting trunk tip
                         (n=N-1). Good for long, story-like trajectories.
"""

import random

TRUNK = 0


def one_step(edge, n, b, N):
    """
    Take the walker from state (edge, n) to its next state.

    Returns (edge, n, absorbed).
    'absorbed' is True only when the walker just died on a branch tip;
    in that case 'edge' tells you which branch caught it.
    """

    # --- at the vertex: pick an edge uniformly, step onto its first site ---
    if n == 0:
        edge = random.randint(0, b)   # 0 = trunk, 1..b = branches, all equally likely
        n = 1
        return edge, n, False

    # --- at a far end ---
    if n == N:
        if edge == TRUNK:
            # reflecting: the attempted step off the end is turned back
            n = N - 1
            return edge, n, False
        else:
            # absorbing branch tip: the walker dies here
            return edge, n, True

    # --- interior: fair coin, step left or right ---
    if random.random() < 0.5:
        n = n - 1
    else:
        n = n + 1
    return edge, n, False


def start_state(start, N):
    """Return the (edge, n) the walker begins in."""
    if start == "vertex":
        return TRUNK, 0
    elif start == "trunk_tip":
        return TRUNK, N - 1
    else:
        raise ValueError("start must be 'vertex' or 'trunk_tip'")


def run_one_walk(b, N, start="trunk_tip", max_steps=5_000_000):
    """
    Run a single particle from `start` until it is absorbed.

    Returns a dict with:
        branch         which branch caught it (1..b)
        steps          how many steps it took
        trunk_touches  how many times it hit the reflecting trunk tip
    """
    edge, n = start_state(start, N)
    trunk_touches = 0

    for step in range(1, max_steps + 1):
        # count a trunk touch before the reflection turns it back
        if edge == TRUNK and n == N:
            trunk_touches += 1

        edge, n, absorbed = one_step(edge, n, b, N)

        if absorbed:
            return {"branch": edge, "steps": step, "trunk_touches": trunk_touches}

    # if we ever get here the walk never finished -- raise max_steps
    return {"branch": None, "steps": max_steps, "trunk_touches": trunk_touches}


if __name__ == "__main__":
    b = 5
    N = 500

    print(f"a few walks from the vertex (b={b}, N={N})\n")
    for i in range(100):
        r = run_one_walk(b, N, start="vertex")
        print(f"  walk {i}: branch {r['branch']:>2}, {r['steps']:>7} steps, "
              f"trunk touches {r['trunk_touches']:>3}")