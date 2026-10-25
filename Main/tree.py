"""
tree.py

A random walk on a rooted tree.

Geometry, going downward from the root:

    a TRUNK of N_0 spacings. Its far end (the root tip) is REFLECTING.
    at the bottom of the trunk sits the first vertex.
    each vertex has b children edges, of N_1 spacings at generation 1,
    N_2 at generation 2, and so on.
    at the deepest generation the far ends are ABSORBING.

depth = 1 is exactly the star graph, so star.py is the test case.

Edge lengths shrink by a fixed ratio:

    N_g = round(N_0 * rho^g)

rho is a parameter, never hardcoded. Self-similarity needs it fixed, and
once advection is switched on it is the knob that decides which generation
the advection-to-diffusion crossover lands in.

HOW THE WALK IS DONE
--------------------
The tree is flattened into a list of sites, each with a list of
neighbours. A walker steps to a uniformly chosen neighbour. That one rule
covers every case:

    interior site      2 neighbours   -> fair coin
    vertex             b+1 neighbours -> uniform branching, which is the
                                         Kirchhoff rule
    reflecting tip     neighbour list is [the site below, ITSELF], so half
                       the time the walker stays put. That is the rejected
                       step, and it costs a step of time.
    absorbing tip      walker is removed on arrival

Writing the reflecting tip as a self-loop is what keeps a single uniform
rule correct everywhere.
"""

import numpy as np


def edge_lengths(N0, depth, rho):
    """N_g for g = 0 (trunk) up to g = depth. At least 1 spacing each."""
    return [max(1, int(round(N0 * rho**g))) for g in range(depth + 1)]


def build_tree(b, depth, N0, rho):
    """
    Flatten the tree into neighbour lists.

    Returns
        nbr        int array (n_sites, max_degree), padded with -1
        degree     int array (n_sites,)
        absorbing  bool array (n_sites,)
        gen        int array (n_sites,), which generation each site is on
        root_tip   index of the reflecting site
        first_vertex  index of the vertex at the bottom of the trunk
        lengths    the N_g list

    `gen` is what makes the screening question measurable: a site on a
    generation-2 edge is labelled 2, so recording where a walker died tells
    us how deep the absorption actually reached.
    """
    lengths = edge_lengths(N0, depth, rho)
    neighbours = []          # list of lists, grown as we go
    absorbing = []
    gen = []

    def new_site(is_absorbing=False, g=0):
        neighbours.append([])
        absorbing.append(is_absorbing)
        gen.append(g)
        return len(neighbours) - 1

    def link(i, j):
        neighbours[i].append(j)
        neighbours[j].append(i)

    def add_chain(start, n_spacings, end_absorbing, g):
        """Attach a chain of n_spacings to `start`; return the far end."""
        prev = start
        for k in range(n_spacings):
            last = (k == n_spacings - 1)
            s = new_site(is_absorbing=(last and end_absorbing), g=g)
            link(prev, s)
            prev = s
        return prev

    # the trunk: reflecting tip at the top, first vertex at the bottom
    root_tip = new_site(g=0)
    first_vertex = add_chain(root_tip, lengths[0], end_absorbing=False, g=0)

    # the reflecting tip is a self-loop so that a uniform choice over its
    # two entries gives "stay put" half the time
    neighbours[root_tip].append(root_tip)

    # grow the generations
    def grow(vertex, g):
        if g > depth:
            return
        for _ in range(b):
            end = add_chain(vertex, lengths[g],
                            end_absorbing=(g == depth), g=g)
            if g < depth:
                grow(end, g + 1)

    grow(first_vertex, 1)

    n_sites = len(neighbours)
    degree = np.array([len(v) for v in neighbours], dtype=int)
    max_deg = degree.max()
    nbr = np.full((n_sites, max_deg), -1, dtype=int)
    for i, v in enumerate(neighbours):
        nbr[i, :len(v)] = v

    return (nbr, degree, np.array(absorbing), np.array(gen),
            root_tip, first_vertex, lengths)


def run_many(nbr, degree, absorbing, gen, root_tip, start_site,
             n_walkers, rng):
    """
    Release n_walkers from start_site and walk until absorbed.

    Returns, one entry per walker:
        steps         how many steps it took
        died_gen      the generation of the tip that absorbed it
        max_gen       the deepest generation it ever reached
        touched_root  True if it ever reached the reflecting root tip

    died_gen is the screening measurement. max_gen says how far the walker
    explored whether or not it died there, so the two together separate
    "never got down there" from "got down there and came back".
    """
    site = np.full(n_walkers, start_site, dtype=int)
    steps = np.zeros(n_walkers, dtype=int)
    max_gen = gen[site].copy()
    touched_root = (site == root_tip)
    alive = ~absorbing[site].copy()

    while alive.any():
        live = np.where(alive)[0]
        steps[live] += 1

        d = degree[site[live]]
        pick = (rng.random(live.size) * d).astype(int)
        site[live] = nbr[site[live], pick]

        here = site[live]
        max_gen[live] = np.maximum(max_gen[live], gen[here])
        touched_root[live] |= (here == root_tip)
        alive[live] = ~absorbing[here]

    return steps, gen[site], max_gen, touched_root


if __name__ == "__main__":
    rng = np.random.default_rng(5)

    # depth 1 is the star: this must match star.py's exact formula
    import star

    b, N = 3, 40
    nbr, deg, absorb, gen, root_tip, v1, lengths = build_tree(b, 1, N, 1.0)

    print(f"depth-1 tree with b = {b}, N = {N}")
    print(f"   sites = {len(deg)}, lengths = {lengths}")
    print(f"   vertex degree = {deg[v1]}  (should be b+1 = {b+1})")
    print(f"   root tip degree = {deg[root_tip]}  (should be 2)")

    steps, died, maxg, touched = run_many(nbr, deg, absorb, gen, root_tip,
                                          v1, 40000, rng)
    exact = star.mfpt_vertex_exact(b, N)
    print(f"\n   mean steps from the vertex = {steps.mean():10.1f}")
    print(f"   star.py exact formula      = {exact:10.1f}")
    print(f"   ratio                      = {steps.mean()/exact:10.4f}")
    print(f"\n   touched the root tip = {touched.mean():.4f}"
          f"   (star h_V = 1/(1+b) = {1/(1+b):.4f})")

    # --------------------------------------------------------------- deeper
    print("\n\nWHERE DOES THE WALKER ACTUALLY DIE?")
    print("released at the first vertex, b = 3, N0 = 40, rho = 0.5\n")
    print(f"  {'depth':>6} {'tips':>6} {'mean steps':>12} "
          f"{'absorbed by generation':>34}")
    for depth in (1, 2, 3, 4):
        nbr, deg, absorb, gen, root_tip, v1, lengths = build_tree(
            3, depth, 40, 0.5)
        steps, died, maxg, touched = run_many(nbr, deg, absorb, gen,
                                              root_tip, v1, 20000, rng)
        frac = [(died == g).mean() for g in range(1, depth + 1)]
        frac_str = " ".join(f"g{g}:{f:.3f}" for g, f in
                            zip(range(1, depth + 1), frac))
        print(f"  {depth:>6} {3**depth:>6} {steps.mean():>12.1f}   {frac_str}")

    print("\n  every walker dies at the DEEPEST generation, because only")
    print("  those tips absorb. The screening question is not where it dies")
    print("  but how much the extra depth costs in time.")