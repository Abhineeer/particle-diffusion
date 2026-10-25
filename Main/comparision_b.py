"""
Compare first-passage time distributions across BRANCH COUNT b.

THE QUESTION FROM THE MEETING
-----------------------------
"If I already sampled the time, how do I get a different b from it?"

The answer is that you cannot, and that is the point. b and L are
different kinds of parameter:

    LENGTH rescales. Pure diffusion has no intrinsic length, so the
    time law at L_2 is the law at L_1 stretched by (L_2/L_1)^2. One
    sample covers every length. (That is comparison.py.)

    BRANCH COUNT does not rescale. b changes what happens AT the vertex:
    a walker standing there picks one of b+1 edges, so changing b changes
    the splitting probability and the whole excursion structure. No
    stretching of the time axis turns a b = 2 walk into a b = 5 walk.
    You re-simulate.

So the sweep below is over b, at fixed length, and every curve is its own
simulation. What is compared is how the time law moves as b grows.

EXACT ANSWERS TO CHECK AGAINST
------------------------------
    trunk-return probability      h_V = 1 / (1 + b)
    mean absorption time          T_V = N(1 + 2N + bN) / b   steps

and in physical time T_V -> L^2 (1/2 + 1/b) / D for large N, which says
the b-dependence SATURATES: the 1/b term dies and every large-b tree
behaves like L^2/(2D).
"""

import numpy as np
import matplotlib.pyplot as plt

import lattice
import star


if __name__ == "__main__":
    rng = np.random.default_rng(99)

    N = 100
    n_walkers = 40000
    b_values = [1, 2, 3, 5, 10, 20]
    L = lattice.length(N)

    print(f"N = {N},  L = {L:.0f},  {n_walkers} walkers per b\n")
    print(f"  {'b':>3} {'mean t':>10} {'exact t':>10} {'ratio':>7} "
          f"{'h_V':>8} {'1/(1+b)':>9}")

    times = {}
    means, hs = [], []
    for b in b_values:
        branch, touched, steps = star.run_many(b, N, n_walkers, rng,
                                               start="vertex")
        t = lattice.to_time(steps)
        times[b] = t
        exact_t = lattice.to_time(star.mfpt_vertex_exact(b, N))
        means.append(t.mean())
        hs.append(touched.mean())
        print(f"  {b:>3} {t.mean():>10.1f} {exact_t:>10.1f} "
              f"{t.mean()/exact_t:>7.4f} {touched.mean():>8.4f} "
              f"{1/(1+b):>9.4f}")

    # --- figure -----------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.3))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(b_values)))

    # 1: the distributions themselves
    hi = np.percentile(times[1], 98)
    for b, c in zip(b_values, colors):
        ax1.hist(times[b], bins=80, range=(0, hi), density=True,
                 histtype="step", lw=2, color=c, label=f"b = {b}")
    ax1.set_xlabel("first-passage time  t")
    ax1.set_ylabel("density")
    ax1.set_title("more branches, faster absorption")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    # 2: survival, log scale, so the late-time slope is visible
    tgrid = np.linspace(hi/300, hi, 300)
    for b, c in zip(b_values, colors):
        s = np.sort(times[b])
        S = 1.0 - np.searchsorted(s, tgrid, side="right") / n_walkers
        ax2.plot(tgrid, S, color=c, lw=2, label=f"b = {b}")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 1)
    ax2.set_xlabel("time  t")
    ax2.set_ylabel("survival  S(t)")
    ax2.set_title("survival (log): steeper tail as b grows")
    ax2.grid(alpha=0.25, which="both")

    # 3: the saturation, measured against the exact curve
    bfine = np.linspace(1, 20, 200)
    exact_curve = [lattice.to_time(star.mfpt_vertex_exact(bb, N))
                   for bb in bfine]
    ax3.plot(bfine, exact_curve, color="0.5", lw=1.5,
             label=r"exact  $N(1+2N+bN)/b$")
    ax3.axhline(L**2/(2*lattice.D), color="k", ls=":", lw=1.2,
                label=r"$L^2/2D$  ($b\to\infty$)")
    ax3.plot(b_values, means, "o", color="#e08214", ms=8,
             label="simulation")
    ax3.set_xlabel("number of branches  b")
    ax3.set_ylabel("mean first-passage time")
    ax3.set_title("the b-dependence saturates")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    fig.suptitle("First-passage time vs branch count: every curve is its "
                 "own simulation, because b does not rescale")
    fig.tight_layout()
    fig.savefig("comparison_b.png", dpi=150)
    print("\nwrote comparison_b.png")