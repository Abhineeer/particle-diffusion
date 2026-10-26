"""
survival_b.py

THE HEADLINE FIGURE.

Claim: more branches does not keep making absorption faster. The gain
saturates, and after about b = 5 extra branches buy almost nothing.

Survival probability is the right way to see it:

    S(t) = probability the particle is still in the network at time t

because the whole story is in one monotone curve per b, with no binning.
Two facts make it readable:

    1. the area under S(t) IS the mean first-passage time,
           <t> = integral of S(t) dt
       so curves sitting on top of each other have equal mean times.

    2. at late times only the slowest eigenmode survives, so
           S(t) -> c0 * exp(-gamma0 * t)
       and on a log plot the tail is a straight line of slope -gamma0.

THE CEILING
-----------
Separation of variables on the star with b branches closes with

    symmetric modes       tan^2(lambda L) = b
    antisymmetric modes   lambda L = n*pi,  b-1 copies of each

so the slowest decay rate is

    gamma0 = D * arctan^2(sqrt b) / L^2

Since arctan saturates at pi/2, gamma0 can never exceed

    gamma0_max = pi^2 D / (4 L^2) = 2.467  (with L = D = 1)

No number of branches makes the survival curve decay faster than that.
That ceiling is the whole point.

Units: L = 1, D = 1, so times are in units of L^2/D and the floor on the
mean first-passage time is L^2/2D = 0.5.
"""

import numpy as np
import matplotlib.pyplot as plt

import lattice
import star


def gamma0(b, L=lattice.L_DEFAULT):
    """Slowest decay rate, D*arctan^2(sqrt b)/L^2."""
    return lattice.D * np.arctan(np.sqrt(b))**2 / L**2


GAMMA_MAX = np.pi**2 / 4          # the b -> infinity ceiling, L = D = 1


def survival_from_times(times, tgrid):
    """Empirical S(t): the fraction of walkers not yet absorbed."""
    s = np.sort(times)
    return 1.0 - np.searchsorted(s, tgrid, side="right") / times.size


if __name__ == "__main__":
    rng = np.random.default_rng(606)

    N = 80
    n_walkers = 40000
    b_values = [1, 2, 3, 5, 10, 20, 30]

    tmax = 3.0
    tgrid = np.linspace(0.0, tmax, 600)

    curves, means = {}, {}
    print(f"N = {N},  L = {lattice.L_DEFAULT},  D = {lattice.D},  "
          f"{n_walkers} walkers per b\n")
    print(f"  {'b':>3} {'mean t':>9} {'exact':>9} {'gain':>8} "
          f"{'gamma0':>9} {'gain':>8} {'% of ceiling':>13}")

    prev_t = prev_g = None
    for b in b_values:
        _, _, steps = star.run_many(b, N, n_walkers, rng, start="vertex")
        t = lattice.to_time(steps, N)
        curves[b] = survival_from_times(t, tgrid)
        means[b] = t.mean()

        g = gamma0(b)
        dt_gain = "" if prev_t is None else f"{prev_t - t.mean():.3f}"
        dg_gain = "" if prev_g is None else f"{g - prev_g:.3f}"
        print(f"  {b:>3} {t.mean():>9.4f} "
              f"{star.mfpt_vertex_continuum(b):>9.4f} {dt_gain:>8} "
              f"{g:>9.4f} {dg_gain:>8} {100*g/GAMMA_MAX:>12.1f}%")
        prev_t, prev_g = t.mean(), g

    print(f"\n  floor on the mean time  L^2/2D   = "
          f"{lattice.mfpt_segment():.4f}")
    print(f"  ceiling on the rate     pi^2/4   = {GAMMA_MAX:.4f}")

    # ---------------------------------------------------------- figure
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.95],
                          wspace=0.28, top=0.80)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(b_values)))

    # --- left: S(t), linear. the gaps, then the crowding
    for b, c in zip(b_values, colors):
        ax1.plot(tgrid, curves[b], color=c, lw=2.2, label=f"b = {b}")
    ax1.set_xlim(0, 1.6)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("time  $t$   (units of $L^2/D$)")
    ax1.set_ylabel("survival  $S(t)$")
    ax1.set_title("1$\\to$2 is a big gap.  After 5 the curves merge.",
                  fontsize=10)
    ax1.legend(fontsize=8.5, title="branches")
    ax1.grid(alpha=0.25)

    # --- middle: log scale, tails against the predicted slopes
    for b, c in zip(b_values, colors):
        ax2.plot(tgrid, curves[b], color=c, lw=2.2, label=f"b = {b}")
    # predicted late-time slopes, anchored on each measured curve
    for b, c in zip(b_values, colors):
        i = np.argmin(np.abs(tgrid - 0.9))
        anchor = curves[b][i]
        if anchor <= 0:
            continue
        tt = tgrid[tgrid >= 0.9]
        ax2.plot(tt, anchor * np.exp(-gamma0(b) * (tt - tgrid[i])),
                 color="k", ls=":", lw=1.0, zorder=5)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 1)
    ax2.set_xlim(0, tmax)
    ax2.set_xlabel("time  $t$   (units of $L^2/D$)")
    ax2.set_ylabel("survival  $S(t)$")
    ax2.set_title("log scale: tail slope is $-\\gamma_0$\n"
                  "dotted = $D\\arctan^2(\\sqrt{b})/L^2$", fontsize=10)
    ax2.grid(alpha=0.25, which="both")

    # --- right: the saturation itself
    bfine = np.linspace(1, 32, 300)
    ax3.plot(bfine, gamma0(bfine), color="0.45", lw=1.8,
             label=r"$\gamma_0 = \arctan^2\sqrt{b}$")
    ax3.axhline(GAMMA_MAX, color="crimson", ls="--", lw=1.4,
                label=r"ceiling $\pi^2/4 = 2.47$")
    for b, c in zip(b_values, colors):
        ax3.plot(b, gamma0(b), "o", color=c, ms=9, zorder=5)
    ax3.set_ylim(0, 2.8)
    ax3.set_xlabel("number of branches  $b$")
    ax3.set_ylabel(r"decay rate  $\gamma_0$")
    ax3.set_title("the rate has a hard ceiling", fontsize=10)
    ax3.legend(fontsize=8.5, loc="lower right")
    ax3.grid(alpha=0.25)

    fig.suptitle("More branches stops helping: the decay rate saturates at "
                 r"$\pi^2 D/4L^2$", fontsize=13, y=0.97)
    fig.savefig("survival_b.png", dpi=150, bbox_inches="tight")
    print("\nwrote survival_b.png")