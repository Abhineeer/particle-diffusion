"""Survival across branch count, no flow, out to b = 150."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lattice, star

plt.rcParams.update({"font.size": 20, "axes.titlesize": 24,
                     "axes.labelsize": 22, "xtick.labelsize": 19,
                     "ytick.labelsize": 19, "legend.fontsize": 18,
                     "legend.title_fontsize": 19})

def survival(times, tgrid):
    s = np.sort(times)
    return 1.0 - np.searchsorted(s, tgrid, side="right") / times.size

if __name__ == "__main__":
    rng = np.random.default_rng(606)
    N, n_walkers = 80, 30000
    b_values = [1, 2, 3, 5, 10, 30, 100, 150]
    tgrid = np.linspace(0.0, 6.0, 900)

    curves, means = {}, {}
    print(f"N = {N}, {n_walkers} walkers, L = D = 1\n")
    print(f"  {'b':>5} {'mean t':>9} {'exact':>9} {'gain':>8}")
    prev = None
    for b in b_values:
        _, _, steps = star.run_many(b, N, n_walkers, rng, start="vertex")
        t = lattice.to_time(steps, N)
        curves[b], means[b] = survival(t, tgrid), t.mean()
        gain = "" if prev is None else f"{prev - t.mean():.4f}"
        print(f"  {b:>5} {t.mean():>9.4f} "
              f"{star.mfpt_vertex_continuum(b):>9.4f} {gain:>8}")
        prev = t.mean()
    print(f"\n  floor L^2/2D = {lattice.mfpt_segment():.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.2))
    colors = plt.cm.viridis(np.linspace(0.05, 0.88, len(b_values)))
    for b, c in zip(b_values, colors):
        ax1.plot(tgrid, curves[b], color=c, lw=3.0, label=f"{b}")
    ax1.set_xlim(0, 6.0); ax1.set_ylim(0, 1)
    ax1.set_xlabel("time  $t$   (units of $L^2/D$)")
    ax1.set_ylabel("survival  $S(t)$")
    ax1.set_title("Survival, no flow")
    ax1.legend(title="branches  $b$", ncol=2); ax1.grid(alpha=0.25)

    bb = np.array(b_values, dtype=float)
    mm = np.array([means[b] for b in b_values])
    bfine = np.linspace(1, 500, 400)
    ax2.plot(bfine, 1.0*(0.5 + 1.0/bfine), color="0.5", lw=2.4,
             label=r"$\frac{L^2}{D}(\frac{1}{2}+\frac{1}{b})$")
    ax2.axhline(0.5, color="crimson", ls="--", lw=2.0,
                label=r"floor  $L^2/2D$")
    ax2.plot(bb, mm, "o", color="#1f6f68", ms=11, label="simulation")
    ax2.set_xscale("log")
    ax2.set_ylim(0.4, 1.6)
    ax2.set_xlabel("number of branches  $b$")
    ax2.set_ylabel("mean first-passage time")
    ax2.set_title("The gain saturates")
    ax2.legend(); ax2.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig("fig_saturation.png", dpi=150)
    print("\nwrote fig_saturation.png")