"""Velocity decay law, and survival with flow under Murray's law."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lattice, star_drift as sd

plt.rcParams.update({"font.size": 21, "axes.titlesize": 24,
                     "axes.labelsize": 23, "xtick.labelsize": 20,
                     "ytick.labelsize": 20, "legend.fontsize": 19,
                     "legend.title_fontsize": 20})

def survival(t, g):
    s = np.sort(t)
    return 1.0 - np.searchsorted(s, g, side="right") / t.size

if __name__ == "__main__":
    rng = np.random.default_rng(77)
    N, n_walkers = 80, 40000
    b_values = [1, 2, 3, 5, 10, 20]
    Pe_values = [0.0, 5.0, 20.0]

    # ---------- figure 1: the two velocity laws ----------
    fig, ax = plt.subplots(figsize=(10, 6.4))
    bf = np.linspace(1, 20, 300)
    ax.plot(bf, 1.0/bf, lw=3.4, color="#b03a2e", label=r"naive:  $v_1/v_0 = 1/b$")
    ax.plot(bf, bf**(-1/3), lw=3.4, color="#1f6f68",
            label=r"Murray:  $v_1/v_0 = b^{-1/3}$")
    for b in (5, 10, 20):
        ax.plot([b, b], [1.0/b, b**(-1/3)], color="0.6", lw=1.6, ls=":")
    ax.set_xlabel("number of branches  $b$")
    ax.set_ylabel(r"branch velocity  $v_1 / v_0$")
    ax.set_title("Murray's law dilutes the flow far less")
    ax.set_ylim(0, 1.02); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("fig_velocity.png", dpi=150)
    print("wrote fig_velocity.png")

    # ---------- figure 2: survival with flow, Murray ----------
    times = {}
    for Pe in Pe_values:
        for b in b_values:
            _, st = sd.run_many(b, N, n_walkers, rng, Pe=Pe,
                                beta=float(b)**(1/3))
            times[(Pe, b)] = lattice.to_time(st, N)

    print(f"\nmean first-passage time, beta = b^(1/3), N = {N}\n")
    print(f"  {'Pe':>5} " + " ".join(f"{('b='+str(b)):>8}" for b in b_values))
    for Pe in Pe_values:
        print(f"  {Pe:>5.0f} " + " ".join(f"{times[(Pe,b)].mean():>8.4f}"
                                          for b in b_values))

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(b_values)))
    for ax, Pe in zip(axes, Pe_values):
        tmax = 2.6 * max(times[(Pe, b)].mean() for b in b_values)
        g = np.linspace(0.0, tmax, 500)
        for b, c in zip(b_values, colors):
            ax.plot(g, survival(times[(Pe, b)], g), color=c, lw=3.0,
                    label=f"{b}")
        ax.set_xlim(0, tmax); ax.set_ylim(0, 1)
        ax.set_title(f"Pe = {Pe:.0f}")
        ax.set_xlabel("time  $t$   (units of $L^2/D$)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("survival  $S(t)$")
    axes[0].legend(title="branches  $b$", ncol=2, loc="upper right")
    fig.suptitle(r"With flow, under Murray's law  $v_1 = v_0 b^{-1/3}$",
                 fontsize=25, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("fig_flow_murray.png", dpi=150)
    print("wrote fig_flow_murray.png")