"""
fpt_distribution.py

First-passage (absorption) time distribution for a particle released on the
trunk of a network with one reflecting trunk and b absorbing branches.
Default: release at the trunk tip (x0 = L), b = 2.

Simulation: lattice walk, Algorithm 1 of the branching write-up.
Theory:     eigenmode expansion over the branch-symmetric modes (Section 3 of
            fpt_distribution.tex):

    tan^2(theta) = b,  theta = lambda L,  both families tan(theta) = +sqrt(b), -sqrt(b)
    S(tau) = sum_n  cos(theta_n (1 - xi0)) * tan(theta_n)/theta_n * exp(-theta_n^2 tau)
    f(tau) = -dS/dtau

Units: tau = t D / L^2, xi0 = x0 / L. One lattice step is Delta t = L^2/(2 D N^2),
so a walk of K steps has tau = K / (2 N^2).

Outputs (next to this script):
    fpt_distribution_linear.pdf / .png   single panel, linear scale
    fpt_distribution.pdf / .png          linear + log panels
    fpt_readings.txt                     measured vs exact statistics
    fpt_readings_table.tex               same, as a LaTeX table (needs booktabs)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
b = 2              # number of branches
N = 75             # sites per edge (2*Delta t = 4e-4 L^2/D, far below bin width)
XI0 = 1.0          # release position x0/L on the trunk (1.0 = trunk tip, 0.0 = vertex)
M = 100000        # walkers
SEED = 7
N_MODES = 400      # modes per root family in the analytic sum
BIN_WIDTH = 0.05   # histogram bin width in tau

OUT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Lattice walk (Algorithm 1), vectorised over walkers
# ----------------------------------------------------------------------------
def simulate(b, n0, M, N, rng):
    """Return step counts K to absorption for M walkers released at trunk site n0."""
    edge = np.zeros(M, dtype=np.int64)          # 0 = trunk, 1..b = branches
    site = np.full(M, n0, dtype=np.int64)
    label = np.arange(M)
    K_alive = np.zeros(M, dtype=np.int64)
    K_out = np.empty(M, dtype=np.int64)

    while label.size:
        at_vertex = site == 0
        at_trunk_tip = (site == N) & (edge == 0)
        interior = ~(at_vertex | at_trunk_tip)

        nv = at_vertex.sum()
        if nv:                                   # vertex: uniform over b+1 edges, Eq. (8)
            edge[at_vertex] = rng.integers(0, b + 1, size=nv)
            site[at_vertex] = 1
        site[at_trunk_tip] = N - 1               # trunk tip: reflect
        ni = interior.sum()
        if ni:                                   # interior: +/- 1
            site[interior] += 2 * rng.integers(0, 2, size=ni) - 1

        K_alive += 1

        absorbed = (site == N) & (edge != 0)     # absorbed on arrival at a branch tip
        if absorbed.any():
            K_out[label[absorbed]] = K_alive[absorbed]
            keep = ~absorbed
            edge, site, label, K_alive = edge[keep], site[keep], label[keep], K_alive[keep]

    return K_out


# ----------------------------------------------------------------------------
# Analytic first-passage distribution (units L = D = 1)
# ----------------------------------------------------------------------------
def symmetric_modes(b, n_modes):
    """theta = lambda L solving tan^2(theta) = b, both root families, sorted."""
    a = np.arctan(np.sqrt(b))
    k = np.arange(n_modes)
    return np.sort(np.concatenate([a + k * np.pi,            # tan = +sqrt(b)
                                   (k + 1) * np.pi - a]))    # tan = -sqrt(b)


theta = symmetric_modes(b, N_MODES)
rates = theta**2                                               # decay rates, units D/L^2
c = np.cos(theta * (1 - XI0)) * np.tan(theta) / theta          # mode weights, Eq. (weights)


def survival(tau):
    return np.exp(-np.outer(tau, rates)) @ c


def density(tau):
    return np.exp(-np.outer(tau, rates)) @ (c * rates)


mean_exact = np.sum(c / rates)                       # E[tau]   = sum c / r
var_exact = 2 * np.sum(c / rates**2) - mean_exact**2 # E[tau^2] = 2 sum c / r^2
tail_rate_exact = rates[0]                           # arctan^2(sqrt b)


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    rng = np.random.default_rng(SEED)
    n0 = int(round(XI0 * N))
    tau = simulate(b, n0, M, N, rng) / (2.0 * N**2)

    # measured statistics
    mean, std = tau.mean(), tau.std(ddof=1)
    se = std / np.sqrt(M)
    median = np.median(tau)

    # analytic peak and median on a fine grid
    grid = np.linspace(1e-3, 8, 8000)
    f_grid = density(grid)
    peak_exact = grid[np.argmax(f_grid)]
    median_exact = grid[np.argmin(np.abs(survival(grid) - 0.5))]

    # measured tail rate: straight-line fit of log S on [2, 5]
    tail = np.linspace(2, 5, 31)
    S_emp = np.array([(tau > t).mean() for t in tail])
    tail_rate_meas = -np.polyfit(tail, np.log(S_emp), 1)[0]

    # histogram as a density: count / (M * bin width)
    bins = np.arange(0, tau.max() + BIN_WIDTH, BIN_WIDTH)
    counts, edges = np.histogram(tau, bins=bins, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])

    # ---- readings -------------------------------------------------------------
    stats = [
        ("mean",            mean,           mean_exact),
        ("std",             std,            np.sqrt(var_exact)),
        ("median",          median,         median_exact),
        ("tail decay rate", tail_rate_meas, tail_rate_exact),
    ]
    lines = [
        f"First-passage times: b = {b}, release x0/L = {XI0}, N = {N}, M = {M:,}",
        "(time in units of L^2/D)", "",
        f"{'quantity':<18}{'measured':>10}{'exact':>10}",
    ]
    lines += [f"{n:<18}{m:>10.4f}{e:>10.4f}" for n, m, e in stats]
    lines += [
        "",
        f"mean: SE = {se:.4f}, z = {(mean - mean_exact) / se:+.2f}",
        f"peak of f (analytic): {peak_exact:.4f}",
        f"checks: S(0.005) = {survival(np.array([0.005]))[0]:.6f} (should be 1), "
        f"exact mean vs Eq.(11) value {0.5 + 1/b + XI0 - XI0**2/2:.4f}",
    ]
    report = "\n".join(lines)
    print(report)
    (OUT / "fpt_readings.txt").write_text(report + "\n")

    tex = [r"\begin{tabular}{lcc}", r"\toprule",
           r"quantity & measured & exact \\", r"\midrule"]
    tex += [f"{n} & {m:.4f} & {e:.4f} \\\\" for n, m, e in stats]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "fpt_readings_table.tex").write_text("\n".join(tex) + "\n")

    # ---- figures ---------------------------------------------------------------
    blue, orange = "#2a78d6", "#eb6834"

    def draw(ax, logy):
        ax.bar(centres, counts, width=BIN_WIDTH * 0.92, color=blue, alpha=0.45,
               label=f"lattice walk ({M:,} walkers)")
        ax.plot(grid, f_grid, color=orange, lw=2, label="eigenmode expansion")
        ax.axvline(mean_exact, color="0.3", lw=1, ls="--")
        ax.set_xlabel(r"first-passage time  $tD/L^2$")
        style(ax)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 2)
            ax.set_xlim(0, 7)
            ax.set_title(r"log scale: straight tail, slope $-\arctan^2\!\sqrt{b}$",
                         fontsize=10)
        else:
            ax.set_xlim(0, 5)
            ax.set_ylabel(r"probability density  $f(t)\,L^2/D$")
            ax.text(mean_exact + 0.05, ax.get_ylim()[1] * 0.9,
                    f"mean = {mean_exact:.2f}", fontsize=9, color="0.3")
            ax.legend(frameon=False, fontsize=9)

    where = "trunk tip" if XI0 == 1.0 else f"x0/L = {XI0}"
    title = f"First-passage time distribution, released at {where}, b = {b}"

    # single panel, linear
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    draw(ax, logy=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fpt_distribution_linear.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fpt_distribution_linear.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # two panels, linear + log
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    draw(ax1, logy=False)
    ax1.set_title("linear scale", fontsize=10)
    draw(ax2, logy=True)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fpt_distribution.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fpt_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote figures and readings to {OUT}")


if __name__ == "__main__":
    main()