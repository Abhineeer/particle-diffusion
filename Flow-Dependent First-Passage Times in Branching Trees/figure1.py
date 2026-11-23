"""
fig1.py

Figure 1: model setup. Full page width (17.8 cm = 7.0 in).

    (a) geometry: trunk of length L, b branches, flow v0, Pe = v0 L / D
    (b) one simulated walker on a b = 2 network (highlighted), plus walkers
        that were absorbed on the other branch (grey)
    (c) the two vessel models, drawn with true radius ratios:
        equal areas    R0 = R1,          v1 = v0 / b
        Murray's law   R0 = b^(1/3) R1,  v1 = v0 b^(-1/3)

Panel (b) uses a small walker defined in this file. Once star.py exists,
replace walk_with_path() with a call to it so the figure uses the same code
as the results.

The model is one-dimensional along each vessel. In panel (b) the sideways
wiggle is a separate bounded walk across the vessel, added only so the path
is visible; it does not affect the dynamics. Say so in the caption.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "cm"

FULL_WIDTH = 7.0          # inches, 17.8 cm
HEIGHT = 2.8              # inches

INK = "#222222"
VESSEL = "#d9d9d9"
HIGHLIGHT = "#c0392b"
OTHER = "#7f7f7f"


# ---------------------------------------------------------------- geometry

def branch_dirs(b, spread_deg):
    """Unit vectors for b branches fanned symmetrically about the +x axis."""
    if b == 1:
        angles = [0.0]
    else:
        angles = np.linspace(spread_deg, -spread_deg, b)
    return [np.array([np.cos(np.radians(t)), np.sin(np.radians(t))])
            for t in angles]


def draw_tube(ax, p0, p1, r, color, z=1):
    """Rectangle of half-width r from point p0 to point p1."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]])
    corners = [p0 + r * n, p1 + r * n, p1 - r * n, p0 - r * n]
    ax.add_patch(Polygon(corners, closed=True, facecolor=color,
                         edgecolor="none", zorder=z))


def arrow(ax, p0, p1, color=INK, lw=1.5):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=12))


# ---------------------------------------------------------------- walker

def walk_with_path(b, N, Pe, beta, rng):
    """
    One walker released at the root of a trunk + b branches.

    Same rules as star.py: bounce at the root, area-weighted (flow-biased)
    choice at the junction, absorbed at a branch tip, drift Pe on the trunk and Pe/beta
    on a branch. Returns (edge, s) arrays along the whole path, where s in
    [0, 1] is the position along that edge measured away from the root.
    """
    eps_t = Pe / (2 * N)
    eps_b = eps_t / beta
    # same junction rule as star.py: area times the coin for that direction
    w_trunk = 1.0 * (1 - eps_t)
    w_branch = (beta / b) * (1 + eps_b)
    total = w_trunk + b * w_branch
    p_edge = [w_trunk / total] + [w_branch / total] * b

    edge, site = 0, 0
    edges, sites = [edge], [site]
    while True:
        if edge == 0 and site == N:                 # at the junction
            k = rng.choice(b + 1, p=p_edge)
            if k == 0:
                site = N - 1
            else:
                edge, site = k, 1
        elif edge == 0 and site == 0:               # reflecting root
            site = 1
        else:
            e = eps_t if edge == 0 else eps_b
            step = 1 if rng.random() < 0.5 * (1 + e) else -1
            site = site + step
            if edge != 0 and site == 0:             # back onto the junction
                edge, site = 0, N
        edges.append(edge)
        sites.append(site)
        if edge != 0 and site == N:                 # absorbed at a tip
            break
    return np.array(edges), np.array(sites) / N


def to_xy(edges, s, dirs, junction, wiggle):
    """Map (edge, s) to the plane, offsetting sideways by `wiggle`."""
    xy = np.zeros((len(s), 2))
    for i in range(len(s)):
        if edges[i] == 0:
            centre = np.array([s[i], 0.0])
            normal = np.array([0.0, 1.0])
        else:
            d = dirs[edges[i] - 1]
            centre = junction + s[i] * d
            normal = np.array([-d[1], d[0]])
        xy[i] = centre + wiggle[i] * normal
    return xy


def bounded_wiggle(n, half_width, rng):
    """Smooth random sideways offset in [-half_width, half_width], display only."""
    w = np.zeros(n)
    for i in range(1, n):
        w[i] = 0.85 * w[i - 1] + 0.35 * half_width * rng.normal()
        w[i] = min(max(w[i], -half_width), half_width)
    return w


# ---------------------------------------------------------------- panels

def panel_a(ax):
    junction = np.array([1.0, 0.0])
    # two branches, an ellipsis for "and so on", then two more branches
    branch_angles = [40, 22, -22, -40]           # degrees
    dirs = [np.array([np.cos(np.radians(t)), np.sin(np.radians(t))])
            for t in branch_angles]

    ax.plot([0, 1], [0, 0], color=INK, lw=2.5, solid_capstyle="butt")
    for d in dirs:
        tip = junction + 0.8 * d
        ax.plot([junction[0], tip[0]], [junction[1], tip[1]],
                color=INK, lw=2.5)
        ax.add_patch(Circle(tip, 0.045, facecolor="white", edgecolor=INK,
                            linewidth=1.5, zorder=3))      # open dot

    # ellipsis: three dots in a vertical column, centred in the gap
    # between the 22 and -22 degree branches
    x_dots = 1.62
    for y in [0.11, 0.0, -0.11]:                 # spacing 0.11
        ax.add_patch(Circle((x_dots, y), 0.02, color=INK, zorder=3))

    # reflecting root
    ax.plot([0, 0], [-0.1, 0.1], color=INK, lw=3)

    # flow arrow and length bar
    arrow(ax, (0.15, 0.13), (0.65, 0.13))
    ax.text(0.40, 0.19, r"$v_0$", ha="center", va="bottom")
    ax.annotate("", xy=(1.0, -0.17), xytext=(0.0, -0.17),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    ax.text(0.5, -0.22, r"$L$", ha="center", va="top")

    # bracket over the tips labelled b
    top = junction + 0.8 * dirs[0]
    bot = junction + 0.8 * dirs[-1]
    xb = junction[0] + 0.8 + 0.12
    ax.plot([xb, xb], [bot[1], top[1]], color=INK, lw=1)
    ax.plot([xb - 0.03, xb], [top[1], top[1]], color=INK, lw=1)
    ax.plot([xb - 0.03, xb], [bot[1], bot[1]], color=INK, lw=1)
    ax.text(xb + 0.05, 0, r"$b$", va="center")

    ax.text(0.05, 0.62, r"$\mathrm{Pe} = v_0 L / D$", ha="left")
    # ax.text(-0.09, 0, "reflecting", rotation=90, ha="right", va="center")
    # ax.text(bot[0], bot[1] - 0.1, "absorbing", ha="center", va="top")

    ax.set_xlim(-0.4, 2.1)
    ax.set_ylim(-0.85, 0.85)


def panel_b(ax, rng, single=False):
    b, N, Pe = 2, 30, 4.0
    every = 6                             # plot every 6th step
    beta = float(b)                       # equal areas for the illustration
    junction = np.array([1.0, 0.0])
    dirs = branch_dirs(b, 30)
    half_width = 0.09

    # vessel outline
    draw_tube(ax, (-0.02, 0), junction, half_width + 0.02, VESSEL)
    for d in dirs:
        draw_tube(ax, junction, junction + d, half_width + 0.02, VESSEL)
    ax.add_patch(Circle(junction, half_width + 0.02, color=VESSEL, zorder=1))

    if single:
        # one walker, absorbed on the upper branch, drawn in red
        while True:
            edges, s = walk_with_path(b, N, Pe, beta, rng)
            if edges[-1] == 1:
                break
        edges, s = edges[::every], s[::every]
        w = bounded_wiggle(len(s), 0.6 * half_width, rng)
        xy = to_xy(edges, s, dirs, junction, w)
        ax.plot(xy[:, 0], xy[:, 1], color=HIGHLIGHT, lw=0.8, zorder=3)
        ax.plot(*xy[-1], "*", color=HIGHLIGHT, ms=9, zorder=4)
        ax.plot(0, 0, "o", color=INK, ms=4, zorder=5)
        ax.set_xlim(-0.1, 2.0)
        ax.set_ylim(-0.85, 0.85)
        return

    # collect walkers so that each branch catches PER_BRANCH of them
    PER_BRANCH = 2
    colours = ["#c0392b", "#2471a3", "#1e8449", "#b9770e"]
    caught = {k: [] for k in range(1, b + 1)}
    while any(len(c) < PER_BRANCH for c in caught.values()):
        edges, s = walk_with_path(b, N, Pe, beta, rng)
        k = edges[-1]
        if len(caught[k]) < PER_BRANCH:
            caught[k].append((edges, s))
    walkers = [w for k in caught for w in caught[k]]

    # each walker wiggles inside its own lane so the paths stay readable
    lanes = np.linspace(-0.6, 0.6, len(walkers)) * half_width
    for (edges, s), colour, lane in zip(walkers, colours, lanes):
        edges, s = edges[::every], s[::every]
        w = lane + bounded_wiggle(len(s), 0.35 * half_width, rng)
        xy = to_xy(edges, s, dirs, junction, w)
        ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=0.7, alpha=0.85, zorder=3)
        ax.plot(*xy[-1], "*", color=colour, ms=8, zorder=4)
    ax.plot(0, 0, "o", color=INK, ms=4, zorder=5)    # common release point

    ax.set_xlim(-0.1, 2.0)
    ax.set_ylim(-0.85, 0.85)


def label_vessel(ax, p0, d, r, frac, side, radius_label, velocity_label,
                 length):
    """
    Same notation on every vessel segment, so trunk and branches match.

    The segment starts at p0, runs along unit vector d for `length`, and has
    radius r. A radius arrow (axis to wall) sits at fraction `frac` of the
    length (near the free end, away from the junction) and a velocity arrow
    along the axis sits around the middle. Both labels go just
    outside the wall on side +1 (left of d) or -1 (right of d).
    """
    p0 = np.asarray(p0, float)
    d = np.asarray(d, float)
    n = side * np.array([-d[1], d[0]])          # chosen outward normal

    # radius: axis -> wall, label just beyond the wall
    c = p0 + frac * length * d
    ax.annotate("", xy=c + r * n, xytext=c,
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9,
                                shrinkA=0, shrinkB=0, mutation_scale=8),
                zorder=5)
    t = c + (r + 0.09) * n
    ax.text(t[0], t[1], radius_label, ha="center", va="center", zorder=5)

    # velocity: arrow on the axis, label just beyond the same wall
    a0 = p0 + 0.28 * length * d
    a1 = p0 + 0.56 * length * d
    arrow(ax, a0, a1)
    mid = 0.5 * (a0 + a1)
    t = mid + 0.05 * n                           # just above the arrow, off the axis
    ax.text(t[0], t[1], velocity_label, ha="center", va="bottom", zorder=6)


def panel_c(ax):
    b = 2
    dirs = branch_dirs(b, 30)
    r0 = 0.2
    trunk_len, branch_len = 0.85, 0.75
    models = [
        ("equal areas", r0, 0.62,
         r"$r_0 = r_1$", r"$v_1 = v_0 / b$"),
        ("Murray's law", r0 / b ** (1 / 3), -0.72,
         r"$r_0 = b^{1/3} r_1$", r"$v_1 = v_0\, b^{-1/3}$"),
    ]
    for title, r1, y0, lab_r, lab_v in models:
        junction = np.array([trunk_len, y0])
        draw_tube(ax, (0.0, y0), junction, r0, VESSEL)
        for d in dirs:
            draw_tube(ax, junction, junction + branch_len * d, r1, VESSEL)
        ax.add_patch(Circle(junction, r0, color=VESSEL, zorder=1))

        # principal (centre) axis of every vessel: thin dash-dot line
        axis_style = dict(color="#666666", lw=0.8, ls=(0, (6, 2, 1, 2)), zorder=2)
        ax.plot([0.0, junction[0]], [y0, y0], **axis_style)
        for d in dirs:
            tip = junction + branch_len * d
            ax.plot([junction[0], tip[0]], [junction[1], tip[1]], **axis_style)

        # trunk: labels above it; upper branch: labels on its outer (upper) side
        label_vessel(ax, (0.0, y0), (1.0, 0.0), r0, 0.12, +1,
                     r"$r_0$", r"$v_0$", trunk_len)
        label_vessel(ax, junction, dirs[0], r1, 0.88, +1,
                     r"$r_1$", r"$v_1$", branch_len)

        # model name and relations on the right
        ax.text(1.7, y0 + 0.36, title, va="center")
        ax.text(1.7, y0 + 0.02, lab_r, va="center")
        ax.text(1.7, y0 - 0.30, lab_v, va="center")

    ax.set_xlim(-0.1, 2.9)
    ax.set_ylim(-1.38, 1.3)


# ---------------------------------------------------------------- figure

def main(single, name):
    rng = np.random.default_rng(7)
    fig, axes = plt.subplot_mosaic([["a", "b", "c"]],
                                   figsize=(FULL_WIDTH, HEIGHT),
                                   width_ratios=[1, 1, 1.35])
    panel_b(axes["a"], rng, single=single)    # trajectories now first
    panel_a(axes["b"])                         # geometry now second
    panel_c(axes["c"])

    for key, ax in axes.items():
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.0, 1.0, f"({key})", transform=ax.transAxes,
                fontweight="bold", va="top")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03,
                        wspace=0.05)
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png", dpi=300)


if __name__ == "__main__":
    main(single=False, name="fig1")          # four coloured walkers
    main(single=True, name="fig1_single")    # one red walker