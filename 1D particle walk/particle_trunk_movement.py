"""
STEP 5  --  what the particle actually does.

Draws the network as a Y: trunk to the left, two branches to the upper and
lower right, all meeting at the vertex. Colour along each arm is the
probability density. Then watch it in time.

The story to look for:
  1. the packet starts on the trunk and spreads both ways
  2. it reaches the vertex and SPLITS -- equally, because the two branches
     are identical (this is the branch-exchange symmetry of Section 7)
  3. part of it reflects off the trunk's closed end and comes back
  4. probability drains out of the two branch tips and never returns
  5. what is left settles into the shape of the slowest mode, lam*L = 0.9553

Needs the finite-difference solver from step 2 in the same folder.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

# the step-2 solver, under whichever name you saved it
try:
    import network_2branches_fd as fd
except ModuleNotFoundError:
    import network_2branches_fd as fd

L, D = 1.0, 1.0
TRUNK, BRANCH_A, BRANCH_B = 0, 1, 2
X0_SEG, X0_POS = TRUNK, 0.5 * L          # where the particle starts


# ------------------------------------------------------------------ geometry
def arm_xy(seg, s):
    """Map arc length s in [0,L] on segment `seg` to a point in the plane.
    The vertex sits at the origin; s = 0 there for all three arms."""
    if seg == TRUNK:
        ang = np.pi                       # trunk points left
    elif seg == BRANCH_A:
        ang = np.pi / 5                   # upper right
    else:
        ang = -np.pi / 5                  # lower right
    return s * np.cos(ang), s * np.sin(ang)


def draw_network(ax, P, x, norm, cmap="magma"):
    """Colour each arm by its density."""
    for seg in (TRUNK, BRANCH_A, BRANCH_B):
        px, py = arm_xy(seg, x)
        pts = np.column_stack([px, py]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=7,
                            capstyle="round")
        lc.set_array(0.5 * (P[seg][:-1] + P[seg][1:]))
        ax.add_collection(lc)
    # mark the ends
    ax.plot(*arm_xy(TRUNK, L), marker="|", ms=22, mew=3, color="#2b8cbe")
    for s in (BRANCH_A, BRANCH_B):
        ax.plot(*arm_xy(s, L), marker="o", ms=9, mfc="none", mew=2,
                color="#d94801")
    ax.plot(0, 0, "o", ms=5, color="w", zorder=5)
    ax.set_xlim(-1.15, 1.0), ax.set_ylim(-0.72, 0.72)
    ax.set_aspect("equal"), ax.axis("off")


# ------------------------------------------------------------------ run
x = fd.make_grid(L, 401)
P0 = fd.point_source(x, X0_SEG, X0_POS)
times = [0.005, 0.02, 0.05, 0.15, 0.4, 1.0]
ts, snaps = fd.evolve(P0, x, D, save_times=times, absorbing=True)

norm = LogNorm(vmin=1e-3, vmax=3.0)

fig = plt.figure(figsize=(13.5, 8.6))
gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 1.15, 1.0], hspace=.55, wspace=.28)

# --- row 1: the network, coloured, at six times
for k, (t, P) in enumerate(zip(ts, snaps)):
    ax = fig.add_subplot(gs[0, k])
    draw_network(ax, P, x, norm)
    ax.set_title(f"t = {t:.3f}\nS = {fd.mass(P, x):.3f}", fontsize=9)

# --- row 2: the same data as profiles along each arm
ax1 = fig.add_subplot(gs[1, :3])
ax2 = fig.add_subplot(gs[1, 3:], sharey=ax1)
cols = plt.cm.viridis(np.linspace(.05, .9, len(ts)))
for (t, P), c in zip(zip(ts, snaps), cols):
    ax1.plot(x, P[TRUNK], color=c, lw=1.8, label=f"t={t:g}")
    ax2.plot(x, P[BRANCH_A], color=c, lw=1.8)
ax1.axvline(X0_POS, color="grey", ls=":", lw=1)
ax1.set_title("trunk   (vertex at x=0, reflecting wall at x=L)", fontsize=10)
ax2.set_title("either branch   (vertex at x=0, absorbing tip at x=L)", fontsize=10)
for a in (ax1, ax2):
    a.set_xlabel("x"), a.set_ylim(0, 2.4), a.grid(alpha=.25)
ax1.set_ylabel("p(x,t)")
ax1.legend(fontsize=8, frameon=False, ncol=2)

# --- row 3: survival, and where the probability sits
ax3 = fig.add_subplot(gs[2, :3])
tfine = np.linspace(1e-4, 3.0, 60)
tt, ss = fd.evolve(P0, x, D, save_times=tfine, absorbing=True)
S = [fd.mass(P, x) for P in ss]
ax3.plot(tt, S, lw=2, color="#7a0177")
for t in ts:
    ax3.axvline(t, color="grey", lw=.7, ls=":")
ax3.set_xlabel("t"), ax3.set_ylabel("S(t)"), ax3.grid(alpha=.25)
ax3.set_title("survival — probability still on the network", fontsize=10)

ax4 = fig.add_subplot(gs[2, 3:])
m_tr = [np.trapezoid(P[TRUNK], x) for P in ss]
m_br = [2 * np.trapezoid(P[BRANCH_A], x) for P in ss]
ax4.plot(tt, m_tr, lw=2, label="on the trunk")
ax4.plot(tt, m_br, lw=2, label="on both branches")
ax4.plot(tt, np.array(m_tr) + np.array(m_br), "k--", lw=1, label="total = S(t)")
ax4.set_xlabel("t"), ax4.set_ylabel("probability"), ax4.grid(alpha=.25)
ax4.legend(fontsize=8, frameon=False)
ax4.set_title("where the probability lives", fontsize=10)

cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"),
                  ax=fig.axes[:6], location="bottom", fraction=.06, pad=.04,
                  aspect=70)
cb.set_label("density p (log scale)", fontsize=9)

fig.suptitle("A particle released on the trunk at x = 0.7L", fontsize=13, y=.99)
fig.savefig("step5_movement.png", dpi=150, bbox_inches="tight")
print("wrote step5_movement.png")

# ------------------------------------------------------------------ numbers
print(f"\n{'t':>7} {'S(t)':>9} {'on trunk':>10} {'on branches':>12} "
      f"{'vertex p':>10}")
for t, P in zip(ts, snaps):
    print(f"{t:7.3f} {fd.mass(P, x):9.5f} {np.trapezoid(P[TRUNK], x):10.5f} "
          f"{2 * np.trapezoid(P[BRANCH_A], x):12.5f} {P[0, 0]:10.5f}")

i_half = int(np.argmin(np.abs(np.array(S) - 0.5)))
print(f"\nhalf the probability is gone by t = {tt[i_half]:.3f}")
print(f"branch A and branch B masses differ by "
      f"{max(abs(np.trapezoid(P[BRANCH_A], x) - np.trapezoid(P[BRANCH_B], x)) for P in snaps):.1e}")