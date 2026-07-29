"""
Fractal (Koch) boundary geometry for the single-particle-in-fractal-boundary problem.

Domain: box [0,L] x [0,H]. Bottom wall replaced by a Koch curve whose folds
point INTO the domain (upward), so a particle diffusing down from above
encounters progressively deeper fjords. Left/right/top walls are plain
reflecting box edges.

Boundary is represented as a polyline: an (N,2) array of vertices, consumed
as N-1 line segments. Everything downstream (collision detection) works on
segments, so this is robust even where the curve is not a single-valued
function of x.
"""
import numpy as np
import matplotlib.pyplot as plt



def koch_generator(p1, p2, depth, fold_sign=+1.0):
    """
    Recursively build a Koch curve from p1 to p2.
    fold_sign=+1 folds the peak to the left of travel direction (into the domain
    if the domain is above the p1->p2 baseline and we're building left-to-right).
    Returns list of (x,y) vertices INCLUDING p1, EXCLUDING p2 (caller appends p2 once).
    """
    if depth == 0:
        return [p1]

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = (p2 - p1) / 3.0
    a = p1 + d
    b = p1 + 2 * d

    # rotate d by 60 degrees (fold_sign controls left/right fold) to get the peak
    theta = fold_sign * np.pi / 3.0
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    peak = a + rot @ d

    verts = []
    verts += koch_generator(p1, a, depth - 1, fold_sign)
    verts += koch_generator(a, peak, depth - 1, fold_sign)
    verts += koch_generator(peak, b, depth - 1, fold_sign)
    verts += koch_generator(b, p2, depth - 1, fold_sign)
    return verts


def build_boundary(L=1.0, H=1.0, depth=4):
    """
    Build the fractal bottom wall of the box domain.
    Baseline runs from (0,0) to (L,0); folds point upward (+y, into the domain).
    Returns:
        boundary : (N,2) array of vertices, left to right, y>=0
        arclen   : (N,) cumulative arc length along the boundary (for later
                   histogramming *where* absorption happens)
    """
    p1, p2 = (0.0, 0.0), (L, 0.0)
    verts = koch_generator(p1, p2, depth, fold_sign=-1.0)  # sign chosen so peaks go +y
    verts.append(p2)
    boundary = np.array(verts)

    # if peaks came out negative-y (folded the wrong way), flip
    if boundary[:, 1].min() < -1e-9:
        boundary[:, 1] *= -1.0

    seg_lengths = np.hypot(np.diff(boundary[:, 0]), np.diff(boundary[:, 1]))
    arclen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    return boundary, arclen


def fractal_dimension_koch():
    return np.log(4) / np.log(3)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, depth in zip(axes, [1, 3, 5]):
        b, s = build_boundary(L=1.0, H=1.0, depth=depth)
        ax.plot(b[:, 0], b[:, 1], color="#2b857f", lw=1.2)
        ax.fill_between(b[:, 0], b[:, 1], -0.05, color="#37bcb1", alpha=0.15)
        ax.set_title(f"depth={depth}, total length={s[-1]:.3f}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.6)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("/home/claude/fractal_mc/boundary_check.png", dpi=140)
    print("dimension (analytic):", fractal_dimension_koch())
    for depth in [1, 2, 3, 4, 5, 6]:
        b, s = build_boundary(depth=depth)
        print(f"depth={depth}: n_vertices={len(b)}, total_length={s[-1]:.4f}")