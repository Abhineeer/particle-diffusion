"""
Each segment carries X_i(x) = A_i cos(lam x) + B_i sin(lam x), so there are
six unknowns:
        u = [A1, B1, A2, B2, A3, B3]
 
and exactly six homogeneous conditions:
 
  (1) trunk reflecting, eq (2):    -A1 sin(lam L) + B1 cos(lam L) = 0
  (2) branch 2 absorbing, eq (3):   A2 cos(lam L) + B2 sin(lam L) = 0
  (3) branch 3 absorbing, eq (3):   A3 cos(lam L) + B3 sin(lam L) = 0
  (4) continuity, eq (12):          A1 - A2 = 0
  (5) continuity, eq (12):          A1 - A3 = 0
  (6) flux conservation, eq (14):   B1 + B2 + B3 = 0
 
Written as M(lam) u = 0. Six equations, six unknowns, all homogeneous, so
u = 0 always works -- that is the empty network, no particle. A REAL mode is
a nonzero u, and a homogeneous system has one only when M is singular:
 
        det M(lam) = 0
 
That single scalar equation IS the eigenvalue condition. Everything in
Section 5 is a hand-computation of this determinant.

"""

import numpy as np
from scipy.optimize import brentq

L = 1.0

def M(lam, L=L):
    # The 6x6 condition matrix. Columns: A1 B1 A2 B2 A3 B3.
    s, c = np.sin(lam * L), np.cos(lam * L)
    return np.array([
        [-s,  c,  0,  0,  0,  0],   # trunk reflecting
        [0,  0,  c,  s,  0,  0],   # branch 2 absorbing
        [0,  0,  0,  0,  c,  s],   # branch 3 absorbing
        [1,  0, -1,  0,  0,  0],   # continuity  A1 = A2
        [1,  0,  0,  0, -1,  0],   # continuity  A1 = A3
        [0,  1,  0,  1,  0,  1],   # flux  B1 + B2 + B3 = 0
    ], dtype=float)

def det(lam, L=L):
    return np.linalg.det(M(lam, L))


def nullvector(lam, L=L):
    # The mode shape: the singular vector of M with the smallest value.
    _, sv, Vt = np.linalg.svd(M(lam, L))
    return Vt[-1], sv[-1]
 

# Finding roots

def find_roots(lam_max=25.0, n_scan=200000, L=L):
    """Scan for sign changes in det, then bracket each with brentq."""
    grid = np.linspace(1e-6, lam_max, n_scan)
    vals = np.array([det(l, L) for l in grid])
    roots = []
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            roots.append(brentq(det, grid[i], grid[i + 1], args=(L,), xtol=1e-13))
    return np.array(roots)
 
 
def classify(lam, L=L, tol=1e-6):
    """Read the null vector and label the mode by its symmetry."""
    v, smin = nullvector(lam, L)
    A1, B1, A2, B2, A3, B3 = v
    if abs(A1) < tol * max(1.0, np.abs(v).max()):
        return "antisym", v, smin
    return "sym    ", v, smin
 
 
if __name__ == "__main__":
    print(type(det), det)
    roots = find_roots(lam_max=25.0)
 
    print(f"found {len(roots)} roots of det M(lam) = 0 in 0 < lam*L < 25\n")
    print(f"{'lam*L':>10} {'kind':>9} {'tan(lam L)':>12} {'A':>8} "
          f"{'B1':>8} {'B2':>8} {'B3':>8} {'sigma_min':>11}")
    for lam in roots[:12]:
        kind, v, smin = classify(lam)
        v = v / (v[0] if abs(v[0]) > 1e-8 else np.abs(v).max())   # normalise
        A1, B1, A2, B2, A3, B3 = v
        tanv = np.tan(lam * L)
        print(f"{lam * L:10.5f} {kind:>9} {tanv:12.5f} {A1:8.4f} "
              f"{B1:8.4f} {B2:8.4f} {B3:8.4f} {smin:11.2e}")
 
    # ---------------------------------------------------------------- checks
    a = np.arctan(np.sqrt(2))
    print("\ncompare with the document:\n")
    print(f"  eq (21) branch tan = +sqrt2 :  "
          f"{', '.join(f'{a + n * np.pi:.5f}' for n in range(3))}")
    print(f"  eq (21) branch tan = -sqrt2 :  "
          f"{', '.join(f'{np.pi - a + n * np.pi:.5f}' for n in range(3))}")
    print(f"  eq (29) antisymmetric n*pi  :  "
          f"{', '.join(f'{n * np.pi:.5f}' for n in range(1, 4))}")
 
    # do the found roots reproduce those three lists exactly?
    predicted = np.sort(np.concatenate([
        a + np.arange(8) * np.pi,
        (np.pi - a) + np.arange(8) * np.pi,
        np.pi * np.arange(1, 9),
    ]))
    predicted = predicted[predicted < 25.0]
    found = np.sort(roots)
    n = min(len(found), len(predicted))
    print(f"\n  max |found - predicted| over {n} modes = "
          f"{np.max(np.abs(found[:n] - predicted[:n])):.2e}")
 
    # ---------------------------------------------------------------- counting
    print("\nmode counting (Weyl, eq 31):\n")
    for cut in (10.0, 15.0, 20.0, 25.0):
        got = np.sum(roots < cut)
        expect = 3 * L * cut / np.pi
        print(f"   lam*L < {cut:5.1f}:  found {got:3d}   Weyl predicts "
              f"{expect:6.2f}   ratio {got / expect:.3f}")
    print("\n-> the count tracks 3L/pi, so nothing is missing. Had we found")
    print("   only the +sqrt2 family plus antisym, the ratio would sit near 2/3.")
