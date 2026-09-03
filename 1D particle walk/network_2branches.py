"""
Single-particle diffusion on a 3-segment network (trunk + 2 branches).

Geometry: every segment is x in [0, L], with x = 0 AT THE VERTEX and x = L
at the far end.
    segment 0 = trunk    -> reflecting at x = L   (dX/dx = 0)
    segment 1 = branch A -> absorbing  at x = L   (X = 0)
    segment 2 = branch B -> absorbing  at x = L   (X = 0)

Vertex conditions: continuity (X_i(0) equal) + zero net flux (sum X_i'(0) = 0).

Spectrum splits into two families:

  SYMMETRIC (X1 = X2, trunk active):     tan^2(lam L) = 2
      -> lam L = arctan(sqrt2) + n*pi   AND   lam L = pi - arctan(sqrt2) + n*pi
         (BOTH branches of the square root; dropping one loses half the basis)
      X_trunk  =  cos(lam(L-x)) / cos(lam L)
      X_branch =  sin(lam(L-x)) / sin(lam L)

  ANTISYMMETRIC (X1 = -X2, trunk silent, node at vertex):   lam L = n*pi
      X_trunk = 0,  X_branchA = sin(lam x),  X_branchB = -sin(lam x)

Counting check: 2/pi symmetric + 1/pi antisymmetric = 3/pi modes per unit lam,
which is Weyl's law for total length 3L. The basis is complete.
"""

import numpy as np

SYMMETRIC = "sym"
ANTISYMMETRIC = "anti"


# ----------------------------------------------------------------------
# spectrum
# ----------------------------------------------------------------------
def solve_eigenvalues(L=1.0, n_sym=20, n_anti=20):
    """Return (lams, kinds) sorted by eigenvalue.

    lams  : ndarray of lambda values
    kinds : ndarray of 'sym' / 'anti' labels, aligned with lams
    """
    a = np.arctan(np.sqrt(2.0))          # 0.955316...

    n = np.arange(n_sym)
    branch_plus = a + n * np.pi          # tan(lam L) = +sqrt2
    branch_minus = (np.pi - a) + n * np.pi  # tan(lam L) = -sqrt2
    lam_sym = np.concatenate([branch_plus, branch_minus]) / L

    m = np.arange(1, n_anti + 1)         # n = 0 is the trivial zero mode
    lam_anti = m * np.pi / L

    lams = np.concatenate([lam_sym, lam_anti])
    kinds = np.array([SYMMETRIC] * lam_sym.size + [ANTISYMMETRIC] * lam_anti.size)

    order = np.argsort(lams)
    return lams[order], kinds[order]


# ----------------------------------------------------------------------
# eigenfunctions
# ----------------------------------------------------------------------
def eigenfunction(lam, kind, x, L=1.0):
    """Evaluate one mode on all three segments.

    Returns array of shape (3, len(x)): rows are [trunk, branchA, branchB].
    """
    x = np.asarray(x, dtype=float)
    X = np.zeros((3, x.size))

    if kind == SYMMETRIC:
        X[0] = np.cos(lam * (L - x)) / np.cos(lam * L)
        X[1] = np.sin(lam * (L - x)) / np.sin(lam * L)
        X[2] = X[1]
    elif kind == ANTISYMMETRIC:
        X[0] = 0.0
        X[1] = np.sin(lam * x)
        X[2] = -X[1]
    else:
        raise ValueError(f"unknown mode kind {kind!r}")
    return X


def mode_norm(lam, kind, L=1.0):
    """<X, X> = sum_i int_0^L X_i^2 dx, computed in closed form."""
    if kind == ANTISYMMETRIC:
        return L  # 2 * (L/2), trunk contributes nothing

    s2 = np.sin(2 * lam * L)
    c = np.cos(lam * L)
    s = np.sin(lam * L)
    trunk = (L / 2 + s2 / (4 * lam)) / c**2
    branch = (L / 2 - s2 / (4 * lam)) / s**2
    return trunk + 2 * branch


# ----------------------------------------------------------------------
# inner products and projection
# ----------------------------------------------------------------------
def inner(f, g, x):
    """<f, g> summed over the three segments. f, g have shape (3, N)."""
    return float(np.sum(np.trapezoid(f * g, x, axis=1)))


def project(p0, x, L=1.0, n_sym=20, n_anti=20):
    """Decompose an initial density p0 (shape (3, N)) onto the eigenbasis.

    Returns (lams, kinds, coeffs).
    """
    lams, kinds = solve_eigenvalues(L, n_sym, n_anti)
    coeffs = np.empty(lams.size)
    for k, (lam, kind) in enumerate(zip(lams, kinds)):
        X = eigenfunction(lam, kind, x, L)
        coeffs[k] = inner(p0, X, x) / mode_norm(lam, kind, L)
    return lams, kinds, coeffs


def density(t, lams, kinds, coeffs, x, L=1.0, D=1.0):
    """Reconstruct p(x, t), shape (3, N)."""
    P = np.zeros((3, np.size(x)))
    for lam, kind, c in zip(lams, kinds, coeffs):
        if c == 0.0:
            continue
        P += c * eigenfunction(lam, kind, x, L) * np.exp(-D * lam**2 * t)
    return P


def survival(t, lams, kinds, coeffs, x, L=1.0, D=1.0):
    """S(t) = total probability remaining on the network."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros(t.size)
    for lam, kind, c in zip(lams, kinds, coeffs):
        X = eigenfunction(lam, kind, x, L)
        mass = float(np.sum(np.trapezoid(X, x, axis=1)))  # zero for antisym
        out += c * mass * np.exp(-D * lam**2 * t)
    return out


def delta_initial(x, seg, x0, L=1.0):
    """Normalised point source: nearest-node delta on segment `seg` at x0."""
    p0 = np.zeros((3, x.size))
    j = int(np.argmin(np.abs(x - x0)))
    dx = x[1] - x[0]
    p0[seg, j] = 1.0 / dx
    return p0