"""
STEP 4  --  Section 6, equation (24).

Step 3 gave us, for each allowed lam, a null vector [A1,B1,A2,B2,A3,B3].
Feeding those into X_i(x) = A_i cos(lam x) + B_i sin(lam x) already gives the
mode. So why does the document rewrite it as eq (24)?

    X1 = cos(lam(L-x)) / cos(lam L)      X2 = X3 = sin(lam(L-x)) / sin(lam L)

Two claims to test here:

  (a) eq (24) is an IDENTITY -- the same function, rewritten. Nothing new.
  (b) the rewrite makes the boundary conditions visible. The argument becomes
      lam*(L-x), the distance measured backwards FROM THE FAR END x=L. So the
      trunk is a cosine centred on its reflecting end (zero slope there,
      automatically) and the branch is a sine centred on its absorbing tip
      (zero value there, automatically).

We also check all six conditions from Section 4-5 directly on the numbers,
and look at the node structure that distinguishes the +sqrt2 modes from the
-sqrt2 ones.

Needs step3_spectrum.py in the same folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import lambda_values_2branches as sp

L = 1.0
TRUNK, BRANCH_A, BRANCH_B = 0, 1, 2


# ---------------------------------------------------------------- two routes
def mode_raw(lam, x, L=L):
    """Route 1: build straight from the null vector. No formula assumed."""
    v, _ = sp.nullvector(lam, L)
    A1, B1, A2, B2, A3, B3 = v
    # normalise so the vertex height A is 1 (symmetric modes only)
    if abs(A1) > 1e-8:
        v = v / A1
        A1, B1, A2, B2, A3, B3 = v
    X = np.empty((3, x.size))
    X[0] = A1 * np.cos(lam * x) + B1 * np.sin(lam * x)
    X[1] = A2 * np.cos(lam * x) + B2 * np.sin(lam * x)
    X[2] = A3 * np.cos(lam * x) + B3 * np.sin(lam * x)
    return X


def mode_eq24(lam, x, L=L):
    """Route 2: the document's closed form, eq (24). Symmetric modes only."""
    X = np.empty((3, x.size))
    X[0] = np.cos(lam * (L - x)) / np.cos(lam * L)
    X[1] = np.sin(lam * (L - x)) / np.sin(lam * L)
    X[2] = X[1]
    return X


if __name__ == "__main__":
    x = np.linspace(0.0, L, 4001)
    roots = sp.find_roots(lam_max=12.0)
    sym = [r for r in roots if sp.classify(r)[0].strip() == "sym"]

    # ------------------------------------------------------- claim (a)
    print("CLAIM (a)  is eq (24) the same function as the raw null vector?\n")
    print(f"   {'lam*L':>9} {'tan(lam L)':>12} {'max |raw - eq24|':>18}")
    for lam in sym[:6]:
        d = np.max(np.abs(mode_raw(lam, x) - mode_eq24(lam, x)))
        print(f"   {lam * L:9.5f} {np.tan(lam * L):12.5f} {d:18.2e}")
    print("\n   -> identical. eq (24) is algebra, not new physics.")

    # ------------------------------------------------------- claim (b)
    # All six conditions, checked on the closed form directly.
    print("\nCLAIM (b)  do the six conditions hold, and are they now obvious?\n")
    h = 1e-6
    print(f"   {'lam*L':>9} {'X1_x(L)':>11} {'X2(L)':>11} {'X3(L)':>11} "
          f"{'X1(0)-X2(0)':>13} {'sum X_x(0)':>12}")
    for lam in sym[:6]:
        def Xf(xx):
            return mode_eq24(lam, np.atleast_1d(xx))

        dX1_L = (Xf(L)[0, 0] - Xf(L - h)[0, 0]) / h          # eq (2)
        X2_L, X3_L = Xf(L)[1, 0], Xf(L)[2, 0]                # eq (3)
        cont = Xf(0.0)[0, 0] - Xf(0.0)[1, 0]                 # eq (12)
        dsum = sum((Xf(h)[i, 0] - Xf(0.0)[i, 0]) / h for i in range(3))  # eq (14)
        print(f"   {lam * L:9.5f} {dX1_L:11.2e} {X2_L:11.2e} {X3_L:11.2e} "
              f"{cont:13.2e} {dsum:12.2e}")
    print("\n   -> reflecting end, both absorbing tips, continuity, flux:")
    print("      all satisfied without ever being imposed on eq (24).")

    # ------------------------------------------------------- node structure
    print("\nNODE STRUCTURE  (the claim made after eq 23)\n")
    print(f"   {'lam*L':>9} {'tan':>8} {'nodes on trunk':>16} "
          f"{'nodes on branch':>17}")
    for lam in sym[:6]:
        X = mode_eq24(lam, x)
        n_tr = np.sum(np.diff(np.sign(X[0])) != 0)
        n_br = np.sum(np.diff(np.sign(X[1][:-1])) != 0)   # exclude the tip zero
        print(f"   {lam * L:9.5f} {np.tan(lam * L):8.3f} {n_tr:16d} {n_br:17d}")

    # ------------------------------------------------------- picture
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, lam in zip(axes, sym[:3]):
        X = mode_eq24(lam, x)
        ax.plot(x, X[0], lw=2, label="trunk (reflecting at L)")
        ax.plot(x, X[1], lw=2, ls="--", label="branches (absorbing at L)")
        ax.axhline(0, color="k", lw=.6)
        ax.axvline(0, color="grey", lw=.8, ls=":")
        ax.set_title(rf"$\lambda L$ = {lam * L:.4f},  "
                     rf"$\tan\lambda L$ = {np.tan(lam * L):+.3f}")
        ax.set_xlabel("x   (0 = vertex,  L = far end)")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("X(x)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Symmetric eigenfunctions: all meet at height 1 at the vertex")
    fig.tight_layout()
    fig.savefig("step4_modes.png", dpi=150)
    print("\nwrote step4_modes.png")