"""
The FD solver, no eigenvalues yet though

Diffusion equation goes forward in time on the network and see what happens.

Geometry:
every segment is x --> [0,L], x = 0 for all is at the vertex, x = L is the far end

segment 0 = trunk --> reflecting at x = L
segment 1 = branch A --> absorbing at x = L
segment 2 = branch B --> absorbing at x = L

Vertex is the tricky par
Continuity and the Junction law is what governs it

Probability flows into it from all three arms and it stores what arrives:
 
d(Pv)/dt = [ sum over arms of  D (P_i[1] - Pv)/dx ] / (3 dx / 2)
         = (2D / (3 dx^2)) * (P_0[1] + P_1[1] + P_2[1] - 3 Pv)


Both vertex conditions from Section 4 are built into this one line:
    - continuity, eq (13):  there is a SINGLE number Pv, not three
    - flux conservation, eq (15):  the three inflows are summed and nothing
      is added or discarded, so the vertex neither creates nor destroys

"""

import numpy as np

TRUNK, BRANCH_A, BRANCH_B = 0, 1, 2

def make_grid(L=1.0, N=401):
    # N nodes per segment. x[0] is the vertex, x[-1] the far end.
    return np.linspace(0.0, L, N)

def rhs(P, dx, D, absorbing=True):
    #  dP/dt for the whole network. P has shape (3, N).
    dP = np.zeros_like(P)
 
    # interior of every segment: the ordinary stencil
    dP[:, 1:-1] = D * (P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]) / dx**2
 
    # the vertex (node 0, shared by all three arms)
    Pv = P[0, 0]
    dP[:, 0] = (2 * D / (3 * dx**2)) * (P[0, 1] + P[1, 1] + P[2, 1] - 3 * Pv)
 
    # --- trunk far end: reflecting. Ghost node P[N] = P[N-2] makes the
    #     slope zero there; substituting into the stencil gives:
    dP[TRUNK, -1] = 2 * D * (P[TRUNK, -2] - P[TRUNK, -1]) / dx**2
 
    # --- branch tips
    if absorbing:
        dP[BRANCH_A, -1] = 0.0      # held pinned at zero
        dP[BRANCH_B, -1] = 0.0
    else:
        for s in (BRANCH_A, BRANCH_B):   # reflecting instead, for the closed test
            dP[s, -1] = 2 * D * (P[s, -2] - P[s, -1]) / dx**2
    return dP

def evolve(P0, x, D=1.0, save_times=(0.1,), safety=0.2, absorbing=True):
    # Forward Euler. Returns (times, [snapshots]).
    dx = x[1] - x[0]
    dt = safety * dx**2 / D          # explicit stability limit is dx^2/(2D)
    P = P0.copy()
    if absorbing:
        P[BRANCH_A, -1] = P[BRANCH_B, -1] = 0.0
 
    targets = np.sort(np.atleast_1d(save_times))
    out_t, snaps = [], []
    t = 0.0
    for target in targets:
        while t < target - 1e-15:
            h = min(dt, target - t)
            P = P + h * rhs(P, dx, D, absorbing)
            if absorbing:
                P[BRANCH_A, -1] = P[BRANCH_B, -1] = 0.0
            t += h
        out_t.append(t)
        snaps.append(P.copy())
    return np.array(out_t), snaps

def mass(P, x):
    # Total probability. Trapezoid weights the vertex half-cell three times,
    # which is exactly its control volume -- so this is consistent with rhs().
    return float(np.sum(np.trapezoid(P, x, axis=1)))

def point_source(x, seg, x0):
    # Normalised delta: one node carrying 1/dx.
    P = np.zeros((3, x.size))
    P[seg, int(np.argmin(np.abs(x - x0)))] = 1.0 / (x[1] - x[0])
    return P


if __name__ == "__main__":
    L, D = 1.0, 1.0
    x = make_grid(L, 401)
 
    # -- TEST 1 --
    # Seal the network: make the branch tips reflecting too. Nothing can
    # escape, so probability must be conserved EXACTLY. This is the real
    # test of the vertex rule -- a wrong vertex leaks.
    print("TEST 1 sealed network (all ends reflecting): is mass conserved?\n")
    x_coarse = make_grid(L, 101)
    P0 = point_source(x_coarse, TRUNK, 0.7 * L)
    ts, snaps = evolve(P0, x_coarse, D, save_times=[0.01, 0.1, 1.0, 5.0], absorbing=False)
    m0 = mass(P0, x_coarse)
    for t, P in zip(ts, snaps):
        print(f"   t = {t:5.2f}   mass = {mass(P, x_coarse):.12f}   "
              f"drift = {mass(P, x_coarse) - m0:+.2e}")
 
    # and it must relax to a flat density 1/(3L) everywhere
    Pf = snaps[-1]
    print(f"\n   final density: min {Pf.min():.6f}  max {Pf.max():.6f}   "
          f"expected uniform {1 / (3 * L):.6f}")
 
    # -- TEST 2 --
    # A uniform density must be a steady state: dP/dt = 0 everywhere,
    # INCLUDING at the vertex. If the vertex coefficient were wrong, the
    # vertex node alone would twitch.
    print("\nTEST 2  is a uniform density a fixed point?\n")
    Pu = np.ones((3, x.size))
    d = rhs(Pu, x[1] - x[0], D, absorbing=False)
    print(f"   max|dP/dt| over interior = {np.max(np.abs(d[:, 1:-1])):.2e}")
    print(f"   |dP/dt| at the vertex    = {abs(d[0, 0]):.2e}")
 
    # -- TEST 3 ---
    # Start symmetrically -> the two branches must stay bit-for-bit identical.
    # This is the branch-exchange symmetry the document leans on in Section 7.
    print("\nTEST 3  symmetric start: do the branches stay identical?\n")
    P0 = point_source(x, TRUNK, 0.5 * L)
    ts, snaps = evolve(P0, x, D, save_times=[0.05, 0.5], absorbing=True)
    for t, P in zip(ts, snaps):
        print(f"   t = {t:4.2f}   max|branchA - branchB| = "
              f"{np.max(np.abs(P[BRANCH_A] - P[BRANCH_B])):.2e}")
 
    # -- TEST 4 --
    # Now let it absorb. Mass should decay monotonically from 1.
    print("\nTEST 4  absorbing tips: survival S(t)\n")
    P0 = point_source(x, TRUNK, 0.5 * L)
    tsave = [0.0 + 1e-9, 0.02, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0]
    ts, snaps = evolve(P0, x, D, save_times=tsave, absorbing=True)
    print(f"   {'t':>6} {'S(t)':>10}   {'vertex height':>14}")
    for t, P in zip(ts, snaps):
        print(f"   {t:6.3f} {mass(P, x):10.6f}   {P[0, 0]:14.6f}")
 
    # -- TEST 5 --
    # Grid convergence: halve dx, the answer should settle.
    print("\nTEST 5  does the answer converge as the grid refines?\n")
    print(f"   {'N':>6} {'dx':>10} {'S(0.5)':>12}")
    for N in (101, 201, 401, 801):
        xx = make_grid(L, N)
        _, sn = evolve(point_source(xx, TRUNK, 0.5 * L), xx, D,
                       save_times=[0.5], absorbing=True)
        print(f"   {N:6d} {xx[1] - xx[0]:10.5f} {mass(sn[0], xx):12.8f}")

