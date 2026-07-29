"""
Finite-difference solver for the continuous point source N/t at the
centre of the unit square, with Robin BCs on all four walls.

PDE:   d(rho)/dt = D * laplacian(rho) + N * delta(x-0.5) delta(y-0.5)

Validates against the analytic eigenfunction-expansion steady state
from diffusion.py / source_K_sweep.py (same D, K, N).

Uses the same ghost-value Robin BC convention as the single-particle
solver: k = D / (D + h*dx), where the ghost point outside the
boundary is set to k times the first interior point (k=1 -> fully
reflecting/Neumann, k=0 -> fully absorbing).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Single Particle in a Square (D, K)")
from diffusion import solve_eigenvalues, X, Y

# ------------------ PARAMETERS ------------------
D = 1.0
K = 0.5
h = (1 - K) / D
N = 10.0            # particles injected per unit time at centre

Ngrid = 101          # interior + boundary points per axis
dx = 1.0 / (Ngrid - 1)
xg = np.linspace(0, 1, Ngrid)
yg = np.linspace(0, 1, Ngrid)

# Robin ghost-node BC, derived from the central-difference form of the
# Robin condition actually satisfied by the eigenfunctions (verified
# numerically): X'(0) = +h*X(0) at the low wall, X'(1) = -h*X(1) at the
# high wall (the sign flips because the outward normal flips). Central
# difference through the boundary node itself gives:
#   ghost_low  = P[1]   - 2*dx*h*P[0]
#   ghost_high = P[N-2] - 2*dx*h*P[N-1]

# Stability: forward Euler, 5-point Laplacian in 2D -> D*dt/dx^2 <= 0.25
safety = 0.4
dt = safety * dx**2 / (4 * D)

n_eigen = 20
T_max = 8.0          # run long enough to reach steady state; adjust if needed
n_steps = int(T_max / dt)

# Source: delta function on a grid must inject total rate N, so the
# grid cell at the centre gets N / dx^2 (since integral of a delta
# spike of height A over a dx*dx cell is A*dx^2, we need A = N/dx^2
# for that integral to equal N).
ic = Ngrid // 2  # centre index (grid has odd Ngrid so 0.5 lands exactly on a node)
source = np.zeros((Ngrid, Ngrid))
source[ic, ic] = N / dx**2

print(f"D={D}  K={K}  h={h:.4f}")
print(f"dx={dx:.5f}  dt={dt:.6f}  steps={n_steps}  (CFL number D*dt/dx^2 = {D*dt/dx**2:.4f})")

# ------------------ FD TIME STEPPING ------------------
rho = np.zeros((Ngrid, Ngrid))

M_t = np.zeros(n_steps)
t_vals = np.arange(n_steps) * dt

for step in range(n_steps):
    # Build padded array with ghost values for Robin BC on all 4 sides
    padded = np.zeros((Ngrid + 2, Ngrid + 2))
    padded[1:-1, 1:-1] = rho
    padded[0, 1:-1]    = rho[1, :]  - 2*dx*h*rho[0, :]    # bottom ghost (y=0 row)
    padded[-1, 1:-1]   = rho[-2, :] - 2*dx*h*rho[-1, :]   # top ghost    (y=1 row)
    padded[1:-1, 0]    = rho[:, 1]  - 2*dx*h*rho[:, 0]    # left ghost   (x=0 col)
    padded[1:-1, -1]   = rho[:, -2] - 2*dx*h*rho[:, -1]   # right ghost  (x=1 col)

    lap = (
        padded[2:, 1:-1] + padded[:-2, 1:-1] +
        padded[1:-1, 2:] + padded[1:-1, :-2] -
        4 * padded[1:-1, 1:-1]
    ) / dx**2

    rho = rho + dt * (D * lap + source)

    M_t[step] = np.trapezoid(np.trapezoid(rho, xg, axis=1), yg)

rho_fd = rho
M_fd = M_t[-1]

# Boundary flux at final time (Robin: flux per length = D*h*rho on each edge)
edge_bottom = np.trapezoid(rho_fd[0, :],  xg)
edge_top    = np.trapezoid(rho_fd[-1, :], xg)
edge_left   = np.trapezoid(rho_fd[:, 0],  yg)
edge_right  = np.trapezoid(rho_fd[:, -1], yg)
absorb_fd = D * h * (edge_bottom + edge_top + edge_left + edge_right)

# ------------------ ANALYTIC STEADY STATE (for comparison) ------------------
mu, h_check = solve_eigenvalues(K, D, n_eigen)
mu = np.array(mu)
assert abs(h_check - h) < 1e-12

Xmat = np.array([X(xg, m, h) for m in mu])
Ymat = np.array([Y(yg, m, h) for m in mu])
normX = np.trapezoid(Xmat**2, xg, axis=1)
normY = np.trapezoid(Ymat**2, yg, axis=1)
Xc = X(0.5, mu, h)
Yc = Y(0.5, mu, h)
cnm = np.outer(Xc, Yc) / np.outer(normX, normY)
lam = mu[:, None] + mu[None, :]
a_ss = (N * cnm) / (D * lam)
rho_analytic = np.einsum('nm,ni,mj->ji', a_ss, Xmat, Ymat)
M_analytic = np.trapezoid(np.trapezoid(rho_analytic, xg, axis=1), yg)

edge_bottom_a = np.trapezoid(rho_analytic[0, :],  xg)
edge_top_a    = np.trapezoid(rho_analytic[-1, :], xg)
edge_left_a   = np.trapezoid(rho_analytic[:, 0],  yg)
edge_right_a  = np.trapezoid(rho_analytic[:, -1], yg)
absorb_analytic = D * h * (edge_bottom_a + edge_top_a + edge_left_a + edge_right_a)

# ------------------ REPORT ------------------
print("=" * 56)
print(" FD vs ANALYTIC STEADY STATE  -  CONTINUOUS SOURCE")
print("=" * 56)
print(f" M_ss   analytic = {M_analytic:.5f}")
print(f" M_ss   FD (t={T_max})  = {M_fd:.5f}")
print(f" M_ss   rel. error       = {abs(M_fd-M_analytic)/M_analytic*100:.3f} %")
print("-" * 56)
print(f" absorb analytic = {absorb_analytic:.5f}  (should be ~N={N})")
print(f" absorb FD       = {absorb_fd:.5f}")
print(f" absorb rel. error        = {abs(absorb_fd-absorb_analytic)/absorb_analytic*100:.3f} %")
print("=" * 56)

# ------------------ PLOTS ------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

im0 = axes[0].imshow(rho_fd, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
axes[0].set_title(f'FD steady state (t={T_max})')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(rho_analytic, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
axes[1].set_title('Analytic steady state')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

mid = Ngrid // 2
axes[2].plot(xg, rho_fd[mid, :], label='FD', lw=2)
axes[2].plot(xg, rho_analytic[mid, :], '--', label='analytic', lw=2)
axes[2].set_xlabel('x'); axes[2].set_ylabel(r'$\rho(x,0.5)$')
axes[2].set_title('Cross-section: FD vs analytic')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('source_fd_vs_analytic.png', dpi=130, bbox_inches='tight')
print("saved source_fd_vs_analytic.png")
plt.show()

# Population approach to steady state
plt.figure(figsize=(6, 4.3))
plt.plot(t_vals, M_t, label='M(t) - FD')
plt.axhline(M_analytic, ls='--', color='crimson', label=f'analytic M_ss={M_analytic:.3f}')
plt.xlabel('t'); plt.ylabel('total population M(t)')
plt.title('FD population approaching analytic steady state')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('source_fd_population.png', dpi=130, bbox_inches='tight')
print("saved source_fd_population.png")
plt.show()