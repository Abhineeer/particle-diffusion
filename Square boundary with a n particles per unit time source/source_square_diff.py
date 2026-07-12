"""
Continuous point source at the centre of the unit square.

A source injects N particles per unit time at (0.5, 0.5). The density obeys d(rho)/dt = D * laplacian(rho) + N * delta(x-0.5) delta(y-0.5)
with the same Robin boundary conditions as the single-particle problem.

Expanding rho in the same eigenbasis X_n(x) Y_m(y), each mode obeys da_nm/dt = -D (mu_n + mu_m) a_nm + N c_nm,     c_nm = X_n(0.5)Y_m(0.5)/(||X_n||^2 ||Y_m||^2) so
a_nm(t)     = (N c_nm)/(D lam_nm) * (1 - exp(-D lam_nm t)),   lam_nm = mu_n + mu_m
a_nm(ss)    = (N c_nm)/(D lam_nm)                              (t -> infinity)

Reuses eigenvalues, X, Y, h, D from diffusion.py so conventions match exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Single Particle in a Square (D, K)")
from diffusion import eigenvalues, X, Y, h, D, K

N = 10.0  # particles injected per unit time at the centre

mu = np.array(eigenvalues)
Ne = len(mu)

Ngrid = 200
xg = np.linspace(0, 1, Ngrid)
yg = np.linspace(0, 1, Ngrid)

# Precompute per-mode eigenfunction tables on the grid
# Xmat[n, i] = X_n(x_i),  Ymat[m, j] = Y_m(y_j)
Xmat = np.array([X(xg, m, h) for m in mu])          # (Ne, Ngrid)
Ymat = np.array([Y(yg, m, h) for m in mu])          # (Ne, Ngrid)

# Norms  ||X_n||^2 = integral X_n^2 dx   and integrals  IX_n = integral X_n dx
normX = np.trapezoid(Xmat**2, xg, axis=1)
normY = np.trapezoid(Ymat**2, yg, axis=1)
IX = np.trapezoid(Xmat, xg, axis=1)
IY = np.trapezoid(Ymat, yg, axis=1)

# Source coefficients c_nm and steady-state amplitudes a_nm(ss)
Xc = X(0.5, mu, h)
Yc = Y(0.5, mu, h)
cnm = np.outer(Xc, Yc) / np.outer(normX, normY)
lam = mu[:, None] + mu[None, :]
a_ss = (N * cnm) / (D * lam)

# ------------------ STEADY-STATE DENSITY FIELD ------------------
# rho_ss[j, i] = sum_nm a_ss[n,m] X_n(x_i) Y_m(y_j)
rho_ss = np.einsum('nm,ni,mj->ji', a_ss, Xmat, Ymat)

# ------------------ STATISTICS ------------------
# Total steady-state population  M = integral rho dA
M_ss = np.trapezoid(np.trapezoid(rho_ss, xg, axis=1), yg)

# Steady-state boundary absorption. Robin BC => outward flux per length = D*h*rho on every edge.
edge_bottom = np.trapezoid(rho_ss[0, :],  xg)
edge_top    = np.trapezoid(rho_ss[-1, :], xg)
edge_left   = np.trapezoid(rho_ss[:, 0],  yg)
edge_right  = np.trapezoid(rho_ss[:, -1], yg)
absorb = D * h * (edge_bottom + edge_top + edge_left + edge_right)

# Slowest mode sets the relaxation time
lam_min = 2 * mu.min()
tau = 1.0 / (D * lam_min)

print("="*52)
print(" CONTINUOUS POINT SOURCE  -  STEADY STATE")
print("="*52)
print(f" D = {D},  K = {K},  h = (1-K)/D = {h}")
print(f" source rate N            = {N}")
print(f" eigenmodes (1D)          = {Ne}   -> {Ne*Ne} 2D modes")
print("-"*52)
print(f" total population  M_ss   = {M_ss:.6f}")
print(f" boundary absorption rate = {absorb:.6f}   (should equal N)")
print(f" conservation error       = {abs(absorb - N)/N*100:.3f} %")
print(f" peak density (centre)    = {rho_ss[Ngrid//2, Ngrid//2]:.4f}  (truncation-limited)")
print(f" mean density             = {M_ss:.4f}")
print(f" slowest mode lam_min     = {lam_min:.4f}")
print(f" relaxation time tau      = {tau:.4f}")
print("="*52)

# ------------------ TIME APPROACH TO STEADY STATE ------------------
# M(t) = sum_nm a_nm(t) IX_n IY_m,  a_nm(t) = a_ss (1 - exp(-D lam t))
IXY = np.outer(IX, IY)                               # (Ne, Ne)
t_vals = np.linspace(0.0, 6*tau, 200)
M_t = np.array([
    np.sum(a_ss * (1 - np.exp(-D * lam * t)) * IXY) for t in t_vals
])

# ------------------ PLOTS ------------------
fig = plt.figure(figsize=(13, 4.2))

ax1 = fig.add_subplot(1, 3, 1)
im = ax1.imshow(rho_ss, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
ax1.set_title(f'Steady-state density  (D={D}, K={K}, N={N})')
ax1.set_xlabel('x'); ax1.set_ylabel('y')
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(1, 3, 2)
mid = Ngrid // 2
ax2.plot(xg, rho_ss[mid, :], lw=2)
ax2.set_title('Cross-section through centre (y = 0.5)')
ax2.set_xlabel('x'); ax2.set_ylabel(r'$\rho(x, 0.5)$')
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(t_vals, M_t, lw=2, label='M(t)')
ax3.axhline(M_ss, ls='--', color='crimson', label=f'M_ss = {M_ss:.3f}')
ax3.axvline(tau, ls=':', color='gray', label=fr'$\tau$ = {tau:.3f}')
ax3.set_title('Population approaching steady state')
ax3.set_xlabel('t'); ax3.set_ylabel('total population M(t)')
ax3.legend(); ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('steady_state.png', dpi=130, bbox_inches='tight')
print("saved steady_state.png")
