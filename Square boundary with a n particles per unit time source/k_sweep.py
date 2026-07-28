import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Single Particle in a Square (D, K)")
from diffusion import solve_eigenvalues, X, Y

D = 1
N = 10.0          
# particles injected per unit time at centre
n_eigen = 20
Ngrid = 200

K_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
# Edgecase check
# K = 1 results in h = 0 and the eigenfunction breaks down
# K must approach 0 but never equal 0, at K = 1 particles never exit and they are performing random walks in the perimeter of the boundary
 
x_vals = np.linspace(0, 1, Ngrid)
y_vals = np.linspace(0, 1, Ngrid)

results = {}

for Kval in K_values:
    mu, h = solve_eigenvalues(Kval, D, n_eigen)
    mu = np.array(mu)
 
    Xmat = np.array([X(x_vals, m, h) for m in mu])
    Ymat = np.array([Y(y_vals, m, h) for m in mu])
    # precomputed table of eigenfunction values
 
    normX = np.trapezoid(Xmat**2, x_vals, axis=1)
    normY = np.trapezoid(Ymat**2, y_vals, axis=1)
    # || X_n ||^2 that is normX
 
    Xc = X(0.5, mu, h)
    Yc = Y(0.5, mu, h)
    cnm = np.outer(Xc, Yc) / np.outer(normX, normY)
    lam = mu[:, None] + mu[None, :]
    a_ss = (N * cnm) / (D * lam)
 
    rho_ss = np.einsum('nm,ni,mj->ji', a_ss, Xmat, Ymat)
    M_ss = np.trapezoid(np.trapezoid(rho_ss, x_vals, axis=1), y_vals)
 
    edge_bottom = np.trapezoid(rho_ss[0, :],  x_vals)
    edge_top    = np.trapezoid(rho_ss[-1, :], x_vals)
    edge_left   = np.trapezoid(rho_ss[:, 0],  y_vals)
    edge_right  = np.trapezoid(rho_ss[:, -1], y_vals)
    absorb = D * h * (edge_bottom + edge_top + edge_left + edge_right)
 
    results[Kval] = dict(rho_ss=rho_ss, M_ss=M_ss, absorb=absorb, h=h, mu=mu)
 
    print(f"K={Kval:5.2f}  h={h:6.3f}  M_ss={M_ss:10.5f}  "
          f"absorb={absorb:10.5f}  cons.err={abs(absorb-N)/N*100:6.3f}%")

# ------------------ PLOTS ------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
 
# (a) M_ss vs K
Ks = list(results.keys())
Mss_vals = [results[k]['M_ss'] for k in Ks]
axes[0].plot(Ks, Mss_vals, 'o-', color='crimson')
axes[0].set_xlabel('K'); axes[0].set_ylabel('steady-state population $M_{ss}$')
axes[0].set_title('Total population vs K')
axes[0].grid(alpha=0.3)
 
# (b) centre-line cross-section, all K overlaid
mid = Ngrid // 2
for Kval in Ks:
    axes[1].plot(x_vals, results[Kval]['rho_ss'][mid, :], label=f'K={Kval}')
axes[1].set_xlabel('x'); axes[1].set_ylabel(r'$\rho(x,0.5)$')
axes[1].set_title('Steady-state cross-section vs K')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
 
# (c) heatmaps for a couple of representative K
im = axes[2].imshow(results[Ks[-1]]['rho_ss'], cmap='viridis',
                     extent=[0, 1, 0, 1], origin='lower')
axes[2].set_title(f'Density field, K={Ks[-1]} (most reflecting)')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
 
plt.tight_layout()
plt.savefig('source_K_sweep.png', dpi=130, bbox_inches='tight')
print("saved source_K_sweep.png")
plt.show()
