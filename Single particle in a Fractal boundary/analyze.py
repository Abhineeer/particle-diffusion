import pickle
import numpy as np
import matplotlib.pyplot as plt
from geometry import build_boundary, fractal_dimension_koch

with open("Single particle in a Fractal boundary/benchmarks/sweep_results.pkl", "rb") as f:
    data = pickle.load(f)

results = data["results"]
meta = data["meta"]
K_values = data["K_values"]
boundary, arclen = build_boundary(L=meta["L"], H=meta["L"], depth=meta["depth"])
total_L = arclen[-1]
d_f = fractal_dimension_koch()

colors = plt.cm.viridis(np.linspace(0, 1, len(K_values)))

# ---------- Figure 1: survival curves ----------
fig, ax = plt.subplots(figsize=(7, 5))
for K, c in zip(K_values, colors):
    r = results[K]
    ax.plot(r["t_grid"], r["survival_curve"], color=c,
            label=f"K={K}  (Λ={r['Lambda']:.4f})")
ax.set_xlabel("time")
ax.set_ylabel("survival probability S(t)")
ax.set_title("Survival probability vs time, fractal Robin boundary (Koch, depth=4)")
ax.legend(fontsize=8)
ax.set_xscale("log")
plt.tight_layout()
plt.savefig("Single particle in a Fractal boundary/fig1_survival.png", dpi=140)
plt.close()

# ---------- Figure 2: absorption-site histograms along the boundary ----------
fig, axes = plt.subplots(len(K_values), 1, figsize=(8, 1.3 * len(K_values)), sharex=True)
bins = np.linspace(0, total_L, 150)
active_length = {}
for ax, K, c in zip(axes, K_values, colors):
    r = results[K]
    s = r["s_absorbed"]
    s = s[~np.isnan(s)]
    counts, edges = np.histogram(s, bins=bins)
    ax.bar(edges[:-1], counts, width=np.diff(edges), color=c, align="edge")
    ax.set_ylabel(f"K={K}", fontsize=8, rotation=0, labelpad=28, va="center")
    ax.set_yticks([])

    # inverse participation ratio -> effective active length
    p = counts / counts.sum() if counts.sum() > 0 else counts
    ipr = np.sum(p**2)
    bin_width = edges[1] - edges[0]
    L_active = (1.0 / ipr) * bin_width if ipr > 0 else np.nan
    active_length[K] = L_active

axes[-1].set_xlabel("arclength along fractal boundary")
fig.suptitle("Where absorption actually happens along the boundary, by K", y=1.0)
plt.tight_layout()
plt.savefig("Single particle in a Fractal boundary/fig2_active_zone.png", dpi=140)
plt.close()

# ---------- Figure 3: measured active length vs Lambda, vs Sapoval scaling ----------
Ks_finite = [K for K in K_values if K < 1.0]
Lambdas = np.array([results[K]["Lambda"] for K in Ks_finite])
L_actives = np.array([active_length[K] for K in Ks_finite])

# fit power law on the finite, non-nan points
mask = np.isfinite(L_actives) & (L_actives > 0) & np.isfinite(Lambdas) & (Lambdas > 0)
logL, logA = np.log(Lambdas[mask]), np.log(L_actives[mask])
slope, intercept = np.polyfit(logL, logA, 1)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(Lambdas[mask], L_actives[mask], color="#f2b035", zorder=3, label="measured (MC)")
fit_x = np.linspace(Lambdas[mask].min(), Lambdas[mask].max(), 50)
ax.plot(fit_x, np.exp(intercept) * fit_x**slope, "--", color="#f2b035",
        label=f"MC fit: slope={slope:.3f}")

predicted_slope = 1 - d_f
# anchor predicted line through the same midpoint for visual comparison
mid = len(fit_x) // 2
pred_intercept = logA.mean() - predicted_slope * logL.mean()
ax.plot(fit_x, np.exp(pred_intercept) * fit_x**predicted_slope, ":", color="#37bcb1",
        label=f"Sapoval prediction: slope={predicted_slope:.3f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("screening length Λ = D/h")
ax.set_ylabel("effective active length (IPR estimate)")
ax.set_title("Active-zone length vs screening length: MC vs Sapoval scaling")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("Single particle in a Fractal boundary", dpi=140)
plt.close()

print(f"fractal dimension d_f = {d_f:.4f}")
print(f"predicted slope (1-d_f) = {predicted_slope:.4f}")
print(f"measured MC slope       = {slope:.4f}")
print()
for K in Ks_finite:
    print(f"K={K:<6} Lambda={results[K]['Lambda']:<10.5f} L_active={active_length[K]:.5f}")