import time
import numpy as np
import pickle
from mc_walk import FractalBoundaryWalk

D = 0.05
dx = 0.006
depth = 4
L = 1.0
N_WALKERS = 1500

dt = dx**2 / (4 * D)
t_cross = L**2 / D
max_steps = int(2 * t_cross / dt)

K_values = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999]

results = {}
t_start_all = time.time()
for K in K_values:
    t0 = time.time()
    sim = FractalBoundaryWalk(L=L, H=L, depth=depth, D=D, dx=dx, K=K, seed=42)
    res = sim.run(n_walkers=N_WALKERS, max_steps=max_steps)
    h = (1 - K) / D
    Lambda = D / h if h > 0 else np.inf
    res["K"] = K
    res["Lambda"] = Lambda
    res["p_absorb_per_collision"] = sim.p_absorb
    results[K] = res
    n_abs = np.sum(~np.isnan(res["t_absorbed"]))
    print(f"K={K:<6} Lambda={Lambda:<10.5f} p_abs/coll={sim.p_absorb:.5f}  "
          f"absorbed={n_abs}/{N_WALKERS}  survival={res['final_alive_frac']:.3f}  "
          f"({time.time()-t0:.1f}s)")

print(f"total time: {time.time()-t_start_all:.1f}s")

meta = dict(D=D, dx=dx, depth=depth, L=L, N_WALKERS=N_WALKERS, dt=dt, max_steps=max_steps)
with open("Single particle in a Fractal boundary/benchmarks/sweep_results.pkl", "wb") as f:
    pickle.dump({"results": results, "meta": meta, "K_values": K_values}, f)
print("saved sweep_results.pkl")