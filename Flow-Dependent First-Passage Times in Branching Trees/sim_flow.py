"""
Data for Figures 3 (equal areas) and 4 (Murray's law), and the SI grid.
Runs b = 1, 2, 3 at Pe = Pe_c * {1/4, 1/2, 1, 2, 4}, Pe_c from analytic.py.
N = 50: with the flow-biased junction rule the lattice error in <T> is
about 0.02%, far below the statistical error.

Usage:  python sim_flow.py equal      ->  data/flow_equal.npz
        python sim_flow.py murray     ->  data/flow_murray.npz
"""

import sys
import numpy as np
from star import run_star
from analytic import beta_of, mean_T, pe_critical

model = sys.argv[1]
N = 50
N_WALKERS = 50_000
B_VALUES = [1, 2, 3, 4, 5]
PE_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]

pe_c = pe_critical(2, model)
rng = np.random.default_rng(11 if model == "equal" else 12)
out = {}
for b in B_VALUES:
    beta = beta_of(b, model)
    for f in PE_FACTORS:
        Pe = f * pe_c
        T, _ = run_star(b, N, N_WALKERS, rng, Pe=Pe, beta=beta)
        out[f"T_b{b}_f{f}"] = T
        se = T.std() / np.sqrt(T.size)
        exact = mean_T(Pe, beta)
        print(f"{model} b = {b}  Pe = {Pe:.3f}:  <T> = {T.mean():.4f} +/- {se:.4f}"
              f"   exact {exact:.4f}   ({(T.mean() - exact) / se:+.2f} sigma)",
              flush=True)

np.savez(f"data/flow_{model}.npz", N=N, pe_c=pe_c, b_values=B_VALUES,
         pe_factors=PE_FACTORS, model=model, **out)