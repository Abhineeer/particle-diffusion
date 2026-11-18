"""
Data for Figure 2: no flow, equal areas, b = 1..5.
Saves absorption times to data/fig2.npz so fig2.py never reruns the walk.
"""

import numpy as np
import os
from star import run_star

N = 50
N_WALKERS = 50_000
B_VALUES = [1, 2, 3, 4, 5]

rng = np.random.default_rng(2026)
out = {}
for b in B_VALUES:
    T, _ = run_star(b, N, N_WALKERS, rng, Pe=0.0, beta=float(b))
    out[f"T_b{b}"] = T
    se = T.std() / np.sqrt(T.size)
    print(f"b = {b}:  <T> = {T.mean():.4f} +/- {se:.4f}   exact 1 + 1/b = {1 + 1/b:.4f}")

os.makedirs("data", exist_ok=True)
np.savez("data/fig2.npz", N=N, b_values=B_VALUES, **out)