import numpy as np
from segment import run_segment

# Checks run_segment against the exact mean first-passage time on one edge,
# reflecting at the root (x = 0), absorbing at the tip (x = 1), start at root:

#     Pe = 0:   <T> = 1/2
#     Pe > 0:   <T> = 1/p - (1 - e^{-p}) / p^2

# The Pe > 0 formula is the parent term of Eq. (6) in the write-up.
# Pass: every |n_sigma| < 3, with signs scattered.

def T_exact(p):
    # Exact <T> in units of tau for a single segment.
    if p == 0:
        return 0.5
    return 1/p - (1 - np.exp(-p)) / p**2


def check(N, Pe, n_walkers, rng):
    # Run one case and print simulated vs exact <T>.
    T = run_segment(N, n_walkers, rng, Pe)
    mean_T = np.mean(T)
    exact = T_exact(Pe)
    se = np.std(T) / np.sqrt(n_walkers) # statistical error of the mean
    n_sigma = (mean_T - exact) / se
    print(f"  N = {N:4d}   Pe = {Pe:4.1f}   <T> = {mean_T:.4f}   "
          f"exact = {exact:.4f}   off by {n_sigma:+.2f} sigma")


rng = np.random.default_rng(1)
n_walkers = 50000

print("Part 1: Pe sweep, N = 50")
for Pe in [0, 1, 3, 6]:
    check(50, Pe, n_walkers, rng)

print("Part 2: convergence, Pe = 6")
check(100, 6, n_walkers, rng)