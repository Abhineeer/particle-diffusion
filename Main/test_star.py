"""
test_star.py

Does the walk on the star graph agree with the exact answers?

Three checks, in increasing order of how much they can catch:

    1. b = 1 reduces to a plain segment of length 2N started at its
       midpoint. This is the strongest single test available: if the
       vertex rule is wrong, b = 1 is where it shows, because the
       answer is known independently of any star reasoning.

    2. h_V, the chance of reaching the reflecting trunk tip before
       being absorbed, must be 1/(1+b) exactly.

    3. The mean absorption time must be N(1 + 2N + bN)/b steps, and the
       branch that catches the walker must be uniform at 1/b.
"""

import numpy as np

import lattice
import star


if __name__ == "__main__":
    rng = np.random.default_rng(3)

    N = 60
    n_walkers = 40000

    # ----------------------------------------------------------- check 1
    print("CHECK 1   b = 1 must reproduce a plain segment of length 2N\n")
    _, _, steps = star.run_many(1, N, n_walkers, rng, start="vertex")
    M = 2 * N
    # segment of M spacings, started m = N from the reflecting end:
    #   T(m) = M(M+1) - m(m+1)
    segment_answer = M*(M + 1) - N*(N + 1)
    print(f"   simulated mean steps     = {steps.mean():10.1f}")
    print(f"   star formula             = {star.mfpt_vertex_exact(1, N):10.1f}")
    print(f"   independent segment calc = {segment_answer:10.1f}")
    print(f"   -> the star formula and the segment calc agree: "
          f"{abs(star.mfpt_vertex_exact(1, N) - segment_answer) < 1e-9}")

    # ----------------------------------------------------------- check 2
    print(f"\nCHECK 2   trunk-return probability h_V      (N = {N}, "
          f"{n_walkers} walkers)\n")
    print(f"   {'b':>3} {'measured':>10} {'1/(1+b)':>10} {'err':>8} "
          f"{'sigma':>8}")
    for b in (1, 2, 3, 4, 5, 6, 10):
        _, touched, _ = star.run_many(b, N, n_walkers, rng, start="vertex")
        hV = touched.mean()
        exact = star.h_vertex_exact(b)
        sigma = np.sqrt(hV * (1 - hV) / n_walkers)
        print(f"   {b:>3} {hV:>10.4f} {exact:>10.4f} {hV - exact:>8.4f} "
              f"{sigma:>8.4f}")

    # ----------------------------------------------------------- check 3
    print(f"\nCHECK 3   mean absorption time and tip split\n")
    print(f"   {'b':>3} {'mean steps':>12} {'exact':>12} {'ratio':>8} "
          f"{'tip split':>28}")
    for b in (1, 2, 3, 5):
        branch, _, steps = star.run_many(b, N, n_walkers, rng, start="vertex")
        exact = star.mfpt_vertex_exact(b, N)
        split = [(branch == i).mean() for i in range(1, b + 1)]
        split_str = " ".join(f"{s:.3f}" for s in split)
        print(f"   {b:>3} {steps.mean():>12.1f} {exact:>12.1f} "
              f"{steps.mean()/exact:>8.4f}   {split_str:>26}")
    print(f"\n   tip split should be uniform at 1/b")

    # ------------------------------------------------- continuum sanity
    print(f"\nin physical time, leading order says <t> -> L^2 (1/2 + 1/b)/D\n")
    L = lattice.length(N)
    print(f"   {'b':>3} {'exact time':>12} {'L^2(1/2+1/b)':>14} {'ratio':>8}")
    for b in (1, 2, 3, 5, 10):
        exact_t = lattice.to_time(star.mfpt_vertex_exact(b, N))
        continuum = L**2 * (0.5 + 1.0/b) / lattice.D
        print(f"   {b:>3} {exact_t:>12.1f} {continuum:>14.1f} "
              f"{exact_t/continuum:>8.4f}")