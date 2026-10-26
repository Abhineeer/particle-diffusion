"""
test_star.py

Does the walk on the star graph agree with the exact answers?

    1. b = 1 must reduce to a plain segment of 2N spacings started at its
       midpoint. The strongest single test: the answer is known without
       any star reasoning, so a wrong vertex rule shows up here.
    2. h_V, the chance of reaching the reflecting trunk tip before being
       absorbed, must be 1/(1+b) EXACTLY. Every N cancels in the
       derivation, so there is no lattice correction to allow for.
    3. The mean time must match N(1+2N+bN)/b steps, and converge to the
       continuum (L^2/D)(1/2 + 1/b) as N grows.
"""

import numpy as np

import lattice
import star


if __name__ == "__main__":
    rng = np.random.default_rng(3)

    N = 60
    n_walkers = 40000

    print("CHECK 1   b = 1 must reproduce a plain segment of 2N spacings\n")
    _, _, steps = star.run_many(1, N, n_walkers, rng, start="vertex")
    M = 2 * N
    segment_answer = M*(M + 1) - N*(N + 1)      # T(m) = M(M+1) - m(m+1)
    print(f"   simulated mean steps     = {steps.mean():10.1f}")
    print(f"   star formula             = {star.mfpt_vertex_exact(1, N):10.1f}")
    print(f"   independent segment calc = {segment_answer:10.1f}")
    print(f"   formula == segment calc: "
          f"{abs(star.mfpt_vertex_exact(1, N) - segment_answer) < 1e-9}")

    print(f"\nCHECK 2   trunk-return probability h_V   (N = {N}, "
          f"{n_walkers} walkers)\n")
    print(f"   {'b':>3} {'measured':>10} {'1/(1+b)':>10} {'err':>8} "
          f"{'sigma':>8}")
    for b in (1, 2, 3, 5, 10):
        _, touched, _ = star.run_many(b, N, n_walkers, rng, start="vertex")
        hV = touched.mean()
        exact = star.h_vertex_exact(b)
        sigma = np.sqrt(hV * (1 - hV) / n_walkers)
        print(f"   {b:>3} {hV:>10.4f} {exact:>10.4f} {hV - exact:>8.4f} "
              f"{sigma:>8.4f}")

    print(f"\nCHECK 3   mean time, in the L = {lattice.L_DEFAULT}, "
          f"D = {lattice.D} convention\n")
    print(f"   {'b':>3} {'sim <t>':>10} {'lattice':>10} {'continuum':>11} "
          f"{'sim/lattice':>12} {'tip split':>22}")
    for b in (1, 2, 3, 5):
        branch, _, steps = star.run_many(b, N, n_walkers, rng,
                                         start="vertex")
        t = lattice.to_time(steps, N)
        lat = lattice.to_time(star.mfpt_vertex_exact(b, N), N)
        con = star.mfpt_vertex_continuum(b)
        split = " ".join(f"{(branch == i).mean():.3f}"
                         for i in range(1, min(b, 3) + 1))
        print(f"   {b:>3} {t.mean():>10.4f} {lat:>10.4f} {con:>11.4f} "
              f"{t.mean()/lat:>12.4f}   {split:>20}")
    print("\n   tip split should be uniform at 1/b (first three shown)")