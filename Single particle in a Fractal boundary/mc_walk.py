"""
Single particle, Monte Carlo random walk, fractal (Koch) Robin boundary.

Domain: box [0,L] x [0,H]. Bottom is the Koch curve (folds pointing up/into
the domain). Left, right, top walls: reflecting. Bottom: Robin, parameterized
by K (K=1 fully reflecting, K=0 fully absorbing), via the SAME convention as
the square-boundary code: h = (1-K)/D, per-collision absorption probability

    p_absorb = h*dx / (D + h*dx) = (1-K)*dx / (D**2 + (1-K)*dx)

"Land" (solid, forbidden) region = polygon formed by the Koch curve + the
baseline. A step is a boundary collision iff it crosses from outside-land
(domain) to inside-land. This is required because a Koch curve is NOT a
single-valued function of x (it folds back on itself), so a naive
y < f(x) test is wrong.
"""
import numpy as np
from matplotlib.path import Path
from geometry import build_boundary


class FractalBoundaryWalk:
    def __init__(self, L=1.0, H=1.0, depth=4, D=1.0, dx=0.01, K=0.5,
                 start=None, seed=None):
        self.L, self.H, self.depth, self.D, self.dx, self.K = L, H, depth, D, dx, K
        self.dt = dx**2 / (4 * D)  # 2D lattice walk, 4 directions

        self.boundary, self.arclen = build_boundary(L, H, depth)
        land_poly = np.vstack([self.boundary, [[L, 0.0], [0.0, 0.0]]])
        self.land_path = Path(land_poly)
        # envelope: land polygon can never extend above this y -> cheap pre-filter
        self.y_envelope = self.boundary[:, 1].max()

        h = (1.0 - K) / D
        self.p_absorb = (h * dx) / (D + h * dx)

        self.start = np.array(start if start is not None else [L / 2, H * 0.8])
        self.rng = np.random.default_rng(seed)

    def _reflect_walls(self, pos):
        # left/right/top box walls: simple mirror reflection
        x, y = pos[:, 0], pos[:, 1]
        x[x < 0] *= -1
        x[x > self.L] = 2 * self.L - x[x > self.L]
        y[y > self.H] = 2 * self.H - y[y > self.H]
        return pos

    def _nearest_arclength(self, points):
        """Project collision points onto the nearest boundary vertex, return arclength."""
        # boundary has up to a few thousand vertices at depth<=6: brute force is fine
        d2 = ((points[:, None, :] - self.boundary[None, :, :]) ** 2).sum(axis=2)
        idx = d2.argmin(axis=1)
        return self.arclen[idx]

    def run(self, n_walkers=2000, max_steps=20000):
        pos = np.tile(self.start, (n_walkers, 1)).astype(float)
        alive = np.ones(n_walkers, dtype=bool)
        t_absorbed = np.full(n_walkers, np.nan)
        s_absorbed = np.full(n_walkers, np.nan)  # arclength of absorption site
        survival_curve = np.zeros(max_steps + 1)
        survival_curve[0] = 1.0

        moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * self.dx

        for step in range(1, max_steps + 1):
            n_alive = alive.sum()
            if n_alive == 0:
                survival_curve[step:] = 0.0
                break

            idx_alive = np.where(alive)[0]
            choice = self.rng.integers(0, 4, size=n_alive)
            proposed = pos[idx_alive] + moves[choice]
            proposed = self._reflect_walls(proposed)

            near_boundary = proposed[:, 1] < (self.y_envelope + 2 * self.dx)
            crosses_land = np.zeros(n_alive, dtype=bool)
            if near_boundary.any():
                crosses_land[near_boundary] = self.land_path.contains_points(
                    proposed[near_boundary]
                )

            # non-colliding walkers: accept move
            free = ~crosses_land
            pos[idx_alive[free]] = proposed[free]

            # colliding walkers: absorb w.p. p_absorb, else reject move (stay put)
            coll_idx_local = np.where(crosses_land)[0]
            if coll_idx_local.size:
                r = self.rng.random(coll_idx_local.size)
                absorb_mask = r < self.p_absorb
                global_idx = idx_alive[coll_idx_local]

                absorbed_global = global_idx[absorb_mask]
                if absorbed_global.size:
                    coll_points = proposed[coll_idx_local][absorb_mask]
                    arcs = self._nearest_arclength(coll_points)
                    t_absorbed[absorbed_global] = step * self.dt
                    s_absorbed[absorbed_global] = arcs
                    alive[absorbed_global] = False
                # rejected (reflected) walkers just stay at old pos[idx] -> no-op

            survival_curve[step] = alive.sum() / n_walkers

        return {
            "survival_curve": survival_curve[: step + 1],
            "t_grid": np.arange(step + 1) * self.dt,
            "t_absorbed": t_absorbed,
            "s_absorbed": s_absorbed,
            "final_alive_frac": alive.sum() / n_walkers,
            "total_boundary_length": self.arclen[-1],
        }


if __name__ == "__main__":
    import time
    dx = 0.01
    D = 1.0
    dt = dx**2 / (4 * D)
    max_steps = int(3 * 0.64 / dt)
    print(f"dt={dt:.2e}, max_steps={max_steps}")

    for K in [0.99, 0.5, 0.05]:
        t0 = time.time()
        sim = FractalBoundaryWalk(depth=4, D=D, dx=dx, K=K, seed=0)
        res = sim.run(n_walkers=500, max_steps=max_steps)
        n_absorbed = np.sum(~np.isnan(res["t_absorbed"]))
        print(f"K={K:.2f}  p_absorb/collision={sim.p_absorb:.4f}  "
              f"absorbed={n_absorbed}/500  final_survival={res['final_alive_frac']:.3f}  "
              f"({time.time()-t0:.1f}s)")