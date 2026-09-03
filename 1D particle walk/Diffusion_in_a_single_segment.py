"""
Claim: p(x,t) = X(x) T(t) with
 
    T(t) = exp(-D lam^2 t)            eq (7)
    X(x) = A cos(lam x) + B sin(lam x)  eq (9)
 
solves the diffusion equation dp/dt = D d2p/dx2 for ANY lam, A, B.
 
No boundary conditions yet. No vertex yet. One segment, floating free.
The point is to confirm that separation of variables really does produce
solutions, and to see what the separation constant controls.
"""

import numpy as np

D = 1.0
L = 1.0

def X(x, lam, A, B):
    # eq (9) the spatial shape X(x)
    return A * np.cos(lam * x) + B * np.sin(lam * x)

def T(t, lam, D=D):
    # eq (7) the temporal decay T(t)
    return np.exp(-D * lam**2 * t)
 
 
def p(x, t, lam, A, B, D=D):
    # probability density
    return X(x, lam, A, B) * T(t, lam, D)


# Testing it satisfies the pde
# dp/dt - D d2p/dx2 = 0 by FD

rng = np.random.default_rng(1)
print("residual of  dp/dt - D d2p/dx2,  at random (lam, A, B):\n")
 
h = 1e-5
for _ in range(5):
    lam = rng.uniform(0.5, 8.0)
    A, B = rng.uniform(-2, 2, 2)
    x0, t0 = rng.uniform(0.2, 0.8), rng.uniform(0.05, 0.5)
 
    dpdt = (p(x0, t0 + h, lam, A, B) - p(x0, t0 - h, lam, A, B)) / (2 * h)
    d2pdx2 = (p(x0 + h, t0, lam, A, B) - 2 * p(x0, t0, lam, A, B)
              + p(x0 - h, t0, lam, A, B)) / h**2
 
    print(f"  lam={lam:5.2f}  A={A:+5.2f}  B={B:+5.2f}   "
          f"residual = {dpdt - D * d2pdx2:+.2e}")


# Test 2, why the separation constant must be negative
# -lam^2 (decay), we can try +lam^2, it will blow up

print("\nwhat the sign of the separation constant does, at t = 0, 1, 5, 20:")
print("(separation constant s;  T' = D s T  =>  T = exp(D s t))\n")
lam = 2.0
for label, s in [("s = -lam^2  (chosen)", -lam**2), ("s = +lam^2  (rejected)", +lam**2)]:
    vals = [np.exp(D * s * tt) for tt in (0, 1, 5, 20)]
    print(f"  {label:<24} " + "  ".join(f"{v:11.3e}" for v in vals))


# test 3, lambda (lam) controls the shape and decay rate both
# they are locked together

print("\nlam sets shape and rate together:\n")
print(f"  {'lam':>6} {'half-wavelength':>16} {'decay rate D lam^2':>20} "
      f"{'relaxation time':>17}")
for lam in [np.pi / 2, np.pi, 2 * np.pi, 4 * np.pi]:
    print(f"  {lam:6.3f} {np.pi / lam:16.4f} {D * lam**2:20.4f} "
          f"{1 / (D * lam**2):17.4f}")
 
print("\nFiner spatial structure decays faster.")
print("This single fact is why the late-time density forgets the initial condition.")
