import numpy as np
from units import dt, steps_to_time, eps

N = 50
n_walkers = 50000
a = 1/N
n_steps = 2 * N**2

rng = np.random.default_rng(100)

x = np.zeros(n_walkers)
# all walkers start at 0, x holds the position of all walkers

for step in range(n_steps):
    coin = rng.choice([-1, 1], size= n_walkers)
    # flipping a coin with two choices -1 or +1
    x = x + a * coin

msd = np.mean(x**2)
target = 2 * steps_to_time(n_steps, N)
#  <x^2> = 2Dt (known formula), D = 1
diff = 100 * (msd - target)/target

print("Pure Diffusion")
print(f"t = {steps_to_time(n_steps, N):.4f}")
print(f"<x^2> = {msd:.4f}")
print(f"target = {target:.4f}")
print(f"diff = {diff:.4f}")


Pe = 3.0
e = eps(Pe, N)
# coin bias, Pe/(2N) = 0.03
p_plus= 0.5 * (1 + e)
# probability of stepping +a

x = np.zeros(n_walkers)
# resetting the position of all walkers to 0

for step in range(n_steps):
    u = rng.random(n_walkers)
    coin = np.where(u < p_plus, 1, -1)
    # +1 with probability p_plus, else -1
    x = x + a * coin

t = steps_to_time(n_steps, N)
mean_x = np.mean(x)
target_mean = Pe * t # <x> = vt with v = Pe
std_err = np.sqrt(np.var(x) / n_walkers)
# statistical error of the mean
n_sigma = abs(mean_x - target_mean)/std_err

var_x = np.var(x)
target_var = 2 * t * (1 - e**2)
# 2D_eff t, D_eff = D(1 - eps^2)

diff_var = 100 * (var_x - target_var)/target_var

print("\nDrift")
print(f"eps = {e:.4f}")
print(f"<x> = {mean_x:.4f}    target = {target_mean:.4f}    off by {n_sigma:.2f} sigma")
print(f"Var(x) = {var_x:.4f}    target = {target_var:.4f}    diff = {diff_var:+.2f}\n")

try:
    eps(20, 50)
    print("GUARD FAILED")
except ValueError:
    print("guard works")
