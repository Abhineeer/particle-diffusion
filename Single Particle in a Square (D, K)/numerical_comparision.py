import numpy as np
import diffusion
import matplotlib.pyplot as plt

N = 100
D = diffusion.D
# Diffusion constant
h = diffusion.h
dx = 1/N
# nnumber of steps in the grid, since it is a square dx = dy. We will use dx for both

# print(dx**2/(4*D))
dt = 0.000015
# Has to be less than the above number - CFL rule for the probabilities to not blow up

P = np.zeros((N+1, N+1))
row_cen = N//2
col_cen = N//2
P[row_cen, col_cen] = 1/dx**2


def update_rule(P, D, dx, dt):
    P_new = P.copy()

    P_new[1:-1, 1:-1] = P[1:-1, 1:-1] + ((dt*D)/dx**2)*(P[2:, 1:-1] - 2*P[1:-1, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] - 2*P[1:-1, 1:-1] + P[1:-1, :-2])
    return P_new


def apply_bc(P, D, h, dx):
    k = D/(D + h*dx)
    P[0, 1:-1] = k*P[1, 1:-1]
    P[-1, 1:-1] = k*P[-2, 1:-1]
    P[1:-1, 0] = k*P[1:-1, 1]
    P[1:-1, -1] = k*P[1:-1, -2]
    # This is the Robin BC being applied to the grid (2D)
    # We use numpy slicing to ease the process

    P[0, 0] = k * (P[1, 0] + P[0, 1]) / 2
    P[0, -1] = k * (P[1, -1] + P[0, -2]) / 2
    P[-1, 0] = k * (P[-2, 0] + P[-1, 1]) / 2
    P[-1, -1] = k * (P[-2, -1] + P[-1, -2]) / 2
    # The corners are missed because of the indices, we could include them in each of the slices above but then we risk setting and then setting corners again because they overlap in the x and y directions.
    # I have decided to average out the value for the corenrs from the two neighbours
    # We can do this and not cost ourselves a huge change in error because the initial probability is confined in the middle of the grid and these are the farthest points - even if there is an error it will be insanely diminished because of how small the weightage of the corners is on the whole probability
    
    return P

def S_numeric(P, dx):
    S = np.trapezoid(np.trapezoid(P, dx=dx, axis=0), dx=dx)
    return S

t = 1
n = int(t//dt)

S_vals = []
t_vals_numeric = []
P_new = P.copy()

for i in range(n):
    P_new = update_rule(P_new, D, dx, dt)
    P_new = apply_bc(P_new, D, h, dx)
    
    if(i%300==0):
        S_vals.append(S_numeric(P_new, dx))
        t_vals_numeric.append(i*dt)
    
# plt.plot(t_vals_numeric, S_vals, label="S(t): Survival Probability")
# plt.xlabel("Time")
# plt.ylabel("Survival Probability")
# plt.title("S(t) v/s t")
# plt.legend()
# plt.show()

t_vals_analytic = np.linspace(0.001, 1, 100)
S_vals_analytic = [diffusion.S(t, diffusion.eigenvalues, diffusion.h, diffusion.D) for t in t_vals_analytic]

plt.plot(t_vals_numeric, S_vals, label="Numerical")
plt.plot(t_vals_analytic, S_vals_analytic, '--', label="Analytic")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.title("S(t): Analytic vs Numerical")
plt.legend()
plt.show()

J_vals_numeric = -np.gradient(S_vals, t_vals_numeric)

t_vals_analytic = np.linspace(0.001, 1, 100)
S_vals_analytic = [diffusion.S(t, diffusion.eigenvalues, diffusion.h, diffusion.D) for t in t_vals_analytic]
J_vals_analytic = -np.gradient(S_vals_analytic, t_vals_analytic)

plt.plot(t_vals_numeric, J_vals_numeric, label="Numerical")
plt.plot(t_vals_analytic, J_vals_analytic, '--', label="Analytic")
plt.xlabel("Time")
plt.ylabel("J(t)")
plt.title("Absorption Flux: Analytic vs Numerical")
plt.legend()
plt.show()