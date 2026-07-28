from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt

# ------------------ EIGENVALUE SOLVER (now a function of K) ------------------

def solve_eigenvalues(K, D=1, n_eigen=20, mu_max=10000, n_scan=10000):
    """
    Find the first n_eigen positive roots mu of the Robin transcendental
    equation for a given K (D fixed at D=1 by default, matching the rest
    of the project's convention h = (1-K)/D).

    Returns (eigenvalues, h).

    NOTE: K should be strictly less than 1. At K=1, h=0, and mu=0 becomes
    a legitimate root (the reflecting/no-flux zero mode), but this scan
    starts at mu=0.001 and the transcendental function f(mu,0) doesn't
    change sign near mu=0, so it will never be bracketed here anyway.
    Physically, K=1 also has no steady state under constant injection
    (a=0 mode has no decay), so a_ss diverges there regardless. Sweep
    K up to but not including 1.
    """
    h = (1 - K) / D
    mu_values = np.linspace(0.001, mu_max, n_scan)

    def f(mu, h):
        sq = np.sqrt(mu)
        return 2*h*np.cos(sq) - np.sin(sq)*(sq - h**2/sq)

    eigenvalues = []
    for i in range(len(mu_values) - 1):
        if f(mu_values[i], h) * f(mu_values[i+1], h) < 0:
            root = brentq(f, mu_values[i], mu_values[i+1], args=(h,))
            eigenvalues.append(root)
            if len(eigenvalues) == n_eigen:
                break

    return eigenvalues, h


# ------------------ CONSTANTS (module-level default, K=0.5) ------------------
D = 1
K = 0.5
n_eigen = 20
eigenvalues, h = solve_eigenvalues(K, D, n_eigen)


def X(x, mu, h):
    B = 1
    sq = np.sqrt(mu)
    A = h*B/sq
    return A*np.sin(sq*x) + B*np.cos(sq*x)

def Y(y, nu, h):
    Dy = 1
    sq = np.sqrt(nu)
    C = h*Dy/sq
    return C*np.sin(sq*y) + Dy*np.cos(sq*y)

def P(x, y, t, eigenvalues, h, D):
    x_vals = np.linspace(0, 1, 100)
    y_vals = np.linspace(0, 1, 100)
    total = 0
    for n in range(len(eigenvalues)):
        for m in range(len(eigenvalues)):
            mu_n = eigenvalues[n]
            mu_m = eigenvalues[m]

            norm_n = np.trapezoid(X(x_vals, mu_n, h)**2, x_vals)
            norm_m = np.trapezoid(Y(y_vals, mu_m, h)**2, y_vals)
            cnm = X(0.5, mu_n, h)*Y(0.5, mu_m, h)/(norm_m*norm_n)

            total += cnm * X(x, mu_n, h) * Y(y, mu_m, h) * np.e**(-D*(mu_m + mu_n)*t)

    return total

x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(0, 1, 100)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

times = [0.01, 0.05, 0.1, 0.5]
t_vals = np.linspace(0.0001, 5, 100)

def S(t, eigenvalues, h, D):
    Z = P(X_grid, Y_grid, t, eigenvalues, h, D)
    result = np.trapezoid(np.trapezoid(Z, x_vals, axis=1), y_vals)
    return result


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for i, t in enumerate(times):
        Z = P(X_grid, Y_grid, times[i], eigenvalues, h, D)
        ax = axes[i//2, i%2]
        ax.imshow(Z, cmap='viridis', extent=[0,1,0,1], origin='lower')
        ax.set_title(f't = {t}')

    plt.tight_layout()
    plt.show()

    S_vals = []
    for i, t in enumerate(t_vals):
        S_vals.append(S(t_vals[i], eigenvalues, h, D))

    J_vals = -np.gradient(S_vals, t_vals)

    plt.plot(t_vals, S_vals)
    plt.xlabel('t'); plt.ylabel('S')
    plt.title('Survival Probability')
    plt.show()

    plt.plot(t_vals, J_vals)
    plt.xlabel('t'); plt.ylabel('J')
    plt.title('Absorption Flux')
    plt.show()