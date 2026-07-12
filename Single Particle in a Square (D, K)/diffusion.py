from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt

# ------------------ CONSTANTS ------------------
D = 1
K = 0.5 # diffusion constant and reflective rate
h = (1 - K)/D
n_eigen = 20 # number of eigenvalues to find (capping it here)

mu_values = np.linspace(0.001, 10000, 10000)

def f(mu, h):
    sq = np.sqrt(mu)
    return 2*h*np.cos(sq) - np.sin(sq)*(sq - h**2/sq)

eigenvalues = [] # will hold the roots

for i in range(len(mu_values) - 1):
    if f(mu_values[i], h) * f(mu_values[i+1], h) < 0:
        root = brentq(f, mu_values[i], mu_values[i+1], args=(h))
        # brentq basically finds out a single root in a continuous function - like the one we have
        # it looks for a sign change within brackets, a lower and upper - mu values i+1 and i in our case
        # we check if the sign change occurs and then give it to brentq to handle finding the root, it also handles extra inputs like in our case h
        eigenvalues.append(root)
        if len(eigenvalues) == n_eigen:
            break

# print(eigenvalues)

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
            # we need a set of eigenvalues
            # a pair, they can be the same or different but the above for loops help us do that

            norm_n = np.trapezoid(X(x_vals, mu_n, h)**2, x_vals)
            norm_m = np.trapezoid(Y(y_vals, mu_m, h)**2, y_vals)
            # Integration with trapezoid
            cnm = X(0.5, mu_n, h)*Y(0.5, mu_m, h)/(norm_m*norm_n)
            # calculating cnm with the help of the IC: P(x,y,0)

            total += cnm * X(x, mu_n, h) * Y(y, mu_m, h) * np.e**(-D*(mu_m + mu_n)*t)
            # Summing up all the values we get, in a way this is summation
    
    return total

# print(P(0.5, 0.5, 0.01, eigenvalues, h, D)) - TEST

x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(0, 1, 100)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
# X_grid and Y_grid are both 2D coordinates
# We use those to get a 2D array of P and then plot that

# --------- TIMES (Changeable) ---------------
times = [0.01, 0.05, 0.1, 0.5]
t_vals = np.linspace(0.001, 5, 100)

def S(t, eigenvalues, h, D):
    Z = P(X_grid, Y_grid, t, eigenvalues, h, D)
    result = np.trapezoid(np.trapezoid(Z, x_vals, axis=1), y_vals)
    return result



if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    for i, t in enumerate(times):
        Z = P(X_grid, Y_grid, times[i], eigenvalues, h, D)
        # print(S(times[i], eigenvalues, h, D))
        ax = axes[i//2, i%2]
        ax.imshow(Z, cmap='viridis', extent=[0,1,0,1], origin='lower')
        ax.set_title(f't = {t}')

        # plt.imshow(Z, cmap='viridis', extent=[0,1,0,1])
        # # imshow is used to display images in a plot, cmap manages color
        # # extent manages the bounding box of the plot - range of x and y values in this case
        # plt.colorbar() 
        # plt.xlabel('x')
        # plt.ylabel('y')

    plt.tight_layout()
    plt.show()

    S_vals = []

    for i, t in enumerate(t_vals):
        S_vals.append(S(t_vals[i], eigenvalues, h, D))

    J_vals = -np.gradient(S_vals, t_vals)

    plt.plot(t_vals, S_vals)
    plt.xlabel('t')
    plt.ylabel('S')
    plt.title('Survival Probability')
    plt.show()

    plt.plot(t_vals, J_vals)
    plt.xlabel('t')
    plt.ylabel('J')
    plt.title('Absorption Flux')
    plt.show()