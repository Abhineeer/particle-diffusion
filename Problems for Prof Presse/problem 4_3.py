import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
xi = 0.0   # prior guess for mean
psi0 = 0   # confidence in prior guess
alpha = 1.0   # Gamma shape
beta = 0.1  # Gamma rate

# True data parameters
mu_true = 0.0   # true mean
tau_main = 1.0   # main precision
mu_outlier = 10.0  # outlier center
tau_outlier = 1.5 # outlier precision (much larger)

# Data size
N_main = 10  # regular points
N_outlier = 10   # outlier points
N = N_main + N_outlier

np.random.seed(42)  # for reproducibility

# Main data points
# sigma = 1/sqrt(tau) — converting precision to std dev
w_main = np.random.normal(
    loc = mu_true, 
    scale = 1/np.sqrt(tau_main), 
    size = N_main
)

# Outlier points
w_outliers = np.random.normal(
    loc = mu_outlier, 
    scale = 1/np.sqrt(tau_outlier), 
    size = N_outlier
)

# Combine into one dataset
w = np.concatenate([w_main, w_outliers])

# Grid of mu values
mu_grid = np.linspace(-5, 15, 1000)

# --- Posterior 1 --- shared precision
def posterior_1(mu_grid, w, xi, psi0, alpha, beta):
    N = len(w)
    result = np.zeros(len(mu_grid))
    
    for i, mu in enumerate(mu_grid):
        # Compute B(mu)
        B = (beta 
             + 0.5 * psi0 * (mu - xi)**2 
             + 0.5 * np.sum((w - mu)**2))
        
        # Raise to negative power
        exponent = -(N/2 + alpha + 0.5)
        result[i] = B**exponent
    
    # Normalize
    result /= np.trapezoid(result, mu_grid)
    return result

# --- Posterior 2 --- individual precisions
def posterior_2(mu_grid, w, xi, psi0, alpha, beta):
    N = len(w)
    log_result = np.zeros(len(mu_grid))  # store log values
    
    for i, mu in enumerate(mu_grid):
        log_product = 0.0
        
        for wn in w:
            Bn = (beta 
                  + 0.5 * psi0 * (mu - xi)**2 
                  + 0.5 * (wn - mu)**2)
            log_product += -(alpha + 0.5) * np.log(Bn)
        
        log_result[i] = log_product  # store, don't exponentiate yet
    
    # NOW stabilize across the full array and exponentiate
    result = np.exp(log_result - np.max(log_result))
    
    # Normalize
    result /= np.trapezoid(result, mu_grid)
    return result



p1 = posterior_1(mu_grid, w, xi, psi0, alpha, beta)
p2 = posterior_2(mu_grid, w, xi, psi0, alpha, beta)

plt.figure(figsize=(10, 5))

plt.plot(mu_grid, p1, 
         label='Posterior 1 — shared precision', 
         color='steelblue', 
         linewidth=2)

plt.plot(mu_grid, p2, 
         label='Posterior 2 — individual precisions', 
         color='crimson', 
         linewidth=2)

# Mark where the data lives
plt.axvline(mu_true,     color='black',  linestyle='--', 
            label=f'True mean = {mu_true}')
plt.axvline(mu_outlier,  color='orange', linestyle='--', 
            label=f'Outlier center = {mu_outlier}')

# Mark actual data points along x axis
plt.scatter(w, np.zeros_like(w), 
            color='gray', zorder=5, 
            label='Data points')

plt.xlabel('μ')
plt.ylabel('p(μ | w₁:N)')
plt.title('Exercise 4.3 — Effect of Outliers on Posterior')
plt.legend()
plt.tight_layout()
plt.savefig('exercise_4_3.png', dpi=150, bbox_inches='tight')
plt.show()