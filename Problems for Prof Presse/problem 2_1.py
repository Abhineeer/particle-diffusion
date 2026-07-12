import numpy as np
import matplotlib.pyplot as plt

lambda_b = 5.0       # transcription (birth) rate
lambda_d = 0.5       # degradation (death) rate
# birth rate > death rate as required by the problem

t_max    = 50.0      # total simulation time
c_init   = 0         # initial RNA count

# Analytical steady state from mass action (Example 2.13):
# d<c>/dt = lambda_b - lambda_d * <c> = 0  =>  <c>_ss = lambda_b / lambda_d
c_ss = lambda_b / lambda_d

def gillespie(lambda_b, lambda_d, t_max, c_init):
    """
    Simulate RNA birth-death process using the Gillespie algorithm.

    Returns:
        times : list of time points (including jump times)
        counts: list of RNA counts at each time point
    """
    t = 0.0
    c = c_init

    times  = [t]
    counts = [c]

    while t < t_max:

        # --- Step 1: compute propensities ---
        mu1 = lambda_b          # birth:  propensity = lambda_b (constant)
        mu2 = c * lambda_d      # death:  propensity = c * lambda_d
        mu_star = mu1 + mu2     # total propensity

        # if no RNA and birth is somehow zero, system is stuck
        if mu_star == 0:
            break

        # --- Step 2: sample time to next event ---
        # H | c ~ Exponential(mu_star)
        # drawn via inverse CDF: h = -ln(U) / mu_star
        u1 = np.random.uniform()
        h  = -np.log(u1) / mu_star

        # --- Step 3: sample which reaction fires ---
        # G | c ~ Categorical(mu1/mu_star, mu2/mu_star)
        u2 = np.random.uniform()
        if u2 < mu1 / mu_star:
            c += 1    # birth fires
        else:
            c -= 1    # death fires

        # --- Step 4: update time and record ---
        t += h
        times.append(t)
        counts.append(c)

    return np.array(times), np.array(counts)

np.random.seed(42)
times, counts = gillespie(lambda_b, lambda_d, t_max, c_init)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Exercise 2.1: RNA Birth-Death Process (Gillespie Simulation)",
             fontsize=13, fontweight='bold')

# --- Top panel: single trajectory ---
ax1 = axes[0]
ax1.step(times, counts, where='post', color='steelblue',
         linewidth=0.8, label='Gillespie trajectory')
ax1.axhline(c_ss, color='red', linestyle='--', linewidth=1.5,
            label=f'Mass action steady state: $\\lambda_b/\\lambda_d$ = {c_ss:.1f}')
ax1.set_xlabel('Time')
ax1.set_ylabel('RNA count')
ax1.set_title('Single stochastic trajectory')
ax1.legend()
ax1.grid(True, alpha=0.3)

dt      = np.diff(times)
running_mean = np.cumsum(counts[:-1] * dt) / np.cumsum(dt)

ax2 = axes[1]
ax2.plot(times[1:], running_mean, color='darkorange',
         linewidth=1.5, label='Running time-average $\\langle c(t) \\rangle$')
ax2.axhline(c_ss, color='red', linestyle='--', linewidth=1.5,
            label=f'Mass action steady state = {c_ss:.1f}')
ax2.set_xlabel('Time')
ax2.set_ylabel('Average RNA count')
ax2.set_title('Running time-average converging to steady state')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gillespie_rna.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 50)
print("Part 3: Mass Action Steady State")
print("=" * 50)
print(f"ODE from Example 2.13:")
print(f"  d<c>/dt = lambda_b - lambda_d * <c>")
print(f"  At steady state: 0 = lambda_b - lambda_d * <c>_ss")
print(f"  => <c>_ss = lambda_b / lambda_d")
print(f"            = {lambda_b} / {lambda_d}")
print(f"            = {c_ss:.2f}")
print()
print(f"Simulation time-average at end: {running_mean[-1]:.2f}")
print(f"Difference: {abs(running_mean[-1] - c_ss):.2f}")