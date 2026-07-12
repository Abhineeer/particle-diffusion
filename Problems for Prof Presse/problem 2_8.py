import numpy as np
import matplotlib.pyplot as plt

# ── Global parameters ────────────────────────────────────
dt = 0.001
T = 10
N = int(T/dt)
t = np.linspace(0, T, N)
q0 = 0.0

# ── Panel 1: Free Diffusion ──────────────────────────────
sigma_free = 0.3

q1 = np.zeros(N)
q1[0] = q0

for i in range(1, N):
    q1[i] = q1[i-1] + sigma_free * np.sqrt(dt) * np.random.randn()

# ── Panel 2: OU Process ──────────────────────────────────
gamma_ou = 1.0
sigma_ou = 0.35

q2 = np.zeros(N)
q2[0] = q0

for i in range(1, N):
    q2[i] = q2[i-1] * (1 - gamma_ou * dt) + sigma_ou * np.sqrt(dt) * np.random.randn()

# ── Panel 3: Hop Diffusion ───────────────────────────────
lam = 2.5
mus = [0.5, -0.5]
gamma_hop = 5.0
sigma_hop = 0.1

q3 = np.zeros(N)
q3[0] = q0

current_state = 0
mu = mus[current_state]
holding_time = np.random.exponential(1.0 / lam)
time_in_state = 0.0

for i in range(1, N):
    time_in_state += dt
    if time_in_state >= holding_time:
        current_state = 1 - current_state
        mu = mus[current_state]
        holding_time = np.random.exponential(1.0 / lam)
        time_in_state = 0.0

    q3[i] = q3[i-1] - gamma_hop * (q3[i-1] - mu) * dt + sigma_hop * np.sqrt(dt) * np.random.randn()

# ── Combined Plot ────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Panel 1
axes[0].plot(t, q1, color='black', linewidth=0.5)
axes[0].axhline(0.5,  linestyle='dotted', color='black', linewidth=0.8)
axes[0].axhline(-0.5, linestyle='dotted', color='black', linewidth=0.8)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_ylabel(r'$q_1(t)$')

# Panel 2
axes[1].plot(t, q2, color='black', linewidth=0.5)
axes[1].axhline(0.25,  linestyle='dotted', color='black', linewidth=0.8)
axes[1].axhline(-0.25, linestyle='dotted', color='black', linewidth=0.8)
axes[1].set_ylim(-1.5, 1.5)
axes[1].set_ylabel(r'$q_2(t)$')

# Panel 3
axes[2].plot(t, q3, color='black', linewidth=0.5)
axes[2].axhline(0.5,  linestyle='dashed', color='black', linewidth=0.8)
axes[2].axhline(-0.5, linestyle='dashed', color='black', linewidth=0.8)
axes[2].axhline(0.25,  linestyle='dotted', color='black', linewidth=0.8)
axes[2].axhline(-0.25, linestyle='dotted', color='black', linewidth=0.8)
axes[2].set_ylim(-1.5, 1.5)
axes[2].set_ylabel(r'$q_3(t)$')
axes[2].set_xlabel('t')

plt.tight_layout()
plt.show()