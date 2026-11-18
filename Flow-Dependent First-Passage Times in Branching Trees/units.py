

def dt(N):
    result = 1/(2 * N**2)
    return result

# dt(N) gets the time step for a lattice step

def steps_to_time(steps, N):
    time = steps*dt(N)
    return time

# Convert step counts (int or array) to time in units of tau = L^2/D.

def eps(Pe, N):
    eps = Pe/(2 * N)
    if eps > 0.1:
        raise ValueError(f"eps = {eps:.3f} at Pe = {Pe}, N = {N}; raise N")
    return eps

# Coin bias for drift. From <xi> = a*eps = v*dt  ->  eps = Pe/(2N). Error in D is O(eps^2).

