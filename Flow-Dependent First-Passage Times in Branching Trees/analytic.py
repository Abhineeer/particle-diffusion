"""
Exact mean first-passage time from the root of a trunk + b branches,
all of length L, in units of tau = L^2/D. Derived in the write-up
(Eqs. 6 and 9); both are this one formula with beta = b A_1 / A_0:

    <T>/tau = (1 + beta)/p - [beta^2 (1 - e^{-p/beta})
                              + (1 - e^{-p}) e^{-p/beta}] / p^2

    p = 0 limit:   1 + 1/beta
    equal areas:   beta = b
    Murray's law:  beta = b^(1/3)
"""

import numpy as np
from scipy.optimize import brentq


def beta_of(b, model):
    """Total branch area over trunk area."""
    if model == "equal":
        return float(b)
    if model == "murray":
        return b ** (1 / 3)
    raise ValueError(f"unknown model {model}")


def mean_T(p, beta):
    """Exact <T>/tau at trunk Peclet number p."""
    if p == 0:
        return 1 + 1 / beta
    return ((1 + beta) / p
            - (beta**2 * (1 - np.exp(-p / beta))
               + (1 - np.exp(-p)) * np.exp(-p / beta)) / p**2)


def pe_critical(b, model):
    """Pe at which b branches and a single branch take equal mean time."""
    beta_b = beta_of(b, model)
    beta_1 = beta_of(1, model)              # = 1 for both models
    gap = lambda p: mean_T(p, beta_b) - mean_T(p, beta_1)
    return brentq(gap, 1e-3, 100)


if __name__ == "__main__":
    for model in ("equal", "murray"):
        for b in (2, 3, 4):
            print(f"{model:7s} b = {b}:  Pe_c = {pe_critical(b, model):.4f}")