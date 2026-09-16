"""
run_ensemble.py

Run many walks at one value of b, count what we care about, compare to theory.
Uses walker_core_logic.py (the plain, readable engine).

Note: h_V = 1/(1+b) is the probability of reaching the trunk tip *starting
from the vertex*. So this file releases at the vertex. Releasing anywhere
else measures h at a different point and will NOT equal 1/(1+b).
"""

from walker_core_logic import run_one_walk


def run_ensemble(b, N, n_walkers, start="vertex"):
    reached_trunk_tip = 0        # walks that touched the trunk tip at least once
    tip_counts = [0] * (b + 1)   # index 1..b used; how many died on each branch
    total_steps = 0

    for _ in range(n_walkers):
        r = run_one_walk(b, N, start=start)
        if r["trunk_touches"] > 0:
            reached_trunk_tip += 1
        tip_counts[r["branch"]] += 1
        total_steps += r["steps"]

    return reached_trunk_tip, tip_counts, total_steps


if __name__ == "__main__":
    b = 6
    N = 100
    n_walkers = 20_000

    reached, tip_counts, total_steps = run_ensemble(b, N, n_walkers, start="trunk_tip")

    frac_reached = reached / n_walkers
    mean_steps = total_steps / n_walkers

    print(f"b = {b}, N = {N}, walkers = {n_walkers}, start = trunk tip\n")

    # print("trunk-return probability h_V (fraction that reach the trunk tip):")
    # print(f"    measured = {frac_reached:.4f}")
    # print(f"    1/(1+b)  = {1/(1+b):.4f}\n")

    print("branch tip distribution (should be uniform, 1/b each):")
    for i in range(1, b + 1):
        print(f"    branch {i}: {tip_counts[i]/n_walkers:.4f}   (theory {1/b:.4f})")

    print(f"\nmean absorption time: {mean_steps:.0f} steps")