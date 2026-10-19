"""
lattice.py

Geometry and units. Everything else imports from here so that the
conversion between lattice steps and physical time is written down
in exactly one place.

Convention:
    a segment of length L is a chain of sites n = 0, 1, ..., N
    the sites are spaced a apart, so L = N * a
    site n sits at position x = n * a

The spacing a is the same everywhere in the project. A longer segment
means more sites, not bigger sites. This matters: if we changed L by
stretching the sites instead, then changing L would not actually
change the geometry and the scaling test would prove nothing.

Time:
    each step moves the walker +a or -a, so after k steps the variance
    is k * a^2. The continuum answer is 2*D*t. Setting them equal with
    t = k*dt gives

        dt = a^2 / (2*D)

    which is the only time conversion in the project.
"""

# --- fixed for the whole project ---
a = 1.0      # lattice spacing
D = 1.0      # diffusion coefficient

dt = a**2 / (2*D)     # physical time of one step


def length(N):
    """Physical length of a segment with N spacings."""
    return N * a


def position(n):
    """Physical position of site n. Works on numpy arrays too."""
    return n * a


def to_time(steps):
    """Convert a step count, or an array of them, to physical time."""
    return steps * dt


def mfpt_segment(N):
    """
    Exact mean first-passage time for one segment:
    released at the reflecting end, absorbed at the far end.

    Solving -D u'' = 1 with u'(0) = 0 and u(L) = 0 gives
    u(x) = (L^2 - x^2) / (2D), so starting from x = 0:

        <t> = L^2 / (2D)
    """
    L = length(N)
    return L**2 / (2*D)


if __name__ == "__main__":
    print("a =", a, "  D =", D, "  dt =", dt)
    print()

    for N in (100, 200):
        print(f"N = {N:4d}   L = {length(N):6.1f}   "
              f"exact MFPT = {mfpt_segment(N):10.2f}")

    print()
    print("doubling N doubles L, so the MFPT should go up by 4:")
    print("   ratio =", mfpt_segment(200) / mfpt_segment(100))
