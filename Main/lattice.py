"""
lattice.py

Geometry and units. Everything else imports from here so that the
conversion between lattice steps and physical time is written down in
exactly one place.

CONVENTION
----------
A segment of physical length L is a chain of sites n = 0, 1, ..., N.
The sites are spaced

    a = L / N

apart, and site n sits at x = n*a.

So L is the physical length and N is the RESOLUTION. Refining N does not
make the segment longer, it makes the lattice finer. The default is
L = 1, D = 1, so the only physical knob left is the flow speed v, and
then the Peclet number is simply

    Pe = L*v/D = v

L is an argument rather than a hard constant so the two-length scaling
comparison can still be run by passing L explicitly.

TIME
----
Each step moves the walker +a or -a, so after k steps the variance is
k*a^2. The continuum answer is 2*D*t, so with t = k*dt

    dt = a^2 / (2*D) = L^2 / (2*D*N^2)

Note this DEPENDS ON N: a finer lattice means more, shorter steps. Any
conversion from steps to time therefore has to be told N.
"""

D = 1.0      # diffusion coefficient, fixed for the project
L_DEFAULT = 1.0


def spacing(N, L=L_DEFAULT):
    """Lattice spacing a = L/N."""
    return L / N


def position(n, N, L=L_DEFAULT):
    """Physical position of site n. Works on numpy arrays."""
    return n * spacing(N, L)


def step_time(N, L=L_DEFAULT):
    """Physical duration of one lattice step, a^2/(2D)."""
    return spacing(N, L)**2 / (2 * D)


def to_time(steps, N, L=L_DEFAULT):
    """Convert a step count, or an array of them, to physical time."""
    return steps * step_time(N, L)


def mfpt_segment(N=None, L=L_DEFAULT):
    """
    Continuum mean first-passage time for one segment, released at the
    reflecting end and absorbed at the far end.

    -D u'' = 1 with u'(0) = 0, u(L) = 0 gives u(x) = (L^2 - x^2)/(2D), so

        <t> = L^2 / (2D)

    independent of N: N is resolution, not length. N is accepted only so
    callers can pass it harmlessly.
    """
    return L**2 / (2 * D)


def mfpt_segment_lattice(N, L=L_DEFAULT):
    """
    The EXACT lattice answer, which is what a simulation at finite N
    actually produces: N*(N+1) steps, so

        <t> = (L^2 / 2D) * (1 + 1/N)

    The 1/N is the lattice being coarse. It is the right thing to compare
    a simulation against; mfpt_segment() is its N -> infinity limit.
    """
    return to_time(N * (N + 1), N, L)


if __name__ == "__main__":
    print(f"D = {D},  default L = {L_DEFAULT}\n")

    print("N is resolution, not length. The continuum answer never moves;")
    print("the lattice answer converges to it as N grows.\n")
    print(f"  {'N':>6} {'a':>10} {'dt':>12} {'lattice <t>':>13} "
          f"{'continuum':>11} {'ratio':>8}")
    for N in (25, 50, 100, 200, 400):
        lat = mfpt_segment_lattice(N)
        con = mfpt_segment(N)
        print(f"  {N:>6} {spacing(N):>10.5f} {step_time(N):>12.3e} "
              f"{lat:>13.6f} {con:>11.6f} {lat/con:>8.5f}")

    print("\n  ratio -> 1 like 1 + 1/N\n")

    print("passing L explicitly still works, for the scaling comparison:")
    for Lx in (1.0, 2.0):
        print(f"  L = {Lx}:  continuum <t> = {mfpt_segment(L=Lx):.4f}")
    print(f"  doubling L quadruples the time: ratio = "
          f"{mfpt_segment(L=2.0)/mfpt_segment(L=1.0)}")