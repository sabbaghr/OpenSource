#!/usr/bin/env python3
"""Ray kinematics for the flat-reflector lag experiment.

Two things are computed here, and they are DIFFERENT things:

  1. DATA times.  When does each arrival reach R?  Straight rays in the
     homogeneous layer above a flat reflector, with the PS/SP conversion
     point found from Snell's law.  Used to (a) set the record length,
     (b) predict where to mute, and (c) be compared against the measured
     shot gather -- the script that does the comparing is the one allowed
     to say whether the model is right.

  2. IMAGE slopes.  Where in (x, z, tau) does a residual at data time T_D
     image, and with what slope?  The extended imaging condition with a
     SYMMETRIC lag puts the forward field at -tau and the adjoint at +tau,
     so a residual event images where

         tau(x) = 0.5 [ T_D - t_a(S, x) - t_b(x, R) ]                   (1)

     with mode a on the source leg and mode b on the receiver leg.  Its
     slope is the gradient of that,

         p(x) = -0.5 [ n_a(x)/v_a + n_b(x)/v_b ]                        (2)

     where n_a is the unit vector pointing from S towards x and n_b the
     unit vector pointing from R towards x.  Both are exact for straight
     rays, i.e. for any point in the homogeneous half-space above the
     reflector, which is where every pivot in this experiment sits.

     Equation (2) is the whole reason the experiment can be judged.  It
     says the slope is set by the two ray DIRECTIONS at the image point
     and by nothing else -- not by the reflector, not by the anomaly, not
     by the residual's amplitude.  Two limiting cases:

       transmission   n_a = -n_b, same mode:  the two terms cancel and
                      p = 0.  A kernel that lies ALONG a ray therefore
                      has |p| = 0 there.  This is the tomographic case.
       backscatter    n_a = +n_b, same mode:  the terms add and
                      |p| = 1/v.  At Vp = 2000 that is 0.5 ms/m, at
                      Vs = 1200 it is 0.833.  This is the specular case.

     So |p| for a same-mode pair runs from 0 to 1/v as the opening angle
     closes, |p| = cos(theta)/v with 2theta the opening angle -- the same
     identity the energy imaging condition uses.  A slope filter with a
     cutoff at 0.20-0.30 ms/m therefore cuts Vp at 2theta = 143-113 deg
     and Vs at 2theta = 166-136 deg: the Vs cut sits much closer to pure
     transmission, which is a prediction this experiment can check and
     not a free parameter.

Conventions.  z is positive DOWN.  Slopes are returned in ms/m, times in
s, lengths in m.  Nothing here reads a file or draws anything; the script
that imports it owns the geometry.
"""
import numpy as np

MS_PER_S = 1.0e3


# ---------------------------------------------------------------------------
# data times
# ---------------------------------------------------------------------------
def direct_time(xs, zs, xr, zr, v):
    """Straight-ray traveltime S -> R at velocity v."""
    return float(np.hypot(xr - xs, zr - zs) / v)


def _refl_time(xc, xs, zs, xr, zr, zrefl, va, vb):
    """Two-leg time S -> (xc, zrefl) -> R, down at va and up at vb."""
    return (np.hypot(xc - xs, zrefl - zs) / va
            + np.hypot(xr - xc, zrefl - zr) / vb)


def reflect_time(xs, zs, xr, zr, zrefl, va, vb, n=200001):
    """Fermat time and bounce point for a flat reflector at depth zrefl.

    Down-leg at va, up-leg at vb.  For va == vb this is the mirror-image
    answer; for va != vb it is Snell's law, and rather than iterate on the
    Snell condition the stationary point of the two-leg time is found by a
    dense scan followed by one parabolic refinement.  The scan spans the
    whole reflector width that can matter (between the two mirror points,
    widened by the offset), so no bracket has to be guessed, and the
    refinement makes the answer independent of the scan step to second
    order.

    Returns (t, xc).
    """
    lo = min(xs, xr) - abs(xr - xs) - 1.0
    hi = max(xs, xr) + abs(xr - xs) + 1.0
    xc = np.linspace(lo, hi, int(n))
    t = _refl_time(xc, xs, zs, xr, zr, zrefl, va, vb)
    i = int(np.argmin(t))
    if 0 < i < len(xc) - 1:                       # parabolic vertex
        y0, y1, y2 = t[i - 1], t[i], t[i + 1]
        d = y0 - 2.0 * y1 + y2
        shift = 0.0 if d == 0.0 else 0.5 * (y0 - y2) / d
        xstar = xc[i] + shift * (xc[1] - xc[0])
    else:
        xstar = xc[i]
    return float(_refl_time(xstar, xs, zs, xr, zr, zrefl, va, vb)), float(xstar)


def arrival_table(xs, zs, xr, zr, zrefl, vp, vs, t0=0.0):
    """Every first-order arrival for one flat reflector, as a list of dicts.

    Keys: name, modes (two-character string or 'P'/'S' for the directs),
    t (traveltime), tpick (t + t0, i.e. where the wavelet PEAK lands, since
    makewave puts its peak at t0), xc (bounce x, None for the directs).
    """
    out = [dict(name='direct P', modes='P',
                t=direct_time(xs, zs, xr, zr, vp), xc=None),
           dict(name='direct S', modes='S',
                t=direct_time(xs, zs, xr, zr, vs), xc=None)]
    for nm, va, vb in (('PP', vp, vp), ('PS', vp, vs),
                       ('SP', vs, vp), ('SS', vs, vs)):
        t, xc = reflect_time(xs, zs, xr, zr, zrefl, va, vb)
        out.append(dict(name=nm, modes=nm, t=t, xc=xc))
    for d in out:
        d['tpick'] = d['t'] + t0
    return out


# ---------------------------------------------------------------------------
# image slopes
# ---------------------------------------------------------------------------
def _unit(dx, dz):
    r = np.hypot(dx, dz)
    r = np.where(r == 0.0, 1.0, r)
    return dx / r, dz / r


def slope(x, z, xs, zs, xr, zr, va, vb):
    """Equation (2): p = -0.5 (n_a/va + n_b/vb), in ms/m.

    n_a points from S towards (x, z); n_b points from R towards (x, z).
    Scalars or arrays.  Straight rays, so this is exact only above the
    reflector -- which is where it is used.
    """
    ax, az = _unit(np.asarray(x, float) - xs, np.asarray(z, float) - zs)
    bx, bz = _unit(np.asarray(x, float) - xr, np.asarray(z, float) - zr)
    px = -0.5 * (ax / va + bx / vb) * MS_PER_S
    pz = -0.5 * (az / va + bz / vb) * MS_PER_S
    return px, pz


def opening_angle(x, z, xs, zs, xr, zr):
    """The full opening angle 2theta between the two ray directions, deg.

    theta = 0 is backscatter, theta = 90 is transmission -- the same
    convention as energy_ic=.  Reported alongside every slope so that a
    measured slope can be read as an angle without a second calculation.
    """
    ax, az = _unit(np.asarray(x, float) - xs, np.asarray(z, float) - zs)
    bx, bz = _unit(np.asarray(x, float) - xr, np.asarray(z, float) - zr)
    # n_a is S->x; the ray ARRIVING at x from R is -n_b, so the angle
    # between the two propagation directions is between n_a and -n_b.
    c = np.clip(ax * (-bx) + az * (-bz), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def isochron(T_D, xs, zs, xr, zr, va, vb, xaxis, zaxis):
    """tau(x, z) from equation (1) on a grid, in SECONDS.

    The tau = 0 contour of this is the isochron the residual images on;
    its zero crossing is what the slant-stack pivots are placed on.
    """
    X = np.asarray(xaxis, float)[:, None]
    Z = np.asarray(zaxis, float)[None, :]
    ta = np.hypot(X - xs, Z - zs) / va
    tb = np.hypot(X - xr, Z - zr) / vb
    return 0.5 * (T_D - ta - tb)


def isochron_depth(T_D, xs, zs, xr, zr, va, vb, xq, zlo, zhi, n=200001):
    """Where the tau = 0 isochron crosses the column x = xq.

    Returns z, or None if it does not cross inside [zlo, zhi].  tau is
    monotone in z along a column well below both S and R, so a scan plus a
    linear crossing is enough and is immune to the bracketing failures of
    a root finder started from a guess.
    """
    z = np.linspace(zlo, zhi, int(n))
    tau = 0.5 * (T_D - np.hypot(xq - xs, z - zs) / va
                 - np.hypot(xq - xr, z - zr) / vb)
    s = np.sign(tau)
    k = np.nonzero(s[:-1] * s[1:] < 0.0)[0]
    if not len(k):
        return None
    i = int(k[0])
    f = tau[i] / (tau[i] - tau[i + 1])
    return float(z[i] + f * (z[i + 1] - z[i]))
