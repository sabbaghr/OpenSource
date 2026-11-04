#!/usr/bin/env python3
"""Local 2D slant stack of a lag-extended 2D gradient or image.

This module does ONE thing: it evaluates

    M(x0, p) = sum_xi  w(xi) * g(x0 + xi, tau0 + px*xi_x + pz*xi_z) * dA

for a set of pivots x0 and a 2D grid of slopes p = (px, pz).  `xi` is the
offset from the pivot, `w` a smooth aperture window, `dA = dx*dz` the grid
cell area.  An event focuses in the (px, pz) map where the stacking plane is
tangent to it, i.e. at p = grad(tau).

It does NOT filter, mute, mask, classify events, invert the transform, or
touch any gradient code.  Nothing here labels a region of the (px, pz) map as
keep or reject -- that is an interpretation, and it is deliberately left to
whoever reads the figure.

---------------------------------------------------------------------------
CONVENTIONS, ESTABLISHED FROM fdelfwi/lag_gradient.c  (not assumed)
---------------------------------------------------------------------------
Producer            lag_gradient.c, called from test_gradient_visual.
Files               <base>_lagvp.su, <base>_lagvs.su, <base>_lagaxis.txt.
                    Nothing else is written; see "no g_f" below.
Cube order          trace = ilag*nx + ix, nz samples per trace, so the SU
                    file reshapes to (nlag, nx, nz): lag slowest, z fastest.
                    (lagEmit, lag_gradient.c:599 and :619)
Axes                dx, dz, x0, z0, nx, nz from <base>_lagaxis.txt.  x and z
                    in metres, z POSITIVE DOWNWARD.  Lags in seconds.
Lag axis            symmetric and odd, tau_i = (i - (nlag-1)/2)*dlag, so
                    tau = 0 is sampled exactly (lag_gradient.c:671, :622).

LAG CONVENTION -- symmetric, relative shift 2*tau:

    g_lam(x,tau) = dt sum_t div(v_f)(x, t-tau) * (psi_txx+psi_tzz)(x, t+tau)

    i.e. the forward field is shifted by -tau and the adjoint by +tau, so
    g_sym(tau) = g_oneside(2*tau).  EVERY statement about the lag axis
    therefore carries a factor 2.  (lag_gradient.c header, :490-492)

PHASE -- the sign is NEGATIVE and the frequency factor is 2:

    g(x,tau) = (dw/pi) sum_k Re{ conj(A^)(x,w_k) B^(x,w_k) exp(-2i w_k tau) }
    A^(w)    = dt sum_t A(t) exp(+i w t)
    weight   = dw/pi = 2*df

    implemented as ct=cos(2 w_k tau), st=sin(2 w_k tau),
    acc += Xr*ct + Xi*st with X = conj(A^) B^.
    (lagSlice, lag_gradient.c:482, :497-498, :521)

    NOTE this is exp(-i 2pi (2f) tau), not exp(+i 2pi (2f) tau).

NO PER-FREQUENCY g_f ON DISK.  The six DFT accumulators (lg_exx, lg_ezz,
lg_shr, lg_pxx, lg_pzz, lg_pxz; frequency-fastest, re/im interleaved at
((size_t)ig*nfreq+k)*2) are internal C state and lagFreeAccum frees them
(lag_gradient.c:320-325, :814-817).  slant2d_freq() therefore takes g_f as an
IN-MEMORY array.  It is fully exercised by the method-agreement test, but it
cannot be run on the on-disk example results without adding a writer to
lagEmit -- which would be a change to the gradient code, so it was not made.

<base>_lagaxis.txt records nfreq and df but NOT fmin, so the band edge fmax
cannot be recovered from it.  Pass fmax= if you want the sampling checks.

---------------------------------------------------------------------------
PUBLIC INTERFACE
---------------------------------------------------------------------------
Slopes are in ms/m at the public interface (px, pz, dp, pmax) and are
converted to s/m internally.  All units are repeated in the result metadata.

    Grid(dx, dz, x0, z0, taus)       axis description
    read_lag_result(base, par)       -> (g, Grid) from the repo's SU files
    slant2d_lag(g, grid, pivots)     -> SlantResult        [lag domain]
    slant2d_freq(gf, freqs, grid, pivots) -> SlantResult   [frequency domain]
    save(res, fn) / load(fn)         .npz round trip
    plot_pivot(res, ip, ...)         one figure per pivot
    slope_to_theta(p, v)             OPTIONAL display only -- see its docstring

Run as a script for the example of section 9:

    python3 lag_slant2d.py --base=dense --par=vp,vs \
        --pivots=2022,1140:2502,1500:2982,1860 \
        --fmax=36 --vmin=1200 --out=slant2d
"""
import os
import sys
import json
import warnings

import numpy as np

# Relative, not absolute -- see the same note in lag_refl_check.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, 'tests', 'kh_misfit'))
import su_io

LAG_CONVENTION = ('symmetric: forward shifted -tau, adjoint +tau, so the '
                  'relative shift is 2*tau; g_sym(tau)=g_oneside(2*tau); '
                  'reconstruction phase exp(-2i w tau), weight 2*df '
                  '(fdelfwi/lag_gradient.c:482,490-492,497-498,521)')

#: factor relating the lag axis to the relative time shift.  2 for the
#: symmetric convention above.  Every sampling criterion uses it, so a change
#: of convention is a change of this one number.
LAG_FACTOR = 2.0

MS_PER_S = 1.0e3            # slopes: public ms/m  <->  internal s/m

BGFILE = {'vp': 'bg_cp.su', 'vs': 'bg_cs.su'}


# ===========================================================================
# axis description and the repo's file format
# ===========================================================================
class Grid(object):
    """Axis description of a lag-extended 2D array.

    g has shape (ntau, nx, nz) for the lag domain, or (nfreq, nx, nz) for the
    frequency domain, matching the repo's cube order (lag/frequency slowest,
    z fastest).  x and z are in metres with z positive downward; taus in s.
    """

    def __init__(self, dx, dz, x0=0.0, z0=0.0, taus=None):
        self.dx = float(dx)
        self.dz = float(dz)
        self.x0 = float(x0)
        self.z0 = float(z0)
        self.taus = None if taus is None else np.asarray(taus, dtype=float)

    @property
    def dtau(self):
        """Mean lag spacing, with a tolerance set by the file format.

        <base>_lagaxis.txt prints float32 lags with %.9g, so for lags of
        order 0.1 s at a 0.002 s spacing the DIFFERENCES carry a relative
        error of about 1e-7 * 0.1 / 0.002 = 5e-6.  A tighter tolerance
        rejects a perfectly uniform axis; the mean is used rather than the
        first difference for the same reason.
        """
        if self.taus is None:
            return None
        d = np.diff(self.taus)
        dm = float(d.mean())
        if np.ptp(d) > 1.0e-4 * abs(dm):
            raise ValueError('the lag axis is not uniform: spacings span '
                             '%g to %g s about a mean of %g s'
                             % (d.min(), d.max(), dm))
        return dm

    @property
    def dA(self):
        return self.dx * self.dz

    def ix_of(self, x):
        return int(round((float(x) - self.x0) / self.dx))

    def iz_of(self, z):
        return int(round((float(z) - self.z0) / self.dz))

    def x_axis(self, nx):
        return self.x0 + self.dx * np.arange(nx)

    def z_axis(self, nz):
        return self.z0 + self.dz * np.arange(nz)

    def __repr__(self):
        return ('Grid(dx=%g, dz=%g, x0=%g, z0=%g, ntau=%s)'
                % (self.dx, self.dz, self.x0, self.z0,
                   'None' if self.taus is None else len(self.taus)))


def read_axis(base):
    """Parse <base>_lagaxis.txt: key/value pairs then a bare list of lags."""
    m, lags = {}, []
    for line in open('%s_lagaxis.txt' % base):
        if line.startswith('#'):
            continue
        p = line.split()
        if len(p) == 2:
            m[p[0]] = float(p[1])
        elif len(p) == 1:
            lags.append(float(p[0]))
    m['lags'] = np.asarray(lags)
    return m


def read_lag_result(base, par='vp'):
    """Load <base>_lag<par>.su -> (g (ntau,nx,nz), Grid).

    The reshape follows lagEmit's documented order, trace = ilag*nx + ix.
    """
    m = read_axis(base)
    nlag, nx, nz = int(m['nlag']), int(m['nx']), int(m['nz'])
    d, _ = su_io.read('%s_lag%s.su' % (base, par))
    if d.shape != (nlag * nx, nz):
        raise ValueError('%s_lag%s.su: shape %s, expected %s'
                         % (base, par, d.shape, (nlag * nx, nz)))
    g = d.reshape(nlag, nx, nz).astype(np.float64)
    grid = Grid(m['dx'], m['dz'], m['x0'], m['z0'], m['lags'])
    return g, grid


def model_vmin(fn):
    """Smallest value of an SU model file, for the default pmax."""
    v, _ = su_io.read(fn)
    return float(v.min())


# ===========================================================================
# aperture windows and the tau-edge taper
# ===========================================================================
def _window_1d(xi, h, kind):
    """Smooth aperture weight on offsets xi [m] with nominal half-width h.

    The window keeps its NOMINAL shape even when the patch is clipped at a
    model edge: clipping then shows up as reduced fold, not as a silently
    re-shaped window.

    hann   0.5(1+cos(pi xi/h))   -- the default; smooth to zero at |xi| = h
    box    1 inside, 0 outside   -- hard edges, kept only for comparison
    tukey  box with a raised-cosine roll over the outer `roll` fraction

    Why the default is smooth.  The far-field tail of M along a direction n
    is governed by the smoothness at the CENTRE of the window's Radon
    projection  P_n(s) = integral of w over the line n.xi = s -- not by the
    1D edge shape, and not by a separable-sinc argument.  Measured on a
    planar event:

        box, n along an axis       |M| ~ 1e-11 (a square projects to a
                                   CONSTANT, so there is no tail at all)
        box, n along a diagonal    p^-2  (triangular projection, KINK at
                                   s = 0), and 2.2e7 times the axis level
        hann, any n                p^-3, diagonal/axis = 0.5

    So a box does not smear uniformly: it beams its leakage into four
    diagonal arms, which is the X seen in a (px, pz) map.  A Hann window is
    smooth in every direction and the arms go away.  The price is a main lobe
    about 1.6x wider, because the effective aperture is smaller.
    """
    a = np.abs(np.asarray(xi, dtype=float))
    if kind == 'box':
        return (a <= h).astype(float)
    if kind == 'hann':
        w = 0.5 * (1.0 + np.cos(np.pi * np.minimum(a / h, 1.0)))
        return np.where(a <= h, w, 0.0)
    if kind.startswith('tukey'):
        roll = 0.5
        if ':' in kind:
            roll = float(kind.split(':', 1)[1])
        roll = min(max(roll, 0.0), 1.0)
        j = np.minimum(a / h, 1.0)
        w = np.ones_like(j)
        if roll > 0.0:
            k = j > 1.0 - roll
            w[k] = 0.5 * (1.0 + np.cos(np.pi * (j[k] - (1.0 - roll)) / roll))
        return np.where(a <= h, w, 0.0)
    raise ValueError('unknown window %r (hann, box, tukey[:roll])' % kind)


def _tau_taper(tau, tlo, thi, te):
    """Smooth 1 -> 0 ramp of length te [s] at each end of [tlo, thi].

    Zero strictly outside.  A hard cutoff in tau creates false arms exactly
    like a hard cutoff in space, so the default is smooth.
    """
    d = np.minimum(tau - tlo, thi - tau)
    if te <= 0.0:
        return (d >= 0.0).astype(float)
    t = 0.5 * (1.0 - np.cos(np.pi * np.clip(d / te, 0.0, 1.0)))
    return np.where(d >= 0.0, t, 0.0)


# ===========================================================================
# interpolation along the lag axis
# ===========================================================================
def _taps(kind):
    """(offsets, coefficient function) for interpolation along tau."""
    if kind == 'linear':
        return np.array([0, 1]), lambda u: (1.0 - u, u)

    if kind == 'cubic':
        # Catmull-Rom cubic convolution (a = -1/2): local, 4 taps, exact at
        # the nodes, C1.  This is the default.
        def c(u):
            u2, u3 = u * u, u * u * u
            return (-0.5 * u3 + u2 - 0.5 * u,
                    1.5 * u3 - 2.5 * u2 + 1.0,
                    -1.5 * u3 + 2.0 * u2 + 0.5 * u,
                    0.5 * u3 - 0.5 * u2)
        return np.array([-1, 0, 1, 2]), c

    if kind == 'sinc':
        # Kaiser-windowed sinc, 8 taps -- band-limited, for when cubic's
        # amplitude roll-off near Nyquist matters.
        half, beta = 4, 6.0
        off = np.arange(-half + 1, half + 1)

        def c(u):
            u = np.asarray(u)
            d = u[..., None] - off                      # distance in samples
            s = np.sinc(d)
            kb = np.i0(beta * np.sqrt(np.maximum(
                0.0, 1.0 - (d / float(half)) ** 2))) / np.i0(beta)
            w = s * kb
            w = w / np.maximum(w.sum(axis=-1, keepdims=True), 1.0e-30)
            return tuple(w[..., m] for m in range(len(off)))
        return off, c

    raise ValueError('unknown interp %r (linear, cubic, sinc)' % kind)


def _gather(gflat, f, offs, cfun, ntau):
    """Interpolate gflat (ntau, npt) along tau at fractional indices f (npt,).

    Indices are clipped; the caller must multiply by a taper that is zero
    wherever the clip would bite, which _tau_taper does.
    """
    i0 = np.floor(f).astype(np.intp)
    u = f - i0
    cols = np.arange(gflat.shape[1], dtype=np.intp)
    out = np.zeros(gflat.shape[1])
    for m, cm in zip(offs, cfun(u)):
        out += cm * gflat[np.clip(i0 + int(m), 0, ntau - 1), cols]
    return out


# ===========================================================================
# the result container
# ===========================================================================
class SlantResult(object):
    """M (n_piv, npx, npz) with everything needed to read it back.

    Attributes
        M          the slant stack.  NOT normalised by fold unless the run
                   asked for it (params['normalize']).
        fold       (n_piv, npx, npz), sum of applied weights times dA [m^2].
                   M/fold is the weighted average; M is the weighted sum.
        px, pz     slope axes in ms/m.  pz is POSITIVE DOWNWARD, because z is.
        pivots     (n_piv, 2) physical (x, z) in m, as requested.
        pivots_xz  (n_piv, 2) the (ix, iz) actually used.
        params     every parameter value, including the effective ones.
        units      units of each field.
        lag_convention / phase    the conventions this was computed under.
        warnings   list of strings; sampling and edge warnings, kept with the
                   data so a figure cannot silently outlive them.
    """

    def __init__(self, M, fold, px, pz, pivots, pivots_ij, params, units,
                 domain, warns):
        self.M = M
        self.fold = fold
        self.px = px
        self.pz = pz
        self.pivots = pivots
        self.pivots_ij = pivots_ij
        self.params = params
        self.units = units
        self.domain = domain
        self.lag_convention = LAG_CONVENTION
        self.phase = 'exp(-2i w tau), weight 2*df'
        self.warnings = list(warns)

    @property
    def n_piv(self):
        return self.M.shape[0]

    def normalized(self):
        """M/fold, the weighted average.  Never applied in place."""
        return self.M / np.maximum(self.fold, 1.0e-30 * self.fold.max())

    def peak(self, ip):
        """((px, pz) in ms/m, value) at the maximum of |M| for one pivot."""
        a = np.abs(self.M[ip])
        j, k = np.unravel_index(np.argmax(a), a.shape)
        return (float(self.px[j]), float(self.pz[k])), float(self.M[ip, j, k])

    def __repr__(self):
        return ('SlantResult(%s, M%s, %d pivots, px %g..%g, pz %g..%g ms/m)'
                % (self.domain, self.M.shape, self.n_piv,
                   self.px[0], self.px[-1], self.pz[0], self.pz[-1]))


# ===========================================================================
# shared set-up: slope axes, patches, sampling checks
# ===========================================================================
def _slope_axes(px, pz, pmax, dp, vmin, warns):
    """Build the px, pz axes in ms/m.  Default is symmetric, pmax = 1.1/vmin.

    vmin is never hard-coded: it comes from the model or from the caller.
    """
    if px is None or pz is None:
        if pmax is None:
            if vmin is None:
                raise ValueError(
                    'need pmax= or vmin= (or vmodel=) to set the default '
                    'slope range pmax = 1.1/vmin; nothing is hard-coded here')
            pmax = 1.1 / float(vmin) * MS_PER_S          # ms/m
        n = int(np.floor(float(pmax) / float(dp)))
        ax = dp * np.arange(-n, n + 1)
        if px is None:
            px = ax
        if pz is None:
            pz = ax
    return (np.asarray(px, dtype=float), np.asarray(pz, dtype=float),
            pmax)


def _patch_index(grid, nx, nz, xp, zp, hx, hz, warns):
    """Offsets and index ranges of the patch around one pivot.

    Near a model edge only the available samples are used.  The window keeps
    its nominal half-width, so the missing samples lower the fold; nothing is
    padded with invented values, and a warning is raised.
    """
    i0, k0 = grid.ix_of(xp), grid.iz_of(zp)
    if not (0 <= i0 < nx and 0 <= k0 < nz):
        raise ValueError('pivot (%g, %g) m is outside the grid' % (xp, zp))

    kx = int(round(hx / grid.dx))
    kz = int(round(hz / grid.dz))
    ia, ib = max(0, i0 - kx), min(nx - 1, i0 + kx)
    ka, kb = max(0, k0 - kz), min(nz - 1, k0 + kz)

    if (ia, ib, ka, kb) != (i0 - kx, i0 + kx, k0 - kz, k0 + kz):
        warns.append('pivot (%g, %g) m is within one aperture of a model '
                     'edge: patch clipped to x[%g, %g] z[%g, %g] m, so its '
                     'fold is reduced; no samples were invented'
                     % (xp, zp, grid.x0 + ia * grid.dx, grid.x0 + ib * grid.dx,
                        grid.z0 + ka * grid.dz, grid.z0 + kb * grid.dz))

    xi_x = (np.arange(ia, ib + 1) - i0) * grid.dx
    xi_z = (np.arange(ka, kb + 1) - k0) * grid.dz
    return (i0, k0), (ia, ib, ka, kb), xi_x, xi_z


def _sampling_checks(grid, px, pz, hx, hz, fmax, warns, dtau=None):
    """Warn, never fail, when the discretisation cannot carry the slopes.

    All three criteria use LAG_FACTOR, so they follow the lag convention
    rather than hard-coding the 2 of the symmetric case.
    """
    eff = {}
    if fmax is None:
        warns.append('fmax not supplied: the sampling checks were SKIPPED.  '
                     '<base>_lagaxis.txt records nfreq and df but not fmin, '
                     'so the band edge cannot be recovered from it')
        return eff

    fe = LAG_FACTOR * float(fmax)          # effective frequency on the tau axis
    eff['f_effective_Hz'] = fe

    if dtau is not None:
        dtau_max = 1.0 / (2.0 * fe)
        eff['dtau_max_s'] = dtau_max
        if dtau > dtau_max:
            warns.append('dtau = %.5f s exceeds 1/(2*%g*fmax) = %.5f s: the '
                         'lag axis is aliased for the %g Hz edge of the band'
                         % (dtau, LAG_FACTOR, dtau_max, fmax))

    pmax_s = max(np.abs(px).max(), np.abs(pz).max()) / MS_PER_S
    eff['pmax_s_per_m'] = pmax_s
    if pmax_s > 0.0:
        dx_max = 1.0 / (2.0 * fe * pmax_s)
        eff['dx_max_m'] = dx_max
        for nm, d in (('dx', grid.dx), ('dz', grid.dz)):
            if d > dx_max:
                warns.append('%s = %g m exceeds 1/(2*%g*fmax*pmax) = %.2f m: '
                             'the steepest slope is spatially aliased'
                             % (nm, d, LAG_FACTOR, dx_max))

    for nm, h, ax in (('px', hx, px), ('pz', hz, pz)):
        D = 2.0 * h
        if D <= 0.0 or len(ax) < 2:
            continue
        dp_max = 1.0 / (fe * D)            # ~ 1/(LAG_FACTOR*fmax*D)
        eff['dp_max_%s_ms_per_m' % nm] = dp_max * MS_PER_S
        dp = abs(ax[1] - ax[0]) / MS_PER_S
        if dp > dp_max:
            warns.append('d%s = %.4f ms/m exceeds 1/(%g*fmax*D) = %.4f ms/m '
                         'for an aperture D = %g m: the slope axis undersamples '
                         'the focus' % (nm, dp * MS_PER_S, LAG_FACTOR,
                                        dp_max * MS_PER_S, D))
    return eff


def _pivot_list(pivots):
    p = np.atleast_2d(np.asarray(pivots, dtype=float))
    if p.shape[1] != 2:
        raise ValueError('pivots must be (x, z) pairs in metres, got shape %s'
                         % (p.shape,))
    return p


# ===========================================================================
# 3.  lag-domain implementation
# ===========================================================================
def slant2d_lag(g, grid, pivots,
                hx=300.0, hz=300.0, window='hann',
                px=None, pz=None, pmax=None, dp=0.02,
                tau0=0.0, tau_edge=None, interp='cubic',
                vmin=None, vmodel=None, fmax=None,
                normalize=False, verbose=1):
    """Local 2D slant stack of g(tau, x, z), one (px, pz) map per pivot.

    g          (ntau, nx, nz), the repo's cube order
    grid       Grid with taus set
    pivots     (x, z) pairs in metres

    hx, hz     patch HALF-widths in metres, default 300 m each
    window     'hann' (default), 'box', 'tukey[:roll]'
    px, pz     slope axes in ms/m; default symmetric with pmax = 1.1/vmin
    pmax       ms/m; overrides the 1.1/vmin default
    dp         slope step in ms/m, default 0.02
    tau0       pivot lag in seconds, default 0
    tau_edge   length of the smooth taper at each end of the tau axis, in
               seconds.  Default 5*dtau, raised to 2*dtau at least when the
               interpolator has taps outside [i0, i0+1].
    interp     'cubic' (default), 'sinc' (band-limited) or 'linear'
    vmin       smallest velocity, for the default pmax.  Not hard-coded: pass
               it, or pass vmodel=, or pass pmax=.
    vmodel     an array or an SU filename; vmin is taken as its minimum
    fmax       band edge in Hz, for the sampling checks.  Omit and they are
               skipped with a warning.
    normalize  divide by fold.  OFF by default, as the fold map is returned
               and dividing hides where the aperture was truncated.

    Returns a SlantResult.  M is the weighted SUM (units of g times m^2).
    """
    g = np.asarray(g, dtype=np.float64)
    if g.ndim != 3:
        raise ValueError('g must be (ntau, nx, nz), got %s' % (g.shape,))
    if grid.taus is None:
        raise ValueError('grid.taus is required for the lag-domain method')
    ntau, nx, nz = g.shape
    if len(grid.taus) != ntau:
        raise ValueError('grid.taus has %d entries but g has %d lags'
                         % (len(grid.taus), ntau))

    dtau = grid.dtau
    tlo, thi = float(grid.taus[0]), float(grid.taus[-1])
    warns = []

    if vmin is None and vmodel is not None:
        vmin = (model_vmin(vmodel) if isinstance(vmodel, str)
                else float(np.asarray(vmodel).min()))
    px, pz, pmax = _slope_axes(px, pz, pmax, dp, vmin, warns)

    offs, cfun = _taps(interp)
    if tau_edge is None:
        tau_edge = 5.0 * dtau
    need = (max(0, -int(offs.min())) + max(0, int(offs.max()) - 1)) * dtau
    if tau_edge < need:
        tau_edge = need
        warns.append('tau_edge raised to %.5f s so the %r interpolator never '
                     'reads a clipped sample with non-zero weight'
                     % (tau_edge, interp))

    piv = _pivot_list(pivots)
    eff = _sampling_checks(grid, px, pz, hx, hz, fmax, warns, dtau=dtau)

    npx, npz = len(px), len(pz)
    M = np.zeros((len(piv), npx, npz))
    FOLD = np.zeros((len(piv), npx, npz))
    pxs, pzs = px / MS_PER_S, pz / MS_PER_S          # s/m

    for ip, (xp, zp) in enumerate(piv):
        (i0, k0), (ia, ib, ka, kb), xi_x, xi_z = _patch_index(
            grid, nx, nz, xp, zp, hx, hz, warns)

        wx = _window_1d(xi_x, hx, window)
        wz = _window_1d(xi_z, hz, window)
        W = (wx[:, None] * wz[None, :]).ravel()      # (npt,)
        keep = W > 0.0

        patch = g[:, ia:ib + 1, ka:kb + 1].reshape(ntau, -1)[:, keep]
        Wk = W[keep] * grid.dA
        ax = xi_x[:, None] * np.ones_like(xi_z)[None, :]
        az = np.ones_like(xi_x)[:, None] * xi_z[None, :]
        ax = ax.ravel()[keep]
        az = az.ravel()[keep]

        for j in range(npx):
            txj = tau0 + pxs[j] * ax                  # fixed for this px
            for k in range(npz):
                tau = txj + pzs[k] * az
                t = _tau_taper(tau, tlo, thi, tau_edge)
                nzmask = t > 0.0
                if not nzmask.any():
                    continue
                f = (tau[nzmask] - tlo) / dtau
                val = _gather(patch[:, nzmask], f, offs, cfun, ntau)
                wgt = Wk[nzmask] * t[nzmask]
                M[ip, j, k] = float(np.dot(wgt, val))
                FOLD[ip, j, k] = float(wgt.sum())

        if verbose:
            (bx, bz), v = ((float(px[np.unravel_index(
                np.argmax(np.abs(M[ip])), M[ip].shape)[0]]),
                float(pz[np.unravel_index(
                    np.argmax(np.abs(M[ip])), M[ip].shape)[1]])),
                M[ip].ravel()[np.argmax(np.abs(M[ip]))])
            print('  pivot %2d  (%7.1f, %7.1f) m   |M| peak at '
                  'p = (%+.3f, %+.3f) ms/m, M = %+.4e   fold %.3g..%.3g m^2'
                  % (ip, xp, zp, bx, bz, v, FOLD[ip].min(), FOLD[ip].max()))

    params = dict(hx=hx, hz=hz, window=window, dp=dp, pmax=pmax, tau0=tau0,
                  tau_edge=tau_edge, interp=interp, vmin=vmin, fmax=fmax,
                  normalize=bool(normalize), dx=grid.dx, dz=grid.dz,
                  dA=grid.dA, dtau=dtau, tau_min=tlo, tau_max=thi,
                  lag_factor=LAG_FACTOR, weighting='dA only (no dtau, no df)',
                  effective=eff)
    units = dict(px='ms/m', pz='ms/m', pivots='m', tau0='s', tau_edge='s',
                 hx='m', hz='m', fold='m^2',
                 M=('units(g) * m^2' if not normalize else 'units(g)'))

    if normalize:
        M = M / np.maximum(FOLD, 1.0e-30 * max(FOLD.max(), 1.0))

    for w in warns:
        warnings.warn(w, RuntimeWarning, stacklevel=2)
    return SlantResult(M, FOLD, px, pz, piv,
                       np.array([[grid.ix_of(a), grid.iz_of(b)]
                                 for a, b in piv]),
                       params, units, 'lag', warns)


# ===========================================================================
# 4.  frequency-domain implementation
# ===========================================================================
def slant2d_freq(gf, freqs, grid, pivots,
                 hx=300.0, hz=300.0, window='hann',
                 px=None, pz=None, pmax=None, dp=0.02,
                 tau0=0.0, fsel=None, df=None,
                 vmin=None, vmodel=None, fmax=None,
                 normalize=False, verbose=1):
    """The same M, evaluated from the per-frequency cross-spectrum.

    gf     COMPLEX (nfreq, nx, nz).  This is X_k = conj(A^) B^ as accumulated
           by lagGradientAccum -- i.e. the quantity lagSlice reconstructs
           from, NOT a Fourier transform of g(tau).
    freqs  (nfreq,) in Hz.

    Uses the phase found in the code, which is NEGATIVE with a factor
    LAG_FACTOR on the frequency:

        M(x0,p) = weight * Re sum_k sum_xi w(xi) X_k(x0+xi)
                           exp(-2i w_k (tau0 + px xi_x + pz xi_z)) dA

    and the reconstruction weight LAG_FACTOR*df of lag_gradient.c:482.  Pass
    df= to override; with one frequency the weight is 1 and only the SHAPE is
    meaningful, exactly as lagSlice warns.

    WEIGHTING, so the two methods are comparable: dA is applied, df enters
    only through the reconstruction weight, and dtau is NEVER applied.  This
    matches slant2d_lag, whose params['weighting'] says the same.

    Evaluated SEPARABLY: for each px the x sum is collapsed once into
    Y[k, iz], then every pz costs one length-nz sum per frequency.

    NOTE there is no tau axis here, so there are no tau limits and no
    tau-edge taper.  On data where slant2d_lag truncates, the two methods
    will legitimately disagree; keep the slopes inside the lag range to
    compare them.
    """
    gf = np.asarray(gf)
    if gf.ndim != 3:
        raise ValueError('gf must be (nfreq, nx, nz), got %s' % (gf.shape,))
    if not np.iscomplexobj(gf):
        raise ValueError('gf must be complex (the cross-spectrum conj(A)B)')
    freqs = np.asarray(freqs, dtype=float)
    if len(freqs) != gf.shape[0]:
        raise ValueError('freqs has %d entries but gf has %d'
                         % (len(freqs), gf.shape[0]))

    warns = []
    if fsel is not None:
        gf, freqs = gf[fsel], freqs[fsel]
        warns.append('frequency subset in use: %d of the original bins, '
                     '%.3f-%.3f Hz' % (len(freqs), freqs.min(), freqs.max()))
    nfreq, nx, nz = gf.shape

    if df is None:
        if nfreq > 1:
            d = np.diff(freqs)
            if np.ptp(d) > 1.0e-6 * abs(d[0]):
                raise ValueError('freqs is not uniform; pass df= explicitly')
            df = float(d[0])
        else:
            df = 0.0
    weight = LAG_FACTOR * df if df > 0.0 else 1.0
    if df <= 0.0:
        warns.append('a single frequency has no spacing, so the '
                     'reconstruction weight is set to 1 and only the SHAPE '
                     'of M is meaningful (as lagSlice warns)')

    if vmin is None and vmodel is not None:
        vmin = (model_vmin(vmodel) if isinstance(vmodel, str)
                else float(np.asarray(vmodel).min()))
    px, pz, pmax = _slope_axes(px, pz, pmax, dp, vmin, warns)

    piv = _pivot_list(pivots)
    eff = _sampling_checks(grid, px, pz, hx, hz, fmax, warns, dtau=None)

    npx, npz = len(px), len(pz)
    M = np.zeros((len(piv), npx, npz))
    FOLD = np.zeros((len(piv), npx, npz))
    pxs, pzs = px / MS_PER_S, pz / MS_PER_S
    om = 2.0 * np.pi * freqs                                   # rad/s
    ph0 = np.exp(-1j * LAG_FACTOR * om * tau0)                 # (nfreq,)

    for ip, (xp, zp) in enumerate(piv):
        (i0, k0), (ia, ib, ka, kb), xi_x, xi_z = _patch_index(
            grid, nx, nz, xp, zp, hx, hz, warns)
        wx = _window_1d(xi_x, hx, window)
        wz = _window_1d(xi_z, hz, window)
        X = gf[:, ia:ib + 1, ka:kb + 1]                        # (nf, nxp, nzp)
        fold1 = float(np.outer(wx, wz).sum()) * grid.dA

        for j in range(npx):
            # collapse x once:  Y[k, iz] = sum_ix wx exp(-2i w_k px xi_x) X
            ex = np.exp(-1j * LAG_FACTOR * np.outer(om, pxs[j] * xi_x))
            Y = np.einsum('fi,fij->fj', ex * wx[None, :], X)   # (nf, nzp)
            for k in range(npz):
                ez = np.exp(-1j * LAG_FACTOR * np.outer(om, pzs[k] * xi_z))
                s = np.einsum('fj,fj->f', ez * wz[None, :], Y)
                M[ip, j, k] = weight * grid.dA * float(np.real(ph0 @ s))
            FOLD[ip, j, :] = fold1

        if verbose:
            a = np.abs(M[ip])
            jj, kk = np.unravel_index(np.argmax(a), a.shape)
            print('  pivot %2d  (%7.1f, %7.1f) m   |M| peak at '
                  'p = (%+.3f, %+.3f) ms/m, M = %+.4e   [frequency domain]'
                  % (ip, xp, zp, px[jj], pz[kk], M[ip, jj, kk]))

    params = dict(hx=hx, hz=hz, window=window, dp=dp, pmax=pmax, tau0=tau0,
                  interp='none (analytic phase)', vmin=vmin, fmax=fmax,
                  normalize=bool(normalize), dx=grid.dx, dz=grid.dz,
                  dA=grid.dA, df=df, weight=weight, nfreq=nfreq,
                  f_lo=float(freqs.min()), f_hi=float(freqs.max()),
                  lag_factor=LAG_FACTOR,
                  weighting='dA and the %g*df reconstruction weight; '
                            'no dtau' % LAG_FACTOR,
                  effective=eff)
    units = dict(px='ms/m', pz='ms/m', pivots='m', tau0='s', hx='m', hz='m',
                 fold='m^2',
                 M=('units(g) * m^2' if not normalize else 'units(g)'))

    if normalize:
        M = M / np.maximum(FOLD, 1.0e-30 * max(FOLD.max(), 1.0))

    for w in warns:
        warnings.warn(w, RuntimeWarning, stacklevel=2)
    return SlantResult(M, FOLD, px, pz, piv,
                       np.array([[grid.ix_of(a), grid.iz_of(b)]
                                 for a, b in piv]),
                       params, units, 'frequency', warns)


# ===========================================================================
# 7.  save / load
# ===========================================================================
def save(res, fn):
    """Write a SlantResult to .npz (NumPy is the repo's only array format for
    analysis output; the gradients themselves stay in SU)."""
    if not fn.endswith('.npz'):
        fn = fn + '.npz'
    np.savez_compressed(
        fn, M=res.M, fold=res.fold, px=res.px, pz=res.pz,
        pivots=res.pivots, pivots_ij=res.pivots_ij,
        params=json.dumps(res.params, default=float),
        units=json.dumps(res.units), domain=res.domain,
        lag_convention=res.lag_convention, phase=res.phase,
        warnings=json.dumps(res.warnings))
    return fn


def load(fn):
    d = np.load(fn, allow_pickle=False)
    r = SlantResult(d['M'], d['fold'], d['px'], d['pz'], d['pivots'],
                    d['pivots_ij'], json.loads(str(d['params'])),
                    json.loads(str(d['units'])), str(d['domain']),
                    json.loads(str(d['warnings'])))
    r.lag_convention = str(d['lag_convention'])
    r.phase = str(d['phase'])
    return r


# ===========================================================================
# 7.  optional slope -> angle display helper
# ===========================================================================
def slope_to_theta(p_ms_per_m, v):
    """theta = arccos(-p v) in degrees.  FOR DISPLAY ONLY.

    p in ms/m, v in m/s.  NaN where |p v| > 1.

    This is a TRUE opening angle only when
      * the event is same-mode (P-P or S-S).  A converted P-S event has
        |k_s||k_r| = w^2/(Vp Vs) and no single v satisfies the identity;
      * v is the correct LOCAL velocity of that mode at the pivot;
      * the FULL slope vector is used -- |p| = sqrt(px^2+pz^2) with the
        bisector direction known.  Feeding one component alone measures
        cos(theta)cos(phi) with phi the bisector dip, not cos(theta).
    The factor LAG_FACTOR of the symmetric lag convention cancels the
    two-way factor of the geometry, which is why this is -cos/v and not
    -2cos/v.
    """
    c = -(np.asarray(p_ms_per_m, dtype=float) / MS_PER_S) * float(v)
    c = np.where(np.abs(c) <= 1.0, c, np.nan)
    return np.degrees(np.arccos(c))


# ===========================================================================
# 7.  one plotting helper per pivot
# ===========================================================================
def plot_pivot(res, ip, g=None, grid=None, out=None, pshow=None,
               pcircles=None, clipq=99.5, cmap='seismic', title=''):
    """One figure for pivot `ip`: optional x and z CIGs, and the (px,pz) map.

    g, grid    pass them to get the two CIGs through the pivot; omit for the
               map alone.
    pshow      (px, pz) in ms/m.  Marked on the map, and drawn as the
               MATCHING line tau = tau0 + p*(coord - pivot) in each CIG.
    pcircles   list of |p| in ms/m, drawn as circles on the map.

    pz is plotted positive DOWNWARD, matching z.  Nothing is labelled keep or
    reject.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    have_cig = g is not None and grid is not None and grid.taus is not None
    ncol = 3 if have_cig else 1
    fig, axs = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.4), squeeze=False)
    axs = axs[0]
    xp, zp = res.pivots[ip]
    tau0 = res.params.get('tau0', 0.0)

    def clip(a):
        c = np.percentile(np.abs(a), clipq) if np.any(a) else 1.0
        return max(c, 1.0e-300)

    if have_cig:
        ntau, nx, nz = g.shape
        i0, k0 = int(res.pivots_ij[ip][0]), int(res.pivots_ij[ip][1])
        hx, hz = res.params['hx'], res.params['hz']
        t1, t2 = grid.taus[0] * 1e3, grid.taus[-1] * 1e3

        # x CIG at the pivot depth
        a = g[:, :, k0]
        c = clip(a)
        xa = grid.x_axis(nx)
        axs[0].imshow(a, extent=[xa[0], xa[-1], t2, t1], aspect='auto',
                      cmap=cmap, vmin=-c, vmax=c, interpolation='nearest')
        axs[0].axvspan(xp - hx, xp + hx, color='k', alpha=0.07, lw=0)
        axs[0].axhline(tau0 * 1e3, color='0.3', lw=0.6)
        axs[0].axvline(xp, color='0.3', lw=0.6)
        axs[0].set_xlabel('x [m]')
        axs[0].set_ylabel(r'$\tau$ [ms]')
        axs[0].set_title('x CIG at z = %g m' % zp, fontsize=9)

        # z CIG at the pivot x
        a = g[:, i0, :]
        c = clip(a)
        za = grid.z_axis(nz)
        axs[1].imshow(a, extent=[za[0], za[-1], t2, t1], aspect='auto',
                      cmap=cmap, vmin=-c, vmax=c, interpolation='nearest')
        axs[1].axvspan(zp - hz, zp + hz, color='k', alpha=0.07, lw=0)
        axs[1].axhline(tau0 * 1e3, color='0.3', lw=0.6)
        axs[1].axvline(zp, color='0.3', lw=0.6)
        axs[1].set_xlabel('z [m]  (positive down)')
        axs[1].set_ylabel(r'$\tau$ [ms]')
        axs[1].set_title('z CIG at x = %g m' % xp, fontsize=9)

        if pshow is not None:
            for ax, p, c0, h in ((axs[0], pshow[0], xp, hx),
                                 (axs[1], pshow[1], zp, hz)):
                s = np.array([c0 - h, c0 + h])
                ax.plot(s, (tau0 + (p / MS_PER_S) * (s - c0)) * 1e3,
                        'k-', lw=1.3)
                ax.set_xlim(s[0] - 4 * h, s[1] + 4 * h)

    # the (px, pz) map
    ax = axs[-1]
    Mi = res.M[ip]
    c = clip(Mi)
    # extent puts pz[0] (most negative) at the top and pz[-1] at the bottom,
    # so pz increases DOWNWARD on the page, matching z.  Do not invert the
    # axis on top of this -- that would put pz positive up and contradict the
    # label.
    ax.imshow(Mi.T, extent=[res.px[0], res.px[-1], res.pz[-1], res.pz[0]],
              aspect='equal', cmap=cmap, vmin=-c, vmax=c,
              origin='upper', interpolation='nearest')
    ax.axhline(0.0, color='0.3', lw=0.6)
    ax.axvline(0.0, color='0.3', lw=0.6)
    if pcircles:
        th = np.linspace(0, 2 * np.pi, 361)
        for r in pcircles:
            ax.plot(r * np.cos(th), r * np.sin(th), 'k--', lw=0.7, alpha=0.8)
            ax.text(0.0, r, '%g' % r, fontsize=7, ha='center', va='bottom')
    if pshow is not None:
        ax.plot([pshow[0]], [pshow[1]], 'ko', mfc='none', ms=9, mew=1.4)
    ax.set_xlabel(r'$p_x$ [ms/m]')
    ax.set_ylabel(r'$p_z$ [ms/m]  (positive down)')
    ax.set_title(r'$M(x_0, p)$   $x_0$ = (%g, %g) m' % (xp, zp), fontsize=9)

    fig.suptitle('%s   aperture %g x %g m, %s window, %s, '
                 r'$\tau_0$ = %g ms, %s'
                 % (title, res.params['hx'], res.params['hz'],
                    res.params['window'], res.params['interp'],
                    tau0 * 1e3,
                    'fold-normalised' if res.params['normalize']
                    else 'weighted sum (not fold-normalised)'),
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if out:
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
    return fig


# ===========================================================================
# 9.  example run
# ===========================================================================
OPT = {'base': 'dense', 'par': 'vp,vs',
       'pivots': '', 'hx': 300.0, 'hz': 300.0, 'window': 'hann',
       'dp': 0.02, 'pmax': '', 'tau0': 0.0, 'interp': 'cubic',
       'vmin': '', 'fmax': '', 'normalize': 0, 'out': 'slant2d',
       'pshow': '', 'pcircles': ''}


def main(argv):
    for a in argv:
        if a.startswith('--') and '=' in a:
            k, v = a[2:].split('=', 1)
            if k not in OPT:
                raise SystemExit('unknown option --%s' % k)
            OPT[k] = v if isinstance(OPT[k], str) else type(OPT[k])(v)

    if not OPT['pivots']:
        raise SystemExit('give --pivots=x1,z1:x2,z2:... in metres')
    piv = [[float(t) for t in s.split(',')]
           for s in OPT['pivots'].split(':') if s]

    odir = OPT['out']
    if os.path.exists(odir) and os.listdir(odir):
        raise SystemExit('%s exists and is not empty: refusing to overwrite '
                         'existing results; pass a new --out=' % odir)
    os.makedirs(odir, exist_ok=True)

    pshow = ([float(t) for t in OPT['pshow'].split(',')]
             if OPT['pshow'] else None)
    pcirc = ([float(t) for t in OPT['pcircles'].split(',')]
             if OPT['pcircles'] else None)

    for par in OPT['par'].split(','):
        par = par.strip()
        if not par:
            continue
        print('--- %s_lag%s.su ---' % (OPT['base'], par))
        g, grid = read_lag_result(OPT['base'], par)
        print('  g %s, dx=%g dz=%g m, tau %g..%g s, dtau=%g s'
              % (g.shape, grid.dx, grid.dz, grid.taus[0], grid.taus[-1],
                 grid.dtau))

        vmin = float(OPT['vmin']) if OPT['vmin'] else None
        vmodel = None
        if vmin is None and os.path.exists(BGFILE.get(par, '')):
            vmodel = BGFILE[par]
            print('  vmin from %s' % vmodel)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = slant2d_lag(
                g, grid, piv, hx=OPT['hx'], hz=OPT['hz'],
                window=OPT['window'], dp=OPT['dp'],
                pmax=float(OPT['pmax']) if OPT['pmax'] else None,
                tau0=OPT['tau0'], interp=OPT['interp'],
                vmin=vmin, vmodel=vmodel,
                fmax=float(OPT['fmax']) if OPT['fmax'] else None,
                normalize=bool(int(OPT['normalize'])))
        for w in caught:
            if issubclass(w.category, RuntimeWarning):
                print('  WARNING: %s' % w.message)

        fn = save(res, os.path.join(odir, 'slant2d_%s_%s'
                                    % (OPT['base'], par)))
        print('  wrote %s' % fn)
        for ip in range(res.n_piv):
            out = os.path.join(odir, 'slant2d_%s_%s_piv%d.png'
                               % (OPT['base'], par, ip))
            plot_pivot(res, ip, g=g, grid=grid, out=out,
                       pshow=pshow, pcircles=pcirc,
                       title='%s  %s' % (OPT['base'], par))
            print('  wrote %s' % out)


if __name__ == '__main__':
    main(sys.argv[1:])
