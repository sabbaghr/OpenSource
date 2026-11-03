#!/usr/bin/env python3
"""3D slope filter for a lag-extended gradient cube g(x, z, tau).

A separate module.  It does not import, modify or depend on lag_slant2d.py
(the production script below imports lag_slant2d only to MEASURE the result,
never to compute it), and it touches no gradient code.

---------------------------------------------------------------------------
WHAT IS FILTERED
---------------------------------------------------------------------------
A local event in the cube is a plane

    tau = tau0 + p . xi,        p = (px, pz)  [s/m]

whose 3D Fourier support is the line  (kx, kz) = -p * nu, so that

    |k| / |nu| = |p|,       k in cycles/m,  nu in cycles/s.

The filter is therefore a cone in (kx, kz, nu).  Publicly the slope magnitude
is quoted in ms/m,

    s = 1000 * sqrt(kx^2 + kz^2) / |nu|          [ms/m]

and the response is

    H = 1                      s <= s_pass   (0.20 ms/m)
    H = cos^2(pi u / 2)        s_pass < s < s_stop,  u = (s-s_pass)/(s_stop-s_pass)
    H = 0                      s >= s_stop   (0.30 ms/m)

cos^2 and sin^2 are the same taper read from the two ends; H(s_pass) = 1,
H(midpoint) = 0.5, H(s_stop) = 0, and the first derivative vanishes at both
corners, so no corner ringing is introduced.

The z-only variant (kmode='z') replaces sqrt(kx^2+kz^2) by |kz| and is
provided for comparison only: it measures dip against the z axis alone, so a
purely horizontal event (kz = 0, kx large) has s = 0 and passes whatever its
true slope is.

nu = 0 IS HANDLED EXPLICITLY, not by an epsilon.  Taking the limit of the
definition:

    k != 0, nu -> 0   =>  s -> infinity   =>  H = 0
    k  = 0, nu  = 0   =>  s = 0/0, the cube mean; slope undefined, H = 1

so the default nu0='limit' keeps only the (0,0,0) sample of the nu = 0 plane.
That plane is the tau-INDEPENDENT part of the cube and it is not negligible,
so slope_filter3d reports the fraction of the total energy that lives there.
nu0='keep' passes the whole plane instead and is available for a sensitivity
check; it is not the limit of the stated formula.

---------------------------------------------------------------------------
EDGES
---------------------------------------------------------------------------
A cone filter is a global convolution, so a hard cube edge wraps.  Two things
are done about it and both are visible in the output:

  * a raised-cosine taper is applied to the x, z and tau edges BEFORE the
    transform.  The tapered input is returned, and reject is defined as
    `tapered - passed`, so pass + reject is the tapered field exactly and no
    energy is created or lost by the split;
  * every axis is zero-padded to `pad` times its length (default 2) before
    the FFT and cropped afterwards, so the circular convolution is linear
    over the original support.

The taper is part of the operator, not a cosmetic step: comparing a filtered
field against the RAW input would credit the taper to the filter.  Compare
against `.tapered`.

---------------------------------------------------------------------------
INTERFACE
---------------------------------------------------------------------------
    taper1d(n, frac)                   raised-cosine edge taper
    slope_response(s, s_pass, s_stop)  H(s), s in ms/m
    slope_filter3d(g, dx, dz, dtau, ...) -> FilterResult
        .passed .rejected .tapered .H_profile .params

Cube order is (ntau, nx, nz), lag slowest and z fastest -- the order written
by fdelfwi/lag_gradient.c and read by lag_slant2d.read_lag_result.

Run as a script for the case_fx_geo example; see main() and --help.
"""
import os
import sys
import json

import numpy as np

try:
    from scipy import fft as _fft
    _HAVE_SCIPY = True
except ImportError:                                    # pragma: no cover
    from numpy import fft as _fft
    _HAVE_SCIPY = False

MS_PER_S = 1.0e3

S_PASS = 0.20          # ms/m, H = 1 at or below
S_STOP = 0.30          # ms/m, H = 0 at or above


# ===========================================================================
# the response and the edge taper
# ===========================================================================
def slope_response(s, s_pass=S_PASS, s_stop=S_STOP):
    """H(s) with a cos^2 (= sin^2 read from the other end) transition.

    s in ms/m, any shape.  Non-finite s (the nu = 0 plane, if it is ever
    passed in here) is treated as infinitely steep and returns 0.
    """
    s_pass, s_stop = float(s_pass), float(s_stop)
    if not s_stop > s_pass:
        raise ValueError('need s_stop > s_pass, got %g and %g'
                         % (s_pass, s_stop))
    s = np.asarray(s, dtype=float)
    H = np.zeros(s.shape, dtype=float)
    H[s <= s_pass] = 1.0
    m = (s > s_pass) & (s < s_stop)
    if m.any():
        u = (s[m] - s_pass) / (s_stop - s_pass)
        H[m] = np.cos(0.5 * np.pi * u) ** 2
    return H


def taper1d(n, frac):
    """Raised-cosine taper over the outer `frac` of each end of n samples.

    frac = 0 gives a box.  frac = 0.5 gives a full Hann.  The taper reaches
    0 at the first and last sample, so the padded region joins smoothly.
    """
    w = np.ones(int(n), dtype=float)
    k = int(round(float(frac) * n))
    if k < 1:
        return w
    r = (np.arange(k) + 0.5) / k                       # (0,1), never 0 or 1
    e = 0.5 * (1.0 - np.cos(np.pi * r))
    w[:k] = e
    w[n - k:] = e[::-1]
    return w


# ===========================================================================
# the filter
# ===========================================================================
class FilterResult(object):
    """passed + rejected == tapered, exactly.

    passed     (ntau, nx, nz)  the s <= s_pass cone
    rejected   (ntau, nx, nz)  tapered - passed
    tapered    (ntau, nx, nz)  the input after the edge taper; compare
                               against THIS, not against the raw input
    H_profile  (s_grid, H)     the 1D response, for plotting
    params     dict, everything needed to reproduce the run
    """

    def __init__(self, passed, rejected, tapered, H_profile, params):
        self.passed = passed
        self.rejected = rejected
        self.tapered = tapered
        self.H_profile = H_profile
        self.params = params


def slope_filter3d(g, dx, dz, dtau,
                   s_pass=S_PASS, s_stop=S_STOP,
                   taper=(0.10, 0.10, 0.10), pad=2,
                   kmode='2d', nu0='limit', workers=-1, verbose=1):
    """Cone filter on the 3D spectrum of g(tau, x, z).

    g        (ntau, nx, nz) real
    dx, dz   grid spacing [m];  dtau  lag spacing [s]
    s_pass   ms/m, H = 1 at or below                        default 0.20
    s_stop   ms/m, H = 0 at or above                        default 0.30
    taper    (frac_tau, frac_x, frac_z) raised-cosine edge fractions
    pad      zero-pad factor per axis, default 2
    kmode    '2d'  s = 1000*sqrt(kx^2+kz^2)/|nu|   (the filter)
             'z'   s = 1000*|kz|/|nu|              (comparison only)
    nu0      'limit' (default) keeps only k = 0 on the nu = 0 plane, which is
             the limit of the definition;  'keep' passes the whole plane
    workers  threads for scipy.fft; ignored on the numpy fallback

    Returns a FilterResult.
    """
    g = np.asarray(g, dtype=np.float64)
    if g.ndim != 3:
        raise ValueError('g must be (ntau, nx, nz), got %s' % (g.shape,))
    if kmode not in ('2d', 'z'):
        raise ValueError("kmode must be '2d' or 'z', got %r" % kmode)
    if nu0 not in ('limit', 'keep'):
        raise ValueError("nu0 must be 'limit' or 'keep', got %r" % nu0)
    pad = int(pad)
    if pad < 1:
        raise ValueError('pad must be >= 1')

    ntau, nx, nz = g.shape
    ft, fx, fz = (float(t) for t in taper)

    wt = taper1d(ntau, ft)
    wx = taper1d(nx, fx)
    wz = taper1d(nz, fz)
    tapered = g * wt[:, None, None] * wx[None, :, None] * wz[None, None, :]

    Nt, Nx, Nz = pad * ntau, pad * nx, pad * nz

    # rfft on the TAU axis (listed last in `axes`), c2c on x and z.  That
    # puts nu >= 0 only, so |nu| = nu and the nu = 0 plane is G[0].
    kw = dict(workers=workers) if _HAVE_SCIPY else {}
    if _HAVE_SCIPY:
        G = _fft.rfftn(tapered, s=(Nx, Nz, Nt), axes=(1, 2, 0), **kw)
        nu = _fft.rfftfreq(Nt, dtau)
        kx = _fft.fftfreq(Nx, dx)
        kz = _fft.fftfreq(Nz, dz)
    else:                                              # pragma: no cover
        G = _fft.rfftn(tapered, s=(Nx, Nz, Nt), axes=(1, 2, 0))
        nu = _fft.rfftfreq(Nt, dtau)
        kx = _fft.fftfreq(Nx, dx)
        kz = _fft.fftfreq(Nz, dz)

    if kmode == '2d':
        K = np.sqrt(kx[:, None] ** 2 + kz[None, :] ** 2)
    else:
        K = np.abs(kz)[None, :] * np.ones((Nx, 1))

    # energy budget before the filter, and how much of it is tau-independent
    e_tot = float(np.vdot(G, G).real)
    e_nu0 = float(np.vdot(G[0], G[0]).real)

    # nu = 0, explicitly
    if nu0 == 'limit':
        keep0 = (K == 0.0)
        G[0] *= keep0

    # one (Nx, Nz) slice at a time: the full s array would be as large as G
    for i in range(1, len(nu)):
        G[i] *= slope_response(MS_PER_S * K / nu[i], s_pass, s_stop)

    e_pass = float(np.vdot(G, G).real)

    if _HAVE_SCIPY:
        out = _fft.irfftn(G, s=(Nx, Nz, Nt), axes=(1, 2, 0), **kw)
    else:                                              # pragma: no cover
        out = _fft.irfftn(G, s=(Nx, Nz, Nt), axes=(1, 2, 0))
    del G
    passed = np.ascontiguousarray(out[:ntau, :nx, :nz])
    del out

    sg = np.linspace(0.0, 1.5 * s_stop, 301)
    params = dict(s_pass=s_pass, s_stop=s_stop, kmode=kmode, nu0=nu0,
                  pad=pad, taper=(ft, fx, fz), dx=dx, dz=dz, dtau=dtau,
                  shape=(ntau, nx, nz), padded=(Nt, Nx, Nz),
                  nu_max=float(nu[-1]), kx_max=float(np.abs(kx).max()),
                  kz_max=float(np.abs(kz).max()),
                  energy_frac_nu0=e_nu0 / e_tot if e_tot > 0 else 0.0,
                  energy_frac_passed=e_pass / e_tot if e_tot > 0 else 0.0,
                  units=dict(s='ms/m', k='cycles/m', nu='cycles/s',
                             dx='m', dz='m', dtau='s'),
                  backend='scipy.fft' if _HAVE_SCIPY else 'numpy.fft')

    if verbose:
        print('  slope filter  kmode=%s  s_pass=%g  s_stop=%g ms/m  '
              'pad=%d  nu0=%s' % (kmode, s_pass, s_stop, pad, nu0))
        print('    cube %s -> padded %s, nu up to %.1f Hz, |k| up to '
              '%.4f cycles/m' % ((ntau, nx, nz), (Nt, Nx, Nz), nu[-1],
                                 float(np.hypot(np.abs(kx).max(),
                                                np.abs(kz).max()))))
        print('    energy on the nu = 0 plane %.3f%% of the total; '
              'spectral energy kept %.3f%%'
              % (100.0 * params['energy_frac_nu0'],
                 100.0 * params['energy_frac_passed']))

    return FilterResult(passed, tapered - passed, tapered,
                        (sg, slope_response(sg, s_pass, s_stop)), params)


# ===========================================================================
# measurement helpers used by the production run (not part of the filter)
# ===========================================================================
def banana_mask(nx, nz, dx, dz, x0, z0, xs, zs, xr, zr, v, fpeak,
                excl=400.0):
    """First-Fresnel-zone fat ray between S and R, with the ends removed.

    inside  = { P : (|PS| + |PR| - |SR|) / v  <=  1/(2 fpeak) }
    valid   = { P : |PS| > excl and |PR| > excl }

    Returns (inside & valid, ~inside & valid).  The half-period criterion is
    the usual first Fresnel zone; `excl` removes the source and receiver
    singularities, where every gradient of this kind is dominated by the
    near-field and by grad_taper rather than by the ray.
    """
    X = x0 + dx * np.arange(nx)
    Z = z0 + dz * np.arange(nz)
    XX, ZZ = X[:, None], Z[None, :]
    ds = np.hypot(XX - xs, ZZ - zs)
    dr = np.hypot(XX - xr, ZZ - zr)
    L = float(np.hypot(xr - xs, zr - zs))
    detour = (ds + dr - L) / v
    inside = detour <= 0.5 / float(fpeak)
    valid = (ds > excl) & (dr > excl)
    return inside & valid, (~inside) & valid


def rms(a, mask=None):
    b = a if mask is None else a[mask]
    return float(np.sqrt(np.mean(b.astype(np.float64) ** 2))) if b.size else 0.0


# ===========================================================================
# the case_fx_geo production run
# ===========================================================================
OPT = {'dir': 'lag_diagonal/case_fx_geo', 'base': 'dense', 'par': 'vp,vs',
       'out': 'slope_filter',
       's_pass': 0.20, 's_stop': 0.30, 'pad': 2,
       'taper': '0.10,0.10,0.10',
       'xsrc': 1302.0, 'zsrc': 600.0, 'xrcv': 3702.0, 'zrcv': 2400.0,
       'vp': 2000.0, 'vs': 1200.0, 'fpeak': 11.06, 'excl': 400.0,
       'hx': 300.0, 'hz': 300.0, 'dp': 0.02,
       'cig': '2502,1500',
       'piv_vp': '4780,1500', 'piv_vs': '2502,466',
       'clipq': 99.5}

HELP = """\
python3 lag_slope_filter.py [--key=value ...]

  --dir=        work directory holding <base>_lag{vp,vs}.su
  --base=dense  file base
  --par=vp,vs   which parameters to filter
  --out=        OUTPUT FOLDER, must not already exist
  --s_pass=0.20 --s_stop=0.30 --pad=2 --taper=ft,fx,fz
  --piv_vp= --piv_vs=   slant-stack pivot per parameter, "x,z" in m
  --cig=x,z     the CIG location
  --excl=400    radius removed around S and R
"""


def _fmt(a, q):
    c = np.percentile(np.abs(a), q)
    return (-c, c) if c > 0 else (-1.0, 1.0)


def main(argv):
    for a in argv:
        if a in ('-h', '--help'):
            print(HELP)
            return 0
        if a.startswith('--') and '=' in a:
            k, v = a[2:].split('=', 1)
            if k not in OPT:
                raise SystemExit('unknown option --%s' % k)
            OPT[k] = type(OPT[k])(v) if not isinstance(OPT[k], str) else v
        else:
            raise SystemExit('unexpected argument %r' % a)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lag_slant2d as ls                       # MEASUREMENT only

    cwd = os.getcwd()
    os.chdir(OPT['dir'])
    out = OPT['out']
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit('refusing to overwrite a non-empty %s' % out)
    if not os.path.exists(out):
        os.makedirs(out)

    tfrac = tuple(float(t) for t in OPT['taper'].split(','))
    xs, zs = OPT['xsrc'], OPT['zsrc']
    xr, zr = OPT['xrcv'], OPT['zrcv']
    cx, cz = (float(t) for t in OPT['cig'].split(','))
    report = {}

    for par in OPT['par'].split(','):
        print('\n=== %s ===' % par)
        g, grid = ls.read_lag_result(OPT['base'], par)
        ntau, nx, nz = g.shape
        dtau = grid.dtau
        it0 = int(np.argmin(np.abs(grid.taus)))
        v = OPT['vp'] if par == 'vp' else OPT['vs']

        res = slope_filter3d(g, grid.dx, grid.dz, dtau,
                             s_pass=OPT['s_pass'], s_stop=OPT['s_stop'],
                             taper=tfrac, pad=int(OPT['pad']), kmode='2d')
        rz_ = slope_filter3d(g, grid.dx, grid.dz, dtau,
                             s_pass=OPT['s_pass'], s_stop=OPT['s_stop'],
                             taper=tfrac, pad=int(OPT['pad']), kmode='z')
        del g

        # ------------------------------------------------------------- 1, 2
        A = res.tapered[it0]
        P, R = res.passed[it0], res.rejected[it0]
        Pz, Rz = rz_.passed[it0], rz_.rejected[it0]
        vmin, vmax = _fmt(A, OPT['clipq'])
        ext = [grid.x0, grid.x0 + grid.dx * (nx - 1),
               grid.z0 + grid.dz * (nz - 1), grid.z0]

        fig, axs = plt.subplots(2, 3, figsize=(15.0, 7.4))
        panels = [(0, 0, A, 'tapered input   tau = 0'),
                  (0, 1, P, '2D pass   s <= %g ms/m' % OPT['s_pass']),
                  (0, 2, R, '2D reject'),
                  (1, 1, Pz, 'z-only pass   |kz| in place of |k|'),
                  (1, 2, Rz, 'z-only reject')]
        for i, j, img, ttl in panels:
            ax = axs[i][j]
            ax.imshow(img.T, extent=ext, aspect='equal', cmap='seismic',
                      vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.plot([xs, xr], [zs, zr], 'k--', lw=0.7, alpha=0.6)
            ax.plot([xs], [zs], 'k*', ms=8)
            ax.plot([xr], [zr], 'kv', ms=7)
            ax.set_title(ttl, fontsize=9)
            ax.set_xlabel('x [m]')
            if j == 0 or (i == 1 and j == 1):
                ax.set_ylabel('z [m]  (positive down)')
        sg, Hg = res.H_profile
        axs[1][0].plot(sg, Hg, 'k-', lw=1.6)
        axs[1][0].axvline(OPT['s_pass'], color='0.5', ls=':')
        axs[1][0].axvline(OPT['s_stop'], color='0.5', ls=':')
        axs[1][0].set_xlabel('s = 1000 |k| / |nu|   [ms/m]')
        axs[1][0].set_ylabel('H')
        axs[1][0].set_title('response (cos$^2$ transition)', fontsize=9)
        axs[1][0].set_ylim(-0.05, 1.05)
        axs[1][0].grid(alpha=0.3)
        fig.suptitle('%s  tau = 0, all image panels on one clip '
                     '(%.1f%% of |tapered input|)' % (par, OPT['clipq']),
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fn = os.path.join(out, 'tau0_%s.png' % par)
        fig.savefig(fn, dpi=120)
        plt.close(fig)
        print('  wrote %s' % fn)

        # ---------------------------------------------------------------- 3
        pv = OPT['piv_vp'] if par == 'vp' else OPT['piv_vs']
        px_, pz_ = (float(t) for t in pv.split(','))
        pmax = 1.1 / OPT['vs'] * MS_PER_S
        sl = {}
        import warnings as _w
        for lab, cube in (('before', res.tapered), ('after', res.passed)):
            with _w.catch_warnings(record=True) as wl:
                _w.simplefilter('always')
                sl[lab] = ls.slant2d_lag(cube, grid, [[px_, pz_]],
                                         hx=OPT['hx'], hz=OPT['hz'],
                                         window='hann', pmax=pmax,
                                         dp=OPT['dp'], tau0=0.0,
                                         interp='cubic', verbose=0)
            for m in wl:
                if 'model edge' in str(m.message):
                    print('  WARNING %s' % m.message)

        fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.8))
        cl = np.percentile(np.abs(sl['before'].M[0]), 99.9)
        for ax, lab in zip(axs, ('before', 'after')):
            M = sl[lab].M[0]
            ax.imshow(M.T, extent=[sl[lab].px[0], sl[lab].px[-1],
                                   sl[lab].pz[-1], sl[lab].pz[0]],
                      aspect='equal', cmap='seismic', vmin=-cl, vmax=cl,
                      origin='upper', interpolation='nearest')
            th = np.linspace(0, 2 * np.pi, 361)
            for rr, st in ((OPT['s_pass'], '-'), (OPT['s_stop'], '--')):
                ax.plot(rr * np.cos(th), rr * np.sin(th), 'k', ls=st, lw=0.8)
            ax.axhline(0, color='k', lw=0.4)
            ax.axvline(0, color='k', lw=0.4)
            ax.set_xlabel('$p_x$ [ms/m]')
            ax.set_ylabel('$p_z$ [ms/m]  (positive down)')
            ax.set_title('%s   |M| peak %.3e' % (lab, np.abs(M).max()),
                         fontsize=9)
        b = np.abs(sl['before'].M[0]).max()
        a_ = np.abs(sl['after'].M[0]).max()
        fig.suptitle('%s  slant stack at (%g, %g) m, hann %g x %g m; '
                     'circles s_pass / s_stop' % (par, px_, pz_,
                                                  OPT['hx'], OPT['hz']),
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fn = os.path.join(out, 'slant_%s.png' % par)
        fig.savefig(fn, dpi=120)
        plt.close(fig)
        print('  wrote %s' % fn)
        np.savez_compressed(os.path.join(out, 'slant_%s.npz' % par),
                            px=sl['before'].px, pz=sl['before'].pz,
                            M_before=sl['before'].M[0],
                            M_after=sl['after'].M[0],
                            fold=sl['before'].fold[0], pivot=[px_, pz_])

        # ---------------------------------------------------------------- 4
        ix, iz = grid.ix_of(cx), grid.iz_of(cz)
        tms = grid.taus * 1e3
        fig, axs = plt.subplots(2, 2, figsize=(12.0, 7.6))
        cigs = [(0, res.tapered, 'before'), (1, res.passed, 'after')]
        cl_x = np.percentile(np.abs(res.tapered[:, :, iz]), OPT['clipq'])
        cl_z = np.percentile(np.abs(res.tapered[:, ix, :]), OPT['clipq'])
        for r, cube, lab in cigs:
            axs[r][0].imshow(cube[:, :, iz], aspect='auto', cmap='seismic',
                             vmin=-cl_x, vmax=cl_x, interpolation='nearest',
                             extent=[grid.x0, grid.x0 + grid.dx * (nx - 1),
                                     tms[-1], tms[0]])
            axs[r][0].set_title('x CIG at z = %g m   %s' % (cz, lab),
                                fontsize=9)
            axs[r][0].set_xlabel('x [m]')
            axs[r][1].imshow(cube[:, ix, :], aspect='auto', cmap='seismic',
                             vmin=-cl_z, vmax=cl_z, interpolation='nearest',
                             extent=[grid.z0, grid.z0 + grid.dz * (nz - 1),
                                     tms[-1], tms[0]])
            axs[r][1].set_title('z CIG at x = %g m   %s' % (cx, lab),
                                fontsize=9)
            axs[r][1].set_xlabel('z [m]  (positive down)')
            for c in (0, 1):
                axs[r][c].axhline(0.0, color='k', lw=0.4)
                axs[r][c].set_ylabel(r'$\tau$ [ms]')
        fig.suptitle('%s  CIGs through (%g, %g) m, same clip per column'
                     % (par, cx, cz), fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fn = os.path.join(out, 'cig_%s.png' % par)
        fig.savefig(fn, dpi=120)
        plt.close(fig)
        print('  wrote %s' % fn)

        # ----------------------------------------------------------- report
        inb, outb = banana_mask(nx, nz, grid.dx, grid.dz, grid.x0, grid.z0,
                                xs, zs, xr, zr, v, OPT['fpeak'],
                                excl=OPT['excl'])
        rec = dict(
            pivot=[px_, pz_],
            peak_before=float(np.abs(sl['before'].M[0]).max()),
            peak_after=float(a_),
            peak_ratio=float(a_ / b) if b > 0 else float('nan'),
            n_in=int(inb.sum()), n_out=int(outb.sum()),
            rms_in_before=rms(A, inb), rms_out_before=rms(A, outb),
            rms_in_after=rms(P, inb), rms_out_after=rms(P, outb),
            rms_in_reject=rms(R, inb), rms_out_reject=rms(R, outb),
            energy_frac_nu0=res.params['energy_frac_nu0'],
            energy_frac_passed=res.params['energy_frac_passed'])
        rec['ratio_before'] = rec['rms_in_before'] / rec['rms_out_before']
        rec['ratio_after'] = rec['rms_in_after'] / rec['rms_out_after']
        rec['ratio_reject'] = rec['rms_in_reject'] / rec['rms_out_reject']
        rec['reject_over_input_in_banana'] = (rec['rms_in_reject']
                                              / rec['rms_in_before'])
        report[par] = rec

        print('  banana  %d points inside, %d outside (|PS|,|PR| > %g m)'
              % (rec['n_in'], rec['n_out'], OPT['excl']))
        print('  rms in/out   before %.4e / %.4e  ratio %.3f'
              % (rec['rms_in_before'], rec['rms_out_before'],
                 rec['ratio_before']))
        print('               after  %.4e / %.4e  ratio %.3f'
              % (rec['rms_in_after'], rec['rms_out_after'],
                 rec['ratio_after']))
        print('               reject %.4e / %.4e  ratio %.3f'
              % (rec['rms_in_reject'], rec['rms_out_reject'],
                 rec['ratio_reject']))
        print('  slant peak |M| %.4e -> %.4e   ratio %.4f'
              % (rec['peak_before'], rec['peak_after'], rec['peak_ratio']))

        np.save(os.path.join(out, 'tau0_%s_input.npy' % par), A)
        np.save(os.path.join(out, 'tau0_%s_pass.npy' % par), P)
        np.save(os.path.join(out, 'tau0_%s_reject.npy' % par), R)
        np.save(os.path.join(out, 'tau0_%s_pass_zonly.npy' % par), Pz)
        del res, rz_

    with open(os.path.join(out, 'report.json'), 'w') as fp:
        json.dump(report, fp, indent=2, sort_keys=True)
    print('\n  wrote %s' % os.path.join(out, 'report.json'))
    os.chdir(cwd)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
