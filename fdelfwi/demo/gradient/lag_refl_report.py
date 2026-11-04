#!/usr/bin/env python3
"""Analysis for test_lag_reflector.sh: slopes, the cone filter, and metrics.

Reads the two lag cubes of each run, measures their slopes against ray
theory, runs the EXISTING cone filter (lag_slope_filter.py) at four cutoffs,
and reports energy fractions inside masks that are built from traveltimes
rather than drawn by eye.  It computes nothing about inversion and makes no
claim about it.

WHAT IS MEASURED, AND HOW THE MASKS ARE DEFINED
-----------------------------------------------
isochron mask      For a residual event at data time T_D imaged with leg
                   modes (a, b), equation (1) of lag_refl_rays gives
                   tau(x, z); the event lies on its tau = 0 contour and is
                   one wavelet thick.  The mask is therefore

                       |tau(x, z)|  <=  1 / (4 f_peak)

                   i.e. the first quarter period, which in metres normal to
                   the isochron is 2/(4 f_peak |p|) -- 109 m for the P/P
                   isochron and 65 m for S/S at this geometry.  Both widths
                   are printed, because a mask whose width is not stated is
                   not a measurement.

ear mask           A transmission kernel lies along a ray, so the mask is the
                   first Fresnel zone (fat ray) around a REFLECTED leg.  The
                   leg is straightened by mirroring the far end of the path
                   in the reflector -- down-leg: S to the mirrored receiver;
                   up-leg: the mirrored source to R -- so the standard
                   two-focus detour criterion applies unchanged:

                       (|xA| + |xB| - |AB|) / v  <=  1 / (2 f_peak)

                   intersected with z <= z_refl and with a 2 sigma ball round
                   the anomaly, so that the two anomaly sites are separated.

exclusion          OFF.  Nothing is blanked around S or R.  test_lag_diagonal
                   .sh -- the test this one is meant to be read against --
                   masks nothing in its analysis, so masking here would make
                   the two incomparable.  The only source/receiver
                   conditioning is grad_taper = 30 grid points (180 m), which
                   lives INSIDE the gradient (grad_taper_mask.c: a parabolic
                   hole w = r^2/R^2, zero at each source AND each receiver)
                   and is applied per lag by lag_gradient.c.  It is therefore
                   already baked into the cube, identically in both tests, and
                   is reported rather than re-applied.  `--excl` is kept as a
                   switch: pass a positive radius to blank a disk anyway.

Energy fractions are reported for the pass and the reject panel separately
and they do NOT sum to one: the split is exact in AMPLITUDE (pass + reject =
tapered input) and so the energies carry a cross term, which is printed
rather than hidden.

Peaks in a (px, pz) map are found as LOCAL maxima and matched to the
prediction by slope distance.  The global argmax is printed too, but it is
not what the lobed patterns of a real kernel should be judged on.
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lag_refl_rays as RY                                    # noqa: E402
import lag_slant2d as LS                                      # noqa: E402
import lag_slope_filter as SF                                 # noqa: E402

SIGMA = 300.0
MIDPOINTS = (0.15, 0.25, 0.40)        # extra cutoff midpoints to sweep
HALFWIDTH = 0.05                      # s_pass = mid - this, s_stop = mid +
RUNS = ('runA', 'runB')
PARS = ('vp', 'vs')
V = {'P': None, 'S': None}            # filled from the command line


# ---------------------------------------------------------------------------
def local_peak_near(M, px, pz, p_pred):
    """The local |M| maximum nearest p_pred, plus the global argmax.

    A real kernel's slant map is lobed -- an interference pattern, not a
    single blob -- so the global argmax can sit on a sidelobe or on an
    aperture arm and say nothing about the event under test.  Returns
    (p_local, val_local, p_global, val_global, n_local).
    """
    a = np.abs(M)
    i = np.zeros(a.shape, dtype=bool)
    i[1:-1, 1:-1] = ((a[1:-1, 1:-1] >= a[:-2, 1:-1])
                     & (a[1:-1, 1:-1] >= a[2:, 1:-1])
                     & (a[1:-1, 1:-1] >= a[1:-1, :-2])
                     & (a[1:-1, 1:-1] >= a[1:-1, 2:])
                     & (a[1:-1, 1:-1] >= a[:-2, :-2])
                     & (a[1:-1, 1:-1] >= a[2:, 2:])
                     & (a[1:-1, 1:-1] >= a[:-2, 2:])
                     & (a[1:-1, 1:-1] >= a[2:, :-2])
                     & (a[1:-1, 1:-1] > 0.0))
    j, k = np.nonzero(i)
    gj, gk = np.unravel_index(int(np.argmax(a)), a.shape)
    pg = (float(px[gj]), float(pz[gk]))
    if not len(j):
        return pg, float(a[gj, gk]), pg, float(a[gj, gk]), 0
    d = np.hypot(px[j] - p_pred[0], pz[k] - p_pred[1])
    m = int(np.argmin(d))
    return ((float(px[j[m]]), float(pz[k[m]])), float(a[j[m], k[m]]),
            pg, float(a[gj, gk]), len(j))


def fit_slope(M, px, pz, p_pred, rad=0.12):
    """Amplitude-weighted centroid of |M| within `rad` of the prediction.

    A centroid, not a peak: it is what a line FITTED to the event in the two
    CIGs converges to, and unlike an argmax it moves continuously, so the
    difference between it and the local peak is itself informative about how
    lobed the map is.
    """
    PX, PZ = px[:, None], pz[None, :]
    w = np.abs(M) * (np.hypot(PX - p_pred[0], PZ - p_pred[1]) <= rad)
    s = w.sum()
    if s <= 0:
        return None
    return float((w * PX).sum() / s), float((w * PZ).sum() / s)


def excl_mask(X, Z, xs, zs, xr, zr, excl):
    """All-true unless a positive exclusion radius is asked for.

    Off by default: test_lag_diagonal.sh masks nothing, and the only
    source/receiver conditioning in either test is grad_taper, which lives
    inside the gradient and is already in the cube.
    """
    if excl <= 0.0:
        return np.ones(np.broadcast(X, Z).shape, dtype=bool)
    return ((np.hypot(X - xs, Z - zs) > excl)
            & (np.hypot(X - xr, Z - zr) > excl))


def tube_mask(X, Z, xa, za, xb, zb, v, fpeak):
    L = float(np.hypot(xb - xa, zb - za))
    return ((np.hypot(X - xa, Z - za) + np.hypot(X - xb, Z - zb) - L) / v
            <= 0.5 / fpeak)


def nrg(a, m=None):
    b = a if m is None else a[m]
    return float(np.sum(b.astype(np.float64) ** 2))


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--work', required=True)
    for k in ('dx', 'xsrc', 'zsrc', 'xrcv', 'zrcv', 'zrefl', 'xspec',
              'zspec', 'vp', 'vs', 't0', 'fpeak', 'xa', 'za', 'xb', 'zb',
              'haper', 'excl'):
        p.add_argument('--' + k, type=float, required=True)
    p.add_argument('--s-pass', dest='s_pass', type=float, required=True)
    p.add_argument('--s-stop', dest='s_stop', type=float, required=True)
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    XS, ZS, XR, ZR = a.xsrc, a.zsrc, a.xrcv, a.zrcv
    VP, VS, ZF, FP = a.vp, a.vs, a.zrefl, a.fpeak
    V['P'], V['S'] = VP, VS
    out = os.path.join(a.work, 'analysis')
    if not os.path.isdir(out):
        os.makedirs(out)
    rep = dict(geometry=dict(xsrc=XS, zsrc=ZS, xrcv=XR, zrcv=ZR, zrefl=ZF,
                             vp=VP, vs=VS, fpeak=FP, t0=a.t0,
                             haper=a.haper, excl=a.excl,
                             s_pass=a.s_pass, s_stop=a.s_stop))

    # -- the prediction catalogue -----------------------------------------
    tab = RY.arrival_table(XS, ZS, XR, ZR, ZF, VP, VS, t0=a.t0)
    T = {d['name']: d['t'] for d in tab}
    XC = {d['name']: d['xc'] for d in tab}
    SM, RM = (XS, 2.0 * ZF - ZS), (XR, 2.0 * ZF - ZR)

    evlist = ('PP', 'PS', 'SP', 'SS', 'direct P', 'direct S')
    print('=== predicted events: every residual event crossed with every '
          'leg-mode pair ===')
    print('A residual event E imaged with legs (a,b) puts a tau = 0 isochron')
    print('wherever t_a(S,x) + t_b(x,R) = T_D(E).  %d events x 4 leg pairs = '
          '%d combinations;' % (len(evlist), 4 * len(evlist)))
    print('only the 4 with legs == event are MATCHED, the rest are MODE')
    print('CROSSTALK.  xq is the column the depth is quoted at; a combination')
    print('with no tau = 0 crossing above the listed depth images nowhere.')
    print('%-9s %-5s %8s %8s %9s %9s %8s %8s %8s'
          % ('residual', 'legs', 'xq', 'z0', 'px', 'pz', '|p|', '2theta',
             'width'))
    cat = []
    for ev in evlist:
        xq = 2502.0 if XC.get(ev) is None else XC[ev]
        for la in 'PS':
            for lb in 'PS':
                z0 = RY.isochron_depth(T[ev], XS, ZS, XR, ZR, V[la], V[lb],
                                       xq, ZS + 1.0, ZF + 700.0)
                if z0 is None:
                    print('%-9s %-5s %8.0f   no tau = 0 crossing above '
                          'z = %.0f m' % (ev, la + lb, xq, ZF + 700.0))
                    continue
                px, pz = RY.slope(xq, z0, XS, ZS, XR, ZR, V[la], V[lb])
                pm = float(np.hypot(px, pz))
                th = float(RY.opening_angle(xq, z0, XS, ZS, XR, ZR))
                wd = 2.0 * (0.25 / FP) * 1.0e3 / pm          # metres
                print('%-9s %-5s %8.0f %8.1f %+9.4f %+9.4f %8.4f %8.2f '
                      '%7.0fm' % (ev, la + lb, xq, z0, px, pz, pm, th, wd))
                cat.append(dict(res=ev, legs=la + lb, xq=xq, z0=z0,
                                px=float(px), pz=float(pz), pmag=pm,
                                theta2=th, width_m=wd,
                                matched=(la + lb == ev)))
    rep['catalogue'] = cat
    mat = [c for c in cat if c['matched']]
    xtk = [c for c in cat if not c['matched']]
    print('')
    print('%d of the %d combinations image somewhere: %d matched, %d mode '
          'crosstalk.' % (len(cat), 4 * len(evlist), len(mat), len(xtk)))
    print('  matched   |p| = %.3f .. %.3f ms/m'
          % (min(c['pmag'] for c in mat), max(c['pmag'] for c in mat)))
    print('  crosstalk |p| = %.3f .. %.3f ms/m'
          % (min(c['pmag'] for c in xtk), max(c['pmag'] for c in xtk)))
    ov = (max(min(c['pmag'] for c in mat), min(c['pmag'] for c in xtk)),
          min(max(c['pmag'] for c in mat), max(c['pmag'] for c in xtk)))
    inov = [c['legs'] for c in mat if ov[0] <= c['pmag'] <= ov[1]]
    print('THE TWO RANGES OVERLAP over |p| = %.3f .. %.3f ms/m, and that '
          'interval holds the' % ov)
    print('matched isochrons %s -- i.e. %d of the %d.  A cone with a '
          '%.2f-%.2f ms/m cutoff' % (', '.join(inov), len(inov), len(mat),
                                     a.s_pass, a.s_stop))
    print('therefore rejects')
    print('the matched specular isochrons and the mode-crosstalk isochrons '
          'alike: high |p|')
    print('does NOT mean specular.  The panels below are labelled by slope, '
          'not by event')
    print('type, and the mask energies are what decide which events a panel '
          'holds.')
    rep['catalogue_summary'] = dict(
        n_imaging=len(cat), n_total=4 * len(evlist), n_matched=len(mat),
        pmag_matched=[min(c['pmag'] for c in mat),
                      max(c['pmag'] for c in mat)],
        pmag_crosstalk=[min(c['pmag'] for c in xtk),
                        max(c['pmag'] for c in xtk)],
        overlap=list(ov))

    # -- the pivots -------------------------------------------------------
    # x snapped to the grid; lag_slant2d reports the index it actually used.
    def sn(x):
        return float(round(x / a.dx) * a.dx)

    PIV = {
        ('runA', 'vp'): [('P/P specular', sn(2502.0), ZF, 'PP', 'PP'),
                         ('S/P conversion', sn(XC['SP']), ZF, 'SP', 'SP'),
                         ('P/S conversion', sn(XC['PS']), ZF, 'PS', 'PS')],
        ('runA', 'vs'): [('S/S specular', sn(2502.0), ZF, 'SS', 'SS'),
                         ('P/S conversion', sn(XC['PS']), ZF, 'PS', 'PS'),
                         ('S/P conversion', sn(XC['SP']), ZF, 'SP', 'SP')],
        ('runB', 'vp'): [('gA  PP down-leg', a.xa, a.za, 'ear_down', None),
                         ('gB  PP up-leg', a.xb, a.zb, 'ear_up', None),
                         ('P/P isochron', sn(2502.0), ZF, 'PP', 'PP')],
        ('runB', 'vs'): [('gA  PP down-leg', a.xa, a.za, 'ear_down', None),
                         ('gB  SS up-leg', a.xb, a.zb, 'ear_up_s', None),
                         ('S/S isochron', sn(2502.0), ZF, 'SS', 'SS')],
    }

    def predict(kind, xp, zp):
        """The predicted (px, pz) at a pivot, and the construction used."""
        if kind == 'ear_down':
            return RY.slope(xp, zp, XS, ZS, RM[0], RM[1], VP, VP), \
                'transmission along the P down-leg, receiver mirrored in ' \
                'the reflector'
        if kind == 'ear_up':
            return RY.slope(xp, zp, SM[0], SM[1], XR, ZR, VP, VP), \
                'transmission along the P up-leg, source mirrored'
        if kind == 'ear_up_s':
            return RY.slope(xp, zp, SM[0], SM[1], XR, ZR, VS, VS), \
                'transmission along the S up-leg, source mirrored'
        la, lb = kind[0], kind[1]
        return RY.slope(xp, zp, XS, ZS, XR, ZR, V[la], V[lb]), \
            'specular %s isochron, direct legs' % kind

    # apex lags: the kernel of a timing residual peaks at HALF the delay
    leff = SIGMA * np.sqrt(2.0 * np.pi)
    dtp = leff * (1.0 / (VP * 0.95) - 1.0 / VP)
    dts = leff * (1.0 / (VS * 0.95) - 1.0 / VS)
    print('')
    print('=== lag anchoring ===')
    print('  tau0 = 0 is used for every pivot, as specified.  For run B the')
    print('  kernel of a TIMING residual has its apex at half the delay, so')
    print('  a sweep over tau0 is reported alongside: ray delay through a')
    print('  sigma = %.0f m, -5%% Gaussian is %.1f ms in P and %.1f ms in S,'
          % (SIGMA, 1.0e3 * dtp, 1.0e3 * dts))
    print('  i.e. apex lags of %.1f and %.1f ms (%.1f and %.1f samples of '
          'the 2 ms lag axis).' % (0.5e3 * dtp, 0.5e3 * dts,
                                   0.25e3 * dtp, 0.25e3 * dts))

    # -- per run / parameter ----------------------------------------------
    rep['runs'] = {}
    tau0_panels = []
    sweep = {}
    for run in RUNS:
        d = os.path.join(a.work, run)
        if not os.path.isdir(d):
            print('  %s missing, skipped' % d)
            continue
        if not all(os.path.isfile(os.path.join(d, 'dense_lag%s.su' % q))
                   for q in PARS):
            print('  %s has no lag cubes yet, skipped' % d)
            continue
        cwd = os.getcwd()
        os.chdir(d)
        for par in PARS:
            key = '%s/%s' % (run, par)
            print('')
            print('=========================================================')
            print(' %s' % key)
            print('=========================================================')
            g, grid = LS.read_lag_result('dense', par)
            ntau, nx, nz = g.shape
            it0 = int(np.argmin(np.abs(grid.taus)))
            X = (grid.x0 + grid.dx * np.arange(nx))[:, None]
            Z = (grid.z0 + grid.dz * np.arange(nz))[None, :]
            val = excl_mask(X, Z, XS, ZS, XR, ZR, a.excl)
            print('  cube %s, dx %.0f dz %.0f, tau %.3f..%.3f s (%d lags), '
                  'tau = 0 at index %d'
                  % ((ntau, nx, nz), grid.dx, grid.dz, grid.taus[0],
                     grid.taus[-1], ntau, it0))
            if a.excl > 0:
                print('  exclusion: %.0f m around S and R removes %d of %d '
                      'points (%.1f%%)'
                      % (a.excl, int((~val).sum()), val.size,
                         100.0 * (~val).sum() / val.size))
            else:
                print('  no exclusion around S or R, as in test_lag_diagonal: '
                      'the only source/receiver conditioning is grad_taper = '
                      '30 pts = 180 m, already inside the cube')

            # ---- the cone filter, primary cutoff, full cube -------------
            res = SF.slope_filter3d(g, grid.dx, grid.dz, grid.dtau,
                                    s_pass=a.s_pass, s_stop=a.s_stop,
                                    taper=(0.10, 0.10, 0.10), pad=2,
                                    kmode='2d', verbose=1)
            del g
            A0, P0, R0 = res.tapered[it0], res.passed[it0], res.rejected[it0]
            tau0_panels.append((key, A0, P0, R0, grid, nx, nz))

            # ---- masks ---------------------------------------------------
            masks = {}
            for c in cat:
                if c['res'] in ('direct P', 'direct S') and run == 'runA':
                    continue           # run A has no direct residual at all
                tau = RY.isochron(T[c['res']], XS, ZS, XR, ZR,
                                  V[c['legs'][0]], V[c['legs'][1]],
                                  X[:, 0], Z[0, :])
                m = (np.abs(tau) <= 0.25 / FP) & val
                if m.sum():
                    masks['iso %s/%s' % (c['res'], c['legs'])] = m
            for nm, xa_, za_, aa, bb, vv in (
                    ('ear gA P down', a.xa, a.za, (XS, ZS), RM, VP),
                    ('ear gB P up', a.xb, a.zb, SM, (XR, ZR), VP),
                    ('ear gB S up', a.xb, a.zb, SM, (XR, ZR), VS)):
                m = (tube_mask(X, Z, aa[0], aa[1], bb[0], bb[1], vv, FP)
                     & (Z <= ZF) & (np.hypot(X - xa_, Z - za_) <= 2.0 * SIGMA)
                     & val)
                if m.sum():
                    masks[nm] = m
            print('  masks (tau = 0 slice, after exclusion):')
            print('  %-22s %8s %10s %10s' % ('mask', 'points', 'area [km2]',
                                             'width [m]'))
            for nm in sorted(masks):
                m = masks[nm]
                wd = np.nan
                for c in cat:
                    if nm == 'iso %s/%s' % (c['res'], c['legs']):
                        wd = c['width_m']
                print('  %-22s %8d %10.4f %10s'
                      % (nm, int(m.sum()),
                         1.0e-6 * m.sum() * grid.dx * grid.dz,
                         '-' if np.isnan(wd) else '%.0f' % wd))

            # ---- energy fractions, primary cutoff -----------------------
            print('  energy at tau = 0 inside each mask, primary cone '
                  's_pass=%.2f s_stop=%.2f ms/m:' % (a.s_pass, a.s_stop))
            print('  %-22s %11s %9s %9s %9s'
                  % ('mask', '||A||^2', 'pass', 'reject', 'cross'))
            emet = {}
            for nm in sorted(masks):
                m = masks[nm]
                ea, ep, er = nrg(A0, m), nrg(P0, m), nrg(R0, m)
                cr = 1.0 - (ep + er) / ea if ea > 0 else 0.0
                emet[nm] = dict(nA=ea, fpass=ep / ea if ea else 0.0,
                                frej=er / ea if ea else 0.0, cross=cr,
                                npts=int(m.sum()))
                print('  %-22s %11.4e %9.4f %9.4f %+9.4f'
                      % (nm, ea, emet[nm]['fpass'], emet[nm]['frej'], cr))
            ev, epv, erv = nrg(A0, val), nrg(P0, val), nrg(R0, val)
            covered = np.zeros(A0.shape, dtype=bool)
            for m in masks.values():
                covered |= m
            unk = val & ~covered
            print('  %-22s %11.4e %9.4f %9.4f %+9.4f   <- everything valid'
                  % ('all valid', ev, epv / ev, erv / ev,
                     1.0 - (epv + erv) / ev))
            eu, pu, ru = nrg(A0, unk), nrg(P0, unk), nrg(R0, unk)
            print('  %-22s %11.4e %9.4f %9.4f %+9.4f   <- UNKNOWN: valid but '
                  'on no predicted event (%.1f%% of the valid area, '
                  '%.4f of the valid energy)'
                  % ('unexplained', eu,
                     pu / eu if eu > 0 else 0.0,
                     ru / eu if eu > 0 else 0.0,
                     1.0 - (pu + ru) / eu if eu > 0 else 0.0,
                     100.0 * unk.sum() / val.sum(),
                     eu / ev if ev > 0 else 0.0))
            print('  (pass and reject fractions do not sum to 1: the split '
                  'is exact in AMPLITUDE,')
            print('   A = P + R, so the energies carry the cross term '
                  'printed above.)')

            # ---- slant stacks, before and after -------------------------
            pivs = PIV[(run, par)]
            print('  slant stacks, Hann aperture half-width %.0f m, '
                  'dp = 0.02 ms/m, tau0 = 0:' % a.haper)
            print('  %-16s %-34s %17s %17s %8s %9s'
                  % ('pivot', 'prediction', 'local peak (in)',
                     'local peak (pass)', 'err', 'fold m2'))
            rows = []
            for (nm, xp, zp, kind, _lg) in pivs:
                (ppx, ppz), how = predict(kind, xp, zp)
                pp = (float(ppx), float(ppz))
                sres = {}
                for lab, cube in (('in', res.tapered),
                                  ('pass', res.passed),
                                  ('rej', res.rejected)):
                    sres[lab] = LS.slant2d_lag(
                        cube, grid, [[xp, zp]], hx=a.haper, hz=a.haper,
                        window='hann', dp=0.02, tau0=0.0, interp='cubic',
                        vmin=min(VP, VS), fmax=36.0, verbose=0)
                pk_i = local_peak_near(sres['in'].M[0], sres['in'].px,
                                       sres['in'].pz, pp)
                pk_p = local_peak_near(sres['pass'].M[0], sres['pass'].px,
                                       sres['pass'].pz, pp)
                fit = fit_slope(sres['in'].M[0], sres['in'].px,
                                sres['in'].pz, pp)
                fold = sres['in'].fold[0]
                f0 = float(fold.max())
                fclip = float(fold.min()) / f0 if f0 > 0 else 0.0
                err = float(np.hypot(pk_i[0][0] - pp[0], pk_i[0][1] - pp[1]))
                print('  %-16s (%+.4f,%+.4f) %-14s (%+.3f,%+.3f) '
                      '(%+.3f,%+.3f) %8.3f %9.3g'
                      % (nm, pp[0], pp[1], kind, pk_i[0][0], pk_i[0][1],
                         pk_p[0][0], pk_p[0][1], err, f0))
                rows.append(dict(pivot=nm, x=xp, z=zp, kind=kind, how=how,
                                 pred=pp, peak_in=pk_i[0], val_in=pk_i[1],
                                 peak_pass=pk_p[0], val_pass=pk_p[1],
                                 gmax_in=pk_i[2], gval_in=pk_i[3],
                                 n_local=pk_i[4],
                                 fit=fit, err=err, fold=f0, fold_clip=fclip,
                                 sres=sres))
            for r in rows:
                print('    %-16s %s' % (r['pivot'], r['how']))
                print('      fold %.4g m2 at best px, %.3f of that at the '
                      'worst px (1.000 = no aperture clipping)'
                      % (r['fold'], r['fold_clip']))
                print('      local maxima in the input map: %d.  The one '
                      'nearest the prediction carries %.3f of the GLOBAL '
                      'argmax (%+.3f,%+.3f) -- so the global argmax is %s'
                      % (r['n_local'],
                         r['val_in'] / max(r['gval_in'], 1e-300),
                         r['gmax_in'][0], r['gmax_in'][1],
                         'the same point' if r['val_in'] >= r['gval_in']
                         else 'a DIFFERENT, stronger lobe'))
                if r['fit']:
                    print('      weighted centroid within 0.12 ms/m of the '
                          'prediction: (%+.3f,%+.3f)'
                          % (r['fit'][0], r['fit'][1]))

            # ---- tau0 sweep ---------------------------------------------
            # Evaluated ONLY on a 5 x 5 slope patch centred on the prediction,
            # not on the whole map.  That is both the cheaper calculation and
            # the sharper question: "at which lag anchor is the PREDICTED
            # event strongest?"  A max over the whole map answers a different
            # question, because for most tau0 the map maximum sits on some
            # other event entirely and the sweep then just tracks that.
            print('  tau0 sweep at each pivot, |M| on a 5x5 slope patch '
                  'centred on the PREDICTED slope:')
            taus0 = np.arange(-0.040, 0.0401, 0.004)
            for (nm, xp, zp, kind, _lg) in pivs:
                (ppx, ppz), _ = predict(kind, xp, zp)
                pxq = float(ppx) + 0.02 * np.arange(-2, 3)
                pzq = float(ppz) + 0.02 * np.arange(-2, 3)
                vals = []
                for t0 in taus0:
                    s = LS.slant2d_lag(res.tapered, grid, [[xp, zp]],
                                       hx=a.haper, hz=a.haper, window='hann',
                                       px=pxq, pz=pzq, tau0=float(t0),
                                       interp='cubic', fmax=36.0, verbose=0)
                    vals.append(float(np.abs(s.M[0]).max()))
                vals = np.array(vals)
                ib = int(np.argmax(vals))
                print('    %-16s best tau0 = %+6.1f ms; |M| at tau0 = 0 is '
                      '%.4f of it' % (nm, 1.0e3 * taus0[ib],
                                      vals[len(taus0) // 2]
                                      / max(vals[ib], 1e-300)))

            # ---- the slant figure ---------------------------------------
            fig, axs = plt.subplots(len(rows), 4,
                                    figsize=(16.0, 3.6 * len(rows)),
                                    squeeze=False)
            for ir, r in enumerate(rows):
                for ic, lab in enumerate(('in', 'pass')):
                    s = r['sres'][lab]
                    M = s.M[0]
                    c = np.percentile(np.abs(M), 99.8) or 1.0
                    ax = axs[ir][ic]
                    ax.imshow(M.T, origin='upper', aspect='auto',
                              cmap='seismic', vmin=-c, vmax=c,
                              extent=[s.px[0], s.px[-1], s.pz[-1], s.pz[0]])
                    ax.plot(r['pred'][0], r['pred'][1], 'k+', ms=14, mew=2)
                    ax.plot(*r['peak_%s' % ('in' if lab == 'in' else 'pass')],
                            marker='o', mfc='none', mec='lime', ms=12, mew=2)
                    if r['fit']:
                        ax.plot(r['fit'][0], r['fit'][1], 'kx', ms=10, mew=2)
                    for s_ in (a.s_pass, a.s_stop):
                        th = np.linspace(0, 2 * np.pi, 181)
                        ax.plot(s_ * np.cos(th), s_ * np.sin(th), 'k--', lw=0.8)
                    ax.set_xlabel('px [ms/m]')
                    ax.set_ylabel('pz [ms/m]')
                    ax.set_title('%s  %s\n%s' % (r['pivot'], lab, key),
                                 fontsize=9)
                # the two CIGs through the pivot, input then pass
                ip_x = grid.ix_of(r['x'])
                ip_z = grid.iz_of(r['z'])
                for ic, (lab, cube) in enumerate((('in', res.tapered),
                                                  ('pass', res.passed))):
                    ax = axs[ir][2 + ic]
                    half = int(a.haper / grid.dx)
                    j0, j1 = max(0, ip_x - half), min(nx, ip_x + half + 1)
                    cig = cube[:, j0:j1, ip_z]
                    c = np.percentile(np.abs(cig), 99.5) or 1.0
                    ax.imshow(cig, origin='upper', aspect='auto',
                              cmap='seismic', vmin=-c, vmax=c,
                              extent=[grid.x0 + grid.dx * j0,
                                      grid.x0 + grid.dx * (j1 - 1),
                                      1e3 * grid.taus[-1], 1e3 * grid.taus[0]])
                    xx = np.array([grid.x0 + grid.dx * j0,
                                   grid.x0 + grid.dx * (j1 - 1)])
                    ax.plot(xx, r['pred'][0] * (xx - r['x']), 'k-', lw=1.2)
                    ax.plot(xx, r['peak_in'][0] * (xx - r['x']), 'g--', lw=1.2)
                    ax.set_xlabel('x [m]')
                    ax.set_ylabel('tau [ms]')
                    ax.set_title('x CIG  %s  z = %.0f m' % (lab, r['z']),
                                 fontsize=9)
            fig.suptitle('%s   slant stacks, Hann %.0f m, tau0 = 0   '
                         '(+ prediction, o local peak, x centroid, dashed '
                         'circles s_pass / s_stop)'
                         % (key, a.haper), fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fn = os.path.join(out, 'slant_%s_%s.png' % (run, par))
            fig.savefig(fn, dpi=110)
            plt.close(fig)
            print('  wrote %s' % fn)

            for r in rows:
                r.pop('sres')
            rep['runs'][key] = dict(masks=emet, pivots=rows,
                                    filter=res.params)
            del res

            # ---- the cutoff sweep ---------------------------------------
            print('  cutoff sweep (energy fractions inside each mask):')
            g, _ = LS.read_lag_result('dense', par)
            sw = {}
            for mid in MIDPOINTS:
                sp, st = mid - HALFWIDTH, mid + HALFWIDTH
                rr = SF.slope_filter3d(g, grid.dx, grid.dz, grid.dtau,
                                       s_pass=sp, s_stop=st,
                                       taper=(0.10, 0.10, 0.10), pad=2,
                                       kmode='2d', verbose=0)
                Aq, Pq, Rq = rr.tapered[it0], rr.passed[it0], rr.rejected[it0]
                row = {}
                for nm in sorted(masks):
                    m = masks[nm]
                    ea = nrg(Aq, m)
                    row[nm] = (nrg(Pq, m) / ea, nrg(Rq, m) / ea)
                sw['%.2f' % mid] = row
                print('    midpoint %.2f (s_pass %.2f, s_stop %.2f): %s'
                      % (mid, sp, st,
                         '  '.join('%s pass %.3f' % (k.replace('iso ', ''),
                                                     v[0])
                                   for k, v in sorted(row.items()))))
                del rr
            del g
            sweep[key] = sw
            rep['runs'][key]['sweep'] = sw
        os.chdir(cwd)

    # -- figure: tau = 0 panels -------------------------------------------
    if tau0_panels:
        n = len(tau0_panels)
        fig, axs = plt.subplots(n, 3, figsize=(16.0, 3.3 * n), squeeze=False)
        for i, (key, A0, P0, R0, grid, nx, nz) in enumerate(tau0_panels):
            c = np.percentile(np.abs(A0), 99.5) or 1.0
            ext = [grid.x0, grid.x0 + grid.dx * (nx - 1),
                   grid.z0 + grid.dz * (nz - 1), grid.z0]
            for j, (lab, D) in enumerate((('tapered input', A0),
                                          ('pass  |p| <= %.2f ms/m'
                                           % a.s_pass, P0),
                                          ('reject  high |p|', R0))):
                ax = axs[i][j]
                ax.imshow(D.T, origin='upper', aspect='equal', cmap='seismic',
                          vmin=-c, vmax=c, extent=ext)
                ax.axhline(ZF, color='k', lw=0.7, ls=':')
                ax.plot([XS, XR], [ZS, ZR], 'k*', ms=9)
                # Only run B HAS anomalies; drawing their sites on a run A
                # panel would invite reading a run A feature as an anomaly
                # response when run A's true model is the bare reflector.
                if key.startswith('runB'):
                    ax.plot([a.xa, a.xb], [a.za, a.zb], 'ko', mfc='none', ms=8)
                ax.set_title('%s   %s' % (key, lab), fontsize=9)
                ax.set_xlabel('x [m]')
                ax.set_ylabel('z [m]')
        fig.suptitle('tau = 0 slice, one clip per row (99.5th percentile of '
                     'the tapered input); dotted line = reflector, '
                     'stars = S and R, circles (run B rows only) = anomalies',
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fn = os.path.join(out, 'tau0_panels.png')
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        print('')
        print('wrote %s' % fn)

    with open(os.path.join(out, 'report.json'), 'w') as fp:
        json.dump(rep, fp, indent=1, default=lambda o: str(o))
    print('wrote %s' % os.path.join(out, 'report.json'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
