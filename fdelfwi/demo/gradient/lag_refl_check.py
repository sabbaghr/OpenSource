#!/usr/bin/env python3
"""Modelling QC for test_lag_reflector.sh, run BEFORE any gradient.

Four things are checked, and the third one can stop the experiment.

  1  ARRIVALS.  Ray-traced times against the measured gather.  The picks are
     local maxima of the analytic-signal envelope, matched to the nearest
     prediction, so a pick is reported with its ERROR and not merely as a
     tick on a plot.  Directs are picked on the homogeneous gather, where
     they are alone; reflections on d_refl - d_homo, where the directs have
     cancelled exactly.

  2  BOUNDARIES.  The absorbing pad is not perfect, so every edge that could
     mirror an arrival into the record gets a predicted time and a measured
     amplitude there, as a fraction of the direct-P peak.  The TOP edge is
     the dangerous one at this geometry: its P mirror lands near 1.28 s,
     between direct P and direct S.

  3  THE DIRECT-S / PP OVERLAP GATE.  Run B needs the direct arrivals muted
     out of its residual, and direct S lands only ~134 ms before PP while the
     wavelet envelope is ~132 ms wide.  Whether a mute can do that cleanly is
     not a question about the wavelet -- it is a question about the residual,
     so it is answered on the residual.

     The four forward runs give an EXACT split.  The direct ray never touches
     the reflector, so the direct arrival is identical in the reflector model
     and in the homogeneous one and

         r_B      = d_refl - d_anom          what the run B gradient sees
         r_direct = d_homo - d_hano          directs only
         r_refl   = r_B - r_direct           reflections only

     with no windowing anywhere in the identity (the printed round-off
     confirms it).  A cosine mute with midpoint t_end and full taper length
     w is then scored on the two numbers that matter,

         leak = energy of r_direct that SURVIVES the mute
         loss = energy of r_refl   that is DESTROYED by it

     scanned over (t_end, w).  If nothing reaches leak < 1% at loss < 5%, the
     geometry does not support the run B design: the script says so, prints
     the least-bad settings for the record, and exits non-zero rather than
     proceeding with a mute that eats into PP.

  4  RECORD LENGTH AND df.  TMOD against the last arrival, and the two
     aliasing conditions the lag reconstruction imposes.

Nothing here is plotted: these are the numbers the later stages are allowed
to assume, so they are printed where they can be read.
"""
from __future__ import print_function

import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# su_io is located RELATIVE TO THIS FILE, not by absolute path: the same
# checkout must work from the MIO tree, and an absolute Seislab10 path would
# make a MIO run silently import from the other machine's tree -- which is
# precisely what a cross-machine reproducibility check must not do.
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, 'tests',
                                'kh_misfit'))
import su_io                                                  # noqa: E402

sys.path.insert(0, HERE)
import lag_refl_rays as R                                     # noqa: E402

COMPS = ('rvx', 'rvz')
LEAK_MAX = 0.01
LOSS_MAX = 0.05


# ---------------------------------------------------------------------------
def env(a):
    """|analytic signal| along the last axis.

    Built in the frequency domain -- negative frequencies zeroed, positives
    doubled -- and inverse transformed as a COMPLEX series, which is the
    definition of the analytic signal and does not depend on any real-FFT
    Nyquist bookkeeping.
    """
    n = a.shape[-1]
    A = np.fft.fft(a, axis=-1)
    k = np.fft.fftfreq(n)
    m = np.ones(n)
    m[k < 0.0] = 0.0
    m[k > 0.0] = 2.0
    return np.abs(np.fft.ifft(A * m, axis=-1))


def local_maxima(e, t, tmin=0.0):
    i = np.nonzero((e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]))[0] + 1
    return i[t[i] >= tmin]


def read_gather(base, comps=COMPS):
    """The components of one shot stacked into (ntrace*ncomp, nt)."""
    out, dt = [], None
    for c in comps:
        d, h = su_io.read('%s_000_%s.su' % (base, c))
        out.append(d)
        if dt is None:
            dt = struct.unpack_from('<h', h[0], 116)[0] * 1.0e-6
    d = np.concatenate(out, axis=0)
    return d, np.arange(d.shape[1]) * dt, dt


def nrg(a):
    return float(np.sum(a.astype(np.float64) ** 2))


def cosine_mute(t, t_end, w):
    """0 below t_end - w/2, 1 above t_end + w/2, raised cosine between.

    A mute here KEEPS THE LATE PART: the direct arrivals are early, so the
    window rises with time.  t_end is the midpoint, i.e. the time at which
    half the amplitude is kept, and that is the time quoted as "the mute
    time".
    """
    m = np.zeros_like(t)
    a, b = t_end - 0.5 * w, t_end + 0.5 * w
    m[t >= b] = 1.0
    k = (t > a) & (t < b)
    m[k] = 0.5 * (1.0 - np.cos(np.pi * (t[k] - a) / w))
    return m


def scan_mute(t, r_direct, r_refl, t_lo, t_hi):
    ed, er = nrg(r_direct), nrg(r_refl)
    rows = []
    for w in (0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20):
        for t_end in np.arange(t_lo, t_hi + 1e-9, 0.005):
            m = cosine_mute(t, t_end, w)[None, :]
            rows.append((float(t_end), float(w),
                         nrg(r_direct * m) / ed,
                         1.0 - nrg(r_refl * m) / er))
    return rows


def span(e, t, frac=0.05):
    ee = e.max(axis=0)
    k = np.nonzero(ee >= frac * ee.max())[0]
    return t[k[0]], t[k[-1]]


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    for k in ('dx', 'sizex', 'sizez', 'xsrc', 'zsrc', 'xrcv', 'zrcv',
              'zrefl', 'vp', 'vs', 'vfac', 't0', 'fpeak', 'tmod', 'ntaper',
              'dfreq'):
        p.add_argument('--' + k, type=float, required=True)
    a = p.parse_args(argv)

    XS, ZS, XR, ZR = a.xsrc, a.zsrc, a.xrcv, a.zrcv
    VP, VS = a.vp, a.vs

    # -- 1  arrivals ------------------------------------------------------
    tab = R.arrival_table(XS, ZS, XR, ZR, a.zrefl, VP, VS, t0=a.t0)
    print('  predicted arrivals (ray theory, flat reflector)')
    print('  %-10s %9s %9s %10s %9s' % ('event', 't [s]', 'peak [s]',
                                        'bounce x', 'inc [deg]'))
    for d in tab:
        inc = ('-' if d['xc'] is None else
               '%.2f' % np.degrees(np.arctan2(abs(d['xc'] - XS),
                                              a.zrefl - ZS)))
        print('  %-10s %9.4f %9.4f %10s %9s'
              % (d['name'], d['t'], d['tpick'],
                 '-' if d['xc'] is None else '%.1f' % d['xc'], inc))

    d_refl, t, dt = read_gather('d_refl')
    d_homo, _, _ = read_gather('d_homo')
    d_anom, _, _ = read_gather('d_anom')
    d_hano, _, _ = read_gather('d_hano')

    m_refl = d_refl - d_homo          # the reflected wavefield, exactly
    e_refl, e_homo, e_all = env(m_refl), env(d_homo), env(d_refl)
    pk = float(e_all.max())
    tr_a = int(np.argmax(e_all.max(axis=1)))
    tr_r = int(np.argmax(e_refl.max(axis=1)))
    tr_h = int(np.argmax(e_homo.max(axis=1)))

    print('')
    print('  measured picks, envelope maxima of the %s traces'
          % '+'.join(COMPS))
    print('  %-10s %9s %9s %9s %9s %12s' % ('event', 'pred [s]', 'meas [s]',
                                            'err [ms]', 'amp/maxP', 'picked on'))
    for d in tab:
        direct = d['xc'] is None
        e = e_homo[tr_h] if direct else e_refl[tr_r]
        src = 'd_homo' if direct else 'd_refl-d_homo'
        i = local_maxima(e, t, tmin=0.3)
        if not len(i):
            print('  %-10s %9.4f   no envelope maximum' % (d['name'],
                                                           d['tpick']))
            continue
        j = i[int(np.argmin(np.abs(t[i] - d['tpick'])))]
        print('  %-10s %9.4f %9.4f %9.1f %9.4f %12s'
              % (d['name'], d['tpick'], t[j], 1.0e3 * (t[j] - d['tpick']),
                 e[j] / pk, src))

    # -- 2  boundaries ----------------------------------------------------
    # The pad EXTENDS the grid, so an edge mirror sits at the model edge
    # displaced outwards by the pad thickness.
    pad = a.ntaper * a.dx
    edges = (('top    z = %8.0f' % (-pad), XS, -2.0 * pad - ZS),
             ('left   x = %8.0f' % (-pad), -2.0 * pad - XS, ZS),
             ('right  x = %8.0f' % (a.sizex + pad),
              2.0 * (a.sizex + pad) - XS, ZS),
             ('bottom z = %8.0f' % (a.sizez + pad), XS,
              2.0 * (a.sizez + pad) - ZS))
    print('')
    print('  boundary check: absorbing pad %.0f m (%d points) on all four '
          'sides, no free surface' % (pad, int(a.ntaper)))
    print('  %-22s %-5s %9s %10s' % ('mirror of S', 'mode', 't [s]',
                                     'amp/maxP'))
    for nm, xm, zm in edges:
        for mode, v in (('P', VP), ('S', VS)):
            tb = np.hypot(XR - xm, ZR - zm) / v + a.t0
            if tb > t[-1]:
                print('  %-22s %-5s %9.4f   after the %.3f s record'
                      % (nm, mode, tb, t[-1]))
                continue
            j = int(round(tb / dt))
            amp = float(e_all[:, max(0, j - 20):j + 21].max()) / pk
            print('  %-22s %-5s %9.4f %10.4f' % (nm, mode, tb, amp))
    # reflector then top edge: mirror S about the reflector, R about the top
    tb = (np.hypot(XR - XS, (2.0 * a.zrefl - ZS) - (-2.0 * pad - ZR)) / VP
          + a.t0)
    if tb <= t[-1]:
        j = int(round(tb / dt))
        print('  %-22s %-5s %9.4f %10.4f'
              % ('reflector + top', 'P', tb,
                 float(e_all[:, max(0, j - 20):j + 21].max()) / pk))
    else:
        print('  %-22s %-5s %9.4f   after the %.3f s record'
              % ('reflector + top', 'P', tb, t[-1]))

    # -- run A residual content -------------------------------------------
    t_s = [d for d in tab if d['name'] == 'direct S'][0]['tpick']
    t_pp = [d for d in tab if d['name'] == 'PP'][0]['tpick']
    i_s = int(round(t_s / dt))
    ee = e_refl[tr_r]
    first = t[int(np.argmax(ee > 0.02 * ee.max()))]
    print('')
    print('  run A residual = syn(homo) - obs(refl), so the directs cancel '
          'by construction, not by muting:')
    print('    energy before the direct-S peak (%.3f s): %.3e of the '
          'residual total, and %.3e of d_refl over that same window'
          % (t_s, nrg(m_refl[:, :i_s]) / nrg(m_refl),
             nrg(m_refl[:, :i_s]) / nrg(d_refl[:, :i_s])))
    print('    reflected-only trace first breaks at %.4f s (2%% of its own '
          'envelope peak) against the PP prediction %.4f s' % (first, t_pp))

    # -- 3  the gate ------------------------------------------------------
    r_B = d_refl - d_anom
    r_direct = d_homo - d_hano
    r_refl = r_B - r_direct
    print('')
    print('  ===== direct-S / PP overlap gate (run B) =====')
    print('  exact split of the run B residual:')
    print('    ||r_B||^2       %.6e' % nrg(r_B))
    print('    ||r_direct||^2  %.6e   %5.1f%%'
          % (nrg(r_direct), 100.0 * nrg(r_direct) / nrg(r_B)))
    print('    ||r_refl||^2    %.6e   %5.1f%%'
          % (nrg(r_refl), 100.0 * nrg(r_refl) / nrg(r_B)))
    print('    identity check  %.3e   (r_B - r_direct - r_refl, so this is '
          'round-off)' % (nrg(r_B - r_direct - r_refl) / nrg(r_B)))

    er_d, er_r = env(r_direct), env(r_refl)
    lo_d, hi_d = span(er_d, t)
    lo_r, hi_r = span(er_r, t)
    print('  envelope overlap, measured on the residual itself:')
    print('    r_direct at the PP peak      (%.3f s) / its own max  %.4f'
          % (t_pp, float(er_d[:, int(round(t_pp / dt))].max() / er_d.max())))
    print('    r_refl   at the direct-S peak (%.3f s) / its own max  %.4f'
          % (t_s, float(er_r[:, int(round(t_s / dt))].max() / er_r.max())))
    print('    r_direct spans %.4f - %.4f s at the 5%% envelope level'
          % (lo_d, hi_d))
    print('    r_refl   spans %.4f - %.4f s' % (lo_r, hi_r))
    print('    overlap        %.1f ms' % (1.0e3 * max(0.0, hi_d - lo_r)))

    rows = scan_mute(t, r_direct, r_refl, t_s - 0.05, t_pp + 0.10)
    ok = [r for r in rows if r[2] < LEAK_MAX and r[3] < LOSS_MAX]
    print('  mute scan: cosine window rising through t_end over a full '
          'length w')
    print('  %9s %7s %12s %12s' % ('t_end [s]', 'w [ms]', 'leak(direct)',
                                   'loss(refl)'))
    if ok:
        ok.sort(key=lambda r: (r[3], r[2]))
        for r in ok[:6]:
            print('  %9.3f %7.0f %12.2e %12.2e'
                  % (r[0], 1.0e3 * r[1], r[2], r[3]))
        b = ok[0]
        print('  GATE PASSED: leak %.2e < %.0e and loss %.2e < %.0e at '
              't_end = %.3f s, w = %.0f ms'
              % (b[2], LEAK_MAX, b[3], LOSS_MAX, b[0], 1.0e3 * b[1]))
    else:
        rows.sort(key=lambda r: r[2] / LEAK_MAX + r[3] / LOSS_MAX)
        print('  (nothing met the bar; the six least-bad, for the record)')
        for r in rows[:6]:
            print('  %9.3f %7.0f %12.2e %12.2e'
                  % (r[0], 1.0e3 * r[1], r[2], r[3]))

    # -- 4  record length and df ------------------------------------------
    last = max(d['tpick'] for d in tab)
    print('')
    print('  record length and df')
    print('    TMOD %.3f s, last predicted peak %.4f s -> %.3f s of coda'
          % (a.tmod, last, a.tmod - last))
    print('    df = %.4f Hz,  df <= 1/TMOD = %.4f Hz   %s'
          % (a.dfreq, 1.0 / a.tmod,
             'OK' if a.dfreq <= 1.0 / a.tmod else 'ALIASES'))
    print('    lag aliasing |tau| < 1/(4 df) = %.3f s against the 0.100 s '
          'asked for   %s'
          % (0.25 / a.dfreq, 'OK' if 0.25 / a.dfreq > 0.1 else 'ALIASES'))
    print('    dtrcv %.4f s -> Nyquist %.1f Hz against the %.1f Hz measured '
          'wavelet edge' % (dt, 0.5 / dt, 1.22 * 25 + 1.46))

    if not ok:
        print('')
        print('  GATE FAILED: no smooth mute separates the direct residual '
              'from the reflected one at this geometry.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
