#!/usr/bin/env python3
"""Mute the DIRECT part of the run B residual, without touching gradient code.

test_gradient_visual forms its own residual internally, as

    res = syn - obs

(verified by correlation on the existing case_fx_geo files: corr +1.000000,
amplitude ratio 1.000000 against syn - obs).  That one fact is enough to
apply an arbitrary mute from outside: form the residual, mute it, and hand
back a modified observed file

    obs' = syn - M r,        r = syn - obs

so that the next run sees  res' = syn - obs' = M r  EXACTLY.  Nothing is
approximated and no source file changes.

Two things make this safe rather than clever, and both are checked here:

  1  syn must be the SAME syn.  It is taken from a probe pass of
     test_gradient_visual itself (lag_nfreq=0, so it writes the synthetic and
     skips the lag accumulation), not from a separately-modelled background
     gather.  The two are compared anyway and the difference reported; if the
     probe syn and the independently modelled background disagree by more
     than round-off that is a finding, not a detail, so it is printed.

  2  the mute must not cut the event it is meant to keep.  The window is not
     taken on faith from the wavelet length: the onset of the reflected
     residual is MEASURED on the envelope, the taper's late edge is required
     to sit below it, and the scan then minimises the direct energy that
     survives subject to that.  The chosen times are printed.

The two onsets are measured on the EXACT SPLIT, not on the full residual:

    r        = syn - obs                       (the whole run B residual)
    r_direct = d_homo - d_hano                 (same anomalies, NO reflector)
    r_refl   = r - r_direct                    (exact, by construction)

That distinction is not cosmetic.  Measuring on the full residual puts the
"reflected onset" at the direct-S time, because the direct-S residual is a
separate, later burst (here 4.6% of the direct-P peak but 15.9% of the
reflected peak) sitting between the direct-P residual and the first
reflection.  A threshold crossing on the full envelope cannot tell that
burst from the event to be kept, so it pins the constraint 70 ms too early
and no window fits.  On the split there is no ambiguity.

Spans are reported at three envelope levels, because the gap between the
direct-S tail and the first reflection is a few tens of ms and a single
threshold would hide how marginal it is.  The hard constraint uses the 1%
level.  If the direct part still overlaps the reflected part there, this
script stops and reports the overlap instead of muting.

Called from test_lag_reflector.sh; see lag_refl_check.py for the same split
used as a standalone gate.
"""
from __future__ import print_function

import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Relative, not absolute -- see the same note in lag_refl_check.py.
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, 'tests',
                                'kh_misfit'))
import su_io                                                  # noqa: E402

sys.path.insert(0, HERE)
import lag_refl_rays as R                                     # noqa: E402
from lag_refl_check import cosine_mute, env, nrg              # noqa: E402

ONSET = 0.01          # envelope fraction that counts as the onset of an event
LEVELS = (0.05, 0.01, 0.003)


def span(e, t, frac):
    """First and last time the envelope `e` reaches `frac` of its own peak."""
    k = np.nonzero(e >= frac * e.max())[0]
    return (float(t[k[0]]), float(t[k[-1]])) if len(k) else (np.nan, np.nan)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--syn', required=True)
    p.add_argument('--bg', required=True)
    p.add_argument('--obs', required=True)
    p.add_argument('--homo', required=True,
                   help='prefix of the no-reflector, no-anomaly gather')
    p.add_argument('--hano', required=True,
                   help='prefix of the no-reflector, WITH-anomaly gather')
    p.add_argument('--out', required=True)
    p.add_argument('--comp', default='rvx,rvz,rtxx,rtzz')
    for k in ('xsrc', 'zsrc', 'xrcv', 'zrcv', 'zrefl', 'vp', 'vs',
              't0', 'fpeak'):
        p.add_argument('--' + k, type=float, required=True)
    a = p.parse_args(argv)
    comps = [c for c in a.comp.split(',') if c]

    tab = R.arrival_table(a.xsrc, a.zsrc, a.xrcv, a.zrcv, a.zrefl,
                          a.vp, a.vs, t0=a.t0)
    T = {d['name']: d['tpick'] for d in tab}

    # ---- read everything once ------------------------------------------
    syn, obs, bg, hdr = {}, {}, {}, {}
    dt = None
    for c in comps:
        syn[c], hdr[c] = su_io.read('%s_000_%s.su' % (a.syn, c))
        obs[c], _ = su_io.read('%s_000_%s.su' % (a.obs, c))
        bg[c], _ = su_io.read('%s_000_%s.su' % (a.bg, c))
        if dt is None:
            dt = struct.unpack_from('<h', hdr[c][0], 116)[0] * 1.0e-6
    nt = syn[comps[0]].shape[1]
    t = np.arange(nt) * dt

    # Only the GRADIENT components decide the window; rtxx/rtzz are carried
    # along so that obs' stays self-consistent for computeHydrophone.
    use = [c for c in comps if c in ('rvx', 'rvz')] or comps
    hom = np.concatenate([su_io.read('%s_000_%s.su' % (a.homo, c))[0]
                          for c in use], axis=0)
    han = np.concatenate([su_io.read('%s_000_%s.su' % (a.hano, c))[0]
                          for c in use], axis=0)

    # ---- check 1: is the probe syn the background gather? ---------------
    ds = sum(nrg(syn[c] - bg[c]) for c in comps)
    nb = sum(nrg(bg[c]) for c in comps)
    print('    probe syn vs the independently modelled background: '
          '||syn-bg||^2/||bg||^2 = %.3e' % (ds / nb))

    # ---- the residual and its EXACT split ------------------------------
    r = np.concatenate([syn[c] - obs[c] for c in use], axis=0)
    r_dir = hom - han
    r_ref = r - r_dir
    print('    exact split: ||r||^2 = %.4e, of which direct %.4f and '
          'reflected %.4f (identity residue %.3e)'
          % (nrg(r), nrg(r_dir) / nrg(r), nrg(r_ref) / nrg(r),
             nrg(r - r_dir - r_ref) / nrg(r)))

    e_dir, e_ref = env(r_dir).max(axis=0), env(r_ref).max(axis=0)
    print('    envelope spans of the two parts, each at a fraction of ITS '
          'OWN peak:')
    print('    %6s %18s %18s %9s' % ('level', 'direct', 'reflected', 'gap'))
    t_dir = t_refl = None
    for fr in LEVELS:
        d0, d1 = span(e_dir, t, fr)
        r0, r1 = span(e_ref, t, fr)
        print('    %5.1f%% %8.4f..%-8.4f %8.4f..%-8.4f %+8.1f ms'
              % (100 * fr, d0, d1, r0, r1, 1.0e3 * (r0 - d1)))
        if fr == ONSET:
            t_dir, t_refl = d1, r0
    print('    direct P peak %.3f s, direct S peak %.3f s, PP peak %.3f s, '
          'SS peak %.3f s'
          % (T['direct P'], T['direct S'], T['PP'], T['SS']))
    print('    the direct residual carries %.4f of its energy after %.2f s: '
          'that late burst is the DIRECT S residual, %.4f of the direct-P '
          'peak but %.4f of the reflected peak -- which is why the window '
          'has to be measured on the split and not on ||r||.'
          % (nrg(r_dir[:, int(round(1.70 / dt)):]) / nrg(r_dir), 1.70,
             e_dir[int(round(1.70 / dt)):].max() / e_dir.max(),
             e_dir[int(round(1.70 / dt)):].max() / e_ref.max()))

    # ---- the gate the user asked for -----------------------------------
    if t_dir >= t_refl:
        print('    GATE FAILED: at the %.0f%% level the direct residual runs '
              'to %.4f s but the reflected residual already begins at '
              '%.4f s -- an overlap of %.1f ms.  Stopping without muting, as '
              'specified: no smooth mute can remove the direct arrivals '
              'without cutting the reflection.'
              % (100 * ONSET, t_dir, t_refl, 1.0e3 * (t_dir - t_refl)))
        return 1

    # ---- choose the window ---------------------------------------------
    # Hard constraint: the taper's LATE edge stays below the measured onset
    # of the reflected residual, so the mute cannot touch it at all.  Within
    # that, minimise the DIRECT energy that survives -- not the total, which
    # would be minimised by muting everything.
    best = None
    for w in (0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20):
        for t_end in np.arange(T['direct P'], t_refl + 1e-9, 0.002):
            if t_end + 0.5 * w > t_refl:
                continue
            m = cosine_mute(t, t_end, w)[None, :]
            cand = (nrg(r_dir * m) / nrg(r_dir), -w, float(t_end))
            if best is None or cand < best:
                best = cand
    if best is None:
        raise SystemExit('    no window fits below %.4f s' % t_refl)
    leak, negw, t_end = best
    w = -negw
    m = cosine_mute(t, t_end, w)
    print('    MUTE: cosine rising through t_end = %.3f s over w = %.0f ms, '
          'i.e. 0 below %.3f s and 1 above %.3f s'
          % (t_end, 1.0e3 * w, t_end - 0.5 * w, t_end + 0.5 * w))
    print('    the taper is %.0f ms against a %.0f ms wavelet period, forced '
          'short by the %.0f ms gap; the resulting edge sits where both parts '
          'are near the noise floor, so:'
          % (1.0e3 * w, 1.0e3 / a.fpeak, 1.0e3 * (t_refl - t_dir)))
    print('    direct energy SURVIVING the mute   %.4e of ||r_direct||^2'
          % leak)
    print('    reflected energy LOST to the mute  %.4e of ||r_refl||^2'
          % (1.0 - nrg(r_ref * m[None, :]) / nrg(r_ref)))

    # ---- write obs' -----------------------------------------------------
    for c in comps:
        rc = syn[c] - obs[c]
        out = syn[c] - rc * m[None, :]
        su_io.write('%s_000_%s.su' % (a.out, c), out.astype(np.float32),
                    hdr[c])
    print('    wrote %s_000_{%s}.su' % (a.out, ','.join(comps)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
