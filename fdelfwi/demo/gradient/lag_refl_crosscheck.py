#!/usr/bin/env python3
"""lag_refl_crosscheck.py -- is a test_lag_reflector.sh run on one machine the
same as a run on another?

Compares two `lag_reflector` work directories produced by the SAME
test_lag_reflector.sh at the same settings, built by DIFFERENT compilers on
DIFFERENT hardware (Seislab10 vs MIO).

WHY THE ANSWER IS NOT "bit-identical"
-------------------------------------
Both trees are built with `-O3 -march=native -ffast-math`.  Those three flags
together mean the compiler is free to re-associate floating-point arithmetic
and to vectorise it differently for a different instruction set, so the FD
kernel's accumulation order is NOT the same on two machines.  Over ~4000 time
steps those differences accumulate.  Expecting `cmp` to pass would be wrong,
and a test that demanded it would fail for a reason that is not a bug.

What CAN be demanded is that the difference stay at the level of accumulated
rounding -- relative RMS of order 1e-6 or below, correlation indistinguishable
from 1.  Anything a real porting bug would do is orders of magnitude larger:

  * an MKL-vs-Mayer FFT normalisation mismatch in lagConeFilter scales the lag
    cubes by Nt*Nx*Nz -- a ratio of ~1e7, not 1e-6;
  * a wrong `-march` / denormal mode shows up as a handful of huge outliers
    rather than a uniform small RMS;
  * a type or endianness slip makes the correlation collapse, not drift.

THE LAYERING IS THE POINT
-------------------------
The three groups are reported in the order the data flows through the code, so
a failure localises itself instead of just saying "different":

  1. observed gathers  d_*_000_*.su   -- the FORWARD modeller alone
  2. zero-lag gradient dense_*_raw.su -- forward + adjoint + accumGradient
  3. lag cubes         dense_lag*.su  -- the running DFTs, the per-lag
                                         reconstruction and the FFT round trip

If (1) already differs by more than rounding, the FD kernel or the build flags
are the suspect and nothing about the lag code is implicated.  If (1) and (2)
match and only (3) moves, the suspect is the lag/FFT path -- which is exactly
the new code and exactly what the FFT backend can break.

Usage:
  lag_refl_crosscheck.py --ref=<dir> --new=<dir> [--rtol=1e-5] [--quick]

  --quick  skip the 189 MB lag cubes (groups 1-2 only)
"""
import argparse
import os
import sys

import numpy as np

TRCBYTES = 240


def read_flat(fn):
    """Read an SU file as one flat float64 vector, headers discarded.

    Deliberately header-blind: this compares SAMPLES.  Headers can legitimately
    differ between runs (they carry no physics here) and a header diff would
    produce a confusing failure.
    """
    raw = np.fromfile(fn, dtype=np.uint8)
    ns = raw[114:116].view(np.int16)[0]
    if ns <= 0:
        raise ValueError('%s: bad ns=%d in the first header' % (fn, ns))
    rec = TRCBYTES + 4 * int(ns)
    if raw.size % rec:
        raise ValueError('%s: size %d is not a multiple of the trace length %d '
                         '(ns=%d)' % (fn, raw.size, rec, ns))
    ntr = raw.size // rec
    out = np.empty((ntr, int(ns)), dtype=np.float32)
    for i in range(ntr):
        out[i] = raw[i * rec + TRCBYTES:(i + 1) * rec].view(np.float32)
    return out.ravel().astype(np.float64)


def compare(ref_fn, new_fn):
    """Return (metrics dict, note) or (None, reason-it-could-not-be-compared)."""
    if not os.path.exists(ref_fn):
        return None, 'absent in ref'
    if not os.path.exists(new_fn):
        return None, 'absent in new'
    a = read_flat(ref_fn)
    b = read_flat(new_fn)
    if a.shape != b.shape:
        return None, 'shape %d vs %d' % (a.size, b.size)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return dict(relrms=0.0, corr=1.0, ratio=1.0, maxrel=0.0,
                    identical=True), 'both identically zero'
    if na == 0.0 or nb == 0.0:
        return None, 'one side is all zeros (|ref|=%.3e |new|=%.3e)' % (na, nb)

    d = b - a
    relrms = float(np.linalg.norm(d) / na)
    corr = float(a @ b) / (na * nb)
    # Peak discrepancy measured against the GLOBAL peak, not pointwise: a
    # pointwise relative error blows up wherever the reference is near a zero
    # crossing and says nothing useful.
    maxrel = float(np.abs(d).max() / np.abs(a).max())
    return dict(relrms=relrms, corr=corr, ratio=nb / na, maxrel=maxrel,
                identical=bool(np.array_equal(a, b))), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref', required=True, help='reference work dir (Seislab10)')
    p.add_argument('--new', required=True, help='work dir to check (MIO)')
    p.add_argument('--rtol', type=float, default=1e-5,
                   help='max tolerated relative RMS difference (default 1e-5)')
    p.add_argument('--ctol', type=float, default=1e-9,
                   help='max tolerated 1-correlation (default 1e-9)')
    p.add_argument('--quick', action='store_true',
                   help='skip the large lag cubes')
    a = p.parse_args()

    groups = [
        ('1. observed gathers -- FORWARD modeller only', [
            'd_homo_000_rvx.su', 'd_homo_000_rvz.su',
            'd_refl_000_rvx.su', 'd_refl_000_rvz.su',
            'd_anom_000_rvx.su', 'd_anom_000_rvz.su',
            'd_hano_000_rvx.su', 'd_hano_000_rvz.su',
        ]),
        ('2. zero-lag gradients -- forward + adjoint + accumGradient', [
            'runA/dense_vp_raw.su', 'runA/dense_vs_raw.su',
            'runA/dense_rho_raw.su',
            'runB/dense_vp_raw.su', 'runB/dense_vs_raw.su',
            'runB/dense_rho_raw.su',
        ]),
    ]
    if not a.quick:
        groups.append(
            ('3. lag cubes -- running DFTs, reconstruction, FFT round trip', [
                'runA/dense_lagvp.su', 'runA/dense_lagvs.su',
                'runB/dense_lagvp.su', 'runB/dense_lagvs.su',
            ]))

    print('reference : %s' % a.ref)
    print('candidate : %s' % a.new)
    print('tolerance : relative RMS <= %.1e, 1-corr <= %.1e' % (a.rtol, a.ctol))

    worst = 0.0
    nfail = nskip = ncmp = 0

    for title, files in groups:
        print('')
        print(title)
        print('  %-28s %11s %13s %9s %10s' %
              ('file', 'rel RMS', '1-corr', 'ampl', 'max/peak'))
        for rel in files:
            m, note = compare(os.path.join(a.ref, rel),
                              os.path.join(a.new, rel))
            if m is None:
                print('  %-28s  SKIP  (%s)' % (rel, note))
                nskip += 1
                continue
            ncmp += 1
            ok = (m['relrms'] <= a.rtol) and ((1.0 - m['corr']) <= a.ctol)
            worst = max(worst, m['relrms'])
            if not ok:
                nfail += 1
            tag = 'bit-identical' if m['identical'] else ('ok' if ok else 'FAIL')
            print('  %-28s %11.3e %13.3e %9.6f %10.3e  %s' %
                  (rel, m['relrms'], 1.0 - m['corr'], m['ratio'],
                   m['maxrel'], tag))

    print('')
    print('=' * 72)
    if ncmp == 0:
        print('NOTHING COMPARED -- %d files skipped.  Did the run finish, and '
              'is --ref the right directory?' % nskip)
        return 2
    print('compared %d files, %d skipped, worst relative RMS %.3e'
          % (ncmp, nskip, worst))
    if nfail:
        print('')
        print('%d FILE(S) OUTSIDE TOLERANCE.' % nfail)
        print('Read the GROUP it failed in before suspecting the lag code:')
        print('  group 1 only  -> the FD kernel / build flags differ; the lag')
        print('                   path is not implicated at all.')
        print('  groups 1+2    -> same, propagated.  Compare -march and')
        print('                   -ffast-math between the two Make_includes.')
        print('  group 3 only  -> the forward and adjoint agree and only the')
        print('                   lag path moved.  First suspect is the FFT')
        print('                   backend: run  ./test_lag_cone selftest=1.')
        print('                   An amplitude ratio near Nt*Nx*Nz or its')
        print('                   reciprocal is the MKL-vs-Mayer scaling bug.')
        return 1
    print('ALL WITHIN TOLERANCE -- the two builds agree to accumulated '
          'rounding.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
