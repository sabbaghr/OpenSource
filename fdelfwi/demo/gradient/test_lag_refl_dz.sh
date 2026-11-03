#!/bin/bash
# test_lag_refl_dz.sh -- how robust is the lag-slope rejection of non-
#                        tomographic kernels when the residual is NOT cleaned?
#
# This is the stress version of test_lag_reflector.sh run B.  Three things are
# made harder on purpose, and nothing is made easier:
#
#   1. the reflector is in the WRONG PLACE in the background model (150 m too
#      shallow), so the dominant residual is a MIS-POSITIONED reflection, not a
#      retimed one.  In run B the reflector was identical in both models.
#   2. two smooth anomalies sit at DIFFERENT places in the true model -- a SLOW
#      Vp anomaly on the P down-leg and a FAST Vs anomaly on the up-leg -- so
#      the tomographic signal and the migration signal are superimposed in the
#      same cube and are opposite in sign.
#   3. the residual is the PLAIN DIFFERENCE syn - obs.  No direct-wave mute, no
#      res_taper, no aper_taper, no data_weight, no misfit reweighting.  The
#      direct P and the direct S residual stay in and are imaged.
#
# WHAT IS BEING MEASURED
# ----------------------
# Only one thing: what the lag-slope cone filter does to that cube.  Pipeline
# is exactly cube -> 2D slant stack (the 3D FK cone) -> keep the tomographic
# angles.  No arrival table, no event catalogue, no isochron or banana masks,
# no predicted-event input of any kind feeds the filter or the figure.  The
# filter sees the cube and one angle pair.
#
# WHAT THIS CANNOT SHOW
# ---------------------
# Nothing about inversion quality: no inversion is run.  And the reject panel
# is NOT a migration panel -- test_lag_reflector.sh already measured that mode
# crosstalk lands at the same high slopes as true specular isochrons
# (crosstalk 1.7x STRONGER than the matched kernel there), so "rejected" means
# "high |p|" and nothing more.
#
# THE CUTOFF IS AN ANGLE, NOT A SLOPE
# -----------------------------------
# The user asked to keep "tomographic angles", so the cutoff is specified as
# the full opening angle 2theta and converted PER MODE through
#
#     |p| = cos(theta)/v        2theta = opening angle
#
# which is the same-mode reduction of p = -0.5 [n_a/v_a + n_b/v_b].  That is
# the only way both modes get the SAME angle; a single slope does not --
# 0.20 ms/m is 132.8 deg for Vp = 2000 but 152.2 deg for Vs = 1200.
#
#     2theta         Vp = 2000        Vs = 1200
#     150 deg (pass) 0.129 ms/m       0.216 ms/m
#     130 deg (stop) 0.211 ms/m       0.352 ms/m
#
# The conversion uses the BACKGROUND velocity of the upper layer.  Below the
# reflector the model is 10% faster, so the effective cutoff there drifts from
# 150 to about 146 deg.  That is reported by the analysis stage, not hidden.
#
# STAGES
#   ./test_lag_refl_dz.sh model   models, wavelet, observed data, kinematics
#   ./test_lag_refl_dz.sh grad    the one lag gradient
#   ./test_lag_refl_dz.sh figs    QC figures: models, residual, raw lag cube
#   ./test_lag_refl_dz.sh anal    the cone filter and its figure
#   ./test_lag_refl_dz.sh all     all four
#
# Re-running `model` rebuilds the work directory from scratch; the later
# stages do not.

set -e
set -o pipefail
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}
export OMP_NUM_THREADS=16

HERE=$(cd "$(dirname "$0")" && pwd)
# lag_refl_dz is the FIRST version of this run and it is kept as a record of a
# design error, not as a result: its anomalies sat 1.50 sigma below the direct
# ray with sigma = 300, so the Gaussian was still at 0.884 of its peak at the
# edge of the direct-P first Fresnel zone.  The direct wave therefore did NOT
# cancel between the two models -- measured shift 5.06 ms, ray theory 6.10 ms
# -- and the direct-P residual came out 26x the PP residual.  The model was
# migration-dominated while the data was transmission-dominated, which is the
# opposite of what the test needs.  See the geometry block below for the fix.
#
# lag_refl_dz2 is version 2: the same test with the instruments lifted to
# 402 m and the anomalies dropped to 1380 m, which fixed the Fresnel leak
# (measured direct-P shift +0.000 ms, direct-P/PP residual 0.009 against
# 26.1 in version 1).  It is a ONE RECEIVER run and is kept as the reference
# the receiver-line run below is compared against.
#
# lag_refl_dz3 is version 3: one shot at x = 1500 and a LINE of 501 receivers
# from x = 1002 to 4002 in a 5004 m domain -- the shot sits at one end of the
# line, so the spread is one sided and the widest offset is 2502 m.
# Measured: 501/501 traces written, worst direct-P shift over the whole line
# -0.000 ms, Fresnel edge 3.21 sigma at the widest offset, misfit 5.561e-19.
#
# lag_refl_dz4 is this version: the SAME one shot, now in the MIDDLE of the
# line, in a domain widened to 6600 m so that centring it buys angle instead
# of costing it.  See the SIZEX comment for the four-way measurement that
# picked 6600 m.  1001 receivers, offsets +/- 3000 m, 2theta 66.8 -> 76.7 deg.
TAG=${TAG:-lag_refl_dz4}
WORK=${HERE}/${TAG}

STAGE=${1:-all}
case ${STAGE} in
    model|grad|figs|anal|all) ;;
    *) echo "usage: $0 [model|grad|figs|anal|all]" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Settings.  Everything is taken UNCHANGED from test_lag_reflector.sh so the
# two cubes are directly comparable; the exceptions are marked CHANGED.
# ---------------------------------------------------------------------------
# CHANGED from SIZEX = 5004.  The domain is widened ONLY because the shot is
# now in the middle of the line.  Centring the shot in a fixed domain HALVES
# the maximum offset even though it nearly doubles the line length, and the
# maximum offset -- not the line length -- is what sets the opening angle the
# reflector is illuminated at.  Measured, at z_refl - z_src = 1896 m:
#     5004 m domain, shot at 1500, line 1002-4002   max |off| 2502   2th 66.8
#     5004 m domain, shot centred, line 300-4704    max |off| 2202   2th 60.3
#     6600 m domain, shot centred, line 300-6300    max |off| 3000   2th 76.7
#     7500 m domain, shot centred, line 300-7200    max |off| 3450   2th 84.6
# The last one FAILS the Fresnel gate below (2.92 sigma against a 3.0 gate),
# so 6600 m is the widest aperture this test can legally have.
DX=6; SIZEX=6600; SIZEZ=3000
VP0=2000; VS0=1200; RO0=2000

VFAC=1.10                     # reflector, +10% Vp and Vs below it, rho equal
ZREFL_TR=2298                 # 383*DX -- the TRUE depth
# CHANGED: the background reflector is 150 m too SHALLOW.  2148 = 358*DX, so
# it is a grid node and the interface is again one flat row.
ZREFL_BG=2148

# CHANGED from ZSRC = ZRCV = 798.  The anomalies must clear the direct-P
# first Fresnel zone, which needs them DEEP, while the two legs converge on
# the specular point, which needs them SHALLOW.  With
#
#     u = (z_a - z_i)/(z_refl - z_i)        r1 = sqrt(lam L)/2 = 301 m
#     clearance  u d >= r1 + 3 sigma        separation  2004 (1-u) >= 4 sigma
#
# both hold only if the reflector is far enough below the instruments,
#
#     d = z_refl - z_i >= (r1 + 3 sigma)/(1 - 4 sigma/2004) = 1499 m at
#     sigma = 200.
#
# The old d = 2148 - 798 = 1350 m FAILS that, which is why version 1 of this
# run had a direct-wave residual.  Dropping the instruments to 402 m gives
# d = 1746 m, which passes with margin.  ntaper extends the grid rather than
# eating into it (getParameters.c:529, mod->naz += bnd->npml), so 402 m is
# physical model, not taper.
# CHANGED from XSRC = 1500.  The shot sits in the MIDDLE of the receiver line
# and in the middle of the domain, so the spread is symmetric: every offset is
# recorded on both sides and the two anomaly sites see mirror-image geometry.
XSRC=3300; ZSRC=402

# A LINE of receivers at the instrument depth, symmetric about the shot.
# Two things set the ends:
#
#   Fresnel.  r1 = sqrt(lam L)/2 grows with offset, so the WIDEST offset sets
#   how close the direct-wave sensitivity gets to the anomalies.  Measured:
#       L = 2004  r1 = 301  edge z = 703  3.39 sigma   OK
#       L = 2502  r1 = 336  edge z = 738  3.21 sigma   OK
#       L = 3000  r1 = 368  edge z = 770  3.05 sigma   OK   <-- used here
#       L = 3161  r1 = 378  edge z = 780  3.00 sigma   the exact ceiling
#       L = 3450  r1 = 395  edge z = 797  2.92 sigma   FAILS
#   so |offset| stops at 3000 m: half-aperture 3000, full aperture 6000 m.
#
#   The absorbing boundary.  300 m of model is left outside each end of the
#   line.  ntaper EXTENDS the grid (getParameters.c:529), so this margin is
#   physical model, not taper -- it is there so the end receivers are not
#   sitting on top of the edge.
#
#   dxrcv = dx.  No receiver-side spatial aliasing to argue about.
#
# XRCV is not the acquisition -- it is the REFERENCE TRACE, picked at the SAME
# 2004 m offset the single-receiver run dz2 used, so the per-arrival residual
# table stays directly comparable across dz2, dz3 and dz4.
XRCV1=300; XRCV2=6300; DXRCV=${DX}
XRCV=5304; ZRCV=402

# derived, so the analysis below never re-types them
NRCV=$(( (XRCV2 - XRCV1) / DXRCV + 1 ))
IREF=$(( (XRCV - XRCV1) / DXRCV ))
# the trace that stresses the Fresnel gate hardest: the largest |offset|.
HMAX=$(python3 -c "print(max(abs(${XRCV1}-${XSRC}), abs(${XRCV2}-${XSRC})))")
IMAX=$(python3 -c "
o1, o2 = abs(${XRCV1}-${XSRC}), abs(${XRCV2}-${XSRC})
print(0 if o1 >= o2 else ${NRCV} - 1)")

# Anomalies, true model only.  CHANGED: they move with the shot and are now
# placed SYMMETRICALLY ABOUT IT, +/- 486 m, keeping the 972 m = 4.86 sigma
# separation that stops the two Gaussians overlapping.
#
# With the shot at one end of the line (dz2, dz3) one privileged trace had gA
# on its PP down-leg and gB on its up-leg.  With the shot in the middle no
# single trace can do that -- the down-leg and the up-leg of any one trace are
# on the SAME side of the shot -- so the illumination is shared across the
# spread instead, and by symmetry the left and right halves illuminate gA and
# gB identically.  At u = (1380-402)/(2298-402) = 0.5158 the traces that put a
# PP leg on a site are, measured:
#     gA (x = 2814)  down-leg at x_r = 1416   up-leg at x_r = 2646
#     gB (x = 3786)  down-leg at x_r = 5184   up-leg at x_r = 3954
# all four inside the line.
# CHANGED earlier: opposite signs and one per parameter -- Vp SLOW, Vs FAST.
SIGMA=200
XA=2814; ZA=1380; APCT_VP=-5
XB=3786; ZB=1380; APCT_VS=+5

# CHANGED from 4.2 s, forced by the wider aperture.  The controlling arrival
# is SS at the LARGEST offset: at 3000 m, SS = 2*hypot(1500, 1896)/1200 =
# 4.029 s, wavelet peak 4.179 s, and the +/- 0.8/f_peak = 72 ms analysis
# window closes at 4.251 s -- PAST the old 4.2 s.  4.5 s leaves 249 ms.
# The constraint this trades against is df <= 1/TMOD: at 4.5 s that is
# 0.222 Hz against the df = 0.2 Hz used below, so it still clears.  The model
# stage gates on this explicitly rather than trusting the comment.
TMOD=4.5; DTRCV=0.001
NTAPER=200
GRAD_TAPER=30
WAV_FMIN=2; WAV_FMAX=25; WAV_T0=0.15
# CHANGED from LAG_NT=101.  The SS reflection is mispositioned by 218.8 ms,
# which images at |tau| = 109.4 ms -- OUTSIDE the old +/-100 ms window.  That
# truncation was measured in version 1: g_Vs peaked AT the window edge and was
# still rising there.  A clipped event has a sharp tau edge, which broadens
# its nu spectrum and smears the very slope the cone filter reads.  +/-160 ms
# clears it; the aliasing limit 1/(2 df) = 1.25 s is far away.
LAG_NT=161; LAG_DT=0.002      # tau in [-0.16, +0.16] s
D_FMIN=2; D_FMAX=36; D_NFREQ=171

# The cone, as an angle.  Full opening angle 2theta: kept entirely at or above
# TH_PASS, rejected entirely at or below TH_STOP, cos^2 between.
TH_PASS=150; TH_STOP=130

FPEAK=$(echo ${WAV_FMAX} | awk '{printf "%.4f", 0.4425*$1}')

# ---------------------------------------------------------------------------
if [ "${STAGE}" = "model" ] || [ "${STAGE}" = "all" ]; then

echo "=========================================================="
echo " test_lag_refl_dz.sh -- stage: model"
echo "=========================================================="
rm -rf ${WORK}
mkdir -p ${WORK}
cd ${WORK}

echo "--- grid and geometry ---"
printf '  domain %d x %d m, dx = dz = %d m\n' ${SIZEX} ${SIZEZ} ${DX}
printf '  S  (%6d, %6d)   Fx force (src_type=6)\n' ${XSRC} ${ZSRC}
printf '  R  line x = %d to %d, dxrcv = %d, z = %d -> %d receivers, Vx and Vz\n' \
    ${XRCV1} ${XRCV2} ${DXRCV} ${ZRCV} ${NRCV}
printf '  offsets %d to %d m;  reference trace x = %d (offset %d), index %d\n' \
    $((XRCV1 - XSRC)) $((XRCV2 - XSRC)) ${XRCV} $((XRCV - XSRC)) ${IREF}
printf '  reflector  TRUE z = %d m   BACKGROUND z = %d m   error %d m\n' \
    ${ZREFL_TR} ${ZREFL_BG} $((ZREFL_BG - ZREFL_TR))

echo ""
echo "--- models ---"
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=homo.su verbose=0
# Two model sets only.  There is no mute, so the fourth QC model of
# test_lag_reflector.sh (reflector-free anomalies, built only to make the mute
# window exact) has nothing to do here and is not built.
#   bgm   reflector at ZREFL_BG, no anomalies      the BACKGROUND
#   trm   reflector at ZREFL_TR + the anomalies    the TRUE model
for P in cp cs ro; do
    cp homo_${P}.su bgm_${P}.su
    cp homo_${P}.su trm_${P}.su
done

python3 - <<PYEOF
import sys
import numpy as np
sys.path.insert(0, '${ROOT}/fdelfwi/tests/kh_misfit')
import su_io

DX = ${DX}
VFAC = ${VFAC}
SIG = ${SIGMA}

# (1) the reflector.  Placed at the first sample with z >= ZREFL in each model,
#     so each is one flat row of nodes with no interpolated half cell.
for base, zr in (('bgm', ${ZREFL_BG}), ('trm', ${ZREFL_TR})):
    k = int(np.ceil(zr / DX))
    for par, fac in (('cp', VFAC), ('cs', VFAC), ('ro', 1.0)):
        fn = '%s_%s.su' % (base, par)
        d, h = su_io.read(fn)
        d[:, k:] *= fac
        su_io.write(fn, d, h)
    print('  %s: reflector at sample %d, z = %.0f m; below it Vp %.0f -> %.0f, '
          'Vs %.0f -> %.0f, rho %.0f (unchanged)'
          % (base, k, k * DX, ${VP0}, ${VP0} * VFAC,
             ${VS0}, ${VS0} * VFAC, ${RO0}))
r = (VFAC - 1.0) / (VFAC + 1.0)
print('  normal-incidence P reflection coefficient R = %.4f in BOTH models '
      '(rho equal, so this is the velocity contrast alone)' % r)

# (2) the two Gaussians, trm only.  One per PARAMETER at its own site and with
#     the OPPOSITE sign, which is the point of this test: perturbing Vs alone
#     moves mu and moves lambda back so Vp = sqrt((lam+2mu)/rho) is unchanged,
#     so the P legs feel only gA (slow) and the S legs only gB (fast).
print('')
print('  %-10s %8s %8s %8s %10s %10s %11s'
      % ('par', 'x', 'z', 'sigma', 'v0', 'v_peak', 'peak pert'))
for fn, lab, xa, za, pct in (('trm_cp.su', 'slow Vp', ${XA}, ${ZA}, ${APCT_VP}),
                             ('trm_cs.su', 'fast Vs', ${XB}, ${ZB}, ${APCT_VS})):
    amp = pct / 100.0
    d, h = su_io.read(fn)
    nx, nz = d.shape
    x = (np.arange(nx) * DX)[:, None]
    z = (np.arange(nz) * DX)[None, :]
    g = np.exp(-((x - xa) ** 2 + (z - za) ** 2) / (2.0 * SIG ** 2))
    v0 = float(d[0, 0])
    out = d.astype(np.float64) * (1.0 + amp * g)
    su_io.write(fn, out.astype(np.float32), h)
    vp_ = out.min() if amp < 0 else out[:, :int(np.ceil(${ZREFL_TR} / DX))].max()
    print('  %-10s %8.0f %8.0f %8.0f %10.1f %10.1f %+10.2f%%'
          % (lab, xa, za, SIG, v0, vp_, 100.0 * (vp_ / v0 - 1.0)))

# The dispersion guard, over BOTH models.  cmin is set by the slowest Vs
# anywhere in either model; the Vs anomaly here is FAST, so cmin stays at the
# background Vs and the guard is looser than in test_lag_reflector.sh.
cmin = min(float(su_io.read('%s_cs.su' % b)[0].min()) for b in ('bgm', 'trm'))
fdisp = cmin / (5.0 * DX)
fmeas = 1.22 * ${WAV_FMAX} + 1.46
print('')
print('  dispersion guard: cmin = %.1f m/s -> fmax < %.2f Hz; measured '
      'wavelet edge %.2f Hz (%.0f%% headroom)'
      % (cmin, fdisp, fmeas, 100.0 * (fdisp / fmeas - 1.0)))
if fmeas >= fdisp:
    raise SystemExit('the wavelet edge is above the dispersion guard')
PYEOF

echo ""
echo "--- wavelet ---"
makewave fmin=${WAV_FMIN} fmax=${WAV_FMAX} \
    dt=0.001 nt=4096 file_out=wave.su t0=${WAV_T0} verbose=0
printf '  fmin=%s fmax=%s t0=%s -> f_peak = %s Hz, period %s ms\n' \
    ${WAV_FMIN} ${WAV_FMAX} ${WAV_T0} ${FPEAK} \
    $(echo ${FPEAK} | awk '{printf "%.1f", 1000/$1}')

echo ""
echo "--- forward modelling (2 runs) ---"
# trm is the observed data.  bgm is modelled too, but ONLY so the kinematics
# below can be checked against real traces -- the gradient run computes its own
# synthetic internally and never reads d_bgm.
for M in trm bgm; do
    printf '  %-5s ... ' ${M}
    ${FDELFWI}/fdelfwi_fwd \
        file_cp=${M}_cp.su file_cs=${M}_cs.su file_den=${M}_ro.su \
        file_src=wave.su file_rcv=d_${M} \
        ischeme=3 iorder=4 src_type=6 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
        xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} \
        dxrcv=${DXRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 > log_fwd_${M}.txt 2>&1
    suop2 d_${M}_000_rtxx.su d_${M}_000_rtzz.su op=sum | \
        sugain scale=0.5 > d_${M}_000_rp.su
    echo "done"
done

echo ""
echo "--- kinematics, and what dominates the residual ---"
python3 - <<PYEOF
import sys
import numpy as np
sys.path.insert(0, '${ROOT}/fdelfwi/tests/kh_misfit')
import su_io

IREF, IMAX, NRCV = ${IREF}, ${IMAX}, ${NRCV}
h_off = abs(${XRCV} - ${XSRC})          # the REFERENCE trace, as in dz2
h_max = float(${HMAX})                  # the widest offset in the line
per = 1000.0 / ${FPEAK}
hw = 0.8 / ${FPEAK}

# the line actually written, read off the file rather than assumed
nt_, _ = su_io.read('d_trm_000_rvx.su')
print('  receiver line: %d traces on disk, %d expected' % (nt_.shape[0], NRCV))
if nt_.shape[0] != NRCV:
    raise SystemExit('the receiver line is not the length the script thinks '
                     'it is -- xrcv1/xrcv2/dxrcv were not honoured')
del nt_

def arrivals(h):
    """straight-ray times at half-offset h/2, true and background."""
    rows = []
    for lab, v in (('direct  P', ${VP0}), ('direct  S', ${VS0})):
        t = h / v
        rows.append((lab, t, t, 0.0))
    for lab, v in (('reflected PP', ${VP0}), ('reflected SS', ${VS0})):
        tt = 2.0 * np.hypot(0.5 * h, ${ZREFL_TR} - ${ZSRC}) / v
        tb = 2.0 * np.hypot(0.5 * h, ${ZREFL_BG} - ${ZSRC}) / v
        rows.append((lab, tt, tb, 1000.0 * (tb - tt)))
    return rows

print('  straight-ray arrival times [s] at the REFERENCE offset %d m:' % h_off)
print('  %-22s %10s %10s %10s' % ('event', 'true', 'background', 'shift [ms]'))
for lab, tt, tb, ds in arrivals(h_off):
    print('  %-22s %10.4f %10.4f %+10.1f' % (lab, tt, tb, ds))
print('  wavelet period at f_peak = %.4g Hz is %.1f ms' % (${FPEAK}, per))

# RECORD-LENGTH GATE.  With a line, the latest event is SS at the widest
# offset, not at the reference offset.  Measured, then checked against tmod.
tss = max(tt for lab, tt, _, _ in arrivals(h_max) if lab.startswith('reflec'))
tend = tss + ${WAV_T0} + hw
print('  latest event over the whole line: SS at offset %.0f m, t = %.4f s; '
      'its analysis window closes at %.4f s, tmod = %.2f s (%.0f ms spare)'
      % (h_max, tss, tend, ${TMOD}, 1000.0 * (${TMOD} - tend)))
if tend > ${TMOD}:
    raise SystemExit('RECORD TOO SHORT: SS at the widest offset is truncated. '
                     'Raise TMOD (watch df <= 1/TMOD) or shorten the line.')

# The anomalies' traveltime contribution, for scale against those shifts.  A
# ray crossing a Gaussian dead centre picks up dt = (dv/v) * sqrt(2 pi) sigma / v.
for lab, pct, v in (('slow Vp on the P leg', ${APCT_VP}, ${VP0}),
                    ('fast Vs on the S leg', ${APCT_VS}, ${VS0})):
    dt = -(pct / 100.0) * np.sqrt(2.0 * np.pi) * ${SIGMA} / v
    print('  %-22s %+10.1f ms  (one centred crossing)' % (lab, 1000.0 * dt))

# ---------------------------------------------------------------------------
# THE FRESNEL GATE.  Version 1 of this run failed here and nothing caught it,
# so the check is now explicit and it refuses to continue.
#
# A ray-theory statement that the anomaly is "off the direct path" is not
# enough: the direct wave is sensitive over its first Fresnel zone, half width
# r1 = sqrt(lam L)/2 at the midpoint.  What matters is the Gaussian amplitude
# at the EDGE of that zone, not at its centre.
# ---------------------------------------------------------------------------
#
# WITH A LINE the gate has to be evaluated at the WIDEST offset, because
# r1 grows as sqrt(L): the far trace is the one whose direct wave reaches
# deepest toward the anomalies.  That is the only hard gate here.
# ---------------------------------------------------------------------------
lam = ${VP0} / ${FPEAK}
u = (${ZA} - ${ZSRC}) / float(${ZREFL_TR} - ${ZSRC})
print('')
print('  direct-P first Fresnel zone, r1 = sqrt(lam L)/2 with lam = %.0f m.'
      % lam)
print('  Anomalies at z = %d, sigma = %d, instruments at z = %d.'
      % (${ZA}, ${SIGMA}, ${ZSRC}))
print('  %10s %8s %10s %10s %10s %10s %10s'
      % ('offset', 'r1', 'z edge', 'n sigma', 'g/g_peak', 'PP legs', 'n sigma'))
fail = False
offs = np.unique(np.abs(np.linspace(${XRCV1}, ${XRCV2}, NRCV) - ${XSRC}))
# the gate is applied to EVERY offset; only six rows are printed, plus every
# offset that fails
show = set(np.round(np.linspace(0, offs.size - 1, 6)).astype(int).tolist())
seps = []
for i_, L in enumerate(offs):
    r1 = 0.5 * np.sqrt(lam * L)
    zedge = ${ZSRC} + r1
    ns = abs(${ZA} - zedge) / ${SIGMA}
    sep = L * (1.0 - u)
    seps.append(sep)
    bad = ns < 3.0
    fail = fail or bad
    if i_ in show or bad:
        print('  %10.0f %8.0f %10.0f %10.2f %10.3e %10.0f %10.2f%s'
              % (L, r1, zedge, ns, np.exp(-0.5 * ns ** 2), sep,
                 sep / ${SIGMA}, '   <-- FAILS' if bad else ''))
seps = np.array(seps)
if fail:
    raise SystemExit('FRESNEL GATE FAILED: at the offsets flagged above the '
                     'anomaly is inside the direct-wave first Fresnel zone. '
                     'Shorten the line (XRCV2), move the instruments up, move '
                     'the anomalies down, or shrink sigma -- see the geometry '
                     'block in the script.')

# the number that actually matters: how much the direct P is retimed
gray = np.exp(-0.5 * ((${ZA} - ${ZSRC}) / ${SIGMA}) ** 2)
dt_dir = (abs(${APCT_VP}) / 100.0) * gray * np.sqrt(2.0 * np.pi) * ${SIGMA} / ${VP0}
print('  predicted direct-P retiming from the Vp Gaussian: %.4f ms '
      '(%.2e of a %.1f ms period)'
      % (1000.0 * dt_dir, dt_dir * ${FPEAK}, 1000.0 / ${FPEAK}))

# Leg separation is a REPORT, not a gate.  It was a gate when there was one
# trace and the geometry could be tuned to satisfy it.  A line necessarily
# contains short offsets whose two PP legs overlap inside one sigma, and that
# is not an error -- those traces simply cannot tell gA from gB, and their
# reflection points are nowhere near either anomaly anyway.  Gating on it
# would make any receiver line impossible.
print('  PP leg separation at z = %d (u = %.4f): %.0f to %.0f m = %.2f to '
      '%.2f sigma;  %d of %d traces clear 4 sigma'
      % (${ZA}, u, seps.min(), seps.max(), seps.min() / ${SIGMA},
         seps.max() / ${SIGMA}, int((seps / ${SIGMA} >= 4.0).sum()), NRCV))

# Measured, not asserted: the RMS of the actual residual, and how much of it
# lives in the window around the PP shift.
print('')
print('  measured residual, res = syn - obs with syn from the BACKGROUND:')
print('  %-8s %12s %12s %12s' % ('comp', 'rms(obs)', 'rms(res)', 'res/obs'))
for c in ('rvx', 'rvz'):
    o, _ = su_io.read('d_trm_000_%s.su' % c)
    b, _ = su_io.read('d_bgm_000_%s.su' % c)
    r = b.astype(np.float64) - o.astype(np.float64)
    ro = float(np.sqrt(np.mean(o.astype(np.float64) ** 2)))
    rr = float(np.sqrt(np.mean(r ** 2)))
    print('  %-8s %12.4e %12.4e %12.3f' % (c, ro, rr, rr / ro))

# per-arrival, which is the split that decides what the gradient images.
# Each window has to be re-centred PER TRACE now: with a line the arrival
# times move with offset, so one fixed window would mix events together.
O, _ = su_io.read('d_trm_000_rvx.su')
B, _ = su_io.read('d_bgm_000_rvx.su')
O = O.astype(np.float64)
B = B.astype(np.float64)
xr = np.linspace(${XRCV1}, ${XRCV2}, NRCV)
tt = np.arange(O.shape[1]) * ${DTRCV}

def windows(h):
    return (('direct P', h / ${VP0} + ${WAV_T0}),
            ('direct S', h / ${VS0} + ${WAV_T0}),
            ('PP', 2.0 * np.hypot(0.5 * h, ${ZREFL_TR} - ${ZSRC}) / ${VP0}
                  + ${WAV_T0}),
            ('SS', 2.0 * np.hypot(0.5 * h, ${ZREFL_TR} - ${ZSRC}) / ${VS0}
                  + ${WAV_T0}))

def win(a, tc):
    m = (tt >= tc - hw) & (tt < tc + hw)
    return float(np.sqrt(np.mean(a[m] ** 2)))

def shift(o_, b_, tc):
    """sub-sample time shift of b_ relative to o_ inside one window."""
    m = (tt >= tc - hw) & (tt < tc + hw)
    oo, bb = o_[m], b_[m]
    if not np.any(oo):
        return 0.0
    cc = np.correlate(bb, oo, 'full')
    k = int(np.argmax(cc))
    s = (k - (len(oo) - 1)) * ${DTRCV}
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        d_ = y0 - 2 * y1 + y2
        if d_ != 0.0:
            s += 0.5 * (y0 - y2) / d_ * ${DTRCV}
    return s

for tag, it in (('REFERENCE trace', IREF), ('WIDEST-offset trace', IMAX)):
    h = abs(xr[it] - ${XSRC})
    o, b = O[it], B[it]
    r = b - o
    print('')
    print('  residual by arrival window, rvx, %s: x = %.0f, offset %.0f m'
          % (tag, xr[it], h))
    print('  %-14s %10s %12s %12s %10s'
          % ('window', 'centre', 'rms(obs)', 'rms(res)', 'res/obs'))
    vals = {}
    for lab, tc in windows(h):
        a, c_ = win(o, tc), win(r, tc)
        vals[lab] = c_
        print('  %-14s %10.3f %12.4e %12.4e %10.3f'
              % (lab, tc, a, c_, c_ / a if a > 0 else np.nan))
    print('  direct-P residual / PP residual: %.3f  '
          '(version 1 of this run had 26.1; dz2, one receiver, had 0.009)'
          % (vals['direct P'] / vals['PP']))

# THE GATE.  The direct arrivals must cancel: both models are identical along
# the direct path by construction, so a direct-P residual means the anomaly
# has leaked into the Fresnel zone.  Measured by cross-correlation, not by
# eye.  Version 1 of this run showed -5.06 ms at its one offset.
#
# With a line the gate runs over EVERY trace and reports the worst, because
# the leak is offset dependent and checking only the reference trace would
# have missed exactly the traces the Fresnel table says are marginal.
sh = np.array([shift(O[i], B[i], windows(abs(xr[i] - ${XSRC}))[0][1])
               for i in range(NRCV)])
iw = int(np.argmax(np.abs(sh)))
print('')
print('  measured direct-P time shift, syn vs obs, over all %d traces:' % NRCV)
print('    median %+.3f ms   rms %.3f ms   worst %+.3f ms at x = %.0f '
      '(offset %.0f)'
      % (1000.0 * np.median(sh), 1000.0 * np.sqrt(np.mean(sh ** 2)),
         1000.0 * sh[iw], xr[iw], abs(xr[iw] - ${XSRC})))
print('    worst is %.2f%% of a %.1f ms period'
      % (100.0 * abs(sh[iw]) * ${FPEAK}, 1000.0 / ${FPEAK}))
if abs(sh[iw]) * ${FPEAK} > 0.01:
    raise SystemExit('DIRECT WAVE DID NOT CANCEL: shift is more than 1%% of a '
                     'period on at least one trace.  The anomaly is still '
                     'inside the direct-P Fresnel zone at that offset.')
PYEOF

echo ""
echo "stage model complete."
fi

# ---------------------------------------------------------------------------
if [ "${STAGE}" = "grad" ] || [ "${STAGE}" = "all" ]; then

echo ""
echo "=========================================================="
echo " test_lag_refl_dz.sh -- stage: grad"
echo "=========================================================="
cd ${WORK}

# ONE run.  background bgm, observed trm, and the residual is whatever
# test_gradient_visual forms internally, which is syn - obs and nothing else:
# res_taper=0, aper_taper=0, data_weight=0, misfit=0, and no mute pass.
mkdir -p run
cd run
for F in bgm_cp bgm_cs bgm_ro trm_cp trm_cs trm_ro wave; do
    ln -sf ../${F}.su .
done
for C in rvx rvz rtxx rtzz rp; do
    ln -sf ../d_trm_000_${C}.su obs_000_${C}.su
done

echo "  lag gradient (nfreq=${D_NFREQ}, ${D_FMIN}-${D_FMAX} Hz, nt=${LAG_NT}, dtau=${LAG_DT} s)"
${FDELFWI}/test_gradient_visual \
    file_cp=bgm_cp.su file_cs=bgm_cs.su file_den=bgm_ro.su \
    file_src=wave.su file_rcv=syn_dense \
    ischeme=3 iorder=4 src_type=6 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} \
    dxrcv=${DXRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
    ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=0 aper_taper=0 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=50 chk_base=chk_dense \
    param=2 scaling=0 precond=0 smooth_grad=0 \
    vp_min=1000 vp_max=4500 vs_min=500 vs_max=2500 \
    rho_min=1500 rho_max=2500 \
    data_weight=0 misfit=0 comp=_rvx,_rvz \
    lag_nfreq=${D_NFREQ} lag_fmin=${D_FMIN} lag_fmax=${D_FMAX} \
    lag_nt=${LAG_NT} lag_dt=${LAG_DT} \
    file_obs=obs out_base=dense > log_dense.txt 2>&1
grep -E "lagGradientInit|lag cube" log_dense.txt | head -4 || true
grep -E "Total misfit" log_dense.txt | tail -1

# nshot=1, so the per-shot cube duplicates the stack byte for byte.
rm -f dense_vp_scaled.su dense_vs_scaled.su dense_rho_scaled.su \
      dense_vp_preco.su  dense_vs_preco.su  dense_rho_preco.su \
      precond_*.su hess_raw_*.su hess_prebeta_*.su \
      dense_shot000_lagvp.su dense_shot000_lagvs.su \
      dense_shot000_lagaxis.txt
rm -rf chk_dense*
cd ${WORK}

echo ""
echo "stage grad complete."
fi

# ---------------------------------------------------------------------------
if [ "${STAGE}" = "figs" ] || [ "${STAGE}" = "all" ]; then

echo ""
echo "=========================================================="
echo " test_lag_refl_dz.sh -- stage: figs"
echo "=========================================================="
# lag_dz_figs.py reads d_trm_000_*.su / d_bgm_000_*.su and the models from the
# CWD and writes its PNGs there, so it has to run from ${WORK}.
cd ${WORK}

python3 ${HERE}/lag_dz_figs.py \
    --run=run --base=dense \
    --dt=${DTRCV} --wt0=${WAV_T0} \
    --vp=${VP0} --vs=${VS0} \
    --zrefl=${ZREFL_TR} --zrefl-bg=${ZREFL_BG} \
    --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} \
    --xrcv1=${XRCV1} --xrcv2=${XRCV2} --dxrcv=${DXRCV} \
    --xa=${XA} --za=${ZA} --xb=${XB} --zb=${ZB} \
    --xgather=$(( (XSRC + XRCV) / 2 )) \
    --zgather ${ZA} ${ZREFL_BG} ${ZREFL_TR} \
    --dx-m=${DX}

echo ""
echo "stage figs complete."
fi

# ---------------------------------------------------------------------------
if [ "${STAGE}" = "anal" ] || [ "${STAGE}" = "all" ]; then

echo ""
echo "=========================================================="
echo " test_lag_refl_dz.sh -- stage: anal"
echo "=========================================================="
cd ${WORK}

python3 ${HERE}/lag_dz_filter.py \
    --run=run --base=dense \
    --vp=${VP0} --vs=${VS0} \
    --theta-pass=${TH_PASS} --theta-stop=${TH_STOP} \
    --vfac=${VFAC} --zrefl=${ZREFL_TR} \
    --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} \
    --xa=${XA} --za=${ZA} --xb=${XB} --zb=${ZB} \
    --zone-r=$((2 * SIGMA)) \
    --out=cone

echo ""
echo "stage anal complete."
fi
