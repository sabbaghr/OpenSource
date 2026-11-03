#!/bin/bash
# test_lag_tomo_inversion.sh -- fwi_inversion driven by the CONE-FILTERED
#                               time-lag gradient (lag_tomo=1)
#
# What this exercises
# -------------------
# With lag_tomo=1 the gradient handed to the optimizer is no longer the
# zero-lag kernel.  Per shot, fwi_inversion now:
#
#   1. folds every time step of the adjoint loop into running DFTs
#      (lag_gradient.c), from which g_Vp(x,tau) and g_Vs(x,tau) are
#      reconstructed on a lag axis of lag_nt points;
#   2. cone-filters that cube in the 3D (tau,x,z) Fourier domain
#      (lag_cone_filter.c), keeping slopes |p| = |k|/|nu| BELOW lag_s_pass;
#   3. takes the tau=0 slice of the passed cube as the shot's gradient.
#
# A SMALL slope is a nearly lag-independent event, i.e. a WIDE opening angle,
# i.e. the transmission (tomographic) kernel.  A LARGE slope is the specular
# reflection imprint.  So the filter keeps the part of the gradient that
# updates the background velocity and discards the part that builds
# reflectivity.
#
# The angle map is |p| = cos(theta)/v with 2*theta the FULL opening angle, so
# at the background Vp of this model (2160 m/s):
#     lag_s_pass=0.20 ms/m  ->  2theta = 2*acos(0.20*2.160) = 129 deg
#     lag_s_stop=0.30 ms/m  ->  2theta = 2*acos(0.30*2.160) = 99 deg
# Opening angles wider than 129 deg pass in full, narrower than 99 deg are
# rejected.
#
# The model: a UNIFORM VELOCITY ERROR over a DENSITY reflector
# ------------------------------------------------------------
#   TRUE        Vp = 2000, Vs = 1150 everywhere
#               rho = 2000, stepping to 2200 below z = 600 m
#
#   INITIAL     Vp = 2000*1.08, Vs = 1150*1.08   (the SAME medium, 8% fast)
#               rho = identical to the true model
#
# The error is entirely in the velocity and nowhere else.  The reflector is in
# DENSITY, so it exists in both models and does not move between them: the
# residual is a traveltime error on a reflection that is present either way,
# which is precisely the situation the tomographic kernel is for.  Nothing is
# done to the data.
#
# The one thing that is CHECKED rather than displayed
# ---------------------------------------------------
# IDENTITY: with the cone opened all the way (lag_s_pass enormous), the edge
# tapers off and pad=1, the filter is the identity, so the lag_tomo=1 gradient
# must reproduce the ordinary lag_tomo=0 gradient.  That one correlation
# exercises the entire new path at once -- the DFT accumulators, the per-lag
# reconstruction, the Lame->velocity chain rule inside lagBuildCubes, the
# FFT round-trip normalisation, the tau=0 extraction, lagTomoInject, and the
# param=1 (strip-only) extraction that keeps the chain rule from being applied
# twice.  A sign error, a missing factor, an index transposition or a double
# conversion all show up in it.
#
# It can NOT be bit-identical: the lag reconstruction is band-limited to
# [lag_fmin, lag_fmax] while the conventional gradient sums the full spectrum.
# The criterion is a normalised correlation near 1 with an amplitude ratio
# near 1 -- the same criterion demo/gradient/test_lag_gradient.sh uses on the
# tau=0 slice.  Measured on this setup, with the cone reporting 100.000% of
# the spectral energy kept:
#
#     g_vp   corr = +0.99997094   ||lag||/||ref|| = 0.999974
#     g_vs   corr = +0.99996261   ||lag||/||ref|| = 1.001485
#
# so the 3e-5 shortfall is the band limiting and nothing else.  The thresholds
# below are set well inside that margin but far too tight for any factor,
# sign or index error to slip through.
#
# Everything after that is a FIGURE, not a test.  The script writes the
# filtered and unfiltered gradients side by side; it does not predict how much
# energy the cone should keep.
#
# Usage:  ./test_lag_tomo_inversion.sh

set -e
set -o pipefail

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

WORK=lag_tomo_inversion
mkdir -p ${WORK}
cd ${WORK}

# ---------------------------------------------------------------------------
# Geometry and the medium
# ---------------------------------------------------------------------------
DX=5; SIZEX=2000; SIZEZ=1000
NX=401; NZ=201

VP0=2000; VS0=1150; RO0=2000
ZREFL=600; RO1=2200          # density reflector: in BOTH models, never moves
FAST=1.08                    # the error: background is 8% fast

VP_BG=$(python3 -c "print(${VP0}*${FAST})")
VS_BG=$(python3 -c "print(${VS0}*${FAST})")

TMOD=1.5; DTRCV=0.004
NTAPER=100
GRAD_TAPER=20

NSHOTS=3
XSRC_START=400; XSRC_STEP=600
ZSRC=10
XRCV1=100; XRCV2=1900; ZRCV=100; DXRCV=20

# ---------------------------------------------------------------------------
# The lag axis and the frequency set
# ---------------------------------------------------------------------------
# DENSITY REQUIREMENT, enforced by lagGradientInit: the lag reconstruction
# replicates with period 1/df in the RELATIVE lag, and conj(A)B is a
# correlation over the whole record, so 1/df must be at least the record
# length T.  Here T = 1.5 s and the band is [2, 18] Hz, so
#     nfreq >= (18-2)*1.5 + 1 = 25.
# LAG_NFREQ=31 gives df = 0.533 Hz, 1/df = 1.875 s > T.  Going coarser is not
# a trade-off, it is an abort with the minimum printed.
LAG_NFREQ=31; LAG_FMIN=2; LAG_FMAX=18

# The lag axis only has to be long enough for the cone to measure a slope
# against; 31 lags at 5 ms is |tau| <= 75 ms.
LAG_NT=31; LAG_DT=0.005

# Cone: see the header for the angle equivalents at this background velocity.
S_PASS=0.20; S_STOP=0.30

echo "========================================================="
echo " LAG-TOMOGRAPHIC INVERSION TEST  (lag_tomo=1)"
echo "   true       Vp=${VP0}  Vs=${VS0}   rho ${RO0}/${RO1} across z=${ZREFL} m"
echo "   initial    Vp=${VP_BG}  Vs=${VS_BG}  (${FAST}x fast), same rho"
echo "   ${LAG_NFREQ} freqs ${LAG_FMIN}-${LAG_FMAX} Hz, ${LAG_NT} lags x ${LAG_DT} s"
echo "   cone  pass |p|<=${S_PASS}  stop |p|>=${S_STOP} ms/m"
echo "========================================================="

# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: models ---"

# TRUE: correct velocity, density reflector.
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=def poly=0 x=0,${SIZEX} z=${ZREFL},${ZREFL} \
    cp=${VP0} cs=${VS0} ro=${RO1} \
    orig=0,0 file_base=model_true.su verbose=0

# INITIAL: the same medium uniformly fast, and the SAME density.  Scaling the
# density file instead of rebuilding it guarantees the reflector is identical
# to the sample, so the only difference between the two models is velocity.
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP_BG} cs0=${VS_BG} ro0=${RO0} \
    intt=def poly=0 x=0,${SIZEX} z=${ZREFL},${ZREFL} \
    cp=${VP_BG} cs=${VS_BG} ro=${RO1} \
    orig=0,0 file_base=model_bg.su verbose=0

cmp model_true_ro.su model_bg_ro.su && \
    echo "  density identical in both models (the reflector does not move)"
echo "  Vp ${VP0} -> ${VP_BG},  Vs ${VS0} -> ${VS_BG}"

# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: wavelet ---"
# fp=12 Hz: the dispersion guard tests the MEASURED wavelet fmax, roughly
# 1.22*fp + 1.46 = 16.1 Hz, and cmin comes from the TRUE model (Vs=1150)
# because the background is the FAST one.  1150/(16.1*5) = 14 points per
# shortest wavelength, comfortably inside the guard.
makewave fp=12 dt=0.001 nt=1024 fmax=20 file_out=wave.su t0=0.10 verbose=0
echo "  fp=12 Hz, fmax=20 Hz, dt=1 ms"

# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: observed data from the true model (${NSHOTS} shots) ---"
# fdelfwi_fwd, NOT fdelmodc: the two have incompatible taper-boundary
# implementations and mixing them leaves a ~1% direct-wave residual that has
# nothing to do with the model error being inverted for.
#
# ONE call, not a shell loop: fdelfwi_fwd runs the whole shot loop itself and
# passes fileno=ishot to fdfwimodc, so writeRec already produces the
# zero-padded obs_000_, obs_001_, obs_002_ that fwi_inversion expects.  There
# is no fileno= parameter to drive from outside -- a loop with nshot=1 would
# write obs_000_ three times.
${FDELFWI}/fdelfwi_fwd \
    file_cp=model_true_cp.su file_cs=model_true_cs.su \
    file_den=model_true_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
    xsrc=${XSRC_START} zsrc=${ZSRC} nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 > /dev/null

for ((ishot=0; ishot<NSHOTS; ishot++)); do
    SHOTID=$(printf "%03d" ${ishot})
    # fdelmodc's rec_type_p is NOT an elastic hydrophone, and fdelfwi_fwd does
    # not write _rp at all, so the OBSERVED pressure is built here.
    suop2 obs_${SHOTID}_rtzz.su obs_${SHOTID}_rtxx.su op=sum | \
        sugain scale=0.5 > obs_${SHOTID}_rp.su
    echo "  shot $((ishot+1))/${NSHOTS}: xsrc=$((XSRC_START + ishot*XSRC_STEP)) m"
done

# ---------------------------------------------------------------------------
# run_inv <file_grad> <log> <extra args...>
#
# Everything that would make the computed gradient differ from the exact
# gradient of the printed J is OFF (res_taper=0, aper_taper=0, smooth_grad=0,
# scaling=0, precond=0, data_weight=0, misfit=0), so the two runs below differ
# ONLY in the lag path.
#
# rho is frozen: lag_tomo has no g_rho (the accumulators carry only the lambda
# and mu kernels), and fwi_inversion refuses to run with rho active.  The
# reference run freezes it too, or the comparison would not be like for like.
run_inv () {
    local grad=$1; local log=$2; shift 2
    ${FDELFWI}/fwi_inversion \
        file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
        file_src=wave.su file_rcv=syn_${grad} \
        ischeme=3 iorder=4 src_type=1 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
        xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
        xsrc=${XSRC_START} zsrc=${ZSRC} nshot=${NSHOTS} dxshot=${XSRC_STEP} \
        ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=0 aper_taper=0 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=50 chk_base=chk_${grad} \
        param=2 active_params=vp,vs \
        scaling=0 precond=0 smooth_grad=0 \
        data_weight=0 misfit=0 comp=_rp \
        vp_min=1500 vp_max=3000 vs_min=800 vs_max=1800 \
        rho_min=1800 rho_max=2400 \
        file_obs=obs file_grad=${grad} \
        algorithm=1 lbfgs_mem=5 niter=1 nls_max=5 conv=1e-6 write_iter=1 \
        "$@" > ${log} 2>&1
    rm -rf chk_${grad}_* fwi_rank*
}

# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4a: reference gradient, lag path OFF ---"
run_inv gref log_ref.txt lag_tomo=0
grep -E "Total misfit|=== " log_ref.txt | head -2 || true

echo ""
echo "--- Step 4b: lag path ON, cone OPENED ALL THE WAY (identity test) ---"
# lag_s_pass huge so H == 1 everywhere, tapers 0 and pad=1 so nothing else
# touches the cube.
#
# lag_nu0_keep=1 is REQUIRED for this to be an identity and is not a way of
# weakening the test.  On the nu=0 plane the slope is |k|/0, so the default
# 'limit' rule zeroes every k != 0 there NO MATTER how large lag_s_pass is --
# it is a separate rule, not part of the cone.  That plane is the
# tau-independent part of the cube and carried 0.7% of the energy here, which
# capped the correlation at 0.9989 instead of ~1.  lagConeSelfTest sets the
# same flag for the same reason.
run_inv gid log_id.txt \
    lag_tomo=1 \
    lag_nfreq=${LAG_NFREQ} lag_fmin=${LAG_FMIN} lag_fmax=${LAG_FMAX} \
    lag_nt=${LAG_NT} lag_dt=${LAG_DT} \
    lag_s_pass=1e9 lag_s_stop=2e9 lag_nu0_keep=1 \
    lag_tap_tau=0 lag_tap_x=0 lag_tap_z=0 lag_pad=1
grep -E "lagGradientInit|kept .* of the spectral energy" log_id.txt | head -4 || true

echo ""
echo "--- Step 4c: lag path ON, cone CLOSED to the tomographic kernel ---"
run_inv gtomo log_tomo.txt \
    lag_tomo=1 \
    lag_nfreq=${LAG_NFREQ} lag_fmin=${LAG_FMIN} lag_fmax=${LAG_FMAX} \
    lag_nt=${LAG_NT} lag_dt=${LAG_DT} \
    lag_s_pass=${S_PASS} lag_s_stop=${S_STOP}
grep -E "kept .* of the spectral energy" log_tomo.txt | head -2 || true

# ---------------------------------------------------------------------------
echo ""
echo "========================= IDENTITY CHECK ========================="
# The only assertion in this script.  Band limiting is why it is a
# correlation and not a cmp.
python3 - <<PYEOF
import numpy as np, sys

def readsu(fn, ns):
    raw = np.fromfile(fn, dtype=np.uint8)
    rec = 240 + 4*ns
    n   = raw.size // rec
    out = np.empty((n, ns), dtype=np.float32)
    for i in range(n):
        out[i] = raw[i*rec+240:(i+1)*rec].view(np.float32)
    return out.ravel()

ns = ${NZ}
fail = 0
for par in ('vp', 'vs'):
    a = readsu('gref_initial_%s.su' % par, ns).astype(np.float64)
    b = readsu('gid_initial_%s.su'  % par, ns).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        print('  g_%s: ZERO gradient (a=%.3e b=%.3e)' % (par, na, nb)); fail = 1; continue
    corr  = float(a @ b) / (na * nb)
    ratio = nb / na
    ok = (corr > 0.9995) and (0.99 < ratio < 1.01)
    print('  g_%-2s  corr = %+.6f   ||lag||/||ref|| = %.4f   %s'
          % (par, corr, ratio, 'PASS' if ok else 'FAIL'))
    if not ok: fail = 1

if fail:
    raise SystemExit('IDENTITY CHECK FAILED: with the cone wide open the '
                     'lag_tomo gradient must reproduce the conventional one.  '
                     'A corr near 1 with ratio near 2*rho*V or 1/(2*rho*V) '
                     'means the chain rule was applied twice or not at all; a '
                     'low corr means the lag reconstruction itself is wrong '
                     '(check df <= 1/T first).')
print('  identity OK -- the whole lag_tomo path reproduces the gradient.')
PYEOF

# ---------------------------------------------------------------------------
echo ""
echo "============================ FIGURES ============================"
# How much the cone actually removed.  Displayed, not asserted: the split
# depends on the acquisition geometry, and this script does not claim to know
# what it should be.
python3 - <<PYEOF
import numpy as np

def readsu(fn, ns):
    raw = np.fromfile(fn, dtype=np.uint8)
    rec = 240 + 4*ns
    n   = raw.size // rec
    out = np.empty((n, ns), dtype=np.float32)
    for i in range(n):
        out[i] = raw[i*rec+240:(i+1)*rec].view(np.float32)
    return out

ns = ${NZ}
for par in ('vp', 'vs'):
    r = readsu('gref_initial_%s.su'   % par, ns).astype(np.float64)
    t = readsu('gtomo_initial_%s.su'  % par, ns).astype(np.float64)
    nr = np.linalg.norm(r)
    print('  g_%-2s  ||tomo||/||ref|| = %.4f   corr = %+.4f'
          % (par, np.linalg.norm(t)/nr,
             float((r*t).sum())/(nr*np.linalg.norm(t) + 1e-300)))
PYEOF

echo ""
echo "  gradients:  gref_initial_{vp,vs}.su   (conventional)"
echo "              gid_initial_{vp,vs}.su    (lag path, cone open)"
echo "              gtomo_initial_{vp,vs}.su  (lag path, tomographic cone)"
echo "  logs:       log_ref.txt log_id.txt log_tomo.txt"
echo "  output in:  $(pwd)"
echo "================================================================="
