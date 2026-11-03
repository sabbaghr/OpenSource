#!/bin/bash
#
# test_gradient_1src_1rcv.sh - Single source / single receiver Vp-Vs gradient,
#                              with everything kept for inspection:
#                              forward snapshots, adjoint snapshots, raw data
#                              residuals per component, and scaled residuals.
#
# Geometry:
#   source   at x=500 m, z=20 m            (src_type=1, monopole)
#   receivers z=50 m, x=0..5000 m every 10 m  (501 receivers, 5 components)
#
# Surface-style acquisition: a shallow source and a receiver line covering the
# full model length, so the anomalies at z=800 m are illuminated over a wide
# aperture rather than by a single path. Offsets run from -500 m to +4500 m.
#
# NOTE on the top boundary. The source sits a few grid points below the top of
# the model and the top boundary is ABSORBING (taper) in both runs. The upgoing
# wavefield is therefore absorbed instead of reflected: there is no free
# surface, hence no source/receiver ghost and no Rayleigh wave. For marine-style
# physics at these depths you can set top=1 in BOTH runs and keep the sides
# absorbing -- that is now safe, because both runs use the same engine. Doing it
# with fdelmodc for the data was NOT safe: the two codes' free surfaces differ
# and that gave a 6-12% residual (worse than the 1% from the absorbing case).
#
# Model (built by this script, same as test_gradient_energy_norm.sh):
#   5000 x 2000 m, dx=dz=5, homogeneous Vp=2000, Vs=1150, rho=2000
#   - smooth +10% velocity anomaly at x=2000, z=800   (Vp 2200, Vs 1265)
#   - smooth -10% velocity anomaly at x=3000, z=800   (Vp 1800, Vs 1035)
#   - density-only reflector at z=1500 (rho 2000 -> 2200)
#   The background model is the plain homogeneous medium, so the gradient is
#   driven by all three. Set NULLTEST=1 in step 1 to strip them all, which must
#   give misfit and gradient exactly 0 (validates the obs/syn pipeline).
#
# Outputs (in ./gradient_1src_1rcv/):
#   grad_vp_raw.su grad_vs_raw.su grad_rho_raw.su     gradients (param=2)
#   grad_vp_scaled.su ... grad_vp_preco.su ...
#   tsnap_s{vx,vz,txx,tzz,txz}.su      TRUE-MODEL wavefield snapshots
#                                      (the run that generates the observed data)
#   fsnap_s{vx,vz,txx,tzz,txz}.su      FORWARD wavefield snapshots (background model)
#   asnap_s{vx,vz,txx,tzz,txz}.su      ADJOINT wavefield snapshots
#     (sna_type_p is not requested: for elastic it writes tzz verbatim,
#      writeSnapTimes.c:169-174, so it would just duplicate _stzz)
#   obs_000_r{vx,vz,txx,tzz,txz}.su    observed data
#   syn_000_r{vx,vz,txx,tzz,txz}.su    synthetic data (background model)
#   dres_r{vx,vz,txx,tzz,txz}.su       RAW residual r = d_syn - d_obs
#   res_000.su                         SCALED residual psi = Q r (adjoint src)
#   sres_r{vx,vz,txx,tzz,txz}.su       scaled residual split per component
#
# NOTE on the two residuals:
#   computeResidual() only writes the *scaled* residual psi = dJ/dd_syn = Q r
#   (computeResidual.c:381-383) -- the raw r lives in a temporary buffer and is
#   never written. The raw residual is therefore reconstructed here exactly, as
#   suop2 syn - obs, which is the same subtraction computeResidual performs.
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# TAG appends to the output directory, so model variants do not clobber each
# other. WITH_REFL / WITH_ANOM select which of the two true-model features are
# present, which is how their separate contributions to the gradient are
# isolated. Defaults reproduce the original run.
TAG=${TAG:-}
WITH_REFL=${WITH_REFL:-1}
WITH_ANOM=${WITH_ANOM:-1}

# BG_REFL=1 puts the SAME density reflector into the INITIAL (background) model.
# The reflector is then no longer part of what the gradient is being asked to
# recover -- it is known -- and the only difference between true and background
# is the pair of velocity anomalies. Two things change as a result:
#
#   * The BACKGROUND wavefield now contains a real reflection, not just the
#     direct wave. That is the field that enters the kernel, so the gradient
#     gets a genuine reflected illumination of the anomalies at z=800 m instead
#     of having to reach them with the direct wave alone.
#   * _ro becomes identical in both models, so g_rho is driven only by the
#     velocity anomalies leaking through the (lam,mu,rho) coupling. A large
#     g_rho here is a cross-talk measurement, which is the useful reading.
#
# Use with WITH_REFL=1 WITH_ANOM=1 (the defaults): the reflector must be in
# BOTH models and the anomalies in the true model only. BG_REFL=1 WITH_ANOM=0
# would make the two models identical and is rejected below.
BG_REFL=${BG_REFL:-0}

WORKDIR=gradient_1src_1rcv${TAG}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " SINGLE SOURCE / SINGLE RECEIVER GRADIENT (Vp, Vs, rho)"
echo "========================================================="

# =========================================================
# Model
# =========================================================
DX=5
SIZEX=5000
SIZEZ=2000
VP0=2000
VS0=1150
RO0=2000

# Velocity-anomaly geometry. Active whenever WITH_ANOM=1, which is the default.
ANOM_Z=800
ANOM_R=250
XANOM_FAST=2000;  VP_FAST=2200;  VS_FAST=1265    # +10%
XANOM_SLOW=3000;  VP_SLOW=1800;  VS_SLOW=1035    # -10%

REFL_Z=1500
RO_REFL=2200

SM_NX=21
SM_NZ=21
SM_ITER=8

# =========================================================
# Acquisition - ONE shallow source, full-length receiver line
# =========================================================
XSRC=${XSRC:-2500}
ZSRC=50                   # 4 grid points below the top of the model

# NSHOT > 1 turns this into a multi-shot run: shots at XSRC + i*DXSHOT, with
# the SAME fixed receiver line for every shot. Defaults keep the original
# single shot at x=2500 so existing runs are untouched.
#
# The default multi-shot spread (NSHOT=15 DXSHOT=200 XSRC=1200) puts the shots
# at 1200..4000 m, i.e. clear of the 200-grid-point (1000 m) absorbing pads at
# both ends. Note the receiver line does NOT move, so the outer shots are
# illuminated asymmetrically -- that is geometry, not a bug, and it shows up as
# weaker outer traces in the shot-index common-image gather.
NSHOT=${NSHOT:-1}
DXSHOT=${DXSHOT:-0}

# Per-shot gradients, written before they are summed (the CIG panels).
SHOT_GRAD=${SHOT_GRAD:-0}

XRCV1=0                   # receiver line spans the whole model in x
XRCV2=${SIZEX}
# DXRCV=DX (every grid point) is what OpenSource-recon_SL10's
# defineRecBoundary.c:35 does (nptx = NINT((x2-x1)/dx)+1). Sampling the
# injection contour any coarser aliases the surface integral: at DXRCV=10
# the above-Nyquist/2 energy fraction in the shallow adjoint field is 0.676,
# at DXRCV=5 it is 0.030 (-95.5%), and the grid-scale stipple disappears.
DXRCV=${DXRCV:-10}        # 501 receivers at dx=5 -> 2 grid points apart
ZRCV=100                  # 10 grid points deep

TMOD=4.0
# dtrcv sets rec_skipdt = dtrcv/dt. The adjoint source is injected only on
# those steps (applyAdjointSource.c:103), so dtrcv > dt makes the injection a
# TEMPORAL COMB with replicas at multiples of 1/dtrcv. Set DTRCV=0.001 to
# remove the decimation and see what the comb was contributing.
DTRCV=${DTRCV:-0.004}

# =========================================================
# Boundaries - IDENTICAL in both runs: taper, left/right/top/bottom=4
# =========================================================
# Step 3 (observed data) and Step 4 (gradient) must use the SAME engine and the
# SAME boundaries, otherwise the residual is not zero at the true model.
#
# Step 3 therefore uses fdelfwi_fwd, not fdelmodc. History of why:
#
#   * fdelmodc's elastic TAPER boundary (type 4) poisons the wavefield with
#     NaN/inf. Verified: with all boundaries=1 or =2 the output is clean; with
#     any boundary=4 NaN appears, independent of taper length (npml=20/50/100)
#     and of grid shape (square 401x401 and 1001x401 both fail). So fdelmodc
#     could only be run with PML here.
#   * fdelfwi cannot use PML: boundaries_adj.c implements the adjoint only for
#     types 4 (taper), 3 (rigid) and 1 (free surface). Its sole mention of PML
#     is a comment inside an "if (mod.ischeme <= 2)" block, i.e. acoustic only.
#     With PML the adjoint wavefield would reflect off the edges.
#   * That forced PML for the data and taper for the gradient, and the two
#     absorb differently (PML reflects ~1e-5, the cosine taper ~1e-2). With a
#     shallow source a few cells below the top boundary, that mismatch left a
#     ~1% residual on the DIRECT WAVE even with model_true bit-identical to
#     model_bg. Measured: 1.17% on _rvx, 4.10% on _rtxx.
#
# The kernels themselves are fine. With source and receivers 200 cells away
# from the pad, and comparing only times before any boundary reflection can
# arrive, fdelmodc and fdelfwi agree to float roundoff (diff/obs = 3e-9). The
# disagreement is entirely in the boundary implementations: fdelfwi was forked
# from fdelmodc before upstream commits b6b5292 -> 60eaadf -> 6f89b30 reworked
# the boundary / P-T index design, and that rework was never propagated. Trying
# top=1 (free surface) in both made it worse, 6-12%, since boundaries.c differs
# by 749 lines between the two codes.
#
# Using fdelfwi_fwd for Step 3 makes obs and syn bit-identical at the true
# model: misfit is exactly 0.000000e+00 and all 5 components compare equal byte
# for byte.
#
# NOTE: fdelmodc IGNORES ntaper (getParameters.c:453-454 reads ntaper into
# bnd->npml, then the next line overwrites it with npml's default of 20), so it
# needed npml=. fdelfwi gets the precedence right (getParameters.c:348-349), so
# both steps here use ntaper=.
NTAPER=200

# =========================================================
# Snapshots (one schedule shared by the true-model, forward and adjoint runs,
# so the three wavefields can be compared frame by frame)
# =========================================================
# DTSNAP=0.2 gives 20 snaps x 5 fields x 1.6 MB = ~160 MB per wavefield, which
# is fine for a movie of the WAVEFIELDS but badly undersamples the gradient
# KERNEL.  The kernel is a PRODUCT of two wavefields, so its spectrum reaches
# ~2*fmax = 50 Hz (period 0.02 s) even though each field only reaches 25 Hz.
# At dtsnap=0.2 s there are ~10 kernel cycles between frames, so summing the
# frames is aliasing, not quadrature -- measured cos(sum K_lam, g_Vp) = +0.22.
#
# For a kernel that can actually be summed, use dtsnap <= 0.01 (Nyquist for
# 50 Hz).  Narrow the window at the same time: the kernel rms is five decades
# down outside t = 0.3-3.0 s, so there is nothing to see at the ends.
#
#   DTSNAP=0.01 TSNAP1=0.3 TSNAP2=3.0   -> 271 frames, ~2.2 GB per wavefield
TSNAP1=${TSNAP1:-0.1}
TSNAP2=${TSNAP2:-${TMOD}}
DTSNAP=${DTSNAP:-0.2}

# Which fields to snapshot. Multi-shot writes one set of files PER SHOT, so
# keep it to vx,vz: that is all the div/curl (P/S) separation needs, and it
# holds the I/O to 2 fields x 15 shots instead of 5 x 15.
if [ ${NSHOT} -gt 1 ]; then
    SNA_ARGS="sna_type_vx=1 sna_type_vz=1"
else
    SNA_ARGS="sna_type_vx=1 sna_type_vz=1 sna_type_txx=1 sna_type_tzz=1 sna_type_txz=1"
fi

# =========================================================
# Gradient parameters
# =========================================================
PARAM=2                   # velocity: Vp, Vs, rho

# misfit=0 -> conventional L2, per-component adjoint injection.  This is the
# case of Fig. 1c of Wang et al. (EAGE 2025): a velocity residual drives a
# point FORCE, which radiates both P and S regardless of what mode the residual
# came from, so the adjoint field carries non-physical arrivals and false mode
# conversions.
#
# misfit=2 -> Kirchhoff-Helmholtz energy flux (kh_misfit.c).  The residual is
# injected TENSORIALLY: the velocity residual drives a stress source and the
# traction residual drives a force source, which is the discrete transpose of
# the correlation-type representation theorem.  Fig. 1d of the paper: the
# false modes are gone.  Requires data_weight=0 and the four-component set
# _rvx,_rvz,_rtxz,_rtzz (computeResidual.c errors out otherwise).
MISFIT=${MISFIT:-0}
KH_NORMAL=${KH_NORMAL:-1}

if [ "${MISFIT}" = "2" ]; then
    DATA_WEIGHT=${DATA_WEIGHT:-0}
    COMP=${COMP:-_rvx,_rvz,_rtxz,_rtzz}
else
    DATA_WEIGHT=${DATA_WEIGHT:-2}   # energy norm Q = diag(rho I, S)
    COMP=${COMP:-_rvx,_rvz,_rtxx,_rtzz,_rtxz}
fi
# Gaussian mask applied to the TOTAL gradient as the very last step
# (test_gradient_visual.c:1024), zeroing a disc of this radius in grid points
# around every source AND every receiver.
#
# Here the receiver line spans the whole model at z=100 m, so GRAD_TAPER=20
# wipes out a continuous band from the surface down to z~200 m. That is the
# usual thing to want for an inversion -- it kills the 1/r singularity where
# the adjoint source is injected -- but it also hides the shallow part of the
# gradient, and it is NOT applied to the per-shot files written by shot_grad=1,
# so sum(per-shot) only equals the total below the band.
#
# Set GRAD_TAPER=0 to see the gradient as the adjoint actually produces it.
# Expect very large amplitudes along the receiver line and at the sources;
# they will dominate any shared colour scale.
GRAD_TAPER=${GRAD_TAPER:-20}
RES_TAPER=100
SMOOTH_GRAD=${SMOOTH_GRAD:-1}

# Aperture taper on the ADJOINT INJECTION, in grid points (0 = off).
# Ramps the injected adjoint amplitude to zero over the outermost APER_TAPER
# grid points at each end of the receiver line, which removes the diffractions
# radiated by the abrupt truncation of the array.
#
# This geometry is the worst case for that artifact: the line spans the whole
# model (XRCV1=0 .. XRCV2=SIZEX) while NTAPER=200, so the outermost ~40
# receivers at each end sit INSIDE the absorbing pad.
#
# 40 grid points = 200 m = 20 receivers ramped at each end (DXRCV=10, DX=5).
#
# CAVEAT: applied outside the misfit, so the gradient is A^T W dd while the
# reported J is 0.5 dd^T Q dd -- a search-direction weight, not the exact
# gradient of the printed J. Same compromise RES_TAPER already makes. See the
# comment block in fdelfwi/aperture_taper.c.
APER_TAPER=200
# No Hessian / preconditioner here: this test is about the gradient itself,
# so precond=0 and grad_*_preco.su is just a copy of grad_*_scaled.su.
PRECOND=0

VP_MIN=1500;  VP_MAX=4500
VS_MIN=500;   VS_MAX=2500
RHO_MIN=1500; RHO_MAX=2500

# =========================================================
# STEP 1: Models
# =========================================================
echo "--- Step 1: Models ---"

# True-model content. Set NULLTEST=1 to strip every anomaly, which makes
# model_true bit-identical to model_bg and must give misfit and gradient
# exactly 0 -- the null test that validates the obs/syn pipeline. Set
# NULLTEST=0 for the real test (density reflector + velocity anomalies).
NULLTEST=0

REFL_ARGS=""
ANOM_ARGS=""
if [ ${NULLTEST} -eq 0 ]; then
    if [ ${WITH_REFL} -eq 1 ]; then
        REFL_ARGS="intt=def poly=0 x=0,${SIZEX} z=${REFL_Z},${REFL_Z} cp=${VP0} cs=${VS0} ro=${RO_REFL}"
    fi
    if [ ${WITH_ANOM} -eq 1 ]; then
        ANOM_ARGS="intt=elipse poly=0 x=${XANOM_FAST} z=${ANOM_Z} var=${ANOM_R},${ANOM_R} cp=${VP_FAST} cs=${VS_FAST} ro=${RO0} intt=elipse poly=0 x=${XANOM_SLOW} z=${ANOM_Z} var=${ANOM_R},${ANOM_R} cp=${VP_SLOW} cs=${VS_SLOW} ro=${RO0}"
    fi
fi

# The reflector goes into the background only when asked for. REFL_ARGS sets
# cp=VP0 cs=VS0, so it perturbs _ro and nothing else -- the background _cp/_cs
# stay exactly homogeneous either way.
BG_REFL_ARGS=""
if [ ${BG_REFL} -eq 1 ]; then
    if [ ${NULLTEST} -eq 1 ] || [ ${WITH_REFL} -eq 0 ]; then
        echo "ERROR: BG_REFL=1 needs WITH_REFL=1 and NULLTEST=0." >&2
        exit 1
    fi
    if [ ${WITH_ANOM} -eq 0 ]; then
        echo "ERROR: BG_REFL=1 with WITH_ANOM=0 makes true and background" >&2
        echo "       identical -- zero residual, nothing to invert for." >&2
        exit 1
    fi
    BG_REFL_ARGS="${REFL_ARGS}"
fi

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    ${BG_REFL_ARGS} \
    orig=0,0 file_base=model_bg.su verbose=0

echo "  true model: reflector=${WITH_REFL} anomalies=${WITH_ANOM} (NULLTEST=${NULLTEST})"
echo "  bg   model: reflector=${BG_REFL}"

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    ${REFL_ARGS} \
    ${ANOM_ARGS} \
    orig=0,0 file_base=model_sharp.su verbose=0

# Smooth _cp/_cs only, so the density reflector in _ro stays sharp. The smooth
# step rounds off the stair-cased edges of the elliptical velocity anomalies;
# with NULLTEST=1 there is nothing to smooth, so just copy.
# IMPORTANT: every branch must write model_true_cp/cs. If a branch leaves them
# unwritten, a STALE pair from an earlier run is silently used as the true
# model -- that cost a debugging session.
if [ ${NULLTEST} -eq 1 ]; then
    cp model_sharp_cp.su model_true_cp.su
    cp model_sharp_cs.su model_true_cs.su
else
    smooth file_in=model_sharp_cp.su file_out=model_true_cp.su \
        ntsm=${SM_NZ} nxsm=${SM_NX} niter=${SM_ITER} verbose=0 2>/dev/null
    smooth file_in=model_sharp_cs.su file_out=model_true_cs.su \
        ntsm=${SM_NZ} nxsm=${SM_NX} niter=${SM_ITER} verbose=0 2>/dev/null
fi
cp model_sharp_ro.su model_true_ro.su

# Report which parameters actually differ between true and background. This is
# the only thing the gradient can respond to, and it catches a stale
# model_true_* file masquerading as a modeling discrepancy.
if [ ${NULLTEST} -eq 1 ]; then
    echo "  NULLTEST=1: true vs bg must be IDENTICAL in all three"
elif [ ${BG_REFL} -eq 1 ]; then
    echo "  BG_REFL=1: cp/cs must differ (anomalies), _ro must be IDENTICAL"
else
    echo "  NULLTEST=0: true vs bg (cp/cs differ via anomalies, ro via reflector)"
fi
for P in cp cs ro; do
    if cmp -s model_true_${P}.su model_bg_${P}.su; then
        echo "    _${P}: identical"
    else
        echo "    _${P}: differs"
    fi
done

for F in model_true_cp model_true_cs model_true_ro; do
    MN=$(sumax < ${F}.su mode=min outpar=/dev/stdout 2>/dev/null)
    MX=$(sumax < ${F}.su mode=max outpar=/dev/stdout 2>/dev/null)
    printf "  %-16s min=%s  max=%s\n" "${F}" "${MN}" "${MX}"
done

# =========================================================
# STEP 2: Wavelet
# =========================================================
echo "--- Step 2: Wavelet ---"
makewave fp=14 fmin=2 fmax=25 dt=0.001 nt=4096 file_out=wave.su t0=0.10 verbose=0

# =========================================================
# STEP 3: Observed data (true model), 1 shot, 1 receiver
# =========================================================
echo "--- Step 3: Observed data (true model) ---"

# True-model snapshots only for the single-shot case. fdelfwi_fwd uses the
# unindexed writeSnapTimes naming, so with nshot>1 every shot would APPEND into
# one tsnap_s*.su file (only test_gradient_visual got the per-shot base name).
if [ ${NSHOT} -gt 1 ]; then
    TSNAP_ARGS=""
else
    TSNAP_ARGS="file_snap=tsnap tsnap1=${TSNAP1} tsnap2=${TSNAP2} dtsnap=${DTSNAP} ${SNA_ARGS}"
fi

${FDELFWI}/fdelfwi_fwd \
    file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=6 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DXRCV} \
    zrcv1=${ZRCV} zrcv2=${ZRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=${NSHOT} dxshot=${DXSHOT} \
    ntaper=${NTAPER} \
    left=4 right=4 top=4 bottom=4 \
    ${TSNAP_ARGS}

# fdelfwi's writeRec already zero-pads the shot number (writeRec.c:77), so the
# files land as obs_000_r*.su and no renaming step is needed (fdelmodc wrote
# obs_0_r*.su and had to be renamed).
ISHOT=0
while [ ${ISHOT} -lt ${NSHOT} ]; do
    S=$(printf "%03d" ${ISHOT})
    suop2 obs_${S}_rtzz.su obs_${S}_rtxx.su op=sum | sugain scale=0.5 > obs_${S}_rp.su
    ISHOT=$((ISHOT + 1))
done

echo "  Observed traces per component (shot 000):"
for EXT in rvx rvz rtxx rtzz rtxz; do
    printf "    %-6s %s\n" "${EXT}" "$(surange < obs_000_${EXT}.su | head -1)"
done

# =========================================================
# STEP 4: Gradient, with forward AND adjoint snapshots
# =========================================================
echo ""
echo "========================================================="
echo " Gradient: param=${PARAM}, data_weight=${DATA_WEIGHT}, misfit=${MISFIT}"
echo "   comp=${COMP}"
echo "   tapers: grad=${GRAD_TAPER} gp, res=${RES_TAPER} samples, aper=${APER_TAPER} gp"
echo "   forward snaps -> fsnap_s*.su    adjoint snaps -> asnap_s*.su"
echo "========================================================="

${FDELFWI}/test_gradient_visual \
    file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=6 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DXRCV} \
    zrcv1=${ZRCV} zrcv2=${ZRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=${NSHOT} dxshot=${DXSHOT} \
    ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=${RES_TAPER} \
    aper_taper=${APER_TAPER} \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_1p1 \
    file_snap=fsnap file_snap_adj=asnap \
    tsnap1=${TSNAP1} tsnap2=${TSNAP2} dtsnap=${DTSNAP} \
    ${SNA_ARGS} \
    shot_grad=${SHOT_GRAD} \
    param=${PARAM} scaling=1 precond=${PRECOND} \
    smooth_grad=${SMOOTH_GRAD} \
    vp_min=${VP_MIN} vp_max=${VP_MAX} \
    vs_min=${VS_MIN} vs_max=${VS_MAX} \
    rho_min=${RHO_MIN} rho_max=${RHO_MAX} \
    data_weight=${DATA_WEIGHT} \
    misfit=${MISFIT} kh_normal=${KH_NORMAL} \
    comp=${COMP} \
    file_obs=obs out_base=grad

# =========================================================
# STEP 5: Residuals
# =========================================================
echo ""
echo "--- Step 5: Residuals ---"

# Raw residual r = d_syn - d_obs, per component (same subtraction
# computeResidual does internally, but it never writes r to disk).
# Reported for shot 000 only; the scaled residual res_NNN.su is on disk for
# every shot either way.
for EXT in rvx rvz rtxx rtzz rtxz; do
    suop2 syn_000_${EXT}.su obs_000_${EXT}.su op=diff > dres_${EXT}.su
    MX=$(sumax < dres_${EXT}.su mode=max outpar=/dev/stdout 2>/dev/null)
    printf "  raw    dres_%-6s max=%s\n" "${EXT}" "${MX}"
done

# Scaled residual psi = Q r. res_000.su is written trace-major with one trace
# per (receiver, component); with a single receiver it is 5 traces, tagged by
# trid: 6=Vx 7=Vz 4=Txx 3=Tzz 2=Txz (computeResidual.c:24, typ=(trid-1)%8+1).
for PAIR in "rvx 6" "rvz 7" "rtxx 4" "rtzz 3" "rtxz 2"; do
    set -- ${PAIR}
    # only the components actually in COMP are present in res_000.su
    case ",${COMP}," in *",_$1,"*) ;; *) continue ;; esac
    suwind < res_000.su key=trid min=$2 max=$2 > sres_$1.su
    MX=$(sumax < sres_$1.su mode=max outpar=/dev/stdout 2>/dev/null)
    printf "  scaled sres_%-6s (trid=%s) max=%s\n" "$1" "$2" "${MX}"
done

# =========================================================
# Report
# =========================================================
echo ""
echo "========================================================="
echo " Done. Files in: $(pwd)"
echo ""
echo " Gradients (sum over ${NSHOT} shot(s)):"
echo "   grad_vp_raw.su     grad_vs_raw.su     grad_rho_raw.su"
echo "   grad_vp_scaled.su  grad_vs_scaled.su  grad_rho_scaled.su"
echo "   grad_vp_preco.su   grad_vs_preco.su   grad_rho_preco.su"
if [ "${SHOT_GRAD}" = "1" ]; then
    echo " Per-shot gradients (CIG panels):"
    echo "   $(ls -1 grad_shot*_raw.su 2>/dev/null | wc -l) files grad_shotNNN_{vp,vs,rho}_raw.su"
fi
echo ""
echo " True-model wavefield snapshots (observed-data run):"
ls -1 tsnap_s*.su 2>/dev/null | sed 's/^/   /' || echo "   (none)"
echo " Forward wavefield snapshots (background model):"
ls -1 fsnap_s*.su 2>/dev/null | sed 's/^/   /' || echo "   (none)"
echo " Adjoint wavefield snapshots:"
ls -1 asnap_s*.su 2>/dev/null | sed 's/^/   /' || echo "   (none)"
echo ""
echo " Data and residuals:"
echo "   obs_000_r*.su   syn_000_r*.su"
echo "   dres_r*.su      raw residual  r   = d_syn - d_obs"
echo "   sres_r*.su      scaled resid. psi = Q r   (also all in res_000.su)"
echo ""
echo " Plot examples:"
echo "   suximage < grad_vp_raw.su  d1=${DX} d2=${DX} perc=99 legend=1 &"
echo "   suximage < tsnap_svz.su    d1=${DX} d2=${DX} perc=99 &"
echo "   suximage < fsnap_svz.su    d1=${DX} d2=${DX} perc=99 &"
echo "   suximage < asnap_svz.su    d1=${DX} d2=${DX} perc=99 &"
echo "   suxgraph < dres_rvz.su  &"
echo "========================================================="
