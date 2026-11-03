#!/bin/bash
#
# test_gradient_energy_norm.sh - Single-shot gradient with the energy-norm data
#                                weighting (data_weight=2) on ALL stress and
#                                velocity receivers.
#
# Objective function:
#     J = 1/2 || W_r dd ||^2 ,   Q = W_r^T W_r = diag( rho*I , S )
# where S is the 2D plane-strain isotropic compliance at each receiver.
# For the (Txx,Tzz) pair Q is NOT diagonal:
#     S_[Txx,Tzz] = 1/D * [ lam+2mu   -lam   ;  -lam   lam+2mu ],  D = 4 mu (lam+mu)
# and for Txz it is 1/mu.  This brings m/s (geophone) and Pa (stress/hydrophone)
# residuals into the same units (energy density, J/m^3) so they can all be
# back-propagated together in a single adjoint pass.
#
# Model: 5000 x 2000 m, dx=dz=5 (nx=1001, nz=401)
#   - homogeneous elastic background Vp=2000, Vs=1150, rho=2000
#   - two smooth velocity anomalies at z=800 m (r=250 m core, ~170 m taper,
#     rho unchanged):
#         x=2000 m  -> Vp=2200, Vs=1265   (+10%, fast)
#         x=3000 m  -> Vp=1800, Vs=1035   (-10%, slow)
#   - density-only reflector at z=1500 m: rho=2200, Vp/Vs unchanged
#
# The anomalies are made with makemod's 'elipse' body (hard rim) and then
# smoothed with the suite's 'smooth' utility applied to _cp/_cs ONLY, so the
# density reflector in _ro keeps its sharp step.  makemod itself has no
# Gaussian option (elipse.c:41-45 and diffraction.c:56-57 are both hard-edged).
#
# Background (starting) model: fully homogeneous, no anomalies, no reflector.
# A correct gradient must therefore light up lam/mu at the two blobs and rho
# at z=1500 m, with limited crosstalk between them.
#
# Acquisition: 1 shot at x=2500, z=10; receivers x=100..4900 dxrcv=10 at z=50.
# Receivers are inside the solid (mu > 0), which exercises the -lam/D
# off-diagonal coupling of Q. (For the seafloor degeneracy check, where
# Txz=0 and Txx=Tzz force Q to collapse onto the hydrophone weight
# 1/(lam+mu), use the OBN-style model instead.)
#
# Output:
#   grad_vp_raw.su, grad_vs_raw.su, grad_rho_raw.su            (raw gradient)
#   grad_vp_scaled.su, grad_vs_scaled.su, grad_rho_scaled.su   (g* = g.dm)
#   grad_vp_preco.su, grad_vs_preco.su, grad_rho_preco.su      (preconditioned)
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

WORKDIR=gradient_energy_norm
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " GRADIENT TEST - ENERGY-NORM WEIGHTING (data_weight=2)"
echo "========================================================="

# =========================================================
# Model parameters
# =========================================================
DX=5
SIZEX=5000
SIZEZ=2000
VP0=2000
VS0=1150
RO0=2000

# Anomalies (z=800 m, radius 250 m before smoothing)
ANOM_Z=800
ANOM_R=250
XANOM_FAST=2000;  VP_FAST=2200;  VS_FAST=1265    # +10%
XANOM_SLOW=3000;  VP_SLOW=1800;  VS_SLOW=1035    # -10%

# Density-only reflector
REFL_Z=1500
RO_REFL=2200

# Smoothing of _cp/_cs to turn the hard elipse rim into a smooth blob.
# nxsm/ntsm are in grid points; niter>1 makes the effective kernel wider and
# essentially Gaussian (repeated boxcar -> central limit).
# nsm=21 (105 m) with niter=8 gives an effective sigma ~86 m, i.e. a taper of
# roughly +/-170 m around the rim, while preserving 99% of the peak amplitude
# (verified: 2200 -> 2197.7).  Widening this further eats the peak: at
# nsm=31/niter=8 the peak drops to 2171.
SM_NX=21
SM_NZ=21
SM_ITER=8

# =========================================================
# Acquisition
# =========================================================
XSRC=2500
ZSRC=10
XRCV1=100
XRCV2=4900
ZRCV=50
DXRCV=10
TMOD=3.0
DTRCV=0.004

# =========================================================
# Gradient / preconditioner parameters
# =========================================================
PARAM=2                   # 1 = Lame (lam,mu,rho), 2 = velocity (Vp,Vs,rho)
DATA_WEIGHT=2             # 0 = off, 1 = Brossier RMS, 2 = energy norm (this test)
GRAD_TAPER=20             # grid points of source-region taper
PRECOND=5
PRECOND_EPS=0.01
SMOOTH_GRAD=10
SMOOTH_HESS=10

VP_MIN=1500;  VP_MAX=4500
VS_MIN=500;   VS_MAX=2500
RHO_MIN=1500; RHO_MAX=2500

# All velocity + all stress components, back-propagated together.
COMP=_rvx,_rvz,_rtxx,_rtzz,_rtxz

# =========================================================
# STEP 1: Create models
# =========================================================
echo "--- Step 1: Creating models ---"

# Starting model: homogeneous, no anomalies, no reflector
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=model_bg.su verbose=0

# True model: density reflector first (it defines everything below z=REFL_Z),
# then the two velocity bodies above it.
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=def poly=0 x=0,${SIZEX} z=${REFL_Z},${REFL_Z} \
        cp=${VP0} cs=${VS0} ro=${RO_REFL} \
    intt=elipse poly=0 x=${XANOM_FAST} z=${ANOM_Z} var=${ANOM_R},${ANOM_R} \
        cp=${VP_FAST} cs=${VS_FAST} ro=${RO0} \
    intt=elipse poly=0 x=${XANOM_SLOW} z=${ANOM_Z} var=${ANOM_R},${ANOM_R} \
        cp=${VP_SLOW} cs=${VS_SLOW} ro=${RO0} \
    orig=0,0 file_base=model_sharp.su verbose=0

# Smooth the velocity blobs only; leave _ro (and its sharp reflector) alone.
smooth file_in=model_sharp_cp.su file_out=model_true_cp.su \
    ntsm=${SM_NZ} nxsm=${SM_NX} niter=${SM_ITER} verbose=0
smooth file_in=model_sharp_cs.su file_out=model_true_cs.su \
    ntsm=${SM_NZ} nxsm=${SM_NX} niter=${SM_ITER} verbose=0
cp model_sharp_ro.su model_true_ro.su

echo "  Background:  Vp=${VP0}, Vs=${VS0}, rho=${RO0} (homogeneous)"
echo "  Anomaly A :  x=${XANOM_FAST} z=${ANOM_Z}  Vp=${VP_FAST} Vs=${VS_FAST} (fast)"
echo "  Anomaly B :  x=${XANOM_SLOW} z=${ANOM_Z}  Vp=${VP_SLOW} Vs=${VS_SLOW} (slow)"
echo "  Reflector :  z=${REFL_Z}  rho=${RO_REFL} (density only)"

# Sanity check: the smoothed _cp/_cs must still bracket the requested
# anomalies, and _ro must still contain the sharp step.
echo "--- Model ranges after smoothing ---"
for F in model_true_cp model_true_cs model_true_ro; do
    MN=$(sumax < ${F}.su mode=min outpar=/dev/stdout 2>/dev/null)
    MX=$(sumax < ${F}.su mode=max outpar=/dev/stdout 2>/dev/null)
    printf "  %-16s min=%s  max=%s\n" "${F}" "${MN}" "${MX}"
done

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=14 fmin=2 fmax=25 dt=0.001 nt=2048 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=14 Hz, fmin=2 Hz, fmax=25 Hz"

# =========================================================
# STEP 3: Generate observed data (1 shot, all components)
# =========================================================
echo "--- Step 3: Generating observed data (1 shot) ---"

fdelmodc \
    file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
    file_src=wave.su file_rcv=obs_0 \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} ntaper=100 \
    left=4 right=4 top=1 bottom=4

# fdelmodc does not zero-pad fileno; fwi expects obs_000_*
for EXT in rvx rvz rtxx rtzz rtxz; do
    mv obs_0_${EXT}.su obs_000_${EXT}.su
done
# Hydrophone (not used by COMP here, but kept for the degeneracy cross-check)
suop2 obs_000_rtzz.su obs_000_rtxx.su op=sum | sugain scale=0.5 > obs_000_rp.su

echo "  Shot at xsrc=${XSRC}, components: rvx rvz rtxx rtzz rtxz (+ rp)"

# =========================================================
# STEP 4: Gradient with the energy norm
# =========================================================
echo ""
echo "========================================================="
echo " Gradient: param=${PARAM}, data_weight=${DATA_WEIGHT} (energy norm)"
echo "   comp=${COMP}"
echo "========================================================="

${FDELFWI}/test_gradient_visual \
    file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=1 dxshot=0 \
    ntaper=100 grad_taper=${GRAD_TAPER} \
    left=4 right=4 top=1 bottom=4 \
    chk_skipdt=100 chk_base=chk_en \
    param=${PARAM} scaling=1 precond=${PRECOND} precond_eps=${PRECOND_EPS} \
    smooth_grad=${SMOOTH_GRAD} smooth_hess=${SMOOTH_HESS} \
    vp_min=${VP_MIN} vp_max=${VP_MAX} \
    vs_min=${VS_MIN} vs_max=${VS_MAX} \
    rho_min=${RHO_MIN} rho_max=${RHO_MAX} \
    data_weight=${DATA_WEIGHT} \
    comp=${COMP} \
    file_obs=obs out_base=grad

# =========================================================
# Cleanup
# =========================================================
rm -f chk_en_*.bin syn_*.su res_*.su

echo ""
echo "========================================================="
echo " Done. Output files in: $(pwd)"
echo "   Raw gradient:    grad_vp_raw.su    grad_vs_raw.su    grad_rho_raw.su"
echo "   Scaled gradient: grad_vp_scaled.su grad_vs_scaled.su grad_rho_scaled.su"
echo "   Preconditioned:  grad_vp_preco.su  grad_vs_preco.su  grad_rho_preco.su"
echo ""
echo " Expected: Vp/Vs gradients peak at (x=2000,z=800) and (x=3000,z=800)"
echo "           with opposite signs; rho gradient peaks along z=1500 m."
echo "========================================================="
