#!/bin/bash
#
# test_gradient_visual_velocity.sh - Gradient visualization in velocity parameterization.
#
# Creates a model with 3 anomalies (Vp, Vs, rho at different x positions),
# generates observed data from the true model, then computes gradients
# using the background (smooth) model and saves as SU files.
#
# Output: grad_vp_raw.su, grad_vs_raw.su, grad_rho_raw.su
#         grad_vp_scaled.su, grad_vs_scaled.su, grad_rho_scaled.su
#         precond_vp.su, precond_vs.su, precond_rho.su             (diagonal of P)
#         precond_vp_vs.su, precond_vp_rho.su, precond_vs_rho.su   (off-diagonal of P)
#         grad_vp_preco.su, grad_vs_preco.su, grad_rho_preco.su    (P^{-1} g)
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=gradient_visual_velocity
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " GRADIENT VISUALIZATION TEST (Velocity param=2)"
echo "========================================================="

# =========================================================
# STEP 1: Create models
# =========================================================
echo "--- Step 1: Creating models ---"

# Domain: 2000m x 1000m, dx=dz=5m -> 401 x 201 points
DX=5
NX=401
NZ=201
SIZEX=2000
SIZEZ=800

# Background model (homogeneous)
VP0=2000
VS0=1150
RO0=2000

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=model_bg.su verbose=0

echo "  Background: Vp=${VP0}, Vs=${VS0}, rho=${RO0}"

# True model: 3 anomalies at different x locations
# makemod intt=def OVERWRITES all parameters below the interface,
# so specify full values (background + perturbation) for all 3 params.
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=def poly=0 x=400,500,600 z=400,400,400 cp=2200 cs=${VS0} ro=${RO0} gradcp=0 gradcs=0 \
    intt=def poly=0 x=900,1000,1100 z=400,400,400 cp=${VP0} cs=1250 ro=${RO0} gradcp=0 gradcs=0 \
    intt=def poly=0 x=1400,1500,1600 z=400,400,400 cp=${VP0} cs=${VS0} ro=2200 gradcp=0 gradcs=0 \
    orig=0,0 file_base=model_true.su verbose=0

echo "  True model: anomalies at x=500(Vp=2200), x=1000(Vs=1250), x=1500(rho=2200)"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=5 dt=0.001 nt=1024 fmax=10 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=5 Hz, fmax=10 Hz, dt=1ms, nt=1024"

# =========================================================
# STEP 3: Generate observed data from true model (5 shots)
# =========================================================
echo "--- Step 3: Generating observed data (5 shots) ---"

NSHOTS=5
XSRC_START=200
XSRC_END=1800
XSRC_STEP=$(( (XSRC_END - XSRC_START) / (NSHOTS - 1) ))

for ((ishot=0; ishot<NSHOTS; ishot++)); do
    XSRC=$((XSRC_START + ishot * XSRC_STEP))

    fdelmodc \
        file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs_${ishot} \
        ischeme=3 iorder=4 src_type=1 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=0.004 tmod=1.5 verbose=0 \
        xrcv1=100 xrcv2=1900 zrcv1=100 zrcv2=100 dxrcv=10 \
        xsrc=${XSRC} zsrc=10 ntaper=100 \
        left=4 right=4 top=4 bottom=4

    # Rename to zero-padded shot IDs (test_gradient_visual expects obs_NNN_*.su)
    SHOTID=$(printf "%03d" ${ishot})
    mv obs_${ishot}_rvx.su obs_${SHOTID}_rvx.su
    mv obs_${ishot}_rvz.su obs_${SHOTID}_rvz.su

    # Compute hydrophone: P = 0.5*(Tzz + Txx)
    suop2 obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su op=sum | sugain scale=0.5 > obs_${SHOTID}_rp.su
    rm -f obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su

    echo "  Shot $((ishot+1))/${NSHOTS}: xsrc=${XSRC}"
done

# =========================================================
# STEP 4: Run gradient computation with background model
# =========================================================
echo ""
echo "========================================================="
echo " Running gradient computation (param=2, velocity)"
echo "========================================================="

# Build shot position string
SHOT_XSRC=""
for ((ishot=0; ishot<NSHOTS; ishot++)); do
    XSRC=$((XSRC_START + ishot * XSRC_STEP))
    if [ -z "$SHOT_XSRC" ]; then
        SHOT_XSRC="${XSRC}"
    else
        SHOT_XSRC="${SHOT_XSRC},${XSRC}"
    fi
done

# Data component selection (change comp= to switch):
#   comp=_rvx    horizontal velocity
#   comp=_rvz    vertical velocity
#   comp=_rp     hydrophone (pressure)
# NOTE: test_gradient_visual supports one component at a time (no comma combos)

${FDELFWI}/test_gradient_visual \
    file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.004 tmod=1.5 verbose=1 \
    xrcv1=100 xrcv2=1900 zrcv1=100 zrcv2=100 dxrcv=10 \
    xsrc=${XSRC_START} zsrc=10 nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=100 grad_taper=100 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_gv \
    param=2 scaling=1 precond_eps=1e-3 \
    comp=_rp \
    file_obs=obs out_base=grad

# =========================================================
# Cleanup
# =========================================================
rm -f chk_gv_*.bin syn_*.su res_*.su

echo ""
echo "========================================================="
echo " Done. Output files in: $(pwd)"
echo " Raw gradient:           grad_vp_raw.su, grad_vs_raw.su, grad_rho_raw.su"
echo " Scaled gradient:        grad_vp_scaled.su, grad_vs_scaled.su, grad_rho_scaled.su"
echo " Preconditioner (diag):  precond_vp.su, precond_vs.su, precond_rho.su"
echo " Preconditioner (off):   precond_vp_vs.su, precond_vp_rho.su, precond_vs_rho.su"
echo " Preconditioned gradient: grad_vp_preco.su, grad_vs_preco.su, grad_rho_preco.su"
echo "========================================================="
