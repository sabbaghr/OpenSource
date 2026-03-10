#!/bin/bash
#
# test_gradient_visual_lame.sh - Gradient visualization in Lamé parameterization.
#
# Same model as test_gradient_visual_velocity.sh but with param=1 (Lamé).
# Output: grad_lam_raw.su, grad_mu_raw.su, grad_rho_raw.su
#         grad_lam_scaled.su, grad_mu_scaled.su, grad_rho_scaled.su
#         precond_lam.su, precond_mu.su, precond_rho.su             (diagonal of P)
#         precond_lam_mu.su, precond_lam_rho.su, precond_mu_rho.su  (off-diagonal of P)
#         grad_lam_preco.su, grad_mu_preco.su, grad_rho_preco.su    (P^{-1} g)
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=gradient_visual_lame
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " GRADIENT VISUALIZATION TEST (Lamé param=1)"
echo "========================================================="

# =========================================================
# STEP 1: Create models
# =========================================================
echo "--- Step 1: Creating models ---"

DX=5
SIZEX=2000
SIZEZ=800
VP0=2000
VS0=1150
RO0=2000

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=model_bg.su verbose=0

echo "  Background: Vp=${VP0}, Vs=${VS0}, rho=${RO0}"

# makemod intt=def OVERWRITES all parameters below the interface.
# Must specify full values (background + perturbation) for all 3 params.
# Anomaly 1: Vp=2200 at x=500 (others unchanged)
# Anomaly 2: Vs=1250 at x=1000 (others unchanged)
# Anomaly 3: rho=2200 at x=1500 (others unchanged)
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=diffr dtype=2 x=500 z=400 cp=2200 cs=${VS0} ro=${RO0} var=20 \
    intt=diffr dtype=2 x=1000 z=400 cp=${VP0} cs=1250 ro=${RO0} var=20 \
    intt=diffr dtype=2 x=1500 z=400 cp=${VP0} cs=${VS0} ro=2200 var=20 \
    orig=0,0 file_base=model_true.su verbose=0

echo "  True model: anomalies at x=500(Vp=2200), x=1000(Vs=1250), x=1500(rho=2200)"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=5 dt=0.001 nt=1024 fmax=10 file_out=wave.su t0=0.10 verbose=0

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
echo " Running gradient computation (param=1, Lamé)"
echo "========================================================="

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
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 taper=100 \
    dtrcv=0.004 tmod=1.5 verbose=1 \
    xrcv1=100 xrcv2=1900 zrcv1=100 zrcv2=100 dxrcv=10 \
    xsrc=${XSRC_START} zsrc=10 nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=100 grad_taper=100 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_gv \
    param=1 scaling=1 precond_eps=1e-3 \
    comp=_rvx \
    file_obs=obs out_base=grad

# Cleanup
rm -f chk_gv_*.bin syn_*.su res_*.su

echo ""
echo "========================================================="
echo " Done. Output files in: $(pwd)"
echo " Raw gradient:           grad_lam_raw.su, grad_mu_raw.su, grad_rho_raw.su"
echo " Scaled gradient:        grad_lam_scaled.su, grad_mu_scaled.su, grad_rho_scaled.su"
echo " Preconditioner (diag):  precond_lam.su, precond_mu.su, precond_rho.su"
echo " Preconditioner (off):   precond_lam_mu.su, precond_lam_rho.su, precond_mu_rho.su"
echo " Preconditioned gradient: grad_lam_preco.su, grad_mu_preco.su, grad_rho_preco.su"
echo "========================================================="
