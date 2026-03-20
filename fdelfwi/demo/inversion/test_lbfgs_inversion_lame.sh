#!/bin/bash
#
# test_lbfgs_inversion_lame.sh - L-BFGS inversion test (Lame param).
#
# Same model/geometry as test_gradient_visual_lame.sh:
#   - Constant background + 3 anomalies (Vp, Vs, rho)
#   - 5 shots, elastic, Lame parameterization
#   - Brossier scaling (scaling=1)
#
# Runs fwi_inversion with verbose=2 for detailed linesearch diagnostics.
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=lbfgs_inversion_lame
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " L-BFGS INVERSION TEST (Lame param=1, scaling=1)"
echo "========================================================="

# =========================================================
# STEP 1: Create models
# =========================================================
echo "--- Step 1: Creating models ---"

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

# True model: 3 Gaussian anomalies at different x locations
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

    # Rename to zero-padded shot IDs (fwi_inversion expects obs_NNN_*.su)
    SHOTID=$(printf "%03d" ${ishot})
    mv obs_${ishot}_rvx.su obs_${SHOTID}_rvx.su
    mv obs_${ishot}_rvz.su obs_${SHOTID}_rvz.su

    # Compute hydrophone: P = 0.5*(Tzz + Txx)
    suop2 obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su op=sum | sugain scale=0.5 > obs_${SHOTID}_rp.su
    rm -f obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su

    echo "  Shot $((ishot+1))/${NSHOTS}: xsrc=${XSRC}"
done

# =========================================================
# STEP 4: Run L-BFGS inversion
# =========================================================
echo ""
echo "========================================================="
echo " Running L-BFGS inversion"
echo "   algorithm=1 (L-BFGS), niter=5, param=1 (Lame)"
echo "   scaling=1 (Brossier), verbose=2 (linesearch diagnostics)"
echo "========================================================="

# Data component selection (change comp= to switch):
#   comp=_rvx              horizontal velocity only
#   comp=_rvz              vertical velocity only
#   comp=_rp               hydrophone (pressure) only
#   comp=_rvx,_rvz         both velocity components
#   comp=_rvx,_rvz,_rp     all three components

${FDELFWI}/fwi_inversion \
    file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.004 tmod=1.5 \
    xrcv1=100 xrcv2=1900 zrcv1=100 zrcv2=100 dxrcv=10 \
    xsrc=${XSRC_START} zsrc=10 nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=100 grad_taper=60 res_taper=100 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_inv \
    param=1 scaling=1 \
    comp=_rp \
    file_obs=obs file_grad=gradient \
    algorithm=1 lbfgs_mem=5 \
    niter=5 nls_max=5 conv=1e-6 \
    write_iter=1 \
    verbose=2

# =========================================================
# Cleanup checkpoints
# =========================================================
rm -rf chk_inv_*.bin fwi_rank*

echo ""
echo "========================================================="
echo " Done. Output files in: $(pwd)"
echo " Model files:  model_iter???_lam.su, model_iter???_muu.su, model_iter???_rho.su"
echo " Gradient:     gradient_lam.su, gradient_muu.su, gradient_rho.su"
echo " Convergence:  iterate_LB.dat"
echo "========================================================="
