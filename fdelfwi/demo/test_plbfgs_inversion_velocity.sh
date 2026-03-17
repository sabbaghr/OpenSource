#!/bin/bash
#
# test_plbfgs_inversion_velocity.sh - Preconditioned L-BFGS inversion test.
#
# Same model as test_lbfgs_inversion_velocity.sh but with:
#   - 10 shots (better illumination)
#   - algorithm=2 (PLBFGS with Shin/Métivier preconditioner)
#   - scaling=3 (range normalization, Δm = m_max - m_min)
#   - User-supplied bounds for normalization
#
# Compares with unpreconditioned L-BFGS (algorithm=1) for reference.
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=plbfgs_inversion_velocity
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " PLBFGS INVERSION TEST (Velocity param=2, scaling=3)"
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

makewave fp=10 dt=0.001 nt=1024 fmax=15 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=10 Hz, fmax=15 Hz, dt=1ms, nt=1024"

# =========================================================
# STEP 3: Generate observed data (10 shots)
# =========================================================
echo "--- Step 3: Generating observed data (10 shots) ---"

NSHOTS=10
XSRC_START=100
XSRC_END=1900
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
# Bounds for range normalization (scaling=3)
# =========================================================
VP_MIN=1500
VP_MAX=4500
VS_MIN=500
VS_MAX=2500
RHO_MIN=1500
RHO_MAX=2500

echo ""
echo "  Bounds: Vp=[${VP_MIN},${VP_MAX}] Vs=[${VS_MIN},${VS_MAX}] rho=[${RHO_MIN},${RHO_MAX}]"
echo "  Δm: Vp=$((VP_MAX-VP_MIN)) Vs=$((VS_MAX-VS_MIN)) rho=$((RHO_MAX-RHO_MIN))"

# =========================================================
# Common FWI parameters
# =========================================================
COMMON="file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.004 tmod=1.5 \
    xrcv1=100 xrcv2=1900 zrcv1=100 zrcv2=100 dxrcv=10 \
    xsrc=${XSRC_START} zsrc=10 nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=100 res_taper=100 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 \
    param=2 \
    comp=_rp \
    file_obs=obs \
    niter=10 nls_max=10 conv=1e-6 \
    write_iter=1 \
    verbose=2"

# =========================================================
# STEP 4a: L-BFGS without preconditioner (reference)
# =========================================================
echo ""
echo "========================================================="
echo " Run 1: L-BFGS (algorithm=1, scaling=3, no preconditioner)"
echo "========================================================="

${FDELFWI}/fwi_inversion \
    ${COMMON} \
    scaling=3 \
    vp_min=${VP_MIN} vp_max=${VP_MAX} \
    vs_min=${VS_MIN} vs_max=${VS_MAX} \
    rho_min=${RHO_MIN} rho_max=${RHO_MAX} \
    grad_taper=20 \
    chk_base=chk_lbfgs \
    file_grad=grad_lbfgs \
    algorithm=1 lbfgs_mem=5 \
    precond=0 2>&1 | tee lbfgs_log.txt

rm -f chk_lbfgs_*.bin

# Save convergence
cp iterate_LB.dat iterate_LBFGS.dat 2>/dev/null || true

# Save final model
for p in vp vs rho; do
    cp model_iter010_${p}.su model_lbfgs_${p}.su 2>/dev/null || \
    cp $(ls -t model_iter*_${p}.su 2>/dev/null | head -1) model_lbfgs_${p}.su 2>/dev/null || true
done

# =========================================================
# STEP 4b: PLBFGS with preconditioner
# =========================================================
echo ""
echo "========================================================="
echo " Run 2: PLBFGS (algorithm=2, scaling=3, Shin preconditioner)"
echo "========================================================="

${FDELFWI}/fwi_inversion \
    ${COMMON} \
    scaling=3 \
    vp_min=${VP_MIN} vp_max=${VP_MAX} \
    vs_min=${VS_MIN} vs_max=${VS_MAX} \
    rho_min=${RHO_MIN} rho_max=${RHO_MAX} \
    grad_taper=0 \
    chk_base=chk_plbfgs \
    file_grad=grad_plbfgs \
    algorithm=2 lbfgs_mem=5 \
    precond=1 precond_eps=0.01 2>&1 | tee plbfgs_log.txt

rm -f chk_plbfgs_*.bin

# Save convergence
cp iterate_PLB.dat iterate_PLBFGS.dat 2>/dev/null || true

# Save final model
for p in vp vs rho; do
    cp model_iter010_${p}.su model_plbfgs_${p}.su 2>/dev/null || \
    cp $(ls -t model_iter*_${p}.su 2>/dev/null | head -1) model_plbfgs_${p}.su 2>/dev/null || true
done

# =========================================================
# Summary
# =========================================================
echo ""
echo "========================================================="
echo " COMPARISON SUMMARY"
echo "========================================================="
echo ""
echo "L-BFGS convergence:"
cat iterate_LBFGS.dat 2>/dev/null || echo "  (not available)"
echo ""
echo "PLBFGS convergence:"
cat iterate_PLBFGS.dat 2>/dev/null || echo "  (not available)"
echo ""
echo "Output directory: $(pwd)"
echo "  L-BFGS models:  model_lbfgs_vp.su, model_lbfgs_vs.su, model_lbfgs_rho.su"
echo "  PLBFGS models:  model_plbfgs_vp.su, model_plbfgs_vs.su, model_plbfgs_rho.su"
echo "  Convergence:    iterate_LBFGS.dat, iterate_PLBFGS.dat"
echo "========================================================="
