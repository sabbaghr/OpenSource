#!/bin/bash
#
# test_crosstalk_comparison.sh — Compare crosstalk effects across preconditioner modes.
#
# Runs test_gradient_visual 3 times with the same model/data but different
# preconditioner modes to visualize inter-parameter crosstalk correction:
#
#   precond=0 : No preconditioner (raw gradient)
#   precond=1 : Yang/Shin diagonal (illumination only, no crosstalk correction)
#   precond=3 : Wang block-diagonal (illumination + crosstalk correction)
#
# The model has 3 isolated anomalies at z=400m:
#   x=1000: Vp=2200 (only Vp perturbation)
#   x=500:  Vs=1250 (only Vs perturbation)
#   x=1500: rho=2200 (only rho perturbation)
#
# If crosstalk is present, the Vp gradient will show artifacts at x=500 and x=1500
# (Vs and rho anomaly positions), and vice versa. The block-diagonal preconditioner
# should suppress these cross-parameter artifacts.
#

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=4

# =========================================================
# Model
# =========================================================
DX=5
SIZEX=2000
SIZEZ=800
VP0=2000
VS0=1150
RO0=2000

# =========================================================
# Acquisition
# =========================================================
NSHOTS=10
XSRC_START=200
XSRC_END=1800
XSRC_STEP=$(( (XSRC_END - XSRC_START) / (NSHOTS - 1) ))
ZSRC=10
ZRCV=20

# Scaling bounds
VP_MIN=1500;  VP_MAX=4500
VS_MIN=500;   VS_MAX=2500
RHO_MIN=1500; RHO_MAX=2500

COMP=_rp
PRECOND_EPS=0.005

# =========================================================
# Setup working directory and create shared model/data
# =========================================================
WORKDIR=crosstalk_comparison
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "========================================================="
echo " CROSSTALK COMPARISON TEST"
echo " 3 preconditioner modes × same model/data"
echo "========================================================="

# Create models
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=model_bg.su verbose=0

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=diffr dtype=2 x=1000 z=400 cp=2200 cs=${VS0} ro=${RO0} var=20 \
    intt=diffr dtype=2 x=500  z=400 cp=${VP0} cs=1250 ro=${RO0} var=20 \
    intt=diffr dtype=2 x=1500 z=400 cp=${VP0} cs=${VS0} ro=2200 var=20 \
    orig=0,0 file_base=model_true.su verbose=0

# Create wavelet
makewave fmin=2 fp=10 dt=0.001 nt=1024 fmax=20 file_out=wave.su t0=0.10 verbose=0

# Generate observed data
echo "--- Generating observed data (${NSHOTS} shots) ---"
for ((ishot=0; ishot<NSHOTS; ishot++)); do
    XSRC=$((XSRC_START + ishot * XSRC_STEP))
    fdelmodc \
        file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs_${ishot} \
        ischeme=3 iorder=4 src_type=1 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=0.004 tmod=1.5 verbose=0 \
        xrcv1=100 xrcv2=1900 zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=10 \
        xsrc=${XSRC} zsrc=${ZSRC} ntaper=100 \
        left=4 right=4 top=4 bottom=4
    SHOTID=$(printf "%03d" ${ishot})
    mv obs_${ishot}_rvx.su obs_${SHOTID}_rvx.su
    mv obs_${ishot}_rvz.su obs_${SHOTID}_rvz.su
    suop2 obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su op=sum | sugain scale=0.5 > obs_${SHOTID}_rp.su
    rm -f obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su
    echo "  Shot $((ishot+1))/${NSHOTS}: xsrc=${XSRC}"
done

# Common parameters for test_gradient_visual
COMMON="file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.004 tmod=1.5 verbose=1 \
    xrcv1=100 xrcv2=1900 zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=10 \
    xsrc=${XSRC_START} zsrc=${ZSRC} nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=100 grad_taper=0 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_gv \
    param=2 scaling=1 \
    precond_eps=${PRECOND_EPS} precond_blend=0 \
    srt_radius=0 srt_filtsize=0 \
    vp_min=${VP_MIN} vp_max=${VP_MAX} \
    vs_min=${VS_MIN} vs_max=${VS_MAX} \
    rho_min=${RHO_MIN} rho_max=${RHO_MAX} \
    comp=${COMP} file_obs=obs"

# =========================================================
# Run 1: No preconditioner (precond=0)
# =========================================================
echo ""
echo "========================================================="
echo " Run 1/3: precond=0 (no preconditioner)"
echo "========================================================="

${FDELFWI}/test_gradient_visual ${COMMON} \
    precond=0 out_base=p0_grad

rm -f chk_gv_*.bin syn_*.su res_*.su

# Rename outputs
for f in p0_grad_*.su precond_*.su; do
    [ -f "$f" ] && mv "$f" "p0_${f#p0_}" 2>/dev/null || true
done

# =========================================================
# Run 2: Yang diagonal (precond=1)
# =========================================================
echo ""
echo "========================================================="
echo " Run 2/3: precond=1 (Yang/Shin diagonal)"
echo "========================================================="

${FDELFWI}/test_gradient_visual ${COMMON} \
    precond=1 out_base=p1_grad

rm -f chk_gv_*.bin syn_*.su res_*.su

for f in p1_grad_*.su precond_*.su; do
    [ -f "$f" ] && mv "$f" "p1_${f#p1_}" 2>/dev/null || true
done

# =========================================================
# Run 3: Wang block-diagonal (precond=3)
# =========================================================
echo ""
echo "========================================================="
echo " Run 3/3: precond=3 (Wang block-diagonal)"
echo "========================================================="

${FDELFWI}/test_gradient_visual ${COMMON} \
    precond=3 out_base=p3_grad

rm -f chk_gv_*.bin syn_*.su res_*.su

for f in p3_grad_*.su precond_*.su; do
    [ -f "$f" ] && mv "$f" "p3_${f#p3_}" 2>/dev/null || true
done

# =========================================================
# Summary
# =========================================================
echo ""
echo "========================================================="
echo " CROSSTALK COMPARISON — OUTPUT FILES"
echo "========================================================="
echo ""
echo " No preconditioner (precond=0):"
echo "   p0_grad_vp_scaled.su  p0_grad_vs_scaled.su  p0_grad_rho_scaled.su"
echo "   p0_grad_vp_preco.su   p0_grad_vs_preco.su   p0_grad_rho_preco.su"
echo ""
echo " Yang diagonal (precond=1):"
echo "   p1_grad_vp_scaled.su  p1_grad_vs_scaled.su  p1_grad_rho_scaled.su"
echo "   p1_grad_vp_preco.su   p1_grad_vs_preco.su   p1_grad_rho_preco.su"
echo ""
echo " Wang block-diagonal (precond=3):"
echo "   p3_grad_vp_scaled.su  p3_grad_vs_scaled.su  p3_grad_rho_scaled.su"
echo "   p3_grad_vp_preco.su   p3_grad_vs_preco.su   p3_grad_rho_preco.su"
echo ""
echo " Look for crosstalk: the Vp gradient should only show the anomaly at x=1000."
echo " Artifacts at x=500 (Vs) or x=1500 (rho) indicate parameter leakage."
echo " The block-diagonal (p3) should have less crosstalk than diagonal (p1)."
echo "========================================================="
