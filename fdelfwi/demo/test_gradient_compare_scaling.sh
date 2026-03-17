#!/bin/bash
#
# test_gradient_compare_scaling.sh - Compare gradient and preconditioned gradient
# for Brossier (scaling=1) vs range normalization (scaling=3).
#
# Uses 10 shots, velocity param=2, hydrophone data.
# Runs test_gradient_visual twice with different scaling, same data.
#

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}

WORKDIR=gradient_compare_scaling
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " GRADIENT SCALING COMPARISON (Brossier vs Range)"
echo "========================================================="

# =========================================================
# Model setup
# =========================================================
DX=5; SIZEX=2000; SIZEZ=800
VP0=2000; VS0=1150; RO0=2000

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=model_bg.su verbose=0

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=def poly=0 x=400,500,600 z=400,400,400 cp=2200 cs=${VS0} ro=${RO0} gradcp=0 gradcs=0 \
    intt=def poly=0 x=900,1000,1100 z=400,400,400 cp=${VP0} cs=1250 ro=${RO0} gradcp=0 gradcs=0 \
    intt=def poly=0 x=1400,1500,1600 z=400,400,400 cp=${VP0} cs=${VS0} ro=2200 gradcp=0 gradcs=0 \
    orig=0,0 file_base=model_true.su verbose=0

makewave fp=15 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

# =========================================================
# Generate observed data (10 shots)
# =========================================================
NSHOTS=10
XSRC_START=200
XSRC_END=1800
XSRC_STEP=$(( (XSRC_END - XSRC_START) / (NSHOTS - 1) ))

echo "--- Generating observed data (${NSHOTS} shots) ---"
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
    SHOTID=$(printf "%03d" ${ishot})
    mv obs_${ishot}_rvx.su obs_${SHOTID}_rvx.su
    mv obs_${ishot}_rvz.su obs_${SHOTID}_rvz.su
    suop2 obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su op=sum | sugain scale=0.5 > obs_${SHOTID}_rp.su
    rm -f obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su
    echo "  Shot $((ishot+1))/${NSHOTS}: xsrc=${XSRC}"
done

# Common parameters
COMMON="file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.004 tmod=1.5 verbose=1 \
    xrcv1=100 xrcv2=1900 zrcv1=100 zrcv2=100 dxrcv=10 \
    xsrc=${XSRC_START} zsrc=10 nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=100 grad_taper=0 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_gv \
    param=2 precond_eps=0.01 \
    comp=_rvz \
    file_obs=obs"

# =========================================================
# Run 1: Brossier scaling (scaling=1)
# =========================================================
echo ""
echo "========================================================="
echo " Run 1: scaling=1 (Brossier, m0 = mean|m|)"
echo "========================================================="

${FDELFWI}/test_gradient_visual ${COMMON} \
    scaling=1 \
    out_base=grad_s1

# Rename outputs
for f in grad_s1_*.su precond_*.su; do
    [ -f "$f" ] && mv "$f" "s1_${f}" 2>/dev/null || true
done

rm -f chk_gv_*.bin syn_*.su res_*.su

# =========================================================
# Run 2: Range scaling (scaling=3)
# =========================================================
echo ""
echo "========================================================="
echo " Run 2: scaling=3 (Range, m0 = Δm = m_max - m_min)"
echo "========================================================="

${FDELFWI}/test_gradient_visual ${COMMON} \
    scaling=3 \
    vp_min=1500 vp_max=4500 vs_min=500 vs_max=2500 rho_min=1500 rho_max=2500 \
    out_base=grad_s3

# Rename outputs
for f in grad_s3_*.su precond_*.su; do
    [ -f "$f" ] && mv "$f" "s3_${f}" 2>/dev/null || true
done

rm -f chk_gv_*.bin syn_*.su res_*.su

# =========================================================
# Summary
# =========================================================
echo ""
echo "========================================================="
echo " OUTPUT FILES"
echo "========================================================="
echo ""
echo " Brossier (scaling=1):"
echo "   Scaled gradient:        s1_grad_s1_vp_scaled.su  s1_grad_s1_vs_scaled.su  s1_grad_s1_rho_scaled.su"
echo "   Preconditioned:         s1_grad_s1_vp_preco.su   s1_grad_s1_vs_preco.su   s1_grad_s1_rho_preco.su"
echo "   Preconditioner (diag):  s1_precond_vp.su         s1_precond_vs.su         s1_precond_rho.su"
echo ""
echo " Range (scaling=3):"
echo "   Scaled gradient:        s3_grad_s3_vp_scaled.su  s3_grad_s3_vs_scaled.su  s3_grad_s3_rho_scaled.su"
echo "   Preconditioned:         s3_grad_s3_vp_preco.su   s3_grad_s3_vs_preco.su   s3_grad_s3_rho_preco.su"
echo "   Preconditioner (diag):  s3_precond_vp.su         s3_precond_vs.su         s3_precond_rho.su"
echo ""
echo " Both use: 10 shots, param=2, comp=_rp, theta=0.01, grad_taper=0"
echo " Norm preservation (Métivier eq 41) ensures ||g_preco|| = ||g_scaled||"
echo "========================================================="
