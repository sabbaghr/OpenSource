#!/bin/bash
#
# test_gradient_visual_acoustic.sh - Acoustic FWI gradient visualization.
#
# Single-shot gradient computation for acoustic Vp inversion (param=2).
# Uses a homogeneous model with a Gaussian velocity anomaly, matching
# the TOY2DAC ball template concept but in time domain.
#
# Outputs:
#   grad_initial_vp.su   - Vp gradient (raw, before scaling)
#   grad_initial_rho.su  - rho gradient (raw)
#
# Usage: bash test_gradient_visual_acoustic.sh
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=../..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=acoustic_gradient_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " ACOUSTIC FWI GRADIENT VISUALIZATION TEST"
echo "========================================================="
echo ""

# =========================================================
# Model parameters
# =========================================================
NX=201
NZ=101
DX=5
DZ=5
SIZEX=$(( (NX-1) * DX ))  # 1000m
SIZEZ=$(( (NZ-1) * DZ ))  # 500m
VP_BG=2000
RHO_BG=2000

echo "--- Creating models ---"

# True model: homogeneous + Gaussian Vp anomaly at center
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DZ} \
    cp0=${VP_BG} cs0=0 ro0=${RHO_BG} \
    intt=def x=500,500 z=250,250 cp=2200 ro=${RHO_BG} cs=0 \
    var=50,25 \
    orig=0,0 file_base=model_true.su verbose=0

# Starting model: homogeneous
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DZ} \
    cp0=${VP_BG} cs0=0 ro0=${RHO_BG} \
    orig=0,0 file_base=model_init.su verbose=0

echo "  True:    Vp=${VP_BG} + 200 m/s Gaussian anomaly at (500,250)m"
echo "  Initial: Vp=${VP_BG} homogeneous"
echo "  Grid: ${NX}x${NZ}, dx=dz=${DX}m"

# =========================================================
# Source wavelet
# =========================================================
echo "--- Creating wavelet ---"

makewave fp=10 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

echo "  Ricker fp=10 Hz, dt=1ms, nt=1024"

# =========================================================
# Step 1: Generate observed data (1 shot, true model)
# =========================================================
echo ""
echo "--- Step 1: Forward modeling (true model, 1 shot) ---"

fdelmodc \
    file_cp=model_true_cp.su file_den=model_true_ro.su \
    file_src=wave.su file_rcv=obs_000 \
    ischeme=1 iorder=4 \
    src_type=1 src_orient=1 \
    rec_type_p=1 \
    dtrcv=0.004 tmod=1.0 verbose=0 \
    xrcv1=50 xrcv2=950 zrcv1=25 zrcv2=25 dxrcv=10 \
    xsrc=500 zsrc=10 \
    ntaper=30 left=4 right=4 top=1 bottom=4 2>/dev/null

echo "  Generated obs_000_rp.su"

# =========================================================
# Step 2: Compute gradient (1 shot, initial model)
# =========================================================
echo ""
echo "--- Step 2: Computing acoustic gradient ---"

${FDELFWI}/fwi_inversion \
    file_cp=model_init_cp.su file_den=model_init_ro.su \
    file_src=wave.su file_rcv=syn file_snap=snap \
    file_obs=obs comp=_rp \
    ischeme=1 iorder=4 param=2 \
    src_type=1 src_orient=1 \
    rec_type_p=1 \
    dtrcv=0.004 tmod=1.0 verbose=1 \
    xrcv1=50 xrcv2=950 zrcv1=25 zrcv2=25 dxrcv=10 \
    xsrc=500 zsrc=10 \
    ntaper=30 left=4 right=4 top=1 bottom=4 \
    nshots=1 niter=0 \
    grad_taper=10 \
    file_grad=grad

echo ""
echo "========================================================="
echo " Output files:"
echo "   grad_initial_vp.su   - Vp gradient"
echo "   grad_initial_rho.su  - rho gradient"
echo ""
echo " To visualize (requires SU):"
echo "   supsimage < grad_initial_vp.su perc=99 title='Acoustic Vp gradient' | gv -"
echo "========================================================="
