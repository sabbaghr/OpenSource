#!/bin/bash
#
# test_lbfgs_inversion_acoustic.sh - Acoustic FWI L-BFGS inversion test.
#
# Tests acoustic Vp inversion using L-BFGS with TOY2DAC-compatible settings:
#   - ischeme=1 (acoustic), param=2 (velocity parameterization)
#   - Pressure source (src_type=1), pressure recording (rec_type_p=1)
#   - 5 shots, 10 iterations, lbfgs_mem=5
#   - Absorbing boundaries on all sides
#   - Optional: scaling_mode=1 for TOY2DAC scaling
#
# Model: homogeneous Vp=2000 + Gaussian anomaly Vp=2200 at center
# Starting model: homogeneous Vp=2000
#
# Comparable to TOY2DAC ball template test case.
#
# Usage: bash test_lbfgs_inversion_acoustic.sh
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=../..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=acoustic_lbfgs_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " ACOUSTIC FWI L-BFGS INVERSION TEST"
echo "========================================================="

# =========================================================
# Parameters
# =========================================================
NX=201
NZ=101
DX=5
DZ=5
SIZEX=$(( (NX-1) * DX ))
SIZEZ=$(( (NZ-1) * DZ ))
VP_BG=2000
RHO_BG=2000
NSHOTS=5
NITER=10

# =========================================================
# Create models
# =========================================================
echo "--- Creating models ---"

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DZ} \
    cp0=${VP_BG} cs0=0 ro0=${RHO_BG} \
    intt=def x=500,500 z=250,250 cp=2200 ro=${RHO_BG} cs=0 \
    var=50,25 \
    orig=0,0 file_base=model_true.su verbose=0

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DZ} \
    cp0=${VP_BG} cs0=0 ro0=${RHO_BG} \
    orig=0,0 file_base=model_init.su verbose=0

echo "  True:    Vp=${VP_BG} + Gaussian anomaly +200 m/s"
echo "  Initial: Vp=${VP_BG} homogeneous"

# =========================================================
# Wavelet
# =========================================================
echo "--- Creating wavelet ---"

makewave fp=10 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

# =========================================================
# Generate observed data (all shots, true model)
# =========================================================
echo "--- Generating observed data (${NSHOTS} shots) ---"

XSRC_FIRST=200
XSRC_STEP=$(( (SIZEX - 2*XSRC_FIRST) / (NSHOTS-1) ))

for (( i=0; i<NSHOTS; i++ )); do
    XSRC=$(( XSRC_FIRST + i * XSRC_STEP ))
    FILENO=$(printf "%03d" $i)

    fdelmodc \
        file_cp=model_true_cp.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs_${FILENO} \
        ischeme=1 iorder=4 \
        src_type=1 src_orient=1 \
        rec_type_p=1 \
        dtrcv=0.004 tmod=1.0 verbose=0 \
        xrcv1=50 xrcv2=950 zrcv1=25 zrcv2=25 dxrcv=10 \
        xsrc=${XSRC} zsrc=10 \
        ntaper=30 left=4 right=4 top=1 bottom=4 2>/dev/null &

    # Throttle parallel jobs
    if (( (i+1) % 4 == 0 )); then wait; fi
done
wait

echo "  Generated ${NSHOTS} observed data files (obs_NNN_rp.su)"

# =========================================================
# Run FWI inversion
# =========================================================
echo ""
echo "========================================================="
echo " Running L-BFGS acoustic FWI inversion"
echo "   nshots=${NSHOTS}, niter=${NITER}, param=2 (Vp)"
echo "   scaling_mode=1 (TOY2DAC), lbfgs_mem=5"
echo "========================================================="

${FDELFWI}/fwi_inversion \
    file_cp=model_init_cp.su file_den=model_init_ro.su \
    file_src=wave.su file_rcv=syn file_snap=snap \
    file_obs=obs comp=_rp \
    ischeme=1 iorder=4 param=2 \
    src_type=1 src_orient=1 \
    rec_type_p=1 \
    dtrcv=0.004 tmod=1.0 verbose=2 \
    xrcv1=50 xrcv2=950 zrcv1=25 zrcv2=25 dxrcv=10 \
    nshots=${NSHOTS} \
    ntaper=30 left=4 right=4 top=1 bottom=4 \
    algorithm=1 niter=${NITER} lbfgs_mem=5 \
    nls_max=10 conv=1e-6 \
    scaling_mode=1 \
    grad_taper=15 \
    active_params=vp \
    write_iter=1 \
    file_grad=grad

echo ""
echo "========================================================="
echo " Inversion complete."
echo ""
echo " Output files:"
echo "   iterate_LB.dat        - convergence history"
echo "   model_iter_NNN_vp.su  - Vp model per iteration"
echo "   grad_initial_vp.su    - initial gradient"
echo ""
echo " Compare with TOY2DAC ball template results."
echo "========================================================="
