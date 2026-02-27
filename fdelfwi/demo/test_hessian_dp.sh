#!/bin/bash
#
# test_hessian_dp.sh - Hessian symmetry dot product test for elastic FWI.
#
# Verifies <H*dm1, dm2> = <dm1, H*dm2> where H = J^T J (Gauss-Newton Hessian).
# A ratio of 1.0 confirms that born_shot + adj_shot correctly implements the
# second-order adjoint state method.
#
# True model:    homogeneous elastic (Vp=2000, Vs=1150, rho=2000)
#                + Vp diffractor at (500,500)
# Starting model: homogeneous elastic (used for Hessian computation)
#
# The diffractor in the true model is NOT used — only the starting model
# matters, since the Hessian depends on the CURRENT model, not the data.
# We do need checkpoints from a forward run through the starting model.
#
# Usage:
#   ./test_hessian_dp.sh              # default: Vz component, iorder=4
#   ./test_hessian_dp.sh 4            # iorder=4
#   ./test_hessian_dp.sh 6            # iorder=6
#   ./test_hessian_dp.sh 8            # iorder=8
#

set -e

IORDER=${1:-4}

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi

export PATH=${BIN}:${PATH}

# Working directory
WORKDIR=hessian_dp_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " HESSIAN SYMMETRY DOT PRODUCT TEST"
echo " <H*dm1, dm2> = <dm1, H*dm2>  (ratio -> 1.0)"
echo "========================================================="
echo " Working directory: $(pwd)"
echo " iorder=${IORDER}"
echo ""

# =========================================================
# STEP 1: Create model (homogeneous elastic)
# =========================================================
echo "--- Step 1: Creating elastic model ---"

makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    orig=0,0 file_base=model.su \
    verbose=0

echo "  Model: Vp=2000, Vs=1150, rho=2000 (homogeneous)"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=15 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=15 Hz, fmax=25 Hz, dt=1ms, nt=1024"

# =========================================================
# STEP 3: Run Hessian symmetry test
# =========================================================
echo ""
echo "========================================================="
echo " Running Hessian symmetry test (iorder=${IORDER})"
echo "========================================================="

${FDELFWI}/test_hessian_dp \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=born \
    ischeme=3 iorder=${IORDER} src_type=7 \
    rec_type_vz=1 \
    dtrcv=0.004 tmod=1.0 verbose=1 \
    xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
    xsrc=500 zsrc=200 ntaper=30 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_hess \
    pert_pct=0.01 seed=12345 \
    comp=_rvz \
    param=1

# Cleanup large temporary files
rm -f chk_hess_*.bin
rm -f born*.su born*_combined.su

echo ""
echo "========================================================="
echo " Done. Working directory: $(pwd)"
echo "========================================================="
