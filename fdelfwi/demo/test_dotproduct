#!/bin/bash
#
# test_dotproduct.sh - Run the Claerbout dot product test for elastic adjoint.
#
# Creates a small homogeneous elastic model and runs the dot product test
# to verify x^T A^T y = y^T A x (Mora Appendix B).
#
# Usage:
#   ./test_dotproduct.sh              # density only (default)
#   ./test_dotproduct.sh 0            # all parameters
#   ./test_dotproduct.sh 1            # Vp only
#   ./test_dotproduct.sh 2            # Vs only
#   ./test_dotproduct.sh 3            # density only
#

set -e

# Which parameter to test (default: 3 = density only)
TEST_PARAM=${1:-3}

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=dotproduct_test
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " CLAERBOUT DOT PRODUCT TEST FOR ELASTIC ADJOINT"
echo "========================================================="
echo " Working directory: $(pwd)"
echo " test_param=${TEST_PARAM} (0=all, 1=Vp, 2=Vs, 3=density)"
echo ""

# =========================================================
# STEP 1: Create homogeneous elastic model
# =========================================================
echo "--- Step 1: Creating homogeneous model ---"

# 200x200 points, 5m spacing = 1000m x 1000m
# Vp=2000, Vs=1150, rho=2000 (safe CFL margin with dt=0.001)
makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    orig=0,0 file_base=model.su \
    verbose=0

echo "  Model: 201x201, dx=dz=5m, Vp=2000, Vs=1150, rho=2000"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=15 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=15 Hz, fmax=25 Hz, dt=1ms, nt=1024"

# =========================================================
# STEP 3: Run dot product test -- density only
# =========================================================
echo ""
echo "========================================================="
echo " Running dot product test (test_param=${TEST_PARAM})"
echo "========================================================="

${FDELFWI}/test_dotproduct \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=7 \
    rec_type_vz=1 \
    dtrcv=0.004 tmod=1.0 verbose=1 \
    xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
    xsrc=500 zsrc=200 ntaper=30 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_dp \
    pert_pct=0.01 seed=12345 \
    test_param=${TEST_PARAM} \
    comp=_rvz

# Cleanup large checkpoint files
rm -f chk_dp_*.bin chk_pert_*.bin

echo ""
echo "========================================================="
echo " Done. Working directory: $(pwd)"
echo "========================================================="
