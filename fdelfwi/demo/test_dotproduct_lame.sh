#!/bin/bash
#
# test_dotproduct_lame.sh - Run the Claerbout dot product test in Lame space.
#
# Creates lambda, mu, rho SU files from a homogeneous elastic model
# using test_readmodel_lame, then runs the dot product test with lame_input=1.
#
# Usage:
#   ./test_dotproduct_lame.sh              # density only (default)
#   ./test_dotproduct_lame.sh 0            # all parameters (lam, mu, rho)
#   ./test_dotproduct_lame.sh 1            # lambda only
#   ./test_dotproduct_lame.sh 2            # mu only
#   ./test_dotproduct_lame.sh 3            # density only
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
WORKDIR=dotproduct_lame_test
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " CLAERBOUT DOT PRODUCT TEST - LAME PARAMETERIZATION"
echo "========================================================="
echo " Working directory: $(pwd)"
echo " test_param=${TEST_PARAM} (0=all, 1=lam, 2=mu, 3=density)"
echo ""

# =========================================================
# STEP 1: Create homogeneous elastic model + wavelet
# =========================================================
echo "--- Step 1: Creating homogeneous model + wavelet ---"

makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    orig=0,0 file_base=model.su \
    verbose=0

makewave fp=15 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

echo "  Model: 201x201, dx=dz=5m, Vp=2000, Vs=1150, rho=2000"
echo "  Wavelet: fp=15 Hz, fmax=25 Hz, dt=1ms, nt=1024"

# =========================================================
# STEP 2: Convert to Lame parameters using test_readmodel_lame
# =========================================================
echo ""
echo "--- Step 2: Converting to Lame parameters ---"

# test_readmodel_lame reads velocity model, computes lambda/mu, writes
# _test_lam.su and _test_mu.su.
${FDELFWI}/test_readmodel_lame \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su ischeme=3 iorder=4 src_type=7 \
    xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
    xsrc=500 zsrc=200 ntaper=30 \
    left=4 right=4 top=4 bottom=4 \
    tmod=1.0 dtrcv=0.004 file_rcv=syn_dummy \
    verbose=1

echo ""
echo "  Created _test_lam.su and _test_mu.su"
echo "  Density file: model_ro.su (unchanged)"

# =========================================================
# STEP 3: Run dot product test in Lame mode
# =========================================================
echo ""
echo "========================================================="
echo " Running dot product test (Lame mode, test_param=${TEST_PARAM})"
echo "========================================================="

${FDELFWI}/test_dotproduct \
    file_cp=_test_lam.su file_cs=_test_mu.su file_den=model_ro.su \
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
    comp=_rvz \
    lame_input=1

# Cleanup large checkpoint files
rm -f chk_dp_*.bin chk_pert_*.bin

echo ""
echo "========================================================="
echo " Done. Working directory: $(pwd)"
echo "========================================================="
