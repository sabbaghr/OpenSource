#!/bin/bash
#
# test_wave_op_dp_acoustic.sh - Wave operator adjoint dot product test
#                                for ACOUSTIC (ischeme=1).
#
# Verifies <Ax, y> = <x, A^T y> for the acoustic wave operator
# using a homogeneous acoustic model.
#
# Tests pressure source (src_type=1) with pressure recording (rec_comp=p).
#
# SUCCESS criterion: relative error < 1e-5
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=../..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=wave_op_dp_acoustic_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " ACOUSTIC WAVE OPERATOR ADJOINT DOT PRODUCT TEST"
echo "========================================================="
echo ""

# =========================================================
# STEP 1: Create homogeneous acoustic model
# =========================================================
echo "--- Step 1: Creating homogeneous acoustic model ---"

makemod sizex=500 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=0 ro0=1800 \
    orig=0,0 file_base=model.su \
    verbose=0

echo "  Model: 100x100, dx=dz=5m, Vp=2000, Vs=0, rho=1800"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=15 dt=0.001 nt=512 fmax=30 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=15 Hz, fmax=30 Hz, dt=1ms, nt=512"

# =========================================================
# Test 1: Pressure source + pressure recording
# =========================================================
echo ""
echo "========================================================="
echo " Test 1: src_type=1 (P), rec_comp=p, ischeme=1"
echo "========================================================="

${FDELFWI}/test_wave_op_dp \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=1 iorder=4 src_type=1 \
    rec_type_p=1 \
    dtrcv=0.001 tmod=0.5 verbose=1 \
    xrcv1=100 xrcv2=400 zrcv1=350 zrcv2=350 dxrcv=10 \
    xsrc=250 zsrc=100 ntaper=50 \
    left=4 right=4 top=4 bottom=4 \
    seed=42 \
    rec_comp=p \
    comp=_rp

echo ""

# =========================================================
# Test 2: Force source + velocity recording
# =========================================================
echo "========================================================="
echo " Test 2: src_type=7 (Fz), rec_comp=vz, ischeme=1"
echo "========================================================="

${FDELFWI}/test_wave_op_dp \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=1 iorder=4 src_type=7 \
    rec_type_vz=1 \
    dtrcv=0.001 tmod=0.5 verbose=1 \
    xrcv1=100 xrcv2=400 zrcv1=350 zrcv2=350 dxrcv=10 \
    xsrc=250 zsrc=100 ntaper=50 \
    left=4 right=4 top=4 bottom=4 \
    seed=42 \
    rec_comp=vz \
    comp=_rvz

echo ""
echo "========================================================="
echo " Done. All acoustic dot product tests completed."
echo "========================================================="
