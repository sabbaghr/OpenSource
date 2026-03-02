#!/bin/bash
#
# test_wave_op_dp.sh - Wave operator adjoint dot product test.
#
# Verifies <Ax, y> = <x, A^T y> for the elastic wave operator
# using a homogeneous elastic model with:
#   - Fz force source (src_type=7)
#   - Vz recording (rec_type_vz=1)
#   - All absorbing (taper) boundaries
#
# Also saves adjoint wavefield snapshots for visualization.
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=../..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=wave_op_dp_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

# FD order to test (can be 4, 6, or 8)
IORDER=${1:-4}

echo "========================================================="
echo " WAVE OPERATOR ADJOINT DOT PRODUCT TEST (iorder=${IORDER})"
echo "========================================================="
echo " Working directory: $(pwd)"
echo ""

# =========================================================
# STEP 1: Create homogeneous elastic model
# =========================================================
echo "--- Step 1: Creating homogeneous model ---"

makemod sizex=500 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=1100 ro0=1800 \
    orig=0,0 file_base=model.su \
    verbose=0

echo "  Model: 100x100, dx=dz=5m, Vp=2000, Vs=1100, rho=1800"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=15 dt=0.001 nt=512 fmax=30 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=15 Hz, fmax=30 Hz, dt=1ms, nt=512"

# =========================================================
# STEP 3: Run wave operator dot product test
# =========================================================
echo ""
echo "========================================================="
echo " Running wave operator dot product test (iorder=${IORDER})"
echo "========================================================="

${FDELFWI}/test_wave_op_dp \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=syn file_snap=adj_snap \
    ischeme=3 iorder=${IORDER} src_type=7 \
    rec_type_vz=1 \
    dtrcv=0.001 tmod=0.5 verbose=1 \
    xrcv1=100 xrcv2=400 zrcv1=350 zrcv2=350 dxrcv=10 \
    xsrc=250 zsrc=100 ntaper=50 \
    left=4 right=4 top=4 bottom=4 \
    seed=42 \
    comp=_rvz \
    tsnap1=0.0 tsnap2=0.5 dtsnap=0.025 \
    sna_type_vz=1 sna_type_vx=1 sna_type_txx=1 sna_type_tzz=1 sna_type_txz=1

echo ""
echo "========================================================="
echo " Adjoint wavefield snapshots saved:"
echo "   adj_snap_svz.su  - Vz component"
echo "   adj_snap_svx.su  - Vx component"
echo "   adj_snap_stxx.su - Txx component"
echo "   adj_snap_stzz.su - Tzz component"
echo "   adj_snap_stxz.su - Txz component"
echo ""
echo " To visualize (requires SU):"
echo "   suxmovie < adj_snap_svz.su clip=1e-10 title='Adjoint Vz' &"
echo "   suxmovie < adj_snap_svx.su clip=1e-10 title='Adjoint Vx' &"
echo "========================================================="
echo ""
echo " Done. Working directory: $(pwd)"
echo "========================================================="
