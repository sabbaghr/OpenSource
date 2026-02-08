#!/bin/bash
#
# test_dotproduct.sh - Run the Claerbout dot product test for elastic adjoint.
#
# Creates a small homogeneous elastic model and runs the dot product test
# to verify x^T A^T y = y^T A x (Mora Appendix B).
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=../..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=dotproduct_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " CLAERBOUT DOT PRODUCT TEST FOR ELASTIC ADJOINT"
echo "========================================================="
echo " Working directory: $(pwd)"
echo ""

# =========================================================
# STEP 1: Create homogeneous elastic model (small grid)
# =========================================================
echo "--- Step 1: Creating homogeneous model ---"

# Small grid for fast testing: 200x200 points, 5m spacing = 1000m x 1000m
makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=3000 cs0=1700 ro0=2000 \
    orig=0,0 file_base=model.su \
    verbose=0

echo "  Model: 200x200, dx=dz=5m, Vp=3000, Vs=1700, rho=2000"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=15 dt=0.001 nt=1024 fmax=30 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=15 Hz, fmax=30 Hz, dt=1ms, nt=1024"

# =========================================================
# STEP 3: Run dot product test -- Vz component
# =========================================================
echo ""
echo "========================================================="
echo " Test: Vz component (trid=7)"
echo "========================================================="

${FDELFWI}/test_dotproduct \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=syn file_snap=adj_snap \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vz=1 \
    dtrcv=0.002 tmod=1.0 verbose=1 \
    xrcv1=100 xrcv2=900 zrcv1=700 zrcv2=700 dxrcv=20 \
    xsrc=500 zsrc=200 ntaper=50 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_dp \
    pert_pct=0.005 seed=12345 \
    comp=_rvz \
    tsnap1=0.0 tsnap2=1.0 dtsnap=0.05 \
    sna_type_vz=1 sna_type_vx=1 sna_type_txx=1 sna_type_tzz=1 sna_type_txz=1

# Cleanup large checkpoint files
rm -f chk_dp_*.bin chk_pert_*.bin

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
