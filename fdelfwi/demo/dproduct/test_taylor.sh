#!/bin/bash
#
# test_taylor.sh - Run the Taylor gradient verification test for elastic FWI.
#
# True model:    homogeneous background (Vp=2000) + point diffractor in Vp
# Initial model: homogeneous background (Vp=2000)
#
# The diffractor creates scattered energy in d_obs that is absent in d_syn(m0),
# giving a meaningful misfit and gradient to test.
#
# Usage:
#   ./test_taylor.sh              # all parameters (default)
#   ./test_taylor.sh 1            # Vp only
#   ./test_taylor.sh 2            # Vs only
#   ./test_taylor.sh 3            # density only (velocity parameterization)
#   ./test_taylor.sh 4            # density Lame direct (rho only, lambda/mu fixed)
#

set -e

# Which parameter to test (default: 0 = all)
TEST_PARAM=${1:-0}

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi

export PATH=${BIN}:${PATH}

# Working directory
WORKDIR=taylor_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " TAYLOR GRADIENT VERIFICATION TEST"
echo "========================================================="
echo " Working directory: $(pwd)"
echo " test_param=${TEST_PARAM} (0=all, 1=Vp, 2=Vs, 3=rho_vel, 4=rho_Lame)"
echo ""

# =========================================================
# STEP 1: Create "true" model (background + Vp diffractor)
# =========================================================
echo "--- Step 1: Creating true model (background + diffractor) ---"

# Background: Vp=2000, Vs=1150, rho=2000
# Diffractor at x=500, z=500 with Vp=2500 (25% anomaly), width=2 grid points
makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    intt=diffr x=500 z=500 cp=2500 cs=1150 ro=2000 var=10 \
    orig=0,0 file_base=true_model.su \
    verbose=0

echo "  True model: Vp=2000 background + diffractor at (500,500) Vp=2500, width=10"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=15 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=15 Hz, fmax=25 Hz, dt=1ms, nt=1024"

# =========================================================
# STEP 3: Generate observed data from true model
# =========================================================
echo "--- Step 3: Generating observed data from true model ---"

${FDELFWI}/test_fdfwimodc \
    file_cp=true_model_cp.su file_cs=true_model_cs.su file_den=true_model_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=7 \
    rec_type_vz=1 \
    dtrcv=0.004 tmod=1.0 verbose=0 \
    xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
    xsrc=500 zsrc=200 ntaper=30 \
    left=4 right=4 top=4 bottom=4

echo "  Observed data generated: obs_000_rvz.su"

# =========================================================
# STEP 4: Create starting model m0 (homogeneous background)
# =========================================================
echo "--- Step 4: Creating starting model (homogeneous) ---"

makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    orig=0,0 file_base=start_model.su \
    verbose=0

echo "  Starting model: Vp=2000, Vs=1150, rho=2000 (no diffractor)"

# =========================================================
# STEP 5: Run Taylor test
# =========================================================
echo ""
echo "========================================================="
echo " Running Taylor test (test_param=${TEST_PARAM})"
echo "========================================================="

${FDELFWI}/test_taylor \
    file_cp=start_model_cp.su file_cs=start_model_cs.su file_den=start_model_ro.su \
    file_src=wave.su file_rcv=syn_taylor \
    file_obs=obs_000 \
    ischeme=3 iorder=4 src_type=7 \
    rec_type_vz=1 \
    dtrcv=0.004 tmod=1.0 verbose=1 \
    xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
    xsrc=500 zsrc=200 ntaper=30 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_taylor \
    pert_pct=0.01 seed=12345 \
    test_param=${TEST_PARAM} \
    comp=_rvz \
    neps=10 eps_min=1e-8 eps_max=1e-1

# Cleanup large temporary files
rm -f chk_taylor_*.bin chk_taylor_dummy_*.bin
rm -f cp_taylor_pert.su cs_taylor_pert.su ro_taylor_pert.su
rm -f residual_taylor*.su syn_taylor_pert_*.su syn_taylor_*.su

echo ""
echo "========================================================="
echo " Done. Working directory: $(pwd)"
echo "========================================================="
