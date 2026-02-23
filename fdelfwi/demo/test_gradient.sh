#!/bin/bash
#
# test_gradient.sh - Multi-component FWI gradient test.
#
# Pipeline:
#   1. Create true and initial 2-layer elastic models (makemod)
#   2. Create source wavelet (makewave)
#   3. Run fdelmodc with true model -> observed data (Vz, Vx, Tzz, Txx)
#   4. Run test_fwi_gradient for each test case:
#      - Different components (Vz, Vx, pressure, combined)
#      - Both parameterizations (Lame: lambda,mu,rho vs Velocity: Vp,Vs,rho)
#   5. Validate gradient output
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=../..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=gradient_test
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=8

echo "========================================================="
echo " FWI GRADIENT MULTI-COMPONENT TEST"
echo "========================================================="
echo " Working directory: $(pwd)"
echo ""

PASS=0
FAIL=0

check_nonzero() {
    local file=$1 label=$2
    if [ ! -f "$file" ]; then
        echo "  [$label] FAIL - file not found: $file"
        FAIL=$((FAIL+1))
        return
    fi
    local filesize
    filesize=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    if [ "$filesize" -le 240 ]; then
        echo "  [$label] FAIL - file is empty or too small: $file ($filesize bytes)"
        FAIL=$((FAIL+1))
        return
    fi
    echo "  [$label] PASS  ($file exists, $filesize bytes)"
    PASS=$((PASS+1))
}


# =========================================================
# Common parameters for all gradient runs
# =========================================================
# Snapshot parameters: enables saving both forward (synthetic) and adjoint wavefields
SNAP_PARAMS="tsnap1=0.0 tsnap2=3.0 dtsnap=0.1 sna_type_vz=1 sna_type_vx=1 sna_type_tzz=1 sna_type_txx=1"

COMMON_PARAMS="file_cp=init_cp.su file_cs=init_cs.su file_den=init_ro.su \
    file_src=wave.su file_rcv=syn file_snap=syn_snap \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vz=1 rec_type_vx=1 rec_type_tzz=1 rec_type_txx=1 rec_type_p=1 \
    dtrcv=0.002 tmod=3.0 verbose=1 \
    xrcv1=500 xrcv2=3500 zrcv1=1800 zrcv2=1800 dxrcv=20\
    xsrc=2000 zsrc=200 ntaper=150 \
    left=4 right=4 top=4 bottom=4 \
    file_obs=obs chk_skipdt=500 chk_base=chk \
    ${SNAP_PARAMS}"

run_gradient_test() {
    local label=$1
    local comp=$2
    local grad_prefix=$3
    local param=$4
    shift 4
    local extra_args="$*"

    local param_label
    if [ "$param" = "1" ]; then
        param_label="Lame (lambda, mu, rho)"
    else
        param_label="Velocity (Vp, Vs, rho)"
    fi

    echo ""
    echo "========================================================="
    echo " Case: $label"
    echo "   Components: $comp"
    echo "   Parameterization: $param_label"
    echo "   Gradient prefix: $grad_prefix"
    echo "========================================================="

    ${FDELFWI}/test_fwi_gradient \
        ${COMMON_PARAMS} \
        file_grad=${grad_prefix} \
        comp=${comp} \
        param=${param} \
        ${extra_args}

    # Cleanup checkpoint files between runs (large)
    rm -f chk_vx.bin chk_vz.bin chk_tzz.bin chk_txx.bin chk_txz.bin
}


# =========================================================
# STEP 1: Create models
# =========================================================
echo "--- Step 1: Creating models ---"

# Background parameters
BG_VP=3000
BG_VS=1500
BG_RHO=2300

# Anomaly parameters (percentage perturbation)
ANOM_VP=2500    # +10% Vp anomaly
ANOM_VS=1150    # +15% Vs anomaly
ANOM_RHO=2000   # +15% density anomaly

# Anomaly positions (three anomalies side by side at z=1000m)
Z_ANOM=1000
X_VP=1200       # Vp anomaly location
X_VS=2000       # Vs anomaly location (center)
X_RHO=2800      # Density anomaly location
VAR_ANOM=80     # Gaussian variance (controls size)

# True model: background + 3 separate anomalies
# Each anomaly only perturbs ONE parameter, others stay at background
makemod sizex=4000 sizez=2000 dx=5 dz=5 \
    cp0=${BG_VP} cs0=${BG_VS} ro0=${BG_RHO} \
    orig=0,0 file_base=true.su \
    intt=diffr dtype=2 x=${X_VP}  z=${Z_ANOM} cp=${ANOM_VP}  cs=${BG_VS}   ro=${BG_RHO}  var=${VAR_ANOM} \
    intt=diffr dtype=2 x=${X_VS}  z=${Z_ANOM} cp=${BG_VP}    cs=${ANOM_VS} ro=${BG_RHO}  var=${VAR_ANOM} \
    intt=diffr dtype=2 x=${X_RHO} z=${Z_ANOM} cp=${BG_VP}    cs=${BG_VS}   ro=${ANOM_RHO} var=${VAR_ANOM} \
    verbose=1

echo "  True model created with 3 separate anomalies at z=${Z_ANOM}m:"
echo "    - Vp anomaly at x=${X_VP}m:  Vp=${ANOM_VP} (background Vs, rho)"
echo "    - Vs anomaly at x=${X_VS}m:  Vs=${ANOM_VS} (background Vp, rho)"
echo "    - Rho anomaly at x=${X_RHO}m: rho=${ANOM_RHO} (background Vp, Vs)"

# Initial model: homogeneous (no anomalies)
makemod sizex=4000 sizez=2000 dx=5 dz=5 \
    cp0=${BG_VP} cs0=${BG_VS} ro0=${BG_RHO} \
    orig=0,0 file_base=init.su \
    verbose=1

echo "  Initial model created (homogeneous background)"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo ""
echo "--- Step 2: Creating wavelet ---"

makewave fp=15 dt=0.001 nt=4096 fmax=30 file_out=wave.su t0=0.10 verbose=1

echo "  Wavelet created: fp=15 Hz, fmax=30 Hz, dt=1ms, nt=4096"

# =========================================================
# STEP 3: Forward model with TRUE model -> observed data
# =========================================================
echo ""
echo "--- Step 3: Forward modeling with true model (observed data) ---"

fdelmodc \
    file_cp=true_cp.su file_cs=true_cs.su file_den=true_ro.su \
    file_src=wave.su \
    file_rcv=obs \
    file_snap=true_snap \
    ischeme=3 iorder=4 \
    src_type=1 \
    dtrcv=0.002 \
    tmod=3.0 \
    verbose=1 \
    xrcv1=500 xrcv2=3500 zrcv1=1800 zrcv2=1800 dxrcv=20\
    rec_type_vz=1 \
    rec_type_vx=1 \
    rec_type_tzz=1 \
    rec_type_txx=1 \
    rec_type_p=1 \
    xsrc=2000 zsrc=200 \
    ntaper=150 \
    left=4 right=4 top=4 bottom=4 \
    tsnap1=0.0 tsnap2=3.0 dtsnap=0.1 \
    sna_type_vz=1 sna_type_vx=1 sna_type_tzz=1 sna_type_txx=1

echo "  Observed data generated:"
ls -la obs_*.su 2>/dev/null || echo "  WARNING: No observed data files found"

# Compute hydrophone (pressure) as P = 0.5*(Txx + Tzz) for elastic
# This is needed because fdelmodc elastic records stress components,
# and we need to combine them for pressure-based inversion
echo "  Computing hydrophone data: P = 0.5*(Txx + Tzz)"
suop2 obs_rtzz.su obs_rtxx.su op=sum > obs_sum_tmp.su
sugain scale=0.5 < obs_sum_tmp.su > obs_rp.su
rm -f obs_sum_tmp.su
echo "  Hydrophone data created: obs_rp.su"

# =========================================================
# STEP 4: Run gradient tests
# =========================================================
echo ""
echo "========================================================="
echo " STEP 4: Running gradient test cases"
echo "========================================================="
echo ""
echo " Snapshots saved:"
echo "   - true_snap_*.su     : Forward wavefield with TRUE model"
echo "   - syn_snap_*.su      : Forward wavefield with INIT model (synthetic)"
echo "   - adj_snap_*_*.su    : Adjoint wavefield (backpropagation)"
echo ""

# # ---------------------------------------------------------
# # Test cases with LAME parameterization (param=1)
# # Output: _lam.su, _muu.su, _rho.su
# # ---------------------------------------------------------

# # Case 1: Vz only - Lame parameterization (with adjoint snapshots)
# run_gradient_test "Vz only (Lame)" "_rvz" "grad_vz_lame" 1 \
#     file_adj_snap=adj_snap_vz_lame

# # Case 2: Pressure (hydrophone) - Lame parameterization
# run_gradient_test "Pressure (Lame)" "_rp" "grad_p_lame" 1 \
#     file_adj_snap=adj_snap_p_lame

# ---------------------------------------------------------
# Test cases with VELOCITY parameterization (param=2)
# Output: _vp.su, _vs.su, _rho.su (with chain rule)
# ---------------------------------------------------------

# Case 3: Vz only - Velocity parameterization
run_gradient_test "Vz only (Velocity)" "_rvz" "grad_vz_vel" 2 \
    file_adj_snap=adj_snap_vz_vel

# Case 4: Pressure (hydrophone) - Velocity parameterization
run_gradient_test "Pressure (Velocity)" "_rp" "grad_p_vel" 2 \
    file_adj_snap=adj_snap_p_vel


# =========================================================
# STEP 5: Validate outputs
# =========================================================
echo ""
echo "========================================================="
echo " STEP 5: Validation"
echo "========================================================="

# Check observed data was produced
echo ""
echo "--- Observed data ---"
for comp in obs_rvz.su obs_rvx.su obs_rtzz.su obs_rtxx.su obs_rp.su; do
    check_nonzero "$comp" "Observed $comp"
done

# Check forward snapshots (true model)
echo ""
echo "--- Forward snapshots (true model) ---"
check_nonzero true_snap_svz.su "True snap Vz"
check_nonzero true_snap_svx.su "True snap Vx"

# # Check Lame gradient files (commented out - Lame tests not run)
# echo ""
# echo "--- Lame parameterization gradients ---"
# for prefix in grad_vz_lame grad_p_lame; do
#     check_nonzero ${prefix}_lam.su "lam ($prefix)"
#     check_nonzero ${prefix}_muu.su "muu ($prefix)"
#     check_nonzero ${prefix}_rho.su "rho ($prefix)"
# done

# Check Velocity gradient files
echo ""
echo "--- Velocity parameterization gradients ---"
for prefix in grad_vz_vel grad_p_vel; do
    check_nonzero ${prefix}_vp.su "Vp ($prefix)"
    check_nonzero ${prefix}_vs.su "Vs ($prefix)"
    check_nonzero ${prefix}_rho.su "rho ($prefix)"
done

# Check adjoint snapshot files
echo ""
echo "--- Adjoint snapshots ---"
check_nonzero adj_snap_vz_vel_svz.su "Adjoint Vz (Vz Velocity)"
check_nonzero adj_snap_p_vel_svz.su "Adjoint Vz (P Velocity)"

# =========================================================
# SUMMARY
# =========================================================
echo ""
echo "========================================================="
echo " SUMMARY"
echo "========================================================="
echo " PASSED: $PASS"
echo " FAILED: $FAIL"
echo " Working dir: $(pwd)"
if [ $FAIL -eq 0 ]; then
    echo " ALL TESTS PASSED"
else
    echo " *** SOME TESTS FAILED ***"
fi
echo "========================================================="

echo ""
echo "Output files preserved in: $(pwd)"
echo ""
echo "  SNAPSHOTS:"
echo "    true_snap_svz.su, true_snap_svx.su   : Forward (true model)"
echo "    syn_snap_svz.su, syn_snap_svx.su     : Forward (init model)"
echo "    adj_snap_*_svz.su, adj_snap_*_svx.su : Adjoint wavefields"
echo ""
echo "  GRADIENTS (Lame param=1):"
echo "    grad_vz_lame_lam.su  grad_vz_lame_muu.su  grad_vz_lame_rho.su"
echo "    grad_p_lame_lam.su   grad_p_lame_muu.su   grad_p_lame_rho.su"
echo ""
echo "  GRADIENTS (Velocity param=2):"
echo "    grad_vz_vel_vp.su  grad_vz_vel_vs.su  grad_vz_vel_rho.su"
echo "    grad_p_vel_vp.su   grad_p_vel_vs.su   grad_p_vel_rho.su"
echo ""
echo "  Chain rule for velocity parameterization:"
echo "    g_Vp = g_lambda * 2*rho*Vp"
echo "    g_Vs = 2*rho*Vs * (g_mu - 2*g_lambda)"
echo "    g_rho = g_rho_direct + g_lambda*(Vp^2 - 2*Vs^2) + g_mu*Vs^2"
echo ""
