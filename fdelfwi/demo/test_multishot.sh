#!/bin/bash
#
# test_multishot.sh - Multi-shot FWI gradient test.
#
# Tests the fwi_driver with multiple shots using:
#   - Same 3-anomaly model from test_gradient.sh
#   - Multiple source positions
#   - Both Lamé and velocity parameterizations
#   - Individual components (Vz, Vx, hydrophone) and combined
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=../..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=multishot_test
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " MULTI-SHOT FWI GRADIENT TEST"
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
# Model parameters (same as test_gradient.sh)
# =========================================================
# Background parameters
BG_VP=3000
BG_VS=1500
BG_RHO=2300

# Anomaly parameters
ANOM_VP=2500
ANOM_VS=1150
ANOM_RHO=2000

# Anomaly positions (three anomalies side by side at z=1000m)
Z_ANOM=1000
X_VP=1200       # Vp anomaly location
X_VS=2000       # Vs anomaly location (center)
X_RHO=2800      # Density anomaly location
VAR_ANOM=80     # Gaussian variance (controls size)

# Grid parameters
SIZEX=4000
SIZEZ=2000
DX=5
DZ=5

# Shot parameters - 5 shots spread across the model
NSHOTS=5
XSRC1=800
XSRC2=3200
DXSRC=600
ZSRC=200

# Receiver parameters
XRCV1=500
XRCV2=3500
ZRCV=1800
DXRCV=20

# Time parameters
TMOD=3.0
DTRCV=0.002

# =========================================================
# STEP 1: Create models
# =========================================================
echo "--- Step 1: Creating models ---"

# True model: background + 3 separate anomalies
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DZ} \
    cp0=${BG_VP} cs0=${BG_VS} ro0=${BG_RHO} \
    orig=0,0 file_base=true.su \
    intt=diffr dtype=2 x=${X_VP}  z=${Z_ANOM} cp=${ANOM_VP}  cs=${BG_VS}   ro=${BG_RHO}  var=${VAR_ANOM} \
    intt=diffr dtype=2 x=${X_VS}  z=${Z_ANOM} cp=${BG_VP}    cs=${ANOM_VS} ro=${BG_RHO}  var=${VAR_ANOM} \
    intt=diffr dtype=2 x=${X_RHO} z=${Z_ANOM} cp=${BG_VP}    cs=${BG_VS}   ro=${ANOM_RHO} var=${VAR_ANOM} \
    verbose=1

echo "  True model created with 3 separate anomalies at z=${Z_ANOM}m:"
echo "    - Vp anomaly at x=${X_VP}m"
echo "    - Vs anomaly at x=${X_VS}m"
echo "    - Rho anomaly at x=${X_RHO}m"

# Initial model: homogeneous (no anomalies)
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DZ} \
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
# STEP 3: Generate observed data for all shots
# =========================================================
echo ""
echo "--- Step 3: Forward modeling with true model (observed data) ---"
echo "  Generating ${NSHOTS} shots from x=${XSRC1} to x=${XSRC2} (dx=${DXSRC})"

# Loop over shots - use _%03d format to match writeRec convention
shot_idx=0
for xsrc in $(seq ${XSRC1} ${DXSRC} ${XSRC2}); do
    shot_str=$(printf "%03d" ${shot_idx})
    echo "  Shot ${shot_idx}: xsrc=${xsrc}m (file suffix: _${shot_str})"

    fdelmodc \
        file_cp=true_cp.su file_cs=true_cs.su file_den=true_ro.su \
        file_src=wave.su \
        file_rcv=obs_${shot_str} \
        ischeme=3 iorder=4 \
        src_type=1 \
        dtrcv=${DTRCV} \
        tmod=${TMOD} \
        verbose=0 \
        xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
        rec_type_vz=1 \
        rec_type_vx=1 \
        rec_type_tzz=1 \
        rec_type_txx=1 \
        xsrc=${xsrc} zsrc=${ZSRC} \
        ntaper=100 \
        left=4 right=4 top=4 bottom=4

    # Compute hydrophone: P = 0.5*(Tzz + Txx)
    suop2 obs_${shot_str}_rtzz.su obs_${shot_str}_rtxx.su op=sum > obs_sum_tmp.su
    sugain scale=0.5 < obs_sum_tmp.su > obs_${shot_str}_rp.su
    rm -f obs_sum_tmp.su

    shot_idx=$((shot_idx + 1))
done

echo "  Observed data generated for ${NSHOTS} shots"
ls -la obs_*_rvz.su 2>/dev/null | head -5

# =========================================================
# Common parameters for gradient runs
# =========================================================
COMMON_PARAMS="file_cp=init_cp.su file_cs=init_cs.su file_den=init_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vz=1 rec_type_vx=1 rec_type_tzz=1 rec_type_txx=1 rec_type_p=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
    xsrc1=${XSRC1} xsrc2=${XSRC2} dxsrc=${DXSRC} zsrc=${ZSRC} \
    ntaper=100 \
    left=4 right=4 top=4 bottom=4 \
    file_obs=obs chk_skipdt=500 chk_base=chk"

run_multishot_test() {
    local label=$1
    local comp=$2
    local grad_prefix=$3
    local param=$4

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

    # Clean up previous run
    rm -rf fwi_rank* chk_*.bin

    ${FDELFWI}/fwi_driver \
        ${COMMON_PARAMS} \
        file_grad=${grad_prefix} \
        comp=${comp} \
        param=${param}

    # Cleanup checkpoint files
    rm -rf fwi_rank* chk_*.bin
}

# =========================================================
# STEP 4: Run multi-shot gradient tests
# =========================================================
echo ""
echo "========================================================="
echo " STEP 4: Running multi-shot gradient tests"
echo "========================================================="

# ---------------------------------------------------------
# Lamé parameterization tests (param=1)
# Output: _lam.su, _muu.su, _rho.su
# ---------------------------------------------------------

# Case 1: Vz only - Lamé
run_multishot_test "Vz only (Lame)" "_rvz" "grad_vz_lame" 1

# Case 2: Vx only - Lamé
run_multishot_test "Vx only (Lame)" "_rvx" "grad_vx_lame" 1

# Case 3: Hydrophone only - Lamé
run_multishot_test "Hydrophone (Lame)" "_rp" "grad_p_lame" 1

# Case 4: All components combined - Lamé
run_multishot_test "All components (Lame)" "_rvz,_rvx,_rp" "grad_all_lame" 1

# ---------------------------------------------------------
# Velocity parameterization tests (param=2)
# Output: _vp.su, _vs.su, _rho.su
# ---------------------------------------------------------

# Case 5: Vz only - Velocity
run_multishot_test "Vz only (Velocity)" "_rvz" "grad_vz_vel" 2

# Case 6: Vx only - Velocity
run_multishot_test "Vx only (Velocity)" "_rvx" "grad_vx_vel" 2

# Case 7: Hydrophone only - Velocity
run_multishot_test "Hydrophone (Velocity)" "_rp" "grad_p_vel" 2

# Case 8: All components combined - Velocity
run_multishot_test "All components (Velocity)" "_rvz,_rvx,_rp" "grad_all_vel" 2

# =========================================================
# STEP 5: Validate outputs
# =========================================================
echo ""
echo "========================================================="
echo " STEP 5: Validation"
echo "========================================================="

# Check observed data was produced
echo ""
echo "--- Observed data (first shot) ---"
for comp in obs_000_rvz.su obs_000_rvx.su obs_000_rp.su; do
    check_nonzero "$comp" "Observed $comp"
done

# Check Lame gradient files
echo ""
echo "--- Lame parameterization gradients ---"
for prefix in grad_vz_lame grad_vx_lame grad_p_lame grad_all_lame; do
    check_nonzero ${prefix}_lam.su "lam ($prefix)"
    check_nonzero ${prefix}_muu.su "muu ($prefix)"
    check_nonzero ${prefix}_rho.su "rho ($prefix)"
done

# Check Velocity gradient files
echo ""
echo "--- Velocity parameterization gradients ---"
for prefix in grad_vz_vel grad_vx_vel grad_p_vel grad_all_vel; do
    check_nonzero ${prefix}_vp.su "Vp ($prefix)"
    check_nonzero ${prefix}_vs.su "Vs ($prefix)"
    check_nonzero ${prefix}_rho.su "rho ($prefix)"
done

# =========================================================
# STEP 6: Compare gradient amplitudes
# =========================================================
echo ""
echo "========================================================="
echo " STEP 6: Gradient amplitude comparison"
echo "========================================================="

echo ""
echo "--- Lame parameterization: max|gradient| ---"
for gfile in grad_vz_lame_lam.su grad_vx_lame_lam.su grad_p_lame_lam.su grad_all_lame_lam.su; do
    if [ -f "$gfile" ]; then
        maxval=$(sumax < "$gfile" mode=abs 2>/dev/null | head -1)
        echo "  $gfile: $maxval"
    fi
done

echo ""
echo "--- Velocity parameterization: max|gradient| ---"
for gfile in grad_vz_vel_vp.su grad_vx_vel_vp.su grad_p_vel_vp.su grad_all_vel_vp.su; do
    if [ -f "$gfile" ]; then
        maxval=$(sumax < "$gfile" mode=abs 2>/dev/null | head -1)
        echo "  $gfile: $maxval"
    fi
done

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
echo ""
echo " Model setup:"
echo "   Grid: ${SIZEX}m x ${SIZEZ}m (dx=${DX}m)"
echo "   Shots: ${NSHOTS} (x=${XSRC1} to ${XSRC2}, dx=${DXSRC}m)"
echo "   Receivers: x=${XRCV1} to ${XRCV2} at z=${ZRCV}m"
echo ""
echo " Anomalies at z=${Z_ANOM}m:"
echo "   Vp: x=${X_VP}m (${ANOM_VP} m/s vs background ${BG_VP} m/s)"
echo "   Vs: x=${X_VS}m (${ANOM_VS} m/s vs background ${BG_VS} m/s)"
echo "   Rho: x=${X_RHO}m (${ANOM_RHO} kg/m3 vs background ${BG_RHO} kg/m3)"
echo ""
if [ $FAIL -eq 0 ]; then
    echo " ALL TESTS PASSED"
else
    echo " *** SOME TESTS FAILED ***"
fi
echo "========================================================="
echo ""
echo "Output files in: $(pwd)"
echo ""
echo "  GRADIENTS (Lame param=1):"
echo "    grad_vz_lame_*.su  - Vz component only"
echo "    grad_vx_lame_*.su  - Vx component only"
echo "    grad_p_lame_*.su   - Hydrophone only"
echo "    grad_all_lame_*.su - All components combined"
echo ""
echo "  GRADIENTS (Velocity param=2):"
echo "    grad_vz_vel_*.su   - Vz component only"
echo "    grad_vx_vel_*.su   - Vx component only"
echo "    grad_p_vel_*.su    - Hydrophone only"
echo "    grad_all_vel_*.su  - All components combined"
echo ""
