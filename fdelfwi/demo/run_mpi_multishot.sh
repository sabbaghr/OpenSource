#!/bin/bash
#
# run_mpi_multishot.sh - Run multi-shot FWI gradient test with MPI + OpenMP
#
# Usage:
#   ./run_mpi_multishot.sh [nranks]    # Default: 2 ranks
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# MPI configuration
NRANKS=1

# OpenMP: use sensible thread count per rank
# Too many threads on a small grid causes overhead; 4-8 is good for 801x401
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Working directory
WORKDIR=mpi_multishot_test
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "========================================================="
echo " MPI MULTI-SHOT FWI GRADIENT TEST"
echo "========================================================="
echo " Working directory: $(pwd)"
echo " MPI ranks: ${NRANKS}"
echo " OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo ""

# =========================================================
# Model parameters
# =========================================================
BG_VP=3000
BG_VS=1500
BG_RHO=2300

ANOM_VP=2500
ANOM_VS=1150
ANOM_RHO=2000

Z_ANOM=1000
X_VP=2000
X_VS=1200
X_RHO=2800
VAR_ANOM=80

SIZEX=4000
SIZEZ=2000
DX=5
DZ=5

# Shot parameters - 4 shots for faster testing
NSHOTS=33
XSRC=400
DXSHOT=100
ZSRC=200

# Receiver parameters
XRCV1=400
XRCV2=3600
ZRCV=1800
DXRCV=20

TMOD=3.0
DTRCV=0.002

# =========================================================
# STEP 1: Create models (if not exist)
# =========================================================
if [ ! -f true_cp.su ]; then
    echo "--- Step 1: Creating models ---"

    makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DZ} \
        cp0=${BG_VP} cs0=${BG_VS} ro0=${BG_RHO} \
        orig=0,0 file_base=true.su \
        intt=diffr dtype=2 x=${X_VP}  z=${Z_ANOM} cp=${ANOM_VP}  cs=${BG_VS}   ro=${BG_RHO}  var=${VAR_ANOM} \
        intt=diffr dtype=2 x=${X_VS}  z=${Z_ANOM} cp=${BG_VP}    cs=${ANOM_VS} ro=${BG_RHO}  var=${VAR_ANOM} \
        intt=diffr dtype=2 x=${X_RHO} z=${Z_ANOM} cp=${BG_VP}    cs=${BG_VS}   ro=${ANOM_RHO} var=${VAR_ANOM} \
        verbose=1

    makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DZ} \
        cp0=${BG_VP} cs0=${BG_VS} ro0=${BG_RHO} \
        orig=0,0 file_base=init.su \
        verbose=1

    echo "  Models created"
else
    echo "--- Step 1: Using existing models ---"
fi

# =========================================================
# STEP 2: Create wavelet (if not exist)
# =========================================================
if [ ! -f wave.su ]; then
    echo "--- Step 2: Creating wavelet ---"
    makewave fp=10 dt=0.001 nt=4096 fmax=20 file_out=wave.su t0=0.10 verbose=1
else
    echo "--- Step 2: Using existing wavelet ---"
fi

# =========================================================
# STEP 3: Generate observed data (serial fdelmodc per shot)
# =========================================================
if [ ! -f obs_000_rvz.su ]; then
    echo "--- Step 3: Generating observed data for ${NSHOTS} shots ---"

    shot_idx=0
    for xsrc in $(seq ${XSRC} ${DXSHOT} $((XSRC + (NSHOTS-1)*DXSHOT))); do
        shot_str=$(printf "%03d" ${shot_idx})
        echo "  Shot ${shot_idx}: xsrc=${xsrc}m"

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
else
    echo "--- Step 3: Using existing observed data ---"
fi

# =========================================================
# Common parameters for gradient computation
# =========================================================
COMMON_PARAMS="file_cp=init_cp.su file_cs=init_cs.su file_den=init_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vz=1 rec_type_vx=1 rec_type_tzz=1 rec_type_txx=1 rec_type_p=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
    nshot=${NSHOTS} xsrc=${XSRC} dxshot=${DXSHOT} zsrc=${ZSRC} \
    ntaper=100 \
    left=4 right=4 top=4 bottom=4 \
    file_obs=obs chk_skipdt=500 chk_base=chk"

# MPI run command
MPIRUN="mpirun -np ${NRANKS} --bind-to none \
    -x OMP_NUM_THREADS=${OMP_NUM_THREADS} \
    -x OMP_PROC_BIND=spread \
    -x OMP_PLACES=threads \
    -x PATH \
    -x LD_LIBRARY_PATH"

# =========================================================
# STEP 4: Run MPI gradient computation
# =========================================================
echo ""
echo "========================================================="
echo " STEP 4: MPI FWI Gradient Computation"
echo "========================================================="

echo ""
# echo "--- Case 1: Vz component only, Velocity parameterization ---"
# rm -rf fwi_rank* chk_*.bin syn_*.su residual.su

# time ${MPIRUN} ${FDELFWI}/fwi_mpi_driver \
#     ${COMMON_PARAMS} \
#     file_grad=grad_vz_vel \
#     comp="_rvz" \
#     param=2

# echo ""
# echo "--- Case 2: Vx component only, Velocity parameterization ---"
# rm -rf fwi_rank* chk_*.bin syn_*.su residual.su

# time ${MPIRUN} ${FDELFWI}/fwi_mpi_driver \
#     ${COMMON_PARAMS} \
#     file_grad=grad_vx_vel \
#     comp="_rvx" \
#     param=2

echo ""
echo "--- Case 3: P component only, Velocity parameterization ---"
rm -rf fwi_rank* chk_*.bin syn_*.su residual.su

time ${MPIRUN} ${FDELFWI}/fwi_mpi_driver \
    ${COMMON_PARAMS} \
    file_grad=grad_p_vel \
    comp="_rp" \
    param=2

echo ""
echo "--- Case 3: All components (Vz+Vx+P), Velocity parameterization ---"
rm -rf fwi_rank* chk_*.bin syn_*.su residual.su

time ${MPIRUN} ${FDELFWI}/fwi_mpi_driver \
    ${COMMON_PARAMS} \
    file_grad=grad_all_vel \
    comp="_rvz,_rvx,_rp" \
    param=2

# =========================================================
# STEP 5: Validation
# =========================================================
echo ""
echo "========================================================="
echo " STEP 5: Validation"
echo "========================================================="

check_gradient() {
    local file=$1
    local name=$2
    if [ -f "$file" ]; then
        local size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
        echo "  [OK] $name: $file ($size bytes)"
    else
        echo "  [FAIL] $name: $file NOT FOUND"
        return 1
    fi
}

echo ""
echo "--- Gradient files ---"
check_gradient grad_vz_vel_vp.su "Vz-only Vp"
check_gradient grad_vz_vel_vs.su "Vz-only Vs"
check_gradient grad_vz_vel_rho.su "Vz-only rho"
check_gradient grad_all_vel_vp.su "All-comp Vp"
check_gradient grad_all_vel_vs.su "All-comp Vs"
check_gradient grad_all_vel_rho.su "All-comp rho"

echo ""
echo "--- Synthetic data files (last shot) ---"
last_shot=$(printf "%03d" $((NSHOTS-1)))
check_gradient syn_${last_shot}_rvz.su "Syn Vz"
check_gradient syn_${last_shot}_rvx.su "Syn Vx"
check_gradient syn_${last_shot}_rp.su "Syn P"

# =========================================================
# SUMMARY
# =========================================================
echo ""
echo "========================================================="
echo " SUMMARY"
echo "========================================================="
echo " MPI ranks: ${NRANKS}"
echo " OpenMP threads per rank: ${OMP_NUM_THREADS}"
echo " Total shots: ${NSHOTS}"
echo " Shots per rank: $((NSHOTS / NRANKS))"
echo ""
echo " Output gradients:"
echo "   grad_vz_vel_*.su   - Vz only, velocity parameterization"
echo "   grad_all_vel_*.su  - All components, velocity parameterization"
echo ""
echo " To visualize gradients:"
echo "   suximage < grad_vz_vel_vp.su perc=99"
echo "   suximage < grad_all_vel_vp.su perc=99"
echo ""
echo " Working directory: $(pwd)"
echo "========================================================="

# Cleanup temporary files
rm -rf fwi_rank* chk_*.bin
