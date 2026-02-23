#!/bin/bash
#
# test_mpi_inversion.sh - Multi-shot MPI FWI inversion test.
#
# Model: 1000x500m, dx=5m (201x101)
# 5 sources at z=10, receivers at z=490m
# fp=20 Hz, fmax=30 Hz
#
# Usage: bash test_mpi_inversion.sh [nproc]
#   nproc: number of MPI ranks (default: 5)
#

set -e

NPROC=${1:-5}

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=4

WORKDIR=mpi_inversion_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

# Source positions (5 shots across the model)
XSRC=(100 300 500 700 900)
ZSRC=10
NSHOTS=${#XSRC[@]}

echo "========================================================="
echo " MPI FWI INVERSION TEST"
echo "========================================================="
echo "  Model: 1000x500m, dx=5m (201x101)"
echo "  Shots: ${NSHOTS} at x=${XSRC[*]}, z=${ZSRC}"
echo "  MPI ranks: ${NPROC}"
echo "========================================================="

# =========================================================
# STEP 1: Models
# =========================================================
echo ""
echo "--- Step 1: Creating models ---"

makemod sizex=1000 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=1000 ro0=2000 \
    orig=0,0 file_base=true.su \
    intt=diffr dtype=2 x=500 z=250 cp=1700 cs=1000 ro=2000 var=20 \
    verbose=0

makemod sizex=1000 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=1000 ro0=2000 \
    orig=0,0 file_base=init.su \
    verbose=0

echo "  True: Vp anomaly 1700 at (500,250), background 2000"
echo "  Init: homogeneous Vp=2000"

# =========================================================
# STEP 2: Wavelet
# =========================================================
echo ""
echo "--- Step 2: Wavelet ---"
makewave fp=20 dt=0.001 nt=2048 fmax=30 file_out=wave.su t0=0.10 verbose=0
echo "  fp=20 Hz, fmax=30 Hz, nt=2048"

# =========================================================
# STEP 3: Observed data (true model, one shot at a time)
# =========================================================
echo ""
echo "--- Step 3: Forward modeling (true model, ${NSHOTS} shots) ---"

for ((i=0; i<${NSHOTS}; i++)); do
    xs=${XSRC[$i]}
    idx=$(printf "%03d" $i)
    echo "  Shot ${idx}: xsrc=${xs} zsrc=${ZSRC}"

    fdelmodc \
        file_cp=true_cp.su file_cs=true_cs.su file_den=true_ro.su \
        file_src=wave.su file_rcv=obs_shot \
        ischeme=3 iorder=4 src_type=1 \
        dtrcv=0.004 tmod=1.0 verbose=0 \
        xrcv1=50 xrcv2=950 zrcv1=490 zrcv2=490 dxrcv=10 \
        rec_type_vz=1 rec_type_vx=1 rec_type_p=1 rec_type_tzz=1 rec_type_txx=1 \
        xsrc=${xs} zsrc=${ZSRC} \
        ntaper=30 \
        left=4 right=4 top=4 bottom=4

    # Rename to shot-indexed naming: obs_000_rp.su, obs_001_rp.su, etc.
    for f in obs_shot_r*.su; do
        comp=$(echo "$f" | sed 's/obs_shot_//' | sed 's/\.su//')
        cp "$f" "obs_${idx}_${comp}.su"
    done

    # Compute observed hydrophone
    suop2 obs_${idx}_rtzz.su obs_${idx}_rtxx.su op=sum > obs_sum_tmp.su
    sugain scale=0.5 < obs_sum_tmp.su > obs_${idx}_rp.su
    rm -f obs_sum_tmp.su obs_shot_r*.su
done

echo "  All ${NSHOTS} shots generated"

# Verify files
echo ""
echo "--- Observed data files ---"
ls -la obs_*_rp.su

# =========================================================
# STEP 4: Run MPI L-BFGS inversion (pressure)
# =========================================================
echo ""
echo "========================================================="
echo " STEP 4: MPI L-BFGS inversion (3 iterations, pressure)"
echo "  ${NSHOTS} shots, ${NPROC} MPI ranks"
echo "========================================================="

# Build xsrc/zsrc comma-separated lists
XSRC_STR=$(IFS=,; echo "${XSRC[*]}")
ZSRC_STR=$(printf "${ZSRC}%.0s," $(seq 1 ${NSHOTS}))
ZSRC_STR=${ZSRC_STR%,}  # remove trailing comma

mpirun -np ${NPROC} ${FDELFWI}/fwi_mpi_inversion \
    file_cp=init_cp.su file_cs=init_cs.su file_den=init_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vz=1 rec_type_vx=1 rec_type_p=1 rec_type_tzz=1 rec_type_txx=1 \
    dtrcv=0.004 tmod=1.0 verbose=1 \
    xrcv1=50 xrcv2=950 zrcv1=490 zrcv2=490 dxrcv=10 \
    xsrc=${XSRC_STR} zsrc=${ZSRC_STR} ntaper=30 \
    left=4 right=4 top=4 bottom=4 \
    file_obs=obs comp=_rp \
    chk_skipdt=100 chk_base=chk \
    param=2 \
    niter=3 algorithm=1 lbfgs_mem=5 grad_taper=5 \
    file_grad=gradient write_iter=1

echo ""
echo "========================================================="
echo " RESULTS"
echo "========================================================="

if [ -f iterate_LB.dat ]; then
    echo ""
    echo "--- Convergence log ---"
    cat iterate_LB.dat
fi

echo ""
echo "--- Output files ---"
ls -la model_iter*.su gradient_*.su 2>/dev/null || echo "  No output files"

echo "========================================================="
