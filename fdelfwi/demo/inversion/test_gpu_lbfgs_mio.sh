#!/bin/bash
#SBATCH --job-name=efwi_gpu_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=efwi_gpu_test_%j.log
#
# test_gpu_lbfgs_mio.sh — GPU vs CPU validation test on MIO (P100)
#
# Runs the same L-BFGS inversion on CPU and GPU, then compares:
#   1. Initial gradient norms (must match to ~4 significant digits)
#   2. Convergence trajectory (misfit reduction)
#   3. Final model files
#
# Model: 2000x800m, homogeneous background, 3 Gaussian anomalies
# 5 shots, elastic (ischeme=3, iorder=4), velocity param
#
# Usage:
#   sbatch test_gpu_lbfgs_mio.sh          # submit as batch job
#   bash test_gpu_lbfgs_mio.sh            # run interactively on GPU node
#

set -e

# =========================================================
# Environment
# =========================================================
module purge
module load compilers/gcc/13
module load libs/cuda/12.5
module load libs/mkl/2024.2
# Note: single GPU test, no MPI needed

# Auto-detect ROOT from script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=4

echo "========================================================="
echo " GPU vs CPU FWI VALIDATION TEST"
echo " Host: $(hostname)"
echo " GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo " Date: $(date)"
echo "========================================================="

# =========================================================
# Step 0: Build GPU binary (if not already built)
# =========================================================
echo ""
echo "--- Step 0: Building GPU binary ---"

cd ${FDELFWI}

# Check if fwi_gpu_inversion exists and is up to date
if [ ! -f fwi_gpu_inversion ] || [ fwi_inversion.c -nt fwi_gpu_inversion ] || \
   [ fdelfwi_gpu.cu -nt fwi_gpu_inversion ]; then
    echo "  Building fwi_gpu_inversion (sm_60 for P100)..."
    make gpu_inversion CUDA_ARCH="-arch=sm_60" 2>&1 | tee /tmp/gpu_build_$$.log | tail -20
    if [ ! -f fwi_gpu_inversion ]; then
        echo "  BUILD FAILED. See /tmp/gpu_build_$$.log"
        exit 1
    fi
    echo "  Build complete."
else
    echo "  fwi_gpu_inversion is up to date."
fi

# Also make sure CPU binary exists
if [ ! -f fwi_inversion ]; then
    echo "  Building fwi_inversion (CPU)..."
    make inversion 2>&1 | tail -5
fi

# =========================================================
# Step 1: Create test data (shared by CPU and GPU runs)
# =========================================================
TESTDIR=/tmp/efwi_gpu_test_$$
mkdir -p ${TESTDIR}
cd ${TESTDIR}

echo ""
echo "--- Step 1: Creating models in ${TESTDIR} ---"

DX=5
SIZEX=2000
SIZEZ=800
VP0=2000
VS0=1150
RO0=2000

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=model_bg.su verbose=0

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=diffr dtype=2 x=500 z=400 cp=2200 cs=${VS0} ro=${RO0} var=20 \
    intt=diffr dtype=2 x=1000 z=400 cp=${VP0} cs=1250 ro=${RO0} var=20 \
    intt=diffr dtype=2 x=1500 z=400 cp=${VP0} cs=${VS0} ro=2200 var=20 \
    orig=0,0 file_base=model_true.su verbose=0

echo "  Background: Vp=${VP0}, Vs=${VS0}, rho=${RO0}"
echo "  Grid: ${SIZEX}x${SIZEZ}m, dx=${DX}m"

# Wavelet
makewave fp=10 dt=0.001 nt=1024 fmax=15 file_out=wave.su t0=0.10 verbose=0

# Observed data (5 shots)
NSHOTS=5
XSRC_START=200
XSRC_END=1800
XSRC_STEP=$(( (XSRC_END - XSRC_START) / (NSHOTS - 1) ))

echo ""
echo "--- Step 2: Generating observed data (${NSHOTS} shots) ---"

for ((ishot=0; ishot<NSHOTS; ishot++)); do
    XSRC=$((XSRC_START + ishot * XSRC_STEP))

    fdelmodc \
        file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs_${ishot} \
        ischeme=3 iorder=4 src_type=1 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=0.004 tmod=1.5 verbose=0 \
        xrcv1=100 xrcv2=1900 zrcv1=100 zrcv2=100 dxrcv=10 \
        xsrc=${XSRC} zsrc=10 ntaper=100 \
        left=4 right=4 top=4 bottom=4

    SHOTID=$(printf "%03d" ${ishot})
    mv obs_${ishot}_rvx.su obs_${SHOTID}_rvx.su
    mv obs_${ishot}_rvz.su obs_${SHOTID}_rvz.su
    suop2 obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su op=sum | sugain scale=0.5 > obs_${SHOTID}_rp.su
    rm -f obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su

    echo "  Shot $((ishot+1))/${NSHOTS}: xsrc=${XSRC}"
done

# =========================================================
# Common FWI parameters
# =========================================================
FWI_PARAMS="file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.004 tmod=1.5 \
    xrcv1=100 xrcv2=1900 zrcv1=100 zrcv2=100 dxrcv=10 \
    xsrc=${XSRC_START} zsrc=10 nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=100 grad_taper=20 res_taper=100 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 \
    param=1 scaling=0 \
    comp=_rvz \
    file_obs=obs file_grad=gradient \
    algorithm=1 lbfgs_mem=5 \
    niter=3 nls_max=10 conv=1e-6 \
    write_iter=1 verbose=2"

# =========================================================
# Step 3: CPU inversion
# =========================================================
echo ""
echo "========================================================="
echo " Running CPU inversion (3 L-BFGS iterations)"
echo "========================================================="

mkdir -p cpu_run
cd cpu_run

# Copy shared data
ln -sf ../model_bg_*.su ../model_true_*.su ../wave.su ../obs_*.su .

T_CPU_START=$(date +%s%N)

${FDELFWI}/fwi_inversion ${FWI_PARAMS} chk_base=chk_cpu 2>&1 | tee cpu_output.log

T_CPU_END=$(date +%s%N)
T_CPU_MS=$(( (T_CPU_END - T_CPU_START) / 1000000 ))

echo ""
echo "CPU time: ${T_CPU_MS} ms"

# Save convergence
cp iterate_LB.dat ../cpu_convergence.dat 2>/dev/null || true

cd ..

# =========================================================
# Step 4: GPU inversion
# =========================================================
echo ""
echo "========================================================="
echo " Running GPU inversion (3 L-BFGS iterations)"
echo "========================================================="

mkdir -p gpu_run
cd gpu_run

# Copy shared data
ln -sf ../model_bg_*.su ../model_true_*.su ../wave.su ../obs_*.su .

T_GPU_START=$(date +%s%N)

${FDELFWI}/fwi_gpu_inversion ${FWI_PARAMS} chk_base=chk_gpu 2>&1 | tee gpu_output.log

T_GPU_END=$(date +%s%N)
T_GPU_MS=$(( (T_GPU_END - T_GPU_START) / 1000000 ))

echo ""
echo "GPU time: ${T_GPU_MS} ms"

# Save convergence
cp iterate_LB.dat ../gpu_convergence.dat 2>/dev/null || true

cd ..

# =========================================================
# Step 5: Compare results
# =========================================================
echo ""
echo "========================================================="
echo " COMPARISON: CPU vs GPU"
echo "========================================================="

echo ""
echo "--- Timing ---"
echo "  CPU: ${T_CPU_MS} ms"
echo "  GPU: ${T_GPU_MS} ms"
if [ ${T_GPU_MS} -gt 0 ]; then
    SPEEDUP=$(echo "scale=1; ${T_CPU_MS} / ${T_GPU_MS}" | bc 2>/dev/null || echo "N/A")
    echo "  Speedup: ${SPEEDUP}x"
fi

echo ""
echo "--- Convergence ---"
echo "  CPU:"
cat cpu_convergence.dat 2>/dev/null || echo "  (not found)"
echo ""
echo "  GPU:"
cat gpu_convergence.dat 2>/dev/null || echo "  (not found)"

echo ""
echo "--- Initial gradient (from verbose output) ---"
echo "  CPU initial misfit:"
grep "Initial misfit" cpu_run/cpu_output.log 2>/dev/null || echo "  (not found)"
echo "  GPU initial misfit:"
grep "Initial misfit" gpu_run/gpu_output.log 2>/dev/null || echo "  (not found)"

echo ""
echo "--- Gradient norms ---"
echo "  CPU:"
grep "Gradient:" cpu_run/cpu_output.log 2>/dev/null | head -1 || echo "  (not found)"
echo "  GPU:"
grep "Gradient:" gpu_run/gpu_output.log 2>/dev/null | head -1 || echo "  (not found)"

echo ""
echo "========================================================="
echo " Test completed. Results in: ${TESTDIR}"
echo "   cpu_run/ — CPU inversion output"
echo "   gpu_run/ — GPU inversion output"
echo "   cpu_convergence.dat, gpu_convergence.dat — iteration history"
echo "========================================================="

# Cleanup checkpoints (keep model files)
rm -rf cpu_run/chk_cpu_*.bin cpu_run/fwi_rank* gpu_run/chk_gpu_*.bin gpu_run/fwi_rank*
