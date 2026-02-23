#!/bin/bash
#
# test_inversion.sh - FWI inversion test.
#
# Model: 1000x500m, dx=5m (201x101)
# Source at (500,10), receivers at z=490m
# fp=20 Hz, fmax=30 Hz
#

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=../..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=8

WORKDIR=inversion_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "========================================================="
echo " FWI INVERSION TEST"
echo "========================================================="

# =========================================================
# STEP 1: Models
# =========================================================
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
# STEP 3: Observed data (true model)
# =========================================================
echo ""
echo "--- Step 3: Forward modeling (true model) ---"

fdelmodc \
    file_cp=true_cp.su file_cs=true_cs.su file_den=true_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=1 \
    dtrcv=0.004 tmod=1.0 verbose=0 \
    xrcv1=50 xrcv2=950 zrcv1=490 zrcv2=490 dxrcv=10 \
    rec_type_vz=1 rec_type_vx=1 rec_type_p=1 rec_type_tzz=1 rec_type_txx=1 \
    xsrc=500 zsrc=10 \
    ntaper=30 \
    left=4 right=4 top=4 bottom=4

# Rename to shot-indexed naming
for f in obs_r*.su; do
    newname=$(echo "$f" | sed 's/obs_/obs_000_/')
    cp "$f" "$newname"
done

# Compute observed hydrophone
suop2 obs_000_rtzz.su obs_000_rtxx.su op=sum > obs_sum_tmp.su
sugain scale=0.5 < obs_sum_tmp.su > obs_000_rp.su
rm -f obs_sum_tmp.su

echo "  Observed data generated"

# Quick sanity check
if cmp -s obs_000_rvz.su <(fdelmodc file_cp=init_cp.su file_cs=init_cs.su file_den=init_ro.su file_src=wave.su file_rcv=/dev/null ischeme=3 iorder=4 2>/dev/null); then
    echo "  WARNING: data may be zero"
fi

# =========================================================
# STEP 4: Run L-BFGS inversion (pressure)
# =========================================================
echo ""
echo "========================================================="
echo " STEP 4: L-BFGS inversion (3 iterations, pressure)"
echo "========================================================="

${FDELFWI}/fwi_inversion \
    file_cp=init_cp.su file_cs=init_cs.su file_den=init_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vz=1 rec_type_vx=1 rec_type_p=1 rec_type_tzz=1 rec_type_txx=1 \
    dtrcv=0.004 tmod=1.0 verbose=1 \
    xrcv1=50 xrcv2=950 zrcv1=490 zrcv2=490 dxrcv=10 \
    xsrc=500 zsrc=10 ntaper=30 \
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
