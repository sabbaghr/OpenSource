#!/bin/bash
#
# test_trn_inversion.sh - TRN (Truncated Newton) inversion test.
#
# Tests algorithm=4 (TRN) and algorithm=5 (Enriched) against L-BFGS baseline.
# Model: 1000x500m, dx=5m (201x101)
# Source at (500,10), receivers at z=490m
#

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi

export PATH=${BIN}:${PATH}
export OMP_NUM_THREADS=4

WORKDIR=trn_inversion_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "========================================================="
echo " TRN INVERSION TEST"
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
# STEP 3: Observed data (true model, explosive source, pressure receivers)
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
CWP=/rcp3/software/codes/cwp/bin
if [ -x "${CWP}/suop2" ]; then
    ${CWP}/suop2 obs_000_rtzz.su obs_000_rtxx.su op=sum > obs_sum_tmp.su
    ${CWP}/sugain scale=0.5 < obs_sum_tmp.su > obs_000_rp.su
    rm -f obs_sum_tmp.su
fi

echo "  Observed data generated (explosive source, pressure receivers)"

# =========================================================
# Common parameters
# =========================================================
COMMON_PARAMS="file_cp=init_cp.su file_cs=init_cs.su file_den=init_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vz=1 rec_type_vx=1 rec_type_p=1 rec_type_tzz=1 rec_type_txx=1 \
    dtrcv=0.004 tmod=1.0 verbose=1 \
    xrcv1=50 xrcv2=950 zrcv1=490 zrcv2=490 dxrcv=10 \
    xsrc=500 zsrc=10 ntaper=30 \
    left=4 right=4 top=4 bottom=4 \
    file_obs=obs comp=_rp \
    chk_skipdt=100 chk_base=chk \
    param=1 \
    grad_taper=5 write_iter=1"

# =========================================================
# STEP 4a: L-BFGS baseline (3 iterations)
# =========================================================
echo ""
echo "========================================================="
echo " STEP 4a: L-BFGS inversion (3 iterations)"
echo "========================================================="

${FDELFWI}/fwi_inversion \
    ${COMMON_PARAMS} \
    niter=3 algorithm=1 lbfgs_mem=5 \
    file_grad=gradient_lbfgs 2>&1 | tail -20

if [ -f iterate_LB.dat ]; then
    echo "--- L-BFGS convergence ---"
    cat iterate_LB.dat
fi

# Save L-BFGS results
mkdir -p results_lbfgs
mv -f model_iter*.su gradient_lbfgs*.su iterate_LB.dat results_lbfgs/ 2>/dev/null || true

# Cleanup working files
rm -rf fwi_rank* syn_*.su

# =========================================================
# STEP 4b: TRN inversion (3 iterations, niter_cg=3)
# =========================================================
echo ""
echo "========================================================="
echo " STEP 4b: TRN inversion (3 iterations, niter_cg=3)"
echo "========================================================="

${FDELFWI}/fwi_inversion \
    ${COMMON_PARAMS} \
    niter=3 algorithm=4 niter_cg=3 \
    file_grad=gradient_trn 2>&1 | tail -30

if [ -f iterate_TRN.dat ]; then
    echo "--- TRN convergence ---"
    cat iterate_TRN.dat
fi

# Save TRN results
mkdir -p results_trn
mv -f model_iter*.su gradient_trn*.su iterate_TRN.dat iterate_TRN_CG.dat results_trn/ 2>/dev/null || true

# Cleanup
rm -rf fwi_rank* syn_*.su

# =========================================================
# STEP 4c: Enriched inversion (5 iterations, enr_l=3, niter_cg=3)
# =========================================================
echo ""
echo "========================================================="
echo " STEP 4c: Enriched inversion (5 iterations, enr_l=3, niter_cg=3)"
echo "========================================================="

${FDELFWI}/fwi_inversion \
    ${COMMON_PARAMS} \
    niter=5 algorithm=5 niter_cg=3 enr_l=3 lbfgs_mem=5 \
    file_grad=gradient_enr 2>&1 | tail -30

if [ -f iterate_ENR.dat ]; then
    echo "--- Enriched convergence ---"
    cat iterate_ENR.dat
fi

# Save Enriched results
mkdir -p results_enr
mv -f model_iter*.su gradient_enr*.su iterate_ENR.dat results_enr/ 2>/dev/null || true

# Cleanup
rm -rf fwi_rank* syn_*.su

echo ""
echo "========================================================="
echo " DONE"
echo "========================================================="
