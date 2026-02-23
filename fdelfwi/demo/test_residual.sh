#!/bin/bash
#
# test_residual.sh - Verify that the data residual makes sense.
#
# 1. Create true model (with Vp anomaly) and initial model (homogeneous)
# 2. Forward model both -> compare data
# 3. Compute residual and check it's non-zero and sensible
#

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=8

WORKDIR=residual_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "========================================================="
echo " RESIDUAL DIAGNOSTIC TEST"
echo "========================================================="

# =========================================================
# Model: 1000x500m, dx=dz=5m (201x101)
# Source at top center (500, 10)
# Receivers at bottom (z=490), x=50..950, every 10m
# =========================================================

# STEP 1: Models
echo "--- Creating models ---"

makemod sizex=1000 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=1000 ro0=2000 \
    orig=0,0 file_base=true.su \
    intt=diffr dtype=2 x=500 z=250 cp=1700 cs=1000 ro=2000 var=20 \
    verbose=0

makemod sizex=1000 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=1000 ro0=2000 \
    orig=0,0 file_base=init.su \
    verbose=0

echo "  True model: Vp anomaly 1700 m/s at (500,250), background 2000"
echo "  Init model: homogeneous Vp=2000"

# STEP 2: Wavelet (low frequency, safe for dx=5, Vs=1000)
echo ""
echo "--- Creating wavelet ---"
makewave fp=20 dt=0.001 nt=2048 fmax=30 file_out=wave.su t0=0.10 verbose=0
echo "  Wavelet: fp=20 Hz, fmax=35 Hz, nt=2048"

# STEP 3: Forward model with TRUE model
echo ""
echo "--- Forward modeling: TRUE model ---"
fdelmodc \
    file_cp=true_cp.su file_cs=true_cs.su file_den=true_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=1 \
    dtrcv=0.004 tmod=1.0 verbose=0 \
    xrcv1=50 xrcv2=950 zrcv1=490 zrcv2=490 dxrcv=10 \
    rec_type_vz=1 rec_type_p=1 rec_type_tzz=1 rec_type_txx=1 \
    xsrc=500 zsrc=10 \
    ntaper=30 \
    left=4 right=4 top=4 bottom=4

echo "  True data generated"

# STEP 4: Forward model with INITIAL model
echo ""
echo "--- Forward modeling: INITIAL model ---"
fdelmodc \
    file_cp=init_cp.su file_cs=init_cs.su file_den=init_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    dtrcv=0.004 tmod=1.0 verbose=0 \
    xrcv1=50 xrcv2=950 zrcv1=490 zrcv2=490 dxrcv=10 \
    rec_type_vz=1 rec_type_p=1 rec_type_tzz=1 rec_type_txx=1 \
    xsrc=500 zsrc=10 \
    ntaper=30 \
    left=4 right=4 top=4 bottom=4

echo "  Synthetic data generated"

# STEP 5: Compute hydrophone for both
echo ""
echo "--- Computing hydrophone P = 0.5*(Txx+Tzz) ---"
suop2 obs_rtzz.su obs_rtxx.su op=sum > tmp.su
sugain scale=0.5 < tmp.su > obs_rp.su; rm tmp.su

suop2 syn_rtzz.su syn_rtxx.su op=sum > tmp.su
sugain scale=0.5 < tmp.su > syn_rp.su; rm tmp.su

echo "  Hydrophone computed for both"

# STEP 6: Data statistics
echo ""
echo "========================================================="
echo " DATA STATISTICS"
echo "========================================================="

echo ""
echo "--- Vz component ---"
echo "  True (obs):"
sumax < obs_rvz.su 2>&1 | head -3
echo "  Init (syn):"
sumax < syn_rvz.su 2>&1 | head -3

echo ""
echo "--- Pressure (hydrophone) ---"
echo "  True (obs):"
sumax < obs_rp.su 2>&1 | head -3
echo "  Init (syn):"
sumax < syn_rp.su 2>&1 | head -3

# STEP 7: Compute residual = obs - syn
echo ""
echo "========================================================="
echo " RESIDUAL = obs - syn"
echo "========================================================="

echo ""
echo "--- Vz residual ---"
suop2 obs_rvz.su syn_rvz.su op=diff > res_rvz.su
echo "  Residual stats:"
sumax < res_rvz.su 2>&1 | head -3

echo ""
echo "--- Pressure residual ---"
suop2 obs_rp.su syn_rp.su op=diff > res_rp.su
echo "  Residual stats:"
sumax < res_rp.su 2>&1 | head -3

# STEP 8: Check that residual is nonzero
echo ""
echo "========================================================="
echo " SANITY CHECKS"
echo "========================================================="

# Check file sizes
obs_size=$(stat -c%s obs_rvz.su)
syn_size=$(stat -c%s syn_rvz.su)
res_size=$(stat -c%s res_rvz.su)
echo "  obs_rvz.su: $obs_size bytes"
echo "  syn_rvz.su: $syn_size bytes"
echo "  res_rvz.su: $res_size bytes"

# Check if obs and syn are identical (they shouldn't be!)
if cmp -s obs_rvz.su syn_rvz.su; then
    echo "  *** WARNING: obs and syn are IDENTICAL! No anomaly signal! ***"
else
    echo "  GOOD: obs and syn differ (anomaly is visible in data)"
fi

if cmp -s obs_rp.su syn_rp.su; then
    echo "  *** WARNING: obs_rp and syn_rp are IDENTICAL! ***"
else
    echo "  GOOD: pressure obs and syn differ"
fi

echo ""
echo "Output files in: $(pwd)"
echo "  obs_rvz.su, syn_rvz.su, res_rvz.su (Vz)"
echo "  obs_rp.su,  syn_rp.su,  res_rp.su  (pressure)"
echo ""
echo "To visualize: suximage < res_rp.su"
echo "========================================================="
