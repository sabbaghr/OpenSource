#!/bin/bash
#
# test_chk_skipdt.sh - Diagnostic: vary checkpoint interval to isolate j=0 skip error.
#
# Tests the Hessian DP ratio with different chk_skipdt values.
# If the j=0 density gradient skip is the cause:
#   - chk_skipdt=2000 (1 segment): error should vanish (no segment boundaries)
#   - chk_skipdt=100 (10 segments): moderate error
#   - chk_skipdt=10 (100 segments): much larger error
#

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi

export PATH=${BIN}:${PATH}
export OMP_NUM_THREADS=4

WORKDIR=hessian_dp_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

# Create model
makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    orig=0,0 file_base=model.su verbose=0

# Create wavelet
makewave fp=15 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

echo "=============================================="
echo " chk_skipdt sweep: tmod=1.0, rho-only (test_block=3)"
echo "=============================================="

for SKIPDT in 2000 100 50 10; do
    echo ""
    echo "--- chk_skipdt=${SKIPDT} ---"

    # Clean previous checkpoint and born files
    rm -f chk_hess_*.bin born*.su

    ${FDELFWI}/test_hessian_dp \
        file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
        file_src=wave.su file_rcv=born \
        ischeme=3 iorder=4 src_type=7 \
        rec_type_vz=1 \
        dtrcv=0.004 tmod=1.0 verbose=0 \
        xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
        xsrc=500 zsrc=200 ntaper=30 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=${SKIPDT} chk_base=chk_hess \
        pert_pct=0.01 seed=12345 \
        comp=_rvz param=1 \
        test_block=3 2>&1 | grep -E "Ratio|RESULT|dp1/dp_data|dp2/dp_data"

    rm -f chk_hess_*.bin born*.su
done

echo ""
echo "--- Same sweep for stiffness (test_block=4) ---"
echo ""

for SKIPDT in 2000 100 10; do
    echo "--- stiffness chk_skipdt=${SKIPDT} ---"
    rm -f chk_hess_*.bin born*.su

    ${FDELFWI}/test_hessian_dp \
        file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
        file_src=wave.su file_rcv=born \
        ischeme=3 iorder=4 src_type=7 \
        rec_type_vz=1 \
        dtrcv=0.004 tmod=1.0 verbose=0 \
        xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
        xsrc=500 zsrc=200 ntaper=30 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=${SKIPDT} chk_base=chk_hess \
        pert_pct=0.01 seed=12345 \
        comp=_rvz param=1 \
        test_block=4 2>&1 | grep -E "Ratio|RESULT|dp1/dp_data|dp2/dp_data"

    rm -f chk_hess_*.bin born*.su
done

echo ""
echo "=============================================="
echo " Done"
echo "=============================================="
