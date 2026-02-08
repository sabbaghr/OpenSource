#!/bin/bash
#
# test_wave_op_dp_all.sh - Wave operator DP test for ALL source/receiver permutations.
#
# Tests all 15 combinations of:
#   Source types: Fz (7), Fx (6), P/explosive (1)
#   Recording types: vz, vx, txx, tzz, p
#
# Usage: ./test_wave_op_dp_all.sh [iorder]
#   iorder: FD order (4, 6, or 8). Default: 4
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=../..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=wave_op_dp_all
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

# FD order to test
IORDER=${1:-4}

echo "============================================================"
echo " WAVE OPERATOR DP TEST - ALL PERMUTATIONS (iorder=${IORDER})"
echo "============================================================"
echo " Working directory: $(pwd)"
echo ""

# =========================================================
# Create homogeneous elastic model (shared by all tests)
# =========================================================
echo "--- Creating homogeneous model ---"

makemod sizex=500 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=1100 ro0=1800 \
    orig=0,0 file_base=model.su \
    verbose=0

echo "  Model: 100x100, dx=dz=5m, Vp=2000, Vs=1100, rho=1800"

# =========================================================
# Create wavelet
# =========================================================
echo "--- Creating wavelet ---"

makewave fp=15 dt=0.001 nt=512 fmax=30 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=15 Hz, fmax=30 Hz, dt=1ms, nt=512"
echo ""

# =========================================================
# Define permutations
# =========================================================
# Source types and their names
SRC_TYPES="7 6 1"
SRC_NAMES="Fz Fx P"

# Recording components
REC_COMPS="vz vx txx tzz p"

# Common parameters
COMMON="file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=${IORDER} \
    rec_type_vz=1 \
    dtrcv=0.001 tmod=0.5 verbose=0 \
    xrcv1=100 xrcv2=400 zrcv1=350 zrcv2=350 dxrcv=10 \
    xsrc=250 zsrc=100 ntaper=50 \
    left=4 right=4 top=4 bottom=4 \
    seed=42 comp=_rvz"

# =========================================================
# Run all permutations
# =========================================================
PASS_COUNT=0
FAIL_COUNT=0
TOTAL=0
RESULTS=""

echo "============================================================"
echo " Running all source/receiver permutations..."
echo "============================================================"
echo ""
printf "%-6s %-6s %-24s %-24s %-12s %s\n" "Src" "Rec" "<Ax,y>" "<x,A^Ty>" "epsilon" "Status"
printf "%-6s %-6s %-24s %-24s %-12s %s\n" "---" "---" "-----" "--------" "-------" "------"

for src_type in ${SRC_TYPES}; do
    # Get source name
    case ${src_type} in
        7) src_name="Fz" ;;
        6) src_name="Fx" ;;
        1) src_name="P" ;;
    esac

    for rec_comp in ${REC_COMPS}; do
        TOTAL=$((TOTAL + 1))

        # Run the test and capture output
        OUTPUT=$(${FDELFWI}/test_wave_op_dp \
            ${COMMON} \
            src_type=${src_type} \
            rec_comp=${rec_comp} 2>/dev/null) || true

        # Parse results from output (use '^ *<' to match indented result lines only,
        # avoiding the header "Verifies: <Ax, y> = <x, A^T y>")
        LHS=$(echo "${OUTPUT}" | grep '^ *<Ax, y>' | awk -F'=' '{print $NF}' | tr -d ' ')
        RHS=$(echo "${OUTPUT}" | grep '^ *<x, A' | awk -F'=' '{print $NF}' | tr -d ' ')
        EPS=$(echo "${OUTPUT}" | grep '^ *epsilon' | awk -F'=' '{print $NF}' | tr -d ' ')
        STATUS=$(echo "${OUTPUT}" | grep -o 'PASS\|FAIL' | tail -1)

        if [ "${STATUS}" = "PASS" ]; then
            PASS_COUNT=$((PASS_COUNT + 1))
            STATUS_STR="PASS"
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            STATUS_STR="FAIL"
        fi

        printf "%-6s %-6s %-24s %-24s %-12s %s\n" \
            "${src_name}" "${rec_comp}" "${LHS}" "${RHS}" "${EPS}" "${STATUS_STR}"

        RESULTS="${RESULTS}${src_name}/${rec_comp}: ${STATUS_STR} (eps=${EPS})\n"
    done
done

echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"
echo ""
echo "  FD order:  ${IORDER}"
echo "  Total:     ${TOTAL}"
echo "  Passed:    ${PASS_COUNT}"
echo "  Failed:    ${FAIL_COUNT}"
echo ""

if [ ${FAIL_COUNT} -eq 0 ]; then
    echo "  ALL TESTS PASSED"
    echo ""
    echo "============================================================"
    exit 0
else
    echo "  SOME TESTS FAILED"
    echo ""
    echo "  Failed tests:"
    echo -e "${RESULTS}" | grep FAIL
    echo ""
    echo "============================================================"
    exit 1
fi
