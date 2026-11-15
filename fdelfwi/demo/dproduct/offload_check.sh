#!/bin/bash
#
# offload_check.sh -- A/B the OpenMP-target offload build against the CPU
# reference on one machine.
#
# Usage:
#     ./offload_check.sh cpu      # after: make clean && make fwd test_wave_op_dp
#     ./offload_check.sh gpu      # after: make clean && make OFFLOAD=p100 fwd test_wave_op_dp
#
# Run `cpu` FIRST.  The `gpu` run then automatically diffs its receiver
# gather against the one the `cpu` run left behind, which is the actual
# verdict for slice 1a.
#
# ---------------------------------------------------------------------------
# WHICH TEST EXERCISES THE DEVICE, AND WHICH DOES NOT
#
# Test A, fdelfwi_fwd, DOES.  It goes through fdfwimodc(), which is the only
# driver that opens the per-shot `target enter data` region and sets
# fdOffloadActive=1.  The elastic4 stencils then run on the GPU.
#
# Test B, test_wave_op_dp, DOES NOT.  It reaches elastic4 through callKernel
# directly, so fdOffloadActive stays 0, the `if(target: ...)` gate evaluates
# false, and the loops execute on the host.  That is by design for slice 1a:
# adj_shot, born_shot and the dot-product drivers do not map their arrays, so
# an ungated target region would dereference host pointers on the device.
# Test B is still worth running -- it is the regression that proves the
# FD_HOST_PARALLEL macro layer did not perturb the adjoint -- but it is NOT
# evidence that anything ran on a GPU.  Only Test A is.
#
# So: a `gpu` run that reports "device data region: not reported" has silently
# fallen back to the host, and its numbers prove nothing.
# ---------------------------------------------------------------------------
#
# WHY THIS IS NOT test_wave_op_dp.sh.  That script cannot run outside the
# seislab10 checkout: it hardcodes ROOT=/rcp3/software/codes/OpenSource_SL10,
# and it sets FDELFWI=../.. before cd-ing into a work directory, so ../..
# resolves to fdelfwi/demo/ -- where a stale COPY OF THE SCRIPT named
# test_wave_op_dp happens to live.  It re-executes itself and recurses.  Both
# bugs are pre-existing and are not fixed here.
#
# ON WALL TIME.  Slice 1a offloads the four elastic4 stencil loops ONLY.
# Source injection, boundaries and receiver recording are still host code, so
# every time step copies five wavefield arrays to the device and back.  On a
# PCIe P100 that transfer costs more than the stencil saves.  Expect the GPU
# build to be SLOWER.  The deliverable of slice 1a is that the gather is
# unchanged, not that the clock moved.
#

set -e
set -o pipefail

LABEL=${1:-cpu}
IORDER=${2:-4}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../../.." && pwd)
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=${CWP:-/rcp3/software/codes/cwp/bin}

FWD=${FDELFWI}/fdelfwi_fwd
DPTEST=${FDELFWI}/test_wave_op_dp

for f in "${FWD}" "${DPTEST}" "${BIN}/makemod" "${BIN}/makewave"; do
	[ -x "$f" ] || { echo "MISSING BINARY: $f"; exit 1; }
done

export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

WORKDIR=${HERE}/offload_check_${LABEL}
REFDIR=${HERE}/offload_check_cpu
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

echo "========================================================="
echo " OFFLOAD CHECK  [${LABEL}]"
echo "========================================================="
echo " host         : $(hostname)"
echo " date         : $(date)"
echo " tree         : ${ROOT}"
echo " iorder       : ${IORDER}"
echo " OMP threads  : ${OMP_NUM_THREADS}"
echo " workdir      : ${WORKDIR}"
echo ""
if command -v strings >/dev/null 2>&1 && strings "${FWD}" 2>/dev/null | grep -qiE '__nv_|nvomp|libomptarget|omp_target'; then
	echo " device runtime symbols in fdelfwi_fwd : PRESENT  (offload build)"
else
	echo " device runtime symbols in fdelfwi_fwd : absent   (plain CPU build)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
		| sed 's/^/ gpu visible                           : /' || true
fi
echo ""

echo "--- building model and wavelet ---"
makemod sizex=500 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=1100 ro0=1800 \
    orig=0,0 file_base=model.su verbose=0
makewave fp=15 dt=0.001 nt=512 fmax=30 file_out=wave.su t0=0.10 verbose=0
echo "  100x100, dx=dz=5m, Vp=2000 Vs=1100 rho=1800; wavelet fp=15 fmax=30"
echo ""

# =====================================================================
# TEST A -- forward modelling through fdfwimodc().  THIS is the device path.
# =====================================================================
echo "--- TEST A: fdelfwi_fwd (goes through fdfwimodc -> GPU) ---"
A0=$(date +%s.%N)
"${FWD}" \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=fwd \
    ischeme=3 iorder=${IORDER} src_type=1 \
    rec_type_vx=1 rec_type_vz=1 \
    dtrcv=0.001 tmod=1.0 verbose=1 \
    xrcv1=100 xrcv2=400 zrcv1=350 zrcv2=350 dxrcv=10 \
    xsrc=250 zsrc=100 nshot=1 \
    ntaper=50 left=4 right=4 top=4 bottom=4 \
    > fwd.log 2>&1
A1=$(date +%s.%N)

NREG=$(grep -c 'OpenMP target data region' fwd.log || true)
if [ "${NREG}" -gt 0 ]; then
	echo "  device data region : ACTIVE (${NREG} shot(s))"
	grep -m1 'OpenMP target data region' fwd.log | sed 's/^/    /'
else
	echo "  device data region : NOT REPORTED  <-- ran on the host"
fi
awk -v a="${A0}" -v b="${A1}" 'BEGIN{printf "  wall time          : %.2f s\n", b-a}' 
ls -1 fwd_*.su 2>/dev/null | sed 's/^/    gather: /'
echo ""

# =====================================================================
# TEST B -- adjoint dot product.  Host path even in the offload build;
# this is the macro-layer regression, not a GPU test.
# =====================================================================
echo "--- TEST B: test_wave_op_dp (host path; macro-layer regression) ---"
B0=$(date +%s.%N)
"${DPTEST}" \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=syn file_snap=adj_snap \
    ischeme=3 iorder=${IORDER} src_type=7 \
    rec_type_vz=1 \
    dtrcv=0.001 tmod=0.5 verbose=1 \
    xrcv1=100 xrcv2=400 zrcv1=350 zrcv2=350 dxrcv=10 \
    xsrc=250 zsrc=100 ntaper=50 \
    left=4 right=4 top=4 bottom=4 \
    seed=42 comp=_rvz \
    > dp.log 2>&1
B1=$(date +%s.%N)
grep -iE 'epsilon|^ +<Ax|^ +<x,|PASS|FAIL' dp.log | sed 's/^/  /' || echo "  (read dp.log)"
awk -v a="${B0}" -v b="${B1}" 'BEGIN{printf "  wall time          : %.2f s\n", b-a}' 
echo ""

# =====================================================================
# Compare this run's gather against the cpu reference run.
# =====================================================================
if [ "${LABEL}" != "cpu" ] && [ -d "${REFDIR}" ]; then
	echo "========================================================="
	echo " GATHER DIFF  [${LABEL}] vs [cpu]"
	echo "========================================================="
	for g in fwd_*.su; do
		[ -f "${REFDIR}/$g" ] || { echo "  $g : no reference"; continue; }
		REF=$(sumax outpar=/dev/stdout < "${REFDIR}/$g" 2>/dev/null | awk '{m=($1<0?-$1:$1); n=($2<0?-$2:$2); print (m>n?m:n)}')
		DIF=$(suop2 "$g" "${REFDIR}/$g" op=diff 2>/dev/null | sumax outpar=/dev/stdout 2>/dev/null | awk '{m=($1<0?-$1:$1); n=($2<0?-$2:$2); print (m>n?m:n)}')
		awk -v g="$g" -v r="${REF:-0}" -v d="${DIF:-0}" \
		    'BEGIN{ printf "  %-20s max|ref|=%12.5e  max|diff|=%12.5e  rel=%10.3e\n", g, r, d, (r>0? d/r : -1) }'
	done
	echo ""
	echo " Slice 1a passes if rel is at float round-off, say <~1e-4."
	echo " It is NOT expected to be 0, and not even ~1e-7, for two reasons"
	echo " that have nothing to do with the GPU:"
	echo "   - the cpu reference is built with -march=native -ffast-math,"
	echo "     the offload build deliberately is not;"
	echo "   - nvc and gcc contract multiply-adds differently."
	echo " Measured on seislab10, host-fallback vs cpu: rel ~2-3e-6."
	echo " A GPU run landing in that same range is the result we want."
	echo " What would be a FAILURE: rel ~1e-2 or larger, or NaN."
	echo "========================================================="
fi

echo "logs: ${WORKDIR}/fwd.log  ${WORKDIR}/dp.log"
