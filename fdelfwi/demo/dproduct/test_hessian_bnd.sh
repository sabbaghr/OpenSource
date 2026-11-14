#!/bin/bash
#
# test_hessian_bnd.sh -- Gauss-Newton Hessian symmetry, taper vs CFS-CPML vs
#                        free surface.
#
# WHY THIS EXISTS
# ---------------
# test_hessian_dp.sh pins left=4 right=4 top=4 bottom=4, so the whole Born /
# Hessian path has only ever been exercised with the taper boundary.  The
# bare-propagator adjoint is now an exact transpose under CFS-CPML and under a
# top free surface with PML sides (demo/test_cpml_dp.sh), and the FWI gradient
# is Taylor-verified for all three (test_taylor_bnd.sh) -- but that only
# covers the FIRST-order engine, fwd_shot + adj_shot.  Nothing yet covers
# born_shot.
#
# WHAT IT MEASURES, AND WHY SYMMETRY IS THE RIGHT PROBE
# -----------------------------------------------------
# hess_shot computes H_GN dm = J^T (J dm) by running born_shot for J and
# adj_shot for J^T.  The test scores
#
#     <dm2, H dm1>  vs  <dm1, H dm2>
#
# If born_shot computes the true Jacobian J, then H = J^T J is symmetric and
# the two agree.  If born_shot computes some other operator Jhat, then
# H = J^T Jhat, which is NOT symmetric, and the mismatch is exactly the
# discrepancy between the implemented Born operator and the transpose of the
# implemented adjoint.  So this is a genuine test of born_shot, not a
# tautology -- the two sides use different code.
#
# THE SUSPECT
# -----------
# born_shot advances TWO wavefields in the same time loop: the background
# re-propagation (callKernel, born_shot.c:331) and the Born scattered field
# (callKernelNoSrc, born_shot.c:339).  Both reach the CFS-CPML through
# elastic4 -> boundariesP/boundariesV, which share the single global psi set
# owned by cpml_state.c.  psi is STATE, not a scratch buffer, so the two
# fields alias each other's PML memory.  born_shot.c contains no cpml or psi
# reference at all.  Under the taper boundary there is no psi and nothing to
# alias, which is why this has never shown up.
#
# Usage:
#   ./test_hessian_bnd.sh [all|taper|pml|fs] [param]
#
# taper is the control: it must keep whatever it scored before, or the change
# under test broke something unrelated to the PML.
#

set -e

WHICH=${1:-all}
PARAM=${2:-1}

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

WORKDIR=hessian_bnd
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "--- model + wavelet ---"
makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    orig=0,0 file_base=model.su verbose=0
makewave fp=15 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

GEOM="ischeme=3 iorder=4 src_type=7 rec_type_vz=1 \
      dtrcv=0.004 tmod=1.0 \
      xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
      xsrc=500 zsrc=200"

run_one () {
    local tag=$1 bnd=$2
    echo ""
    echo "========================================================="
    echo " GAUSS-NEWTON HESSIAN SYMMETRY -- ${tag}"
    echo "   ${bnd}"
    echo "   param=${PARAM}"
    echo "========================================================="

    ${FDELFWI}/test_hessian_dp \
        file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
        file_src=wave.su file_rcv=born_${tag} \
        ${GEOM} ${bnd} verbose=1 \
        chk_skipdt=100 chk_base=chk_${tag} \
        pert_pct=0.01 seed=12345 \
        comp=_rvz param=${PARAM} \
        2>&1 | tee log_${tag}.txt \
        | grep -iE "dm2.*H.*dm1|dm1.*H.*dm2|relative|symmetr|PASS|FAIL"

    rm -f chk_${tag}_*.bin born_${tag}*.su
}

BND_TAPER="left=4 right=4 top=4 bottom=4 ntaper=30"
BND_PML="left=2 right=2 top=2 bottom=2 npml=${NPML:-30}"
BND_FS="left=2 right=2 top=1 bottom=2 npml=${NPML:-30}"

case ${WHICH} in
    taper) run_one taper "${BND_TAPER}" ;;
    pml)   run_one pml   "${BND_PML}"   ;;
    fs)    run_one fs    "${BND_FS}"    ;;
    all)   run_one taper "${BND_TAPER}" ; run_one pml "${BND_PML}" ; run_one fs "${BND_FS}" ;;
    *) echo "usage: $0 [all|taper|pml|fs] [param]" ; exit 1 ;;
esac

echo ""
echo "========================================================="
echo " $(pwd)"
echo "========================================================="
