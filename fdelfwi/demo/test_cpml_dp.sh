#!/bin/bash
#
# test_cpml_dp.sh -- dot-product test of the bare elastic propagator,
#                    boundaries included.  Validates the CFS-CPML transpose.
#
# WHAT IT RUNS
#   <F^N S0, R> == <S0, (F^T)^N R>
# over the full padded wavefield, with NO source and NO receivers, so the only
# thing that can break it is the propagator and its boundary conditions.
#
# THE CASES
#   case 0  nstep=0            F = identity, so the two sides are the same sum
#                              in a different order.  Checks the bookkeeping
#                              (state size, inner product, copies) and nothing
#                              else.  Must pass at 1e-15.
#   case 1  pml                left=2 right=2 top=2 bottom=2 -- the case under
#                              test, boundaries_cpml_adj.c.
#   case 2  pml, layered       same, but lam/mu/rho vary inside the PML zone so
#                              the coefficients are not one constant.
#   case 3  pml, nstep=1       isolates ONE step.  If case 1 fails and this
#                              passes, the error is in the psi feedback (b),
#                              not in the spatial transpose.
#   case 6  top=1 + PML        left=2 right=2 top=1 bottom=2 -- a TOP free
#                              surface with PML sides, the production marine /
#                              land configuration.  Must pass.
#   case 4  taper          (EXPECTED FAIL, eps ~ 5e-1)
#   case 5  rigid          (EXPECTED FAIL, eps ~ 6e-2)
#   case 7  free x4        (EXPECTED FAIL, eps ~ 3e-2)
#
# Case 6 used to fail at ~2e-3 and no longer does.  Two independent defects,
# both found by brute-force column extraction with test_fs_col (which measures
# (F^T e_o)_j entry by entry by running the FORWARD on e_j, instead of scoring
# one scalar):
#
#   1. boundaries_adj.c blanket-zeroed adj_vx in the two outermost columns on
#      each side whenever top==1.  True for taper/rigid; false under CPML,
#      where boundariesP's ghost fill writes the left columns and the
#      free-surface txx stencil legitimately reads vx[nax-2] on the right.
#      Now guarded on !pml.
#   2. elastic4_adj.c's "free surface txz correction" ran on both paths, but
#      cpmlAdjF1T already gathers txz over the whole padded grid.  It doubled
#      exactly the txz[izs] and txz[izs-1] entries of F^T.  Now guarded on
#      !pml as well.
#
# Case 7 still fails: free surfaces on the LEFT and RIGHT (bnd.lef/rig==1) are
# a separate, still-approximate code path.
#
# WHY TAPER AND RIGID ARE EXPECTED TO FAIL -- READ THIS BEFORE "FIXING" THEM
# -------------------------------------------------------------------------
# My first guess was that these two would be controls: a taper is a pointwise
# multiply, hence diagonal, hence self-adjoint.  That is wrong, and the reason
# is worth writing down.  The forward boundariesP with taper does TWO things
# (boundaries.c:645-663), not one:
#
#     1. it runs the FULL velocity FD stencil over the taper pad, because
#        elastic4's main loop stops at [ioXx,ieXx) and excludes that pad
#     2. THEN it multiplies by bnd.tapx
#
# boundaries_adj.c replays only step 2 and drops the transpose of step 1.  That
# is a deliberate, documented approximation -- see the comment at
# boundaries_adj.c:64-79, "the taper damping alone is sufficient for practical
# FWI".  Rigid is the same story in a different costume: `vz[ghost] =
# -vz[interior]` is a mirror ASSIGNMENT, not a diagonal, and boundaries_adj.c
# labels it "self-adjoint, same as forward" (line 222) and repeats it verbatim.
# Repeating a non-diagonal assignment is not transposing it.
#
# So these two cases are not controls, they are a SENSITIVITY check: they prove
# this harness can actually see a broken boundary transpose.  A test that only
# ever prints PASS is not evidence of anything.
#
# nstep MATTERS: with nstep=1 the psi recursion only ever contributes its
# c*D(u) term and the b*psi feedback is never exercised, so a wrong b passes.
# The multi-step cases use nstep=12.
#
# Usage:  ./test_cpml_dp.sh            all cases
#         ./test_cpml_dp.sh pml        just the PML cases

set -u

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
CWP=/rcp3/software/codes/cwp/bin
FDELFWI=$(cd "$(dirname "$0")/.." && pwd)
export PATH=${BIN}:${CWP}:${PATH}

WHICH=${1:-all}
NSTEP=${NSTEP:-12}
NPML=${NPML:-20}

WORKDIR=$(dirname "$0")/cpml_dp_test
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

[ -x "${FDELFWI}/test_cpml_dp" ] || {
    echo "ERROR: ${FDELFWI}/test_cpml_dp not built.  cd ${FDELFWI} && make test_cpml_dp" >&2
    exit 1; }

# ---------------------------------------------------------------------------
# models.  Small on purpose: this test is O(nstep) not O(nt), and a 100x100
# grid already exercises every corner, edge and PML block.
# ---------------------------------------------------------------------------
if [ ! -f model_cp.su ]; then
    echo "--- building homogeneous model (100 x 100, dx=dz=5) ---"
    makemod sizex=500 sizez=500 dx=5 dz=5 \
        cp0=2000 cs0=1100 ro0=1800 orig=0,0 \
        file_base=model.su verbose=0
fi

if [ ! -f modelh_cp.su ]; then
    echo "--- building layered model (100 x 100, two interfaces) ---"
    makemod sizex=500 sizez=500 dx=5 dz=5 \
        cp0=1800 cs0=900 ro0=1600 orig=0,0 \
        intt=def poly=0 cp=2400 cs=1350 ro=2000 x=0,500 z=180,180 \
        intt=def poly=0 cp=3000 cs=1700 ro=2300 x=0,500 z=340,340 \
        file_base=modelh.su verbose=0
fi

# a wavelet is required by getParameters even though src.n is zeroed
if [ ! -f wave.su ]; then
    makewave fp=15 dt=0.0005 nt=256 fmax=30 file_out=wave.su t0=0.05 verbose=0
fi

# ---------------------------------------------------------------------------
NBAD=0
run_case () {
    local label="$1" base="$2" bndstr="$3" nstep="$4" want="$5" eps rc
    eps=$(${FDELFWI}/test_cpml_dp \
        file_cp=${base}_cp.su file_cs=${base}_cs.su file_den=${base}_ro.su \
        file_src=wave.su \
        ischeme=3 iorder=4 \
        dt=0.0005 tmod=0.1 \
        ${bndstr} npml=${NPML} ntaper=${NPML} \
        nstep=${nstep} seed=12345 verbose=0 \
        | sed -n 's/ *epsilon *= *//p')
    # the binary's exit status is unreachable through the pipe, so re-apply the
    # same 1e-5 threshold to the epsilon it printed
    rc=$(awk -v e="${eps:-9}" 'BEGIN{print (e+0 < 1e-5) ? 0 : 1}')
    if [ "${want}" = "pass" ]; then
        if [ ${rc} -eq 0 ]; then printf '  %-52s eps=%-11s OK\n'    "${label}" "${eps}"
        else                     printf '  %-52s eps=%-11s FAILED\n' "${label}" "${eps}"
                                 NBAD=$((NBAD+1)); fi
    else
        if [ ${rc} -ne 0 ]; then printf '  %-52s eps=%-11s fails as documented\n' "${label}" "${eps}"
        else                     printf '  %-52s eps=%-11s UNEXPECTEDLY PASSED\n' "${label}" "${eps}"
                                 NBAD=$((NBAD+1)); fi
    fi
}

echo ""
echo "--- exact transpose: must pass -----------------------------------------"
run_case "case 0  identity (nstep=0), bookkeeping only" \
         model "left=2 right=2 top=2 bottom=2" 0 pass
run_case "case 1  CFS-CPML x4, homogeneous, ${NSTEP} steps" \
         model "left=2 right=2 top=2 bottom=2" ${NSTEP} pass
run_case "case 2  CFS-CPML x4, layered,     ${NSTEP} steps" \
         modelh "left=2 right=2 top=2 bottom=2" ${NSTEP} pass
run_case "case 3  CFS-CPML x4, ONE step" \
         model "left=2 right=2 top=2 bottom=2" 1 pass
run_case "case 6  free surface on top, PML sides" \
         model "left=2 right=2 top=1 bottom=2" ${NSTEP} pass

if [ "${WHICH}" = "all" ]; then
echo ""
echo "--- documented approximations: must FAIL (sensitivity check) -----------"
run_case "case 4  taper x4  (stencil-in-pad transpose dropped)" \
         model "left=4 right=4 top=4 bottom=4" ${NSTEP} fail
run_case "case 5  rigid x4  (mirror assignment repeated, not transposed)" \
         model "left=3 right=3 top=3 bottom=3" ${NSTEP} fail
run_case "case 7  free surface x4 (no PML: lef/rig free surface is approximate)" \
         model "left=1 right=1 top=1 bottom=1" ${NSTEP} fail
fi

echo ""
echo "==========================================================="
if [ ${NBAD} -eq 0 ]; then echo " ALL CASES BEHAVED AS EXPECTED"
else                       echo " ${NBAD} CASE(S) DID NOT BEHAVE AS EXPECTED"; fi
echo "==========================================================="
exit $(( NBAD > 0 ))
