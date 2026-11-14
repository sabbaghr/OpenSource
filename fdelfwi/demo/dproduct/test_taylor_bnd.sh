#!/bin/bash
#
# test_taylor_bnd.sh -- Taylor gradient verification, taper vs CFS-CPML,
#                       on a setup where the test is actually sensitive.
#
# WHY THIS EXISTS (do not merge it back into test_taylor.sh without reading)
# -------------------------------------------------------------------------
# test_taylor.sh FAILS, and has always failed, on BOTH boundaries:
#
#        phi(m0)     <grad,dm>    best|ratio-1|
#   taper   6.8571e-22   2.0300e-23   1.00   FAIL
#   pml     6.7965e-22   2.2054e-23   1.00   FAIL
#
# The two agree to 1%, so the failure is nothing to do with the boundary.  It
# is the experiment design.  The Taylor test measures
#
#     phi(m+eps*dm) - phi(m) = eps*<g,dm> + 0.5*eps^2*|J dm|^2 + ...
#
# and it can only confirm g when the LINEAR term dominates.  test_taylor.sh
# arranges for it to lose, twice over:
#
#   1. dm is grid-scale WHITE NOISE (test_taylor.c:395-401, drand48 per cell).
#      White noise is the worst possible Taylor direction: it scatters hard,
#      so |J dm|^2 is large, and it is near-orthogonal to the smooth gradient,
#      so <g,dm> is a sqrt(N) random-sign sum instead of an O(N) one.  At
#      eps=1e-1 the quadratic term is 26x the linear one.
#
#   2. The model error is a var=10 point diffractor in an otherwise identical
#      background, so the residual is ~0.1% of the data.  phi is then a
#      difference of two nearly equal float32 .su gathers, and dphi floors at
#      ~1e-4 of phi -- reached while the quadratic term is STILL dominant.
#      There is no eps that is both linear-dominated and above the noise.
#
# Consequence: the reported "ord2 ~ 2" is vacuous.  With the linear term
# negligible, err2 = phi_pert - phi - eps<g,dm> is just dphi, which is O(eps^2)
# whether or not g is the gradient.  The old test could not fail a wrong
# gradient and could not pass a right one.
#
# WHAT THIS SCRIPT CHANGES
# ------------------------
# Only the experiment, not the code: the model error goes into the BACKGROUND
# VELOCITY (start 2000/1150, true 2100/1200) instead of a weak diffractor.
# That scales <r0, J dm> linearly with |r0| while leaving |J dm|^2 alone, so
# the linear/quadratic crossover moves up by the same factor the float32 noise
# floor moves down.  Both media stay homogeneous, so the residual is a clean
# ~17 ms shift of the direct arrival -- about a quarter period at fp=15, large
# enough to dominate and small enough not to cycle-skip.
#
# and band-limits dm (smooth_dm=, new option in test_taylor.c, default off so
# every existing caller is bit-identical).  dm stays an INDEPENDENT random
# direction -- it is not set to g, which would only probe one direction -- it
# is just stripped of the grid-scale content that inflates |J dm|^2.  Its RMS
# is restored after smoothing so pert_pct keeps its old meaning.
#
# chk_skipdt=100 is deliberate for the PML case: it forces re-propagation
# between checkpoints, which is the path psi checkpointing had to fix
# (cpml_state.c + the psi blob in checkpoint.c).  Do not set it to 1.
#
# Usage:
#   ./test_taylor_bnd.sh            # both boundaries, all parameters
#   ./test_taylor_bnd.sh pml        # CPML only
#   ./test_taylor_bnd.sh taper      # taper only
#   ./test_taylor_bnd.sh both 1     # both, Vp only  (1=Vp 2=Vs 3=rho 4=rho_Lame)

set -e

WHICH=${1:-both}
TEST_PARAM=${2:-0}

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
export PATH=${BIN}:${PATH}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

WORKDIR=$(cd "$(dirname "$0")" && pwd)/taylor_bnd
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

# ---------------------------------------------------------------- models ---
# Both homogeneous.  The error is the background itself, so the residual is a
# traveltime shift of the direct arrival rather than a weak scattered field.
#   src z=200 -> rcv z=700, ~500-700 m path
#   700/2000 - 700/2100 = 16.7 ms  ~ 0.25 period at fp=15
echo "--- true model (homogeneous, 5% fast) ---"
makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2100 cs0=1200 ro0=2000 \
    orig=0,0 file_base=true_model.su verbose=0

echo "--- starting model (homogeneous) ---"
makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    orig=0,0 file_base=start_model.su verbose=0

# Dispersion guard checks the MEASURED wavelet fmax (~1.22*f+1.5) against
# cmin/(5*dx) = 1150/25 = 46 Hz.  fmax=25 measures ~32 Hz and clears it.
echo "--- wavelet ---"
makewave fp=15 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

GEOM="dtrcv=0.004 tmod=1.0 \
      xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
      xsrc=500 zsrc=200 \
      ischeme=3 iorder=4 src_type=7 rec_type_vz=1"

run_one () {
    local tag=$1 bnd=$2
    echo ""
    echo "========================================================="
    echo " TAYLOR GRADIENT VERIFICATION -- ${tag}"
    echo "   ${bnd}"
    echo "   test_param=${TEST_PARAM} (0=all 1=Vp 2=Vs 3=rho_vel 4=rho_Lame)"
    echo "========================================================="

    # Observed data MUST use the same boundaries as the gradient run, or the
    # boundary mismatch itself shows up in the residual.
    ${FDELFWI}/fdelfwi_fwd \
        file_cp=true_model_cp.su file_cs=true_model_cs.su file_den=true_model_ro.su \
        file_src=wave.su file_rcv=obs_${tag} \
        ${GEOM} ${bnd} verbose=0

    ${FDELFWI}/test_taylor \
        file_cp=start_model_cp.su file_cs=start_model_cs.su file_den=start_model_ro.su \
        file_src=wave.su file_rcv=syn_${tag} \
        file_obs=obs_${tag}_000 \
        ${GEOM} ${bnd} verbose=1 \
        chk_skipdt=100 chk_base=chk_${tag} \
        pert_pct=0.01 seed=12345 smooth_dm=${SMOOTH_DM:-6} \
        test_param=${TEST_PARAM} \
        comp=_rvz \
        neps=10 eps_min=1e-8 eps_max=1e-1 \
        2>&1 | tee log_${tag}.txt | sed -n '/TAYLOR GRADIENT TEST RESULTS/,/^$/p;/Best |ratio/,/PASS\|FAIL/p'

    rm -f chk_${tag}_*.bin
    rm -f cp_taylor_pert.su cs_taylor_pert.su ro_taylor_pert.su
    rm -f residual_taylor*.su syn_${tag}*.su
}

BND_TAPER="left=4 right=4 top=4 bottom=4 ntaper=30"
BND_PML="left=2 right=2 top=2 bottom=2 npml=${NPML:-30}"
# Free surface on top, PML sides -- the production configuration.  Its bare
# propagator adjoint is exact (test_cpml_dp.sh case 6), so this must pass too.
BND_FS="left=2 right=2 top=1 bottom=2 npml=${NPML:-30}"

case ${WHICH} in
    taper) run_one taper "${BND_TAPER}" ;;
    pml)   run_one pml   "${BND_PML}"   ;;
    fs)    run_one fs    "${BND_FS}"    ;;
    both)  run_one taper "${BND_TAPER}" ; run_one pml "${BND_PML}" ;;
    all)   run_one taper "${BND_TAPER}" ; run_one pml "${BND_PML}" ; run_one fs "${BND_FS}" ;;
    *) echo "usage: $0 [both|all|taper|pml|fs] [test_param]" ; exit 1 ;;
esac

echo ""
echo "========================================================="
echo " ${WORKDIR}"
echo "========================================================="
