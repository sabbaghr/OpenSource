#!/bin/bash
#
# test_cpml_checkpoint.sh -- does the checkpointed gradient still depend on
#                            the checkpoint INTERVAL when a CFS-CPML is on?
#
# WHAT IS BEING TESTED
# --------------------
# adj_shot does not store the forward wavefield; it stores checkpoints and
# re-propagates between them.  The gradient must therefore be INVARIANT to
# chk_skipdt: the checkpoint interval is an implementation detail of how the
# forward field is recovered, not a physical parameter.  chk_skipdt=1 stores
# every single time step, so no re-propagation happens at all and that run is
# the ground truth.  Any other interval must reproduce it exactly.
#
# Under a CFS-CPML it did not, because {vx,vz,txx,tzz,txz} is NOT the complete
# state.  The memory variables obey
#
#       psi^{n+1} = b * psi^n + c * D(u^n)
#
# so psi depends on the whole history of the field inside the absorbing band.
# checkpoint.c used to store the five wavefield arrays and nothing else, so
# every segment restarted with psi = 0 and the RECONSTRUCTED forward field was
# wrong in and near the PML.  cpml_state.c + the psi blob in checkpoint.c fix
# that.  This script is the regression test.
#
# WHY THE GEOMETRY IS WHAT IT IS
# ------------------------------
# The source sits 20 cells from the left edge and the receiver line runs the
# full width of the model at z=40 m, so a large part of the acquisition is
# inside or immediately next to the absorbing band.  That is deliberate.  With
# a source and receivers parked in the middle of a big model the PML barely
# sees the wavefield, psi stays small, and the bug this is hunting is invisible
# -- the test would print PASS whether or not the fix was in.  Here it is loud.
#
# THE CONTROL
# -----------
# Case "taper" repeats the identical comparison with left/right/top/bottom=4.
# A taper boundary has no memory variables, so chk_skipdt invariance there held
# before the fix and holds after it.  It proves this harness is measuring
# reconstruction error specifically and not just generic run-to-run noise --
# a test that can only ever print PASS is not evidence of anything.
#
# MEASURED, 2026-11-14
# --------------------
#                            rel = rms(g_37 - g_1)/rms(g_1)
#                            vp          vs          rho
#   psi restored (fixed)     0.0         0.0         0.0        PASS
#   psi zeroed  (pre-fix)    8.06e-04    8.67e-04    6.21e-04   FAIL
#   taper, both builds       0.0         0.0         0.0        PASS
#
# The pre-fix row was produced by swapping cpmlStateLoad() for cpmlStateZero()
# in readCheckpoint, which is exactly what the old code did implicitly.  Two
# things to read off it.  First, the fixed run is BIT-IDENTICAL, not merely
# within tolerance: re-propagation replays the same kernel over the same steps
# in the same order, so once the state is complete the float arithmetic is
# reproducible.  That makes TOL=1e-5 very slack; it is kept slack on purpose so
# the test does not become a tripwire for unrelated compiler or thread changes.
# Second, the pre-fix error is 1e-3, not 1e-1.  It is a quiet bug.  With
# grad_taper on, or a source parked in the middle of the model, it would
# disappear into the noise entirely -- which is why the geometry above puts the
# acquisition in the absorbing band and switches every post-process off.
#
# COST
# ----
# chk_skipdt=1 writes one checkpoint per time step, so the model is kept small
# on purpose (800 x 600 m).  Expect ~1 GB of scratch in the work directory,
# removed at the end unless KEEP=1.
#
# Usage:  ./test_cpml_checkpoint.sh          both cases
#         ./test_cpml_checkpoint.sh pml      just the PML case
#         KEEP=1 ./test_cpml_checkpoint.sh   leave the gradients on disk

set -u

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}

WHICH=${1:-all}
KEEP=${KEEP:-0}

WORKDIR=$(cd "$(dirname "$0")" && pwd)/cpml_checkpoint_test
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# ---- model / acquisition -------------------------------------------------
DX=5
SIZEX=800
SIZEZ=600
VP0=2000
VS0=1150
RO0=2000

NPML=20
TMOD=0.5
DTRCV=0.002

XSRC=100          # 20 cells from the left edge -> deep inside the PML's reach
ZSRC=40
XRCV1=0
XRCV2=${SIZEX}
DXRCV=10
ZRCV=40

# The two checkpoint intervals under comparison.  SKIP_REF=1 means every time
# step is a checkpoint, i.e. adj_shot never re-propagates and that gradient is
# exact by construction.
SKIP_REF=1
SKIP_TST=${SKIP_TST:-37}

# Tolerance on rms(g_test - g_ref)/rms(g_ref).  See MEASURED above: with the
# fix in, the two runs come out bit-identical and this is 0.  The margin is
# here for compiler/thread-count changes, not because any is expected.
TOL=${TOL:-1.0e-5}

echo "========================================================="
echo " CFS-CPML checkpoint regression"
echo "   gradient(chk_skipdt=${SKIP_REF})  vs  gradient(chk_skipdt=${SKIP_TST})"
echo "   model ${SIZEX} x ${SIZEZ} m, dx=${DX}, npml=${NPML}"
echo "   src x=${XSRC} z=${ZSRC}   rcv z=${ZRCV}, x=${XRCV1}..${XRCV2}"
echo "========================================================="

# ---- step 1: models ------------------------------------------------------
# Background is homogeneous.  The true model adds one smooth velocity blob, so
# the residual -- and hence the gradient -- is non-trivial.  What the gradient
# actually looks like is beside the point here; only its invariance matters.
echo "--- models ---"
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=model_bg.su verbose=0 || exit 1

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=elipse poly=0 x=350 z=250 var=90,90 \
        cp=2200 cs=1265 ro=${RO0} \
    orig=0,0 file_base=model_true.su verbose=0 || exit 1

echo "--- wavelet ---"
# fmax is set by the dispersion guard, not by taste: it measures the ACTUAL
# fmax of the wavelet (~1.22*fmax+1.5) and refuses to run above cmin/(5*dx) =
# 1150/25 = 46 Hz.  fmax=30 measures ~38 Hz and clears it.
makewave fp=15 fmin=3 fmax=30 dt=0.0005 nt=2048 file_out=wave.su t0=0.05 verbose=0 || exit 1

# ---------------------------------------------------------------------------
# run_case <name> <bndcode>
#   bndcode 2 = CFS-CPML, 4 = taper
# ---------------------------------------------------------------------------
run_case()
{
    NAME=$1
    BND=$2

    if [ "${BND}" = "2" ]; then
        BNDARG="left=2 right=2 top=2 bottom=2 npml=${NPML}"
    else
        BNDARG="left=4 right=4 top=4 bottom=4 ntaper=${NPML}"
    fi

    echo ""
    echo "---------------------------------------------------------"
    echo " case ${NAME}:  ${BNDARG}"
    echo "---------------------------------------------------------"

    # observed data, true model, SAME engine and SAME boundaries as the
    # gradient run -- otherwise the residual is not zero at the true model and
    # the comparison is polluted by a modelling discrepancy.
    ${FDELFWI}/fdelfwi_fwd \
        file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs_${NAME} \
        ischeme=3 iorder=4 src_type=1 \
        rec_type_vx=1 rec_type_vz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
        xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DXRCV} \
        zrcv1=${ZRCV} zrcv2=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ${BNDARG} > log_obs_${NAME}.txt 2>&1 || { echo "  obs FAILED"; return 1; }

    for SKIP in ${SKIP_REF} ${SKIP_TST}; do
        OUT=g_${NAME}_${SKIP}
        # grad_taper / res_taper / smooth_grad are all OFF.  Every one of them
        # post-processes the gradient and would mask a reconstruction error
        # exactly where it lives -- next to the sources, receivers and edges.
        ${FDELFWI}/test_gradient_visual \
            file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
            file_src=wave.su file_rcv=syn_${NAME}_${SKIP} \
            ischeme=3 iorder=4 src_type=1 \
            rec_type_vx=1 rec_type_vz=1 \
            dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
            xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DXRCV} \
            zrcv1=${ZRCV} zrcv2=${ZRCV} \
            xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
            ${BNDARG} \
            grad_taper=0 res_taper=0 smooth_grad=0 \
            chk_skipdt=${SKIP} chk_base=chk_${NAME}_${SKIP} \
            param=2 scaling=0 precond=0 \
            data_weight=0 comp=_rvx,_rvz \
            file_obs=obs_${NAME} out_base=${OUT} \
            > log_grad_${NAME}_${SKIP}.txt 2>&1 \
            || { echo "  gradient (skipdt=${SKIP}) FAILED, see log_grad_${NAME}_${SKIP}.txt"; return 1; }
    done

    echo ""
    printf "  %-8s %-14s %-14s %-12s %s\n" param "rms(g_ref)" "rms(diff)" "rel" verdict
    FAILED=0
    for P in vp vs rho; do
        REF=g_${NAME}_${SKIP_REF}_${P}_raw.su
        TST=g_${NAME}_${SKIP_TST}_${P}_raw.su
        if [ ! -f "${REF}" ] || [ ! -f "${TST}" ]; then
            printf "  %-8s MISSING (%s / %s)\n" "${P}" "${REF}" "${TST}"
            FAILED=1
            continue
        fi
        suop2 "${TST}" "${REF}" op=diff > d_${NAME}_${P}.su

        RR=$(sumax < "${REF}" mode=rms outpar=/dev/stdout 2>/dev/null | tr ' ' '\n' | grep -E '^[-0-9.eE+]+$' | tail -1)
        RD=$(sumax < d_${NAME}_${P}.su mode=rms outpar=/dev/stdout 2>/dev/null | tr ' ' '\n' | grep -E '^[-0-9.eE+]+$' | tail -1)
        REL=$(awk -v a="${RD}" -v b="${RR}" 'BEGIN{ if (b+0==0) print "nan"; else printf "%.4e", a/b }')
        OK=$(awk -v r="${REL}" -v t="${TOL}" 'BEGIN{ print (r+0 <= t+0) ? "PASS" : "FAIL" }')
        [ "${OK}" = "FAIL" ] && FAILED=1
        printf "  %-8s %-14s %-14s %-12s %s\n" "${P}" "${RR}" "${RD}" "${REL}" "${OK}"
    done

    echo ""
    if [ ${FAILED} -eq 0 ]; then
        echo "  case ${NAME}: PASS  (gradient is independent of chk_skipdt)"
    else
        echo "  case ${NAME}: FAIL  (rel > ${TOL})"
        if [ "${BND}" = "2" ]; then
            echo "    The CFS-CPML psi memory variables are not being carried"
            echo "    across the checkpoint.  Check cpmlStateSave/Load in"
            echo "    checkpoint.c and chk->npsi -- npsi=0 means the psi blob"
            echo "    was never written, which silently skips the restore."
        fi
    fi
    return ${FAILED}
}

RC=0
case "${WHICH}" in
    pml)   run_case pml   2 || RC=1 ;;
    taper) run_case taper 4 || RC=1 ;;
    *)     run_case pml   2 || RC=1
           run_case taper 4 || RC=1 ;;
esac

# ---- scratch ------------------------------------------------------------
# The chk_skipdt=1 run leaves one checkpoint per time step on disk; that is
# the bulk of the ~1 GB.  cleanCheckpoints removes them on a clean exit, so
# this only matters after a crash.
rm -f chk_*_chk_*.bin
if [ "${KEEP}" != "1" ]; then
    rm -f g_*_raw.su g_*_scaled.su g_*_preco.su d_*.su \
          obs_*.su syn_*.su model_*.su wave.su
fi

echo ""
echo "========================================================="
echo " ${WORKDIR}"
[ "${KEEP}" = "1" ] && echo "   KEEP=1: gradients and data left in place"
echo "========================================================="
exit ${RC}
