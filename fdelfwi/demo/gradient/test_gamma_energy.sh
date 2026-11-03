#!/bin/bash
# test_gamma_energy.sh -- energy_gamma=0 versus energy_gamma=2 on the Rocha
#                         Fig 9 transmission geometry.
#
# =====================================================================
# THE QUESTION
# =====================================================================
# energy_imaging.c splits the gradient by opening angle using the two
# halves of an elastic energy inner product.  Which inner product is a
# CHOICE, because the members of the family
#
#   V_gamma(v,v~) = (lam+2mu) div.div~ + mu curl.curl~ - gamma*mu*N
#   N(v,v~)       = div.div~ - d_i v_j d_j v~_i
#
# all integrate to the same total energy: N is a null Lagrangian, its
# integral is a boundary term.  They are identical for ONE field and
# different for the POINTWISE PRODUCT OF TWO fields -- which is exactly
# what an imaging condition forms.  So the choice is invisible to any
# energy budget and fully visible in the gradient.
#
#   gamma = 2   eps:C:eps~, the strain energy.  What the code has always
#               done.
#   gamma = 1   Rocha, Tanushev & Sava (2016), the Laplacian form.  NOT
#               implemented -- it needs the full d_i v_j d_j v~_i tensor.
#   gamma = 0   (lam+2mu) div.div~ + mu curl.curl~, the Navier form.
#
# Plane-wave algebra for a state/adjoint pair at opening angle phi =
# 2 theta, normalised by rho w^2 (tests/gamma_energy/gamma_check.py,
# every assertion verified symbolically):
#
#              K            V_gamma                   kappa = Vs^2/Vp^2
#   PP     -cos(phi)     1 - gamma*kappa*sin^2(phi)
#   SS     -cos(phi)     1 - gamma*sin^2(phi)
#   PS     +-sin(phi)    -+gamma*sqrt(kappa) sin(phi) cos(phi)
#
# so, with H = K + V and R = V - K,
#
#   PP   R = 2cos^2(th) (1 - 2 gamma kappa sin^2 th)
#        H = 2sin^2(th) (1 - 2 gamma kappa cos^2 th)
#   SS   the same with kappa -> 1.
#
# AT gamma = 0 THE MODULATION AND kappa DROP OUT COMPLETELY:
#
#   PP and SS BOTH give   H = 2 sin^2(theta),  R = 2 cos^2(theta)
#   PS gives              V_0 == 0 identically, hence c2 = 1/2 exactly
#
# That is the whole hypothesis.  Three predictions follow, and this script
# is built to test exactly those three and nothing else:
#
#   (1) NO ISOTROPIC MODULATION.  At gamma=2 the clean 2sin^2/2cos^2 is
#       multiplied by (lam -+ 2mu cos2theta), which CHANGES SIGN at
#       2theta = 59 deg (PP, from cos^2 th = 1/(4 kappa) with
#       kappa = (1150/2000)^2 = 0.331) and at 30 deg for SS.  A sign
#       change in the weight is not a soft error: it moves energy into
#       the WRONG output.  At gamma=0 there is no zero on 0..90 deg at
#       all.  -> cell abs_s, which has the most S energy.
#
#   (2) MODE INDEPENDENCE.  At gamma=2 the PP and SS weights differ by
#       kappa; at gamma=0 they are the same function.  -> comparing
#       abs_p and abs_s.
#
#   (3) CONVERTED MODES.  At gamma=2 a P-S pair has V = -gamma sqrt(kappa)
#       sin cos, which is neither 0 nor 1 and lands wherever the geometry
#       puts it; the documented consequence is that conversions DO NOT
#       SEPARATE.  At gamma=0, V_0 vanishes identically for a converted
#       pair, which pins c2 to exactly 1/2 -- a single known value.
#       -> cell abs_mixed.
#
# =====================================================================
# WHY gamma=0 ALONE IS NOT ENOUGH, AND WHAT energy_alpha IS FOR
# =====================================================================
# c2 = 1/2 is NOT rejection.  The linear weight
#
#     a_tomo = |k+p| / (|k+p| + |k-p|) = 1 - c2
#
# gives a converted pair a_tomo = a_migr = 1/2: HALF-PASSED INTO BOTH
# outputs, not removed from either.  gamma=0 only guarantees that every
# conversion sits at the SAME c2.  Removing them then needs a mask that
# is small at c2 = 1/2, i.e. Rocha's exponential sharpening (his eq 45),
#
#     a_tomo = exp(-alpha c2)        a_migr = exp(-alpha (1-c2))
#
# which at alpha=100 gives exp(-50) ~ 2e-22.  gamma and alpha are
# COMPLEMENTARY, and the run matrix below separates their contributions
# rather than confounding them: abs_mixed is run at gamma=0/alpha=100 AND
# at gamma=2/alpha=100, so a gain can be attributed to the pairing, to
# the mask, or to the pair.
#
# PRICE OF alpha>0: a_tomo + a_migr != 1, so g*a_tomo + g*a_migr != g.
# The partition-of-unity regression is therefore checked ONLY on the
# alpha=0 runs.  It is not a weaker test there -- it is the only place it
# is a theorem.
#
# =====================================================================
# SIGN CALIBRATION -- DO THIS FIRST, AND ONLY HERE
# =====================================================================
# psi is a DUAL vector and psi_v carries a time-reversal sign relative to
# psi_sigma (section 2a of energy_imaging.c).  Getting that relative sign
# wrong does not produce a small error: IT EXCHANGES THE TWO FILTERS.
# gamma=0 does not remove the hazard (K now comes from psi_sigma and V_0
# from psi_v -- still two different halves) but it does remove the mode
# AMBIGUITY, so a sign error can no longer hide behind legitimate
# modulation.
#
# The calibration is the abs_p cell, because gamma=0 and gamma=2 MUST
# agree there: both are exact at 2theta = 0 and 180, and this geometry
# lives at precisely those two extremes.  The direct-path BANANA
# (2theta ~ 180) must land in a_tomo and the ELLIPSE (2theta ~ 0) in
# a_migr, for BOTH gammas.  If the banana lands in a_migr at gamma=0 but
# not at gamma=2, the gamma=0 relative sign is flipped.
#
# NEVER calibrate on anything at an intermediate angle: there the two
# gammas are legitimately different and a sign error is unfalsifiable.
#
# =====================================================================
# THE MATRIX: 9 NEW RUNS
# =====================================================================
#              ei1  ei2   | why
#   abs_p      g0a0  g0a0 | SIGN CALIBRATION + partition of unity
#              g0a100   - | does the mask damage the banana?
#   abs_s      g0a0  g0a0 | prediction (1): no 59/121 deg sign change
#   abs_mixed  g0a0  g0a0 | prediction (3): the converted-mode rings
#              g0a100   - | gamma=0 + mask: the predicted kill
#              g2a100   - | CONTROL: mask alone, without gamma=0
#
# energy_ic=0 is NOT re-run.  energyImagingInit returns before touching
# gamma or alpha when mode==EI_OFF, and both accumulators are guarded on
# it, so the unfiltered gradient is bit-identical to the one already in
# rocha_fig9/<cell>/l2/ei0.  Those directories are symlinked in, which
# also guarantees the partition check compares against the SAME g0.
#
# misfit=0 (L2) throughout.  It is the objective Rocha uses, and on the
# existing 36-run matrix it gave the cleaner P kernel than K-H did
# (abs_p banana/ellipse 2.59->10.08 for L2 against 2.20->7.70 for K-H).
# Mixing in a second objective here would confound the gamma test.
#
# Everything else -- model, wavelet, geometry, components, boundary,
# ntaper, all tapers off -- is copied from test_rocha_fig9.sh verbatim,
# and the observed data are REUSED from it, so gamma is the only thing
# that varies.
#
# =====================================================================
# Usage
#   ./test_rocha_fig9.sh          must have been run first (BNDS=abs is enough)
#   ./test_gamma_energy.sh                     all three cells
#   CELLS=abs_p ./test_gamma_energy.sh         sign calibration only
#   CELLS=abs_mixed ./test_gamma_energy.sh     the discriminator only
# =====================================================================

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}

HERE=$(cd "$(dirname "$0")" && pwd)
cd "${HERE}"

SRCWORK=${SRCWORK:-rocha_fig9}      # where test_rocha_fig9.sh put its output
WORK=${WORK:-gamma_energy}

CELLS=${CELLS:-"abs_p abs_s abs_mixed"}
OMP=${OMP:-16}                      # 3 runs in parallel x 16 = 48 of 64 threads

# ------------------------------------------------------------------ model
DX=5
VP_BG=2600;    VS_BG=1495;    RO_BG=2000
XSRC=3000;  ZSRC=1250
XRCV=3000;  ZRCV=3750
TMOD=3.0
DTRCV=0.001
DT=0.0005
NTAPER=150
PARAM=2
COMP=_rvx,_rvz,_rtxz,_rtzz
MISFIT=0                            # L2

if [ ! -d "${SRCWORK}" ]; then
    echo "ERROR: ${SRCWORK}/ not found.  Run ./test_rocha_fig9.sh first" >&2
    echo "       (BNDS=abs ./test_rocha_fig9.sh is enough for this script)." >&2
    exit 1
fi

src_type_of () {
    case "$1" in
        p)     echo 1 ;;
        s)     echo 5 ;;
        mixed) echo 6 ;;
        *)     echo "ERROR: unknown source variant '$1'" >&2; exit 1 ;;
    esac
}

echo "========================================================="
echo " energy_gamma=0 vs energy_gamma=2, Rocha Fig 9 geometry"
echo "   cells    ${CELLS}"
echo "   reusing  ${SRCWORK}/<cell>/obs_000_r*.su  and  l2/ei0/"
echo "   objective L2 (misfit=0), top absorbing"
echo "========================================================="

mkdir -p ${WORK}

# run_one <cell> <variant> <energy_ic> <gamma> <alpha> <src_type>
run_one () {
    local CELL=$1 VAR=$2 EI=$3 GAM=$4 ALP=$5 ST=$6
    local DIR=${WORK}/${CELL}/${VAR}/ei${EI}
    mkdir -p ${DIR}
    ( cd ${DIR}
      for f in ../../../../${SRCWORK}/model_*.su \
               ../../../../${SRCWORK}/wave.su \
               ../../../../${SRCWORK}/${CELL}/obs_*.su; do ln -sf $f .; done
      OMP_NUM_THREADS=${OMP} ${FDELFWI}/test_gradient_visual \
        file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
        file_src=wave.su file_rcv=syn \
        ischeme=3 iorder=4 src_type=${ST} \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} grad_taper=0 res_taper=0 aper_taper=0 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=100 chk_base=chk_g${EI} \
        param=${PARAM} scaling=0 precond=0 smooth_grad=0 \
        vp_min=1000 vp_max=4500 vs_min=500 vs_max=2500 \
        rho_min=1500 rho_max=2500 \
        data_weight=0 misfit=${MISFIT} \
        comp=${COMP} \
        file_obs=obs energy_ic=${EI} \
        energy_gamma=${GAM} energy_alpha=${ALP} \
        out_base=g${EI} > ../log_ei${EI}.txt 2>&1 )
}

# Symlink the already-computed unfiltered gradient into a variant directory,
# so the partition check and the plotter find ei0 where they expect it.
link_ei0 () {   # link_ei0 <cell> <variant>
    local CELL=$1 VAR=$2
    mkdir -p ${WORK}/${CELL}/${VAR}
    ln -sfn "${HERE}/${SRCWORK}/${CELL}/l2/ei0" "${WORK}/${CELL}/${VAR}/ei0"
}

for CELL in ${CELLS}; do
    SV=${CELL#abs_}
    ST=$(src_type_of ${SV})

    # Which variants this cell needs, and whether each gets ei2.
    case "${SV}" in
        p)     VARIANTS="g0a0 g0a100" ;;
        s)     VARIANTS="g0a0" ;;
        mixed) VARIANTS="g0a0 g0a100 g2a100" ;;
    esac

    echo ""
    echo "=== cell ${CELL}  (src_type=${ST}) ======================"
    if [ ! -f "${SRCWORK}/${CELL}/l2/ei0/g0_vp_raw.su" ]; then
        echo "  SKIP: ${SRCWORK}/${CELL}/l2/ei0/g0_vp_raw.su missing" >&2
        continue
    fi

    PIDS=""
    for VAR in ${VARIANTS}; do
        GAM=${VAR%a*}; GAM=${GAM#g}
        ALP=${VAR#*a}
        link_ei0 ${CELL} ${VAR}
        # ei2 only for the alpha=0 variants: the partition-of-unity check
        # needs both halves, and with alpha>0 it does not hold anyway.
        if [ "${ALP}" = "0" ]; then EIS="1 2"; else EIS="1"; fi
        for EI in ${EIS}; do
            echo "    gamma=${GAM} alpha=${ALP} energy_ic=${EI}"
            run_one ${CELL} ${VAR} ${EI} ${GAM} ${ALP} ${ST} & PIDS="${PIDS} $!"
        done
    done
    date +"    start %H:%M:%S  (${OMP} threads each)"
    wait ${PIDS}
    date +"    done  %H:%M:%S"

    # ----------------------------------------------------- diagnostics
    for VAR in ${VARIANTS}; do
        ALP=${VAR#*a}
        echo "    --- ${VAR} ---"
        for EI in 1 2; do
            L=${WORK}/${CELL}/${VAR}/log_ei${EI}.txt
            [ -f "${L}" ] || continue
            echo "      ei=${EI}: $(grep 'rms(P)/rms(K)' ${L} | tail -1 \
                                     | sed 's/.*energy_ic: //')"
            echo "             $(grep 'mean .* weight' ${L} | tail -1 \
                                     | sed 's/.*energy_ic: //')"
        done

        if [ "${ALP}" = "0" ]; then
            python3 - "${WORK}/${CELL}/${VAR}" <<'PY'
import sys
import numpy as np
sys.path.insert(0, '/rcp3/software/codes/OpenSource_SL10/fdelfwi/tests/kh_misfit')
import su_io

cell = sys.argv[1]
NX, NZ = 1201, 1001
out = []
for par in ('vp', 'vs', 'rho'):
    g = []
    for k in (0, 1, 2):
        d, _ = su_io.read('%s/ei%d/g%d_%s_raw.su' % (cell, k, k, par))
        g.append(d.astype(np.float64).reshape(NX, NZ))
    r = np.linalg.norm(g[1] + g[2] - g[0]) / max(np.linalg.norm(g[0]), 1e-300)
    out.append('%s %.2e %s' % (par, r, 'PASS' if r < 2e-2 else 'FAIL'))
print('      partition of unity:  ' + '   '.join(out))
PY
        else
            echo "      partition of unity: not applicable (alpha>0)"
        fi
    done
done

echo ""
echo "========================================================="
echo " Done.  Results in ${HERE}/${WORK}/"
echo " Next:  python3 plot_gamma_energy.py"
echo "========================================================="
