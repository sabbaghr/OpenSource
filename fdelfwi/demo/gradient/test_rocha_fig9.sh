#!/bin/bash
# test_rocha_fig9.sh -- elastic translation of Rocha, Tanushev & Sava (2016),
#                       GEOPHYSICS 81(4), S151-S163, Figures 8 and 9,
#                       driven by the Kirchhoff-Helmholtz ENERGY-FLUX objective
#                       (misfit=2) instead of acoustic L2.
#
# =====================================================================
# WHAT ROCHA'S FIGURE 9 ACTUALLY IS
# =====================================================================
# It is NOT a reflector experiment.  It is a pure TRANSMISSION test, which
# is exactly why it isolates the tomographic kernel so cleanly:
#
#   homogeneous v = 2.5 km/s, model 6 x 3.5 km, NO reflector, NO anomaly
#   ONE source at (1, 2.25) km, ONE receiver at (5, 2.25) km, both buried
#   starting model = a uniform +30% on the true velocity.  That is the
#   ONLY difference between the two models.
#
# The residual (his Fig 9a) is therefore just TWO SPIKES: the synthetic
# direct wave and the observed direct wave, separated in time.  The
# gradient that results has exactly two pieces, and the paper names both
# (p. S160):
#
#   1. a BANANA along the direct source-receiver path -- "the well-known
#      low-frequency waveform inversion sensitivity kernel".  This is the
#      thing we want to KEEP.  Opening angle 2theta ~ 180 deg.
#
#   2. an ELLIPSE where t_s + t_r = t_obs -- the adjoint field correlates
#      with the forward field at every point whose total traveltime matches
#      the observed arrival, "placing an apparent reflector where it does
#      not exist".  This is the thing we want to KILL.  2theta ~ 0 deg.
#
# His Fig 9c is the conventional gradient (banana buried under ellipse
# rings); Fig 9e is the energy-filtered one (clean banana).
#
# WHY THIS IS THE RIGHT TEST FOR energy_ic.  The documented isotropic
# modulation of our elastic filter, (lam -/+ 2 mu cos2theta), is EXACT at
# 2theta = 0 and 180 degrees and worst in between (it changes sign at 59
# and 121 deg for Vp=2000/Vs=1150/rho=2000).  This experiment lives at
# precisely those two extremes, so it is where the filter should look its
# best.  The z=1500 reflector test (test_energy_ic.sh) spans intermediate
# angles and is the filter's worst case.
#
# =====================================================================
# WHY THE GEOMETRY IS ROTATED 90 DEGREES
# =====================================================================
# Rocha puts source and receiver at the SAME DEPTH, so the direct wave
# travels HORIZONTALLY.  That is fine for his acoustic L2 objective and
# fatal for ours.  kh_misfit.c:316-317 computes
#
#     r = n_z * ( dsig_xz * dv_x  +  dsig_zz * dv_z )
#
# which is the residual energy flux THROUGH A HORIZONTAL PLANE.  For a
# horizontally-travelling P wave, dv_z = 0 and dsig_xz = 0 at the
# receiver, so r == 0.  The Poynting vector says the same thing:
# S_z = -(sig_xz v_x + sig_zz v_z) = 0.  Rocha's literal geometry gives an
# IDENTICALLY ZERO energy-flux misfit.
#
# So the source-receiver axis is rotated to VERTICAL.  The direct wave
# then crosses the recording plane at normal incidence -- maximum vertical
# flux, which is the regime the K-H objective is built for.  The medium is
# homogeneous, so the banana/ellipse geometry is rotation invariant: this
# is still physically Rocha's Fig 9, turned on its side.
#
#   model     6000 x 5000 m, dx = 5
#   source    (3000, 1250)
#   receiver  (3000, 3750)     separation L = 2500 m, VERTICAL
#   true      Vp=2000  Vs=1150  rho=2000     (homogeneous)
#   start     Vp=2600  Vs=1495  rho=2000     (x1.30 on velocities only)
#
# ANALYTIC LOCI (both closed form -- this is what makes the test
# quantitative rather than a visual impression):
#
#   banana    the vertical segment x=3000, z in [1250, 3750].
#             First Fresnel radius at the midpoint, sqrt(lambda L)/2:
#             P at 10 Hz, v_bg=2600 -> lambda=260 m -> r_F = 403 m
#
#   ellipse   d_s + d_r = v_bg * t_obs = 2a, with
#             a = v_bg * (L/v_true) / 2 = 1.30 * L / 2 = 1625 m
#             c = L/2 = 1250 m,  b = sqrt(a^2-c^2) = 1038 m
#             centred (3000, 2500) -> z in [875,4125], x in [1962,4038]
#
#             NOTE a = 1.30*L/2 is INDEPENDENT OF MODE, because the 30%
#             scaling is applied to Vp and Vs alike.  The PP and SS
#             ellipses therefore COINCIDE exactly.  Anything landing off
#             that ellipse is a mode-conversion artifact and is directly
#             measurable -- which is how this experiment probes the
#             documented P-S limitation of energy_ic.
#
# =====================================================================
# THE OBJECTIVE
# =====================================================================
# misfit=2 (kh_misfit.c), the Kirchhoff-Helmholtz energy flux.  Three
# consequences worth knowing while reading the output:
#
#   * data_weight MUST be 0 (computeResidual.c enforces it).
#   * comp must contain Vx, Vz, Txz and Tzz -- khResolveChannels hard
#     errors otherwise.  _rtxx IS recorded (costs nothing) but is left out
#     of comp because a_txx = 0 for a horizontal normal, kh_misfit.c:52-60.
#     A later dipping-normal test then needs no remodelling.
#   * J is a SIGNED bilinear form, not a quadratic one (khApply returns
#     w*(rs+rn), not 0.5*r^2).  kh_normal therefore flips the sign of both
#     J and the gradient.  kh_normal=+1 here: the receiver is BELOW the
#     source, so the direct flux is downward.
#   * ONE receiver is explicitly supported: khWeightsBuild sets w=1.0 and
#     reports that J is a flux DENSITY in W/m^2, not a flux
#     (kh_misfit.c:206-214).
#
# =====================================================================
# THE MATRIX: 36 RUNS
# =====================================================================
# 3 energy_ic x 2 objectives x 2 boundary conditions x 3 source types.
#
#   energy_ic  0 unfiltered / 1 tomographic / 2 migration
#   objective  l2 = conventional waveform L2 (misfit=0) -- what Rocha uses
#              kh = K-H energy flux (misfit=2)
#   boundary   abs = top absorbing (Rocha 9a,c,e,g)
#              fs  = top free surface (Rocha 9b,d,f,h)
#   source     p      src_type=1 (P/explosion) -- the direct Fig 9 analogue
#              s      src_type=5 (S-pot)       -- no acoustic counterpart
#              mixed  src_type=6 (Fx)          -- both modes
#
# The l2/kh pair is the controlled comparison: SAME observed data, SAME
# model, SAME four components, SAME data_weight=0.  The only thing that
# differs is dJ/dd, and hence the adjoint source -- independent
# per-component point injection for L2 versus the tensorial
# traction->force / velocity->deformation-rate split for K-H.  Ravasi &
# Curtis (2013) predict the L2 injection radiates both up and down and
# emits both P and S from a single-mode residual; this geometry shows
# that directly, and then asks whether the energy_ic angle filter can
# clean up what is left in each case.
#
# Even the pure-P variant shows a forward-P x adjoint-S cross term,
# because the K-H adjoint injects tractions as FORCES (kh_misfit.c:48-51)
# and a point force radiates both modes.  That interaction between the
# tensorial injection and the angle filter is the point of the s/mixed
# variants.
#
# The three energy_ic runs of a given (boundary, source) pair are
# INDEPENDENT and run in parallel, each in its own subdirectory.  Separate
# directories are required, not just separate out_base: file_rcv (syn_*),
# chk_base and res_*.su would otherwise collide.
#
# =====================================================================
# WHAT IS DELIBERATELY NOT DONE
# =====================================================================
# Rocha's Fig 9e/9f apply his EXPONENTIAL SHARPENING (eq 45, alpha=100).
# energy_imaging.c deliberately does not implement it -- he drops it
# himself for Marmousi (p. S161), where eq 45 reduces to eq 35.  Expect
# our banana to be softer-edged than his.
#
# Ours is also the IMAGE-domain flavour (his 9e/9f), a per-point weight on
# the accumulated kernel, not the wavefield-domain one (his 9g/9h).
#
# =====================================================================
# =====================================================================
# EI_GAMMA: WHICH ENERGY PAIRING, AND WHY THIS SCRIPT CALIBRATES ITS SIGN
# =====================================================================
# EI_GAMMA selects energy_gamma (2 = eps:C:eps~, 0 = Navier div/curl,
# -1 = EI_MODESEP, the per-mode acoustic norm).  Changing it changes only
# how the two halves K and V are built; the mask is |K+V| / (|K+V|+|K-V|)
# in every case.
#
# This experiment is the ONLY place the SIGN convention can be settled.
# test_ei_modesep.c feeds the adjoint in analytically with no relative
# flip between psi_v and psi_sigma, so it measures cos(psi) perfectly
# while being blind to which end of that range a real adj_shot run puts
# the direct path at.  Getting it backwards EXCHANGES a_tomo and a_migr
# and changes no number in that unit test.
#
# The calibration is the abs/p cell: the direct-path BANANA must land in
# energy_ic=1 (tomographic) and the ELLIPSE in energy_ic=2 (migration).
# If they come out swapped, the relative K/V sign is wrong.  Every gamma
# must agree with every other one here, because 2theta ~ 0 and ~180 deg
# are the two angles at which all the pairings coincide exactly.
#
# =====================================================================
# Usage
#   ./test_rocha_fig9.sh                            full 36-run matrix
#   MISFITS=kh BNDS=abs SRCS=p ./test_rocha_fig9.sh one triple (smoke test)
#   MISFITS="l2 kh" BNDS="abs fs" SRCS="p s mixed" ...  explicit
#   EI_GAMMA=-1 WORK=rocha_fig9_ms MISFITS=l2 BNDS=abs SRCS=p \
#       ./test_rocha_fig9.sh                        sign calibration cell
# =====================================================================

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}

HERE=$(cd "$(dirname "$0")" && pwd)
cd "${HERE}"

WORK=${WORK:-rocha_fig9}

# Which cells of the matrix to run.
BNDS=${BNDS:-"abs fs"}
SRCS=${SRCS:-"p s mixed"}
# l2 = conventional waveform L2 (misfit=0), the objective Rocha actually
# uses; kh = Kirchhoff-Helmholtz energy flux (misfit=2).  Both are driven
# with the SAME four components and data_weight=0, so the ONLY difference
# between the two columns is the objective -- and therefore the adjoint
# source: per-component point injection for L2 versus the tensorial
# traction->force / velocity->deformation-rate split for K-H.
MISFITS=${MISFITS:-"l2 kh"}

# Which potential-energy pairing: 2 = eps:C:eps~, 0 = Navier div/curl,
# -1 = EI_MODESEP (per-mode acoustic norm).  See the header block.
EI_GAMMA=${EI_GAMMA:-2}

# 3 runs in parallel x 16 threads = 48 of 64 hardware threads.
OMP=${OMP:-16}

# ------------------------------------------------------------------ model
DX=5
SIZEX=6000
SIZEZ=5000

VP_TRUE=2000;  VS_TRUE=1150;  RO_TRUE=2000
# Rocha: "an increase of 30% in velocity from the true model".  Velocities
# only -- density is identical in both models, so g_rho here is pure
# crosstalk and is a useful read in its own right.
VP_BG=2600;    VS_BG=1495;    RO_BG=2000

# ------------------------------------------------- acquisition (ROTATED)
XSRC=3000;  ZSRC=1250
XRCV=3000;  ZRCV=3750          # a SINGLE receiver, directly below the source

TMOD=3.0
DTRCV=0.001                    # = dt, so no rec_skipdt temporal comb
DT=0.0005                      # CFL: 0.606*dx/Vp_bg = 1.17e-3, so 5e-4 is safe

# 150 cells = 750 m of pad.  The ellipse spans z in [875,4125] and
# x in [1962,4038]; the pads cover z<750, z>4250, x<750, x>5250, so the
# whole structure of interest is clear of them.
NTAPER=150

# ------------------------------------------------------------ gradient
# Everything that would modify the gradient is OFF, so what comes out is
# the kernel as the adjoint actually produces it.  grad_taper=0 in
# particular means the 1/r injection singularities at the source and the
# receiver are left in; they dominate any shared colour scale and are
# masked out analytically in plot_rocha_fig9.py.
PARAM=2
COMP=_rvx,_rvz,_rtxz,_rtzz
KH_NORMAL=1

echo "========================================================="
echo " Rocha Fig 8/9, elastic, K-H energy-flux objective"
echo "   model   ${SIZEX} x ${SIZEZ} m, dx=${DX}"
echo "   src     (${XSRC}, ${ZSRC})    rcv (${XRCV}, ${ZRCV})   L=$((ZRCV-ZSRC)) m"
echo "   true    Vp=${VP_TRUE} Vs=${VS_TRUE} rho=${RO_TRUE}"
echo "   start   Vp=${VP_BG} Vs=${VS_BG} rho=${RO_BG}   (x1.30 velocities)"
echo "   matrix  misfit={${MISFITS}}  bnd={${BNDS}}  src={${SRCS}}  energy_ic={0,1,2}"
echo "   energy_gamma=${EI_GAMMA}"
echo "========================================================="

rm -rf ${WORK}
mkdir -p ${WORK}
cd ${WORK}

# ------------------------------------------------------------- models
echo "--- models (both homogeneous) ---"
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP_TRUE} cs0=${VS_TRUE} ro0=${RO_TRUE} \
    orig=0,0 file_base=model_true.su verbose=0
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP_BG} cs0=${VS_BG} ro0=${RO_BG} \
    orig=0,0 file_base=model_bg.su verbose=0

# ------------------------------------------------------------ wavelet
# A plain Ricker, closest to Rocha's 9 Hz peak (his Fig 10).  fp alone,
# with no fmin/fmax: makewave.c:105 DISCARDS fp when fmax is given, which
# would silently turn this into a flat 2-25 Hz band and invalidate the
# Fresnel radius quoted in the header.
echo "--- wavelet: Ricker fp=10 Hz ---"
makewave fp=10 dt=${DT} nt=2048 file_out=wave.su t0=0.15 verbose=0

# traveltimes, for reading the residual plot
python3 - <<PY
L = ${ZRCV} - ${ZSRC}
for nm, vt, vb in (('P', ${VP_TRUE}, ${VP_BG}), ('S', ${VS_TRUE}, ${VS_BG})):
    print('    %s direct: obs %.3f s   syn %.3f s   separation %.3f s'
          % (nm, L / vt, L / vb, L / vt - L / vb))
print('    ellipse semi-axes: a=%.0f  b=%.0f  c=%.0f m (mode independent)'
      % (1.30 * L / 2.0, (((1.30 * L / 2.0) ** 2 - (L / 2.0) ** 2) ** 0.5), L / 2.0))
PY

# --------------------------------------------------------------- runs
# src_type: 1=P(explosion)  5=S-pot  6=Fx
src_type_of () {
    case "$1" in
        p)     echo 1 ;;
        s)     echo 5 ;;
        mixed) echo 6 ;;
        *)     echo "ERROR: unknown source variant '$1'" >&2; exit 1 ;;
    esac
}

# top=4 absorbing (Rocha 9a,c,e,g) / top=1 free surface (Rocha 9b,d,f,h)
top_of () {
    case "$1" in
        abs) echo 4 ;;
        fs)  echo 1 ;;
        *)   echo "ERROR: unknown boundary variant '$1'" >&2; exit 1 ;;
    esac
}

misfit_of () {
    case "$1" in
        l2) echo 0 ;;
        kh) echo 2 ;;
        *)  echo "ERROR: unknown misfit variant '$1'" >&2; exit 1 ;;
    esac
}

run_grad () {   # run_grad <cell_dir> <misfit_tag> <energy_ic> <top> <src_type>
    local CELL=$1 MF=$2 EI=$3 TOP=$4 ST=$5
    local DIR=${CELL}/${MF}/ei${EI}
    local MFNUM=$(misfit_of ${MF})
    mkdir -p ${DIR}
    ( cd ${DIR}
      for f in ../../../model_*.su ../../../wave.su ../../obs_*.su; do ln -sf $f .; done
      OMP_NUM_THREADS=${OMP} ${FDELFWI}/test_gradient_visual \
        file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
        file_src=wave.su file_rcv=syn \
        ischeme=3 iorder=4 src_type=${ST} \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} grad_taper=0 res_taper=0 aper_taper=0 \
        left=4 right=4 top=${TOP} bottom=4 \
        chk_skipdt=100 chk_base=chk_g${EI} \
        param=${PARAM} scaling=0 precond=0 smooth_grad=0 \
        vp_min=1000 vp_max=4500 vs_min=500 vs_max=2500 \
        rho_min=1500 rho_max=2500 \
        data_weight=0 misfit=${MFNUM} kh_normal=${KH_NORMAL} \
        comp=${COMP} \
        file_obs=obs energy_ic=${EI} energy_gamma=${EI_GAMMA} \
        out_base=g${EI} > ../log_ei${EI}.txt 2>&1 )
}

for BND in ${BNDS}; do
  TOP=$(top_of ${BND})
  for SV in ${SRCS}; do
    ST=$(src_type_of ${SV})
    CELL=${BND}_${SV}
    echo ""
    echo "=== cell ${CELL}  (top=${TOP}, src_type=${ST}) ==========="
    mkdir -p ${CELL}

    # ---------------------------------------------------- observed data
    # fdelfwi_fwd, NOT fdelmodc: the two have incompatible boundary
    # implementations and mixing them leaves a ~1% direct-wave residual
    # even at the true model.  fdelfwi_fwd also zero-pads fileno, so the
    # files land as obs_000_r*.su with no renaming step.
    ( cd ${CELL}
      for f in ../model_*.su ../wave.su; do ln -sf $f .; done
      OMP_NUM_THREADS=$(nproc) ${FDELFWI}/fdelfwi_fwd \
        file_cp=model_true_cp.su file_cs=model_true_cs.su \
        file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs \
        ischeme=3 iorder=4 src_type=${ST} \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} left=4 right=4 top=${TOP} bottom=4 \
        > log_fwd.txt 2>&1 )
    echo "    observed data: $(ls ${CELL}/obs_000_r*.su | wc -l) components"

    # ------------------------------------------------------- gradients
    # The observed data depend only on (boundary, source), so both
    # objectives reuse the same obs_000_r*.su.  That is what makes the
    # l2/kh comparison exact: identical data, identical model, identical
    # components -- only dJ/dd differs.
    for MF in ${MISFITS}; do
      echo "    [${MF}] energy_ic 0,1,2 in parallel (${OMP} threads each)"
      date +"      start %H:%M:%S"
      run_grad ${CELL} ${MF} 0 ${TOP} ${ST} &  P0=$!
      run_grad ${CELL} ${MF} 1 ${TOP} ${ST} &  P1=$!
      run_grad ${CELL} ${MF} 2 ${TOP} ${ST} &  P2=$!
      wait ${P0} ${P1} ${P2}
      date +"      done  %H:%M:%S"

      for EI in 0 1 2; do
          echo "      ei=${EI}: $(grep -E '=== Total misfit' \
                                   ${CELL}/${MF}/log_ei${EI}.txt \
                                   | tail -1 | sed 's/.*=== //')"
      done

      # partition of unity -- the one regression check that must hold no
      # matter what the physics does
      python3 - "${CELL}/${MF}" <<'PY'
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
    done
  done
done

echo ""
echo "========================================================="
echo " Done.  Results in ${HERE}/${WORK}/"
echo " Next:  python3 plot_rocha_fig9.py"
echo "========================================================="
