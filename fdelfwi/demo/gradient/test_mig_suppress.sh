#!/bin/bash
# test_mig_suppress.sh -- suppress the migration kernel in a long-offset
#                         transmission gradient, using energy_gamma=0 with
#                         energy_alpha=5.
#
# =====================================================================
# THE EXPERIMENT
# =====================================================================
# One source at (1000,2500) and one receiver at (5000,1500) in a homogeneous
# model: 4000 m of horizontal offset with 1000 m of rise, so the path is
# 4123 m long and DIPS 14.04 deg.  The background is 30% FAST (velocities
# x1.30), so the synthetic direct arrivals come in early and the residual is
# a pair of time-shifted direct wavetrains -- one P, one S.
#
# The tilt is not cosmetic.  With source and receiver at a common depth the
# whole problem is mirror-symmetric about that depth, the transmission
# corridor is exactly horizontal, and it is then impossible to tell a
# correctly-placed transmission kernel from a horizontally-mis-mapped
# arrival -- the two lie on top of each other.  Breaking the symmetry
# separates them, and it also puts the kernel at intermediate opening angles
# instead of only the 0/180 deg extremes.  Every mask below is therefore
# built from true point-to-segment distance and from d_s+d_r, not from
# "distance in z", so the geometry is not assumed to be flat anywhere.
#
# Backpropagating that residual and correlating against the forward field
# puts energy in two geometrically distinct places:
#
#   TRANSMISSION ("banana")  along the source-receiver line.  This is the
#       update we want: it is what moves a smooth background velocity.
#       Half-width is the Fresnel radius, RF = sqrt(lam_P L)/2 = 579 m
#       with lam_P = Vp_bg/fp = 325 m.
#
#   MIGRATION / CONVERTED loci  on the isochrone ellipses
#       d_s/V1 + d_r/V2 = t_obs.  These are the artifact: there is no
#       reflector anywhere in either model, so every bit of energy off the
#       transmission corridor is a mis-mapped arrival.  For the same-mode
#       pairs the dominant one is d_s + d_r = 1.30 L = 5360 m, an ellipse
#       centred on (3000,2000) with a=2680 m, b=1712 m, and -- because its
#       major axis IS the source-receiver segment -- rotated by the dip.
#
# A homogeneous model is the right test bed precisely BECAUSE it has no
# reflector: anything the filter removes from the ellipses is provably
# artifact, and anything it removes from the banana is provably damage.
# There is no interpretation step and nothing to argue about.
#
# =====================================================================
# WHAT IS BEING TESTED
# =====================================================================
# energy_imaging.c splits the gradient by the opening angle between the
# forward and adjoint fields, using the two halves of an elastic energy
# inner product, and then masks with Rocha, Tanushev & Sava (2016) eq 45:
#
#     a_tomo(x) = exp(-alpha * c2(x)),   c2 = |k-p| / (|k+p| + |k-p| + eps)
#
# with k,p the RMS-normalised kinetic and potential accumulators.  Two
# knobs, and this script fixes both at the values the alpha ROC study
# picked out:
#
#   energy_gamma=0   V_0 = (lam+2mu) div.div~ + mu curl.curl~, the Navier
#                    pairing.  At gamma=2 (the historical default) the
#                    clean weight is modulated by (lam -+ 2mu cos2theta),
#                    which CHANGES SIGN -- moving energy into the wrong
#                    output.  THIS MODEL IS AN UNUSUALLY HARSH TEST OF
#                    THAT: Vs=1700 against Vp=2500 is a Poisson ratio of
#                    0.07, so lam/mu = 0.163 and the sign change sits at
#                    2theta = 85 deg, right in the middle of the range the
#                    kernel actually occupies.  At gamma=0 the modulation
#                    and the Vs/Vp ratio drop out entirely.
#
#   energy_alpha     ALPHAS="5 2" by default, one filtered run each.  From
#                    the alpha scan on the vertical-offset cells, alpha=5-10
#                    reached within ~1 point of the purity ceiling while
#                    keeping the kernel width at 1.05 x RF, whereas
#                    alpha=50-100 bought those last few points by
#                    COLLAPSING the kernel to 0.52 x RF and discarding 97%
#                    of the transmission energy -- high purity by
#                    annihilation, which is useless to an optimiser.  The
#                    pair 5 and 2 brackets the knee of that curve from
#                    BELOW, which is the half the earlier scan never
#                    sampled: at alpha=2 the mask is gentle enough that
#                    retained energy should stay high, and the question is
#                    whether purity holds up with it.
#
# =====================================================================
# GEOMETRY CAVEAT -- READ BEFORE QUOTING ANY PURITY NUMBER
# =====================================================================
# ntaper=150 at dx=5 is a 750 m pad, so the usable region is
# x in [750,5250], z in [750,3450].  The transmission banana is fully
# inside it -- that is what SIZEZ=4200 buys (see the note on the model).
#
# The artifact ellipse is NOT.  Tilted, it spans x in [367,5633] and
# z in [216,3784], so it is clipped on all four sides.  Containing it
# would need roughly 8600 x 6800 m with a 1500 m pad, 2.8x the cost.
#
# CONSEQUENCE: artifact-energy shares here are computed over a CLIPPED
# artifact set and are a LOWER BOUND on what the filter removes.  The
# banana numbers, the kernel widths and the retained fractions are
# unaffected and can be quoted as they stand.
#
# =====================================================================
# COMPONENTS
# =====================================================================
# All five receiver types are RECORDED (vx,vz,txx,tzz,txz) but only
# comp=_rvx drives the adjoint.  With a SINGLE channel the question of
# commensurate units does not arise at all, so data_weight=0 is exact
# rather than merely defensible -- any weight would be one global scalar
# and could not change the gradient's shape.  (Mixing in the Pa channels
# at data_weight=0 would instead make the misfit numerically all-stress.)
#
# vx and not vz: the source is src_type=6, Fx, so vx is the component the
# source actually radiates strongly along this near-horizontal path, and
# it carries both the P and the S direct arrival.  The cost is that the
# adjoint source is now a single-component vector field, which is a
# HARDER test of the filter, not an easier one: a one-component injection
# radiates its own spurious lobes, and those lobes are off the
# transmission corridor, i.e. they land in the artifact set.
#
# =====================================================================
# RUNS -- three propagations, not six
# =====================================================================
#   obs/   fdelfwi_fwd on the TRUE model.  Observed data + TRUE-model
#          forward snapshots.  fdelfwi_fwd and NOT fdelmodc: their
#          boundary implementations are incompatible and mixing them
#          leaves a ~1% direct-wave residual even at the true model.
#   ei0/   energy_ic=0.  Unfiltered gradient + INITIAL-model forward
#          snapshots + ADJOINT snapshots.
#   a5/    energy_ic=1 gamma=0 alpha=5.  Filtered gradient, plus the
#   a2/    ei_{kin,pot,tomo,migr} maps.  One directory per alpha.
#
# All of them PROPAGATE IDENTICALLY -- energy_ic only touches the two
# accumulators and a post-hoc pointwise weight -- so snapshots are
# written by ei0 only, and energy_ic=0 is bit-identical to an
# un-instrumented run.  The REJECTED (migration) gradient and any other
# alpha are then recoverable OFFLINE from any run's kin/pot maps (they are
# alpha-INDEPENDENT -- alpha enters only in the final mask), exactly and
# for free, because the weight is a pure function of those two fields and
# a pointwise scalar commutes with the Lame -> velocity chain rule.  That
# offline path reproduces in-code runs to 5e-5 (see scan_alpha.py), so
# there is no reason to spend a propagation on it.
#
# =====================================================================
# Usage
#   ./test_mig_suppress.sh                  full run, ~30 min
#   GAMMA=2 ./test_mig_suppress.sh          control: mask without gamma=0
#   ALPHAS="10 5 2" ./test_mig_suppress.sh  more mask strengths (concurrent)
#   SKIP_OBS=1 ./test_mig_suppress.sh       reuse existing observed data
# =====================================================================

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}

HERE=$(cd "$(dirname "$0")" && pwd)
cd "${HERE}"

WORK=${WORK:-mig_suppress}
GAMMA=${GAMMA:-0}
ALPHAS=${ALPHAS:-"5 2"}         # one filtered run per value, all concurrent
SKIP_OBS=${SKIP_OBS:-0}

# ------------------------------------------------------------------ model
# SIZEZ is 4200 and not 3500 for one reason: the source sits at z=2500 and
# the transmission kernel has a Fresnel half-width of RF=579 m, so it
# reaches z=3079.  At SIZEZ=3500 with a 750 m pad the usable region stops
# at z=2750 and the kernel would be 330 m INSIDE the sponge -- the width
# diagnostic would then be measuring the absorbing layer.  Growing the
# model DOWNWARD moves the pad edge to z=3450 and leaves every specified
# coordinate exactly where it is, because the origin stays at (0,0).
SIZEX=6000;  SIZEZ=4200;  DX=5
VP_TRUE=2500;  VS_TRUE=1700;  RO_TRUE=2000
VP_BG=3250;    VS_BG=2210;    RO_BG=2000          # x1.30, rho unchanged

# Tilted path: 4000 m of horizontal offset with 1000 m of rise.
XSRC=1000;  ZSRC=2500
XRCV=5000;  ZRCV=1500

SRC_TYPE=6                      # Fx, horizontal body force (fdelmodc.c:144)
FP=10                           # Ricker peak frequency

# CFL: 0.606*dx/Vp_bg = 9.32e-4, so 5e-4 is safe (Courant 0.325).
# Dispersion: the slowest wave is Vs_true=1700 -> 34 cells per wavelength.
DT=0.0005
# TMOD is overridable only so that TMOD=0.4 gives a fast smoke test of the
# parameter list.  At anything below ~2.5 s the S arrival is cut off and the
# gradient is not the one this script is about.
TMOD=${TMOD:-3.0}               # S arrives at 2.353 s in the true model
# dtrcv = dt deliberately.  Any dtrcv > dt gives rec_skipdt > 1, which
# combs the residual in time and puts a visible checkerboard on the
# ADJOINT SNAPSHOTS this script exists to produce.  It does not reach the
# gradient, but there is no reason to look at it.
DTRCV=0.0005

# 750 m pad = 2.3 dominant P wavelengths.  This CANNOT simply be raised:
# the source is at x=1000, so a thicker pad eats its clearance from the
# left edge (250 m at ntaper=150 already).  Thickening the sponge requires
# growing the model in x as well, which moves the specified coordinates.
# See the GEOMETRY CAVEAT above.
NTAPER=150

# ------------------------------------------------------------------ gradient
# Everything that would modify the gradient is OFF, so what comes out is
# the kernel as the adjoint actually produces it, and the only difference
# between ei0 and the a* runs is the energy filter.  grad_taper=0 in particular
# leaves the 1/r injection singularities at the source and receiver in;
# they dominate any shared colour scale and are masked analytically in the
# plotting script.
PARAM=2
COMP=_rvx                       # vx only -- see COMPONENTS above
MISFIT=0                        # L2

# ------------------------------------------------------------------ snapshots
# 13 snapshots at 0.2 s, spanning the window that contains every arrival
# of interest (wavelet t0=0.15, latest arrival S_true at 2.353 s).
TSNAP1=0.2;  TSNAP2=2.6;  DTSNAP=0.2
# vx/vz are the raw fields; pp/ss are div(v) and curl(v)
# (fdelmodc/writeSnapTimes.c:191,199) -- a genuine model-domain P/S
# separation, and the natural QC for gamma=0 specifically, whose two
# halves ARE the div path and the curl path.  They are 2nd-order
# differences against a 4th-order kernel, which is fine for looking at.
SNA="sna_type_vx=1 sna_type_vz=1 sna_type_pp=1 sna_type_ss=1"
SNAPSCHED="tsnap1=${TSNAP1} tsnap2=${TSNAP2} dtsnap=${DTSNAP} ${SNA}"

# Every gradient run goes at once -- 1 unfiltered + one per alpha -- so the
# per-run thread count is the budget divided by that count, not a fixed
# number.  Hard-coding it would oversubscribe the moment a third alpha is
# added, and oversubscription on a memory-bound FD kernel is a net loss.
OMP_ONE=${OMP_ONE:-64}          # whole-node budget; obs/ runs alone with it
NRUN=$(( 1 + $(echo ${ALPHAS} | wc -w) ))
OMP_EACH=$(( OMP_ONE / NRUN ))
[ ${OMP_EACH} -lt 1 ] && OMP_EACH=1

# Grid size, derived -- never written twice.  The diagnostics block below is
# handed these, so changing SIZEZ cannot leave a stale NZ behind.
NXG=$(( SIZEX / DX + 1 ))
NZG=$(( SIZEZ / DX + 1 ))

echo "========================================================="
echo " Migration-kernel suppression, long-offset transmission"
echo "   model    ${SIZEX} x ${SIZEZ} m, dx=${DX}  (${NXG} x ${NZG})"
echo "   src      (${XSRC}, ${ZSRC})  src_type=${SRC_TYPE} (Fx)"
echo "   rcv      (${XRCV}, ${ZRCV})  dx=$((XRCV-XSRC)) dz=$((ZRCV-ZSRC)) m"
echo "   true     Vp=${VP_TRUE} Vs=${VS_TRUE} rho=${RO_TRUE}"
echo "   start    Vp=${VP_BG} Vs=${VS_BG} rho=${RO_BG}   (x1.30 velocities)"
echo "   adjoint  comp=${COMP}   misfit=L2   data_weight=0"
echo "   filter   energy_gamma=${GAMMA}  energy_alpha={${ALPHAS}}"
echo "   runs     ${NRUN} concurrent x ${OMP_EACH} threads"
echo "========================================================="

mkdir -p ${WORK}
cd ${WORK}

# --------------------------------------------------------------- models
echo "--- models (both homogeneous) ---"
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP_TRUE} cs0=${VS_TRUE} ro0=${RO_TRUE} \
    orig=0,0 file_base=model_true.su verbose=0
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP_BG} cs0=${VS_BG} ro0=${RO_BG} \
    orig=0,0 file_base=model_bg.su verbose=0

# -------------------------------------------------------------- wavelet
# fp alone, with no fmin/fmax: makewave.c:105 DISCARDS fp when fmax is
# given, which would silently turn this into a flat band and invalidate
# the Fresnel radius the plotting masks are built from.
echo "--- wavelet: Ricker fp=${FP} Hz ---"
makewave fp=${FP} dt=${DT} nt=4096 file_out=wave.su t0=0.15 verbose=0

# ------------------------------------------------- geometry, for reading
python3 - <<PY
import math
xs, zs, xr, zr = ${XSRC}.0, ${ZSRC}.0, ${XRCV}.0, ${ZRCV}.0
pad = ${NTAPER} * float(${DX})
L   = math.hypot(xr - xs, zr - zs)
phi = math.atan2(zr - zs, xr - xs)
print('    path length L = %.1f m   dip %.2f deg   midpoint (%.0f, %.0f)'
      % (L, math.degrees(phi), 0.5*(xs+xr), 0.5*(zs+zr)))
for nm, vt, vb in (('P', ${VP_TRUE}, ${VP_BG}), ('S', ${VS_TRUE}, ${VS_BG})):
    tt, tb = L / vt, L / vb
    print('    %s direct: obs %.3f s   syn %.3f s   shift %.3f s  '
          '(= %.1f cycles at %g Hz)'
          % (nm, tt, tb, tt - tb, (tt - tb) * ${FP}, ${FP}))

lam = ${VP_BG} / float(${FP})
rf  = (lam * L) ** 0.5 / 2.0
# Banana = the segment dilated by RF.  Its bounding box needs the tilt:
# the perpendicular offset has components RF*(sin phi, cos phi).
bx, bz = rf * abs(math.sin(phi)), rf * abs(math.cos(phi))
print('    Fresnel radius RF = sqrt(%.0f*%.0f)/2 = %.0f m' % (lam, L, rf))
print('      banana bbox: x in [%.0f, %.0f], z in [%.0f, %.0f]'
      % (min(xs, xr) - bx, max(xs, xr) + bx,
         min(zs, zr) - bz, max(zs, zr) + bz))

# Artifact isochrone d_s + d_r = 1.30 L: an ellipse with the s-r segment
# as its major axis, hence ROTATED by phi.
a = 1.30 * L / 2.0
c = L / 2.0
b = (a * a - c * c) ** 0.5
hx = math.hypot(a * math.cos(phi), b * math.sin(phi))
hz = math.hypot(a * math.sin(phi), b * math.cos(phi))
cx, cz = 0.5 * (xs + xr), 0.5 * (zs + zr)
print('    artifact ellipse d_s+d_r=%.0f: a=%.0f b=%.0f rotated %.1f deg'
      % (2 * a, a, b, math.degrees(phi)))
print('      bbox: x in [%.0f, %.0f], z in [%.0f, %.0f]'
      % (cx - hx, cx + hx, cz - hz, cz + hz))

print('    usable region (pads %.0f m): x in [%.0f, %.0f], z in [%.0f, %.0f]'
      % (pad, pad, ${SIZEX} - pad, pad, ${SIZEZ} - pad))

# Explicit pass/fail on each mask, so a clipped run cannot be mistaken for
# a clean one when the numbers are read months later.
ok_ban = (min(xs, xr) - bx >= pad and max(xs, xr) + bx <= ${SIZEX} - pad and
          min(zs, zr) - bz >= pad and max(zs, zr) + bz <= ${SIZEZ} - pad)
ok_ell = (cx - hx >= pad and cx + hx <= ${SIZEX} - pad and
          cz - hz >= pad and cz + hz <= ${SIZEZ} - pad)
print('      banana  %s' % ('CLEAR of the pads' if ok_ban else
                            '*** CLIPPED -- widths are meaningless ***'))
print('      ellipse %s' % ('CLEAR of the pads' if ok_ell else
                            'CLIPPED -- artifact shares are a LOWER BOUND'))

lm = ${RO_BG} * (${VP_BG}**2 - 2*${VS_BG}**2)
mu = ${RO_BG} * ${VS_BG}**2
print('    lam/mu = %.3f  ->  gamma=2 weight changes sign at 2theta = %.1f deg'
      % (lm / mu, math.degrees(math.acos(lm / (2 * mu)))))
PY

# --------------------------------------------------------- observed data
if [ "${SKIP_OBS}" = "1" ] && [ -f obs/obs_000_rvx.su ]; then
    echo "--- observed data: reusing obs/ (SKIP_OBS=1) ---"
else
    echo "--- observed data + TRUE-model forward snapshots ---"
    mkdir -p obs
    ( cd obs
      for f in ../model_*.su ../wave.su; do ln -sf $f .; done
      OMP_NUM_THREADS=${OMP_ONE} ${FDELFWI}/fdelfwi_fwd \
        file_cp=model_true_cp.su file_cs=model_true_cs.su \
        file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs \
        ischeme=3 iorder=4 src_type=${SRC_TYPE} \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        rec_type_txz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 \
        file_snap=snap_true ${SNAPSCHED} \
        > log_fwd_true.txt 2>&1 )
    echo "    components: $(ls obs/obs_000_r*.su | wc -l)"
    echo "    snapshots:  $(ls obs/snap_true_s*.su 2>/dev/null | wc -l) files"
fi

# ------------------------------------------------------------- gradients
# run_grad <dir> <energy_ic> <extra args...>
#
# The directory is an argument rather than being derived from energy_ic,
# because several runs now share energy_ic=1 and differ only in alpha.  Each
# one keeps its own checkpoint base too: chk files are per-run scratch and
# two concurrent runs sharing a base would overwrite each other's wavefield.
run_grad () {
    local DIR=$1; local EI=$2; shift 2
    mkdir -p ${DIR}
    ( cd ${DIR}
      for f in ../model_*.su ../wave.su ../obs/obs_*.su; do ln -sf $f .; done
      OMP_NUM_THREADS=${OMP_EACH} ${FDELFWI}/test_gradient_visual \
        file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
        file_src=wave.su file_rcv=syn \
        ischeme=3 iorder=4 src_type=${SRC_TYPE} \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        rec_type_txz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} grad_taper=0 res_taper=0 aper_taper=0 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=100 chk_base=chk_${DIR} \
        param=${PARAM} scaling=0 precond=0 smooth_grad=0 \
        vp_min=1000 vp_max=4500 vs_min=500 vs_max=2800 \
        rho_min=1500 rho_max=2500 \
        data_weight=0 misfit=${MISFIT} \
        comp=${COMP} \
        file_obs=obs energy_ic=${EI} \
        out_base=g${EI} "$@" > ../log_${DIR}.txt 2>&1 )
}

echo ""
echo "--- gradients: ei0 (unfiltered, + snapshots) and a{${ALPHAS}} (filtered) ---"
date +"    start %H:%M:%S"

# Snapshots on ei0 only: the filtered runs propagate identically, so a second
# copy would be bit-for-bit redundant I/O.
PIDS=""
run_grad ei0 0 file_snap=snap_bg file_snap_adj=snap_adj ${SNAPSCHED} &
PIDS="${PIDS} $!"
for A in ${ALPHAS}; do
    run_grad a${A} 1 energy_gamma=${GAMMA} energy_alpha=${A} &
    PIDS="${PIDS} $!"
done
wait ${PIDS}
date +"    done  %H:%M:%S"

# ----------------------------------------------------------- diagnostics
echo ""
echo "--- diagnostics ---"
for A in ${ALPHAS}; do
    echo "  alpha=${A} filter report:"
    grep 'energy_ic:' log_a${A}.txt | tail -2 | sed 's/^/    /' || true
done
echo "  wall times:"
grep -H 'Wall time' log_ei0.txt $(for A in ${ALPHAS}; do echo log_a${A}.txt; done) \
    | sed 's/^/    /' || true
echo "  snapshots: fwd_true $(ls obs/snap_true_s*.su 2>/dev/null | wc -l)" \
     " fwd_bg $(ls ei0/snap_bg_s*.su 2>/dev/null | wc -l)" \
     " adj $(ls ei0/snap_adj_s*.su 2>/dev/null | wc -l)"

# How much of the gradient did the filter keep, and WHERE.  The banana and
# the ellipse are built analytically from the geometry at the top of this
# file -- no thresholding, no hand-drawn polygons.
python3 - "${GAMMA}" "${NXG}" "${NZG}" "${DX}" \
         "${XSRC}" "${ZSRC}" "${XRCV}" "${ZRCV}" \
         "${VP_BG}" "${FP}" "${NTAPER}" ${ALPHAS} <<'PY'
import sys
import numpy as np
sys.path.insert(0, '/rcp3/software/codes/OpenSource_SL10/fdelfwi/tests/kh_misfit')
import su_io

# Everything geometric comes in from the shell, so there is exactly one
# definition of the model and the acquisition in this file.  A stale NZ here
# silently reshapes the gradient and every number below becomes garbage,
# which is precisely the failure that hard-coding invites.
gamma = sys.argv[1]
NX, NZ = int(sys.argv[2]), int(sys.argv[3])
DX = float(sys.argv[4])
XSRC, ZSRC, XRCV, ZRCV = [float(v) for v in sys.argv[5:9]]
VP_BG, FP = float(sys.argv[9]), float(sys.argv[10])
PAD = int(sys.argv[11]) * DX
ALPHAS = sys.argv[12:]

xx = (np.arange(NX) * DX)[:, None] * np.ones((1, NZ))
zz = np.ones((NX, 1)) * (np.arange(NZ) * DX)[None, :]
d_s = np.hypot(xx - XSRC, zz - ZSRC)
d_r = np.hypot(xx - XRCV, zz - ZRCV)

# Path frame.  s runs along the source-receiver line, q perpendicular to it.
# The path DIPS, so neither "distance in z" nor a z-profile means anything
# here; both would measure a sheared slice through the kernel and report a
# width inflated by 1/cos(dip).
bx, bz = XRCV - XSRC, ZRCV - ZSRC
L = np.hypot(bx, bz)
s = ((xx - XSRC) * bx + (zz - ZSRC) * bz) / L
q = ((xx - XSRC) * (-bz) + (zz - ZSRC) * bx) / L
# True point-to-SEGMENT distance: perpendicular offset inside the span,
# endpoint distance outside it.  This is what makes the banana mask a
# dilated segment rather than a dilated infinite line.
d_seg = np.where((s >= 0.0) & (s <= L), np.abs(q), np.minimum(d_s, d_r))

RF = np.sqrt(VP_BG / FP * L) / 2.0

interior = ((xx > PAD) & (xx < NX * DX - PAD) &
            (zz > PAD) & (zz < NZ * DX - PAD))
# Exclude the injection singularities: they are 1/r, not kernel.
total = interior & (d_s > 400.0) & (d_r > 400.0)
banana = (d_seg < RF) & total
# Same-mode isochrone for a 1.30x fast background.  Band half-width is
# half a dominant wavelength, the same convention the alpha study used.
ell = (np.abs(d_s + d_r - 1.30 * L) < 0.5 * VP_BG / FP) & total & ~banana

# Width is measured on a thin strip ACROSS the path at its midpoint, so it
# is a true perpendicular half-width independent of the dip.
mid = (np.abs(s - 0.5 * L) < DX) & (np.abs(q) < 1500.0)


def width(g):
    e = g[mid] ** 2
    if e.sum() <= 0.0:
        return np.nan
    return np.sqrt(np.sum(e * q[mid] ** 2) / e.sum())


print('    path L = %.1f m, dip %.2f deg, RF = %.0f m'
      % (L, np.degrees(np.arctan2(bz, bx)), RF))
print('    masks: banana %d cells, ellipse %d cells, total %d cells,'
      ' mid-strip %d cells'
      % (banana.sum(), ell.sum(), total.sum(), mid.sum()))

cases = [('ei0', 0, 'unfiltered')]
cases += [('a%s' % a, 1, 'gamma=%s alpha=%s' % (gamma, a)) for a in ALPHAS]

print('    %-4s %12s %12s %10s %10s %9s   %s'
      % ('par', 'E_banana', 'E_ellipse', 'ban/ell', 'retained', 'width/RF',
         'case'))
for par in ('vp', 'vs'):
    rows = []
    for d, ei, label in cases:
        dat, _ = su_io.read('%s/g%d_%s_raw.su' % (d, ei, par))
        g = dat.astype(np.float64).reshape(NX, NZ)
        rows.append((np.sum(g[banana] ** 2), np.sum(g[ell] ** 2),
                     np.sum(g[total] ** 2), width(g), label))
    et0, r0b, r0e = rows[0][2], rows[0][0], rows[0][1]
    for eb, ea, et, wid, label in rows:
        print('    %-4s %12.4g %12.4g %10.2f %9.1f%% %9.2f   %s'
              % (par, eb, ea, eb / ea if ea > 0 else np.nan,
                 100.0 * et / et0 if et0 > 0 else np.nan, wid / RF, label))
    for eb, ea, et, wid, label in rows[1:]:
        if r0e > 0 and ea > 0 and r0b > 0:
            # Two numbers, both needed: the ratio improvement says the
            # filter cleaned up, the banana-kept fraction says whether it
            # did so by removing artifact or by removing everything.
            print('         %-18s ban/ell x%.2f   banana kept %5.1f%%'
                  % (label, (eb / ea) / (r0b / r0e), 100.0 * eb / r0b))
PY

echo ""
echo "========================================================="
echo " Done.  Results in ${HERE}/${WORK}/"
echo "   ei0/g0_{vp,vs}_raw.su   unfiltered gradient"
for A in ${ALPHAS}; do
echo "   a${A}/g1_{vp,vs}_raw.su   gamma=${GAMMA} alpha=${A} filtered"
echo "   a${A}/g1_shot000_ei_{kin,pot,tomo,migr}.su"
done
echo "   snapshots, each _s{vx,vz,pp,ss}.su with ${DTSNAP}s traces:"
echo "     obs/snap_true_*   true forward"
echo "     ei0/snap_bg_*     initial-model forward"
echo "     ei0/snap_adj_*    adjoint"
echo "========================================================="
