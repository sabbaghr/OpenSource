#!/bin/bash
# test_lag_reflector.sh -- does the LAG SLOPE separate a tomographic kernel
#                          from a migration kernel?
#
# This is a KINEMATIC and KERNEL-SEPARATION test.  It asks one question and
# does not answer a second: whether a kernel's slope in the lag cube is where
# ray theory says it should be, and whether a cone filter on that slope puts
# the two kinds of kernel in different panels.  It says NOTHING about whether
# the filtered gradient inverts better.  No inversion is run.
#
# WHY A SLOPE SEPARATES THEM AT ALL
# ---------------------------------
# With the symmetric lag convention (forward at -tau, adjoint at +tau) a
# residual event at data time T_D images at
#
#     tau(x) = 0.5 [ T_D - t_a(S,x) - t_b(x,R) ]                        (1)
#     p(x)   = grad tau = -0.5 [ n_a/v_a + n_b/v_b ]                    (2)
#
# with n_a pointing from S towards x and n_b from R towards x.  (2) is the
# whole basis of the test.  For a SAME-MODE pair it reduces to
#
#     |p| = cos(theta)/v,   2theta = the opening angle
#
# so the two extremes are
#
#     transmission (2theta = 180)   |p| = 0        TOMOGRAPHIC
#     backscatter  (2theta = 0)     |p| = 1/v      SPECULAR
#
# and a cone cutoff at 0.20-0.30 ms/m cuts Vp = 2000 at 2theta = 143-113 deg
# and Vs = 1200 at 166-136 deg.  Note already that the two cuts are NOT the
# same angle: the same filter is much closer to pure transmission for S than
# for P.  That is a consequence of (2), not a tuning choice, and it is one of
# the things measured below.
#
# THE TWO RUNS
# ------------
# The error is in the BACKGROUND MODEL in both runs -- never in the data.
#
#   run A  MIGRATION-dominated.  true = flat reflector.  background = the
#          homogeneous half space, NO reflector.  The two models are identical
#          above the reflector, so the direct P and direct S cancel EXACTLY and
#          the residual is nothing but the reflected wavefield.  No mute is
#          needed or used.  Expected: isochrons tangent to the reflector, at
#          the specular slopes 0.416 (P/P) and 0.693 (S/S) ms/m.
#
#   run B  TOMOGRAPHY-dominated.  background = the reflector model.  true = the
#          same reflector PLUS two smooth Gaussian anomalies, sigma = 300 m,
#          one on the P down-leg of the PP reflection and one on its up-leg.
#          Both models have the same reflector, so the reflection is not
#          created by the error -- only retimed.  Expected: rabbit-ear kernels
#          lying ALONG the reflected legs, where (2) gives |p| = 0.
#          The direct P and direct S residual is MUTED, because an unmuted
#          direct residual puts a second, high-slope event at the same sites
#          (0.29 ms/m for a P/P direct pairing, 0.48 for S/S) and would be
#          counted as a filter failure when it is really a different event.
#
# WHAT THE PREDICTION TABLE ALREADY SAYS, BEFORE ANY DATA
# -------------------------------------------------------
# lag_refl_rays.py evaluates (1)-(2) for every residual event crossed with
# every leg-mode pair -- 16 combinations, of which 12 are MODE CROSSTALK (a PP
# residual imaged with S legs, and so on).  Every single one of them, crosstalk
# included, comes out at |p| = 0.32 to 0.60 ms/m.  So:
#
#     the cone CANNOT distinguish a true specular isochron from a mode-
#     crosstalk isochron.  Both are high slope.  The reject panel will hold
#     both, and the figures below label it "high |p|", not "migration".
#
# What the cone CAN do, if (2) holds in the measured cubes, is separate either
# of those from a transmission kernel at |p| ~ 0.  That is the claim under
# test, and it is narrower than "separates reflections from crosstalk".
#
# THE ONE KINEMATIC HAZARD: direct S against PP
# ---------------------------------------------
# At this geometry direct S lands 134 ms before PP, while the wavelet envelope
# is 132 ms wide at the 10% level and 156 ms at the 5% level.  The two arrivals
# are therefore MARGINALLY separated at best and overlapping at worst, and that
# decides whether the run B mute can remove the direct S without also cutting
# into the PP residual it is supposed to keep.  The `model` stage measures the
# overlap on the actual gather and refuses to continue if it is too large.
# This is checked on data, not asserted from the wavelet length.
#
# STAGES
#   ./test_lag_reflector.sh model     models, wavelet, observed data, and the
#                                     arrival / overlap / boundary QC.  Stops
#                                     before any gradient.
#   ./test_lag_reflector.sh grad      the lag gradients for both runs
#   ./test_lag_reflector.sh anal      slant stacks, the cone filter, metrics
#   ./test_lag_reflector.sh all       all three
#
# Re-running `model` rebuilds the work directory from scratch.  The later
# stages do not, so a gradient can be re-analysed without being recomputed.

set -e
set -o pipefail
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}
export OMP_NUM_THREADS=16

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${HERE}/lag_reflector

STAGE=${1:-all}
case ${STAGE} in
    model|grad|anal|all) ;;
    *) echo "usage: $0 [model|grad|anal|all]" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Settings.  Everything not commented is taken UNCHANGED from
# test_lag_diagonal.sh / case_fx_geo so that the two experiments are
# comparable; the exceptions are marked CHANGED and justified.
# ---------------------------------------------------------------------------
DX=6; SIZEX=5004; SIZEZ=3000
VP0=2000; VS0=1200; RO0=2000

# CHANGED: a reflector, +10% in Vp and Vs below it, rho unchanged.  Rho
# unchanged is the specification and it matters -- the reflection coefficient
# is then purely a velocity contrast.
VFAC=1.10
ZREFL=2298                    # 383*DX; the requested 2300 is not on the grid

# Geometry.  The requested S = (1500,800), R = (3500,800) is snapped onto the
# dx = 6 grid, and XRCV is taken to 3504 rather than 3498 so that the SPECULAR
# MIDPOINT 0.5(XSRC+XRCV) = 2502 is itself a grid node -- every slant-stack
# pivot then sits on a sample instead of between two.  The cost is a 4 m
# change in offset, 0.2%.
XSRC=1500; ZSRC=798
XRCV=3504; ZRCV=798
XSPEC=2502; ZSPEC=2298

# Run B anomalies.  gA is on the PP DOWN-leg and gB on the PP UP-leg: the
# straight line S -> (XSPEC,ZSPEC) crosses z = 1248 at x = 1800.6, and the
# up-leg crosses it at x = 3203.4, so both sites are on the ray to within
# 1 m.  The requested (3200,1250) is moved to (3204,1248) for that reason --
# 4 m, against a 300 m sigma.
SIGMA=300; APCT=-5
XA=1800; ZA=1248              # Vp only
XB=3204; ZB=1248              # Vs only

# CHANGED from TMOD=3.1.  SS arrives at 3.0065 s and makewave puts the wavelet
# PEAK at t0 = 0.15, so the SS peak is at 3.157 s and 3.1 would truncate it.
# 4.0 leaves 0.84 s of coda.  The constraint this trades against is
# df <= 1/TMOD, which at 4.0 s is 0.25 Hz against the df = 0.2 Hz used below:
# still satisfied, with less margin than before (df*T 0.80 against 0.62).
TMOD=4.0; DTRCV=0.001
NTAPER=200
GRAD_TAPER=30
WAV_FMIN=2; WAV_FMAX=25; WAV_T0=0.15
LAG_NT=101; LAG_DT=0.002      # tau in [-0.10, +0.10] s, as specified
D_FMIN=2; D_FMAX=36; D_NFREQ=171

# Slant stack and cone filter
HAPER=300                     # Hann aperture HALF width, as specified
S_PASS=0.20; S_STOP=0.30      # the primary cone; midpoints 0.15/0.25/0.40 are
                              # swept in the anal stage
# No exclusion radius around S and R.  test_lag_diagonal.sh masks nothing in
# its analysis, and this test is meant to be read against it, so the only
# source/receiver conditioning is grad_taper=30 (180 m) -- which is applied
# INSIDE the gradient, identically in both tests, and per lag by
# lag_gradient.c.  Set this to a positive radius to blank a disk around S and
# R in the metrics instead.
EXCL=0

FPEAK=$(echo ${WAV_FMAX} | awk '{printf "%.4f", 0.4425*$1}')
DFREQ=$(echo "${D_FMIN} ${D_FMAX} ${D_NFREQ}" | \
        awk '{printf "%.6f", ($2-$1)/($3-1)}')

# ---------------------------------------------------------------------------
if [ "${STAGE}" = "model" ] || [ "${STAGE}" = "all" ]; then

echo "=========================================================="
echo " test_lag_reflector.sh -- stage: model"
echo "=========================================================="
rm -rf ${WORK}
mkdir -p ${WORK}
cd ${WORK}

echo "--- grid and geometry ---"
printf '  domain %d x %d m, dx = dz = %d m\n' ${SIZEX} ${SIZEZ} ${DX}
printf '  S  (%6d, %6d)   Fx force (src_type=6)\n' ${XSRC} ${ZSRC}
printf '  R  (%6d, %6d)   Vx and Vz\n' ${XRCV} ${ZRCV}
printf '  offset %d m, reflector z = %d m (%d m below the instruments)\n' \
    $((XRCV - XSRC)) ${ZREFL} $((ZREFL - ZSRC))
printf '  specular midpoint (%d, %d)\n' ${XSPEC} ${ZSPEC}

echo ""
echo "--- models ---"
# The homogeneous half space first; the other two are built from it.
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=homo.su verbose=0
# Four model sets, because the fourth makes the run B residual split EXACT
# rather than windowed (see lag_refl_check.py):
#   homo   homogeneous half space                 run A background
#   refl   + the reflector                       run A observed = run B bg
#   anom   + the reflector + the two Gaussians   run B observed
#   hano   + the two Gaussians, NO reflector     QC only, never a gradient
for P in cp cs ro; do
    cp homo_${P}.su refl_${P}.su
    cp homo_${P}.su anom_${P}.su
    cp homo_${P}.su hano_${P}.su
done

python3 - <<PYEOF
import sys
import numpy as np
sys.path.insert(0, '${ROOT}/fdelfwi/tests/kh_misfit')
import su_io

DX = ${DX}
ZREFL = ${ZREFL}
VFAC = ${VFAC}
AMP = ${APCT} / 100.0
SIG = ${SIGMA}

# (1) the reflector, in refl_* and anom_*: velocities VFAC times higher at and
#     below z = ZREFL, rho untouched.  The interface is placed at the FIRST
#     sample with z >= ZREFL, so it is one flat row of nodes and there is no
#     interpolated half cell.
for base in ('refl', 'anom'):
    for par, fac in (('cp', VFAC), ('cs', VFAC), ('ro', 1.0)):
        fn = '%s_%s.su' % (base, par)
        d, h = su_io.read(fn)
        k = int(np.ceil(ZREFL / DX))
        d[:, k:] *= fac
        su_io.write(fn, d, h)
j = int(np.ceil(ZREFL / DX))
print('  reflector at sample %d, z = %.0f m; below it Vp %.0f -> %.0f, '
      'Vs %.0f -> %.0f, rho %.0f (unchanged)'
      % (j, j * DX, ${VP0}, ${VP0} * VFAC, ${VS0}, ${VS0} * VFAC, ${RO0}))
r = (VFAC - 1.0) / (VFAC + 1.0)
print('  normal-incidence P reflection coefficient R = %.4f  (rho equal, so '
      'this is the velocity contrast alone)' % r)

# (2) the two Gaussians, in anom_* only.  One per PARAMETER at its own site,
#     so each event in the data is retimed by ONE anomaly:
#       the P legs feel gA (Vp) and not gB, because perturbing Vs alone moves
#       mu and moves lambda back so that Vp = sqrt((lam+2mu)/rho) is unchanged
#       the S legs feel gB (Vs) and not gA
print('  %-12s %8s %8s %10s %10s %11s' % ('par', 'x', 'z', 'v0', 'v_min',
                                          'peak pert'))
for fn, lab, xa, za in (('anom_cp.su', 'anom Vp', ${XA}, ${ZA}),
                        ('anom_cs.su', 'anom Vs', ${XB}, ${ZB}),
                        ('hano_cp.su', 'hano Vp', ${XA}, ${ZA}),
                        ('hano_cs.su', 'hano Vs', ${XB}, ${ZB})):
    d, h = su_io.read(fn)
    nx, nz = d.shape
    x = (np.arange(nx) * DX)[:, None]
    z = (np.arange(nz) * DX)[None, :]
    g = np.exp(-((x - xa) ** 2 + (z - za) ** 2) / (2.0 * SIG ** 2))
    v0 = float(d[0, 0])
    out = d.astype(np.float64) * (1.0 + AMP * g)
    su_io.write(fn, out.astype(np.float32), h)
    print('  %-12s %8.0f %8.0f %10.1f %10.1f %10.2f%%'
          % (lab, xa, za, v0, out.min(), 100.0 * (out.min() / v0 - 1.0)))

# The dispersion guard, over ALL FOUR models: fdelmodc aborts if the measured
# wavelet edge exceeds cmin/(5 dx), and cmin is set by the SLOW Vs anomaly.
cmin = min(float(su_io.read('%s_cs.su' % b)[0].min())
           for b in ('homo', 'refl', 'anom', 'hano'))
fdisp = cmin / (5.0 * DX)
fmeas = 1.22 * ${WAV_FMAX} + 1.46
print('  dispersion guard: cmin = %.1f m/s -> fmax < %.2f Hz; measured '
      'wavelet edge %.2f Hz (%.0f%% headroom)'
      % (cmin, fdisp, fmeas, 100.0 * (fdisp / fmeas - 1.0)))
if fmeas >= fdisp:
    raise SystemExit('the wavelet edge is above the dispersion guard')

# Backscatter from the Gaussians: the transform is exp(-k^2 sig^2/2) and
# backscatter samples 2k, so the amplitude goes as exp(-2 k^2 sig^2).  This is
# the quantitative version of "the anomalies do not reflect", which is what
# makes run B a TOMOGRAPHY run and not a second migration run.
for lab, f in (('peak', ${FPEAK}), ('edge', fmeas)):
    ks = 2.0 * np.pi * f * SIG / cmin
    print('  backscatter off a Gaussian at the %s (%5.2f Hz): k*sigma = %5.1f '
          '-> amplitude ~ e^%-6.0f' % (lab, f, ks, -2.0 * ks ** 2))
PYEOF

echo ""
echo "--- wavelet ---"
makewave fmin=${WAV_FMIN} fmax=${WAV_FMAX} \
    dt=0.001 nt=4096 file_out=wave.su t0=${WAV_T0} verbose=0
printf '  fmin=%s fmax=%s t0=%s -> f_peak = %s Hz, period %s ms\n' \
    ${WAV_FMIN} ${WAV_FMAX} ${WAV_T0} ${FPEAK} \
    $(echo ${FPEAK} | awk '{printf "%.1f", 1000/$1}')

# ---------------------------------------------------------------------------
# Four forward runs, and only three of them feed a gradient.  Run B's
# BACKGROUND is the reflector model, which is also run A's TRUE model, so that
# gather is computed once and reused -- being the same bytes is not a
# convenience, it is what makes "the reflector is identical in both of run B's
# models" exact rather than approximate.
#
#   d_refl   reflector             = run A observed = run B background
#   d_anom   reflector + Gaussians = run B observed
#   d_homo   homogeneous           = run A background
#   d_hano   Gaussians, no reflector -- QC ONLY.  It buys the exact split of
#            the run B residual into a direct part and a reflected part, which
#            is what turns the mute from a guessed window into a measured one.
# ---------------------------------------------------------------------------
echo ""
echo "--- forward modelling (4 runs) ---"
for M in refl anom homo hano; do
    printf '  %-5s ... ' ${M}
    ${FDELFWI}/fdelfwi_fwd \
        file_cp=${M}_cp.su file_cs=${M}_cs.su file_den=${M}_ro.su \
        file_src=wave.su file_rcv=d_${M} \
        ischeme=3 iorder=4 src_type=6 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 > log_fwd_${M}.txt 2>&1
    suop2 d_${M}_000_rtxx.su d_${M}_000_rtzz.su op=sum | \
        sugain scale=0.5 > d_${M}_000_rp.su
    echo "done"
done

echo ""
echo "--- arrival QC and the direct-S / PP overlap gate ---"
python3 ${HERE}/lag_refl_check.py \
    --dx=${DX} --sizex=${SIZEX} --sizez=${SIZEZ} \
    --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} \
    --zrefl=${ZREFL} --vp=${VP0} --vs=${VS0} --vfac=${VFAC} \
    --t0=${WAV_T0} --fpeak=${FPEAK} --tmod=${TMOD} \
    --ntaper=${NTAPER} --dfreq=${DFREQ}

echo ""
echo "stage model complete.  Read the gate above before running stage grad."
fi

# ---------------------------------------------------------------------------
if [ "${STAGE}" = "grad" ] || [ "${STAGE}" = "all" ]; then

echo ""
echo "=========================================================="
echo " test_lag_reflector.sh -- stage: grad"
echo "=========================================================="
cd ${WORK}

# name : background : observed : mute
RUNS=(
    "runA:homo:refl:0"
    "runB:refl:anom:1"
)

for SPEC in "${RUNS[@]}"; do
    IFS=: read -r NAME BG OBS MUTE <<< "${SPEC}"
    echo ""
    echo "----- ${NAME}:  background ${BG}, observed ${OBS}, mute=${MUTE} -----"
    mkdir -p ${NAME}
    cd ${NAME}
    for F in homo_cp homo_cs homo_ro refl_cp refl_cs refl_ro \
             anom_cp anom_cs anom_ro wave; do
        ln -sf ../${F}.su .
    done
    for C in rvx rvz rtxx rtzz rp; do
        ln -sf ../d_${OBS}_000_${C}.su obs_000_${C}.su
        ln -sf ../d_${BG}_000_${C}.su  bg_000_${C}.su
    done

    FILE_OBS=obs
    if [ "${MUTE}" = "1" ]; then
        # test_gradient_visual computes the residual INTERNALLY as
        # res = syn - obs (verified by correlation on the case_fx_geo files:
        # corr +1.000000, ratio 1.000000 against syn - obs).  So a mute can be
        # applied without touching the gradient code: form r = syn - obs, mute
        # it, and hand back obs' = syn - r_muted.  The run then sees
        # res = syn - obs' = r_muted, exactly.
        #
        # syn is taken from a cheap pass with lag_nfreq=0, which writes the
        # synthetic and skips the whole lag accumulation.  Using THAT syn
        # rather than the separately-modelled bg gather is what makes the
        # identity exact; lag_refl_mute.py reports how far the two differ.
        echo "  mute pass 1: syn only (lag_nfreq=0)"
        ${FDELFWI}/test_gradient_visual \
            file_cp=${BG}_cp.su file_cs=${BG}_cs.su file_den=${BG}_ro.su \
            file_src=wave.su file_rcv=syn_probe \
            ischeme=3 iorder=4 src_type=6 \
            rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
            dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
            xrcva=${XRCV} zrcva=${ZRCV} \
            xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
            ntaper=${NTAPER} grad_taper=${GRAD_TAPER} \
            res_taper=0 aper_taper=0 \
            left=4 right=4 top=4 bottom=4 \
            chk_skipdt=50 chk_base=chk_probe \
            param=2 scaling=0 precond=0 smooth_grad=0 \
            vp_min=1000 vp_max=4500 vs_min=500 vs_max=2500 \
            rho_min=1500 rho_max=2500 \
            data_weight=0 misfit=0 comp=_rvx,_rvz \
            file_obs=obs out_base=probe > log_probe.txt 2>&1
        python3 ${HERE}/lag_refl_mute.py \
            --syn=syn_probe --bg=bg --obs=obs --out=obsm \
            --homo=../d_homo --hano=../d_hano \
            --comp=rvx,rvz,rtxx,rtzz \
            --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} \
            --zrefl=${ZREFL} --vp=${VP0} --vs=${VS0} \
            --t0=${WAV_T0} --fpeak=${FPEAK}
        suop2 obsm_000_rtxx.su obsm_000_rtzz.su op=sum | \
            sugain scale=0.5 > obsm_000_rp.su
        FILE_OBS=obsm
        rm -f probe_*.su probe_*.txt
        rm -rf chk_probe*
    fi

    echo "  lag gradient (nfreq=${D_NFREQ}, ${D_FMIN}-${D_FMAX} Hz, nt=${LAG_NT}, dtau=${LAG_DT} s)"
    ${FDELFWI}/test_gradient_visual \
        file_cp=${BG}_cp.su file_cs=${BG}_cs.su file_den=${BG}_ro.su \
        file_src=wave.su file_rcv=syn_dense \
        ischeme=3 iorder=4 src_type=6 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=0 aper_taper=0 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=50 chk_base=chk_dense \
        param=2 scaling=0 precond=0 smooth_grad=0 \
        vp_min=1000 vp_max=4500 vs_min=500 vs_max=2500 \
        rho_min=1500 rho_max=2500 \
        data_weight=0 misfit=0 comp=_rvx,_rvz \
        lag_nfreq=${D_NFREQ} lag_fmin=${D_FMIN} lag_fmax=${D_FMAX} \
        lag_nt=${LAG_NT} lag_dt=${LAG_DT} \
        file_obs=${FILE_OBS} out_base=dense > log_dense.txt 2>&1
    grep -E "lagGradientInit|lag cube" log_dense.txt | head -4 || true
    grep -E "Total misfit" log_dense.txt | tail -1

    # NOT a per-frequency gradient.  lag_gradient.c writes only the two lag
    # cubes and the axis file -- its DFT accumulators are freed by
    # lagFreeAccum and no option dumps them -- so the requested
    # "per-frequency gradients, if a simple option already exists" cannot be
    # produced without changing gradient code, which is out of scope here.

    # Files nothing downstream reads: see the same block in
    # test_lag_diagonal.sh.  nshot=1, so the per-shot cube duplicates the
    # stack byte for byte.
    rm -f dense_vp_scaled.su dense_vs_scaled.su dense_rho_scaled.su \
          dense_vp_preco.su  dense_vs_preco.su  dense_rho_preco.su \
          precond_*.su hess_raw_*.su hess_prebeta_*.su \
          dense_shot000_lagvp.su dense_shot000_lagvs.su \
          dense_shot000_lagaxis.txt
    rm -rf chk_dense*
    cd ${WORK}
done
echo ""
echo "stage grad complete."
fi

# ---------------------------------------------------------------------------
if [ "${STAGE}" = "anal" ] || [ "${STAGE}" = "all" ]; then

echo ""
echo "=========================================================="
echo " test_lag_reflector.sh -- stage: anal"
echo "=========================================================="
cd ${WORK}
python3 ${HERE}/lag_refl_report.py \
    --work=${WORK} --dx=${DX} \
    --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} \
    --zrefl=${ZREFL} --xspec=${XSPEC} --zspec=${ZSPEC} \
    --vp=${VP0} --vs=${VS0} --t0=${WAV_T0} --fpeak=${FPEAK} \
    --xa=${XA} --za=${ZA} --xb=${XB} --zb=${ZB} \
    --haper=${HAPER} --s-pass=${S_PASS} --s-stop=${S_STOP} --excl=${EXCL}
echo ""
echo "stage anal complete.  Everything is in ${WORK}"
fi
