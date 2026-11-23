#!/bin/bash
#
# test_petro_examples.sh - the four reference inversion demos, rerun with the
#                          probabilistic petrophysical constraint J_P
#                          (Aragao & Sava, 2020, GEOPHYSICS 85(2), R101-R111)
#
# Same model, geometry, wavelet and data component as
#     test_lbfgs_inversion_lame.sh       -> CASE=lbfgs_lame
#     test_lbfgs_inversion_velocity.sh   -> CASE=lbfgs_velocity
#     test_trn_inversion_lame.sh         -> CASE=trn_lame
#     test_trn_inversion_velocity.sh     -> CASE=trn_velocity
# i.e. 2000x800 m, dx=5, homogeneous Vp=2000/Vs=1150/rho=2000 with three
# Gaussian anomalies (Vp at x=500, Vs at x=1000, rho at x=1500), 5 shots,
# ischeme=3 iorder=4, scaling=1, starting from the homogeneous background.
#
# Each case runs TWICE, identical apart from the constraint:
#     <case>/baseline     no constraint
#     <case>/petro_s<N>   petro_file=wells_*.txt petro_space=N petro_balance=B
#
# Deliberate differences from the reference scripts (they would not run, or
# would not test what their names say, as they stand):
#   - vp_min/vp_max/vs_*/rho_* are passed.  scaling=1 now verr()s without
#     them, so the reference scripts abort at startup.
#   - lbfgs_velocity runs param=2.  The reference script is NAMED velocity but
#     passes param=1.
#   - Observed data come from fdelfwi_fwd, not fdelmodc: the two boundary
#     implementations differ and fdelmodc data leave a ~1% direct-wave
#     residual at the true model.
#
# WELL LOGS.  Three vertical wells at x = 500, 1000, 1500 m, i.e. straight
# through the three anomalies, sampled from the TRUE model every dz, with
# 2.5% sonic-log scatter (NREAL independent realizations per depth sample).
# Each anomaly is a single-parameter perturbation, so on the crossplot it is
# a short ARM leaving the background blob: Vp-only -> +lambda only;
# Vs-only -> +mu and -2*dmu in lambda; rho-only -> both scale.  The PDF
# therefore permits each anomaly's own direction and penalises the OTHER
# directions -- which is exactly the crosstalk (e.g. the Vp anomaly leaking
# into Vs).  The background blob holds ~90% of the samples, so expect J_P to
# also shave some amplitude off the anomalies: they are rare in the logs.
#
# WHAT TO LOOK AT (printed at the end, measured from the final models):
#   per anomaly zone, the recovered perturbation of EACH parameter, as a
#   fraction of the true perturbation of the anomaly's own parameter.  The
#   diagonal is recovery (want ~1), the off-diagonal is crosstalk (want 0).
#   The misfit is NOT comparable across runs: J_P is part of the objective.
#
# Usage:
#   ./test_petro_examples.sh                  # all four cases, sequential
#   ./test_petro_examples.sh trn_lame         # one case
#   PETRO_BALANCE=0.2 SPACES="1 2" ./test_petro_examples.sh lbfgs_velocity
#
# Environment overrides:
#   PETRO_BALANCE (0.1)   petro_balance=
#   SPACES        ("1")   crossplot planes to run: 1=(lambda,mu) GPa, 2=(Vp,Vs) km/s
#   NITER         (5)     outer iterations, as the reference scripts
#   NREAL         (10)    scatter realizations per log sample
#   OMP_NUM_THREADS (4)
#
# Cases are independent (separate workdirs), so they can be run in parallel:
#   for c in lbfgs_lame lbfgs_velocity trn_lame trn_velocity; do
#       ./test_petro_examples.sh $c > petro_examples_$c.log 2>&1 &
#   done; wait
#
# J_P's own correctness (Taylor, attraction, chain rules, Hessian) is
# test_petro, which needs no wave propagation:
#   ../../test_petro petro_file=petro_examples/<case>/wells_lammu.txt petro_space=1
#

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

PETRO_BALANCE=${PETRO_BALANCE:-0.1}
SPACES=${SPACES:-1}
NITER=${NITER:-5}
NREAL=${NREAL:-10}

HERE=$(cd "$(dirname "$0")" && pwd)
TOP=${HERE}/petro_examples

CASES=${1:-"lbfgs_lame lbfgs_velocity trn_lame trn_velocity"}

# ---------------------------------------------------------------------------
# Geometry (identical to the reference scripts)
# ---------------------------------------------------------------------------
DX=5
SIZEX=2000
SIZEZ=800
VP0=2000
VS0=1150
RO0=2000
NSHOTS=5
XSRC1=200
DXSHOT=400
ZSRC=10
XRCV1=100
XRCV2=1900
ZRCV=100
DXRCV=10
DTRCV=0.004
TMOD=1.5
NTAPER=100
# Bounds bracket all three anomalies AND keep lambda_min = rho_min*(vp_min^2 -
# 2 vs_max^2) > 0: scaling=1 derives the Lame ranges from these, and a
# negative lambda_min inflates the lambda range (lbfgs_lame then failed its
# first linesearch with a 6000x cost overshoot).
BOUNDS="vp_min=1900 vp_max=2400 vs_min=1000 vs_max=1300 rho_min=1800 rho_max=2400"

# ---------------------------------------------------------------------------
run_case ()
{
    local case=$1
    local param fp fmax gtaper comp nls alg algname

    # Per-case settings, copied from the reference scripts
    case ${case} in
      lbfgs_lame)     param=1 fp=5  fmax=10 gtaper=60 comp=_rp  nls=5
                      alg="algorithm=1 lbfgs_mem=5" ;;
      lbfgs_velocity) param=2 fp=10 fmax=15 gtaper=20 comp=_rvz nls=10
                      alg="algorithm=1 lbfgs_mem=5" ;;
      trn_lame)       param=1 fp=5  fmax=10 gtaper=60 comp=_rp  nls=10
                      alg="algorithm=4 niter_cg=5" ;;
      trn_velocity)   param=2 fp=5  fmax=10 gtaper=20 comp=_rvz nls=10
                      alg="algorithm=4 niter_cg=5" ;;
      *) echo "unknown case '${case}'"; exit 1 ;;
    esac

    local W=${TOP}/${case}
    mkdir -p ${W}
    cd ${W}

    echo ""
    echo "#########################################################"
    echo "# CASE ${case}: param=${param} ${alg} comp=${comp}"
    echo "#   wavelet fp=${fp} fmax=${fmax}, grad_taper=${gtaper}, nls_max=${nls}"
    echo "#   petro_balance=${PETRO_BALANCE}, spaces: ${SPACES}"
    echo "#   workdir ${W}"
    echo "#########################################################"

    # --- models (always rebuilt: cheap, and keeps obs/model/wells in step) ---
    makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
        cp0=${VP0} cs0=${VS0} ro0=${RO0} \
        orig=0,0 file_base=model_bg.su verbose=0
    makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
        cp0=${VP0} cs0=${VS0} ro0=${RO0} \
        intt=diffr dtype=2 x=500  z=400 cp=2200   cs=${VS0} ro=${RO0} var=20 \
        intt=diffr dtype=2 x=1000 z=400 cp=${VP0} cs=1250   ro=${RO0} var=20 \
        intt=diffr dtype=2 x=1500 z=400 cp=${VP0} cs=${VS0} ro=2200   var=20 \
        orig=0,0 file_base=model_true.su verbose=0

    makewave fp=${fp} dt=0.001 nt=1024 fmax=${fmax} file_out=wave.su t0=0.10 verbose=0

    # --- observed data (regenerated every time: no stale-gather reuse) ---
    rm -f obs_*.su
    ${FDELFWI}/fdelfwi_fwd \
        file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs \
        ischeme=3 iorder=4 src_type=1 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
        xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
        xsrc=${XSRC1} zsrc=${ZSRC} nshot=${NSHOTS} dxshot=${DXSHOT} \
        ntaper=${NTAPER} left=4 right=4 top=4 bottom=4
    # hydrophone P = 0.5*(Txx+Tzz); fwi_inversion builds the synthetic one
    # the same way
    for ((i=0; i<NSHOTS; i++)); do
        S=$(printf "%03d" $i)
        if [ -f obs_${S}_rtzz.su ] && [ -f obs_${S}_rtxx.su ]; then
            suop2 obs_${S}_rtzz.su obs_${S}_rtxx.su op=sum | sugain scale=0.5 > obs_${S}_rp.su
        fi
    done
    ls obs_000_*.su > /dev/null || { echo "no observed data produced"; exit 1; }
    [ -f obs_000${comp}.su ] || { echo "obs_000${comp}.su missing"; ls obs_000_*; exit 1; }
    echo "  observed data: $(ls obs_*${comp}.su | wc -l) shots of ${comp}"

    # --- well logs from the true model, written on both crossplot planes ---
    python3 -I - "${NREAL}" <<'PYEOF'
import sys, struct, random, math
import numpy as np

def rd(f):
    b = open(f, 'rb').read()
    ns = struct.unpack('<H', b[114:116])[0]
    n = len(b) // (240 + 4*ns)
    a = np.frombuffer(b, dtype=np.uint8).reshape(n, 240 + 4*ns)[:, 240:]
    return a.copy().view('<f4')           # [nx, nz]

cp, cs, ro = (rd('model_true_%s.su' % k) for k in ('cp', 'cs', 'ro'))
nreal = int(sys.argv[1])
dx = 5.0
SCAT = 0.025                               # sonic-log scatter, as test_petro_inversion.sh
random.seed(20200101)
lm, vv = [], []
for xw in (500.0, 1000.0, 1500.0):
    ix = int(round(xw / dx))
    for iz in range(cp.shape[1]):
        for _ in range(nreal):
            vp = cp[ix, iz] * (1.0 + SCAT*random.gauss(0, 1))
            vs = cs[ix, iz] * (1.0 + SCAT*random.gauss(0, 1))
            r  = ro[ix, iz] * (1.0 + 0.4*SCAT*random.gauss(0, 1))
            mu, lam = r*vs*vs, r*(vp*vp - 2.0*vs*vs)
            lm.append("%.6f %.6f" % (lam*1e-9, mu*1e-9))
            vv.append("%.6f %.6f" % (vp*1e-3, vs*1e-3))
hdr = "wells at x=500,1000,1500 through the true model, %d realizations/sample, %.1f%% scatter" % (nreal, 100*SCAT)
open('wells_lammu.txt', 'w').write("# lambda mu [GPa], " + hdr + "\n" + "\n".join(lm) + "\n")
open('wells_vpvs.txt',  'w').write("# Vp Vs [km/s], "    + hdr + "\n" + "\n".join(vv) + "\n")
print("  %d well samples -> wells_lammu.txt, wells_vpvs.txt" % len(lm))
PYEOF

    # --- inversion runner ---
    run_inv ()
    {
        local tag=$1; shift
        echo ""
        echo "========================================================="
        echo " ${case}: RUN ${tag}"
        echo "========================================================="
        rm -rf ${tag}; mkdir -p ${tag}
        ( cd ${tag}
          ${FDELFWI}/fwi_inversion \
            file_cp=../model_bg_cp.su file_cs=../model_bg_cs.su file_den=../model_bg_ro.su \
            file_src=../wave.su file_rcv=syn \
            ischeme=3 iorder=4 src_type=1 \
            rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
            dtrcv=${DTRCV} tmod=${TMOD} \
            xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
            xsrc=${XSRC1} zsrc=${ZSRC} nshot=${NSHOTS} dxshot=${DXSHOT} \
            ntaper=${NTAPER} grad_taper=${gtaper} res_taper=100 \
            left=4 right=4 top=4 bottom=4 \
            chk_skipdt=100 chk_base=chk_inv \
            param=${param} scaling=1 ${BOUNDS} \
            comp=${comp} \
            file_obs=../obs file_grad=gradient \
            ${alg} \
            niter=${NITER} nls_max=${nls} conv=1e-6 \
            write_iter=1 verbose=2 "$@" > run.log 2>&1
          status=$?
          rm -rf chk_inv_*.bin fwi_rank*
          exit ${status} ) \
        && echo "  ${tag}: finished" \
        || echo "  ${tag}: fwi_inversion FAILED -- see ${W}/${tag}/run.log"
        grep -h "calibrated a=" ${tag}/run.log 2>/dev/null | sed 's/^ *[^:]*: /  /' || true
    }

    run_inv baseline
    local s
    for s in ${SPACES}; do
        local wf=../wells_lammu.txt
        [ "${s}" = 2 ] && wf=../wells_vpvs.txt
        run_inv petro_s${s} petro_file=${wf} petro_space=${s} petro_balance=${PETRO_BALANCE}
    done

    # --- report: recovery (diagonal) and crosstalk (off-diagonal) ---
    python3 -I - "${param}" baseline $(for s in ${SPACES}; do echo petro_s${s}; done) <<'PYEOF'
import sys, os, glob, struct
import numpy as np

def rd(f):
    b = open(f, 'rb').read()
    ns = struct.unpack('<H', b[114:116])[0]
    n = len(b) // (240 + 4*ns)
    a = np.frombuffer(b, dtype=np.uint8).reshape(n, 240 + 4*ns)[:, 240:]
    return a.copy().view('<f4').astype(np.float64)

param = int(sys.argv[1]); runs = sys.argv[2:]
tru = [rd('model_true_%s.su' % k) for k in ('cp', 'cs', 'ro')]
bg  = [rd('model_bg_%s.su'   % k) for k in ('cp', 'cs', 'ro')]
nx, nz = tru[0].shape
X, Z = np.meshgrid(np.arange(nx)*5.0, np.arange(nz)*5.0, indexing='ij')
zones = [('Vp  anomaly x=500 ', 500.0, 0), ('Vs  anomaly x=1000', 1000.0, 1), ('rho anomaly x=1500', 1500.0, 2)]
names = ['dVp', 'dVs', 'drho']

def final(run):
    if param == 1:
        fs = sorted(glob.glob(os.path.join(run, 'model_iter*_lam.su')))
        if not fs: return None, 0
        it = fs[-1].split('model_iter')[1][:3]
        lam = rd(os.path.join(run, 'model_iter%s_lam.su' % it))
        mu  = rd(os.path.join(run, 'model_iter%s_muu.su' % it))
        r   = rd(os.path.join(run, 'model_iter%s_rho.su' % it))
        return [np.sqrt(np.maximum(lam + 2*mu, 0)/r), np.sqrt(np.maximum(mu, 0)/r), r], int(it)
    fs = sorted(glob.glob(os.path.join(run, 'model_iter*_vp.su')))
    if not fs: return None, 0
    it = fs[-1].split('model_iter')[1][:3]
    return [rd(os.path.join(run, 'model_iter%s_%s.su' % (it, k))) for k in ('vp', 'vs', 'rho')], int(it)

print("")
print("  Recovered perturbation inside each anomaly (r < 30 m), as a fraction of")
print("  the TRUE perturbation of that anomaly's own parameter.  Diagonal = recovery")
print("  (want 1), off-diagonal = crosstalk (want 0).  Plus ||m - m_true|| / ||m_bg - m_true||.")
for run in runs:
    m, it = final(run)
    if m is None:
        print("\n  %-12s no model output" % run); continue
    print("\n  %-12s (iteration %d)" % (run, it))
    print("    %-20s %8s %8s %8s" % ('zone', *names))
    for zn, xc, own in zones:
        msk = (X - xc)**2 + (Z - 400.0)**2 < 30.0**2
        ref = np.mean(tru[own][msk] - bg[own][msk])
        row = [np.mean(m[k][msk] - bg[k][msk]) / ref for k in range(3)]
        print("    %-20s %8.3f %8.3f %8.3f" % (zn, *row))
    err = [np.linalg.norm(m[k] - tru[k]) / max(np.linalg.norm(bg[k] - tru[k]), 1e-30) for k in range(3)]
    print("    %-20s %8.3f %8.3f %8.3f" % ('rel. model error', *err))
PYEOF
    cd ${HERE}
}

mkdir -p ${TOP}
for c in ${CASES}; do
    run_case ${c}
done

echo ""
echo "========================================================="
echo " Done.  Per-run logs: ${TOP}/<case>/{baseline,petro_s*}/run.log"
echo "========================================================="
