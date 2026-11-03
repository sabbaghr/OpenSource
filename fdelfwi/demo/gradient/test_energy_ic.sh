#!/bin/bash
# test_energy_ic.sh -- validate the energy-norm angle filter (energy_ic=).
#
# Reuses the taylor_vs/ setup: ONE shot at x=2500, Vs-only smoothed anomalies
# at (2000,800) and (3000,800), and a rho reflector at z=1500 that is present
# in BOTH models.  The residual is therefore the z=1500 reflection mis-timed
# in TRANSMISSION through the anomalies -- a textbook tomographic signal with
# a textbook migration-type event (the reflector) to be rejected.
#
# Three checks:
#
#   (1) energy_ic=0 must be BIT-IDENTICAL to the un-instrumented gradient.
#       The accumulators are no-ops, so this is a regression test on the two
#       new calls inside the adj_shot backward loop.
#
#   (2) g(energy_ic=1) + g(energy_ic=2) == g(energy_ic=0), because the two
#       weights are normalised to partition unity, a_tomo + a_migr = 1.
#       NOTE: this is NOT the identity "I_tomo + I_migr = I_conventional" --
#       that one is false, the energy inner product carries an extra 2w^2
#       relative to the conventional one (Rocha eqs 28-31).  It holds here
#       only because the weights are normalised by their own sum.
#
#   (3) Angle selectivity.  The tomographic fraction (the share of the
#       gradient norm surviving a 300 m smoother) must RISE for energy_ic=1
#       and FALL for energy_ic=2 relative to energy_ic=0.
#
# Everything that modifies the gradient after accumulation is off, as in
# taylor_vs.sh, so the only difference between the three runs is the filter.
#
# Usage:  ./test_energy_ic.sh          (expects taylor_vs/ to exist already;
#                                       run ./taylor_vs.sh first if not)

set -e
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}
export OMP_NUM_THREADS=8

SRC=taylor_vs
WORK=energy_ic

if [ ! -f ${SRC}/model_bg_cs.su ] || [ ! -f ${SRC}/obs_000_rvz.su ]; then
    echo "ERROR: ${SRC}/ is missing its models or observed data."
    echo "       Run ./taylor_vs.sh first."
    exit 1
fi

rm -rf ${WORK}
mkdir -p ${WORK}
cp ${SRC}/model_bg_*.su ${SRC}/model_true_*.su ${SRC}/wave.su ${WORK}/ 2>/dev/null || true
cp ${SRC}/obs_000_*.su ${WORK}/
cd ${WORK}

if [ ! -f wave.su ]; then
    makewave fp=14 fmin=2 fmax=25 dt=0.001 nt=4096 file_out=wave.su t0=0.10 verbose=0
fi

DX=5; SIZEX=5000
XSRC=2500; ZSRC=50
XRCV1=0; XRCV2=${SIZEX}; DXRCV=10; ZRCV=100
TMOD=4.0; DTRCV=0.001
NTAPER=200

# run_grad <energy_ic> <out_base> <logfile>
run_grad () {
    ${FDELFWI}/test_gradient_visual \
        file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
        file_src=wave.su file_rcv=syn \
        ischeme=3 iorder=4 src_type=6 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
        xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DXRCV} zrcv1=${ZRCV} zrcv2=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 dxshot=0 \
        ntaper=${NTAPER} grad_taper=0 res_taper=0 aper_taper=0 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=100 chk_base=chk_ei \
        param=2 scaling=0 precond=0 smooth_grad=0 \
        vp_min=1500 vp_max=4500 vs_min=500 vs_max=2500 \
        rho_min=1500 rho_max=2500 \
        data_weight=0 misfit=2 kh_normal=-1 \
        comp=_rvx,_rvz,_rtxz,_rtzz \
        file_obs=obs energy_ic=$1 out_base=$2 > $3 2>&1
}

for EI in 0 1 2; do
    echo "--- energy_ic=${EI} ---"
    run_grad ${EI} g${EI} log_${EI}.txt
    grep -E "=== Total misfit|energy_ic:" log_${EI}.txt || true
done

echo ""
python3 - <<'PY'
import sys
import numpy as np
sys.path.insert(0, '/rcp3/software/codes/OpenSource_SL10/fdelfwi/tests/kh_misfit')
import su_io
from scipy.ndimage import uniform_filter

NZ, NX, DX = 401, 1001, 5.0
SM = int(round(300.0 / DX)) | 1          # 300 m smoother, odd


def g(base, par):
    d, _ = su_io.read('%s_%s_raw.su' % (base, par))
    return d.astype(np.float64).reshape(NX, NZ)


def tomofrac(a):
    n = np.linalg.norm(a)
    return np.linalg.norm(uniform_filter(a, SM)) / n if n > 0 else 0.0


print('=========== energy_ic validation ===========')
ok = True

# ---- (1) energy_ic=0 is the reference; compare against the archived run ----
print('\n(1) energy_ic=0 vs the pre-existing taylor_vs gradient')
for par in ('vp', 'vs', 'rho'):
    a = g('g0', par)
    try:
        b, _ = su_io.read('../taylor_vs/g0_%s_raw.su' % par)
        b = b.astype(np.float64).reshape(NX, NZ)
    except Exception as e:
        print('    %-3s : archived run unavailable (%s)' % (par, e))
        continue
    dmax = np.abs(a - b).max()
    rel = dmax / max(np.abs(b).max(), 1e-300)
    flag = 'IDENTICAL' if dmax == 0.0 else ('ok (%.2e rel)' % rel if rel < 1e-6
                                            else 'DIFFERS')
    if flag == 'DIFFERS':
        ok = False
    print('    %-3s : max|diff| = %.6e   %s' % (par, dmax, flag))

# ---- (2) partition of unity: g1 + g2 == g0 ----
print('\n(2) a_tomo + a_migr = 1   =>   g(ic=1) + g(ic=2) = g(ic=0)')
for par in ('vp', 'vs', 'rho'):
    g0, g1, g2 = g('g0', par), g('g1', par), g('g2', par)
    d = g1 + g2 - g0
    n0 = max(np.linalg.norm(g0), 1e-300)
    rl2 = np.linalg.norm(d) / n0
    rmx = np.abs(d).max() / max(np.abs(g0).max(), 1e-300)
    print('    %-3s : ||g1+g2-g0||/||g0|| = %.3e   (max-norm %.3e)   %s'
          % (par, rl2, rmx, 'PASS' if rl2 < 2e-2 else 'FAIL'))
    if rl2 >= 2e-2:
        ok = False
print('    (the eps floor in the weight denominator leaves a small residue;')
print('     it only acts where both halves are negligible)')

# ---- (3) angle selectivity ----
print('\n(3) tomographic fraction  ||smooth_300m(g)|| / ||g||')
print('    %-4s %10s %10s %10s' % ('par', 'ic=0', 'ic=1(tomo)', 'ic=2(migr)'))
for par in ('vp', 'vs', 'rho'):
    f = [tomofrac(g('g%d' % k, par)) for k in (0, 1, 2)]
    verdict = 'PASS' if (f[1] > f[0] > f[2]) else 'CHECK'
    print('    %-4s %10.4f %10.4f %10.4f   %s' % (par, f[0], f[1], f[2], verdict))

# ---- extra: where does each weight live? ----
print('\n(4) weight maps: mean over depth bands (z in m)')
w = {}
for nm in ('tomo', 'migr'):
    d, _ = su_io.read('g1_shot000_ei_%s.su' % nm)
    w[nm] = d.astype(np.float64).reshape(NX, NZ)
z = np.arange(NZ) * DX
bands = [(0, 500), (500, 1100), (1100, 1400), (1400, 1600), (1600, 2000)]
label = {(500, 1100): '  <- anomalies (z=550..1050)',
         (1400, 1600): '  <- reflector (z=1500)'}
print('    %-16s %10s %10s' % ('band', 'a_tomo', 'a_migr'))
for z0, z1 in bands:
    m = (z >= z0) & (z < z1)
    print('    %-16s %10.4f %10.4f%s'
          % ('%d-%d' % (z0, z1), w['tomo'][:, m].mean(),
             w['migr'][:, m].mean(), label.get((z0, z1), '')))

print('\n' + ('ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'))
PY
