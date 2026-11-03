#!/bin/bash
# test_energy_ic_20shot.sh -- energy-norm angle filter on a 20-shot survey.
#
# Same physics as test_energy_ic.sh (which is one shot at x=2500), but with a
# 20-shot surface spread.  The point is NOT to re-check the identities -- those
# are already verified per shot -- it is to see what the filter does once many
# shots are stacked, because stacking is itself a (crude) angle filter:
# summing over source position attenuates the specular reflector imprint
# relative to the transmission update even with energy_ic=0.  The question this
# script answers is how much is LEFT for the filter to do at 20 shots.
#
# Model (inherited from taylor_vs/):
#   5000 x 2000 m, dx=5, Vp=2000, Vs=1150, rho=2000
#   smoothed Vs-only anomalies at (2000,800) +10% and (3000,800) -10%
#   rho reflector at z=1500 present in BOTH models -> everything the gradient
#   puts there is artifact, which is what makes this a clean test.
#
# Geometry: 20 shots at x = 400 + i*220  (400 .. 4580), z = 50
#           receivers z=100, x=0..5000 every 10 m
#
# The three energy_ic runs are INDEPENDENT, so they run in parallel, each in
# its own subdirectory.  Separate directories are required, not just separate
# out_base: file_rcv (syn_*), chk_base and res_*.su would otherwise collide.
#
# Usage:  ./test_energy_ic_20shot.sh        (expects taylor_vs/ to exist;
#                                            run ./taylor_vs.sh first)

set -e
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}

HERE=$(cd "$(dirname "$0")" && pwd)
cd "${HERE}"

SRC=taylor_vs
WORK=energy_ic_20

NSHOT=${NSHOT:-20}
XSRC=${XSRC:-400}
DXSHOT=${DXSHOT:-220}
ZSRC=50

# 3 runs in parallel x 16 threads = 48 of 64 hardware threads.
OMP=${OMP:-16}

DX=5; SIZEX=5000
XRCV1=0; XRCV2=${SIZEX}; DXRCV=10; ZRCV=100
TMOD=4.0; DTRCV=0.001
NTAPER=200

if [ ! -f ${SRC}/model_bg_cs.su ]; then
    echo "ERROR: ${SRC}/ is missing its models.  Run ./taylor_vs.sh first."
    exit 1
fi

rm -rf ${WORK}
mkdir -p ${WORK}
cp ${SRC}/model_bg_*.su ${SRC}/model_true_*.su ${SRC}/wave.su ${WORK}/
cd ${WORK}

# ----------------------------------------------------------------- obs data
# fdelfwi_fwd, NOT fdelmodc: the two have incompatible boundary
# implementations and mixing them leaves a ~1% direct-wave residual even at the
# true model.  fdelfwi_fwd also zero-pads fileno, so files land as
# obs_000_r*.su with no renaming step.
echo "--- observed data, ${NSHOT} shots (true model) ---"
OMP_NUM_THREADS=$(nproc) ${FDELFWI}/fdelfwi_fwd \
    file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=6 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DXRCV} zrcv1=${ZRCV} zrcv2=${ZRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=${NSHOT} dxshot=${DXSHOT} \
    ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 > log_fwd.txt 2>&1
echo "    $(ls obs_*_rvz.su | wc -l) shots written"

# ----------------------------------------------------------------- gradients
run_grad () {   # run_grad <energy_ic> <subdir> <out_base>
    local EI=$1 DIR=$2 OB=$3
    mkdir -p ${DIR}
    ( cd ${DIR}
      for f in ../model_*.su ../wave.su ../obs_*.su; do ln -sf $f .; done
      OMP_NUM_THREADS=${OMP} ${FDELFWI}/test_gradient_visual \
        file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
        file_src=wave.su file_rcv=syn \
        ischeme=3 iorder=4 src_type=6 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
        xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DXRCV} zrcv1=${ZRCV} zrcv2=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=${NSHOT} dxshot=${DXSHOT} \
        ntaper=${NTAPER} grad_taper=0 res_taper=0 aper_taper=0 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=100 chk_base=chk_${OB} \
        param=2 scaling=0 precond=0 smooth_grad=0 \
        vp_min=1500 vp_max=4500 vs_min=500 vs_max=2500 \
        rho_min=1500 rho_max=2500 \
        data_weight=0 misfit=2 kh_normal=-1 \
        comp=_rvx,_rvz,_rtxz,_rtzz \
        file_obs=obs energy_ic=${EI} out_base=${OB} > ../log_${EI}.txt 2>&1 )
}

echo "--- gradients: energy_ic = 0, 1, 2 in parallel (${OMP} threads each) ---"
date +"    start %H:%M:%S"
run_grad 0 ei0 g0 &  P0=$!
run_grad 1 ei1 g1 &  P1=$!
run_grad 2 ei2 g2 &  P2=$!
wait ${P0} ${P1} ${P2}
date +"    done  %H:%M:%S"

for EI in 0 1 2; do
    echo "  energy_ic=${EI}: $(grep -E '=== Total misfit' log_${EI}.txt | tail -1 | sed 's/.*=== //')"
done

# ----------------------------------------------------------------- analysis
echo ""
python3 - <<'PY'
import sys
import numpy as np
sys.path.insert(0, '/rcp3/software/codes/OpenSource_SL10/fdelfwi/tests/kh_misfit')
import su_io
from scipy.ndimage import uniform_filter

NZ, NX, DX = 401, 1001, 5.0
SM = int(round(300.0 / DX)) | 1
DIR = {0: 'ei0/g0', 1: 'ei1/g1', 2: 'ei2/g2'}


def g(k, par):
    d, _ = su_io.read('%s_%s_raw.su' % (DIR[k], par))
    return d.astype(np.float64).reshape(NX, NZ)


z = np.arange(NZ) * DX
anom = (z >= 550) & (z <= 1050)
refl = (z >= 1400) & (z <= 1600)

print('=========== energy_ic, 20 shots ===========')

print('\n(1) partition of unity   g(ic=1) + g(ic=2) = g(ic=0)')
for par in ('vp', 'vs', 'rho'):
    g0, g1, g2 = g(0, par), g(1, par), g(2, par)
    r = np.linalg.norm(g1 + g2 - g0) / max(np.linalg.norm(g0), 1e-300)
    print('    %-3s : %.3e   %s' % (par, r, 'PASS' if r < 2e-2 else 'FAIL'))

print('\n(2) tomographic fraction  ||smooth_300m(g)|| / ||g||')
print('    %-4s %10s %10s %10s' % ('par', 'ic=0', 'ic=1', 'ic=2'))
for par in ('vp', 'vs', 'rho'):
    f = []
    for k in (0, 1, 2):
        a = g(k, par)
        n = np.linalg.norm(a)
        f.append(np.linalg.norm(uniform_filter(a, SM)) / n if n > 0 else 0.0)
    print('    %-4s %10.4f %10.4f %10.4f   %s'
          % (par, f[0], f[1], f[2], 'PASS' if f[1] > f[0] > f[2] else 'CHECK'))

print('\n(3) share of ||g||^2 : true anomaly band vs reflector artifact')
print('    %-4s %-6s %9s %9s %9s' % ('par', 'run', 'anomaly', 'reflector', 'ratio'))
for par in ('vp', 'vs'):
    for k, nm in ((0, 'ic=0'), (1, 'ic=1'), (2, 'ic=2')):
        a = g(k, par)
        tot = (a ** 2).sum()
        ea = (a[:, anom] ** 2).sum() / tot
        er = (a[:, refl] ** 2).sum() / tot
        print('    %-4s %-6s %9.4f %9.4f %9.3f' % (par, nm, ea, er, ea / er))

print('\n(4) correlation with the true Vs perturbation (the thing to recover)')
bg, _ = su_io.read('model_bg_cs.su')
tr, _ = su_io.read('model_true_cs.su')
dm = (tr - bg).astype(np.float64).reshape(NX, NZ)
dmn = np.linalg.norm(dm)
for k, nm in ((0, 'ic=0'), (1, 'ic=1'), (2, 'ic=2')):
    a = g(k, 'vs')
    c = float((a * dm).sum() / (np.linalg.norm(a) * dmn))
    print('    g_vs %-6s cos(g, dVs) = %+.4f' % (nm, c))
print('    (sign is convention; |cos| is what matters -- higher = the gradient')
print('     points more at the real anomaly and less at the artifact)')
PY
