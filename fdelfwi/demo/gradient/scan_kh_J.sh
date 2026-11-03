#!/bin/bash
# scan_kh_J.sh -- is the K-H misfit actually MINIMISED at the true model?
#
# J_KH = sum_r w_r n_z ( dTxz*dVx + dTzz*dVz ) is a BILINEAR CROSS-TERM: it
# pairs a stress residual with a velocity residual.  Unlike the L2 norm it is
# therefore not obviously sign-definite, and "J = 0 at m_true" does not by
# itself make m_true a MINIMUM -- it could be a saddle.
#
# This walks the straight line m(eps) = m_bg + eps*(m_true - m_bg) for
# eps = 0, 0.25, 0.5, 0.75, 1.0 and records J at each point, reusing the
# observed data already in the run directory.  No adjoint is needed for J, but
# test_gradient_visual computes the gradient anyway, so snapshots and per-shot
# output are switched off to keep it cheap.
#
# Read the result like this:
#   J falling monotonically to 0   -> m_true is a minimum, the objective is usable
#   J going NEGATIVE on the way    -> m_true is a saddle, descent runs away
#
# Usage:  ./scan_kh_J.sh [srcdir]        (default gradient_1src_1rcv_kh)

set -e
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}
export OMP_NUM_THREADS=4

SRC=${1:-gradient_1src_1rcv_kh}
WORK=scan_kh_J
mkdir -p ${WORK}
cp ${SRC}/wave.su ${SRC}/model_bg_*.su ${SRC}/model_true_*.su ${WORK}/
cp ${SRC}/obs_000_r*.su ${WORK}/
cd ${WORK}

for EPS in 0.00 0.25 0.50 0.75 1.00; do
    python3 - "${EPS}" <<'PY'
import sys, os
sys.path.insert(0, '/rcp3/software/codes/OpenSource_SL10/fdelfwi/tests/kh_misfit')
import numpy as np, su_io
eps = float(sys.argv[1])
for p in ('cp', 'cs', 'ro'):
    bg, h = su_io.read('model_bg_%s.su' % p)
    tr, _ = su_io.read('model_true_%s.su' % p)
    su_io.write('m_%s.su' % p, (bg + eps * (tr - bg)).astype(np.float32), h)
PY

    echo "=== eps=${EPS} ==="
    ${FDELFWI}/test_gradient_visual \
        file_cp=m_cp.su file_cs=m_cs.su file_den=m_ro.su \
        file_src=wave.su file_rcv=syn \
        ischeme=3 iorder=4 src_type=6 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
        dtrcv=0.001 tmod=4.0 verbose=1 \
        xrcv1=0 xrcv2=5000 dxrcv=10 zrcv1=100 zrcv2=100 \
        xsrc=2500 zsrc=50 nshot=1 dxshot=0 \
        ntaper=200 grad_taper=0 res_taper=100 aper_taper=200 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=100 chk_base=chk_scan \
        param=2 scaling=1 precond=0 smooth_grad=1 \
        vp_min=1500 vp_max=2500 vs_min=800 vs_max=1500 \
        rho_min=1800 rho_max=2400 \
        data_weight=0 misfit=2 kh_normal=-1 \
        comp=_rvx,_rvz,_rtxz,_rtzz \
        file_obs=obs out_base=g_${EPS} 2>&1 | tee log_${EPS}.txt | grep -i "misfit\|flux"
done

echo ""
echo "================ J( m_bg + eps*(m_true-m_bg) ) ================"
for EPS in 0.00 0.25 0.50 0.75 1.00; do
    printf "  eps=%s  %s\n" "${EPS}" "$(grep -i 'Total misfit\|Shot 0 misfit' log_${EPS}.txt | tail -1)"
done
