#!/bin/bash
# taylor_vs.sh -- is g_Vs the true gradient of J with respect to Vs?
#
# The decisive test is the first-order Taylor expansion along a perturbation
# that lives ONLY in Vs:
#
#     J(m0 + eps*dm) = J(m0) + eps * <g, dm> + O(eps^2)
#
# so the ratio
#
#     r(eps) = ( J(eps) - J(0) ) / ( eps * <g, dm> )
#
# must tend to 1 as eps -> 0, and the remainder
#
#     R(eps) = | J(eps) - J(0) - eps*<g,dm> |
#
# must fall like eps^2 (halving eps quarters R). If g_Vs had the wrong sign,
# wrong scale factor or a missing term, r would tend to something other than 1
# and R would fall only like eps.
#
# dm is constructed so that dVp and drho are EXACTLY zero (model_true_cp and
# model_true_ro are byte copies of the background), which is what isolates
# g_Vs: <g,dm> = <g_Vs, dVs> with no contamination from the other two.
#
# EVERYTHING that makes the computed gradient differ from the exact gradient of
# the printed J is switched off:
#   grad_taper=0   masks a disc around sources/receivers
#   res_taper=0    tapers the residual outside the misfit
#   aper_taper=0   tapers the adjoint injection outside the misfit
#   smooth_grad=0  smooths the gradient after accumulation
#   scaling=0      Brossier scaling
#   precond=0      Hessian preconditioner
#   data_weight=0  per-component reweighting
# With any of these on the test MUST fail, and that would not be a bug.
#
# Usage:  ./taylor_vs.sh

set -e
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}
export OMP_NUM_THREADS=8

WORK=taylor_vs
rm -rf ${WORK}
mkdir -p ${WORK}
cd ${WORK}

DX=5; SIZEX=5000; SIZEZ=2000
VP0=2000; VS0=1150; RO0=2000
ANOM_Z=800; ANOM_R=250
XANOM_FAST=2000; VS_FAST=1265      # +10% in Vs
XANOM_SLOW=3000; VS_SLOW=1035      # -10% in Vs
REFL_Z=1500; RO_REFL=2200
SM_NX=21; SM_NZ=21; SM_ITER=8

XSRC=2500; ZSRC=50
XRCV1=0; XRCV2=${SIZEX}; DXRCV=10; ZRCV=100
TMOD=4.0; DTRCV=0.001
NTAPER=200

echo "--- models: background (reflector in rho) ---"
REFL="intt=def poly=0 x=0,${SIZEX} z=${REFL_Z},${REFL_Z} cp=${VP0} cs=${VS0} ro=${RO_REFL}"
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} ${REFL} \
    orig=0,0 file_base=model_bg.su verbose=0

# Vs-only anomalies: cp is held at VP0 and ro at RO0 inside the ellipses.
ANOM="intt=elipse poly=0 x=${XANOM_FAST} z=${ANOM_Z} var=${ANOM_R},${ANOM_R} cp=${VP0} cs=${VS_FAST} ro=${RO0} intt=elipse poly=0 x=${XANOM_SLOW} z=${ANOM_Z} var=${ANOM_R},${ANOM_R} cp=${VP0} cs=${VS_SLOW} ro=${RO0}"
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} ${REFL} ${ANOM} \
    orig=0,0 file_base=model_sharp.su verbose=0

smooth file_in=model_sharp_cs.su file_out=model_true_cs.su \
    ntsm=${SM_NZ} nxsm=${SM_NX} niter=${SM_ITER} verbose=0 2>/dev/null
# Force dVp = drho = 0 EXACTLY -- byte copies, not recomputed models.
cp model_bg_cp.su model_true_cp.su
cp model_bg_ro.su model_true_ro.su

echo "  perturbation support (must be cs ONLY):"
for P in cp cs ro; do
    if cmp -s model_true_${P}.su model_bg_${P}.su; then
        echo "    _${P}: identical"
    else
        echo "    _${P}: differs"
    fi
done

echo "--- wavelet ---"
makewave fp=14 fmin=2 fmax=25 dt=0.001 nt=4096 file_out=wave.su t0=0.10 verbose=0

echo "--- observed data (Vs-perturbed true model) ---"
${FDELFWI}/fdelfwi_fwd \
    file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=6 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DXRCV} zrcv1=${ZRCV} zrcv2=${ZRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=1 dxshot=0 \
    ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 > /dev/null

# run_grad <model_cs> <out_base> <logfile>
run_grad () {
    ${FDELFWI}/test_gradient_visual \
        file_cp=model_bg_cp.su file_cs=$1 file_den=model_bg_ro.su \
        file_src=wave.su file_rcv=syn \
        ischeme=3 iorder=4 src_type=6 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
        xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DXRCV} zrcv1=${ZRCV} zrcv2=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 dxshot=0 \
        ntaper=${NTAPER} grad_taper=0 res_taper=0 aper_taper=0 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=100 chk_base=chk_tv \
        param=2 scaling=0 precond=0 smooth_grad=0 \
        vp_min=1500 vp_max=4500 vs_min=500 vs_max=2500 \
        rho_min=1500 rho_max=2500 \
        data_weight=0 misfit=2 kh_normal=-1 \
        comp=_rvx,_rvz,_rtxz,_rtzz \
        file_obs=obs out_base=$2 > $3 2>&1
}

echo "--- gradient at m0 (eps = 0) ---"
run_grad model_bg_cs.su g0 log_g0.txt
grep -E "Total misfit" log_g0.txt | tail -1

# <g, dm> with dm = m_true - m_bg, supported in Vs only
python3 - <<'PY' | tee gdm.txt
import sys, numpy as np
sys.path.insert(0,'/rcp3/software/codes/OpenSource_SL10/fdelfwi/tests/kh_misfit')
import su_io
g,_  = su_io.read('g0_vs_raw.su')
bg,_ = su_io.read('model_bg_cs.su')
tr,_ = su_io.read('model_true_cs.su')
d = (tr-bg).astype(np.float64)
print('%.12e' % (g.astype(np.float64)*d).sum())
PY
GDM=$(tail -1 gdm.txt)
echo "  <g_Vs, dVs> = ${GDM}"

echo "--- J( m0 + eps*dm ) ---"
for EPS in 1.0 0.5 0.25 0.125 0.0625; do
    python3 - "${EPS}" <<'PY'
import sys, numpy as np
sys.path.insert(0,'/rcp3/software/codes/OpenSource_SL10/fdelfwi/tests/kh_misfit')
import su_io
eps=float(sys.argv[1])
bg,h = su_io.read('model_bg_cs.su')
tr,_ = su_io.read('model_true_cs.su')
su_io.write('m_cs.su', (bg+eps*(tr-bg)).astype(np.float32), h)
PY
    run_grad m_cs.su ge_${EPS} log_${EPS}.txt
    echo "  eps=${EPS}  $(grep -E '=== Total misfit' log_${EPS}.txt | tail -1 | sed 's/.*=== //')"
done

echo ""
echo "================ Taylor test on g_Vs ================"
python3 - "${GDM}" <<'PY'
import re, sys
gdm = float(sys.argv[1])
def J_of(fn):
    return float(re.findall(r'=== Total misfit: ([0-9.e+-]+)', open(fn).read())[-1])
J0 = J_of('log_g0.txt')
print('J(0)        = %+.8e' % J0)
print('<g_Vs,dVs>  = %+.8e' % gdm)
print()
print('%8s %18s %12s %14s %10s' % ('eps','J(eps)','r(eps)','remainder R','R/eps^2'))
prev=None
for e in ('1.0','0.5','0.25','0.125','0.0625'):
    J = J_of('log_%s.txt' % e)
    eps=float(e)
    r = (J-J0)/(eps*gdm)
    R = abs(J-J0-eps*gdm)
    print('%8s %18.8e %12.5f %14.4e %10.3e' % (e,J,r,R,R/eps**2))
print()
print('PASS if r -> 1 and R/eps^2 is roughly constant (R falls 4x when eps halves).')
PY
