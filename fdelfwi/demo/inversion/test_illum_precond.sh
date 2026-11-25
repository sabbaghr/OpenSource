#!/bin/bash
# test_illum_precond.sh -- illum_precond=1 (source-illumination PLBFGS H0)
#                          on the lag_tomo=1, Vp-only, hydrophone inversion
#
# Reuses the model and observed data of test_lag_tomo_inversion.sh by symlink
# (run that script first): 2000 x 1000 m, density reflector at 600 m in BOTH
# models, background 8% fast, 3 shots, _rp data.
#
# The run is the production configuration in miniature:
#   comp=_rp  param=2 active_params=vp  lag_tomo=1  lag_nv=2
#   lag_theta_pass/stop (full opening angle)  algorithm=2 illum_precond=1
#
# The one thing CHECKED (code correctness of the new path):
#   PRECO:  <grad>_initial_preco_vp.su == nu / (I + beta) * <grad>_initial_vp.su
#           with I = <grad>_illum.su, beta = illum_eps * max(I outside the
#           taper mask), nu = ||g|| / ||P g||.  Checked as: the ratio
#           preco/g is proportional to 1/(I+beta) (corr ~ 1), and
#           ||preco|| == ||g|| (norm preservation).
# Everything else (cost per iteration) is printed, not asserted.
#
# Usage:  ./test_illum_precond.sh

set -e
set -o pipefail

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

HERE=$(pwd)
REF=${HERE}/lag_tomo_inversion
[ -f ${REF}/obs_002_rp.su ] || { echo "run ./test_lag_tomo_inversion.sh first"; exit 1; }

WORK=illum_precond
mkdir -p ${WORK}
cd ${WORK}
for f in ${REF}/model_bg_*.su ${REF}/model_true_*.su ${REF}/wave.su ${REF}/obs_*.su; do
    ln -sf ${f} .
done

NZ=201
DTRCV=0.004; TMOD=1.5; NTAPER=100; GRAD_TAPER=20
NSHOTS=3; XSRC_START=400; XSRC_STEP=600; ZSRC=10
XRCV1=100; XRCV2=1900; ZRCV=100; DXRCV=20
LAG_NFREQ=31; LAG_FMIN=2; LAG_FMAX=18; LAG_NT=31; LAG_DT=0.005
THETA_PASS=${THETA_PASS:-82}; THETA_STOP=${THETA_STOP:-72}
ILLUM_EPS=${ILLUM_EPS:-1e-2}
NITER=${NITER:-3}

echo "========================================================="
echo " ILLUMINATION-PRECONDITIONED lag_tomo INVERSION (Vp only, _rp)"
echo "   cone 2theta: stop < ${THETA_STOP} deg, pass > ${THETA_PASS} deg, lag_nv=2"
echo "   PLBFGS + illum_precond=1  illum_eps=${ILLUM_EPS}  niter=${NITER}"
echo "========================================================="

${FDELFWI}/fwi_inversion \
    file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn_ill \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
    xsrc=${XSRC_START} zsrc=${ZSRC} nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=0 aper_taper=0 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=50 chk_base=chk_ill \
    param=2 active_params=vp \
    scaling=0 precond=0 smooth_grad=0 \
    data_weight=0 misfit=0 comp=_rp \
    vp_min=1500 vp_max=3000 vs_min=800 vs_max=1800 rho_min=1800 rho_max=2400 \
    file_obs=obs file_grad=gill \
    lag_tomo=1 lag_nfreq=${LAG_NFREQ} lag_fmin=${LAG_FMIN} lag_fmax=${LAG_FMAX} \
    lag_nt=${LAG_NT} lag_dt=${LAG_DT} lag_nv=2 \
    lag_theta_pass=${THETA_PASS} lag_theta_stop=${THETA_STOP} \
    algorithm=2 illum_precond=1 illum_eps=${ILLUM_EPS} \
    lbfgs_mem=5 niter=${NITER} nls_max=5 conv=1e-6 write_iter=1 \
    > log_illum.txt 2>&1 || { tail -30 log_illum.txt; exit 1; }
rm -rf chk_ill_* fwi_rank*

grep -E "illum_precond" log_illum.txt | head -20

echo ""
echo "========================= PRECO CHECK ========================="
python3 - <<PYEOF
import numpy as np
def readsu(fn, ns):
    raw = np.fromfile(fn, dtype=np.uint8); rec = 240 + 4*ns
    return np.stack([raw[i*rec+240:(i+1)*rec].view(np.float32)
                     for i in range(raw.size//rec)]).astype(np.float64)
ns = ${NZ}
g  = readsu('gill_initial_vp.su', ns)
gp = readsu('gill_initial_preco_vp.su', ns)
I  = readsu('gill_illum.su', ns)
P  = readsu('gill_illum_precond.su', ns)
beta = 1.0/P.max() - I.min()          # P = 1/(I+beta) at the least-lit cell
P_expect = 1.0/(I + beta)
fail = 0
e = np.abs(P - P_expect).max()/P.max()
print('  P == 1/(I+beta)            max rel err = %.2e  %s' % (e, 'PASS' if e < 1e-4 else 'FAIL')); fail |= e >= 1e-4
nz = np.abs(g) > 1e-6*np.abs(g).max()
nu = np.median(gp[nz]/(P[nz]*g[nz]))
e = np.abs(gp - nu*P*g).max()/np.abs(gp).max()
print('  preco == nu P g            max rel err = %.2e  (nu=%.4e)  %s' % (e, nu, 'PASS' if e < 1e-4 else 'FAIL')); fail |= e >= 1e-4
r = np.linalg.norm(gp)/np.linalg.norm(g)
print('  ||preco|| / ||g||          = %.6f  %s' % (r, 'PASS' if abs(r-1) < 1e-4 else 'FAIL')); fail |= abs(r-1) >= 1e-4
print('  illumination dynamic range (unmasked interior) = %.1f dB' % (10*np.log10(I.max()/max(I.min(),1e-300))))
if fail: raise SystemExit('PRECO CHECK FAILED')
PYEOF

echo ""
echo "======================== CONVERGENCE ========================"
cat iterate_PL*.dat 2>/dev/null || ls *.dat
echo ""
echo "  QC: gill_illum.su  gill_illum_precond.su  gill_initial_{vp,preco_vp}.su"
echo "  output in: $(pwd)"
