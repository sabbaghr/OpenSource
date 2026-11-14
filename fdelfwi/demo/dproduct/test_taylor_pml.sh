#!/bin/bash
#
# test_taylor_pml.sh -- Taylor gradient verification under a CFS-CPML.
#
# WHY THIS EXISTS SEPARATELY FROM test_taylor.sh
# ----------------------------------------------
# test_taylor.sh is hardcoded to left=4 right=4 top=4 bottom=4 ntaper=30, and so
# are test_dotproduct.sh and test_hessian_dp.sh.  Every gradient-level check in
# the tree therefore runs on a TAPER boundary.  What has been proven under a PML
# is the bare propagator (test_cpml_dp) and the wave operator (test_wave_op_dp)
# -- never the gradient.
#
# The blocker was that the CFS-CPML psi memory variables were not checkpointed,
# so adj_shot restarted every segment with psi = 0 and the reconstructed forward
# field was wrong in and near the PML.  That is fixed (cpml_state.c + the psi
# blob in checkpoint.c, regression-tested by demo/test_cpml_checkpoint.sh), so
# this test should now pass.  Before the fix it could not.
#
# WHAT A PASS MEANS
# -----------------
# The Taylor test measures
#
#     J(m + eps*dm) - J(m) - eps*<g, dm>  =  O(eps^2)
#
# so it checks the gradient against the misfit functional the forward operator
# actually implements.  It is the end-to-end statement that g is the gradient
# -- stronger than a dot-product test, which only says the adjoint is the
# transpose of whatever the forward happens to be.
#
# chk_skipdt=100 is deliberate: it forces re-propagation between checkpoints,
# which is exactly the path psi checkpointing had to fix.  Do not set it to 1
# "to be safe" -- that would skip the code under test.
#
# Usage:
#   ./test_taylor_pml.sh        # all parameters
#   ./test_taylor_pml.sh 1      # Vp only
#   ./test_taylor_pml.sh 2      # Vs only
#   ./test_taylor_pml.sh 3      # rho (velocity parameterization)
#   ./test_taylor_pml.sh 4      # rho (Lame direct)

set -e

TEST_PARAM=${1:-0}

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
export PATH=${BIN}:${PATH}

NPML=${NPML:-30}
BND="left=2 right=2 top=2 bottom=2 npml=${NPML}"

WORKDIR=$(cd "$(dirname "$0")" && pwd)/taylor_test_pml
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "========================================================="
echo " TAYLOR GRADIENT VERIFICATION -- CFS-CPML"
echo "   ${BND}"
echo "   test_param=${TEST_PARAM} (0=all, 1=Vp, 2=Vs, 3=rho_vel, 4=rho_Lame)"
echo "========================================================="

# Same model/geometry/wavelet as test_taylor.sh so the two are comparable:
# homogeneous background + a Vp point diffractor, which puts scattered energy
# in d_obs that is absent from d_syn(m0) and so makes the gradient non-trivial.
echo "--- true model (background + Vp diffractor) ---"
makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    intt=diffr x=500 z=500 cp=2500 cs=1150 ro=2000 var=10 \
    orig=0,0 file_base=true_model.su verbose=0

echo "--- wavelet ---"
makewave fp=15 dt=0.001 nt=1024 fmax=25 file_out=wave.su t0=0.10 verbose=0

echo "--- observed data (true model, SAME boundaries as the gradient run) ---"
${FDELFWI}/fdelfwi_fwd \
    file_cp=true_model_cp.su file_cs=true_model_cs.su file_den=true_model_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=7 \
    rec_type_vz=1 \
    dtrcv=0.004 tmod=1.0 verbose=0 \
    xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
    xsrc=500 zsrc=200 ${BND}

echo "--- starting model (homogeneous) ---"
makemod sizex=1000 sizez=1000 dx=5 dz=5 \
    cp0=2000 cs0=1150 ro0=2000 \
    orig=0,0 file_base=start_model.su verbose=0

echo ""
echo "--- Taylor test ---"
${FDELFWI}/test_taylor \
    file_cp=start_model_cp.su file_cs=start_model_cs.su file_den=start_model_ro.su \
    file_src=wave.su file_rcv=syn_taylor \
    file_obs=obs_000 \
    ischeme=3 iorder=4 src_type=7 \
    rec_type_vz=1 \
    dtrcv=0.004 tmod=1.0 verbose=1 \
    xrcv1=200 xrcv2=800 zrcv1=700 zrcv2=700 dxrcv=10 \
    xsrc=500 zsrc=200 ${BND} \
    chk_skipdt=100 chk_base=chk_taylor \
    pert_pct=0.01 seed=12345 \
    test_param=${TEST_PARAM} \
    comp=_rvz \
    neps=10 eps_min=1e-8 eps_max=1e-1

rm -f chk_taylor_*.bin chk_taylor_dummy_*.bin
rm -f cp_taylor_pert.su cs_taylor_pert.su ro_taylor_pert.su
rm -f residual_taylor*.su syn_taylor_pert_*.su syn_taylor_*.su

echo ""
echo "========================================================="
echo " ${WORKDIR}"
echo "========================================================="
