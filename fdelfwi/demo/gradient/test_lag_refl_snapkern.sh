#!/bin/bash
#
# test_lag_refl_snapkern.sh -- case B of test_lag_reflector.sh, re-run so that
# the GRADIENT INTEGRAND can be plotted frame by frame next to the two
# wavefields that produce it.
#
# Why a re-run at all.  lag_reflector/snapB already holds forward and adjoint
# snapshots, but only vx and vz.  The kernel the code actually accumulates
# (fwi_gradient.c:58 accumGradient) is
#
#     K_lam(x,t) = (psi_txx + psi_tzz) * (D+x vx_f + D+z vz_f)
#     K_mu (x,t) = 2(psi_txx D+x vx_f + psi_tzz D+z vz_f)
#                  + psi_txz (D-z vx_f + D-x vz_f)
#
# so it needs the ADJOINT STRESSES, which were never written.  No amount of
# post-processing recovers them from psi_vx/psi_vz -- that would require
# integrating the adjoint system -- so the snapshots are simply re-run with
# sna_type_txx/tzz/txz=1 on both passes.
#
# Two things are deliberately different from snapB:
#
#   1. FULL spatial resolution, dxsnap=dzsnap=dx.  snapB used dxsnap=12, a
#      factor-2 decimation.  The stencils above are 4th-order staggered
#      differences on the dx grid; evaluating them on a decimated grid would
#      not be the operator the code uses, so the comparison would be of my
#      arithmetic rather than of the code's.
#   2. lag_nfreq=0.  The lag cubes are not needed here and they dominate both
#      the runtime and the memory, so the pass is cheap.  Everything else --
#      model, geometry, mute, comp, taper -- is copied from case B verbatim.
#   3. sna_type_pp=1, which is NOT decoration.  writeSnapTimes.c SHIFTS the
#      staggered fields before storing them -- vx is written from vx[ix+1,iz]
#      (line 150), vz from vz[ix,iz+1] (line 159), txz from txz[ix+1,iz+1]
#      (line 186), while txx and tzz are unshifted -- so the code's stencils
#      cannot be applied to the snapshot arrays as if they were the raw
#      wavefields.  The _spp file is a divergence that the C code computes
#      ITSELF at the P node (line 193), so reproducing it in python from the
#      shifted vx/vz arrays is an exact, independent check that the index
#      algebra in lag_refl_snapkern.py is right.  The plotter asserts it.
#
# This stage CONSUMES runB/obsm_000_*.su, the already-muted observed data, so
# the mute pass is not repeated and the residual is bit-identical to case B's.
# test_lag_reflector.sh must have been run first.
#
# Usage
#   ./test_lag_refl_snapkern.sh [run|figs|all]

set -e
set -o pipefail
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}
export OMP_NUM_THREADS=16

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${HERE}/lag_reflector
SNAP=${WORK}/snapB_kern

STAGE=${1:-all}
case ${STAGE} in
    run|figs|all) ;;
    *) echo "usage: $0 [run|figs|all]" >&2; exit 2 ;;
esac

# ---- copied verbatim from test_lag_reflector.sh -------------------------
DX=6; SIZEX=5004; SIZEZ=3000
VP0=2000; VS0=1200; RO0=2000
VFAC=1.10
ZREFL=2298
XSRC=1500; ZSRC=798
XRCV=3504; ZRCV=798
XSPEC=2502; ZSPEC=2298
SIGMA=300; APCT=-5
XA=1800; ZA=1248              # Vp anomaly, on the PP DOWN-leg
XB=3204; ZB=1248              # Vs anomaly, on the PP UP-leg
TMOD=4.0; DTRCV=0.001
NTAPER=200
GRAD_TAPER=30
WAV_FMIN=2; WAV_FMAX=25; WAV_T0=0.15
FPEAK=$(echo ${WAV_FMAX} | awk '{printf "%.4f", 0.4425*$1}')

# ---- snapshot schedule ---------------------------------------------------
# CHANGED from the snapB schedule (1.0 to 3.4 s every 0.2 s, 13 frames).  Two
# reasons, and the second is the important one:
#
#   a) 0.2 s is coarser than the wavelet period, 1/11.0625 = 90.4 ms, so
#      consecutive frames were uncorrelated snapshots of an oscillating
#      integrand.  0.01 s gives ~9 frames per period.
#   b) at 0.01 s the frames can be SUMMED.  The code forms
#      sum_t dt K over all 4000 steps; sampling every 10th step and weighting
#      by dtsnap is a quadrature of the same integral, so the running total
#      can be compared against the code's own <out_base>_vp_raw.su.  That
#      turns the figure from a plausible picture into a verified one, and it
#      is the only way to see WHEN each limb of the kernel is deposited.
#
# Window: the wavelet is centred at t0 = 0.15 s and the latest event of
# interest is SS at 3.007 + 0.15 = 3.157 s, so 0.10 to 3.90 s covers the
# record bar a quiet head and tail.  381 frames x 835 x 501 x 4 bytes =
# 638 MB per field, 6 fields x 2 passes = 7.6 GB.
TSNAP1=0.10; TSNAP2=3.90; DTSNAP=0.01

# Frames drawn in the figure, chosen on the case B kinematics.  Straight-ray
# times with z_refl - z_src = 1500 m and half-offset 1002 m, plus t0 = 0.15:
#   1.152  direct P reaches the receiver
#   1.954  PP specular
#   2.820  direct S reaches the receiver
# 1.00 and 1.40 bracket the direct P, 1.80/1.96 straddle the PP specular,
# and 2.40/2.80 follow the S.
SNAP_TIME=1.00,1.40,1.80,1.96,2.40,2.80

if [ "${STAGE}" = "run" ] || [ "${STAGE}" = "all" ]; then
echo "=========================================================="
echo " test_lag_refl_snapkern.sh -- stage: run"
echo "=========================================================="

if [ ! -f ${WORK}/runB/obsm_000_rvx.su ]; then
    echo "ERROR: ${WORK}/runB/obsm_000_rvx.su is missing." >&2
    echo "Run ./test_lag_reflector.sh grad first -- this stage reuses case" >&2
    echo "B's MUTED observed data so that the residual is identical." >&2
    exit 1
fi

rm -rf ${SNAP}
mkdir -p ${SNAP}
cd ${SNAP}
for F in refl_cp refl_cs refl_ro wave; do ln -sf ../${F}.su .; done
for C in rvx rvz rtxx rtzz rp; do
    ln -sf ../runB/obsm_000_${C}.su obsm_000_${C}.su
done

echo "  gradient pass with full-resolution fwd+adj snapshots (vx vz txx tzz txz)"
${FDELFWI}/test_gradient_visual \
    file_cp=refl_cp.su file_cs=refl_cs.su file_den=refl_ro.su \
    file_src=wave.su file_rcv=syn_snap \
    ischeme=3 iorder=4 src_type=6 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
    xrcva=${XRCV} zrcva=${ZRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
    ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=0 aper_taper=0 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=50 chk_base=chk_snap \
    param=2 scaling=0 precond=0 smooth_grad=0 \
    vp_min=1000 vp_max=4500 vs_min=500 vs_max=2500 \
    rho_min=1500 rho_max=2500 \
    data_weight=0 misfit=0 comp=_rvx,_rvz \
    lag_nfreq=0 \
    file_snap=snapf file_snap_adj=snapa \
    tsnap1=${TSNAP1} tsnap2=${TSNAP2} dtsnap=${DTSNAP} \
    dxsnap=${DX} dzsnap=${DX} \
    sna_type_vx=1 sna_type_vz=1 \
    sna_type_txx=1 sna_type_tzz=1 sna_type_txz=1 sna_type_p=0 \
    sna_type_pp=1 \
    file_obs=obsm out_base=snapg > log_snapkern.txt 2>&1

grep -E "Total misfit" log_snapkern.txt | tail -1
echo "  snapshot files:"
ls -1 snapf_s*.su snapa_s*.su | sed 's/^/    /'
rm -rf chk_snap*
echo ""
echo "stage run complete."
fi

if [ "${STAGE}" = "figs" ] || [ "${STAGE}" = "all" ]; then
echo ""
echo "=========================================================="
echo " test_lag_refl_snapkern.sh -- stage: figs"
echo "=========================================================="
cd ${SNAP}
# The FD time step is CHOSEN BY THE CODE (stability), not by us, so it is read
# back out of the log rather than assumed equal to dtrcv.  It is the dt that
# multiplies the kernel in accumGradient.
FDDT=$(awk '/ntsam *=/{for(i=1;i<=NF;i++) if($i=="dt")
            {v=$(i+2); sub(/\(.*/,"",v); print v; exit}}' log_snapkern.txt)
if [ -z "${FDDT}" ]; then
    echo "ERROR: could not read the FD dt out of log_snapkern.txt" >&2
    exit 1
fi
echo "  FD time step read from the log: dt = ${FDDT} s"
# The chain rule Lame -> (Vp,Vs) is pointwise in the BACKGROUND model
# (convertGradientToVelocity, fwi_gradient.c:580), and case B's background is
# not homogeneous -- it is VFAC faster below ZREFL -- so the model files are
# passed in rather than the scalars VP0/VS0/RO0.
python3 ${HERE}/lag_refl_snapkern.py \
    --dir=${SNAP} --dx=${DX} \
    --tsnap1=${TSNAP1} --tsnap2=${TSNAP2} --dtsnap=${DTSNAP} \
    --dt=${FDDT} --times=${SNAP_TIME} \
    --model-cp=refl_cp.su --model-cs=refl_cs.su --model-ro=refl_ro.su \
    --base=snapg --zrefl=${ZREFL} \
    --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} \
    --xa=${XA} --za=${ZA} --xb=${XB} --zb=${ZB} --sigma=${SIGMA}
echo ""
echo "stage figs complete."
fi
