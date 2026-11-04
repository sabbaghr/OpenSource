#!/bin/bash
#SBATCH --job-name=efwi_lag_tomo
#SBATCH --partition=geop
#SBATCH --nodes=11
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=efwi_lag_tomo_%j.log
#
# test_lag_tomo_mpi_mio.sh -- multi-node L-BFGS inversion driven by the
#                             CONE-FILTERED time-lag gradient (lag_tomo=1)
#
# This is the production version of demo/inversion/test_lag_tomo_inversion.sh.
# Same medium, same lag axis, same cone; what changes is that the shots are
# spread over MPI ranks on several nodes and the run does a real L-BFGS
# inversion instead of a single gradient.
#
#   sbatch test_lag_tomo_mpi_mio.sh
#
# Rank layout
# -----------
# 22 ranks = 11 nodes x 2 ranks (the whole geop partition), 14 OpenMP threads
# each (28 cores/node).
# NSHOTS is taken from $SLURM_NTASKS, so one shot per rank and the shot loop
#   for (ishot = mpi_rank; ishot < nshots; ishot += mpi_size)
# runs exactly once per rank.  Change --nodes / --ntasks-per-node and the shot
# spacing re-derives itself; nothing else needs editing -- NSHOTS is never
# hardcoded, so the job stays load-balanced at any node count.
#
# Why the lag path costs memory and not communication
# ---------------------------------------------------
# The lag accumulators are 6 complex fields x lag_nfreq x nx*nz and live
# ENTIRELY inside one rank -- the cone filter is applied per shot, before the
# reduction, so the only MPI traffic is the usual Allreduce of the tau=0
# slices.  Cost is therefore per-rank memory, independent of the shot count.
# On this 401x201 grid at lag_nfreq=31 the six accumulators are 229 MB and
# lagGradientInit reports 348 MB/rank once the reconstruction cubes and FFT
# workspace are counted.  With 2 ranks/node that is ~0.7 GB/node, which is why
# --mem=0 + --exclusive is plenty.  The REPORT section echoes the figure the
# code itself prints, so a grid or lag_nfreq change is self-documenting.
#
# TRAP: every path below must be on SHARED storage
# ------------------------------------------------
# Unlike the single-node GPU test, this job spans nodes.  The observed data,
# the residuals and the checkpoints are passed between ranks THROUGH THE
# FILESYSTEM, so a node-local /tmp work directory silently gives each node a
# different view: ranks find no obs files, or read someone else's checkpoints.
# WORK therefore sits under /rcp3, which every node mounts.
#
# lag_tomo=1 constrains the inversion (fwi_inversion hard-errors otherwise)
# -------------------------------------------------------------------------
#   ischeme=3            elastic only
#   param=2              velocity parameterisation
#   active_params=vp,vs  rho MUST be frozen -- there is no g_rho lag cube
#   precond=0            no preconditioner
#   algorithm<=1         SD or L-BFGS; the Newton variants are rejected
#

set -e
set -o pipefail

# =========================================================
# Environment -- MIO modules, matching ROOT/Make_include
# =========================================================
module purge
module load compilers/gcc/13
module load libs/cuda/12.5
module load libs/mkl/2024.2
module load mpi/openmpi/gcc-cuda/4.1.6

ROOT=/rcp3/software/codes/OpenSource_MIO
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}

# One rank per socket, threads = the cores SLURM gave that rank.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-14}

NPROCS=${SLURM_NTASKS:-22}

# Absolute path: under SLURM $(dirname "$0") resolves to the spool copy of the
# script, not to the directory it was submitted from.
WORK=/rcp3/software/codes/OpenSource_MIO/fdelfwi/demo/inversion/lag_tomo_mpi_${SLURM_JOB_ID:-local}
mkdir -p ${WORK}
cd ${WORK}

# =========================================================
# Geometry and the medium -- identical to the validated demo
# =========================================================
DX=5; SIZEX=2000; SIZEZ=1000
NX=401; NZ=201

VP0=2000; VS0=1150; RO0=2000
ZREFL=600; RO1=2200          # density reflector: in BOTH models, never moves
FAST=1.08                    # the error: background is 8% fast

VP_BG=$(python3 -c "print(${VP0}*${FAST})")
VS_BG=$(python3 -c "print(${VS0}*${FAST})")

TMOD=1.5; DTRCV=0.004
NTAPER=100
GRAD_TAPER=20

# One shot per rank.
NSHOTS=${NPROCS}
XSRC_START=110
if [ ${NSHOTS} -gt 1 ]; then
    # Keep the spacing an exact multiple of dx so every source lands on a grid
    # node; dxshot is in metres but fdelmodc places it by index.
    XSRC_STEP=$(python3 -c "print(int(((${SIZEX}-2*${XSRC_START})/(${NSHOTS}-1))//${DX}*${DX}))")
else
    XSRC_STEP=0
fi
XSRC_LAST=$((XSRC_START + (NSHOTS-1)*XSRC_STEP))
ZSRC=10
XRCV1=100; XRCV2=1900; ZRCV=100; DXRCV=20

# =========================================================
# Lag axis, frequency set and cone
# =========================================================
# DENSITY REQUIREMENT, enforced by lagGradientInit: the reconstruction
# replicates with period 1/df in the relative lag and conj(A)B correlates over
# the whole record, so 1/df >= T.  T=1.5 s, band [2,18] Hz ->
#     lag_nfreq >= (18-2)*1.5 + 1 = 25.
# 31 gives df=0.533 Hz, 1/df=1.875 s > T.  Going coarser is not a trade-off,
# it is an abort with the minimum printed.
LAG_NFREQ=31; LAG_FMIN=2; LAG_FMAX=18

# 31 lags x 5 ms is |tau| <= 75 ms -- long enough for the cone to measure a
# slope, and well inside the |tau| < 1/(4 df) = 469 ms aliasing limit.
LAG_NT=31; LAG_DT=0.005

# |p| = cos(theta)/v with 2*theta the FULL opening angle.  At the background
# Vp of 2160 m/s:  0.20 ms/m -> 129 deg passes in full, 0.30 -> 99 deg is
# rejected.  Small slope = wide angle = transmission = the tomographic kernel.
S_PASS=0.20; S_STOP=0.30

NITER=10
NLS_MAX=5

echo "========================================================="
echo " LAG-TOMOGRAPHIC MPI INVERSION  (lag_tomo=1)"
echo " job ${SLURM_JOB_ID:-<interactive>} on $(hostname)"
echo " nodes=${SLURM_JOB_NUM_NODES:-?}  ranks=${NPROCS}  threads/rank=${OMP_NUM_THREADS}"
echo " date $(date)"
echo "---------------------------------------------------------"
echo "   true       Vp=${VP0}  Vs=${VS0}   rho ${RO0}/${RO1} across z=${ZREFL} m"
echo "   initial    Vp=${VP_BG}  Vs=${VS_BG}  (${FAST}x fast), same rho"
echo "   grid       ${NX} x ${NZ}  (dx=${DX} m)"
echo "   shots      ${NSHOTS} from x=${XSRC_START} to ${XSRC_LAST} m, step ${XSRC_STEP} m"
echo "   lag        ${LAG_NFREQ} freqs ${LAG_FMIN}-${LAG_FMAX} Hz, ${LAG_NT} lags x ${LAG_DT} s"
echo "   cone       pass |p|<=${S_PASS}  stop |p|>=${S_STOP} ms/m"
echo "   L-BFGS     niter=${NITER}  nls_max=${NLS_MAX}"
echo "   work       ${WORK}"
echo "========================================================="

if [ ! -x ${FDELFWI}/fwi_mpi_inversion ]; then
    echo "ERROR: ${FDELFWI}/fwi_mpi_inversion not found."
    echo "       Build it on MIO first:  cd ${FDELFWI} && make mpi_inversion"
    exit 1
fi

# =========================================================
echo ""
echo "--- Step 1: models ---"

# TRUE: correct velocity, density reflector.
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=def poly=0 x=0,${SIZEX} z=${ZREFL},${ZREFL} \
    cp=${VP0} cs=${VS0} ro=${RO1} \
    orig=0,0 file_base=model_true.su verbose=0

# INITIAL: the same medium uniformly fast, and the SAME density.  The error is
# entirely in velocity; the reflector exists in both models and never moves,
# so the residual is a traveltime error on a reflection that is present either
# way -- precisely what the tomographic kernel is for.
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP_BG} cs0=${VS_BG} ro0=${RO0} \
    intt=def poly=0 x=0,${SIZEX} z=${ZREFL},${ZREFL} \
    cp=${VP_BG} cs=${VS_BG} ro=${RO1} \
    orig=0,0 file_base=model_bg.su verbose=0

cmp model_true_ro.su model_bg_ro.su && \
    echo "  density identical in both models (the reflector does not move)"
echo "  Vp ${VP0} -> ${VP_BG},  Vs ${VS0} -> ${VS_BG}"

# =========================================================
echo ""
echo "--- Step 2: wavelet ---"
# fp=12 Hz: the dispersion guard tests the MEASURED wavelet fmax, about
# 1.22*fp + 1.46 = 16.1 Hz, and cmin comes from the TRUE model (Vs=1150)
# because the background is the FAST one.  1150/(16.1*5) = 14 points per
# shortest wavelength, comfortably inside the guard.
makewave fp=12 dt=0.001 nt=1024 fmax=20 file_out=wave.su t0=0.10 verbose=0
echo "  fp=12 Hz, fmax=20 Hz, dt=1 ms"

# =========================================================
echo ""
echo "--- Step 3: observed data from the true model (${NSHOTS} shots) ---"
# fdelfwi_fwd, NOT fdelmodc: the two have incompatible taper-boundary
# implementations and mixing them leaves a ~1% direct-wave residual that has
# nothing to do with the model error being inverted for.
#
# ONE call, not a loop and not srun: fdelfwi_fwd runs the whole shot loop
# itself and passes fileno=ishot to fdfwimodc, so writeRec already produces
# the zero-padded obs_000_, obs_001_, ... that fwi_inversion expects.  There
# is no fileno= parameter to drive from outside, so parallel instances would
# all write obs_000_.  This step is serial on purpose; it is minutes, while
# the inversion below is the part worth 24 ranks.
time ${FDELFWI}/fdelfwi_fwd \
    file_cp=model_true_cp.su file_cs=model_true_cs.su \
    file_den=model_true_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
    xsrc=${XSRC_START} zsrc=${ZSRC} nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 > fwd_obs.log 2>&1

for ((ishot=0; ishot<NSHOTS; ishot++)); do
    SHOTID=$(printf "%03d" ${ishot})
    # fdelmodc's rec_type_p is NOT an elastic hydrophone and fdelfwi_fwd does
    # not write _rp at all, so the OBSERVED pressure is built here.
    suop2 obs_${SHOTID}_rtzz.su obs_${SHOTID}_rtxx.su op=sum | \
        sugain scale=0.5 > obs_${SHOTID}_rp.su
done
echo "  ${NSHOTS} shots written, hydrophone P = 0.5*(Txx+Tzz) built"

# =========================================================
echo ""
echo "========================================================="
echo " Step 4: MPI L-BFGS inversion on the cone-filtered gradient"
echo "========================================================="

srun -n ${NPROCS} --cpus-per-task=${OMP_NUM_THREADS} \
    ${FDELFWI}/fwi_mpi_inversion \
    file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
    xsrc=${XSRC_START} zsrc=${ZSRC} nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=100 aper_taper=0 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=50 chk_base=chk \
    param=2 active_params=vp,vs \
    scaling=0 precond=0 smooth_grad=0 \
    data_weight=0 misfit=0 comp=_rp \
    vp_min=1500 vp_max=3000 vs_min=800 vs_max=1800 \
    rho_min=1800 rho_max=2400 \
    file_obs=obs file_grad=gradient \
    algorithm=1 lbfgs_mem=5 niter=${NITER} nls_max=${NLS_MAX} \
    conv=1e-6 write_iter=1 \
    lag_tomo=1 \
    lag_nfreq=${LAG_NFREQ} lag_fmin=${LAG_FMIN} lag_fmax=${LAG_FMAX} \
    lag_nt=${LAG_NT} lag_dt=${LAG_DT} \
    lag_s_pass=${S_PASS} lag_s_stop=${S_STOP} \
    2>&1 | tee inversion.log

# =========================================================
echo ""
echo "========================= REPORT ========================="
echo "--- lag accumulator footprint (as reported by the code) ---"
grep -m4 -E "lagGradientInit:" inversion.log || echo "  (not found)"

echo ""
echo "--- how much the cone kept, first and last reported shot ---"
grep -E "kept .* of the spectral energy" inversion.log | head -2 || echo "  (not found)"
grep -E "kept .* of the spectral energy" inversion.log | tail -2 || true

echo ""
echo "--- L-BFGS convergence ---"
cat iterate_LB.dat 2>/dev/null || echo "  (iterate_LB.dat not found)"

echo ""
echo "--- model error against the true model ---"
# The whole point of a tomographic update: the 8% background error should come
# down.  Reported, not asserted -- how far it gets in ${NITER} iterations
# depends on the geometry.
python3 - <<PYEOF
import glob, numpy as np

def readsu(fn, ns):
    raw = np.fromfile(fn, dtype=np.uint8)
    rec = 240 + 4*ns
    n   = raw.size // rec
    out = np.empty((n, ns), dtype=np.float32)
    for i in range(n):
        out[i] = raw[i*rec+240:(i+1)*rec].view(np.float32)
    return out.astype(np.float64)

ns = ${NZ}
for par, true_f, init_f in (('vp', 'model_true_cp.su', 'model_bg_cp.su'),
                            ('vs', 'model_true_cs.su', 'model_bg_cs.su')):
    t = readsu(true_f, ns)
    i = readsu(init_f, ns)
    # write_iteration_model() names these model_iter%03d_{vp,vs}.su, and the
    # final model is written the same way with the last iteration number, so
    # the highest-numbered file IS the result.
    finals = sorted(glob.glob('model_iter*_%s.su' % par))
    e0 = np.linalg.norm(i - t) / np.linalg.norm(t)
    line = '  %s  initial rel. error = %6.2f %%' % (par, 100*e0)
    if finals:
        f = readsu(finals[-1], ns)
        if f.shape == t.shape:
            e1 = np.linalg.norm(f - t) / np.linalg.norm(t)
            line += '   final = %6.2f %%   (%s)' % (100*e1, finals[-1])
    else:
        line += '   final = n/a (no per-iteration model file found)'
    print(line)
PYEOF

echo ""
echo "  output in: ${WORK}"
echo "    inversion.log      full verbose=2 linesearch diagnostics"
echo "    iterate_LB.dat     L-BFGS convergence history"
echo "    gradient_*.su      per-iteration gradients and models (write_iter=1)"
echo "=========================================================="

# Checkpoints are the bulk of the footprint and are worthless after the run.
rm -rf chk_*.bin fwi_rank*
