#!/bin/bash
#SBATCH --job-name=lag_refl_xcheck
#SBATCH --partition=geop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=lag_refl_xcheck_%j.log
#
# test_lag_reflector_mio.sh -- run the single-shot lag_reflector experiment on
#                              MIO and check it against the Seislab10 result
#
#   sbatch test_lag_reflector_mio.sh
#
# WHY THIS RUNS BEFORE THE MULTI-SHOT MPI JOB
# -------------------------------------------
# test_lag_tomo_mpi_mio.sh changes two things at once: it moves to a new
# machine AND it goes to 22 ranks on 11 nodes.  If its numbers look odd there
# is no way to tell which change caused it.  This job changes only the
# MACHINE: one shot, one rank, no MPI, the same script and the same settings
# that already ran on Seislab10, with the reference output still on disk.  A
# clean result here means any later surprise belongs to the parallelisation,
# not to the port.
#
# It is also the strongest available test of the FFT backend.  lag_nfreq=171
# and a 101-lag cube exercise the running DFTs, the per-lag reconstruction and
# the FFT round trip far harder than the 31/31 inversion demo does, and the
# MKL-vs-Mayer normalisation question in lagConeFilter is a build-time
# property -- it cannot be settled on Seislab10, only on the machine that
# links the library.
#
# HOW THE COMPARISON IS POSSIBLE AT ALL
# -------------------------------------
# /rcp3 is mounted on both machines, so the Seislab10 work directory is
# readable from here and the comparison is against the ACTUAL reference
# samples -- not against numbers transcribed into a header, which is the usual
# way a reproducibility check rots.  Nothing is written to the reference tree;
# it is opened read-only by lag_refl_crosscheck.py.
#
# Not bit-identical, by construction: both trees use -O3 -march=native
# -ffast-math, so FP re-association differs between the two architectures.
# See the header of lag_refl_crosscheck.py.
#
# Cost, measured on the Seislab10 run:
#   accumulators 8777.2 MB  (171 freqs x 834 x 501, independent of lag count)
#   runA dense gradient  675 s      runB dense gradient  1767 s
#   plus 4 forward runs.  --mem=48G leaves room for the checkpoints and the
#   two 189 MB lag cubes; 3 h is roughly double the measured total.

set -e
set -o pipefail

# =========================================================
# Environment -- MIO modules, matching OpenSource_MIO/Make_include
# =========================================================
module purge
module load compilers/gcc/13
module load libs/cuda/12.5
module load libs/mkl/2024.2

MIO_ROOT=/rcp3/software/codes/OpenSource_MIO
SL10_ROOT=/rcp3/software/codes/OpenSource_SL10

# Drive the MIO binaries.  test_lag_reflector.sh honours FDELFWI_ROOT; without
# it the script's default would run the SEISLAB10 binaries and the check would
# compare a build against itself and trivially pass.
export FDELFWI_ROOT=${MIO_ROOT}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

HERE=${MIO_ROOT}/fdelfwi/demo/gradient
REF=${SL10_ROOT}/fdelfwi/demo/gradient/lag_reflector
NEW=${HERE}/lag_reflector

echo "=========================================================="
echo " lag_reflector CROSS-MACHINE CHECK"
echo " job ${SLURM_JOB_ID:-<interactive>} on $(hostname)"
echo " $(date)"
echo "----------------------------------------------------------"
echo " binaries under test : ${MIO_ROOT}"
echo " reference run       : ${REF}"
echo " threads             : ${OMP_NUM_THREADS}"
echo "=========================================================="

# ---- pre-flight, so a 45-minute job does not fail at the last step --------
fail=0
for b in ${MIO_ROOT}/fdelfwi/test_gradient_visual \
         ${MIO_ROOT}/fdelfwi/fdelfwi_fwd; do
    if [ ! -x "$b" ]; then
        echo "MISSING BINARY: $b"
        fail=1
    fi
done
[ ${fail} -eq 0 ] || {
    echo ""
    echo "Build them on MIO first:"
    echo "  cd ${MIO_ROOT}/fdelfwi && make test_gradient_visual fdelfwi_fwd"
    exit 1
}

for f in ${HERE}/test_lag_reflector.sh ${HERE}/lag_refl_check.py \
         ${HERE}/lag_refl_mute.py ${HERE}/lag_refl_report.py \
         ${HERE}/lag_refl_crosscheck.py \
         ${MIO_ROOT}/fdelfwi/tests/kh_misfit/su_io.py; do
    [ -f "$f" ] || { echo "MISSING FILE: $f"; fail=1; }
done
[ ${fail} -eq 0 ] || {
    echo ""
    echo "The MIO checkout is behind.  Pull first:"
    echo "  cd ${MIO_ROOT} && git pull --ff-only origin master"
    exit 1
}

if [ ! -d "${REF}" ]; then
    echo "NO REFERENCE at ${REF}"
    echo "Re-create it on Seislab10 with:"
    echo "  ${SL10_ROOT}/fdelfwi/demo/gradient/test_lag_reflector.sh all"
    exit 1
fi
echo "pre-flight OK: binaries, scripts and reference all present."

# ---- the FFT backend question, settled on THIS build ---------------------
# Cheap, and it is the single most likely thing to differ between two builds
# of the lag path, so it runs before the expensive stages.
echo ""
echo "--- lagConeSelfTest on the MIO build ---"
if [ -x ${MIO_ROOT}/fdelfwi/test_lag_cone ]; then
    ${MIO_ROOT}/fdelfwi/test_lag_cone selftest=1
else
    echo "  test_lag_cone not built -- SKIPPED."
    echo "  Build it:  cd ${MIO_ROOT}/fdelfwi && make test_lag_cone"
    echo "  This is the FFT-normalisation check; do not skip it permanently."
fi

# ---- the experiment ------------------------------------------------------
echo ""
echo "--- running test_lag_reflector.sh all (MIO binaries) ---"
cd ${HERE}
time ./test_lag_reflector.sh all

# ---- the comparison ------------------------------------------------------
echo ""
echo "=========================================================="
echo " CROSS-MACHINE COMPARISON  (MIO vs Seislab10)"
echo "=========================================================="
python3 ${HERE}/lag_refl_crosscheck.py --ref=${REF} --new=${NEW}
RC=$?

echo ""
if [ ${RC} -eq 0 ]; then
    echo "MIO reproduces Seislab10 on the single-shot lag_reflector case."
    echo "Next:  sbatch ${MIO_ROOT}/fdelfwi/demo/inversion/test_lag_tomo_mpi_mio.sh"
else
    echo "Differences outside tolerance -- read the group breakdown above."
    echo "Do NOT start the multi-node job until this is understood: it would"
    echo "add 22 ranks on top of an unexplained single-shot discrepancy."
fi
exit ${RC}
