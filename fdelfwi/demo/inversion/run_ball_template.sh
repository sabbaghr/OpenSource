#!/bin/bash
#
# run_ball_template.sh - Acoustic FWI ball template (fdelfwi equivalent of TOY2DAC)
#
# Reproduces the TOY2DAC V2.6 run_ball_template test case in time domain.
#
# TOY2DAC setup:
#   Grid:        101×101, h=20m, 2000×2000m
#   True model:  Vp=1500 + Gaussian anomaly (+300 m/s) at (z=920,x=1000)
#   Init model:  Vp=1500 homogeneous
#   rho=1000, Q=1000 (no attenuation)
#   Acquisition: 196 sources/receivers on all 4 boundaries, 40m spacing
#   Frequency:   single 3 Hz
#   Optimizer:   PLBFGS, 10 iterations, lbfgs_m=5
#   Precond:     Shin diagonal, threshold=1e-4
#   Bounds:      [1000, 4000] m/s
#   Source est:  off (cc1=1)
#   Regulariz:   none
#
# Time-domain adaptation:
#   - Ricker wavelet fp=5 Hz, fmax=14 Hz (dispersion limit for h=20m)
#   - tmod=1.5s (sufficient propagation for 2km domain at 1500 m/s)
#   - PML absorbing boundaries
#   - 20 sources on top boundary (reduced from 196 for speed)
#   - Receivers on top boundary at 40m spacing
#
# Usage:
#   bash run_ball_template.sh
#   sbatch run_ball_template.sh
#
#SBATCH --job-name=ball_fdelfwi
#SBATCH --partition=rcp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=ball_fdelfwi_%j.out
#SBATCH --error=ball_fdelfwi_%j.err

set -e

# =========================================================
# Paths
# =========================================================
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
BIN=${ROOT}/bin
CWP=/rcp3/software/codes/cwp/bin
TOY2DAC_DIR=/rcp3/software/codes/TOY2DAC_V2.6_2019_05_24/run_ball_template

export PATH=${BIN}:${CWP}:${PATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# =========================================================
# Configuration
# =========================================================
NSHOTS=20
NITER=10
ALGORITHM=1         # 1=LBFGS (no precond needed)
LBFGS_MEM=5
FP=5                # Ricker peak frequency
FMAX=14             # Dispersion limit for dx=20m, Vp_min=1500
DT=0.002            # Time step
TMOD=1.5            # Recording time
DTRCV=0.004         # Receiver sampling
GRAD_TAPER=5        # Gradient taper near sources

# Working directory
WORKDIR=$(dirname "$(readlink -f "$0")")/ball_template_acoustic
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "========================================================="
echo " ACOUSTIC FWI BALL TEMPLATE (fdelfwi)"
echo "========================================================="
echo " OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo " NSHOTS=${NSHOTS}, NITER=${NITER}, fp=${FP} Hz"
echo ""

# =========================================================
# STEP 1: Convert TOY2DAC models to SU format
# =========================================================
echo "=== Step 1: Converting TOY2DAC models to SU ==="

python3 << PYEOF
import numpy as np, subprocess, os

nz, nx = 101, 101
dx = 20.0

# Read TOY2DAC binary models (Fortran column-major: n1=nz fast, n2=nx slow)
vp_ball = np.fromfile('${TOY2DAC_DIR}/vp_ball', dtype=np.float32).reshape(nx, nz).T
vp_homo = np.fromfile('${TOY2DAC_DIR}/vp_homogeneous', dtype=np.float32).reshape(nx, nz).T
rho     = np.fromfile('${TOY2DAC_DIR}/rho', dtype=np.float32).reshape(nx, nz).T

print(f'  vp_ball:  [{vp_ball.min():.0f}, {vp_ball.max():.0f}] m/s')
print(f'  vp_homo:  [{vp_homo.min():.0f}, {vp_homo.max():.0f}] m/s')
print(f'  rho:      [{rho.min():.0f}, {rho.max():.0f}] kg/m3')

# Write as raw binary then convert with suaddhead
for name, data in [('model_true_cp', vp_ball), ('model_init_cp', vp_homo),
                    ('model_true_ro', rho), ('model_init_ro', rho)]:
    raw = name + '.raw'
    # Column-by-column: each trace = one column (nz samples)
    for ix in range(nx):
        with open(raw, 'ab' if ix > 0 else 'wb') as f:
            data[:, ix].astype(np.float32).tofile(f)
    # Convert to SU with proper headers
    cmd = (f'suaddhead < {raw} ns={nz} '
           f'| sushw key=dt,d1,d2,f1,f2,tracl,tracr '
           f'a={int(dx*1e6)},{dx},{dx},0,0,1,1 b=0,0,0,0,0,1,1 '
           f'> {name}.su')
    subprocess.run(cmd, shell=True, check=True)
    os.remove(raw)

# Read TOY2DAC final model for comparison
vp_final = np.fromfile('${TOY2DAC_DIR}/param_vp_final', dtype=np.float32).reshape(nx, nz).T
rms_err = np.sqrt(np.mean((vp_final - vp_ball)**2))
print(f'  TOY2DAC final: [{vp_final.min():.1f}, {vp_final.max():.1f}] m/s, RMS err = {rms_err:.2f} m/s')
PYEOF

echo ""

# =========================================================
# STEP 2: Create source wavelet
# =========================================================
echo "=== Step 2: Creating source wavelet ==="

NT=$(echo "${TMOD} / ${DT}" | bc)
makewave fp=${FP} fmax=${FMAX} dt=${DT} nt=${NT} file_out=wave.su t0=0.20 verbose=0

echo "  Ricker wavelet: fp=${FP} Hz, fmax=${FMAX} Hz, dt=${DT}s, nt=${NT}"
echo ""

# =========================================================
# STEP 3: Generate observed data (true model)
# =========================================================
echo "=== Step 3: Generating observed data (${NSHOTS} shots) ==="

# Source positions: evenly spaced across top boundary
XSRC_FIRST=100
XSRC_LAST=1900
XSRC_STEP=$(( (XSRC_LAST - XSRC_FIRST) / (NSHOTS - 1) ))
ZSRC=20

echo "  Sources: x=${XSRC_FIRST} to ${XSRC_LAST}, step=${XSRC_STEP}m, z=${ZSRC}m"
echo "  Receivers: x=20..1980, step=40m, z=20m"

for (( i=0; i<NSHOTS; i++ )); do
    XSRC=$(( XSRC_FIRST + i * XSRC_STEP ))
    FILENO=$(printf "%03d" $i)

    fdelmodc \
        file_cp=model_true_cp.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs_${FILENO} \
        ischeme=1 iorder=4 fmax=${FMAX} \
        src_type=1 src_orient=1 \
        rec_type_p=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
        xrcv1=20 xrcv2=1980 zrcv1=20 zrcv2=20 dxrcv=40 \
        xsrc=${XSRC} zsrc=${ZSRC} \
        npml=20 \
        left=2 right=2 top=2 bottom=2 2>/dev/null

    echo "  Shot ${i}/${NSHOTS} (xsrc=${XSRC}m) done"
done

echo ""

# =========================================================
# STEP 4: Run FWI inversion (starting from homogeneous)
# =========================================================
echo "=== Step 4: Running FWI inversion ==="
echo "  Algorithm: L-BFGS (${ALGORITHM}), niter=${NITER}, lbfgs_mem=${LBFGS_MEM}"
echo "  Scaling: TOY2DAC (scaling_mode=1)"
echo "  Bounds: [1000, 4000] m/s"
echo ""

${FDELFWI}/fwi_inversion \
    file_cp=model_init_cp.su file_den=model_init_ro.su \
    file_src=wave.su file_rcv=syn \
    file_obs=obs comp=_rp \
    ischeme=1 iorder=4 param=2 fmax=${FMAX} \
    src_type=1 src_orient=1 \
    rec_type_p=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
    xrcv1=20 xrcv2=1980 zrcv1=20 zrcv2=20 dxrcv=40 \
    nshots=${NSHOTS} \
    npml=20 \
    left=2 right=2 top=2 bottom=2 \
    algorithm=${ALGORITHM} niter=${NITER} lbfgs_mem=${LBFGS_MEM} \
    nls_max=20 conv=1e-4 \
    scaling_mode=1 \
    vp_min=1000 vp_max=4000 \
    grad_taper=${GRAD_TAPER} \
    active_params=vp \
    write_iter=1 \
    file_grad=grad

# =========================================================
# STEP 5: Summary and comparison
# =========================================================
echo ""
echo "========================================================="
echo " RESULTS"
echo "========================================================="

echo ""
echo "--- TOY2DAC convergence ---"
cat ${TOY2DAC_DIR}/iterate_PLB.dat 2>/dev/null | grep -E "^\s+[0-9]"

echo ""
echo "--- fdelfwi convergence ---"
for f in iterate_LB.dat iterate_PLB.dat; do
    if [ -f "$f" ]; then
        cat "$f"
        break
    fi
done

echo ""
echo "--- Output files ---"
echo "  Convergence:  iterate_LB.dat or iterate_PLB.dat"
echo "  Models:       model_iter_*_vp.su"
echo "  Gradient:     grad_initial_vp.su"
echo ""

# Compare final models
python3 << PYEOF2
import numpy as np

nz, nx = 101, 101

# Read TOY2DAC final
vp_true = np.fromfile('${TOY2DAC_DIR}/vp_ball', dtype=np.float32).reshape(nx, nz).T
vp_toy  = np.fromfile('${TOY2DAC_DIR}/param_vp_final', dtype=np.float32).reshape(nx, nz).T

# Read fdelfwi final (last model_iter file)
import glob, os
model_files = sorted(glob.glob('model_iter_*_vp.su'))
if model_files:
    last = model_files[-1]
    print(f'  fdelfwi final model: {last}')
    # Read SU file (skip 240-byte header per trace)
    vp_fd = np.zeros((nz, nx), dtype=np.float32)
    with open(last, 'rb') as f:
        for ix in range(nx):
            f.seek(240, 1)  # skip header
            vp_fd[:, ix] = np.frombuffer(f.read(nz*4), dtype=np.float32)

    rms_toy = np.sqrt(np.mean((vp_toy - vp_true)**2))
    rms_fd  = np.sqrt(np.mean((vp_fd - vp_true)**2))
    print(f'  TOY2DAC  RMS error: {rms_toy:.2f} m/s')
    print(f'  fdelfwi  RMS error: {rms_fd:.2f} m/s')
    print(f'  fdelfwi  Vp range:  [{vp_fd.min():.1f}, {vp_fd.max():.1f}] m/s')
else:
    print('  No model_iter files found')
PYEOF2

echo ""
echo "========================================================="
echo " Done."
echo "========================================================="
