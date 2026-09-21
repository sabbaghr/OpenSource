#!/bin/bash
#
# test_ball_template_acoustic.sh - Acoustic FWI ball template test.
#
# Reproduces the TOY2DAC V2.6 ball template in time domain:
#   - Grid: 101x101, h=20m → 2000×2000m
#   - True: Vp=1500 + 300 m/s Gaussian anomaly at center (z=920,x=1000)
#   - Initial: Vp=1500 homogeneous
#   - rho=1000 (constant), Q=1000 (no attenuation)
#   - PLBFGS, 10 iterations, lbfgs_mem=5, precond_eps=1e-4
#   - Bounds: [1000, 4000] m/s
#   - TOY2DAC scaling (scaling_mode=1)
#
# Time-domain differences from TOY2DAC (frequency-domain, 3 Hz):
#   - Uses Ricker wavelet f0=3 Hz (broadband, then bandpass filtered)
#   - 20 sources on top boundary (TOY2DAC uses 196 circumferential)
#   - Absorbing boundaries (PML) instead of complex-damped Helmholtz
#
# Comparison: cost reduction curves and final Vp model.
#
# Usage: bash test_ball_template_acoustic.sh
#        sbatch test_ball_template_acoustic.sh  (SLURM)
#
#SBATCH --job-name=ball_acoustic
#SBATCH --partition=rcp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=ball_acoustic_%j.out
#SBATCH --error=ball_acoustic_%j.err

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=$(dirname "$(readlink -f "$0")")/../..
CWP=/rcp3/software/codes/cwp/bin
TOY2DAC=/rcp3/software/codes/TOY2DAC_V2.6_2019_05_24/run_ball_template

export PATH=${BIN}:${CWP}:${PATH}

# Working directory
WORKDIR=ball_template_acoustic
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "========================================================="
echo " ACOUSTIC FWI BALL TEMPLATE TEST (fdelfwi vs TOY2DAC)"
echo "========================================================="
echo ""

# =========================================================
# Parameters (matching TOY2DAC)
# =========================================================
NX=101
NZ=101
DX=20
SIZEX=$(( (NX-1) * DX ))  # 2000m
SIZEZ=$(( (NZ-1) * DX ))  # 2000m
VP_BG=1500
RHO_BG=1000
NSHOTS=20
NITER=10
F0=3       # Ricker peak frequency (matching TOY2DAC 3 Hz)

echo "Grid: ${NX}x${NZ}, dx=dz=${DX}m, domain=${SIZEX}x${SIZEZ}m"
echo "Background: Vp=${VP_BG} m/s, rho=${RHO_BG} kg/m3"
echo "Anomaly: +300 m/s Gaussian at center"
echo "Sources: ${NSHOTS} on top, Ricker f0=${F0} Hz"
echo "Optimizer: PLBFGS, ${NITER} iterations"
echo ""

# =========================================================
# Step 1: Convert TOY2DAC models to SU format
# =========================================================
echo "--- Step 1: Converting TOY2DAC models to SU ---"

python3 << 'PYEOF'
import numpy as np
import struct, sys

def write_su(filename, data, nz, nx, dz, dx, z0=0, x0=0):
    """Write 2D array to SU file (one trace per column)."""
    with open(filename, 'wb') as f:
        for ix in range(nx):
            # SU header (240 bytes)
            hdr = bytearray(240)
            struct.pack_into('<i', hdr, 0, ix+1)      # tracl
            struct.pack_into('<i', hdr, 4, ix+1)      # tracr
            struct.pack_into('<i', hdr, 72, int((x0 + ix*dx)*1000))  # sx (mm)
            struct.pack_into('<h', hdr, 114, int(nz))  # ns
            struct.pack_into('<h', hdr, 116, int(dz*1e6))  # dt (us)
            struct.pack_into('<i', hdr, 180, int((x0 + ix*dx)*1000))  # gx (mm)
            struct.pack_into('<h', hdr, 70, -3)   # scalco = -1000 (mm)
            struct.pack_into('<h', hdr, 202, 1)   # trid = 1
            f.write(hdr)
            f.write(data[:, ix].astype(np.float32).tobytes())

nz, nx = 101, 101
dx = 20.0

# Read TOY2DAC models (column-major Fortran: n1=nz fast, n2=nx slow)
vp_ball = np.fromfile(f'{sys.argv[1]}/vp_ball', dtype=np.float32).reshape(nx, nz).T
vp_homo = np.fromfile(f'{sys.argv[1]}/vp_homogeneous', dtype=np.float32).reshape(nx, nz).T
rho     = np.fromfile(f'{sys.argv[1]}/rho', dtype=np.float32).reshape(nx, nz).T
vp_final = np.fromfile(f'{sys.argv[1]}/param_vp_final', dtype=np.float32).reshape(nx, nz).T

# Write SU files
write_su('model_true_cp.su', vp_ball, nz, nx, dx, dx)
write_su('model_init_cp.su', vp_homo, nz, nx, dx, dx)
write_su('model_true_ro.su', rho, nz, nx, dx, dx)
write_su('model_init_ro.su', rho, nz, nx, dx, dx)
write_su('toy2dac_vp_final.su', vp_final, nz, nx, dx, dx)

# Also create Cs=0 files for fdelmodc
cs = np.zeros_like(vp_ball)
write_su('model_true_cs.su', cs, nz, nx, dx, dx)
write_su('model_init_cs.su', cs, nz, nx, dx, dx)

print(f'  vp_true:  [{vp_ball.min():.0f}, {vp_ball.max():.0f}] m/s')
print(f'  vp_init:  [{vp_homo.min():.0f}, {vp_homo.max():.0f}] m/s')
print(f'  rho:      [{rho.min():.0f}, {rho.max():.0f}] kg/m3')
print(f'  vp_final (TOY2DAC): [{vp_final.min():.1f}, {vp_final.max():.1f}] m/s')
print(f'  TOY2DAC RMS error: {np.sqrt(np.mean((vp_final-vp_ball)**2)):.2f} m/s')
PYEOF
echo "${TOY2DAC}" | xargs -I{} python3 -c "
import numpy as np, struct, sys
def write_su(fn, data, nz, nx, dz, dx):
    with open(fn, 'wb') as f:
        for ix in range(nx):
            hdr = bytearray(240)
            struct.pack_into('<i', hdr, 0, ix+1)
            struct.pack_into('<i', hdr, 4, ix+1)
            struct.pack_into('<h', hdr, 114, int(nz))
            struct.pack_into('<h', hdr, 116, int(dz*1e6))
            f.write(hdr)
            f.write(data[:, ix].astype(np.float32).tobytes())
nz, nx, dx = 101, 101, 20.0
vp_b = np.fromfile('{}' + '/vp_ball', dtype=np.float32).reshape(nx,nz).T
vp_h = np.fromfile('{}' + '/vp_homogeneous', dtype=np.float32).reshape(nx,nz).T
rho  = np.fromfile('{}' + '/rho', dtype=np.float32).reshape(nx,nz).T
vp_f = np.fromfile('{}' + '/param_vp_final', dtype=np.float32).reshape(nx,nz).T
write_su('model_true_cp.su', vp_b, nz, nx, dx, dx)
write_su('model_init_cp.su', vp_h, nz, nx, dx, dx)
write_su('model_true_ro.su', rho, nz, nx, dx, dx)
write_su('model_init_ro.su', rho, nz, nx, dx, dx)
write_su('toy2dac_vp_final.su', vp_f, nz, nx, dx, dx)
cs = np.zeros_like(vp_b)
write_su('model_true_cs.su', cs, nz, nx, dx, dx)
write_su('model_init_cs.su', cs, nz, nx, dx, dx)
print(f'  vp_true:  [{vp_b.min():.0f}, {vp_b.max():.0f}] m/s')
print(f'  vp_init:  [{vp_h.min():.0f}, {vp_h.max():.0f}] m/s')
print(f'  rho:      [{rho.min():.0f}, {rho.max():.0f}] kg/m3')
print(f'  TOY2DAC final: [{vp_f.min():.1f}, {vp_f.max():.1f}], RMS err={np.sqrt(np.mean((vp_f-vp_b)**2)):.2f} m/s')
"

# =========================================================
# Step 2: Create wavelet (Ricker f0=3 Hz)
# =========================================================
echo ""
echo "--- Step 2: Creating wavelet ---"

# dt stability: Vp_max=1800, dx=20 -> dt < dx/(Vp*1.65) ~ 6.7ms
# Use dt=2ms for safety
DT=0.002
TMOD=4.0
NT=$(echo "$TMOD / $DT" | bc)

makewave fp=${F0} dt=${DT} nt=${NT} file_out=wave.su t0=0.50 verbose=0

echo "  Ricker f0=${F0} Hz, dt=${DT}s, tmod=${TMOD}s, nt=${NT}"

# =========================================================
# Step 3: Generate observed data (true model)
# =========================================================
echo ""
echo "--- Step 3: Generating observed data (${NSHOTS} shots) ---"

# Source positions: evenly spaced on top boundary
XSRC_FIRST=100
XSRC_LAST=1900
XSRC_STEP=$(( (XSRC_LAST - XSRC_FIRST) / (NSHOTS - 1) ))
ZSRC=20

# Receiver positions: all around the boundary at every 40m
# Top: z=20, x=20..1980 (50 receivers)
# Bottom: z=1980, x=20..1980 (50)
# Left: x=20, z=60..1940 (48)
# Right: x=1980, z=60..1940 (48)
# Total: 196 receivers (same as TOY2DAC)

for (( i=0; i<NSHOTS; i++ )); do
    XSRC=$(( XSRC_FIRST + i * XSRC_STEP ))
    FILENO=$(printf "%03d" $i)

    fdelmodc \
        file_cp=model_true_cp.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs_${FILENO} \
        ischeme=1 iorder=4 \
        src_type=1 src_orient=1 \
        rec_type_p=1 \
        dtrcv=0.004 tmod=${TMOD} verbose=0 \
        xrcv1=20 xrcv2=1980 zrcv1=20 zrcv2=20 dxrcv=40 \
        xsrc=${XSRC} zsrc=${ZSRC} \
        npml=20 \
        left=2 right=2 top=2 bottom=2 2>/dev/null &

    # Throttle parallel jobs
    if (( (i+1) % ${OMP_NUM_THREADS} == 0 )); then wait; fi
done
wait

echo "  Generated ${NSHOTS} shot gathers (obs_NNN_rp.su)"
echo "  Source spacing: ${XSRC_STEP}m, from x=${XSRC_FIRST} to ${XSRC_LAST}m at z=${ZSRC}m"

# =========================================================
# Step 4: Run fdelfwi acoustic inversion (PLBFGS)
# =========================================================
echo ""
echo "========================================================="
echo " Running PLBFGS acoustic FWI (scaling_mode=1, shin_precond=1)"
echo "========================================================="

${FDELFWI}/fwi_inversion \
    file_cp=model_init_cp.su file_den=model_init_ro.su \
    file_src=wave.su file_rcv=syn file_snap=snap \
    file_obs=obs comp=_rp \
    ischeme=1 iorder=4 param=2 \
    src_type=1 src_orient=1 \
    rec_type_p=1 \
    dtrcv=0.004 tmod=${TMOD} verbose=2 \
    xrcv1=20 xrcv2=1980 zrcv1=20 zrcv2=20 dxrcv=40 \
    nshots=${NSHOTS} \
    npml=20 \
    left=2 right=2 top=2 bottom=2 \
    algorithm=2 niter=${NITER} lbfgs_mem=5 \
    nls_max=10 conv=1e-4 \
    scaling_mode=1 \
    shin_precond=1 shin_eps=1e-4 \
    vp_min=1000 vp_max=4000 \
    grad_taper=5 \
    active_params=vp \
    write_iter=1 \
    file_grad=grad

# =========================================================
# Step 5: Compare with TOY2DAC
# =========================================================
echo ""
echo "========================================================="
echo " COMPARISON: fdelfwi vs TOY2DAC"
echo "========================================================="

echo ""
echo "TOY2DAC convergence (iterate_PLB.dat):"
cat ${TOY2DAC}/iterate_PLB.dat | grep -E "^\s+[0-9]"

echo ""
echo "fdelfwi convergence (iterate_PLB.dat):"
if [ -f iterate_PLB.dat ]; then
    cat iterate_PLB.dat | grep -E "^\s+[0-9]"
elif [ -f iterate_LB.dat ]; then
    cat iterate_LB.dat | grep -E "^\s+[0-9]"
fi

echo ""
echo "========================================================="
echo " Done. Compare:"
echo "   - Cost curves: iterate_PLB.dat vs TOY2DAC"
echo "   - Final models: model_iter_*_vp.su vs toy2dac_vp_final.su"
echo "========================================================="
