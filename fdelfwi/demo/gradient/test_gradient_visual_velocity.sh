#!/bin/bash
#
# test_gradient_visual_velocity.sh - Gradient visualization (velocity parameterization).
#
# Model: 2000x800m, homogeneous background (Vp=2000, Vs=1150, rho=2000)
#        with 3 Gaussian anomalies at z=400m:
#        x=1000 (Vp=2200), x=500 (Vs=1250), x=1500 (rho=2200)
#
# Acquisition: 10 shots (x=200-1800), 181 receivers (x=100-1900) at z=20m
# Scaling: range normalization (scaling=1, m* = m/Δm)
# Preconditioner: Plessix & Mulder (2004) — sqrt(Ws) × asinh receiver approx
#                 with per-shot erf source taper and Métivier norm preservation
#
# Output:
#   grad_vp_raw.su, grad_vs_raw.su, grad_rho_raw.su          (raw gradient)
#   grad_vp_scaled.su, grad_vs_scaled.su, grad_rho_scaled.su  (g* = g·Δm)
#   grad_vp_preco.su, grad_vs_preco.su, grad_rho_preco.su     (preconditioned)
#   precond_vp.su, precond_vs.su, precond_rho.su               (P diagonal)
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

WORKDIR=gradient_visual_velocity
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " GRADIENT VISUALIZATION TEST (Velocity param=2)"
echo "========================================================="

# =========================================================
# Model parameters
# =========================================================
DX=5
SIZEX=2000
SIZEZ=800
VP0=2000
VS0=1150
RO0=2000

# =========================================================
# Acquisition
# =========================================================
NSHOTS=10
XSRC_START=200
XSRC_END=1800
XSRC_STEP=$(( (XSRC_END - XSRC_START) / (NSHOTS - 1) ))

# =========================================================
# Preconditioner parameters
# =========================================================
PRECOND=1                # 1=Yang/Shin (K_i²), 2=Plessix&Mulder (sqrt(Ws)*asinh)
PRECOND_EPS=0.01          # θ damping (Métivier eq 39)
PRECOND_BLEND=0           # depth taper zone in meters (0=disabled)
PRECOND_ALPHA=0.30        # depth taper steepness
SRT_RADIUS=0        # per-shot source taper radius in meters (0=disabled)
SRT_FILTSIZE=0         # hard-zero half-width in grid points (zone = (2n+1)²)

# Scaling bounds (for range normalization) 
VP_MIN=1500;  VP_MAX=4500
VS_MIN=500;   VS_MAX=2500
RHO_MIN=1500; RHO_MAX=2500

# Data component: _rvx, _rvz, or _rp
COMP=_rp

# =========================================================
# STEP 1: Create models
# =========================================================
echo "--- Step 1: Creating models ---"

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=model_bg.su verbose=0

makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    intt=diffr dtype=2 x=1000 z=400 cp=2200 cs=${VS0} ro=${RO0} var=20 \
    intt=diffr dtype=2 x=500 z=400 cp=${VP0} cs=1250 ro=${RO0} var=20 \
    intt=diffr dtype=2 x=1500 z=400 cp=${VP0} cs=${VS0} ro=2200 var=20 \
    orig=0,0 file_base=model_true.su verbose=0

echo "  Background: Vp=${VP0}, Vs=${VS0}, rho=${RO0}"
echo "  Anomalies: x=1000(Vp=2200), x=500(Vs=1250), x=1500(rho=2200)"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=15 fmin=2 fmax=20 dt=0.001 nt=1024 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=15 Hz, fmin=2 Hz, fmax=20 Hz"

# =========================================================
# STEP 3: Generate observed data
# =========================================================
echo "--- Step 3: Generating observed data (${NSHOTS} shots) ---"

for ((ishot=0; ishot<NSHOTS; ishot++)); do
    XSRC=$((XSRC_START + ishot * XSRC_STEP))

    fdelmodc \
        file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs_${ishot} \
        ischeme=3 iorder=4 src_type=1 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=0.004 tmod=1.5 verbose=0 \
        xrcv1=100 xrcv2=1900 zrcv1=20 zrcv2=20 dxrcv=10 \
        xsrc=${XSRC} zsrc=10 ntaper=100 \
        left=4 right=4 top=4 bottom=4

    SHOTID=$(printf "%03d" ${ishot})
    mv obs_${ishot}_rvx.su obs_${SHOTID}_rvx.su
    mv obs_${ishot}_rvz.su obs_${SHOTID}_rvz.su
    suop2 obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su op=sum | sugain scale=0.5 > obs_${SHOTID}_rp.su
    rm -f obs_${ishot}_rtzz.su obs_${ishot}_rtxx.su

    echo "  Shot $((ishot+1))/${NSHOTS}: xsrc=${XSRC}"
done

# =========================================================
# STEP 4: Run gradient computation
# =========================================================
echo ""
echo "========================================================="
echo " Running gradient computation (param=2, velocity)"
echo "   scaling=1 (range), theta=${PRECOND_EPS}, srt_radius=${SRT_RADIUS}m"
echo "   comp=${COMP}, ${NSHOTS} shots"
echo "========================================================="

${FDELFWI}/test_gradient_visual \
    file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.004 tmod=1.5 verbose=1 \
    xrcv1=100 xrcv2=1900 zrcv1=20 zrcv2=20 dxrcv=10 \
    xsrc=${XSRC_START} zsrc=10 nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=100 grad_taper=0 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=100 chk_base=chk_gv \
    param=2 scaling=1 precond=${PRECOND} \
    precond_eps=${PRECOND_EPS} precond_blend=${PRECOND_BLEND} precond_alpha=${PRECOND_ALPHA} \
    srt_radius=${SRT_RADIUS} srt_filtsize=${SRT_FILTSIZE} \
    vp_min=${VP_MIN} vp_max=${VP_MAX} \
    vs_min=${VS_MIN} vs_max=${VS_MAX} \
    rho_min=${RHO_MIN} rho_max=${RHO_MAX} \
    comp=${COMP} \
    file_obs=obs out_base=grad

# =========================================================
# Cleanup
# =========================================================
rm -f chk_gv_*.bin syn_*.su res_*.su

echo ""
echo "========================================================="
echo " Done. Output files in: $(pwd)"
echo "   Raw gradient:    grad_vp_raw.su    grad_vs_raw.su    grad_rho_raw.su"
echo "   Scaled gradient: grad_vp_scaled.su grad_vs_scaled.su grad_rho_scaled.su"
echo "   Preconditioned:  grad_vp_preco.su  grad_vs_preco.su  grad_rho_preco.su"
echo "   Preconditioner:  precond_vp.su     precond_vs.su     precond_rho.su"
echo "========================================================="
