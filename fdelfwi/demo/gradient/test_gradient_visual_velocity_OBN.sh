#!/bin/bash
#
# test_gradient_visual_velocity_OBN.sh - Gradient visualization with water layer + OBN.
#
# Same as test_gradient_visual_velocity.sh but with:
#   - Water layer 0-100m (Vp=1500, Vs=0, rho=1000)
#   - Sources at z=10m (in water, near surface)
#   - Receivers on sea floor at z=100m (OBN geometry)
#   - Free surface at top (top=1)
#   - Rock below 100m: same background + layer + 3 anomalies
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

WORKDIR=gradient_visual_velocity_OBN
mkdir -p ${WORKDIR}
cd ${WORKDIR}

export OMP_NUM_THREADS=4

echo "========================================================="
echo " GRADIENT VISUALIZATION TEST — OBN (water layer + sea floor)"
echo "   Velocity param=2"
echo "========================================================="

# =========================================================
# Model parameters
# =========================================================
DX=5
SIZEX=2000
SIZEZ=600

# Water layer
VP_WATER=1500
VS_WATER=0
RO_WATER=1000
ZSEAFLOOR=100

# Rock below sea floor
VP0=2000
VS0=1150
RO0=2000

# =========================================================
# Acquisition
# =========================================================
NSHOTS=21
XSRC_START=0
XSRC_END=2000
XSRC_STEP=$(( (XSRC_END - XSRC_START) / (NSHOTS - 1) ))
ZSRC=10
ZRCV=${ZSEAFLOOR}

# =========================================================
# Preconditioner parameters
# =========================================================
PRECOND=5                # 5=Yang block pseudo-Hessian
PRECOND_EPS=0.01          # θ damping (Métivier eq 39)
PRECOND_BLEND=0           # depth taper zone in meters (0=disabled)
PRECOND_ALPHA=0.30        # depth taper steepness
SRT_RADIUS=0              # per-shot source taper radius in meters (0=disabled)
SRT_FILTSIZE=0            # hard-zero half-width in grid points
SMOOTH_GRAD=10            # gradient smoothing half-width (grid pts, 0=off)
SMOOTH_HESS=10            # Hessian smoothing half-width (grid pts, 0=off)

# Preconditioner boost (Yang et al. 2018, eq 48)
BOOST_1=1                 # param 1 (Vp) boost
BOOST_2=1                 # param 2 (Vs) boost
BOOST_3=1                 # param 3 (rho) boost

# Scaling bounds (for range normalization)
VP_MIN=1500;  VP_MAX=4500
VS_MIN=500;   VS_MAX=2500
RHO_MIN=1500; RHO_MAX=2500

# Data component: _rvx, _rvz, or _rp
COMP=_rp

# =========================================================
# STEP 1: Create models (with water layer)
# =========================================================
echo "--- Step 1: Creating models (water layer 0-${ZSEAFLOOR}m) ---"

# Background: water + homogeneous rock + deep layer
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP_WATER} cs0=${VS_WATER} ro0=${RO_WATER} \
    intt=def poly=0 x=0,2000 z=${ZSEAFLOOR},${ZSEAFLOOR} cp=${VP0} cs=${VS0} ro=${RO0} \
    intt=def poly=0 x=0,2000 z=500,500 cp=3000 cs=1700 ro=2200 \
    orig=0,0 file_base=model_bg.su verbose=0

# True: water + rock + deep layer + 3 Gaussian anomalies
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP_WATER} cs0=${VS_WATER} ro0=${RO_WATER} \
    intt=def poly=0 x=0,2000 z=${ZSEAFLOOR},${ZSEAFLOOR} cp=${VP0} cs=${VS0} ro=${RO0} \
    intt=def poly=0 x=0,2000 z=500,500 cp=3000 cs=1700 ro=2200 \
    intt=diffr dtype=2 x=1000 z=300 cp=2200 cs=${VS0} ro=${RO0} var=20 \
    intt=diffr dtype=2 x=500  z=300 cp=${VP0} cs=1250 ro=${RO0} var=20 \
    intt=diffr dtype=2 x=1500 z=300 cp=${VP0} cs=${VS0} ro=2200 var=20 \
    orig=0,0 file_base=model_true.su verbose=0

echo "  Water:  0-${ZSEAFLOOR}m (Vp=${VP_WATER}, Vs=${VS_WATER}, rho=${RO_WATER})"
echo "  Rock:   ${ZSEAFLOOR}-500m (Vp=${VP0}, Vs=${VS0}, rho=${RO0})"
echo "  Layer:  500-${SIZEZ}m (Vp=3000, Vs=1700, rho=2200)"
echo "  Anomalies: x=1000(Vp=2200), x=500(Vs=1250), x=1500(rho=2200) at z=300m"

# =========================================================
# STEP 2: Create wavelet
# =========================================================
echo "--- Step 2: Creating wavelet ---"

makewave fp=10 fmin=2 fmax=15 dt=0.001 nt=1024 file_out=wave.su t0=0.10 verbose=0

echo "  Wavelet: fp=14 Hz, fmin=2 Hz, fmax=25 Hz"

# =========================================================
# STEP 3: Generate observed data
# =========================================================
echo "--- Step 3: Generating observed data (${NSHOTS} shots, OBN) ---"
echo "  Sources at z=${ZSRC}m (in water), Receivers at z=${ZRCV}m (sea floor)"

for ((ishot=0; ishot<NSHOTS; ishot++)); do
    XSRC=$((XSRC_START + ishot * XSRC_STEP))

    fdelmodc \
        file_cp=model_true_cp.su file_cs=model_true_cs.su file_den=model_true_ro.su \
        file_src=wave.su file_rcv=obs_${ishot} \
        ischeme=3 iorder=4 src_type=1 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=0.004 tmod=1.5 verbose=0 \
        xrcv1=100 xrcv2=1900 zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=10 \
        xsrc=${XSRC} zsrc=${ZSRC} ntaper=100 \
        left=4 right=4 top=1 bottom=4

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
echo " Running gradient computation (param=2, velocity, OBN)"
echo "   scaling=1 (range), theta=${PRECOND_EPS}"
echo "   comp=${COMP}, ${NSHOTS} shots"
echo "   Sources z=${ZSRC}m, Receivers z=${ZRCV}m (sea floor)"
echo "========================================================="

${FDELFWI}/test_gradient_visual \
    file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.004 tmod=1.5 verbose=1 \
    xrcv1=100 xrcv2=1900 zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=10 \
    xsrc=${XSRC_START} zsrc=${ZSRC} nshot=${NSHOTS} dxshot=${XSRC_STEP} \
    ntaper=100 grad_taper=0 \
    left=4 right=4 top=1 bottom=4 \
    chk_skipdt=100 chk_base=chk_gv \
    param=2 scaling=1 precond=${PRECOND} \
    precond_eps=${PRECOND_EPS} precond_blend=${PRECOND_BLEND} precond_alpha=${PRECOND_ALPHA} \
    srt_radius=${SRT_RADIUS} srt_filtsize=${SRT_FILTSIZE} \
    smooth_grad=${SMOOTH_GRAD} smooth_hess=${SMOOTH_HESS} \
    vp_min=${VP_MIN} vp_max=${VP_MAX} \
    vs_min=${VS_MIN} vs_max=${VS_MAX} \
    rho_min=${RHO_MIN} rho_max=${RHO_MAX} \
    precond_boost_1=${BOOST_1} precond_boost_2=${BOOST_2} precond_boost_3=${BOOST_3} \
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
