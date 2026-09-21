#!/bin/bash
#
# test_bandpass_filter.sh - Simple 1-shot test of bandpass filtering.
#
# 1. Create a homogeneous model
# 2. Generate 1 shot with fdelmodc (elastic, records txx+tzz)
# 3. Compute hydrophone P = 0.5*(Tzz+Txx)
# 4. Filter the rp data with sufilter at 3 bands
# 5. Display all for visual comparison
#
set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}

WORKDIR=test_bandpass
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "========================================================="
echo " BANDPASS FILTER TEST (1 shot, hydrophone)"
echo "========================================================="

# =========================================================
# Model
# =========================================================
DX=5
makemod sizex=2000 sizez=600 dx=${DX} dz=${DX} \
    cp0=2000 cs0=1150 ro0=2000 \
    intt=def poly=0 x=0,2000 z=400,400 cp=3000 cs=1700 ro=2200 \
    orig=0,0 file_base=model.su verbose=0

echo "  Model: homogeneous + layer at z=400m"

# =========================================================
# Wavelet
# =========================================================
makewave fp=12 fmin=2 fmax=24 dt=0.001 nt=1024 file_out=wave.su t0=0.10 verbose=0
echo "  Wavelet: fp=12, fmin=2, fmax=24 Hz"

# =========================================================
# Forward modeling (1 shot)
# =========================================================
echo "  Running fdelmodc..."
fdelmodc \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=shot \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.002 tmod=1.5 verbose=0 \
    xrcv1=100 xrcv2=1900 zrcv1=50 zrcv2=50 dxrcv=10 \
    xsrc=1000 zsrc=10 ntaper=100 \
    left=4 right=4 top=1 bottom=4

# =========================================================
# Compute hydrophone: P = 0.5*(Tzz + Txx)
# =========================================================
suop2 shot_rtzz.su shot_rtxx.su op=sum | sugain scale=0.5 > shot_rp.su
echo "  Hydrophone computed: shot_rp.su"

# Report trace info
echo ""
echo "  Trace info:"
echo "    rvz: ns=$(sugethw < shot_rvz.su key=ns 2>&1 | head -1 | awk '{print $1}' | cut -d= -f2)"
echo "    rp:  ns=$(sugethw < shot_rp.su key=ns 2>&1 | head -1 | awk '{print $1}' | cut -d= -f2)"

# =========================================================
# Filter at 3 bands using sufilter
# =========================================================
# Band [2, 5] Hz:  taper=0.5 -> f1=1.5, f2=2, f3=5, f4=5.5
echo ""
echo "  Filtering [2, 5] Hz..."
sufilter < shot_rp.su f=1.50,2.00,5.00,5.50 > shot_rp_2_5.su

# Band [2, 10] Hz: taper=0.8 -> f1=1.2, f2=2, f3=10, f4=10.8
echo "  Filtering [2, 10] Hz..."
sufilter < shot_rp.su f=1.20,2.00,10.00,10.80 > shot_rp_2_10.su

# Band [2, 20] Hz: taper=1.8 -> f1=0.2, f2=2, f3=20, f4=21.8
echo "  Filtering [2, 20] Hz..."
sufilter < shot_rp.su f=0.20,2.00,20.00,21.80 > shot_rp_2_20.su

# Also filter rvz for comparison
echo "  Filtering rvz [2, 5] Hz..."
sufilter < shot_rvz.su f=1.50,2.00,5.00,5.50 > shot_rvz_2_5.su
echo "  Filtering rvz [2, 10] Hz..."
sufilter < shot_rvz.su f=1.20,2.00,10.00,10.80 > shot_rvz_2_10.su
echo "  Filtering rvz [2, 20] Hz..."
sufilter < shot_rvz.su f=0.20,2.00,20.00,21.80 > shot_rvz_2_20.su

echo ""
echo "========================================================="
echo " DONE. Output in: $(pwd)"
echo "========================================================="
echo ""
echo " Unfiltered:"
echo "   shot_rp.su          (hydrophone, full band)"
echo "   shot_rvz.su         (vertical velocity, full band)"
echo ""
echo " Filtered hydrophone:"
echo "   shot_rp_2_5.su      [2, 5] Hz"
echo "   shot_rp_2_10.su     [2, 10] Hz"
echo "   shot_rp_2_20.su     [2, 20] Hz"
echo ""
echo " Filtered rvz:"
echo "   shot_rvz_2_5.su     [2, 5] Hz"
echo "   shot_rvz_2_10.su    [2, 10] Hz"
echo "   shot_rvz_2_20.su    [2, 20] Hz"
echo ""
echo " Quick look:"
echo "   suximage < shot_rp.su title='rp full' &"
echo "   suximage < shot_rp_2_5.su title='rp [2-5Hz]' &"
echo "   suximage < shot_rp_2_10.su title='rp [2-10Hz]' &"
echo "   suximage < shot_rp_2_20.su title='rp [2-20Hz]' &"
echo "========================================================="
