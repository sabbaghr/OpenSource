#!/bin/bash
set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=${ROOT}/fdelfwi

export PATH=${BIN}:${PATH}
export OMP_NUM_THREADS=4

WORKDIR=trn_debug_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

# Models
makemod sizex=1000 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=1000 ro0=2000 \
    orig=0,0 file_base=true.su \
    intt=diffr dtype=2 x=500 z=250 cp=1700 cs=1000 ro=2000 var=20 \
    verbose=0

makemod sizex=1000 sizez=500 dx=5 dz=5 \
    cp0=2000 cs0=1000 ro0=2000 \
    orig=0,0 file_base=init.su \
    verbose=0

# Wavelet
makewave fp=20 dt=0.001 nt=2048 fmax=30 file_out=wave.su t0=0.10 verbose=0

# Observed data
fdelmodc \
    file_cp=true_cp.su file_cs=true_cs.su file_den=true_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=1 \
    dtrcv=0.004 tmod=1.0 verbose=0 \
    xrcv1=50 xrcv2=950 zrcv1=490 zrcv2=490 dxrcv=10 \
    rec_type_vz=1 rec_type_vx=1 rec_type_p=1 rec_type_tzz=1 rec_type_txx=1 \
    xsrc=500 zsrc=10 ntaper=30 \
    left=4 right=4 top=4 bottom=4

for f in obs_r*.su; do
    newname=$(echo "$f" | sed 's/obs_/obs_000_/')
    cp "$f" "$newname"
done

CWP=/rcp3/software/codes/cwp/bin
if [ -x "${CWP}/suop2" ]; then
    ${CWP}/suop2 obs_000_rtzz.su obs_000_rtxx.su op=sum > obs_sum_tmp.su
    ${CWP}/sugain scale=0.5 < obs_sum_tmp.su > obs_000_rp.su
    rm -f obs_sum_tmp.su
fi

echo "=== Running TRN inversion (algorithm=4) ==="

# Enable glibc extra heap checking (disable for normal testing)
# export MALLOC_CHECK_=3

${FDELFWI}/fwi_inversion \
    file_cp=init_cp.su file_cs=init_cs.su file_den=init_ro.su \
    file_src=wave.su file_rcv=syn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vz=1 rec_type_vx=1 rec_type_p=1 rec_type_tzz=1 rec_type_txx=1 \
    dtrcv=0.004 tmod=1.0 verbose=1 \
    xrcv1=50 xrcv2=950 zrcv1=490 zrcv2=490 dxrcv=10 \
    xsrc=500 zsrc=10 ntaper=30 \
    left=4 right=4 top=4 bottom=4 \
    file_obs=obs comp=_rp \
    chk_skipdt=100 chk_base=chk \
    param=1 \
    grad_taper=5 write_iter=1 \
    niter=2 algorithm=4 niter_cg=2 \
    file_grad=gradient_trn 2>&1

echo "=== Exit code: $? ==="
