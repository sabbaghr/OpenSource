#!/bin/bash
#
# Acoustic 1-shot modeling on the Marmousi model + 3 smoothed versions.
# Model is resampled from dx=24m to dx=5m. The "true" model is lightly
# smoothed; three progressively smoother versions are also produced.

set -e

# ---- paths ------------------------------------------------------------
OSSL=/rcp3/software/codes/OpenSource_SL10
CWP_MARM=/rcp3/software/codes/cwp/src/demos/Velocity_Profiles/Marmousi
export PATH=${OSSL}/bin:/rcp3/software/codes/cwp/bin:$PATH

# ---- model geometry (resampled) ---------------------------------------
NZ_ORIG=122
NX_ORIG=384
DX_ORIG=24.0

DX=5.0          # target dx (m)
DZ=5.0          # target dz (m)

# resampled grid: depth=(NZ_ORIG-1)*DX_ORIG, width=(NX_ORIG-1)*DX_ORIG
NZ=$(python3 -c "print(int(round((${NZ_ORIG}-1)*${DX_ORIG}/${DX}))+1)")
NX=$(python3 -c "print(int(round((${NX_ORIG}-1)*${DX_ORIG}/${DX}))+1)")
echo "resampled grid: NZ=${NZ} NX=${NX} at dx=${DX} m"

# ---- resample Marmousi -------------------------------------------------
a2b n1=1 < ${CWP_MARM}/marmhard.dat > marm_raw.bin

python3 <<EOF
import numpy as np
from scipy.ndimage import zoom
# marmhard.dat (n1=122 depth, n2=384 x): depth is fast axis -> shape (nx, nz)
a = np.fromfile('marm_raw.bin', dtype=np.float32).reshape(${NX_ORIG}, ${NZ_ORIG})
zy = ${NX}/${NX_ORIG}
zx = ${NZ}/${NZ_ORIG}
b = zoom(a, (zy, zx), order=3).astype(np.float32)
b.tofile('marm_fine.bin')
print('resampled shape:', b.shape, 'min/max Vp:', b.min(), b.max())
EOF

# ---- light smoothing -> "true" model ----------------------------------
smooth2 n1=${NZ} n2=${NX} r1=10 r2=10 < marm_fine.bin > marm_true.bin

suaddhead < marm_true.bin ns=${NZ} |
    sushw key=dt,d1,d2,f1,f2,trid a=$(awk "BEGIN{print ${DZ}*1000}"),${DZ},${DX},0,0,130 |
    sushw key=tracl,tracf,cdp a=1,1,1 b=1,1,1 > marm_true_cp.su

# ---- homogeneous density ----------------------------------------------
python3 -c "
import struct
nz, nx, rho = ${NZ}, ${NX}, 1000.0
with open('marm_rho.bin','wb') as f:
    f.write(struct.pack('%df' % (nz*nx), *([rho]*(nz*nx))))
"
suaddhead < marm_rho.bin ns=${NZ} |
    sushw key=dt,d1,d2,f1,f2,trid a=$(awk "BEGIN{print ${DZ}*1000}"),${DZ},${DX},0,0,130 |
    sushw key=tracl,tracf,cdp a=1,1,1 b=1,1,1 > marm_true_ro.su

# ---- 3 smoother versions ----------------------------------------------
# r1/r2 in samples (z, x). At dx=5m, r=20 ~ 100 m smoothing length.
for pair in "smooth1:10.5:10.5" "smooth2:40:40" "smooth3:80:80"; do
    tag=${pair%%:*}; rest=${pair#*:}; r1=${rest%%:*}; r2=${rest#*:}
    echo "generating ${tag}: r1=${r1}, r2=${r2}"
    smooth2 n1=${NZ} n2=${NX} r1=${r1} r2=${r2} < marm_fine.bin > marm_${tag}.bin
    suaddhead < marm_${tag}.bin ns=${NZ} |
        sushw key=dt,d1,d2,f1,f2,trid a=$(awk "BEGIN{print ${DZ}*1000}"),${DZ},${DX},0,0,130 |
        sushw key=tracl,tracf,cdp a=1,1,1 b=1,1,1 > marm_${tag}_cp.su
done

# ---- source wavelet ----------------------------------------------------
makewave w=g1 fp=15 fmax=30 t0=0.1 dt=0.0005 nt=16384 \
         db=-40 file_out=wavelet.su verbose=1

# ---- common acquisition parameters -------------------------------------
XMAX=$(awk "BEGIN{print (${NX}-1)*${DX}}")
XSRC=1000.0
MAXOFF=6000.0
XRCV1=$(awk "BEGIN{v=${XSRC}-${MAXOFF}; if (v<0) v=0; print v}")
XRCV2=$(awk "BEGIN{v=${XSRC}+${MAXOFF}; if (v>${XMAX}) v=${XMAX}; print v}")
ZSRC=10.0
ZRCV=10.0
TMOD=4.0

run_shot() {
    local tag=$1
    local cp=$2
    echo "=== modeling shot on ${tag} ==="
    fdelmodc \
        ischeme=1 \
        file_cp=${cp} \
        file_den=marm_true_ro.su \
        file_src=wavelet.su \
        file_rcv=shot_${tag}.su \
        src_type=1 \
        xsrc=${XSRC} zsrc=${ZSRC} \
        nshot=1 \
        rec_type_p=1 rec_type_vz=0 \
        xrcv1=${XRCV1} xrcv2=${XRCV2} dxrcv=${DX} \
        zrcv1=${ZRCV} zrcv2=${ZRCV} \
        dtrcv=0.004 \
        tmod=${TMOD} \
        ntaper=60 \
        left=2 right=2 top=1 bottom=2 \
        verbose=2
}

run_shot true    marm_true_cp.su
run_shot smooth1 marm_smooth1_cp.su
run_shot smooth2 marm_smooth2_cp.su
run_shot smooth3 marm_smooth3_cp.su

# ---- quick-look PostScript plots --------------------------------------
for tag in true smooth1 smooth2 smooth3; do
    supsimage < shot_${tag}_rp.su perc=99 \
        label1="Time (s)" label2="Offset (m)" \
        title="Shot gather - ${tag}" > shot_${tag}.eps
    supsimage < marm_${tag}_cp.su wbox=8 hbox=3 legend=1 \
        label1="Depth (m)" label2="Distance (m)" \
        title="Vp - ${tag}" > model_${tag}.eps
    epstopdf shot_${tag}.eps  --outfile=shot_${tag}.pdf
    epstopdf model_${tag}.eps --outfile=model_${tag}.pdf
done

# ---- residuals: d_true - d_smoothN ------------------------------------
for tag in smooth1 smooth2 smooth3; do
    suop2 shot_true_rp.su shot_${tag}_rp.su op=diff > resid_${tag}.su
done

# shared symmetric color scale across the 3 residuals (99th pct of |amp|)
CLIP=$(python3 <<'PY'
import numpy as np, struct, glob
vals = []
for f in ['resid_smooth1.su','resid_smooth2.su','resid_smooth3.su']:
    with open(f,'rb') as fh:
        data = fh.read()
    ns = struct.unpack('<h', data[114:116])[0]   # ns header (short at byte 114)
    tlen = 240 + 4*ns
    ntr = len(data)//tlen
    a = np.empty(ntr*ns, dtype=np.float32)
    for i in range(ntr):
        off = i*tlen + 240
        a[i*ns:(i+1)*ns] = np.frombuffer(data[off:off+4*ns], dtype=np.float32)
    vals.append(np.abs(a))
v = np.concatenate(vals)
print(f"{np.percentile(v, 99):.6e}")
PY
)
echo "common residual clip = ${CLIP}"

for tag in smooth1 smooth2 smooth3; do
    supsimage < resid_${tag}.su bclip=${CLIP} wclip=-${CLIP} \
        label1="Time (s)" label2="Offset (m)" \
        title="Residual: true - ${tag}" > resid_${tag}.eps
    epstopdf resid_${tag}.eps --outfile=resid_${tag}.pdf
done

echo
echo "Done. Outputs:"
ls -1 shot_*_rp.su *.eps
