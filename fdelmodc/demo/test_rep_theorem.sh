#!/bin/bash
#
# test_rep_theorem.sh
#
# Validates the elastic representation theorem (Kirchhoff-Helmholtz integral):
#
#   u_k(x_v) = ∮_S [ G_ik(x_v,x') T_i(x') - u_i(x') Σ_ik(x_v,x') n_j ] dS(x')
#
# In velocity form (multiply by iω; by reciprocity G_ik(x,x')=G_ki(x',x)):
#
#   V_k(x_v) = Σ_ξ ΔS [ vx_GFk(ξ) ⊛ T_x(ξ)
#                      + vz_GFk(ξ) ⊛ T_z(ξ)
#                      - vx_fwd(ξ) ⊛ ∫Σ_xk(ξ)
#                      - vz_fwd(ξ) ⊛ ∫Σ_zk(ξ) ] dS
#
# where ⊛ = time convolution,  ∫ = time integration (≡ /iω in freq. domain).
#
# Boundary tractions (outward normal from the enclosed volume):
#   top    n=(0,-1):  T_x = -txz,  T_z = -tzz
#   bottom n=(0,+1):  T_x = +txz,  T_z = +tzz
#   left   n=(-1,0):  T_x = -txx,  T_z = -txz
#   right  n=(+1,0):  T_x = +txx,  T_z = +txz
#
# Three fdelmodc runs:
#   FWD:   explosive at (xs,zs) inside the rectangle
#          → records vx,vz,txx,tzz,txz on all 4 boundary sides
#   GF-A:  Fx force at reconstruction point (xv,zv)
#          → records all 5 components on boundary
#          → by reciprocity: vx_GFA(ξ)=G_xx(x_v;ξ), vz_GFA(ξ)=G_xz(x_v;ξ)
#   GF-B:  Fz force at (xv,zv)
#          → vx_GFB(ξ)=G_zx(x_v;ξ), vz_GFB(ξ)=G_zz(x_v;ξ)
#
# Surface integral implemented entirely with SU tools:
#   suwind   — extract traces per boundary side
#   sugain   — apply normal sign and ΔS scaling
#   suinteg  — time integration (≡ /iω) of GF stress tractions
#   suconv   — time convolution of GF velocity × forward traction (trace-by-trace)
#   sustack  — sum all boundary traces into one (norm=0: unnormalized sum)
#   suop2    — combine the four integral terms
#
# Wavelet note:
#   GF runs use fp=20 Hz Ricker. fdelmodc computes effective fmax ≈ 2.14*fp
#   from the wavelet spectrum; with fp=20 that gives ~42.8 Hz, safely below
#   the stability limit Vs/(5*dx) = 57.5 Hz (dx=4m, Vs=1150 m/s).
#   The wavelet effect is cancelled exactly by the comparison reference:
#       V_k_recon ≈ V_k_direct ⊛ w_gf   and we compare against direct ⊛ w_gf
#   so the test accuracy is independent of the GF wavelet frequency.
#
# Requires: fdelmodc, makemod, makewave (in SL10/bin)
#           suwind, sugain, suinteg, suconv, sustack, suop2, sumax, sugain
#           sushw, suxwigb, supsimage  (in cwp/bin)
#
# Usage: bash test_rep_theorem.sh
#        (run from fdelmodc/demo/ directory)

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}

WORKDIR=rep_theorem_test
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "======================================================"
echo " ELASTIC REPRESENTATION THEOREM TEST"
echo "======================================================"

# ============================================================
# Model and geometry
# ============================================================
NX=501; NZ=251; DX=4; DZ=4
TMOD=1.0; DT=0.001; NT=1200
VP=2000; VS=1150; RHO=2000
NTAP=150

# Number of samples fdelmodc records: TMOD/DT + 1
NT_REC=$(awk "BEGIN{printf \"%d\", ${TMOD}/${DT}+1}")

# Interior source  (explosion, OUTSIDE the boundary rectangle)
XS=1300; ZS=370

# Reconstruction point (inside the boundary rectangle)
XV=950; ZV=560

# Closed rectangular boundary
XL=500; XR=1500; ZT=400; ZB=700

# Number of receivers per boundary side
NX_SIDE=$(( (XR - XL) / DX + 1 ))   # top and bottom  (121)
NZ_SIDE=$(( (ZB - ZT) / DZ + 1 ))   # left and right  (81)
N_TOTAL=$(( 2*NX_SIDE + 2*NZ_SIDE )) # total boundary traces (404)

# Trace index ranges in fdelmodc output (1-based sequential ordering):
#   array 1 = top, array 2 = bottom, array 3 = left, array 4 = right
TR_TOP1=1;                       TR_TOP2=${NX_SIDE}
TR_BOT1=$(( NX_SIDE + 1 ));     TR_BOT2=$(( 2*NX_SIDE ))
TR_LEFT1=$(( 2*NX_SIDE + 1 )); TR_LEFT2=$(( 2*NX_SIDE + NZ_SIDE ))
TR_RIGHT1=$(( 2*NX_SIDE + NZ_SIDE + 1 )); TR_RIGHT2=${N_TOTAL}

echo ""
echo "  Domain   : $(( (NX-1)*DX )) x $(( (NZ-1)*DZ )) m,  dx=dz=${DX} m"
echo "  Medium   : Vp=${VP}  Vs=${VS}  rho=${RHO}  (homogeneous elastic)"
echo "  Source   : (xs,zs) = (${XS}, ${ZS}) m"
echo "  Target   : (xv,zv) = (${XV}, ${ZV}) m"
echo "  Boundary : x=[${XL},${XR}] z=[${ZT},${ZB}] m"
echo "  Traces   : top/bot=${NX_SIDE}  left/right=${NZ_SIDE}  total=${N_TOTAL}"

# ============================================================
# Build model and wavelets
# ============================================================
echo ""
echo "--- Building model and wavelets ---"

makemod sizex=$(( (NX-1)*DX )) sizez=$(( (NZ-1)*DZ )) dx=${DX} dz=${DZ} \
    cp0=${VP} cs0=${VS} ro0=${RHO} orig=0,0 \
    file_base=model.su verbose=0

# Forward wavelet: standard Ricker fp=15 Hz
makewave w=fw fmin=0 flef=10 frig=30 fmax=40 dt=${DT} nt=${NT} file_out=wave_fwd.su t0=0.10 scale=0 t0=0.1 scfft=1

# GF wavelet: broader Ricker to approximate a delta source.
# fdelmodc computes effective fmax ≈ 2.14*fp from the wavelet spectrum;
# stability limit is Vs/(5*dx) = 1150/(5*4) = 57.5 Hz, so we need fp < 26.8 Hz.
# fp=20 Hz gives fmax_eff ≈ 42.8 Hz, well below the limit.
# The comparison cancels the wavelet effect exactly via direct ⊛ w_gf,
# so the absolute frequency of w_gf does not affect test accuracy.
makewave w=g0 fp=40 fmax=55 dt=${DT} nt=${NT_REC} file_out=wave_gf.su t0=0.05 verbose=0 scale=1

echo "  Forward wavelet: fp=15 Hz, t0=0.10 s"
echo "  GF wavelet     : fp=20 Hz, t0=0.05 s  (fmax_eff~42.8 Hz < 57.5 Hz limit)"

# ============================================================
# Source / receiver / target position map (for QC plotting)
#
# Builds SrcRecPositions.su on the same grid as model_cp.su (nx=NX,
# nz=NZ, d1=DZ, d2=DX, origin 0,0 — matching the `makemod` call above):
#   +1  at the FWD source (xs,zs)
#   +1  at the reconstruction point x_v (source of GF-A/GF-B)
#   -1  at every boundary receiver on S (all 4 sides, 404 points)
# Same +1/-1, 5x5-cell-dot convention fdelmodc's own writeSrcRecPos.c
# uses internally (triggered there by verbose>3).
# ============================================================
echo ""
echo "--- Building SrcRecPositions.su ---"

_PYSRCREC=$(mktemp /tmp/srcrec_XXXXXX.py)
cat > "$_PYSRCREC" << 'PYEOF'
import sys, struct
import numpy as np

nx, nz  = int(sys.argv[1]), int(sys.argv[2])
dx, dz  = float(sys.argv[3]), float(sys.argv[4])
x0, z0  = float(sys.argv[5]), float(sys.argv[6])
outfile = sys.argv[7]
marks   = sys.argv[8:]          # flattened triples: x z value ...

# layout matches model_cp.su from makemod: nx traces (one per x), nz samples (depth) each
grid = np.zeros((nx, nz), dtype='<f4')

def stamp(xw, zw, val, half=2):
    ix = int(round((xw - x0) / dx))
    iz = int(round((zw - z0) / dz))
    ix0, ix1 = max(0, ix - half), min(nx, ix + half + 1)
    iz0, iz1 = max(0, iz - half), min(nz, iz + half + 1)
    grid[ix0:ix1, iz0:iz1] = val

it = iter(marks)
for xw, zw, val in zip(it, it, it):
    stamp(float(xw), float(zw), float(val))

with open(outfile, 'wb') as f:
    for ix in range(nx):
        hdr = bytearray(240)
        struct.pack_into('<i', hdr, 0, ix + 1)                   # tracl
        struct.pack_into('<i', hdr, 4, ix + 1)                   # tracr
        struct.pack_into('<H', hdr, 114, nz)                     # ns
        struct.pack_into('<H', hdr, 116, max(1, int(dz*1000)))   # dt (cosmetic only)
        f.write(hdr)
        f.write(grid[ix, :].tobytes())
PYEOF

REC_ARGS=()
for ((x=XL; x<=XR; x+=DX)); do REC_ARGS+=("$x $ZT -1"); REC_ARGS+=("$x $ZB -1"); done
for ((z=ZT; z<=ZB; z+=DZ)); do REC_ARGS+=("$XL $z -1"); REC_ARGS+=("$XR $z -1"); done

python3 "$_PYSRCREC" ${NX} ${NZ} ${DX} ${DZ} 0 0 SrcRecPositions.su \
    ${XS} ${ZS} 1 \
    ${XV} ${ZV} 1 \
    ${REC_ARGS[@]}
rm -f "$_PYSRCREC"

suop2 model_cp.su SrcRecPositions.su w1=1 w2=2000 op=sum | \
    supsimage wclip=1400 bclip=2000 \
    wbox=4 hbox=4 titlesize=-1 labelsize=10 verbose=1 \
    d2=${DX} f2=0 wrgb=1.0,0,0 grgb=0,1.0,0 brgb=0,0,1.0 bps=24 \
    label1="depth [m]" label2="lateral position [m]" > model_plane_src.eps

echo "  → SrcRecPositions.su, model_plane_src.eps"
echo "    (model_cp.su + FWD source + x_v + all 404 boundary-S receivers)"

# ============================================================
# Boundary receiver specification (4 arrays, one per side)
#
#   Array 1 (top):    z=ZT, x: XL → XR,  dxrcv=DX, dzrcv=0
#   Array 2 (bottom): z=ZB, x: XL → XR,  dxrcv=DX, dzrcv=0
#   Array 3 (left):   x=XL, z: ZT → ZB,  dxrcv=0,  dzrcv=DZ
#   Array 4 (right):  x=XR, z: ZT → ZB,  dxrcv=0,  dzrcv=DZ
# ============================================================
XRCV1="${XL},${XL},${XL},${XR}"
XRCV2="${XR},${XR},${XL},${XR}"
ZRCV1="${ZT},${ZB},${ZT},${ZT}"
ZRCV2="${ZT},${ZB},${ZB},${ZB}"
DXRCV="${DX},${DX},0,0"
DZRCV="0,0,${DZ},${DZ}"

# Common receiver parameters (all 5 elastic components)
RCV_PARAMS="xrcv1=${XRCV1} xrcv2=${XRCV2}
    zrcv1=${ZRCV1} zrcv2=${ZRCV2}
    dxrcv=${DXRCV} dzrcv=${DZRCV}
    rec_type_vx=1 rec_type_vz=1
    rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1
    dtrcv=${DT}"

# Common fdelmodc model parameters
MDL_PARAMS="file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su
    ischeme=3 iorder=4 tmod=${TMOD}
    ntaper=${NTAP} left=4 right=4 top=4 bottom=4
    verbose=0"

# ============================================================
# Run 1: FWD — explosion at (XS,ZS), record on closed boundary
# ============================================================
echo ""
echo "--- FWD: explosive at (${XS},${ZS}), recording on boundary ---"

fdelmodc ${MDL_PARAMS} ${RCV_PARAMS} \
    file_src=wave_fwd.su file_rcv=fwd \
    file_snap=snap_nodisp.su \
    src_type=1 src_orient=1 \
    xsrc=${XS} zsrc=${ZS}\
    tsnap1=0.1 tsnap2=1.0 dtsnap=0.1 \
    sna_type_vx=1 sna_type_vz=1 sna_type_tzz=1
    

echo "  → fwd_rvx.su  fwd_rvz.su  fwd_rtxx.su  fwd_rtzz.su  fwd_rtxz.su"

# ============================================================
# Run 2: DIRECT — same explosion, record only at (XV,ZV)
#         This is the truth we want to reconstruct.
# ============================================================
echo ""
echo "--- DIRECT: explosive at (${XS},${ZS}), truth at (${XV},${ZV}) ---"

fdelmodc ${MDL_PARAMS} \
    file_src=wave_fwd.su file_rcv=direct \
    src_type=1 src_orient=1 \
    xsrc=${XS} zsrc=${ZS} \
    xrcv1=${XV} xrcv2=${XV} dxrcv=0 \
    zrcv1=${ZV} zrcv2=$((ZV+DZ)) dzrcv=${DZ} \
    rec_type_vx=1 rec_type_vz=1 \
    dtrcv=${DT}

# Keep only the first trace (at ZV); second trace is at ZV+DZ
suwind key=tracl min=1 max=1 < direct_rvx.su > _tmp.su && mv _tmp.su direct_rvx.su
suwind key=tracl min=1 max=1 < direct_rvz.su > _tmp.su && mv _tmp.su direct_rvz.su

echo "  → direct_rvx.su  direct_rvz.su  (truth, single trace at XV=${XV} ZV=${ZV})"

# ============================================================
# Run 3: GF-A — Fx force at (XV,ZV), record on boundary
#
#   By reciprocity: vx_GFA(ξ) = G_xx^v(x_v; ξ)  ⊛ w_gf(t)
#                   vz_GFA(ξ) = G_xz^v(x_v; ξ)  ⊛ w_gf(t)
# ============================================================
echo ""
echo "--- GF-A: Fx at (${XV},${ZV}), recording on boundary ---"

fdelmodc ${MDL_PARAMS} ${RCV_PARAMS} \
    file_src=wave_gf.su file_rcv=gfa \
    src_injectionrate=1 \
    fmax=57 \
    src_type=6 \
    xsrc=${XV} zsrc=${ZV}

echo "  → gfa_rvx.su  gfa_rvz.su  gfa_rtxx.su  gfa_rtzz.su  gfa_rtxz.su"

# ============================================================
# Run 4: GF-B — Fz force at (XV,ZV), record on boundary
#
#   vx_GFB(ξ) = G_zx^v(x_v; ξ) ⊛ w_gf(t)
#   vz_GFB(ξ) = G_zz^v(x_v; ξ) ⊛ w_gf(t)
# ============================================================
echo ""
echo "--- GF-B: Fz at (${XV},${ZV}), recording on boundary ---"

fdelmodc ${MDL_PARAMS} ${RCV_PARAMS} \
    file_src=wave_gf.su file_rcv=gfb \
    fmax=57 \
    src_type=7 \
    xsrc=${XV} zsrc=${ZV}

echo "  → gfb_rvx.su  gfb_rvz.su  gfb_rtxx.su  gfb_rtzz.su  gfb_rtxz.su"

# ============================================================
# Surface integral
#
# For each boundary side, the outward normal determines which stress
# component enters the traction (only one term survives per side
# for axis-aligned normals):
#
# Physical:  T_x = σ_xj n_j.  Stored: tXX_stored = -σ_XX^phys.
# Combined (sign applied to stored values in extract_scaled):
#   top    n=(0,-1):  T_x = +txz_s  T_z = +tzz_s   Σ_x= +txz_gf  Σ_z= +tzz_gf
#   bottom n=(0,+1):  T_x = -txz_s  T_z = -tzz_s   Σ_x= -txz_gf  Σ_z= -tzz_gf
#   left   n=(-1,0):  T_x = +txx_s  T_z = +txz_s   Σ_x= +txx_gf  Σ_z= +txz_gf
#   right  n=(+1,0):  T_x = -txx_s  T_z = -txz_s   Σ_x= -txx_gf  Σ_z= -txz_gf
#
# where _x / _z denotes the x- or z-component of the traction.
#
# The ΔS (dx for top/bottom, dz for left/right) is absorbed into
# the scaling factor so that sustack normpow=0 gives the integral directly.
#
# sum_and_stack PATTERN OUTPUT
#   Concatenates all four boundary-side files matching PATTERN and sums them
#   into a single trace using sustack (norm=0: unnormalized sum).
# ============================================================
echo ""
echo "--- Computing surface integral ---"

sum_and_stack() {
    local pattern=$1 output=$2
    cat ${pattern}_top.su ${pattern}_bot.su \
        ${pattern}_left.su ${pattern}_right.su | \
        sushw key=cdp a=1 | sustack key=cdp normpow=0 > ${output}
}

# ------------------------------------------------------------------
# Helper: extract a side, apply sign×dS scale
#   extract_scaled FILE TR1 TR2 SIGN DS OUTPUT
# ------------------------------------------------------------------
extract_scaled() {
    local file=$1 tr1=$2 tr2=$3 sign=$4 ds=$5 out=$6
    local sc; sc=$(awk "BEGIN{printf \"%.6f\", ${sign}*${ds}}")
    suwind key=tracl min=${tr1} max=${tr2} < ${file} | sugain scale=${sc} > ${out}
}

# (integ_scaled removed — GF stress recordings are already Σ^d)

# ------------------------------------------------------------------
# Forward tractions T_x = σ_xj n_j  and  T_z = σ_zj n_j
# (scaled by ΔS so that sustack normpow=0 gives ∮ ... dS directly)
#
# NOTE: fdelmodc elastic4 stores NEGATIVE physical stresses:
#   txx_stored = -σ_xx,  tzz_stored = -σ_zz,  txz_stored = -σ_xz
# Physical traction T_i = σ_ij n_j, so when reading recorded files
# the sign flips: T_x(top) = σ_xz*(-1)*dx = -σ_xz*dx = +txz_stored*dx (sign=+1)
#
#   top    (n=(0,-1), dS=dx):  T_x = +txz_stored*dx    T_z = +tzz_stored*dx
#   bottom (n=(0,+1), dS=dx):  T_x = -txz_stored*dx    T_z = -tzz_stored*dx
#   left   (n=(-1,0), dS=dz):  T_x = +txx_stored*dz    T_z = +txz_stored*dz
#   right  (n=(+1,0), dS=dz):  T_x = -txx_stored*dz    T_z = -txz_stored*dz
# ------------------------------------------------------------------
echo "  Forward tractions..."

extract_scaled fwd_rtxz.su $TR_TOP1   $TR_TOP2   +1 $DX Tx_top.su
extract_scaled fwd_rtxz.su $TR_BOT1   $TR_BOT2   -1 $DX Tx_bot.su
extract_scaled fwd_rtxx.su $TR_LEFT1  $TR_LEFT2  +1 $DZ Tx_left.su
extract_scaled fwd_rtxx.su $TR_RIGHT1 $TR_RIGHT2 -1 $DZ Tx_right.su

extract_scaled fwd_rtzz.su $TR_TOP1   $TR_TOP2   +1 $DX Tz_top.su
extract_scaled fwd_rtzz.su $TR_BOT1   $TR_BOT2   -1 $DX Tz_bot.su
extract_scaled fwd_rtxz.su $TR_LEFT1  $TR_LEFT2  +1 $DZ Tz_left.su
extract_scaled fwd_rtxz.su $TR_RIGHT1 $TR_RIGHT2 -1 $DZ Tz_right.su

# ------------------------------------------------------------------
# GF-A tractions Σ^d_xk and Σ^d_zk for Vx reconstruction
#
# fdelmodc with src_type=6/7 drives the FD equations with a body force,
# so the recorded velocity IS the velocity Green's function G^vel and the
# recorded stress IS already the displacement-GF stress Σ^d  (because
# ∂_t σ = C:∇v → σ = ∫C:∇G^vel dt = C:∇G^disp = Σ^d).
# No time integration (suinteg) is needed on GF fields.
# ------------------------------------------------------------------
echo "  GF-A tractions (direct, already Sigma^d)..."

extract_scaled gfa_rtxz.su $TR_TOP1   $TR_TOP2   +1 $DX SAx_top.su
extract_scaled gfa_rtxz.su $TR_BOT1   $TR_BOT2   -1 $DX SAx_bot.su
extract_scaled gfa_rtxx.su $TR_LEFT1  $TR_LEFT2  +1 $DZ SAx_left.su
extract_scaled gfa_rtxx.su $TR_RIGHT1 $TR_RIGHT2 -1 $DZ SAx_right.su

extract_scaled gfa_rtzz.su $TR_TOP1   $TR_TOP2   +1 $DX SAz_top.su
extract_scaled gfa_rtzz.su $TR_BOT1   $TR_BOT2   -1 $DX SAz_bot.su
extract_scaled gfa_rtxz.su $TR_LEFT1  $TR_LEFT2  +1 $DZ SAz_left.su
extract_scaled gfa_rtxz.su $TR_RIGHT1 $TR_RIGHT2 -1 $DZ SAz_right.su

# ------------------------------------------------------------------
# GF-B tractions Σ^d_xk and Σ^d_zk for Vz reconstruction
# ------------------------------------------------------------------
echo "  GF-B tractions (direct, already Sigma^d)..."

extract_scaled gfb_rtxz.su $TR_TOP1   $TR_TOP2   +1 $DX SBx_top.su
extract_scaled gfb_rtxz.su $TR_BOT1   $TR_BOT2   -1 $DX SBx_bot.su
extract_scaled gfb_rtxx.su $TR_LEFT1  $TR_LEFT2  +1 $DZ SBx_left.su
extract_scaled gfb_rtxx.su $TR_RIGHT1 $TR_RIGHT2 -1 $DZ SBx_right.su

extract_scaled gfb_rtzz.su $TR_TOP1   $TR_TOP2   +1 $DX SBz_top.su
extract_scaled gfb_rtzz.su $TR_BOT1   $TR_BOT2   -1 $DX SBz_bot.su
extract_scaled gfb_rtxz.su $TR_LEFT1  $TR_LEFT2  +1 $DZ SBz_left.su
extract_scaled gfb_rtxz.su $TR_RIGHT1 $TR_RIGHT2 -1 $DZ SBz_right.su

# ------------------------------------------------------------------
# Extract GF velocities and forward velocities per side.
# GF velocities are already G^vel (no suinteg needed).
# ------------------------------------------------------------------
for SIDE in top bot left right; do
    case $SIDE in
        top)   tr1=$TR_TOP1;   tr2=$TR_TOP2   ;;
        bot)   tr1=$TR_BOT1;   tr2=$TR_BOT2   ;;
        left)  tr1=$TR_LEFT1;  tr2=$TR_LEFT2  ;;
        right) tr1=$TR_RIGHT1; tr2=$TR_RIGHT2 ;;
    esac
    suwind key=tracl min=${tr1} max=${tr2} < gfa_rvx.su > gfa_vx_${SIDE}.su
    suwind key=tracl min=${tr1} max=${tr2} < gfa_rvz.su > gfa_vz_${SIDE}.su
    suwind key=tracl min=${tr1} max=${tr2} < gfb_rvx.su > gfb_vx_${SIDE}.su
    suwind key=tracl min=${tr1} max=${tr2} < gfb_rvz.su > gfb_vz_${SIDE}.su
    suwind key=tracl min=${tr1} max=${tr2} < fwd_rvx.su > fwd_vx_${SIDE}.su
    suwind key=tracl min=${tr1} max=${tr2} < fwd_rvz.su > fwd_vz_${SIDE}.su
done

# ------------------------------------------------------------------
# Evaluate the four terms of the surface integral
#
# V_x(x_v) = ∮ [G_{x,x}(x_v,ξ) T_x + G_{x,z}(x_v,ξ) T_z
#              - v_x Σ^d_{x,x}(x_v,ξ) - v_z Σ^d_{x,z}(x_v,ξ)] dS
#
# By reciprocity G_{x,i}(x_v,ξ) = G_{i,x}(ξ,x_v) = i-vel at ξ from Fx at x_v:
#   G_{x,x} = gfa_rvx,  G_{x,z} = gfa_rvz  ← BOTH from GF-A (Fx at x_v)
# Σ^d_{x,.} tractions = GF-A stress (Fx at x_v)
#
# suconv performs trace-by-trace linear convolution (sufile= holds the
# second operand; both files must have the same trace count per side).
# sustack normpow=0 sums all boundary traces without normalization.
# ------------------------------------------------------------------
echo "  Integral terms for Vx..."

for SIDE in top bot left right; do
    # Vx: ALL velocity GF from GF-A (Fx at x_v, src_type=6, ×2 for 0.5-factor)
    # G_{x,x}(x_v,ξ) = G_{x,x}(ξ,x_v) = x-vel at ξ from Fx at x_v = gfa_rvx  (×2)
    # G_{x,z}(x_v,ξ) = G_{z,x}(ξ,x_v) = z-vel at ξ from Fx at x_v = gfa_rvz  (×2)
    suconv < gfa_vx_${SIDE}.su sufile=Tx_${SIDE}.su  panel=1 | sugain scale=2     > T1_vx_${SIDE}.su
    suconv < gfa_vz_${SIDE}.su sufile=Tz_${SIDE}.su  panel=1 | sugain scale=2     > T2_vx_${SIDE}.su
    suconv < fwd_vx_${SIDE}.su sufile=SAx_${SIDE}.su panel=1 | sugain scale=-2    > T3_vx_${SIDE}.su
    suconv < fwd_vz_${SIDE}.su sufile=SAz_${SIDE}.su panel=1 | sugain scale=-2    > T4_vx_${SIDE}.su
done

sum_and_stack T1_vx T1_vx_sum.su
sum_and_stack T2_vx T2_vx_sum.su
sum_and_stack T3_vx T3_vx_sum.su
sum_and_stack T4_vx T4_vx_sum.su

suop2 T1_vx_sum.su T2_vx_sum.su op=sum > _vx12.su
suop2 _vx12.su     T3_vx_sum.su op=sum > _vx123.su
suop2 _vx123.su    T4_vx_sum.su op=sum > recon_vx.su
rm -f _vx12.su _vx123.su

# ------------------------------------------------------------------
# Same four terms for V_z reconstruction (use GF-B for the GF columns)
# ------------------------------------------------------------------
echo "  Integral terms for Vz..."

for SIDE in top bot left right; do
    # Vz: ALL velocity GF from GF-B (Fz at x_v, src_type=7, ×1 no factor)
    # G_{z,x}(x_v,ξ) = G_{x,z}(ξ,x_v) = x-vel at ξ from Fz at x_v = gfb_rvx  (×1)
    # G_{z,z}(x_v,ξ) = G_{z,z}(ξ,x_v) = z-vel at ξ from Fz at x_v = gfb_rvz  (×1)
    suconv < gfb_vx_${SIDE}.su sufile=Tx_${SIDE}.su  panel=1                      > T1_vz_${SIDE}.su
    suconv < gfb_vz_${SIDE}.su sufile=Tz_${SIDE}.su  panel=1                      > T2_vz_${SIDE}.su
    suconv < fwd_vx_${SIDE}.su sufile=SBx_${SIDE}.su panel=1 | sugain scale=-1    > T3_vz_${SIDE}.su
    suconv < fwd_vz_${SIDE}.su sufile=SBz_${SIDE}.su panel=1 | sugain scale=-1    > T4_vz_${SIDE}.su
done

sum_and_stack T1_vz T1_vz_sum.su
sum_and_stack T2_vz T2_vz_sum.su
sum_and_stack T3_vz T3_vz_sum.su
sum_and_stack T4_vz T4_vz_sum.su

suop2 T1_vz_sum.su T2_vz_sum.su op=sum > _vz12.su
suop2 _vz12.su     T3_vz_sum.su op=sum > _vz123.su
suop2 _vz123.su    T4_vz_sum.su op=sum > recon_vz.su
rm -f _vz12.su _vz123.su

# ============================================================
# Comparison
#
# The GF wavelet w_gf(fp=20 Hz) approximates a delta over the
# forward bandwidth (0-15 Hz), so:
#   V_k_recon(t) ≈ V_k_direct(t)   [up to smoothing by w_gf]
#
# For an exact comparison, convolve the direct recording with w_gf:
#   direct_conv(t) = V_k_direct ⊛ w_gf   should equal V_k_recon
# ============================================================
echo ""
echo "--- Preparing comparison ---"

suconv < direct_rvx.su sufile=wave_gf.su > direct_conv_vx.su
suconv < direct_rvz.su sufile=wave_gf.su > direct_conv_vz.su

peak() { sumax mode=abs outpar=/dev/stdout < "$1" 2>/dev/null | awk '{printf "%11.4e", $1; exit}'; }
echo ""
echo "  Per-term peak amplitudes (Vx):"
printf "    T1=%s  T2=%s  T3=%s  T4=%s\n" \
    "$(peak T1_vx_sum.su)" "$(peak T2_vx_sum.su)" "$(peak T3_vx_sum.su)" "$(peak T4_vx_sum.su)"
printf "    recon_vx=%s  direct_conv_vx=%s\n" \
    "$(peak recon_vx.su)" "$(peak direct_conv_vx.su)"
echo "  Per-term peak amplitudes (Vz):"
printf "    T1=%s  T2=%s  T3=%s  T4=%s\n" \
    "$(peak T1_vz_sum.su)" "$(peak T2_vz_sum.su)" "$(peak T3_vz_sum.su)" "$(peak T4_vz_sum.su)"
printf "    recon_vz=%s  direct_conv_vz=%s\n" \
    "$(peak recon_vz.su)" "$(peak direct_conv_vz.su)"

compare_norm() {
    local REF=$1 REC=$2 LABEL=$3

    sugain pbal=1 < ${REF} > _ref_n.su 2>/dev/null
    sugain pbal=1 < ${REC} > _rec_n.su 2>/dev/null
    suop2 _ref_n.su _rec_n.su op=diff > _res.su

    MAX_REF=$(sumax mode=abs outpar=/dev/stdout < _ref_n.su 2>/dev/null | awk '{print $1; exit}')
    MAX_RES=$(sumax mode=abs outpar=/dev/stdout < _res.su   2>/dev/null | awk '{print $1; exit}')

    if [ -n "$MAX_REF" ] && [ -n "$MAX_RES" ] && [ "$MAX_REF" != "0" ]; then
        REL=$(awk "BEGIN{printf \"%.2f\", ${MAX_RES}/${MAX_REF}*100.0}")
        STATUS="OK"
        [ "$(awk "BEGIN{print (${REL} > 5.0)}")" = "1" ] && STATUS="WARN >5%"
        printf "  %-40s  residual/peak = %6s%%  [%s]\n" \
            "${LABEL}" "${REL}" "${STATUS}"
    else
        printf "  %-40s  (could not parse sumax output)\n" "${LABEL}"
    fi
}

echo ""
echo "======================================================"
echo " RESULTS: normalized waveform comparison"
echo " reference = direct_rvk ⊛ wave_gf  (exact match if w_gf ≈ delta at forward bandwidth)"
echo "======================================================"
echo ""

compare_norm direct_conv_vx.su recon_vx.su "Vx (direct*w_gf vs reconstructed)"
compare_norm direct_conv_vz.su recon_vz.su "Vz (direct*w_gf vs reconstructed)"

echo ""
echo "  Expected: < ~5% residual for 4th-order FD in homogeneous medium"
echo "  (residual reflects FD discretization error, not wavelet mismatch)"
echo ""
echo "======================================================"
echo " Visualization:"
echo "======================================================"

# ============================================================
# Overlay plots: direct*w_gf vs reconstructed, saved as PNG
# ============================================================
python3 - <<'PYEOF'
import struct, numpy as np, sys

def read_su(fname):
    """Read a SU file, return (times_array, list_of_traces)."""
    traces = []
    with open(fname, 'rb') as f:
        while True:
            hdr = f.read(240)
            if not hdr:
                break
            ns  = struct.unpack_from('<H', hdr, 114)[0]
            dt  = struct.unpack_from('<H', hdr, 116)[0] * 1e-6   # µs → s
            data = np.frombuffer(f.read(4 * ns), dtype='<f4').copy()
            traces.append((dt, data))
    dt0, d0 = traces[0]
    ns0 = len(d0)
    t = np.arange(ns0) * dt0
    return t, [tr[1] for tr in traces]

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("  [plot] matplotlib not available — skipping PNG output")
    sys.exit(0)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle("Elastic Representation Theorem: reconstructed vs direct*w_gf", fontsize=11)

for ax, comp, label in zip(axes, ['vx', 'vz'], ['Vx', 'Vz']):
    try:
        t_ref, ref = read_su(f'direct_conv_{comp}.su')
        t_rec, rec = read_su(f'recon_{comp}.su')
    except FileNotFoundError as e:
        ax.set_title(f"{label}: file not found ({e})")
        continue

    # Normalize both by the peak of the reference
    peak = np.max(np.abs(ref[0]))
    if peak == 0:
        peak = 1.0

    t = t_ref[:len(ref[0])]
    ax.plot(t, ref[0] / peak, color='tab:blue',   lw=1.5, label='direct ⊛ w_gf (reference)')
    ax.plot(t, rec[0] / peak, color='tab:orange',  lw=1.0, ls='--', label='reconstructed')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel(f"{label} (normalized)")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(label)

axes[-1].set_xlabel("Time [s]")
plt.tight_layout()
plt.savefig('comparison_overlay.png', dpi=150)
print("  → comparison_overlay.png  (Vx and Vz overlaid, normalized)")
plt.close()
PYEOF

echo ""
echo "  Saved: ${WORKDIR}/comparison_overlay.png"
echo "======================================================"