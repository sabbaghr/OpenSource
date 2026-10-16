#!/bin/bash
#
# wfl_case.sh <case-name> <wfl_tshift 0|1> <wfl_corner2 0|1>
#
# Self-contained experiment (no earlier run needed). Everything is created inside
# $OUTROOT/<case-name>/ (default wfl_cases/<case-name>/):  model, wavelets, forward run (real
# source, records the boundary on Omega, writes forward snapshots + point traces), reconstruction
# run (silent source, boundary injection only, snapshots + point traces), then
#   - point-receiver table(s): forward vs time-reversed reconstruction, all five fields
#   - leakage outside Omega per zone (reconstruction should be zero there)
#   - windows: forward / time-reversed reconstruction / difference snapshots, trace overlays
#
# Parameters come from $WFL_PARAMS (default wfl_params.sh). Optional settings a params file may define:
#   make_model()            shell function that creates model_cp.su model_cs.su model_ro.su
#                           (default: homogeneous model from cp0 cs0 ro0)
#   BND_TOP_FWD  (default 2) top boundary of the forward run: 1 = free surface, 2 = CPML
#   BND_TOP_RECON (default = BND_TOP_FWD) top boundary of the reconstruction run
#   ptx_list / ptz_list     comma-separated point receivers (default: single ptx / ptz)
#   PLOTCOMPS               fields to plot as trace overlays (default: vx vz txx tzz txz)
#   OUTROOT                 output root directory (default wfl_cases)
#   WFL_OPEN (default 0)    1 = single-sided (open) surface: only the top line z=wfl_z1, x=wfl_x1..wfl_x2
#   WFL_FWD  (default 0)    1 = forward-time-order (retarded) injection, no time reversal: the valid
#                           reconstruction of the downgoing field from an open line (implies the forward
#                           kernel; traces and snapshots are compared WITHOUT time reversal)
#   MUTE=1 (default 0)      mute the direct arrival on the recorded boundary/line data before injecting
#                           it (auto-pick per contour point from the recorded vz, cosine-taper mute);
#                           MUTE_THRESH (0.15), MUTE_PAD (0.05 s), MUTE_TAPER (0.05 s) tune the pick
#   SNAP_TSNAP1/SNAP_TSNAP2/SNAP_DTSNAP   override tsnap1/tsnap2/dtsnap from the params file (for
#                           imaging you generally want a much finer dtsnap than the default QC movies)
#   REFSUB=1 (default 0)    isolate the SCATTERED field by subtracting a second forward run's boundary
#                           recording, made through a background model with no reflectors (params file
#                           must define make_model_ref(); same source/geometry, no snapshots needed) --
#                           this is the rigorous alternative to MUTE's amplitude-picked taper. REFSUB is
#                           applied first, then MUTE (if also set) on the residual.
#   INJECT_FIELDS=vx,vz,... (default empty = all 5) zero out every recorded field NOT listed before
#                           injection (applied last, after REFSUB/MUTE) -- an ablation of which of the
#                           5 recorded fields actually drives the reconstruction. Recall for a horizontal
#                           line the split is: tzz,txz -> FORCE term (updates vx,vz); vx,vz -> MOMENT
#                           term (updates txx,tzz,txz); txx is never read regardless. So e.g.
#                           INJECT_FIELDS=vx,vz keeps only the moment/dipole term, INJECT_FIELDS=tzz,txz
#                           keeps only the force/monopole term -- the "vectorial vs tensorial" wavefield
#                           extrapolation distinction (Ravasi & Curtis 2013).

set -e
name=$1; TS=$2; C2=$3
[ -n "$name" ] && [ -n "$TS" ] && [ -n "$C2" ] || { echo "usage: $0 name tshift corner2" >&2; exit 1; }

DEMO=$(cd "$(dirname "$0")" && pwd)
BIN=$DEMO/../fdelmodc
export PATH=$DEMO/../../bin:${CWPROOT:-$HOME/cwp}/bin:$PATH   # OpenSource/bin + Seismic Unix (set CWPROOT if SU is not in ~/cwp)
[ -x $BIN ] || { echo "$BIN not found -- run make in fdelmodc/ first" >&2; exit 1; }
source ${WFL_PARAMS:-$DEMO/wfl_params.sh}

BND_TOP_FWD=${BND_TOP_FWD:-2}
BND_TOP_RECON=${BND_TOP_RECON:-$BND_TOP_FWD}
ptx_list=${ptx_list:-$ptx}
ptz_list=${ptz_list:-$ptz}
PLOTCOMPS=${PLOTCOMPS:-"vx vz txx tzz txz"}
OUTROOT=${OUTROOT:-wfl_cases}
WFL_OPEN=${WFL_OPEN:-0}
EDGESIGN=${EDGESIGN:-1}   # factor on the sources of the first/last point of the horizontal line(s); -1 = reversed polarity
BASE_NAME=$name
TAPER=${TAPER:-0}         # cosine ramp over this many points at each end of the horizontal line(s); 0 = off
CHANGED=0
if [ "$EDGESIGN" != "1" ]; then name=${name}_edge; CHANGED=1; fi   # separate folder so the baseline run is not overwritten
if [ "$TAPER" != "0" ]; then name=${name}_taper; CHANGED=1; fi
MUTE=${MUTE:-0}
MUTE_THRESH=${MUTE_THRESH:-0.15}
MUTE_PAD=${MUTE_PAD:-0.05}
MUTE_TAPER=${MUTE_TAPER:-0.05}
REFSUB=${REFSUB:-0}
if [ "$REFSUB" = "1" ]; then name=${name}_refsub; CHANGED=1; fi
if [ "$MUTE" = "1" ]; then name=${name}_mute; CHANGED=1; fi
INJECT_FIELDS=${INJECT_FIELDS:-}   # comma list of vx,vz,txx,tzz,txz to KEEP in the injected data (others
                                    # zeroed); default empty = keep all 5 (unchanged behaviour). For a
                                    # horizontal line, recall injectBoundary(Bwd).c's own split: recorded
                                    # tzz,txz drive the FORCE term (-> vx,vz update); recorded vx,vz drive
                                    # the MOMENT term (-> txx,tzz,txz update); recorded txx is never read.
                                    # So INJECT_FIELDS=vx,vz keeps only the moment term (stresses still get
                                    # updated, from vx,vz); INJECT_FIELDS=tzz,txz keeps only the force term.
if [ -n "$INJECT_FIELDS" ]; then name=${name}_inj-$(echo $INJECT_FIELDS | tr ',' '-'); CHANGED=1; fi
VECTORIAL=${VECTORIAL:-0}   # Ravasi & Curtis (2013) eq. 8 "vectorial" (common-practice) wavefield
                            # extrapolation: recorded vx,vz routed through the FORCE channel directly
                            # (-> vx,vz update), nothing injected into txx,tzz,txz -- as opposed to this
                            # code's default TENSORIAL injection (full traction+velocity, eq. 5). Distinct
                            # from INJECT_FIELDS=vx,vz, which keeps vx,vz routed through the MOMENT channel
                            # instead (a different, non-paper reduction of the tensorial formula).
if [ "$VECTORIAL" = "1" ]; then name=${name}_vectorial; CHANGED=1; fi
RECON_SHARP=${RECON_SHARP:-0}  # REFSUB=1 normally back-propagates through the SMOOTH/reference model
                                # (modelref, no reflector -- see RECON_MODEL below). RECON_SHARP=1 instead
                                # back-propagates the SAME scattered-field data through the TRUE/sharp model
                                # (model.su, reflector present), to reproduce the backscattering artifacts
                                # Rocha et al. (2015/2016) report when the reflector is present in the
                                # migration velocity -- same injected data, different reconstruction medium.
if [ "$RECON_SHARP" = "1" ]; then name=${name}_sharp; CHANGED=1; fi
if [ -n "$INJ_SOURCE" ]; then name=${name}_inj-$INJ_SOURCE; CHANGED=1; fi
tsnap1=${SNAP_TSNAP1:-$tsnap1}
tsnap2=${SNAP_TSNAP2:-$tsnap2}
dtsnap=${SNAP_DTSNAP:-$dtsnap}
WFL_FWD=${WFL_FWD:-0}
if [ "$WFL_FWD" = "1" ]; then BWD_EFF=0; TREV=0; else BWD_EFF=$BWD; TREV=1; fi

out=$DEMO/$OUTROOT/$name
mkdir -p $out && cd $out
echo "############ case '$name': wfl_edgesign=$EDGESIGN wfl_taper=$TAPER wfl_tshift=$TS wfl_corner2=$C2 wfl_bwd=$BWD_EFF wfl_open=$WFL_OPEN wfl_fwd=$WFL_FWD wfl_vectorial=$VECTORIAL top(fwd)=$BND_TOP_FWD top(recon)=$BND_TOP_RECON"
echo "############ params: ${WFL_PARAMS:-$DEMO/wfl_params.sh}   outputs: $out"

# ---------- model, wavelets ----------
if declare -F make_model >/dev/null; then
	make_model
else
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz orig=0,0 \
		cp0=$cp0 cs0=$cs0 ro0=$ro0 verbose=0
fi
makewave fp=$fp dt=$dt fmax=$fmax file_out=wav.su nt=$nt_wav t0=$t0 scale=1
# fdelmodc has no "no source" mode: use a wavelet multiplied by exactly zero (verified) for the
# reconstruction run, so it is driven only by the boundary injection
sugain scale=0.0 < wav.su > wav_dead.su
python3 -c "
import struct, sys
d=open('wav_dead.su','rb').read()[240:]
m=max(abs(v) for v in struct.unpack('<%df'%(len(d)//4),d))
sys.exit('FATAL: dead wavelet not silent') if m!=0.0 else print('dead wavelet verified silent')"

# ---------- common parameter strings ----------
base="ischeme=3 iorder=4 file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
	dtrcv=$dt npml=$npml R=1e-5 m=2.0 left=2 right=2 bottom=2 tmod=$tmod verbose=2"
snaps="xsnap1=0 xsnap2=$sizex zsnap1=0 zsnap2=$sizez dxsnap=$dx dzsnap=$dz \
	tsnap1=$tsnap1 tsnap2=$tsnap2 dtsnap=$dtsnap"
for f in $SNAPFIELDS; do snaps="$snaps sna_type_$f=1"; done
omega="wfl_x1=$wfl_x1 wfl_x2=$wfl_x2 wfl_z1=$wfl_z1 wfl_z2=$wfl_z2"
rtypes="rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1"
point="xrcva=$ptx_list zrcva=$ptz_list"

# ---------- 1) forward run ----------
$BIN $base top=$BND_TOP_FWD $snaps \
	file_src=wav.su xsrc=$xsrc zsrc=$zsrc src_type=$src_type_fwd src_orient=$src_orient \
	file_snap=snap_forw.su \
	wfl_recon=1 wfl_corner2=$C2 wfl_open=$WFL_OPEN $omega wfl_file=boundary.bin \
	file_rcv=rec_fwd.su $point $rtypes

# the recorded line/boundary data (what is injected later) as SU gathers, one trace per contour point:
# line_r{vx,vz,txx,tzz,txz}.su   (view with suximage/suxwigb, see wfl_bin2su.py)
python3 $DEMO/wfl_bin2su.py boundary.bin $dt line

# ---------- 1b) isolate the scattered field by subtracting a background-model recording (optional) ----------
INJ_FILE=boundary.bin
if [ "$REFSUB" = "1" ]; then
	declare -F make_model_ref >/dev/null || { echo "REFSUB=1 needs make_model_ref() in the params file" >&2; exit 1; }
	make_model_ref
	# also record full snapshots of this reference/background-model forward wavefield -- this is the
	# correct "source wavefield" S(x,z,t) for the RTM imaging condition (eq. 4 of Ravasi & Curtis 2013
	# uses the REFERENCE Green's function G^0 on the source side, not the true/total field), since the
	# true-model forward field's own internal reflections otherwise crosscorrelate with the receiver
	# wavefield near the source and swamp the image (see snap_forw for the true/total field instead)
	$BIN $base top=$BND_TOP_FWD $snaps \
		file_src=wav.su xsrc=$xsrc zsrc=$zsrc src_type=$src_type_fwd src_orient=$src_orient \
		file_snap=snap_ref.su \
		wfl_recon=1 wfl_corner2=$C2 wfl_open=$WFL_OPEN $omega wfl_file=boundary_ref.bin \
		file_cp=modelref_cp.su file_cs=modelref_cs.su file_den=modelref_ro.su \
		file_rcv=rec_fwd_ref.su $point $rtypes
	python3 - <<PY
import struct
a=open('boundary.bin','rb').read(); b=open('boundary_ref.bin','rb').read()
na,nta=struct.unpack('<ii',a[:8]); nb,ntb=struct.unpack('<ii',b[:8])
assert (na,nta)==(nb,ntb), 'true vs reference boundary geometry mismatch: (%d,%d) vs (%d,%d)'%(na,nta,nb,ntb)
hdr=8+4*4*na  # n,nt then xr,zr,nx,nz
va=struct.unpack('<%df'%(5*na*nta),a[hdr:hdr+4*5*na*nta])
vb=struct.unpack('<%df'%(5*nb*ntb),b[hdr:hdr+4*5*nb*ntb])
scat=[x-y for x,y in zip(va,vb)]
out=bytearray(); out+=struct.pack('<ii',na,nta); out+=a[8:hdr]; out+=struct.pack('<%df'%len(scat),*scat)
open('boundary_scat.bin','wb').write(out)
print('wrote boundary_scat.bin = boundary.bin - boundary_ref.bin (%d points x %d samples x 5 fields)'%(na,nta))
PY
	INJ_FILE=boundary_scat.bin
	python3 $DEMO/wfl_bin2su.py boundary_scat.bin $dt line_scat
	# INJ_SOURCE lets a nonlinear-imaging-condition workflow (Ravasi & Curtis 2013 eq. 9/21) pick which
	# recorded boundary data to back-propagate independently of REFSUB's own scattered-data default:
	# direct = boundary_ref.bin (the reference-model's own recording, i.e. direct arrivals only, for
	# G-tilde^0), total = boundary.bin (full recording, for the Phi^t term). Default (unset) keeps the
	# normal REFSUB=1 behaviour (scattered data) unchanged.
	case "$INJ_SOURCE" in
		direct) INJ_FILE=boundary_ref.bin ;;
		total)  INJ_FILE=boundary.bin ;;
	esac
fi
if [ "$MUTE" = "1" ]; then
	python3 - <<PY
import struct
DT=$dt; THRESH=$MUTE_THRESH; PAD=$MUTE_PAD; TAP=$MUTE_TAPER
d=open('$INJ_FILE','rb').read()
n,nt=struct.unpack('<ii',d[:8]); off=8
off+=4*4*n   # skip xr,zr,nx,nz
names=['vx','vz','txx','tzz','txz']
blocks={}
for nm in names:
    blocks[nm]=list(struct.unpack('<%df'%(n*nt),d[off:off+4*n*nt])); off+=4*n*nt
pad_s=int(round(PAD/DT)); tap_s=max(1,int(round(TAP/DT)))
mutef=[0]*n
for i in range(n):
    tr=blocks['vz'][i*nt:(i+1)*nt]
    m=max(abs(x) for x in tr) or 1.0
    onset=nt
    for k,x in enumerate(tr):
        if abs(x)>THRESH*m: onset=k; break
    cut=onset+pad_s
    mutef[i]=cut
    for nm in names:
        row=blocks[nm]; base=i*nt
        for k in range(min(cut,nt)): row[base+k]=0.0
        for k in range(cut,min(cut+tap_s,nt)):
            w=0.5*(1.0-__import__('math').cos(3.14159265*(k-cut)/tap_s))
            row[base+k]*=w
print('muted %d contour points: median onset+pad = %.3f s (min %.3f, max %.3f)'%(n,
    sorted(mutef)[n//2]*DT, min(mutef)*DT, max(mutef)*DT))
out=bytearray(); out+=struct.pack('<ii',n,nt); out+=d[8:8+4*4*n]
for nm in names: out+=struct.pack('<%df'%(n*nt),*blocks[nm])
open('boundary_mute.bin','wb').write(out)
PY
	INJ_FILE=boundary_mute.bin
	python3 $DEMO/wfl_bin2su.py boundary_mute.bin $dt line_mute
fi
if [ -n "$INJECT_FIELDS" ]; then
	KEEP="$INJECT_FIELDS" IN="$INJ_FILE" python3 - <<PY
import struct, os
keep = set(os.environ['KEEP'].split(','))
names = ['vx','vz','txx','tzz','txz']
d = open(os.environ['IN'],'rb').read()
n, nt = struct.unpack('<ii', d[:8])
hdr = 8 + 4*4*n
out = bytearray(); out += struct.pack('<ii', n, nt); out += d[8:hdr]
off = hdr
for nm in names:
	block = d[off:off+4*n*nt]; off += 4*n*nt
	out += block if nm in keep else struct.pack('<%df'%(n*nt), *([0.0]*(n*nt)))
open('boundary_inject.bin','wb').write(out)
print('kept %s, zeroed %s -> boundary_inject.bin'%(sorted(keep), sorted(set(names)-keep)))
PY
	INJ_FILE=boundary_inject.bin
	python3 $DEMO/wfl_bin2su.py boundary_inject.bin $dt line_inject
fi
echo "injecting: $INJ_FILE"

# ---------- 2) reconstruction run ----------
# REFSUB: back-propagate through the SAME smooth/background model used for the source-side reference
# wavefield (snap_ref), not the true model -- real migration never has access to the true (reflector)
# model for either side; using it here would let the reconstructed field interact with the very
# reflector being imaged a second time, which is not a fair test of the method.
RECON_MODEL=""
if [ "$REFSUB" = "1" ]; then
	if [ "$RECON_SHARP" = "1" ]; then RECON_MODEL="file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su"
	else RECON_MODEL="file_cp=modelref_cp.su file_cs=modelref_cs.su file_den=modelref_ro.su"; fi
fi
$BIN $base top=$BND_TOP_RECON $snaps \
	file_src=wav_dead.su xsrc=$xsrc zsrc=$zsrc src_type=1 src_orient=1 \
	file_snap=snap_recon.su \
	wfl_recon=2 wfl_bwd=$BWD_EFF wfl_tshift=$TS wfl_corner2=$C2 wfl_open=$WFL_OPEN wfl_fwd=$WFL_FWD wfl_edgesign=$EDGESIGN wfl_taper=$TAPER wfl_vectorial=$VECTORIAL $omega wfl_file=$INJ_FILE \
	file_rcv=rec_recon.su $point $rtypes $RECON_MODEL

# stress sign of the reconstruction: +1 with the backward kernel, -1 with the forward kernel
if [ "$BWD_EFF" = "1" ] || [ "$WFL_FWD" = "1" ]; then ssign=1; sop=diff; else ssign=-1; sop=sum; fi

# ---------- 3) point receiver(s): time-reverse the reconstructed trace, tables, residual files ----------
PTX="$ptx_list" PTZ="$ptz_list" SSIGN=$ssign TREV=$TREV CASE=$name python3 - <<'PY'
import struct, os
ssign=int(os.environ['SSIGN']); name=os.environ['CASE']; trev=int(os.environ['TREV'])
px=[v for v in os.environ['PTX'].split(',')]; pz=[v for v in os.environ['PTZ'].split(',')]
NP=len(px)
def readall(fn):
    d=open(fn,'rb').read(); ns=struct.unpack('<H',d[114:116])[0]; tr=240+4*ns
    assert len(d)//tr>=NP, '%s has %d traces, expected %d point receivers'%(fn,len(d)//tr,NP)
    return [(d[i*tr:i*tr+240], list(struct.unpack('<%df'%ns,d[i*tr+240:i*tr+240+4*ns]))) for i in range(NP)], ns
for p in range(NP):
    tag='' if NP==1 else '_p%d'%(p+1)
    print()
    print("case %s  point %d/%d (%s,%s):  forward vs %s"%(name,p+1,NP,px[p],pz[p],'time-reversed reconstruction' if trev else 'reconstruction (same time axis)'))
    print("%-4s %4s %8s %8s %12s"%('comp','lag','corr','k','resid/|fwd|'))
    for c in ['vx','vz','txx','tzz','txz']:
        F,ns=readall('rec_fwd_r%s.su'%c); R,ns2=readall('rec_recon_r%s.su'%c)
        h,f=F[p]; h2,r=R[p]
        if trev: r=r[::-1]
        if c[0]=='t': r=[ssign*x for x in r]
        open('fwd%s_%s.su'%(tag,c),'wb').write(h+struct.pack('<%df'%ns,*f))
        open('recon_trev%s_%s.su'%(tag,c),'wb').write(h2+struct.pack('<%df'%ns2,*r))
        if max(abs(x) for x in f)<1e-30:
            print("%-4s   (forward trace is identically zero here)"%c); continue
        best=None
        for lag in range(-6,7):
            a=f[max(0,lag):ns+min(0,lag)]; b=r[max(0,-lag):ns2-max(0,lag)]
            m=min(len(a),len(b)); a=a[:m]; b=b[:m]
            sab=sum(x*y for x,y in zip(a,b)); saa=sum(x*x for x in a); sbb=sum(y*y for y in b)
            if saa==0 or sbb==0: continue
            corr=sab/(saa*sbb)**0.5
            if best is None or corr>best[1]: best=(lag,corr,sab/saa,a,b)
        lag,corr,k,a,b=best
        res=(sum((y/k-x)**2 for x,y in zip(a,b))/sum(x*x for x in a))**0.5
        print("%-4s %4d %8.4f %8.4f %12.4f"%(c,lag,corr,k,res))
        # residual x10 on the forward time axis (best lag and amplitude factor applied), for the overlay plot
        rs=[(10.0*(r[t-lag]/k-f[t]) if 0<=t-lag<ns2 else 0.0) for t in range(ns)]
        open('resid10%s_%s.su'%(tag,c),'wb').write(h+struct.pack('<%df'%ns,*rs))
PY

# ---------- 4) leakage outside Omega (reconstruction should be 0 there) ----------
fl=""
for f in $SNAPFIELDS; do case $f in vx|vz) fl="$fl,s$f";; esac; done
if [ -n "$fl" ]; then
	WFL_OPEN=$WFL_OPEN python3 $DEMO/wfl_ablate_analyze.py $sizex $sizez $dx $dz $wfl_x1 $wfl_x2 $wfl_z1 $wfl_z2 ${fl#,} snap_recon:$name
fi

# ---------- 5) snapshots: reverse the reconstruction frame order, difference vs forward ----------
n1=$(python3 -c "print(int(round($sizez/$dz))+1)")
for f in $SNAPFIELDS; do
	s=s$f
	if [ "$TREV" = "1" ]; then
	python3 - <<PY
nx=int(round($sizex/$dx))+1; ns=int(round($sizez/$dz))+1; fs=nx*(240+ns*4)
d=open('snap_recon_$s.su','rb').read(); n=len(d)//fs
assert n*fs==len(d), 'snapshot size mismatch'
fr=[d[i*fs:(i+1)*fs] for i in range(n)]
open('snap_recon_trev_$s.su','wb').write(b''.join(reversed(fr)))
PY
	else
		ln -sf snap_recon_$s.su snap_recon_trev_$s.su   # forward-order: frame k already matches forward frame k
	fi
	case $f in vx|vz) op=diff;; *) op=$sop;; esac
	suop2 snap_forw_$s.su snap_recon_trev_$s.su op=$op > snap_diff_$s.su
done
# effect of the edge change (EDGESIGN / TAPER) on the wavefield: this reconstruction minus the baseline reconstruction
if [ "$CHANGED" = "1" ]; then
	for f in $SNAPFIELDS; do
		s=s$f
		if [ -e $DEMO/$OUTROOT/$BASE_NAME/snap_recon_trev_$s.su ]; then
			suop2 snap_recon_trev_$s.su $DEMO/$OUTROOT/$BASE_NAME/snap_recon_trev_$s.su op=diff > snap_effect_$s.su
		else
			echo "baseline $DEMO/$OUTROOT/$BASE_NAME not found: run the case with EDGESIGN=1 first to get the edge-effect difference"
		fi
	done
fi
echo
echo "files in $out:"
echo "  snap_forw_s*.su        forward snapshots"
echo "  snap_recon_trev_s*.su  reconstruction aligned with the forward frames (reversed frame order, or a link to snap_recon when WFL_FWD=1)"
echo "  snap_diff_s*.su        forward minus reconstruction (should be ~0 inside Omega)"
echo "  line_r<c>.su           recorded surface data, one trace per contour point (what is injected)"
echo "  fwd[_pN]_<c>.su / recon_trev[_pN]_<c>.su / resid10[_pN]_<c>.su   point traces (forward / time-reversed reconstruction / residual x10)"

# ---------- 6) windows ----------
if [ "$VIEW" = "1" ]; then
	for f in $SNAPFIELDS; do
		s=s$f
		clip=$(python3 -c "
import array
nx=int(round($sizex/$dx))+1; ns=int(round($sizez/$dz))+1
m=0.0
with open('snap_forw_$s.su','rb') as fh:
    while True:
        b=fh.read(240+ns*4)
        if not b: break
        a=array.array('f'); a.frombytes(b[240:]); m=max(m,max(a),-min(a))
print('%.6e'%($CLIPFRAC*m))")
		suxmovie < snap_forw_$s.su       clip=$clip n1=$n1 title="[$name] FORWARD $s"                              &
		suxmovie < snap_recon_trev_$s.su clip=$clip n1=$n1 title="[$name] RECONSTRUCTED (time-reversed) $s"       &
		suxmovie < snap_diff_$s.su       clip=$clip n1=$n1 title="[$name] forward - reconstructed $s (same clip)" &
	done
	NP=$(echo "$ptx_list" | tr ',' '\n' | wc -l | tr -d ' ')
	for p in $(seq 1 $NP); do
		if [ "$NP" = "1" ]; then tag=""; else tag="_p$p"; fi
		px=$(echo "$ptx_list" | cut -d, -f$p); pz=$(echo "$ptz_list" | cut -d, -f$p)
		for c in $PLOTCOMPS; do
			[ -e resid10${tag}_$c.su ] || continue
			# nplot=3 is REQUIRED: each trace header has ntr=1, so without it suxgraph draws only the first trace
			# 3 curves: forward (thick black), time-reversed reconstruction (thin red, drawn on top),
			# residual x10 (blue) -- forward and reconstruction agree to ~1%, so the residual shows the difference
			cat fwd${tag}_$c.su recon_trev${tag}_$c.su resid10${tag}_$c.su | suxgraph style=normal nplot=3 \
				linewidth=5,1,1 linecolor=0,2,4 \
				title="[$name] point ($px,$pz) $c: forward (black) / reconstructed time-reversed (red) / residual x10 (blue)" &
		done
	done
	# recorded line/boundary data (trace = contour point, x axis = point number along the contour)
	if [ "$CHANGED" = "1" ]; then
		for f in $SNAPFIELDS; do
			[ -e snap_effect_s$f.su ] && suxmovie < snap_effect_s$f.su clip=$(python3 -c "
import array
nx=int(round($sizex/$dx))+1; ns=int(round($sizez/$dz))+1
m=0.0
with open('snap_forw_s$f.su','rb') as fh:
    while True:
        b=fh.read(240+ns*4)
        if not b: break
        a=array.array('f'); a.frombytes(b[240:]); m=max(m,max(a),-min(a))
print('%.6e'%($CLIPFRAC*m))") n1=$n1 title="[$name] EFFECT (edgesign $EDGESIGN, taper $TAPER): reconstruction - baseline reconstruction s$f" &
		done
	fi
	for c in $PLOTCOMPS; do
		suximage < line_r$c.su perc=99 d1=$dt f1=0 d2=1 f2=1 xlabel="contour point" ylabel="time (s)" \
			title="[$name] RECORDED $c on the reconstruction surface" &
	done
	echo "windows open; close them to finish"
	wait
fi
