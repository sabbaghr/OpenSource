#!/bin/bash
# Reconstruct using a PHYSICALLY REALISTIC injection: vx,vz from a geophone coupled to the solid
# (z=505, wfl_seafloor1's line) spliced with tzz,txz from a hydrophone in the water (z=495,
# wfl_seafloor3's line; tzz=-p, txz=0 there since it's a fluid). txx is dropped -- the horizontal-face
# injection formula never reads it. Both the geophone (GEO_DIR) and hydrophone (HYDRO_DIR) cases must
# already have been run (wfl_seafloor1_refl_timereversed_refsub_taper.scr and
# wfl_seafloor3_hydro_timereversed_refsub_taper.scr, same TAPER).
# Output: $GEO_DIR/snap_recon_realistic_s*.su (aligned with snap_forw the same way snap_recon_trev is).
set -e
DEMO=$(cd "$(dirname "$0")" && pwd)
BIN=$DEMO/../fdelmodc
export PATH=$DEMO/../../bin:${CWPROOT:-$HOME/cwp}/bin:$PATH   # OpenSource/bin + Seismic Unix (set CWPROOT if SU is not in ~/cwp)

GEO_DIR=$DEMO/wfl_cases_seafloor_refl/timereversed_taper_refsub
HYDRO_DIR=$DEMO/wfl_cases_seafloor_refl/timereversed_hydro_taper_refsub
[ -f $GEO_DIR/boundary_scat.bin ]   || { echo "missing $GEO_DIR/boundary_scat.bin -- run wfl_seafloor1_refl_timereversed_refsub_taper.scr first" >&2; exit 1; }
[ -f $HYDRO_DIR/boundary_scat.bin ] || { echo "missing $HYDRO_DIR/boundary_scat.bin -- run wfl_seafloor3_hydro_timereversed_refsub_taper.scr first" >&2; exit 1; }

cd $GEO_DIR
python3 $DEMO/wfl_splice_geo_hydro.py boundary_scat.bin $HYDRO_DIR/boundary_scat.bin boundary_hybrid_scat.bin
python3 $DEMO/wfl_bin2su.py boundary_hybrid_scat.bin 0.0004 line_hybrid_scat

# same parameters wfl_seafloor1 used (WFL_PARAMS=wfl_params_seafloor_refl.sh, WFL_Z1=505 default)
source $DEMO/wfl_params_seafloor_refl.sh
# match snap_recon_trev's frame timing so the two reconstructions can be compared frame-by-frame
tsnap1=${SNAP_TSNAP1:-$tsnap1}; tsnap2=${SNAP_TSNAP2:-$tsnap2}; dtsnap=${SNAP_DTSNAP:-$dtsnap}

base="ischeme=3 iorder=4 file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
	dtrcv=$dt npml=$npml R=1e-5 m=2.0 left=2 right=2 bottom=2 tmod=$tmod verbose=2"
snaps="xsnap1=0 xsnap2=$sizex zsnap1=0 zsnap2=$sizez dxsnap=$dx dzsnap=$dz \
	tsnap1=$tsnap1 tsnap2=$tsnap2 dtsnap=$dtsnap"
for f in ${SNAPFIELDS:-"vx vz"}; do snaps="$snaps sna_type_$f=1"; done
omega="wfl_x1=$wfl_x1 wfl_x2=$wfl_x2 wfl_z1=$wfl_z1 wfl_z2=$wfl_z2"
rtypes="rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1"
point="xrcva=$ptx_list zrcva=$ptz_list"

$BIN $base top=$BND_TOP_RECON $snaps \
	file_src=wav_dead.su xsrc=$xsrc zsrc=$zsrc src_type=1 src_orient=1 \
	file_snap=snap_recon_realistic.su \
	wfl_recon=2 wfl_bwd=1 wfl_tshift=1 wfl_corner2=0 wfl_open=$WFL_OPEN wfl_fwd=0 wfl_edgesign=1 wfl_taper=${TAPER:-300} $omega wfl_file=boundary_hybrid_scat.bin \
	file_rcv=rec_recon_realistic.su $point $rtypes file_cp=modelref_cp.su file_cs=modelref_cs.su file_den=modelref_ro.su

# [BUG FIX] wfl_bwd=1 reconstruction runs in REVERSED chronological order, same as wfl_case.sh's own
# snap_recon.su -- wfl_case.sh reverses the frame order into snap_recon_trev.su before it is ever
# compared/plotted against snap_forw.su (see its step 5); this standalone script must do the same or
# frame k here will NOT be the same instant as frame k of snap_forw/snap_recon_trev.
nx=$(python3 -c "print(int(round($sizex/$dx))+1)"); ns=$(python3 -c "print(int(round($sizez/$dz))+1)")
for f in ${SNAPFIELDS:-"vx vz"}; do
	python3 - <<PY
nx=$nx; ns=$ns; fs=nx*(240+ns*4)
d=open('snap_recon_realistic_s$f.su','rb').read(); n=len(d)//fs
assert n*fs==len(d), 'snapshot size mismatch'
fr=[d[i*fs:(i+1)*fs] for i in range(n)]
open('snap_recon_realistic_trev_s$f.su','wb').write(b''.join(reversed(fr)))
PY
done

echo "wrote $GEO_DIR/snap_recon_realistic_trev_s*.su (geophone vx,vz @ z=505 + hydrophone tzz,txz @ z=495, time-aligned with snap_forw)"
