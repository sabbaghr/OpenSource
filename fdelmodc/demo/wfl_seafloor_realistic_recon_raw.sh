#!/bin/bash
# Same PHYSICALLY REALISTIC injection as wfl_seafloor_realistic_recon.sh (vx,vz from a geophone at
# z=505 spliced with tzz,txz from a hydrophone at z=495), but with NOTHING subtracted: injects the RAW
# recorded field (direct wave, seafloor reflection/multiples, AND the target reflection all still in
# it), through the TRUE model (no migration/reference model, since there is nothing to migrate against
# here -- this is just "what would injecting the actual full recording reconstruct").
# Output: $GEO_DIR/snap_recon_realistic_raw_trev_s*.su (time-aligned with snap_forw).
set -e
DEMO=$(cd "$(dirname "$0")" && pwd)
BIN=$DEMO/../fdelmodc
export PATH=$DEMO/../../bin:${CWPROOT:-$HOME/cwp}/bin:$PATH   # OpenSource/bin + Seismic Unix (set CWPROOT if SU is not in ~/cwp)

GEO_DIR=$DEMO/wfl_cases_seafloor_refl/timereversed_taper_refsub
HYDRO_DIR=$DEMO/wfl_cases_seafloor_refl/timereversed_hydro_taper_refsub
[ -f $GEO_DIR/boundary.bin ]   || { echo "missing $GEO_DIR/boundary.bin -- run wfl_seafloor1_refl_timereversed_refsub_taper.scr first" >&2; exit 1; }
[ -f $HYDRO_DIR/boundary.bin ] || { echo "missing $HYDRO_DIR/boundary.bin -- run wfl_seafloor3_hydro_timereversed_refsub_taper.scr first" >&2; exit 1; }

cd $GEO_DIR
python3 $DEMO/wfl_splice_geo_hydro.py boundary.bin $HYDRO_DIR/boundary.bin boundary_hybrid_raw.bin
python3 $DEMO/wfl_bin2su.py boundary_hybrid_raw.bin 0.0004 line_hybrid_raw

source $DEMO/wfl_params_seafloor_refl.sh
tsnap1=${SNAP_TSNAP1:-$tsnap1}; tsnap2=${SNAP_TSNAP2:-$tsnap2}; dtsnap=${SNAP_DTSNAP:-$dtsnap}

base="ischeme=3 iorder=4 file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
	dtrcv=$dt npml=$npml R=1e-5 m=2.0 left=2 right=2 bottom=2 tmod=$tmod verbose=2"
snaps="xsnap1=0 xsnap2=$sizex zsnap1=0 zsnap2=$sizez dxsnap=$dx dzsnap=$dz \
	tsnap1=$tsnap1 tsnap2=$tsnap2 dtsnap=$dtsnap"
for f in ${SNAPFIELDS:-"vx vz"}; do snaps="$snaps sna_type_$f=1"; done
omega="wfl_x1=$wfl_x1 wfl_x2=$wfl_x2 wfl_z1=$wfl_z1 wfl_z2=$wfl_z2"
rtypes="rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1"
point="xrcva=$ptx_list zrcva=$ptz_list"

# NOTE: no file_cp=modelref_* override here -- reconstruct through the TRUE model, since nothing was
# subtracted (there is no "migration model" role to play; injecting the raw field through the smoothed
# model would be an inconsistent, arbitrary choice).
$BIN $base top=$BND_TOP_RECON $snaps \
	file_src=wav_dead.su xsrc=$xsrc zsrc=$zsrc src_type=1 src_orient=1 \
	file_snap=snap_recon_realistic_raw.su \
	wfl_recon=2 wfl_bwd=1 wfl_tshift=1 wfl_corner2=0 wfl_open=$WFL_OPEN wfl_fwd=0 wfl_edgesign=1 wfl_taper=${TAPER:-300} $omega wfl_file=boundary_hybrid_raw.bin \
	file_rcv=rec_recon_realistic_raw.su $point $rtypes

# [see wfl_seafloor_realistic_recon.sh] wfl_bwd=1 runs in REVERSED chronological order -- reverse the
# frame order to align frame k with real time tsnap1+k*dtsnap, same as snap_forw/snap_recon_trev.
nx=$(python3 -c "print(int(round($sizex/$dx))+1)"); ns=$(python3 -c "print(int(round($sizez/$dz))+1)")
for f in ${SNAPFIELDS:-"vx vz"}; do
	python3 - <<PY
nx=$nx; ns=$ns; fs=nx*(240+ns*4)
d=open('snap_recon_realistic_raw_s$f.su','rb').read(); n=len(d)//fs
assert n*fs==len(d), 'snapshot size mismatch'
fr=[d[i*fs:(i+1)*fs] for i in range(n)]
open('snap_recon_realistic_raw_trev_s$f.su','wb').write(b''.join(reversed(fr)))
PY
done

echo "wrote $GEO_DIR/snap_recon_realistic_raw_trev_s*.su (geophone vx,vz @ z=505 + hydrophone tzz,txz @ z=495, RAW/no subtraction, true model, time-aligned with snap_forw)"
