#!/bin/bash
# wfl_cig_rocha.sh -- multi-shot energy-norm RTM on the Rocha et al. (2016) reflector geometry, then
# shot-index common image gathers (CIGs).
#
# For each source position in SHOTS: run wfl_rocha1_refl_timereversed_refsub_taper.scr (REFSUB, taper,
# reverse-time reconstruction through the SMOOTH reference model), image with wfl_rtm_energy_stress.py
# (--src snap_ref, model=modelref), keep the small image files and DELETE the ~14 GB of snapshots /
# boundary binaries (disk cannot hold more than a few shots' wavefields). Finally wfl_cig.py extracts
# the column at each CIG_X from every shot image -> cig_x<x0>_{full,atten}.su/.png and the stack.
#
# Outputs: demo/wfl_cases_rocha/cig_smooth/shot_x<xsrc>/reflector_abs_taper_refsub/energy_smooth_*.su
#          demo/wfl_cases_rocha/cig_smooth/cig_*.{su,png}, stack_*.{su,png}
# Options: SHOTS="2000 3500 5000 6500 8000" CIG_X="3000 5000 7000" KEEP_SNAPS=0 SKIP_DONE=1
#          JOBS=1  shots run concurrently (fdelmodc and the imaging are single-threaded; the limit is DISK:
#                  ~14 GB peak per shot until its snapshots are deleted). JOBS>1 logs each shot to
#                  $ROOT/shot_x<xs>.log.
set -e
DEMO=$(cd "$(dirname "$0")" && pwd)
SHOTS=${SHOTS:-"2000 3500 5000 6500 8000"}
CIG_X=${CIG_X:-"3000 5000 7000"}
KEEP_SNAPS=${KEEP_SNAPS:-0}
SKIP_DONE=${SKIP_DONE:-1}
JOBS=${JOBS:-1}
ROOT=wfl_cases_rocha/cig_smooth
CASE=reflector_abs_taper_refsub
nx=1001; nz=251; dx=10; dz=10     # wfl_params_rocha_refl.sh: 10000 x 2500 m at 10 m

run_shot() {
	local xs=$1 d=$DEMO/$ROOT/shot_x$1/$CASE
	echo "== shot x=$xs  ($(date +%H:%M:%S))"
	XSRC=$xs ROCHA_OUTROOT=$ROOT/shot_x$xs SNAPFIELDS="vx vz txx tzz txz" VIEW=0 \
		$DEMO/wfl_rocha1_refl_timereversed_refsub_taper.scr
	# stale-stress guard (see project memory): every source/receiver snapshot must come from THIS run
	for f in snap_ref snap_recon_trev; do for c in vx vz txx tzz txz; do
		[ $d/${f}_s$c.su -nt $d/model_cp.su ] || { echo "FATAL: $d/${f}_s$c.su older than this run's model" >&2; return 1; }
	done; done
	echo "== imaging shot x=$xs  ($(date +%H:%M:%S))"
	python3 $DEMO/wfl_rtm_energy_stress.py $nx $nz $dx $dz $d modelref $d/energy_smooth --src snap_ref
	if [ "$KEEP_SNAPS" != "1" ]; then rm -f $d/snap_*.su $d/boundary*.bin; fi
	echo "== done shot x=$xs  ($(date +%H:%M:%S))"
}

pids=()
for xs in $SHOTS; do
	if [ "$SKIP_DONE" = "1" ] && [ -e $DEMO/$ROOT/shot_x$xs/$CASE/energy_smooth_full.su ]; then echo "== shot x=$xs already imaged, skipping"; continue; fi
	if [ "$JOBS" = "1" ]; then run_shot $xs; continue; fi
	while [ $(jobs -rp | wc -l) -ge $JOBS ]; do sleep 10; done   # bash 3.2 (macOS): no wait -n
	mkdir -p $DEMO/$ROOT
	( set -e; run_shot $xs ) > $DEMO/$ROOT/shot_x$xs.log 2>&1 &
	pids+=($!); echo "== launched shot x=$xs (pid $!, log $ROOT/shot_x$xs.log)  ($(date +%H:%M:%S))"
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail = 0 ] || { echo "FATAL: at least one shot failed, see $ROOT/shot_x*.log" >&2; exit 1; }

python3 $DEMO/wfl_cig.py $DEMO/$ROOT $CASE/energy_smooth $dx $dz "$SHOTS" "$CIG_X"
