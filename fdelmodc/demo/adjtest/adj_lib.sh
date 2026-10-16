#!/bin/bash
# adj_lib.sh -- reusable shell functions for the elastic adjoint dot-product test.
# Glue only: every line below calls makemod, fdelmodc, or a Seismic Unix program.
# Source this after adj_params.sh.

# adj_build_model DIR
#   Builds model_cp.su / model_cs.su / model_ro.su in DIR: a smooth heterogeneous background
#   (see adj_params.sh) plus one exact 3x3 constant patch (rho0,Vp0,Vs0) at every P_x and P_y
#   point. One makemod call, one "def" background interface followed by one "elipse" interface
#   per point (each needs its own gradcp=0 gradcs=0 gradro=0, since makemod requires the count
#   of gradXX= occurrences to match the count of intt= occurrences once any are given at all).
adj_build_model() {
	local dir="$1"
	mkdir -p "$dir" && cd "$dir" || return 1

	local cmd=(makemod file_base=model.su
		sizex=$SIZEX sizez=$SIZEZ dx=$DX dz=$DZ orig=0,0
		cp0=$CP_LEFT cs0=$CS_LEFT ro0=$RO_LEFT
		intt=def x=0,$SIZEX z=0,0 cp=$CP_LEFT,$CP_RIGHT cs=$CS_LEFT,$CS_RIGHT ro=$RO_LEFT,$RO_RIGHT
		gradcp=$GRADCP gradcs=$GRADCS gradro=$GRADRO gradunit=0 above=0)

	local xs="$PX_X $PY_X"
	local zs="$PX_Z $PY_Z"
	set -- $xs
	local xarr=("$@")
	set -- $zs
	local zarr=("$@")
	local i
	for i in "${!xarr[@]}"; do
		cmd+=(intt=elipse x=${xarr[$i]} z=${zarr[$i]} var=$PATCH_R,$PATCH_R \
			cp=$VP0 cs=$VS0 ro=$RHO0 gradcp=0 gradcs=0 gradro=0)
	done

	"${cmd[@]}" verbose=1
}

# adj_make_signal SEED OUTFILE
#   Builds one nt=$NT, dt=$DT trace: band-limited (BP_F1..BP_F4 Hz) Gaussian noise, tapered
#   to exactly zero at both t=0 and t=(NT-1)*DT. Pipeline verified in scratchpad/modeltest:
#     sunull -> suaddnoise (plain, no internal f=/amps= -- see note below) -> sufilter ->
#     sutaper -> sumute (hard-zero fix for a confirmed sutaper end-taper off-by-one, see below).
#   Note: suaddnoise's own f=/amps= band-limiting option shells out internally to a second
#   sufilter via a temp file and failed ("efopen: fopen failed") in this environment; band-
#   limiting is done instead by a separate, top-level sufilter call, which works.
#   Note: sutaper's end-of-trace cosine ramp has an off-by-one (writes trace[nt-i] instead of
#   trace[nt-1-i], nt=tr.ns), so the true last sample is left at a small (~1e-5 relative)
#   residual instead of exact zero, while the start ramp is correct (trace[0] lands on the i=0
#   term and is exactly zero). The trailing sumute hard-zeroes exactly that one sample.
adj_make_signal() {
	local seed="$1" outfile="$2"
	local tmp
	tmp=$(mktemp -d "${TMPDIR:-/tmp}/adjsig.XXXXXX") || return 1

	sunull ntr=1 nt=$NT dt=$DT > "$tmp/null.su" &&
	suaddnoise < "$tmp/null.su" seed=$seed noise=gauss dt=$DT > "$tmp/noisy.su" &&
	sufilter  < "$tmp/noisy.su" f=$BP_F1,$BP_F2,$BP_F3,$BP_F4 amps=0,1,1,0 dt=$DT > "$tmp/filt.su" &&
	sutaper   < "$tmp/filt.su" ntr=1 tbeg=$TAPER_MS tend=$TAPER_MS taper=3 > "$tmp/tap.su" &&
	sumute    < "$tmp/tap.su" xmute=0 tmute=$TLAST key=tracl mode=1 ntaper=0 > "$outfile"
	local rc=$?
	rm -rf "$tmp"
	return $rc
}

# adj_csv "300 700 500" -> "300,700,500"   (pure bash, no external tool)
adj_csv() {
	local arr=($1) IFS=,
	echo "${arr[*]}"
}

# adj_seed BASE SIDE POINTIDX COMPIDX
#   Deterministic per-trace seed derived from a single test-group seed BASE, so every active
#   (point,component) pair gets its own independent random signal instead of a shared/correlated
#   one. SIDE: 1=P_x (source side), 2=P_y (receiver side). POINTIDX, COMPIDX: 1-based.
adj_seed() {
	echo $(( $1*10000 + $2*1000 + $3*10 + $4 ))
}

# adj_make_multisrc SIDE COMPIDX BASESEED OUTFILE
#   Builds a multi-trace SU file (one trace per P_x, or P_y, point, in the same left-to-right
#   order as PX_X/PX_Z or PY_X/PY_Z) of independent adj_make_signal traces, for use as
#   src_multiwav=1's file_src (source side) or as the random "y" adjoint-input vector
#   (receiver side, used later in Step 2's adjoint half).
adj_make_multisrc() {
	local side="$1" compidx="$2" baseseed="$3" outfile="$4"
	local pts; if [ "$side" = "1" ]; then pts="$PX_X"; else pts="$PY_X"; fi
	local parr=($pts) n=${#parr[@]}
	local tmp; tmp=$(mktemp -d "${TMPDIR:-/tmp}/adjmsrc.XXXXXX") || return 1
	: > "$tmp/all.su"
	local i seed
	for ((i=1; i<=n; i++)); do
		seed=$(adj_seed "$baseseed" "$side" "$i" "$compidx")
		adj_make_signal "$seed" "$tmp/p$i.su" || { rm -rf "$tmp"; return 1; }
		cat "$tmp/p$i.su" >> "$tmp/all.su"
	done
	mv "$tmp/all.su" "$outfile"
	rm -rf "$tmp"
}

# fdelmodc BUG, found while building the adjoint run: defineSource.c's "shift wavelet with
# +1/2 DeltaT" block (src.type>6 only, e.g. Fz=7 but NOT Fx=6) indexes shot.x[i]/shot.z[i] for
# i=0..wav.nx-1 (wav.nx = number of src_multiwav points), but shot.x/shot.z are calloc'd with
# only shot.n (=nshot, normally 1) elements (getParameters.c:652-653,598) -- an out-of-bounds
# heap read for i>=1. Confirmed by crash (Bus error) running src_type=7 with 5 src_multiwav
# points, and confirmed fixed by NOT using src_multiwav for these types: a single-point
# xsrc=/zsrc= run (wav.nx=1, shot.n=1, i only ever 0) completes cleanly. Per the task's "do not
# modify fdelmodc source" rule, _adj_run_multi below works around this for src_type>6 by running
# one single-source fdelmodc call per point and summing by linearity instead of one
# src_multiwav call -- mathematically identical, just N fdelmodc invocations instead of 1.

# _adj_run_multi COMPIDX SRCFILE SRC_X SRC_Z REC_X REC_Z MODELDIR OUTDIR TAG
#   Core worker shared by the forward and adjoint runs. SRCFILE is a multi-trace SU file (one
#   trace per point, in the same order as SRC_X/SRC_Z, e.g. from adj_make_multisrc or
#   adj_build_adjsrc). Always writes OUTDIR/rec_<name>_r{vx,vz,txx,tzz,txz}.su, regardless of
#   which internal path (multiwav vs. per-point-and-sum) was used.
_adj_run_multi() {
	local compidx="$1" srcfile="$2" srcx="$3" srcz="$4" recx="$5" recz="$6" modeldir="$7" outdir="$8" tag="$9"
	local name=${COMP_NAMES[$((compidx-1))]} stype=${COMP_SRC_TYPE[$((compidx-1))]}
	mkdir -p "$outdir" && cd "$outdir" || return 1
	local recx_csv=$(adj_csv "$recx") recz_csv=$(adj_csv "$recz")

	if [ "$stype" -le 6 ]; then
		local srcx_csv=$(adj_csv "$srcx") srcz_csv=$(adj_csv "$srcz")
		fdelmodc \
			file_cp=$modeldir/model_cp.su file_cs=$modeldir/model_cs.su file_den=$modeldir/model_ro.su \
			ischeme=3 iorder=4 dt=$DT dtrcv=$DT \
			file_src=$srcfile src_multiwav=1 src_type=$stype src_orient=1 src_injectionrate=1 \
			xsrca=$srcx_csv zsrca=$srcz_csv \
			file_rcv=rec_${name}.su xrcva=$recx_csv zrcva=$recz_csv \
			rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 rec_type_p=0 \
			rec_int_vx=0 rec_int_vz=0 rec_int_p=0 \
			left=$BND right=$BND top=$BND bottom=$BND \
			verbose=2 > log_${tag}_${name}.txt 2>&1
	else
		local sxarr=($srcx) szarr=($srcz) n=${#sxarr[@]} i
		: > "log_${tag}_${name}.txt"
		for ((i=1; i<=n; i++)); do
			suwind < "$srcfile" skip=$((i-1)) count=1 > "pt${i}_${name}.su"
			fdelmodc \
				file_cp=$modeldir/model_cp.su file_cs=$modeldir/model_cs.su file_den=$modeldir/model_ro.su \
				ischeme=3 iorder=4 dt=$DT dtrcv=$DT \
				file_src=pt${i}_${name}.su src_type=$stype src_orient=1 src_injectionrate=1 \
				xsrc=${sxarr[$((i-1))]} zsrc=${szarr[$((i-1))]} \
				file_rcv=rec_${name}_pt${i}.su xrcva=$recx_csv zrcva=$recz_csv \
				rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 rec_type_p=0 \
				rec_int_vx=0 rec_int_vz=0 rec_int_p=0 \
				left=$BND right=$BND top=$BND bottom=$BND \
				verbose=2 >> "log_${tag}_${name}.txt" 2>&1 || return 1
		done
		for r in vx vz txx tzz txz; do
			cp "rec_${name}_pt1_r${r}.su" "rec_${name}_r${r}.su"
			for ((i=2; i<=n; i++)); do
				suop2 "rec_${name}_r${r}.su" "rec_${name}_pt${i}_r${r}.su" op=sum > "rec_${name}_r${r}.su.tmp"
				mv "rec_${name}_r${r}.su.tmp" "rec_${name}_r${r}.su"
			done
		done
	fi
}

# adj_run_forward COMPIDX BASESEED MODELDIR OUTDIR [SNAP]
#   Runs the forward operator for ONE src_type (COMPIDX in 1..5, see COMP_NAMES/COMP_SRC_TYPE in
#   adj_params.sh) with an independent random band-limited signal at every P_x point, recording
#   all 5 field components at every P_y point. SNAP=1 also writes vx/vz/tzz snapshots (only
#   available on the src_multiwav path, i.e. src_type<=6; not used for Fz in this test anyway
#   since Step 2's forward run only turns snapshots on for illustration).
#   Boundary is rigid ($BND) on all four sides for Steps 2-3 (Step 4 sweeps this separately).
adj_run_forward() {
	local compidx="$1" baseseed="$2" modeldir="$3" outdir="$4" snap="${5:-0}"
	local name=${COMP_NAMES[$((compidx-1))]} stype=${COMP_SRC_TYPE[$((compidx-1))]}
	mkdir -p "$outdir" && cd "$outdir" || return 1
	adj_make_multisrc 1 "$compidx" "$baseseed" "src_${name}.su" || return 1

	if [ "$snap" = "1" ] && [ "$stype" -le 6 ]; then
		local xsrca=$(adj_csv "$PX_X") zsrca=$(adj_csv "$PX_Z")
		local xrcva=$(adj_csv "$PY_X") zrcva=$(adj_csv "$PY_Z")
		fdelmodc \
			file_cp=$modeldir/model_cp.su file_cs=$modeldir/model_cs.su file_den=$modeldir/model_ro.su \
			ischeme=3 iorder=4 dt=$DT dtrcv=$DT \
			file_src=src_${name}.su src_multiwav=1 src_type=$stype src_orient=1 src_injectionrate=1 \
			xsrca=$xsrca zsrca=$zsrca \
			file_rcv=rec_${name}.su xrcva=$xrcva zrcva=$zrcva \
			rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 rec_type_p=0 \
			rec_int_vx=0 rec_int_vz=0 rec_int_p=0 \
			left=$BND right=$BND top=$BND bottom=$BND \
			file_snap=snap_${name}.su tsnap1=0 tsnap2=$TLAST dtsnap=0.02 \
			sna_type_vx=1 sna_type_vz=1 sna_type_tzz=1 \
			verbose=2 > log_fwd_${name}.txt 2>&1
	else
		_adj_run_multi "$compidx" "$outdir/src_${name}.su" "$PX_X" "$PX_Z" "$PY_X" "$PY_Z" "$modeldir" "$outdir" fwd
	fi
}

# adj_build_y BASESEED OUTDIR
#   Builds y_{vx,vz,txx,tzz,txz}.su: one independent random band-limited signal per P_y point,
#   per component (10 traces each) -- the second, independent random test vector used for
#   a=<Fx,y> and b=<x,F^t y>. Uses side=2 in adj_seed()/adj_make_multisrc(), so these never
#   collide with the P_x-side "x" signals even for the same BASESEED.
adj_build_y() {
	local baseseed="$1" outdir="$2"
	mkdir -p "$outdir" && cd "$outdir" || return 1
	local names=(vx vz txx tzz txz) i
	for i in 1 2 3 4 5; do
		adj_make_multisrc 2 "$i" "$baseseed" "y_${names[$((i-1))]}.su" || return 1
	done
}

# adj_build_adjsrc YDIR OUTDIR
#   From YDIR/y_{vx,vz,txx,tzz,txz}.su (at P_y), builds the 5 src_multiwav injection files
#   src_{Fx,Fz,Txx,Tzz,Txz}.su ready to run as the adjoint sources AT P_y: apply the combined
#   M^{-1}*(1/S_c) literal scale (AS_* in adj_params.sh), then time-reverse (suflip flip=3).
#   NOTE: no extra half-step v/tau realignment is applied yet -- per the Step 0 report this is
#   an open item to pin down from the diagnostic eps pattern, not asserted a priori.
adj_build_adjsrc() {
	local ydir="$1" outdir="$2"
	mkdir -p "$outdir" && cd "$outdir" || return 1

	sugain < "$ydir/y_vx.su"  scale=$AS_FX | suflip flip=3 > src_Fx.su
	sugain < "$ydir/y_vz.su"  scale=$AS_FZ | suflip flip=3 > src_Fz.su
	sugain < "$ydir/y_txz.su" scale=$AS_TXZ | suflip flip=3 > src_Txz.su

	sugain < "$ydir/y_txx.su" scale=$AS_TXX_SELF  > txx_self.su
	sugain < "$ydir/y_tzz.su" scale=$AS_TXX_CROSS > tzz_cross.su
	suop2 txx_self.su tzz_cross.su op=sum | suflip flip=3 > src_Txx.su

	sugain < "$ydir/y_txx.su" scale=$AS_TZZ_CROSS > txx_cross.su
	sugain < "$ydir/y_tzz.su" scale=$AS_TZZ_SELF  > tzz_self.su
	suop2 txx_cross.su tzz_self.su op=sum | suflip flip=3 > src_Tzz.su

	rm -f txx_self.su tzz_cross.su txx_cross.su tzz_self.su
}

# adj_run_adjoint_component COMPIDX ADJSRCDIR MODELDIR OUTDIR [SNAP]
#   Runs ONE src_type's adjoint pass: injects ADJSRCDIR/src_<name>.su (already time-reversed and
#   prescaled) AT the P_y points, records all 5 field components AT the P_x points -- source and
#   receiver roles swapped relative to adj_run_forward, same model/boundary/dt. Same fdelmodc
#   src_multiwav+src_type>6 bug as adj_run_forward (see the note above _adj_run_multi) applies
#   here too -- delegates to _adj_run_multi for src_type>6 instead of calling fdelmodc directly.
#   SNAP=1 also writes vx/vz/tzz snapshots (src_type<=6 only, same limitation as adj_run_forward).
adj_run_adjoint_component() {
	local compidx="$1" adjsrcdir="$2" modeldir="$3" outdir="$4" snap="${5:-0}"
	local name=${COMP_NAMES[$((compidx-1))]} stype=${COMP_SRC_TYPE[$((compidx-1))]}
	mkdir -p "$outdir" && cd "$outdir" || return 1

	if [ "$snap" = "1" ] && [ "$stype" -le 6 ]; then
		local xsrca=$(adj_csv "$PY_X") zsrca=$(adj_csv "$PY_Z")
		local xrcva=$(adj_csv "$PX_X") zrcva=$(adj_csv "$PX_Z")
		fdelmodc \
			file_cp=$modeldir/model_cp.su file_cs=$modeldir/model_cs.su file_den=$modeldir/model_ro.su \
			ischeme=3 iorder=4 dt=$DT dtrcv=$DT \
			file_src=$adjsrcdir/src_${name}.su src_multiwav=1 src_type=$stype src_orient=1 src_injectionrate=1 \
			xsrca=$xsrca zsrca=$zsrca \
			file_rcv=rec_${name}.su xrcva=$xrcva zrcva=$zrcva \
			rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 rec_type_p=0 \
			rec_int_vx=0 rec_int_vz=0 rec_int_p=0 \
			left=$BND right=$BND top=$BND bottom=$BND \
			file_snap=snap_${name}.su tsnap1=0 tsnap2=$TLAST dtsnap=0.02 \
			sna_type_vx=1 sna_type_vz=1 sna_type_tzz=1 \
			verbose=2 > log_adj_${name}.txt 2>&1
	else
		_adj_run_multi "$compidx" "$adjsrcdir/src_${name}.su" "$PY_X" "$PY_Z" "$PX_X" "$PX_Z" "$modeldir" "$outdir" adj
	fi
}

# adj_apply_M REVDIR OUTDIR
#   REVDIR/rev_{vx,vz,txx,tzz,txz}.su are the summed adjoint recordings at P_x, already
#   time-reversed a second time (the "v~, tau~" of the Step 0 report). Applies the final M
#   rescale (literal CINV_*/RHO0 constants) to produce wdag_{vx,vz,txx,tzz,txz}.su = F^t y at P_x.
adj_apply_M() {
	local revdir="$1" outdir="$2"
	mkdir -p "$outdir" && cd "$outdir" || return 1

	sugain < "$revdir/rev_vx.su" scale=$RHO0 > wdag_vx.su
	sugain < "$revdir/rev_vz.su" scale=$RHO0 > wdag_vz.su

	sugain < "$revdir/rev_txx.su" scale=-$CINV_DIAG > a1.su
	sugain < "$revdir/rev_tzz.su" scale=$CINV_OFFDIAG > a2.su
	suop2 a1.su a2.su op=sum > wdag_txx.su

	sugain < "$revdir/rev_txx.su" scale=$CINV_OFFDIAG > b1.su
	sugain < "$revdir/rev_tzz.su" scale=-$CINV_DIAG > b2.su
	suop2 b1.su b2.su op=sum > wdag_tzz.su

	sugain < "$revdir/rev_txz.su" scale=-$CINV_SHEAR > wdag_txz.su

	rm -f a1.su a2.su b1.su b2.su
}

# adj_dotprod FILE_A FILE_B
#   Prints the scalar plain-L2 dot product sum(A.*B) over all traces and samples of two
#   matching-shape SU files, using the recipe verified in the Step 0 report:
#   suop2 op=prod -> sushw cdp -> sustack normpow=0 (sum traces) -> transpose (suflip flip=0) ->
#   sushw cdp -> sustack normpow=0 (sum samples) -> suascii bare=1.
adj_dotprod() {
	local fa="$1" fb="$2"
	local tmp; tmp=$(mktemp -d "${TMPDIR:-/tmp}/adjdot.XXXXXX") || return 1
	suop2 "$fa" "$fb" op=prod > "$tmp/prod.su"
	sushw < "$tmp/prod.su" key=cdp a=1 > "$tmp/prod_h.su"
	sustack < "$tmp/prod_h.su" key=cdp normpow=0 > "$tmp/s1.su"
	suflip < "$tmp/s1.su" flip=0 > "$tmp/tr.su"
	sushw < "$tmp/tr.su" key=cdp a=1 > "$tmp/tr_h.su"
	sustack < "$tmp/tr_h.su" key=cdp normpow=0 > "$tmp/total.su"
	suascii < "$tmp/total.su" bare=1 2>/dev/null
	rm -rf "$tmp"
}
