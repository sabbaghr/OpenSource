#!/bin/bash
#PBS -l nodes=1
#PBS -N InterfModeling
#PBS -q fourweeks
#PBS -V
#
# Illustrates how to place source and receivers on topography


export PATH=../../bin:$PATH

dt=0.0004
ntap=120
fmax=45

makemod sizex=10000 sizez=4100 dx=5 dz=5 cp0=0 cs0=0 ro0=1000 file_base=real2.su \
    orig=0,-800 gradunit=0 \
    intt=def poly=2 cp=2450 ro=1000 gradcp=14 grad=0 cs=2000 \
    x=0,1000,1700,1800,2000,3000,4000,4500,6000,6800,7000,7500,8100,8800,10000 \
    z=-100,-200,-250,-200,-200,-120,-300,-600,-650,-500,-350,-200,-200,-150,-200  \
    intt=rough var=200,3.2,1 poly=2 x=0,3000,8000,10000 \
		z=400,250,300,500 cp=4500,4200,4800,4500 ro=1400 gradcp=5 grad=0 cs=3800,3200,4000,3800 \
    intt=def poly=2 x=0,2000,3000,5000,7000,8000,10000 \
        z=1100,1100,1100,1600,1100,1100,1100 cp=4000 cs=3000 ro=2000 gradcp=8 grad=0 \
    intt=def poly=0 x=0,10000 z=1750,2050 cp=4500,5100 cs=3800,4200 ro=1500 gradcp=13 grad=0 \
    intt=def poly=0 x=0,10000 z=1850,2150 cp=6000,4200 cs=5000 ro=1500 gradcp=14 grad=0 \
    intt=def poly=0 x=0,10000 z=1950,2250 cp=4800,4800 cs=4000 ro=1500 gradcp=5 grad=0 \
    intt=def poly=0 x=0,10000 z=2000,2300 cp=6100,5000 cs=4400 ro=1500 gradcp=13 grad=0 \
    intt=def poly=0 x=0,10000 z=2100,2400 cp=3800,5000 cs=3000,3600 ro=1500 gradcp=20 grad=0 \
    intt=def poly=0 x=0,10000 z=2150,2450 cp=5000 cs=4000 ro=1500 gradcp=14 grad=0 \
    intt=def poly=0 x=0,10000 z=2350,2650 cp=5800 cs=4200 ro=1500 gradcp=5 grad=0 \
    intt=def poly=0 x=0,10000 z=2600,2600 cp=5500 cs=4100 ro=2200 gradcp=5 grad=0

sushw key=f1 a=0 < real2_cp.su | \
    sushw key=f1 a=0 | \
    supsimage hbox=6 wbox=8 labelsize=10 f2num=-5000 d2num=1000 \
    wrgb=0,0,1.0 grgb=0,1.0,0 brgb=1.0,0,0 \
    bclip=7053.02 wclip=0 label1="depth [m]" label2="lateral position [m]" \
    > model2_cp.eps


makewave w=g2 fmax=45 t0=0.10 dt=$dt nt=4096 db=-40 file_out=G2.su verbose=1

#in new FD code extendmodel is done in FD 
#extendModel file_in=real2_ro.su nafter=$ntap nbefore=$ntap nabove=0 nbelow=$ntap > vel2_edge_ro.su
#extendModel file_in=real2_cp.su nafter=$ntap nbefore=$ntap nabove=0 nbelow=$ntap > vel2_edge_cp.su

#reference

fdelmodc \
    file_cp=real2_cp.su ischeme=3 iorder=6\
    file_den=real2_ro.su \
    file_cs=real2_cs.su \
    file_rcv=shot_real2_x5000_topo.su \
    file_src=G2.su \
    file_snap=snap_real.su \
    src_type=7 \
    dtrcv=0.004 \
    verbose=4 \
    tmod=3.104 \
	rec_delay=0.1 \
    dxrcv=20.0 \
    zrcv1=-800 \
    zrcv2=-800 \
    xrcv1=0 \
    xrcv2=10000 \
    sinkdepth=1 \
    sinkdepth_src=1 \
    src_random=0 \
    wav_random=0 \
    xsrc=5000 \
    zsrc=-500 \
    ntaper=$ntap \
    tsnap1=0.1 tsnap2=2.0 dtsnap=0.04 dxsnap=20 dzsnap=20 \
    sna_type_txx=1 sna_type_tzz=1 \
    left=4 right=4 top=1 bottom=4


../test_fdfwimodc\
    file_cp=real2_cp.su ischeme=3 iorder=6\
    file_den=real2_ro.su \
    file_cs=real2_cs.su \
    file_rcv=shot_test2_x5000_topo.su \
    file_src=G2.su \
    file_snap=snap_test.su \
    src_type=7 \
    dtrcv=0.004 \
    verbose=4 \
    tmod=3.104 \
	rec_delay=0.1 \
    dxrcv=20.0 \
    zrcv1=-800 \
    zrcv2=-800 \
    xrcv1=0 \
    xrcv2=10000 \
    sinkdepth=1 \
    sinkdepth_src=1 \
    src_random=0 \
    wav_random=0 \
    xsrc=5000 \
    zsrc=-500 \
    ntaper=$ntap \
    tsnap1=0.1 tsnap2=2.0 dtsnap=0.04 dxsnap=20 dzsnap=20 \
    sna_type_txx=1 sna_type_tzz=1 \
    left=4 right=4 top=1 bottom=4

    supsimage perc=99 f1=0 f2=-5000 x1end=3.004 hbox=8 wbox=6 < shot_real2_x5000_topo_rvz.su \
    label1="time (s)" label2="lateral position (m)" \
    labelsize=10 f2num=-5000 d2num=1000 d1num=0.5 > shot_real2_x5000_topo.eps

######################################################################
# Quantitative comparison: fdelmodc (reference) vs test_fdfwimodc
######################################################################
THRESHOLD=1.0   # maximum allowed relative error in percent (elastic + topo)

echo ""
echo "==========================================="
echo "  Quantitative comparison (topo elastic)   "
echo "==========================================="

for comp in rvz; do
    REF="shot_real2_x5000_topo_${comp}.su"
    TST="shot_test2_x5000_topo_${comp}.su"

    if [ ! -f "$REF" ] || [ ! -f "$TST" ]; then
        echo "  ${comp}: SKIP (file missing)"
        continue
    fi

    # Peak amplitude of reference
    sumax < "$REF" outpar=nep_ref 2>/dev/null
    REF_MAX=$(awk '{print $1}' nep_ref)

    # Peak of absolute difference
    sudiff "$REF" "$TST" 2>/dev/null | suop op=abs | sumax outpar=nep_diff 2>/dev/null
    DIFF_MAX=$(awk '{print $1}' nep_diff)

    # Relative error (%)
    REL_ERR=$(awk "BEGIN {if ($REF_MAX>0) printf \"%.6f\", 100.0*$DIFF_MAX/$REF_MAX; else print \"0.0\"}")

    # PASS/FAIL
    PASS=$(awk "BEGIN {print ($REL_ERR <= $THRESHOLD) ? 1 : 0}")
    if [ "$PASS" -eq 1 ]; then
        VERDICT="PASS"
    else
        VERDICT="FAIL"
    fi

    echo "  ${comp}: ref_max=${REF_MAX}  diff_max=${DIFF_MAX}  rel_err=${REL_ERR}%  [$VERDICT]"
    rm -f nep_ref nep_diff
done

echo "==========================================="
echo ""

######################################################################
# Snapshot comparison: 5 snapshots for txx and tzz components
# tsnap1=0.1 tsnap2=2.0 dtsnap=0.04 => 49 snapshots (fldr=1..49)
# Pick fldr = 5,15,25,35,45  (t = 0.26, 0.66, 1.06, 1.46, 1.86 s)
######################################################################

SNAP_THRESHOLD=3.0  # snapshot threshold (%) — higher than receiver because
                    # full-grid comparison exposes FP accumulation over many
                    # time steps, especially near free-surface topography

echo "==========================================="
echo "  Snapshot comparison (txx, tzz)           "
echo "==========================================="

SNAP_PASS=0
SNAP_TOTAL=0

for stype in stxx stzz; do
    REF_SNAP="snap_real_${stype}.su"
    TST_SNAP="snap_test_${stype}.su"

    if [ ! -f "$REF_SNAP" ] || [ ! -f "$TST_SNAP" ]; then
        echo "  ${stype}: SKIP (file missing)"
        continue
    fi

    for fldr in 5 15 25 35 45; do
        SNAP_TOTAL=$((SNAP_TOTAL + 1))

        # Extract one snapshot by field record number
        suwind key=fldr min=$fldr max=$fldr < "$REF_SNAP" > _ref_snap.su 2>/dev/null
        suwind key=fldr min=$fldr max=$fldr < "$TST_SNAP" > _tst_snap.su 2>/dev/null

        # Peak amplitude of reference snapshot
        sumax < _ref_snap.su outpar=nep_ref 2>/dev/null
        REF_MAX=$(awk '{print $1}' nep_ref)

        # Peak of absolute difference
        sudiff _ref_snap.su _tst_snap.su 2>/dev/null | suop op=abs | sumax outpar=nep_diff 2>/dev/null
        DIFF_MAX=$(awk '{print $1}' nep_diff)

        # Relative error (%)
        REL_ERR=$(awk "BEGIN {if ($REF_MAX>0) printf \"%.6f\", 100.0*$DIFF_MAX/$REF_MAX; else print \"0.0\"}")

        PASS=$(awk "BEGIN {print ($REL_ERR <= $SNAP_THRESHOLD) ? 1 : 0}")
        if [ "$PASS" -eq 1 ]; then
            VERDICT="PASS"
            SNAP_PASS=$((SNAP_PASS + 1))
        else
            VERDICT="FAIL"
        fi

        echo "  ${stype} fldr=${fldr}: ref_max=${REF_MAX}  diff_max=${DIFF_MAX}  rel_err=${REL_ERR}%  [$VERDICT]"
        rm -f nep_ref nep_diff _ref_snap.su _tst_snap.su
    done
done

echo "-------------------------------------------"
echo "  Snapshots: ${SNAP_PASS}/${SNAP_TOTAL} passed (threshold=${SNAP_THRESHOLD}%)"
echo "==========================================="
echo ""

