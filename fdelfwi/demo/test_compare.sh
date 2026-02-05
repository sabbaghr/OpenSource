#!/bin/bash
#
# test_compare.sh - Compare test_fdfwimodc output against fdelmodc reference
#
# Test 1: Homogeneous acoustic model (Green's function style, like FigureGreenAppendixA)
# Test 2: Two-layer elastic model with topography (like fdelmodc_topography)
#
# Each test runs fdelmodc and test_fdfwimodc with identical parameters,
# then uses sudiff to verify the outputs are identical.
#

set -e

# Paths
ROOT=/rcp3/software/codes/OpenSource_SL10
BIN=${ROOT}/bin
FDELFWI=..
CWP=/rcp3/software/codes/cwp/bin

export PATH=${BIN}:${CWP}:${PATH}

PASS=0
FAIL=0

# -----------------------------------------------------------
# Comparison helper function
# Usage: compare_component ref_file test_file label
# -----------------------------------------------------------
compare_component() {
    local ref=$1 test=$2 label=$3
    if [ ! -f "$ref" ]; then
        echo "  [$label] SKIP - $ref not found"
        return
    fi
    if [ ! -f "$test" ]; then
        echo "  [$label] FAIL - $test not found"
        FAIL=$((FAIL+1))
        return
    fi
    sudiff "$ref" "$test" > diff_tmp.su 2>/dev/null
    sumax < diff_tmp.su outpar=nep_tmp 2>/dev/null
    maxval=$(cat nep_tmp | awk '{print $1}')
    rm -f diff_tmp.su nep_tmp

    # Check if maxval is zero (or very small)
    is_zero=$(echo "$maxval" | awk '{if ($1+0 == 0.0) print "yes"; else print "no"}')
    if [ "$is_zero" = "yes" ]; then
        echo "  [$label] PASS  (max diff = $maxval)"
        PASS=$((PASS+1))
    else
        echo "  [$label] FAIL  (max diff = $maxval)"
        FAIL=$((FAIL+1))
    fi
}


echo "========================================================="
echo " TEST 1: Homogeneous Acoustic (Green's function style)"
echo "========================================================="

# Clean
rm -f simple_cp.su simple_ro.su wave.su
rm -f ref_ac_*.su test_ac_*.su

cp=2000
rho=1000
dx=2.5
dt=0.0001

# Model: homogeneous acoustic
makemod sizex=2000 sizez=2000 dx=$dx dz=$dx cp0=$cp ro0=$rho \
    orig=-1000,0 file_base=simple.su

# Wavelet
makewave fp=15 dt=$dt file_out=wave.su nt=4096 t0=0.1

# --- Reference: fdelmodc ---
fdelmodc \
    file_cp=simple_cp.su ischeme=1 iorder=4 \
    file_den=simple_ro.su \
    file_src=wave.su \
    file_rcv=ref_ac.su \
    src_type=1 \
    rec_type_p=1 rec_type_vz=1 rec_type_vx=1 \
    rec_int_vz=2 \
    dtrcv=0.0005 \
    verbose=1 \
    tmod=0.5115 \
    dxrcv=5.0 \
    xrcv1=-500 xrcv2=500 \
    zrcv1=500 zrcv2=500 \
    xsrc=0 zsrc=1000 \
    ntaper=80 \
    left=4 right=4 top=4 bottom=4

# --- Test: test_fdfwimodc ---
${FDELFWI}/test_fdfwimodc \
    file_cp=simple_cp.su ischeme=1 iorder=4 \
    file_den=simple_ro.su \
    file_src=wave.su \
    file_rcv=test_ac.su \
    src_type=1 \
    rec_type_p=1 rec_type_vz=1 rec_type_vx=1 \
    rec_int_vz=2 \
    dtrcv=0.0005 \
    verbose=1 \
    tmod=0.5115 \
    dxrcv=5.0 \
    xrcv1=-500 xrcv2=500 \
    zrcv1=500 zrcv2=500 \
    xsrc=0 zsrc=1000 \
    ntaper=80 \
    left=4 right=4 top=4 bottom=4
    

echo ""
echo "--- Test 1 Results ---"
compare_component ref_ac_rp.su  test_ac_rp.su  "Acoustic P "
compare_component ref_ac_rvz.su test_ac_rvz.su "Acoustic Vz"
compare_component ref_ac_rvx.su test_ac_rvx.su "Acoustic Vx"


echo ""
echo "========================================================="
echo " TEST 2: Two-layer Elastic"
echo "========================================================="

# Clean
rm -f model_cp.su model_cs.su model_ro.su wav.su
rm -f ref_el_*.su test_el_*.su

# Wavelet
makewave dt=0.0004 nt=4096 fp=15 fmax=35 t0=0.10 file_out=wav.su verbose=1

# Two-layer elastic model
makemod file_base=model.su \
    cp0=2200 cs0=1300 ro0=1200 \
    sizex=2000 sizez=1000 dx=5 dz=5 orig=0,0 \
    intt=def poly=0 \
        cp=2800,2800 cs=1500,1500 ro=1500,1500 \
        x=0,2000 z=400,400 \
    verbose=1

# --- Reference: fdelmodc ---
fdelmodc \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wav.su \
    ischeme=3 iorder=6 \
    file_rcv=ref_el.su \
    rec_type_vx=1 rec_type_vz=1 rec_type_pp=1 \
    dtrcv=0.004 \
    xrcv1=100 xrcv2=1900 dxrcv=10 \
    zrcv1=0 zrcv2=0 \
    xsrc=1000 zsrc=0 \
    src_type=1 nshot=1 \
    ntaper=50 left=4 right=4 bottom=4 top=1 \
    tmod=1.0 \
    verbose=1

# --- Test: test_fdfwimodc ---
${FDELFWI}/test_fdfwimodc \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wav.su \
    ischeme=3 iorder=6 \
    file_rcv=test_el.su \
    rec_type_vx=1 rec_type_vz=1 rec_type_pp=1 \
    dtrcv=0.004 \
    xrcv1=100 xrcv2=1900 dxrcv=10 \
    zrcv1=0 zrcv2=0 \
    xsrc=1000 zsrc=0 \
    src_type=1 nshot=1 \
    ntaper=50 left=4 right=4 bottom=4 top=1 \
    tmod=1.0 \
    verbose=1

echo ""
echo "--- Test 2 Results ---"
compare_component ref_el_rvx.su test_el_rvx.su "Elastic Vx"
compare_component ref_el_rvz.su test_el_rvz.su "Elastic Vz"
compare_component ref_el_rpp.su  test_el_rpp.su  "Elastic P "


echo ""
echo "========================================================="
echo " TEST 3: Elastic with Topography"
echo "========================================================="

# Clean
rm -f topo_cp.su topo_cs.su topo_ro.su G2.su
rm -f ref_topo_*.su test_topo_*.su

dt=0.0004
ntap=120

# Topography model (simplified from fdelmodc_topography.scr)
makemod sizex=6000 sizez=2000 dx=5 dz=5 cp0=0 cs0=0 ro0=1000 \
    file_base=topo.su orig=0,-400 gradunit=0 \
    intt=def poly=2 cp=2450 ro=1000 gradcp=14 grad=0 cs=2000 \
        x=0,1500,3000,4500,6000 \
        z=-100,-250,-120,-200,-150 \
    intt=def poly=0 \
        x=0,6000 z=500,500 \
        cp=4500 cs=3800 ro=1400 gradcp=5 grad=0 \
    verbose=1

# Wavelet
makewave w=g2 fmax=45 t0=0.10 dt=$dt nt=4096 db=-40 file_out=G2.su verbose=1

# --- Reference: fdelmodc ---
fdelmodc \
    file_cp=topo_cp.su ischeme=3 \
    file_den=topo_ro.su \
    file_cs=topo_cs.su \
    file_rcv=ref_topo.su \
    file_src=G2.su \
    src_type=7 \
    rec_type_vz=1 rec_type_vx=1 rec_type_pp=1 \
    dtrcv=0.004 \
    verbose=1 \
    tmod=1.5 \
    dxrcv=20.0 \
    zrcv1=-400 zrcv2=-400 \
    xrcv1=0 xrcv2=6000 \
    sinkdepth=1 \
    sinkdepth_src=1 \
    xsrc=3000 zsrc=-250 \
    ntaper=$ntap \
    left=4 right=4 top=1 bottom=4

# --- Test: test_fdfwimodc ---
${FDELFWI}/test_fdfwimodc \
    file_cp=topo_cp.su ischeme=3 \
    file_den=topo_ro.su \
    file_cs=topo_cs.su \
    file_rcv=test_topo.su \
    file_src=G2.su \
    src_type=7 \
    rec_type_vz=1 rec_type_vx=1 rec_type_pp=1 \
    dtrcv=0.004 \
    verbose=1 \
    tmod=1.5 \
    dxrcv=20.0 \
    zrcv1=-400 zrcv2=-400 \
    xrcv1=0 xrcv2=6000 \
    sinkdepth=1 \
    sinkdepth_src=1 \
    xsrc=3000 zsrc=-250 \
    ntaper=$ntap \
    left=4 right=4 top=1 bottom=4

echo ""
echo "--- Test 3 Results ---"
compare_component ref_topo_rvx.su test_topo_rvx.su "Topo Vx"
compare_component ref_topo_rvz.su test_topo_rvz.su "Topo Vz"
compare_component ref_topo_rpp.su  test_topo_rpp.su  "Topo P "


echo ""
echo "========================================================="
echo " SUMMARY"
echo "========================================================="
echo " PASSED: $PASS"
echo " FAILED: $FAIL"
if [ $FAIL -eq 0 ]; then
    echo " ALL TESTS PASSED"
else
    echo " *** SOME TESTS FAILED ***"
fi
echo "========================================================="
