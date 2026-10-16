#!/bin/bash
# adj_params.sh -- fixed parameters for the elastic adjoint dot-product test (Step 1).
# Sourced by adj_step1_setup.scr and later step2/3/4 drivers. Pure shell variables only --
# every number below is a literal, hand-derived constant (see docs/adjoint_dotproduct_step0_report.md),
# not something computed at run time.

# ---------------- grid / model ----------------
DX=5; DZ=5
SIZEX=1000; SIZEZ=1000          # -> nx=nz=201 (sizex/dx+1), matches the ~201x201 spec

# smooth heterogeneous background: linear lateral gradient (left->right, x=0..SIZEX) plus a
# linear depth gradient (gradunit=0 => units are [quantity]/dz-step), verified exactly in
# scratchpad/modeltest: cp(x,z) = lerp(CP_LEFT,CP_RIGHT; x/SIZEX) + GRADCP*(z/DZ), same form
# for cs/ro. Range ends up cp in [2850,3350], cs in [1350,1750], ro in [1900,2130] -- Vs>0 everywhere.
CP_LEFT=2850;  CP_RIGHT=3150;  GRADCP=1.0
CS_LEFT=1350;  CS_RIGHT=1650;  GRADCS=0.5
RO_LEFT=1900;  RO_RIGHT=2100;  GRADRO=0.1

# Phase-1 constant-patch material (must match exactly at every P_x/P_y point):
RHO0=2000; VP0=3000; VS0=1500
# derived (hand-computed, see Step 0 report):
MU0=4500000000          # rho0*Vs0^2
LAM0=9000000000         # rho0*Vp0^2 - 2*MU0
L2M0=18000000000        # rho0*Vp0^2 = LAM0+2*MU0
CINV_DET=243000000000000000000   # 4*MU0*(LAM0+MU0) = 4*4.5e9*1.35e10
CINV_DIAG=7.407407e-11            # L2M0/CINV_DET
CINV_OFFDIAG=3.703704e-11         # LAM0/CINV_DET  (used with a minus sign)
CINV_SHEAR=2.222222e-10           # 1/MU0

# patch radius (meters): elipse var=r1,r2 with r1=r2=8 rasterizes to an exact 3x3 block of
# base-grid nodes around the center at dx=dz=5 (diag distance 7.07 <= 8 < 10 = 2-cell distance),
# verified in scratchpad/modeltest (patch center, its 8 neighbors, and the 2-cell-away points
# all checked by hand against the model file).
PATCH_R=8

# ---------------- P_x (source) and P_y (receiver) points ----------------
# All >=40 grid cells (200 m) from every boundary (spec only requires >=20 cells / 100 m).
PX_X="300 700 500 300 700"
PX_Z="300 300 500 700 700"
PY_X="200 500 800 200 800 200 500 800 400 600"
PY_Z="200 200 200 500 500 800 800 800 600 400"

# ---------------- timing ----------------
DT=0.0004
NT=1000                  # -> trace length (NT-1)*DT = 0.3996 s
TLAST=0.3996             # (NT-1)*DT, literal, in SECONDS (sumute's tmute= unit) -- used to
                         # hard-zero the final sample (see adj_lib.sh adj_make_signal)
# Courant check (4th-order 2D staggered elastic, safe threshold ~0.6): cp_max ~ 3350+ (background)
# but never exceeds ~3350 anywhere (patches sit at cp0=3000, inside range) =>
#   Courant = cp_max*DT/DX = 3350*0.0004/5 = 0.268  (comfortable margin)

# ---------------- random band-limited source/adjoint-source signals ----------------
# Dispersion rule of thumb (4th order, >=8 pts/wavelength): fmax <= Vs_min/(8*DX) = 1350/40 = 33.75 Hz.
# Passband chosen well below that:
BP_F1=2; BP_F2=5; BP_F3=15; BP_F4=18     # sufilter f=,amps=0,1,1,0 (Hz)
TAPER_MS=60                               # sutaper tbeg=tend=60 ms (15% of the 400 ms record)
SEEDS="101 102 103 104 105"               # 5 fixed seeds, per spec

# ---------------- component <-> src_type / rec_type mapping (Step 0 report, fdelmodc.c:138) ----------------
COMP_NAMES=(Fx Fz Txx Tzz Txz)            # index 1..5 (bash arrays are 0-based; see COMP index convention below)
COMP_SRC_TYPE=(6 7 4 3 2)
REC_NAMES=(vx vz txx tzz txz)
# Component index convention used by adj_seed(): 1=vx/Fx 2=vz/Fz 3=txx/Txx 4=tzz/Tzz 5=txz/Txz
# (same order as COMP_NAMES/COMP_SRC_TYPE/REC_NAMES, 1-based).

# boundary used for Steps 2-3 (Step 4 sweeps this separately): rigid on all four sides.
BND=3   # 1=free 2=pml 3=rigid 4=taper

# ---------------- adjoint-source pre-scaling: pure M^{-1}, no extra S_c conversion ----------------
# CORRECTED (was wrong): earlier version also divided by S_c to "convert physical forcing back to
# file amplitude", reasoning applySource.c's own scaling needed compensating first. That was an
# extra, unwanted round-trip -- confirmed by direct measurement: a single-point-pair Txz->txz test
# gave a/b matching L2M0/(DX*DX) to within 0.05%, exactly the erroneous factor that extra step
# introduced. s~=M^{-1}y is meant to be injected AS the file amplitude directly, the same
# convention x itself uses (both sides are "raw" quantities; only M/M^{-1} apply, nothing else):
#   ampl_file(Fx)  = y_vx / RHO0
#   ampl_file(Fz)  = y_vz / RHO0
#   ampl_file(Txx) = -L2M0*y_txx -LAM0*y_tzz
#   ampl_file(Tzz) = -LAM0*y_txx -L2M0*y_tzz
#   ampl_file(Txz) = -MU0*y_txz
# Confirmed by direct raw-reciprocity measurement (Fx-at-A->vz-at-B vs Fz-at-B->vx-at-A, same
# wavelet, no adjoint math at all) that applySource.c's Fx branch (the explicit 0.5x noted in the
# Step 0 report) injects at roughly half the strength Fz does for the same file amplitude (B/A
# came out ~1.8-1.9, close to the expected 2). This is invisible in a pure Fx<->Fx self-test
# (cancels in the ratio: both a and b use the SAME native, uncorrected mechanism) but breaks any
# test that pairs Fx against Fz. AS_FX (native, 1/RHO0) stays uncorrected -- it's for adjoint
# constructions where Fx is tested against itself. AS_FX_CLEAN carries the extra x2, and is used
# only when an Fx-type adjoint injection stands in for "the vx-slot of M^{-1}y" in a test whose
# OTHER side is Fz (native, uncorrected) -- i.e. only in the off-diagonal Fz->vx group, to bring
# Fx's native weakness up to Fz's native strength so the two sides of that specific test match.
AS_FX=0.0005
AS_FX_CLEAN=0.001
AS_FZ=0.0005
# AS_FZ_HALF: for the Fx-forward/Fz-adjoint cross group (G5-style), the correction belongs on the
# Fz-adjoint side (native Fz is clean/2x-strong relative to native-weak Fx), NOT on the shared x
# test vector (scaling x is a no-op for a/b's ratio, caught and fixed this pass). Confirmed via
# energy ratio (phase-invariant, so isolated from the separate Fz tshift issue) that Fx is weaker
# than Fz by a clean factor of 2.000 (not the ~1.8-1.9 the raw sample-by-sample comparison showed
# -- that discrepancy was the tshift timing mismatch, a separate effect, not amplitude).
AS_FZ_HALF=0.00025
AS_TXX_SELF=-18000000000;  AS_TXX_CROSS=-9000000000
AS_TZZ_CROSS=-9000000000;  AS_TZZ_SELF=-18000000000
AS_TXZ=-4500000000

# ---------------- final M rescale applied to the (re-time-reversed) adjoint recordings ----------------
#   v+  = RHO0 * v~        (v+_vx = 2000*v~_vx, v+_vz = 2000*v~_vz)
#   tau+= -C^{-1} tau~ :
#     tau+_txx = -CINV_DIAG*txx~ + CINV_OFFDIAG*tzz~
#     tau+_tzz =  CINV_OFFDIAG*txx~ - CINV_DIAG*tzz~
#     tau+_txz = -CINV_SHEAR*txz~
# (CINV_DIAG, CINV_OFFDIAG, CINV_SHEAR, RHO0 already defined above.)
