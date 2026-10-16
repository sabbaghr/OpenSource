# wfl_params_open.sh -- homogeneous half-space, FREE SURFACE on top, single-sided (open) reconstruction
# surface = one horizontal line of receivers (the "top face" only).
# Used by wfl_open1_fwdorder.scr and wfl_open2_timereversed.scr (via WFL_PARAMS); outputs in wfl_cases_open/<case>/
#
#   z = 0    ---------------------------------------- free surface (top=1)
#   z = 100         * source (Fz)
#   z = 600  ======= recording/injection line, x = 500 .. 2500 (open surface, outward normal = up)
#            reconstruction region: below the line (downgoing field), receivers below at z = 1000..1500
#
# Physics: an open line can only radiate DOWNWARD, so the valid reconstruction is the forward-time-order
# (retarded) one (WFL_FWD=1): inject the recorded data in the recorded time order and get the downgoing
# wavefield back. Time-reversed injection (the closed-box workflow) does NOT reconstruct the forward
# field from one-sided data (checked: correlation ~0); wfl_open2_timereversed.scr shows that.

dx=4; dz=4
sizex=3000; sizez=2000
cp0=2500; cs0=1400; ro0=2000

dt=0.0005; fp=15; t0=0.15; fmax=40; nt_wav=1024

# forward-order injection has no "field must vanish at the end" requirement (it starts from zero like the
# forward run), so 2 s is enough to record the arrivals below the line.  tmod = tsnap1 + tsnap2.
tsnap1=0.1; tsnap2=1.9; dtsnap=0.05
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
npml=${NPML:-100}

# free surface on top in both runs
BND_TOP_FWD=${BND_TOP_FWD:-1}      # 1 = free surface, 2 = absorbing (CPML)
BND_TOP_RECON=${BND_TOP_RECON:-1}

# open line: x1..x2 at depth z1 (wfl_z2 is ignored by fdelmodc when wfl_open=1; it only bounds the region
# used by the leakage/rms diagnostics, i.e. the region below the line)
WFL_OPEN=1
wfl_x1=${WFL_X1:-500}; wfl_x2=${WFL_X2:-2500}   # line extent, overridable with WFL_X1 / WFL_X2
wfl_z1=600; wfl_z2=1800

# source between the free surface and the line: vertical force (P and S waves)
xsrc=1500; zsrc=100; src_type_fwd=7; src_orient=1

# point receivers below the line (inside the aperture, off the source axis so vx/txz are non-zero)
ptx_list=1800,2100,1200
ptz_list=1000,1300,1500

WFL_FWD=${WFL_FWD:-1}
BWD=${BWD:-1}
SNAPFIELDS=${SNAPFIELDS:-"vx vz"}
PLOTCOMPS=${PLOTCOMPS:-"vz tzz"}
VIEW=${VIEW:-1}
CLIPFRAC=${CLIPFRAC:-0.1}
OUTROOT=wfl_cases_open
