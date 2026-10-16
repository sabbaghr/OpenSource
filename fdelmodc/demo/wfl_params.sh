# wfl_params.sh -- parameters shared by the four wfl_case*.scr experiments.
# (sourced by wfl_case.sh; edit here to change all four cases at once)

# model (homogeneous elastic)
dx=4; dz=4
sizex=3300; sizez=3300
cp0=2500; cs0=1400; ro0=2000

# source wavelet and time
dt=0.0005; fp=15; t0=0.15; fmax=40; nt_wav=1024
tsnap1=0.1; tsnap2=1.5; dtsnap=0.05
# tmod = tsnap1+tsnap2 makes the snapshot grid an exact time mirror (frame k <-> frame N-1-k)
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
npml=150

# reconstruction rectangle Omega (absolute model coordinates)
wfl_x1=400; wfl_x2=2500
wfl_z1=400; wfl_z2=2500

# real source of the forward run: vertical force (P and S waves), OUTSIDE Omega
xsrc=1500; zsrc=1000; src_type_fwd=1; src_orient=1

# independent point receiver INSIDE Omega
ptx=2000; ptz=900

# reconstruction kernel: 1 = backward kernel (all fields with true sign), 0 = original forward kernel
# (with 0 the stresses come out with flipped sign; the scripts account for that in tables/plots)
BWD=${BWD:-1}

# snapshot fields to write/show (fdelmodc names: vx vz txx tzz txz); each field costs ~4 x 114 MB per case
SNAPFIELDS=${SNAPFIELDS:-"vx vz"}

# 1 = open suxmovie/suxgraph windows at the end, 0 = only write files and print tables
VIEW=${VIEW:-1}
# movie clip = CLIPFRAC * (max |forward snapshot|), identical for forward/reconstruction/difference windows
CLIPFRAC=${CLIPFRAC:-0.1}
