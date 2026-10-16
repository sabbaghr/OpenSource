# wfl_params_layered.sh -- layered model: water + 2 elastic layers, free surface on top.
# Used by wfl_layered1_original.scr .. wfl_layered4_both.scr (via WFL_PARAMS); outputs in wfl_cases_layered/<case>/
#
#   z =    0 ---------------------------------------- free surface
#          water           cp=1500 cs=0    rho=1000
#   z =  500 ---------------------------------------- interface 1
#          elastic layer 1 cp=2100 cs=600  rho=2000
#   z = 1000 ---------------------------------------- interface 2
#          elastic layer 2 cp=2500 cs=1700 rho=2200   (continues below the model)
# Reduced version (smaller, 2 elastic layers only) of the 6000 x 2100 reference model.

dx=2.5; dz=2.5
sizex=3000; sizez=1500

make_model() {
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=1500 ro0=1000 cs0=0 orig=0,0 gradunit=0 verbose=0 \
		intt=def poly=0 x=0,$sizex z=500,500   cp=2100 ro=2000 cs=600  gradcp=0 gradcs=0 grad=0 \
		intt=def poly=0 x=0,$sizex z=1000,1000 cp=2500 ro=2200 cs=1700 gradcp=0 gradcs=0 grad=0
}

# wavelet: cs_min=600 m/s at dx=2.5 allows at most 48 Hz (5 points/wavelength, checked by fdelmodc),
# so fp=10 Hz (spectrum up to ~36 Hz) instead of the 15 Hz used in the homogeneous experiment
dt=0.0004; fp=10; t0=0.2; fmax=30; nt_wav=1500

# time: energy inside Omega decays slowly (water reverberation, slow interface waves): about 1.5% of
# the peak (energy) at 2 s, 0.1% at 3 s, 0.03% at 4 s in the forward test -> record 3.5 s.
# tmod = tsnap1 + tsnap2 keeps the snapshot grid an exact time mirror.
tsnap1=0.1; tsnap2=3.4; dtsnap=0.05
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
npml=100

# free surface on top in the forward run; the reconstruction run uses the same top boundary
# (override with BND_TOP_RECON=2 to absorb instead: Omega does not touch the free surface)
BND_TOP_FWD=1
BND_TOP_RECON=${BND_TOP_RECON:-1}

# reconstruction rectangle Omega: water (z<500) + layer 1 (500-1000) + layer 2 (z>1000)
wfl_x1=700; wfl_x2=2300
wfl_z1=250; wfl_z2=1250

# source in the water (100 m below the free surface, above Omega): pressure/P source
xsrc=1500; zsrc=100; src_type_fwd=1; src_orient=1

# point receivers inside Omega: water, layer 1, layer 2
ptx_list=1900,1900,1900
ptz_list=400,750,1100

BWD=${BWD:-1}
SNAPFIELDS=${SNAPFIELDS:-"vx vz"}
PLOTCOMPS=${PLOTCOMPS:-"vz tzz"}
VIEW=${VIEW:-1}
CLIPFRAC=${CLIPFRAC:-0.1}
OUTROOT=wfl_cases_layered
