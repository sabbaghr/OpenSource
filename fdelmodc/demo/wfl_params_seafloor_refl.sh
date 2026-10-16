# wfl_params_seafloor_refl.sh -- water + KNOWN seafloor (z=500) + ONE target reflector (z=1100),
# model bottom at z=1500, FREE SURFACE on top, open (one-sided) receiver line just below the seafloor,
# reverse-time reconstruction of the SCATTERED field (REFSUB) with edge taper on the line.
#
#   z =    0  free surface
#          water                cp=1500 cs=0    rho=1000
#   z =  100        * source (P)
#   z =  500  ---------------------------------------- seafloor (KNOWN, kept exact in the migration model)
#   z =  505  ======= open receiver line, x = 10 .. 2990
#          layer 1              cp=2100 cs=600  rho=2000
#   z = 1100  ---------------------------------------- target reflector (UNKNOWN -- smoothed in the
#                                                        migration model, same as the flat-reflector case)
#          layer 2              cp=2500 cs=1700 rho=2200
#   z = 1500  model bottom
#
# Migration/background model = water + seafloor EXACT, z=1100 contrast SMOOTHED (Gaussian-like, via
# wfl_smooth_model.py's --niter repeated-boxcar, protected above z0 so the seafloor stays exact).

dx=2.5; dz=2.5
sizex=3000; sizez=1500

seafloor_z=500
refl_z=1100
WATER_CS0=${WATER_CS0:-0}   # opt-in test: give the "water" layer a nonzero shear velocity (e.g. matching
                            # the sediment's cs=600) to check whether the fluid (cs=0) discontinuity at the
                            # seafloor is what's causing the above-line leak in tensorial reconstruction.
make_model() {
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=1500 ro0=1000 cs0=$WATER_CS0 orig=0,0 gradunit=0 verbose=0 \
		intt=def poly=0 x=0,$sizex z=$seafloor_z,$seafloor_z cp=2100 ro=2000 cs=600  gradcp=0 gradcs=0 grad=0 \
		intt=def poly=0 x=0,$sizex z=$refl_z,$refl_z         cp=2500 ro=2200 cs=1700 gradcp=0 gradcs=0 grad=0
}

SMOOTH_Z0=${SMOOTH_Z0:-700}          # protect exactly everything above this depth (seafloor is at 500)
SMOOTH_HALFWIDTH=${SMOOTH_HALFWIDTH:-80}   # metres, single-pass boxcar half-width
SMOOTH_NITER=${SMOOTH_NITER:-4}            # repeated passes -> Gaussian-like taper (see wfl_smooth_model.py)
make_model_ref() {
	makemod file_base=modelref.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=1500 ro0=1000 cs0=$WATER_CS0 orig=0,0 gradunit=0 verbose=0 \
		intt=def poly=0 x=0,$sizex z=$seafloor_z,$seafloor_z cp=2100 ro=2000 cs=600  gradcp=0 gradcs=0 grad=0 \
		intt=def poly=0 x=0,$sizex z=$refl_z,$refl_z         cp=2500 ro=2200 cs=1700 gradcp=0 gradcs=0 grad=0
	python3 $DEMO/wfl_smooth_model.py modelref_cp.su $dz $SMOOTH_Z0 $SMOOTH_HALFWIDTH --niter $SMOOTH_NITER
	python3 $DEMO/wfl_smooth_model.py modelref_cs.su $dz $SMOOTH_Z0 $SMOOTH_HALFWIDTH --niter $SMOOTH_NITER
	python3 $DEMO/wfl_smooth_model.py modelref_ro.su $dz $SMOOTH_Z0 $SMOOTH_HALFWIDTH --niter $SMOOTH_NITER
}

dt=0.0004; fp=10; t0=0.2; fmax=30; nt_wav=1500

# direct + PP reflection (~1.0s) + converted waves; generous margin
tsnap1=0.1; tsnap2=1.9; dtsnap=0.02
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
npml=${NPML:-100}

BND_TOP_FWD=${BND_TOP_FWD:-1}      # free surface
BND_TOP_RECON=${BND_TOP_RECON:-1}

WFL_OPEN=1
wfl_x1=10; wfl_x2=2990
# wfl_z1 overridable: 505 (5 m below the seafloor, in the solid -- "geophone" line: vx,vz meaningful)
# or 495 (5 m above the seafloor, in the water -- "hydrophone" line: tzz(=-p),txz(=0) meaningful, see
# wfl_seafloor_realistic_recon.sh which runs both and splices them into one physically-realistic injection)
wfl_z1=${WFL_Z1:-505}; wfl_z2=1500

xsrc=1500; zsrc=100; src_type_fwd=1; src_orient=1

ptx_list=1800,2100,1200
ptz_list=700,900,1300

WFL_FWD=${WFL_FWD:-0}
BWD=${BWD:-1}
SNAPFIELDS=${SNAPFIELDS:-"vx vz"}
PLOTCOMPS=${PLOTCOMPS:-"vz tzz"}
VIEW=${VIEW:-0}
CLIPFRAC=${CLIPFRAC:-0.1}
OUTROOT=wfl_cases_seafloor_refl
