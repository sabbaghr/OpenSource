# wfl_params_obn.sh -- ONE-SIDED OBN (ocean-bottom node) reconstruction.
#
#   z =    0 ---------------------------------------- top of the model: ABSORBING boundary (CPML), no free surface
#   z =  100        * source in the water (P source, airgun-like)
#          water           cp=1500 cs=0    rho=1000
#   z =  500 ---------------------------------------- SEAFLOOR
#   z =  505 ======= OBN line (x = 10 .. 2990): recorded vx, vz, txx, tzz (= -p), txz  -> the one-sided surface
#          elastic layer 1 cp=2100 cs=600  rho=2000      (line 2 cells below the interface, inside the solid)
#   z = 1000 ---------------------------------------- interface 2
#          elastic layer 2 cp=2500 cs=1700 rho=2200
#
# The subsurface wavefield (below the line) is reconstructed from the OBN line alone (wfl_open=1):
#   wfl_obn1_fwdorder.scr     forward-time-order (retarded) injection  -> the total field below the seafloor
#   wfl_obn2_timereversed.scr reverse-time injection                    -> the reversed UP-going part (RTM-type
#                                                                           receiver wavefield), see notes there
# What an OBN measures: vx, vz (geophones) and p = -tzz (hydrophone); at a fluid-solid interface txz = 0 on the
# solid side, so 4C data give exactly the traction (txz, tzz) and velocity (vx, vz) the equivalent sources need.
source "$(dirname "${BASH_SOURCE[0]}")/wfl_params_layered.sh"

# time: forward-order needs only the arrivals; reverse-time additionally needs the subsurface field to have left
tsnap1=0.1; tsnap2=2.9; dtsnap=0.05
tmod=$(python3 -c "print($tsnap1+$tsnap2)")

# reference/background (migration) model for Born (scattered-field) subtraction (REFSUB=1 in
# wfl_case.sh): water (z<500) and the seafloor itself are kept EXACT (that boundary is known/measured
# in a real OBN survey, and keeping it identical to the true model cancels its own primary reflection
# out of the "scattered" residual); everything below z=500 is SMOOTHED (boxcar, SMOOTH_HALFWIDTH m) so
# the sharp z=1000 layer1/layer2 interface -- the actual imaging target -- is NOT reproduced in the
# reference model and so remains in the scattered residual, with correct background kinematics below it
# (unlike an earlier version that just extended layer 1's exact velocity flat to the bottom, which kept
# the right seafloor cancellation but gave the WRONG velocity for everything below z=1000, so reference
# and true wavefields also desynchronise in time there, not just at the interface).
SMOOTH_HALFWIDTH=${SMOOTH_HALFWIDTH:-150}
make_model_ref() {
	make_model_from() { # reuse the true model's own construction, just a different output name
		makemod file_base=$1 sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
			cp0=1500 ro0=1000 cs0=0 orig=0,0 gradunit=0 verbose=0 \
			intt=def poly=0 x=0,$sizex z=500,500   cp=2100 ro=2000 cs=600  gradcp=0 gradcs=0 grad=0 \
			intt=def poly=0 x=0,$sizex z=1000,1000 cp=2500 ro=2200 cs=1700 gradcp=0 gradcs=0 grad=0
	}
	make_model_from modelref.su
	python3 $DEMO/wfl_smooth_model.py modelref_cp.su $dz 500 $SMOOTH_HALFWIDTH
	python3 $DEMO/wfl_smooth_model.py modelref_cs.su $dz 500 $SMOOTH_HALFWIDTH
	python3 $DEMO/wfl_smooth_model.py modelref_ro.su $dz 500 $SMOOTH_HALFWIDTH
}

# absorbing (CPML) top boundary in BOTH runs: no sea-surface reflections / water multiples from the free surface
# (the layered params file sets these to 1, so they are assigned explicitly here; OBN_TOP=1 restores the free surface)
BND_TOP_FWD=${OBN_TOP:-2}
BND_TOP_RECON=${OBN_TOP:-2}

# open OBN line just below the seafloor
WFL_OPEN=1
wfl_x1=10; wfl_x2=2990
wfl_z1=505; wfl_z2=1400          # wfl_z2 is ignored by fdelmodc (open line); it only bounds the diagnostics region

# source in the water, above the line
xsrc=1500; zsrc=100; src_type_fwd=1; src_orient=1

# subsurface observation points (NOT used in the reconstruction): layer 1, layer 1 (off-centre), layer 2
ptx_list=1900,1200,1900
ptz_list=750,900,1100

WFL_FWD=${WFL_FWD:-1}
BWD=${BWD:-1}
PLOTCOMPS=${PLOTCOMPS:-"vz tzz"}
OUTROOT=wfl_cases_obn
