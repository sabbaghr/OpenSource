# wfl_params_uniform_diffr.sh -- simplest possible test: a FULLY UNIFORM elastic background (no water
# layer, no seafloor) with a single point diffractor. Source, receiving line and diffractor all sit in
# the same homogeneous medium -- removes the seafloor/water complications entirely, leaving only the
# one scattered event to look at.
#
#   z =    0  top of model: ABSORBING boundary (CPML)
#          uniform elastic background   cp=2100 cs=600 rho=2000
#   z =  100        * source (P)
#   z =  505  ======= recording line, x = 10 .. 2990 (open surface, outward normal = up)
#   z =  900        o  point diffractor: same cp,cs, rho+600, width 10 m, circular
#
# Reference/background model for Born (scattered-field) subtraction = the SAME uniform medium with NO
# diffractor at all -- as exact as it gets, the true and reference models differ at exactly one point.

dx=2.5; dz=2.5
sizex=3000; sizez=1500

make_model() {
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=2100 ro0=2000 cs0=600 orig=0,0 gradunit=0 verbose=0 \
		intt=diffr x=1500 z=900 cp=2100 ro=2600 cs=600 var=10,2
}
make_model_ref() {
	makemod file_base=modelref.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=2100 ro0=2000 cs0=600 orig=0,0 gradunit=0 verbose=0
}

dt=0.0004; fp=10; t0=0.2; fmax=30; nt_wav=1500

tsnap1=0.1; tsnap2=1.5; dtsnap=0.02
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
npml=100

BND_TOP_FWD=${OBN_TOP:-2}
BND_TOP_RECON=${OBN_TOP:-2}

WFL_OPEN=1
wfl_x1=10; wfl_x2=2990
wfl_z1=505; wfl_z2=1400

xsrc=1500; zsrc=100; src_type_fwd=1; src_orient=1

ptx_list=1500,2000,1500
ptz_list=700,900,1100

WFL_FWD=${WFL_FWD:-0}
BWD=${BWD:-1}
SNAPFIELDS=${SNAPFIELDS:-"vx vz"}
PLOTCOMPS=${PLOTCOMPS:-"vz tzz"}
VIEW=${VIEW:-0}
CLIPFRAC=${CLIPFRAC:-0.1}
OUTROOT=wfl_cases_uniform_diffr
