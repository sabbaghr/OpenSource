# wfl_params_obn_diffr.sh -- ONE-SIDED OBN reconstruction, simplified to a single point diffractor
# below the seafloor instead of a second layer, so the reconstructed wavefield is a single clean
# converging/diverging circular event (like Ravasi & Curtis 2013, Fig. 4-5) instead of a layered
# section full of multiples and interface artifacts.
#
#   z =    0  top of model: ABSORBING boundary (CPML), no free surface
#          water            cp=1500 cs=0   rho=1000
#   z =  100        * source (P)
#   z =  500  ---------------------------------------- seafloor
#   z =  505  ======= OBN line, x = 10 .. 2990 (open surface, outward normal = up)
#          homogeneous elastic half-space   cp=2100 cs=600 rho=2000
#   z =  900        o  point diffractor: same cp,cs, rho+600 (pure density scatterer, as in the
#                      reference paper's example), width 10 m, circular
#
# The reference/background model for Born (scattered-field) subtraction is EXACTLY this model minus
# the diffractor (water + homogeneous half-space, no anomaly) -- an EXACT match everywhere except at
# the single point, so REFSUB needs no smoothing approximation here (unlike the layered case).

dx=2.5; dz=2.5
sizex=3000; sizez=1500
cp0=1500; cs0=0; ro0=1000   # background = water properties (only used if make_model_ref's makemod call needs a cp0; the real background is set explicitly below)

make_model() {
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=1500 ro0=1000 cs0=0 orig=0,0 gradunit=0 verbose=0 \
		intt=def   poly=0 x=0,$sizex z=500,500 cp=2100 ro=2000 cs=600 gradcp=0 gradcs=0 grad=0 \
		intt=diffr x=1500 z=900 cp=2100 ro=2600 cs=600 var=10,2 gradcp=0 gradcs=0 grad=0
}
make_model_ref() {
	makemod file_base=modelref.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=1500 ro0=1000 cs0=0 orig=0,0 gradunit=0 verbose=0 \
		intt=def poly=0 x=0,$sizex z=500,500 cp=2100 ro=2000 cs=600 gradcp=0 gradcs=0 grad=0
}

dt=0.0004; fp=10; t0=0.2; fmax=30; nt_wav=1500

# direct(source->scatterer) + scattered(scatterer->line) round trip ~0.65s; keep some margin
tsnap1=0.1; tsnap2=1.5; dtsnap=0.02
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
npml=100

BND_TOP_FWD=${OBN_TOP:-2}
BND_TOP_RECON=${OBN_TOP:-2}

WFL_OPEN=1
wfl_x1=10; wfl_x2=2990
wfl_z1=505; wfl_z2=1400

xsrc=1500; zsrc=100; src_type_fwd=1; src_orient=1

# observation points around the diffractor: above, beside, below (not used in the reconstruction itself)
ptx_list=1500,2000,1500
ptz_list=700,900,1100

WFL_FWD=${WFL_FWD:-0}
BWD=${BWD:-1}
SNAPFIELDS=${SNAPFIELDS:-"vx vz"}
PLOTCOMPS=${PLOTCOMPS:-"vz tzz"}
VIEW=${VIEW:-0}
CLIPFRAC=${CLIPFRAC:-0.1}
OUTROOT=wfl_cases_obn_diffr
