# wfl_params_nonlinear_diffr.sh -- test case for the nonlinear (eq. 9/21, Ravasi & Curtis 2013;
# eq. 3, Ravasi, Vasconcelos & Curtis 2014) source-receiver imaging condition. Deliberately NOT the
# Rocha single-flat-reflector geometry: that model has essentially no multiply-scattered energy for a
# nonlinear term to find, and an open receiver line doesn't match the theory's enclosing-boundary
# requirement. Here:
#
#   - constant cp,cs everywhere (NO velocity contrast anywhere -- isolates density scattering only,
#     matching this project's established density-only-reflector convention)
#   - 5 point diffractors (density-only perturbations, ro0=2000 -> ro=2600), arranged in a quincunx
#     (center + 4 corners) so they produce genuine inter-scatterer multiple scattering
#   - a CLOSED SQUARE of receivers (Omega) fully enclosing all 5 diffractors -- matches the
#     representation theorem's enclosing-boundary requirement, unlike the Rocha open-line case
#   - ONE real source genuinely OUTSIDE Omega (same discipline as wfl_params_closed_refl.sh)
#
#   z =    0  top of model: ABSORBING boundary (CPML)
#          uniform elastic background   cp=2100 cs=600 rho=2000
#   z =  200        * source (P)
#   z =  800  ---------------------------------------- Omega top (wfl_z1)
#          5 diffractors (same cp,cs, rho+600, width 10m, circular) at:
#            (1500,1500) center, (1100,1100) (1900,1100) (1100,1900) (1900,1900) corners
#   z = 2200  ---------------------------------------- Omega bottom (wfl_z2)
#
# Reference/background model for G^0/G-tilde^0 (and for REFSUB-style scattered-data isolation) = the
# SAME uniform medium with NO diffractors at all -- exact everywhere except at the 5 points.

dx=4; dz=4
sizex=3000; sizez=3000
cp0=2100; cs0=600; ro0=2000

DIFFR_RO=${DIFFR_RO:-2600}
DIFFR_VAR=${DIFFR_VAR:-10,2}
make_model() {
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=$cp0 cs0=$cs0 ro0=$ro0 orig=0,0 gradunit=0 verbose=0 \
		intt=diffr x=1500 z=1500 cp=$cp0 ro=$DIFFR_RO cs=$cs0 var=$DIFFR_VAR \
		intt=diffr x=1100 z=1100 cp=$cp0 ro=$DIFFR_RO cs=$cs0 var=$DIFFR_VAR \
		intt=diffr x=1900 z=1100 cp=$cp0 ro=$DIFFR_RO cs=$cs0 var=$DIFFR_VAR \
		intt=diffr x=1100 z=1900 cp=$cp0 ro=$DIFFR_RO cs=$cs0 var=$DIFFR_VAR \
		intt=diffr x=1900 z=1900 cp=$cp0 ro=$DIFFR_RO cs=$cs0 var=$DIFFR_VAR
}
make_model_ref() {
	makemod file_base=modelref.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=$cp0 cs0=$cs0 ro0=$ro0 orig=0,0 gradunit=0 verbose=0
}

# dispersion check: cmin=cs0=600, dx=4 -> fmax < cmin/(5*dx) = 30 Hz; leave margin
dt=0.0005; fp=12; t0=0.15; fmax=25; nt_wav=1024

# source (1500,200) to farthest corner diffractor (1900,1900): ~1909m; diffractor to farthest Omega
# receiver point (e.g. (800,800) from (1900,1900)): ~1556m; round trip at cs0=600 m/s dominates timing
tsnap1=0.1; tsnap2=3.0; dtsnap=0.01
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
npml=100

BND_TOP_FWD=${BND_TOP_FWD:-2}
BND_TOP_RECON=${BND_TOP_RECON:-2}

# closed-box reconstruction boundary Omega (WFL_OPEN unset = closed contour, default)
wfl_x1=800; wfl_x2=2200
wfl_z1=800; wfl_z2=2200

# real source of the forward run: outside Omega (above it: zsrc < wfl_z1)
xsrc=1500; zsrc=200; src_type_fwd=1; src_orient=1

# point receivers near representative diffractors (diagnostics only, not used by the imaging condition)
ptx_list=1500,1100,1900
ptz_list=1500,1100,1900

WFL_FWD=${WFL_FWD:-0}
BWD=${BWD:-1}
SNAPFIELDS=${SNAPFIELDS:-"vx vz txx tzz txz"}
PLOTCOMPS=${PLOTCOMPS:-"vz tzz"}
VIEW=${VIEW:-0}
CLIPFRAC=${CLIPFRAC:-0.1}
OUTROOT=wfl_cases_nonlinear
