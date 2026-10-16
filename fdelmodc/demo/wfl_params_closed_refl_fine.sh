# wfl_params_closed_refl_fine.sh -- same model/geometry as wfl_params_closed_refl.sh (closed box, source
# outside Omega, reflector inside Omega) but at HALF the grid spacing (dx=dz=2m instead of 4m), to test
# whether the ~5.5-6% interior reconstruction residual shrinks with finer discretization (i.e. is a
# genuine numerical/quadrature convergence error, not a formula/sign bug).
# dt, npml, nt_wav scaled to keep the same physical CFL margin, PML width, and wavelet duration.

dx=2; dz=2
sizex=3300; sizez=3300
cp0=2500; cs0=1400; ro0=2000

refl_z=1500
make_model() {
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=$cp0 cs0=$cs0 ro0=$ro0 orig=0,0 gradunit=0 verbose=0 \
		intt=def poly=0 x=0,$sizex z=$refl_z,$refl_z cp=$cp0 cs=$cs0 ro=2500 gradcp=0 gradcs=0 grad=0
}

dt=0.00025; fp=15; t0=0.15; fmax=40; nt_wav=2048
tsnap1=0.1; tsnap2=1.5; dtsnap=0.05
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
npml=300

wfl_x1=400; wfl_x2=2500
wfl_z1=400; wfl_z2=2500

xsrc=1500; zsrc=100; src_type_fwd=1; src_orient=1

ptx=2000; ptz=900

BWD=${BWD:-1}
SNAPFIELDS=${SNAPFIELDS:-"vx vz"}
VIEW=${VIEW:-0}
CLIPFRAC=${CLIPFRAC:-0.1}
OUTROOT=wfl_cases_closed_refl
