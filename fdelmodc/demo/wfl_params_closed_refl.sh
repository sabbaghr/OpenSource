# wfl_params_closed_refl.sh -- closed-box reconstruction test with the source genuinely OUTSIDE Omega
# (above the box, matching the representation theorem's source-free-interior assumption -- the original
# wfl_params.sh's own comment claimed this but its actual xsrc/zsrc were inside the box) and a horizontal
# REFLECTOR inside Omega, so the box must reconstruct a genuinely scattered (not just direct) wavefield.
#
#   z =    0
#          background   cp=2500 cs=1400 rho=2000
#   z =  100   * source (outside Omega, above it)
#   z =  400  ---------------------------------------- Omega top (wfl_z1)
#          Omega interior
#   z = 1500  ---------------------------------------- reflector: cp/cs unchanged (pure density contrast),
#                                                        rho=2000 -> 2500
#   z = 2500  ---------------------------------------- Omega bottom (wfl_z2)
#
# No REFSUB here: a closed contour's exact reconstruction doesn't need reference-model subtraction --
# unlike the open-line migration cases, Omega has no source inside it and the full boundary is used, so
# the representation theorem reconstructs the TOTAL field (direct + reflected) exactly inside Omega.

dx=4; dz=4
sizex=3300; sizez=3300
cp0=2500; cs0=1400; ro0=2000

refl_z=1500
make_model() {
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=$cp0 cs0=$cs0 ro0=$ro0 orig=0,0 gradunit=0 verbose=0 \
		intt=def poly=0 x=0,$sizex z=$refl_z,$refl_z cp=$cp0 cs=$cs0 ro=2500 gradcp=0 gradcs=0 grad=0
}

dt=0.0005; fp=15; t0=0.15; fmax=40; nt_wav=1024
tsnap1=0.1; tsnap2=1.5; dtsnap=0.05
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
npml=150

# reconstruction rectangle Omega (absolute model coordinates)
wfl_x1=400; wfl_x2=2500
wfl_z1=400; wfl_z2=2500

# real source of the forward run: vertical force, genuinely OUTSIDE Omega (above it: zsrc < wfl_z1)
xsrc=1500; zsrc=100; src_type_fwd=1; src_orient=1

# independent point receiver INSIDE Omega
ptx=2000; ptz=900

BWD=${BWD:-1}
SNAPFIELDS=${SNAPFIELDS:-"vx vz"}
VIEW=${VIEW:-0}
CLIPFRAC=${CLIPFRAC:-0.1}
OUTROOT=wfl_cases_closed_refl
