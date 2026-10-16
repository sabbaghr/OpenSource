# wfl_params_rocha_refl.sh -- model geometry matching Figures 3-5 of Rocha, Tanushev & Sava (2016),
# "Isotropic elastic wavefield imaging using the energy norm" (GEOPHYSICS 81(4), S207-S219,
# geo2015-0487): horizontal reflector at z=1500m, vertical-displacement source at x=5000m, domain
# 10000m x 2500m. The paper does not give explicit vp/vs/density values for this simple example (only
# "a velocity contrast" causing backscattering artifacts). Here the reflector is a PURE DENSITY contrast:
# cp0=2500,cs0=1400,rho0=2000 above the reflector -> cp/cs unchanged, rho=2500 below (P reflection
# coefficient at normal incidence = (rho2-rho1)/(rho2+rho1) ~= 0.11), consistent with the other flat-
# reflector test cases in this project (wfl_params_open_refl.sh, wfl_params_closed_refl.sh).
source "$(dirname "${BASH_SOURCE[0]}")/wfl_params_open.sh"

dx=10; dz=10
sizex=10000; sizez=2500

src_type_fwd=7; src_orient=1   # vertical-displacement-generating force source (matches wfl_params_open.sh's own default)
fmax=20   # dx=10m dispersion limit (5 pts/wavelength, cmin=1400 m/s) is ~28Hz; fmax= isn't a hard spectral
          # cutoff for the wavelet (fmax=25 still left actual content at 29.3Hz), so leave more margin
          # rather than refine the grid on this much larger (10000x2500m) domain

refl_z=1500
make_model() {
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=$cp0 cs0=$cs0 ro0=$ro0 orig=0,0 gradunit=0 verbose=0 \
		intt=def poly=0 x=0,$sizex z=$refl_z,$refl_z cp=$cp0 cs=$cs0 ro=2500 gradcp=0 gradcs=0 grad=0
}
SMOOTH_NTSM=${SMOOTH_NTSM:-41}
SMOOTH_NITER=${SMOOTH_NITER:-4}
make_model_ref() {
	makemod file_base=modelref.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=$cp0 cs0=$cs0 ro0=$ro0 orig=0,0 gradunit=0 verbose=0 \
		intt=def poly=0 x=0,$sizex z=$refl_z,$refl_z cp=$cp0 cs=$cs0 ro=2500 gradcp=0 gradcs=0 grad=0
	for f in cp cs ro; do
		smooth file_in=modelref_$f.su file_out=modelref_${f}_sm.su ntsm=$SMOOTH_NTSM nxsm=0 niter=$SMOOTH_NITER verbose=0
		mv modelref_${f}_sm.su modelref_$f.su
	done
}

xsrc=${XSRC:-5000}; zsrc=10   # XSRC overridable for multi-shot runs (wfl_cig_rocha.sh)

# Recording time must cover the SLOWEST (SS) reflection to the FARTHEST receiver: source at x=5000,
# receivers to x=10/9990 (offset ~4990m), reflector at z=1500, source/receivers near z=0 -> image-method
# path = sqrt(4990^2 + (2*1500)^2) ~= 5822m, at cs0=1400 m/s -> ~4.16s travel + t0=0.15s onset -> need
# tmod >= ~4.3s. Measured reconstruction amplitude actually decays to the noise floor by t~4.16s, so
# tsnap2=4.65 (tmod=4.75s) keeps a ~0.6s margin past that without paying for the extra, empty 1.25s
# tail the earlier tsnap2=5.9 (tmod=6.0s) setting was recording.
tsnap1=0.1; tsnap2=4.65; dtsnap=0.05
tmod=$(python3 -c "print($tsnap1+$tsnap2)")

WFL_OPEN=1
wfl_x1=${WFL_X1:-10}; wfl_x2=${WFL_X2:-9990}
wfl_z1=20; wfl_z2=1800    # open receiver line (z2 unused when WFL_OPEN=1, kept >z1 only to pass the degeneracy check)

ptx_list=4500,5500,6500
ptz_list=1000,1300,1600

OUTROOT=${ROCHA_OUTROOT:-wfl_cases_rocha}   # ROCHA_OUTROOT: per-shot folders for wfl_cig_rocha.sh
