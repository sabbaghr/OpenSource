# wfl_params_open_refl.sh -- the one-sided (open line + free surface) case of wfl_params_open.sh with a
# flat REFLECTOR at z=1600. Everything else is inherited from wfl_params_open.sh.
#
#   z =    0  free surface,  z = 100 source,  z = 600 recording/injection line (x=500..2500)
#   z = 1600 ---- interface: cp/cs unchanged (pure density contrast), rho 2000 -> 2500
#                 (P reflection coeff at normal incidence = (rho2-rho1)/(rho2+rho1) ~= 0.11)
#
# The line now records downgoing AND reflected (upgoing) waves. Forward-order injection reconstructs the total
# field below the line (uniqueness of the causal solution), so the reflection should be reproduced too.
source "$(dirname "${BASH_SOURCE[0]}")/wfl_params_open.sh"

# source: P source (src_type=1) at (1500,100) instead of the vertical force of the homogeneous open cases
src_type_fwd=1; src_orient=1

refl_z=1600
make_model() {
	makemod file_base=model.su sizex=$sizex sizez=$sizez dx=$dx dz=$dz \
		cp0=$cp0 cs0=$cs0 ro0=$ro0 orig=0,0 gradunit=0 verbose=0 \
		intt=def poly=0 x=0,$sizex z=$refl_z,$refl_z cp=$cp0 cs=$cs0 ro=2500 gradcp=0 gradcs=0 grad=0
}
# migration/background model (REFSUB=1 in wfl_case.sh) = a SMOOTHED version of the TRUE model (the z=refl_z
# step blurred into a gradual ramp), not a separate model with no trace of the deeper velocity at all.
# Uses OpenSource's own `smooth` utility (utils/smooth.c) with niter>1: REPEATED boxcar convolution, which
# by the Central Limit Theorem approximates a GAUSSIAN kernel -- a smooth, tapered onset with no sharp
# derivative discontinuity, unlike a single hard-edged boxcar (wfl_smooth_model.py's approach, kept for the
# OBN layered case, which had a hard cutoff at the smoothing window edges and was found to generate its own
# small spurious "reflections" from those two edges -- see the reflector case notes). ntsm=41,niter=4
# (empirically tuned) gives about the same overall transition extent (flat by ~300 m either side of
# refl_z) as the old boxcar's SMOOTH_HALFWIDTH=300, but with a genuinely smooth taper.
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

# longer recording: reflected P reaches the line/receivers at ~1.2-1.4 s, reflected S at ~2.0-2.3 s
tsnap1=0.1; tsnap2=2.9; dtsnap=0.05
tmod=$(python3 -c "print($tsnap1+$tsnap2)")
