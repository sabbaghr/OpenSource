#!/bin/bash
# test_lag_gradient.sh -- validation of the time-lag extended gradient
#                         (lag_gradient.c, lag_* parameters)
#
# What is being tested
# -------------------
# lag_gradient.c reconstructs the lag-extended FWI gradient
#
#     g(x,tau) = sum_t  A(x,t-tau) B(x,t+tau)
#
# from running DFTs of the forward strain A and the adjoint stress multiplier
# B (psi_sigma), using the cross-correlation theorem
#
#     g(x,tau) = 2 df sum_k Re{ conj(Ahat(w_k)) Bhat(w_k) exp(-2i w_k tau) }.
#
# The lag is applied SYMMETRICALLY: the forward field is shifted back by tau
# and the adjoint field forward by tau, so tau is the ONE-WAY shift each
# wavefield carries and the RELATIVE shift between them is 2*tau.  That is
# where the 2 in the exponent comes from, and it propagates into two places
# downstream -- the aliasing guard (|tau| < 1/(4 df), not 1/(2 df)) and the
# slope-to-angle map of plot_lag_slant.py (p = -cos(theta)/v, not -2cos/v).
#
# The adjoint source is the DATA RESIDUAL, so g(x,0) is the ordinary FWI
# gradient dJ/dm and the whole gather goes to zero at the solution.  The point
# of the demo is simply to produce that family of gradients, one per lag, in a
# setting where there is something for the lag axis to respond to.
#
# The setting: a HOMOGENEOUS MEDIUM, STARTED FROM A UNIFORMLY FAST MODEL.
#
#   TRUE       Vp = VP0, Vs = VS0, rho = RO0, constant everywhere
#
#   INITIAL    Vp = VP0*(1+FAST), Vs = VS0*(1+FAST), rho = RO0
#              -- the same medium, uniformly FAST by FAST_PCT per cent
#
# FAST and not slow, because that is what puts an open migration ellipse on the
# tau = 0 slice instead of the degenerate S-R segment; the medium block below
# derives this.
#
# This is the simplest setting that still has something on the lag axis, and
# it is deliberately a PURE TRANSMISSION experiment: with no contrast of any
# kind in the true model there is no reflection and, the velocity being
# constant, no diving wave either.  The only arrival is the direct wave, the
# only thing wrong with it is its traveltime, and the gradient is therefore a
# single transmission (banana) kernel per source-receiver pair.
#
# And there is exactly ONE such pair: one source, one receiver, both at
# mid-depth (see the geometry block).  With a single pair there is nothing to
# make the imaging condition interfere constructively anywhere, so the gradient
# is NOT an image -- it is the migration operator itself.  Every time sample of
# the residual maps to the locus
#     |x - S| + |x - R| = v_bg (t_res - t0 - 2 tau),
# which with two point foci is by definition an ELLIPSE through S and R.  A
# band-limited event therefore images as a band of ellipses, and that ringing
# is what the panels show.  It is the experiment, not a bug.
#
# The one hard constraint, used throughout below: no point in the plane has
# |x-S| + |x-R| < |S-R|.  So any residual time whose 2a falls below 2000 m
# images NOTHING, which is why the lag gather is sharply one-sided.
#
# Nothing is done to the data -- the error is entirely in the model, which is
# the whole point of running it this way.  This script GENERATES the family of
# gradients; it does not predict where the energy lands on tau.
#
# There is ONE initial model and ONE lag run, with a DENSE frequency set.  A
# sparse set used to be run alongside it as a counter-example; lagGradientInit
# now rejects df > 1/T outright, so there is nothing left to compare against.
#
# The data and the residual are PRESSURE ONLY, P = (Txx + Tzz)/2 (comp=_rp).
# Txx and Tzz are still recorded because that is what P is built from.
#
# The one number that is CHECKED rather than displayed
# ----------------------------------------------------
# The tau = 0 slice of the SHOT-STACKED cube (<base>_lagvp.su, no shot index)
# must reproduce the conventional gradient written by accumGradient
# (<base>_vp_raw.su, <base>_vs_raw.su).  Both are sums over shots.  That one
# comparison
# exercises the accumulation, the 2*df weight, the conj(A)B complex algebra,
# the 0.25 Txz scatter, the Lame -> velocity chain rule and the loop bounds at
# once: a sign error, a missing factor or a bounds mismatch all show up.
#
# It can NOT be bit-identical.  The DFT slice is band-limited to the requested
# frequencies while accumGradient sums the full spectrum, so the criterion is
# a normalised correlation near 1 with an amplitude ratio near 1.
#
# And tau = 0 is only HALF the check: it is invariant under both a factor-2
# error in the lag scale and a sign flip of tau.  The lag AXIS is calibrated
# separately, by test_lag_gradient lag_shift=N -- delaying the adjoint source
# by N samples must translate the whole cube by exactly +N/2 samples in tau,
# because the lag is applied symmetrically and the relative shift is 2*tau.
#
# grad_taper is ON, to kill the source and receiver singularities.  Without it
# ~99.9% of the gradient energy sits in a disc around the source, whose lag
# response is just the wavelet autocorrelation at tau = 0, and it swamps every
# display.  The taper reaches the lag cube because lagGradientWrite is handed
# the SAME mask that test_gradient_visual applies to the conventional
# gradient -- it cannot inherit it, since adj_shot tapers g*_shot while the lag
# cube is built from lag_gradient's own accumulators.  Applying one mask to
# both sides is also what keeps the tau = 0 check honest.
#
# Everything else that would make the computed gradient differ from the exact
# gradient of the printed J is still off (res_taper, aper_taper, smooth_grad,
# scaling, precond, data_weight).
#
# A third, cheap pass writes wavefield SNAPSHOTS -- the forward in the true
# model, the forward in the background, and the adjoint of the residual -- so
# the thing the gradient cross-correlates can be looked at directly.  It is a
# separate pass rather than a flag on the gradient runs only because the true-
# model forward is not one of them.
#
# The source is an explosion (src_type=1): isotropic radiation puts a strong
# downgoing P directly below it, whereas an Fx source has a P radiation NULL in
# that direction, which would leave the column below the source S-illuminated.
# The module does not care either way.
#
# Usage:  ./test_lag_gradient.sh

# pipefail matters here: without it, running this script under `| tail` reports
# the exit status of tail, so a `set -e` abort inside a modelling call looks
# like success and the only evidence is in log_*.txt.
set -e
set -o pipefail
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}
export OMP_NUM_THREADS=8

WORK=lag_gradient
rm -rf ${WORK}
mkdir -p ${WORK}
cd ${WORK}

# Coarser grid than taylor_vs.sh on purpose: the module holds 6 complex fields
# of nfreq values per grid point, so memory is nfreq * sizem * 48 bytes.
# SIZEZ is 2400, not 2000, so the tau=0 migration ellipse (semi-minor axis
# 1118 m) fits between the source depth and both edges.  See the medium block.
DX=10; SIZEX=4000; SIZEZ=2400

# The medium: homogeneous, and a background that is the same medium uniformly
# FASTER.  Vp and Vs are scaled by the SAME fraction, so Vp/Vs is unchanged and
# the error is purely a uniform time contraction of everything.
#
#                 true     background    error
#   Vp [m/s]      1800        2700       +50%  (too FAST -> synthetic is EARLY)
#   Vs [m/s]      1050        1575       +50%
#   rho [kg/m3]   2000        2000         0
#
# FAST, not slow, and that is the whole point of this configuration.  The zero
# lag imaging condition puts the residual at time t_res on the ellipse
#     |x-S| + |x-R| = 2a = v_bg (t_res - t0 - 2 tau),
# and at tau = 0 the two events in the residual land in completely different
# places:
#
#   SYNTHETIC event, t_syn - t0 = offset/v_bg.  Then
#       2a = v_bg * offset/v_bg = offset,  EXACTLY, for any v_bg.
#   That is an identity, not a coincidence: the background's own direct arrival
#   always images on the DEGENERATE ellipse, the straight S-R segment.  It is
#   the flattest ellipse there is, its only thickness comes from the length of
#   the wavelet, and no velocity error of any size or sign changes it.
#
#   OBSERVED event, t_obs - t0 = offset/v_true.  Then
#       2a = offset * v_bg/v_true,
#   and the sign of the error decides everything, because no point in the plane
#   has |x-S| + |x-R| < |S-R|:
#       v_bg < v_true  ->  2a < offset  ->  images NOTHING at tau = 0.  The
#                          ellipse only opens at tau = -dt/2, i.e. it is pushed
#                          OFF zero lag, further the bigger the error.
#       v_bg > v_true  ->  2a > offset  ->  a REAL, OPEN ELLIPSE AT TAU = 0,
#                          with semi-axes a = (offset/2)(v_bg/v_true) and
#                          b = sqrt(a^2 - (offset/2)^2).
#
# So a slow background can never show a migration ellipse at zero lag, and a
# fast one always does.  Here a = 1500 m and b = 1118 m, and the obs ellipse
# stays open until it degenerates at tau = +185 ms -- so POSITIVE lag is no
# longer empty either, which it was in every slow version of this script.
#
# Separation of the two arrivals.  Over the 2000 m offset
#   syn (background):  2000/2700 + t0 = 0.891 s   <- EARLY
#   obs (true model):  2000/1800 + t0 = 1.261 s
# i.e. 370 ms apart, against a wavelet about 290 ms long, so the residual is
# TWO resolved events rather than one interference blob.
#
# These are not round numbers because four constraints fight each other:
#
#   1. Separation SATURATES, unlike the slow case:
#          dt = offset*(1/v_true - 1/v_bg) = (offset/v_true) * p/(1+p)
#      -> dt < offset/v_true however large p gets.  v_true must be slow enough
#      to give a budget at all.  Here the ceiling is 1.111 s and p=50% spends
#      370 ms of it.  (At VP0=2200 the ceiling is 909 ms and p=40% yields only
#      260 ms, which the wavelet does not resolve.)
#
#   2. The tau=0 ellipse must FIT:  b = (offset/2)*sqrt((1+p)^2 - 1) = 1118 m.
#      That is why SIZEZ is 2400 and not 2000 -- at the old depth the model
#      half-height was 1000 m and the ellipse would be cut off top and bottom,
#      which is precisely the feature this configuration exists to show.
#
#   3. The dispersion guard, fmax < Vmin/(5*dx), is set by the slowest S
#      velocity ANYWHERE, and with a fast background that is the TRUE model,
#      not the background.  A 19.8 Hz wavelet needs Vs_true > 989 m/s.  This is
#      the trap in flipping the sign of the error: the fast background sails
#      through (1575/50 = 31.5 Hz) while the true model, which never had a
#      problem before, becomes the binding one.  Vs0 = 1050 gives 21.0 Hz.
#
#   4. Vp/Vs must stay physical.  Constraint 3 pushes Vs0 UP while constraint 1
#      pushes Vp0 DOWN, and they meet at an unphysical ratio if pressed:
#      Vp0/Vs0 = 1800/1050 = 1.714 is about as far as this can be taken before
#      lambda = rho(Vp^2 - 2Vs^2) goes negative.
#
# Net effect: no `fmax=` override anywhere in this script.  Both models clear
# the guard on their own.
VP0=1800; VS0=1050; RO0=2000
FAST_PCT=50
VP_BG=$(echo "${VP0} ${FAST_PCT}" | awk '{printf "%.4f", $1*(1+$2/100)}')
VS_BG=$(echo "${VS0} ${FAST_PCT}" | awk '{printf "%.4f", $1*(1+$2/100)}')

# Geometry: ONE source and ONE receiver, both at MID-DEPTH, 2000 m apart and
# straddling the centre of the model.  That is the whole point of this version
# of the script -- a single source-receiver pair in a homogeneous medium gives
# exactly ONE transmission kernel, a single banana lying horizontally across
# the middle of the model, and the lag gather of that one object is as simple
# as a lag gather gets.
#
#        x=1000              x=2000              x=3000
#   z=1200   S  ----------------+----------------  R
#                            midpoint
#
# Both are at z = SIZEZ/2 so the banana is as far from every boundary as it can
# be (1200 m of model plus 2000 m of taper on each side), and so that at the
# midpoint the ray from S and the ray to R are mirror images about the
# VERTICAL -- which is the one assumption the theta axis of plot_lag_slant.py
# makes.  Putting the source back at the surface is a one-line change (ZSRC=50)
# but it tilts the bisector and inflates the theta it reports.
XSRC=1000;  ZSRC=$((SIZEZ/2))
XRCV=3000;  ZRCV=$((SIZEZ/2))
OFFSET=$((XRCV - XSRC))

# tmod has to outlast the S arrival in the slowest model -- now the TRUE one,
# 2000/1050 + t0 = 2.06 s -- and the forward field has to have crossed the
# largest isochron the lag axis asks for.  At tau = -300 ms the obs ellipse has
# 2a = 2700*(1.111 + 0.6) = 4620 m, reached at 4620/2700 + t0 = 1.86 s, so the
# S arrival is the binding one here.
TMOD=2.5; DTRCV=0.001

# Absorbing boundary.  ntaper is ADDED outside the model grid (getParameters.c
# naz/nax += bnd->ntap), so a larger value costs compute, never model.  The
# taper profile is exp(-wfct^2) per time step with wfct running from tapfact
# =0.30 at the outer edge to 0 at the model edge, and scl = tapfact/ntap -- so
# the SHAPE is fixed and doubling ntap doubles the number of steps a wave
# spends inside it, i.e. it SQUARES the attenuation.  At 200 points = 2000 m
# that is 6.6 wavelengths at the 6.6 Hz wavelet peak, against the 3-wavelength
# rule of thumb that getParameters uses for its own default.
# Cost: the padded grid goes from 600x400 to 800x600, so runtime roughly x2.
NTAPER=200

# Radius, in grid points, of the parabolic hole punched around every source
# and every receiver.  30 points = 300 m at dx=10.
GRAD_TAPER=30

# Wavelet.  makewave builds a second-derivative-of-Gaussian, and fmax=
# OVERRIDES fp= (verified: fp=8,12,14,16 with fmax=16 all give the same
# wavelet), so the peak sits at 0.4425*fmax.
#
# fmax is bounded by fdelmodc's dispersion guard (getParameters.c:313),
#     wav->fmax < cmin/(dx*5),
# with cmin the SLOWEST velocity in the model.  Two things make this trickier
# than it looks, both of which have already cost a dead run here:
#
#   - wav->fmax is NOT the makewave parameter.  getWaveletInfo re-measures the
#     wavelet's own high-frequency edge from file_src, and that edge is well
#     above the parameter.  Measured:
#         makewave fmax=15  ->  reported 19.775 Hz
#         makewave fmax=12  ->  reported 16.113 Hz
#     i.e. roughly 1.22*F + 1.46.  Budget against 19.8 Hz, not against 15.
#
#   - cmin is an S velocity, and WHICH model owns it depends on the sign of the
#     velocity error.  With a slow background it is the background's Vs; with
#     the FAST background used here it is the TRUE model's Vs = 1050 m/s.
#     Ceiling 1050/50 = 21.0 Hz, cleared by 19.775 with ~6% to spare.
#
# If that margin is ever lost, the honest fixes are a lower WAV_FMAX (longer
# pulse, less event separation) or a smaller dx (more compute).  `fmax=` does
# exist as an override and only relaxes the dispersion comparison -- unlike
# `disable_check=1`, which would also switch off the CFL STABILITY test, a hard
# limit that must stay enforced.  Nothing here needs either.
WAV_FMIN=2; WAV_FMAX=15; WAV_T0=0.15

# Lag axis, in the ONE-WAY tau of the symmetric convention: the relative shift
# between the two wavefields is 2*tau, so this axis spans +-0.60 s of relative
# lag.
#
# Each event rides its own ellipse, 2a(tau) = v_bg (t_res - t0 - 2 tau), which
# SHRINKS as tau increases and dies when 2a reaches the focal separation,
# 2000 m.  With the fast background both events are alive at tau = 0:
#     syn event, t_res = 0.891 s:  2a = 2000 m at tau = 0   -> degenerate, the
#                                  S-R segment; dies immediately for tau > 0
#     obs event, t_res = 1.261 s:  2a = 3000 m at tau = 0   -> OPEN ELLIPSE,
#                                  a = 1500, b = 1118; dies at tau = +185 ms
# so the interval 0 < tau < +185 ms isolates the observed event completely --
# the synthetic one cannot reach there.  Going negative inflates both.
#
# +-300 ms covers that whole positive window with room to spare, and on the
# negative side carries both ellipses until they outgrow the model.
LAG_NT=121; LAG_DT=0.005           # tau in [-0.30, +0.30] s

# Display window: the clips and the ||g(tau)|| curves are measured here, not
# globally.  It is centred on a DEPTH, z = ZSRC, not on a reflector -- there is
# no reflector.  It is the FULL depth range, because the tau = 0 ellipse alone
# already has b = 1118 m and anything narrower would clip the main feature.
ZFOCUS=$((SIZEZ/2)); ZWIN=$((SIZEZ/2))   # display window 0 - 2400 m

# Aliasing: the reconstruction repeats every 1/(2 df) on the tau axis, so the
# usable half-range is 1/(4 df).
#
# Dense set: df = 0.25 Hz -> usable |tau| < 1.0 s, well outside the +-0.4 s
# axis.  2-24 Hz covers the whole wavelet including its taper (its measured
# edge is 19.8 Hz).
#
# That df is set by the tau = 0 CHECK, not by the lag axis, and the two are not
# the same requirement.  The check compares the tau = 0 slice against
# accumGradient over the WHOLE grid, and the relative lag a grid point carries,
#     lambda = (t_res - t0) - (|x-S| + |x-R|)/v_bg,
# runs over the FULL grid, not just the illuminated banana.  Here the sum of
# distances reaches 6400 m in the far corners, so lambda spans about 2.7 s.
# Anything beyond the 1/df reconstruction period wraps -- onto tau = 0 and
# everywhere else.
#
# Measured on the earlier 8% version of this script, where lambda spanned
# ~2.3 s:
#     df = 1.0 Hz (nfreq=23)  ->  corr = 0.884 / 0.944, ratio 1.13 / 1.06
#     df = 0.5 Hz (nfreq=45)  ->  corr = 0.999998       , ratio 1.0002
# and the 1 Hz error was NOT a loss of band -- the band was identical in both
# runs.  It was a flat, grid-filling wrap-around floor, visible as a spurious
# outer arc near x = 3800 and as a gradient at tau = +150 ms that was the same
# order as the one at tau = 0, where the geometry says it must be exactly zero.
#
# So: pick df from the relative-lag SUPPORT, not from the lag range you intend
# to display.  conj(A)B is a correlation over the whole record, so that support
# is bounded by the record length itself and the requirement is simply
#
#     df <= 1/T,    T = tmod
#
# i.e. at least the natural frequency resolution of the record.  Here T = 2.5 s
# gives df <= 0.4 Hz, so nfreq >= (24-2)*2.5 + 1 = 56 over this band; 89
# (df = 0.25) is used because it also covers the ~2.7 s of relative lag that
# the far corners of THIS grid carry.  lagGradientInit enforces df <= 1/T and
# aborts below it -- a sparser set is not a cheaper approximation of the lag
# gradient, it folds real correlation energy back into the lag window.
# Cost: 2.1 GB of accumulators, independent of the number of lags.
D_FMIN=2; D_FMAX=24; D_NFREQ=89

# Wavefield snapshots.  Written by a SEPARATE pass (see below).  tsnap2 must
# stay inside tmod; the S arrival in the slowest (TRUE) model is at
# 2000/1050 + t0 = 2.06 s.
TSNAP1=0.1; TSNAP2=2.4; DTSNAP=0.1 # 24 frames

echo "--- true and initial models ---"
# Both are homogeneous, so makemod builds them directly -- no interfaces, no
# numpy.  (makemod is inclusive of both endpoints, so the grid is 401 x 241,
# not 400 x 240; do not hardwire SIZEX/DX anywhere downstream.)
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=model.su verbose=0
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP_BG} cs0=${VS_BG} ro0=${RO0} \
    orig=0,0 file_base=bg.su verbose=0
printf '  %-10s %10s %10s %9s\n' '' true background error
printf '  %-10s %10.1f %10.1f %8.1f%%\n' \
    Vp   ${VP0} ${VP_BG} +${FAST_PCT} \
    Vs   ${VS0} ${VS_BG} +${FAST_PCT} \
    rho  ${RO0} ${RO0}   0
echo "  grid $(surange < model_cp.su | \
    awk '/traces:/{nx=$1} $1=="ns"{nz=$2} END{print nx" x "nz}')  (nx x nz)"

echo "--- wavelet ---"
makewave fmin=${WAV_FMIN} fmax=${WAV_FMAX} \
    dt=0.001 nt=4096 file_out=wave.su t0=${WAV_T0} verbose=0

echo "--- observed data (TRUE model, explosive source, 1 shot, 1 receiver) ---"
# fdelfwi_fwd writes <file_rcv>_<NNN>_r<comp>.su already zero-padded, which is
# what the gradient code reads.
#
# ONE receiver, given as an explicit position list (xrcva/zrcva), NOT as a
# degenerate linear array.  The xrcv1/xrcv2 path cannot express a single
# receiver: recvPar.c has a branch for dxrcv=dzrcv=0 that sets dxrcv=1 and
# announces "Placing 1 receiver at ..." (line 207), but the very next loop
# (line 216) sees xrcv1==xrcv2 and resets dxrcv back to 0, after which the
# receiver count errors out with "receiver distance dzrcv is not given".  Any
# nonzero dxrcv hits the same reset.  xrcva/zrcva bypasses all of it -- the
# positions are taken verbatim and the linear-array block is never entered.
${FDELFWI}/fdelfwi_fwd \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=obs \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
    xrcva=${XRCV} zrcva=${ZRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
    ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 > /dev/null
echo "  S at (${XSRC}, ${ZSRC}) m,  R at (${XRCV}, ${ZRCV}) m,  offset ${OFFSET} m"
echo "  traces per component: $(surange < obs_000_rtzz.su | \
    awk '/traces:/{print $1}')"

# The hydrophone, P = 0.5*(Txx + Tzz).  fdelmodc's rec_type_p does NOT give a
# proper elastic hydrophone, and fdelfwi_fwd does not write _rp at all, so the
# OBSERVED pressure has to be built here.  (test_gradient_visual builds the
# SYNTHETIC one itself, in computeHydrophone, whenever ischeme > 2 -- which is
# why rec_type_txx/tzz stay switched on in the runs below even though the only
# component used is _rp.)
echo "--- observed hydrophone P = (Txx+Tzz)/2 ---"
suop2 obs_000_rtxx.su obs_000_rtzz.su op=sum | \
    sugain scale=0.5 > obs_000_rp.su
echo "  obs_000_rp.su"

# ---------------------------------------------------------------------------
# Wavefield snapshots, one shot only
# ---------------------------------------------------------------------------
# Three wavefields for the SAME shot, so they can be compared frame by frame:
#   snap_true_*   forward in the TRUE model        -> what the data contain
#   snap_syn_*    forward in the INITIAL model     -> what the model predicts
#   snap_adj_*    the adjoint of the residual      -> what is backpropagated
#
# The pair that matters is snap_true vs snap_syn.  Both are a single expanding
# circular front -- there is nothing in either model to scatter off -- but the
# background front is 30% smaller at every instant, so by the time the fronts
# reach the receiver they are 600 m apart and the difference between the two
# frames is TWO separate rings, not one annulus.  That separation is the whole
# point of the large error, and it is what puts two events in the residual.
# The gap grows linearly with radius, which is exactly the uniform time
# dilation being inverted for.
#
# There is only one shot, so these read the SAME obs_000_* files as the
# gradient runs; no copy is needed.  Everything else (ntaper, grad_taper,
# chk_skipdt) matches the gradient runs so the snapshots describe those runs.
#
# NOTE: the adjoint marches BACKWARD, so its frames come out in reverse
# chronological order -- frame 1 is at TSNAP2, frame nsnap at TSNAP1.  The
# headers are relabelled to look forward (adj_shot.c, fake_it), so only the
# plot script knows this; it flips them.
#
# sna_type_p is FORCED to 0 for ischeme>2 (getParameters.c), so ask for tzz.
# pp = div(v) and ss = curl(v): the P and S parts of the wavefield, which is
# what makes a mode conversion visible in a single frame.
SNAP_TYPES="sna_type_vx=1 sna_type_vz=1 sna_type_tzz=1 sna_type_pp=1 sna_type_ss=1"
SNAP_TIME="tsnap1=${TSNAP1} tsnap2=${TSNAP2} dtsnap=${DTSNAP}"

echo "--- snapshots: S at (${XSRC}, ${ZSRC}) m ---"

# (a) true model
${FDELFWI}/fdelfwi_fwd \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=snaptrue \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
    xrcva=${XRCV} zrcva=${ZRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
    ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 \
    file_snap=snap_true ${SNAP_TIME} ${SNAP_TYPES} > log_snap_true.txt 2>&1

# (b) + (c) background forward and the adjoint of its residual
${FDELFWI}/test_gradient_visual \
    file_cp=bg_cp.su file_cs=bg_cs.su file_den=bg_ro.su \
    file_src=wave.su file_rcv=snapsyn \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=${DTRCV} tmod=${TMOD} verbose=1 \
    xrcva=${XRCV} zrcva=${ZRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
    ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=0 aper_taper=0 \
    left=4 right=4 top=4 bottom=4 \
    chk_skipdt=50 chk_base=chk_snap \
    param=2 scaling=0 precond=0 smooth_grad=0 \
    vp_min=1000 vp_max=4500 vs_min=500 vs_max=2500 \
    rho_min=1500 rho_max=2500 \
    data_weight=0 misfit=0 \
    comp=_rp \
    file_obs=obs out_base=snaprun \
    file_snap=snap_syn file_snap_adj=snap_adj \
    ${SNAP_TIME} ${SNAP_TYPES} > log_snap_syn.txt 2>&1
echo "  snap_true_*, snap_syn_*, snap_adj_*  ($(ls snap_*_s*.su | wc -l) files)"

# run_lag <out_base> <lag_nfreq> <lag_fmin> <lag_fmax> <lag_nt> <log>
# Called once; kept as a function so the frequency set and the lag axis stay
# the only things a second call would have to change.
run_lag () {
    ${FDELFWI}/test_gradient_visual \
        file_cp=bg_cp.su file_cs=bg_cs.su file_den=bg_ro.su \
        file_src=wave.su file_rcv=syn_$1 \
        ischeme=3 iorder=4 src_type=1 \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=0 aper_taper=0 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=50 chk_base=chk_$1 \
        param=2 scaling=0 precond=0 smooth_grad=0 \
        vp_min=1000 vp_max=4500 vs_min=500 vs_max=2500 \
        rho_min=1500 rho_max=2500 \
        data_weight=0 misfit=0 \
        comp=_rp \
        lag_nfreq=$2 lag_fmin=$3 lag_fmax=$4 \
        lag_nt=$5 lag_dt=${LAG_DT} \
        file_obs=obs out_base=$1 > $6 2>&1
}

echo "--- lag run: dense frequency set (nfreq=${D_NFREQ}, ${D_FMIN}-${D_FMAX} Hz) ---"
run_lag dense ${D_NFREQ} ${D_FMIN} ${D_FMAX} ${LAG_NT} log_dense.txt
grep -E "lagGradientInit|lagGradientWriteStack" log_dense.txt | head -4 || true
grep -E "Total misfit" log_dense.txt | tail -1

echo ""
echo "================ figures ================"
# CIG locations: the midpoint of the single S-R pair and two on either side,
# all inside the banana (x = 1000 .. 3000).  The midpoint is where the two rays
# are mirror images about the vertical, so it is the only column where the
# theta axis of the slant stack is free of the bisector-tilt bias; the others
# are there to show that bias growing.
XMID=$(( (XSRC + XRCV) / 2 ))
XCIG=1400,1700,${XMID},2300,2600
python3 ../plot_lag_gradient.py \
    --zfocus=${ZFOCUS} --zwin=${ZWIN} --xgather=${XMID} \
    --xcig=${XCIG} dense

echo ""
# The same cube as three faces at once, then one animation per axis.  Those
# three faces ARE the static figures above -- the main panel is one lag of
# lag_dense_panels.png, the right face one column of lag_dense_cig.png -- so
# the only thing the animation adds is that they move together.  Worth the
# ~2 min and ~50 MB the first time you look at a new cube and not after, hence
# the switch.  LAG_GIF=0 ./test_lag_gradient.sh skips it.
if [ "${LAG_GIF:-1}" != "0" ]; then
    python3 ../plot_lag_cube.py --par=vp,vs --clipq=99.9 \
        --zfocus=${ZFOCUS} --zwin=${ZWIN} --xslice=${XMID} \
        --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} dense
fi

echo ""
# Slant stack of those same CIGs: collapses the tilt of each band to a point on
# a slope axis p = dtau/dz, then relabels it as theta = arccos(-p v).  Display
# only -- nothing is picked or inverted.
#
# The window is the WHOLE depth axis.  The tau = 0 ellipse spans z = 82 to
# 2318 m at the midpoint, so a window around z = ZSRC sits on its flat APEX and
# cuts both flanks off -- which is why it used to report p = 0 / theta = 90 deg
# at every x.  Over the full depth the flanks dominate and the stack resolves
# them as the mirror pair at +p and -p that they are: g_vp near +-210 ms/km
# (theta = 125 deg and its bisector-flipped twin near 56 deg), g_vs near
# +-220 ms/km (110 / 70 deg).  Read |p|; the sign only says which flank.
python3 ../plot_lag_slant.py --xcig=${XCIG} dense

echo ""
# DSR Radon of the same CIGs.  The bands are not straight lines, so the linear
# stack above can only ever report their flank dips as a +p/-p pair.  Their
# EXACT shape is tau(z) = T/2 - D(z)/(2v) with D the double square root
# |x-S| + |x-R|, so stacking along that curve collapses each band to a single
# point at its own APEX LAG tau0 -- half that event's traveltime residual.
#
# tau0 is the axis the two kernels separate on, and the separation is a
# property of the imaging condition, not of this model: the background's own
# arrival images on the degenerate ellipse at tau0 = 0 identically (see the
# medium block at the top), so the transmission kernel sits at zero lag while
# anything mistimed against it sits at tau0 = dt/2.  That is what makes a mute
# on tau0 a usable separation.  The v axis is a focusing QC, not an inversion:
# both kernels share one background, so both focus at the same v.
#
# The geometry is passed in rather than duplicated so the moveout cannot drift
# away from the acquisition actually modelled.
python3 ../plot_lag_radon.py --xcig=${XCIG} \
    --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} dense

echo ""
# Act on that separation: keep only the tomographic kernel in g_vp, with a
# Gaussian in the SLANT-STACK domain centred on (p, tau0) = (0, 0) and cut at
# RMAX sigma.  Referred to the apex depth, the transmission kernel IS that
# origin and the migration kernel is a second point at tau0 = dt/2, so the two
# separate on both axes at once -- the migration kernel is rejected by its
# apex lag as well as by its flank dip.
#
# The slant stack is built per lag-frequency, where a straight line is a phase
# multiply, so there is no interpolation and the model is a damped LEAST
# SQUARES fit rather than the adjoint.  That makes the transform an identity
# when the Gaussian is 1 (the run prints the measured round trip), which is
# what lets the difference between raw and filtered be attributed to the
# Gaussian instead of to the operator.
#
# SIGTHETA is not a free knob: it also sets how much of the ray survives,
# |z - z_apex| <= a_s*tan(SIGTHETA).
#
# The reference depth is z_ref = 0, the sutaup convention (fwd_tx_sstack uses
# delay = p*x/dt about the coordinate origin, taup.c:545; suradon exposes the
# same thing as interoff, also 0 by default).  That is deliberate: at p = 0 the
# stack line is tau = tau0 whatever z_ref is, so BOTH kernels keep their bowtie
# vertex on the tau0 axis -- transmission at 0, migration at dt/2 -- and the
# Gaussian picks the right one without being told where the kernel sits.  The
# panel is then directly comparable with slant_<base>.png, which uses the same
# reference and the same convention.
#
# The cost is that only the CENTRE is z_ref invariant, not the width: a slope p
# displaces the line by p*z, so at z = 1200 m the migration kernel is reachable
# at p = 154 ms/km = 1.6 sig_p and some of it leaks.  Measured here (energy
# within +-200 m of the S-R segment vs of the tau = 0 isochron, in the
# aperture):
#
#     raw                                   ray 35.1%  ellipse 18.16%  ratio 1.9
#     z_ref = 0  (default, sutaup)          ray 57.1%  ellipse  2.18%  ratio  26
#     --lwin=800 (sliding, no geometry)     ray 63.0%  ellipse  0.51%  ratio 123
#     --zref=1200 (an ORACLE, see below)    ray 77.7%  ellipse  0.09%  ratio 838
#
# The last row is not an option on real data: 1200 m is the ray depth, which is
# known here only because the source and receiver were put at mid-depth.  It is
# quoted as the ceiling, not as a setting.  --lwin=L stacks over a sliding
# window whose reference IS the output depth, which needs no geometry at all.
#
# --xcig gives lag_filter_xcig_dense_vp.png: the same Gaussian at each of those x,
# with the slant stack, the filtered stack and the resulting g(tau = 0) profile
# on a COMMON clip, so the x dependence is read off instead of inferred from the
# single --xgather panel.  The filter itself already ran at every x -- these are
# the same XCIG locations used by plot_lag_slant.py and plot_lag_radon.py.
python3 ../plot_lag_filter.py --mode=gauss --par=vp \
    --sigtheta=15 --sigtau=40 --rmax=3 --xgather=${XMID} --xcig=${XCIG} \
    --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} dense

echo ""
# The same angle filter in the Sava & Fomel formulation, which differs in two
# ways that are worth having side by side.
#
# (1) The pivot.  The lines are tau = p (z - z0), pivoted on the ZERO-LAG axis
#     at the output depth, so a row of the output is the angle gather at that
#     depth and cos(theta) = v|p| labels it directly.  There is no z_ref to
#     choose, so the whole --zref / --lwin discussion above does not arise.
# (2) The inverse is EXACT, not damped least squares.  Ramp-filter along tau,
#     slant stack, sum the row: g(z0, 0) = (dp/2pi) sum_p A_rho(z0, p), because
#     every line in row z0 crosses tau = 0 at z0.  Filtering is then a weighted
#     row sum and there is no matrix anywhere.  The script prints the W = 1
#     identity (corr +0.9998 here, against +0.9997 for the damped solve) and
#     the agreement between its own two code paths (1e-14, i.e. exact).
#
# The Hann window costs resolution, NOT exactness -- windowing replaces g by
# w(u) g(z0+u) and the same change of variable returns w(0) g(z0,0), so any
# window that is 1 at its centre reconstructs.  Printed dp_res ~ T_tau/(2W).
python3 ../plot_lag_angle.py --par=vp,vs --mode=tomo \
    --sigtheta=15 --lwin=800 --xcig=${XCIG} dense

echo ""
# The same thing with the NEIGHBOURING CIGs folded in.  One CIG only has
# p_z = dtau/dz, so it cannot tell a steep event at small dip from a shallow
# one at large dip -- it folds both onto the same p_z, and Sava & Fomel's dip
# correction cos(theta) = v|tau_z|sqrt(1+z_x^2) needs a z_x the panel does not
# contain.  Pivoting in x as well, tau = p_x(x-x0) + p_z(z-z0), measures the
# slope VECTOR, so |p| = |k|/omega_tau is the real thing and the dip
# correction is obtained exactly rather than estimated.  The ramp goes from
# |f| to f^2 (d2k = omega^2 d2p) and the measure to dpx dpz dx dz.
#
# lag_angle_k_dense_vp.png is the payoff: A(p_x,p_z) on a grid of points, with
# radius = angle and azimuth = bisector dip.  The PP / PS / SS rings on it are
# |p| <= (1/v_f + 1/v_a)/2, which is 1/Vp only when both legs are P -- so the
# radial axis is a MODE axis too, and cos(theta) = v|p| labels the PP disk and
# nothing outside it.  On this run the transmission kernel is 100.0% inside
# the PP ring on the S-R segment and 91% off it, the rest being conversion.
python3 ../plot_lag_angle.py --par=vp --mode=tomo \
    --sigtheta=15 --lwin=800 --neigh=800 --xcig=${XCIG} dense

echo ""
# One model, so one residual.  The
# geometry and the velocities are passed in rather than duplicated, so the
# analytic arrival markers cannot drift away from the model actually built.
python3 ../plot_lag_residual.py \
    --offset=${OFFSET} --vp=${VP0} --vs=${VS0} \
    --vpbg=${VP_BG} --vsbg=${VS_BG} --t0=${WAV_T0} dense

echo ""
# pp, vx and vz.  pp = div v is the P wavefield and is what the pressure data
# sees; vx and vz are the two components the elastic kernel actually steps, and
# they carry the radiation pattern that pp hides -- vx has a node on the
# vertical through the source and vz on the horizontal, because the particle
# velocity of this explosive source is radial.  The velocities and t0 are used
# only by the direction check the script prints for that pair.
python3 ../plot_lag_snapshots.py \
    --dx=${DX} --tsnap1=${TSNAP1} --dtsnap=${DTSNAP} \
    --xsrc=${XSRC} --zsrc=${ZSRC} --field=pp,vx,vz \
    --vp=${VP0} --vpbg=${VP_BG} --t0=${WAV_T0}

echo ""
echo "Figures and numbers are in $(pwd)"
