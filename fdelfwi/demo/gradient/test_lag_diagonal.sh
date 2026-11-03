#!/bin/bash
# test_lag_diagonal.sh -- the lag-extended Vp and Vs gradients for ONE source
#                         and ONE receiver on a DIAGONAL, with a smooth Vp
#                         anomaly and a smooth Vs anomaly at DIFFERENT places
#                         on the ray, under two acquisitions that share one
#                         Fx source and differ ONLY in the receiver:
#
#                             a)  Fx source + hydrophone   (blind to S)
#                             b)  Fx source + 2C geophone  (sees both modes)
#
# What this experiment is for
# ---------------------------
# Everything else in this directory has the source and the receiver at the same
# depth, which is a geometric accident with physical consequences:
#
#   - An Fx force radiates P as cos(phi) and S as sin(phi), phi measured from
#     the x axis.  On a HORIZONTAL path phi = 0: maximum P, ZERO S.  On a
#     vertical path the reverse.  Only a DIAGONAL path gets both modes out of a
#     single force, and that is the whole premise here: the geophone case needs
#     a real S arrival to record, and a horizontal geometry would not produce
#     one.
#   - A dipping kernel also has p_x =/= 0, so the lag lines are not symmetric
#     about the vertical.  That matters for any later slope/angle analysis of
#     these cubes, which is a separate exercise -- this script only generates
#     the lag gradients and the CIGs.
#
# S = (XSRC, ZSRC), R = (XRCV, ZRCV) with dx = 2400, dz = 1800: a 3-4-5
# triangle, so the path is exactly 3000 m long and dips at 36.87 deg.  The Fx
# radiation is then cos = 0.8 in P and sin = 0.6 in S -- both modes present,
# neither marginal.
#
# The models -- the error is in the BACKGROUND, and it does not scatter
# ----------------------------------------------------------------------
# The background is HOMOGENEOUS: Vp0, Vs0, RO0, no contrast anywhere.  The true
# model is the same thing with TWO Gaussian velocity anomalies added
# multiplicatively, v = v0 (1 + a exp(-r^2/2 sigma^2)), a = APCT/100 < 0:
#
#     site A, at 30% of the path,   Vp ONLY,   Vs and rho untouched
#     site B, at 70% of the path,   Vs ONLY,   Vp and rho untouched
#
# Perturbing Vp alone at A leaves mu alone and moves lambda; perturbing Vs
# alone at B moves mu and moves lambda back so that Vp = sqrt((lam+2mu)/rho) is
# unchanged.  So the two events in the data are delayed by ONE anomaly each:
#
#     the P arrival is late because of site A only
#     the S arrival is late because of site B only
#
# That separation is the point of the experiment, and it is what makes the
# comparison measurable: the hydrophone case holds only the P arrival, so the
# Vs site is reached only indirectly, while the geophone case records the S
# arrival that is delayed by the Vs site itself.  The CIGs say how much that
# changes g_vs, and at which lag.
#
# "Very smooth, so there is no scattering" is a statement about wavenumber, and
# it is quantitative.  The spatial transform of a Gaussian of width sigma is
# exp(-k^2 sigma^2 / 2); backscatter needs 2k, so its amplitude goes as
# exp(-2 k^2 sigma^2).  At the 11.1 Hz wavelet peak, k_S sigma = 18.3 and that
# factor is e^-669.  There is no reflection off these anomalies at any
# frequency this wavelet carries -- only transmission.  The script prints the
# number; it does not ask to be believed.
#
# What survives is the FORWARD-scattering (banana-doughnut) kernel.  The
# Fresnel half width at mid path, sqrt(lambda L)/2, is 368 m in P and 285 m in
# S, against sigma = 300 m.  Raising the frequency moved S across that line:
# the S Fresnel zone is now NARROWER than the anomaly, so the S delay no longer
# heals and the measured S shift should approach the ray value, while P
# (368 > 300) still heals partially and stays BELOW the ray-theoretical
# Leff (1/v_an - 1/v0) quoted in the table.  That is the correct physics for a
# finite-frequency kernel, not an error, and it is the reason the cycle-skip
# check below is a measurement rather than the ray prediction.
#
# SLOW anomalies, i.e. a FAST background, for the usual reason: at tau = 0 a
# residual at time t images on |x-S| + |x-R| = v_bg (t - t0 - 2 tau), the
# background's own arrival always images on the degenerate S-R segment whatever
# the error, and the observed one opens into a real ellipse only when the
# background is the fast model.  A slow background would push the structure off
# zero lag instead of opening it there.
#
# NO CYCLE SKIPPING
# -----------------
# Budget is |dt| < T/2 at the wavelet peak.  makewave puts the peak at
# 0.4425*fmax, so WAV_FMAX = 25 gives f_peak = 11.06 Hz, T = 90.4 ms, half
# period 45.2 ms.  Ray-theoretical shifts through Leff = sigma sqrt(2 pi) =
# 752 m:
#
#     P through site A   19.8 ms   0.219 cycles
#     S through site B   33.0 ms   0.365 cycles    <- binding
#
# So both events are sub-half-cycle mistimings: obs and syn overlap, the
# residual is a timing residual, and the gradient points downhill.
#
# This is the trade that sets APCT.  The budget is a budget in CYCLES, so
# raising the frequency FORCES a weaker anomaly: the same a = -8% that gave
# 0.362 cycles at WAV_FMAX = 15 gives 0.584 at 25 and is cycle skipped.  Hence
# a = -5%, which is as large as the budget allows at this frequency.
#
# APCT is bounded from the other side by the dispersion guard,
# wav->fmax < cmin/(5 dx) with cmin the slowest velocity ANYWHERE in either
# model: a slow Vs anomaly lowers cmin, and the re-measured wavelet edge is
# about 1.22*WAV_FMAX + 1.46 = 32.0 Hz, so Vs0 (1 + a) must stay above
# 5 dx * 32.0 = 959 m/s.  At Vs0 = 1200, a = -0.05 and dx = 6 that is 1140
# against a guard of 38.0 Hz, a 19% headroom, and the script prints it.  dx had
# to come down from 10 to 6 for that: at dx = 10 the guard is 22.8 Hz and a
# 32 Hz wavelet aborts the run.
#
# The two cases
# -------------
# Both use src_type=6, Fx, a horizontal point force (src_type is from
# fdelmodc.c, where 6 = Fx).  Holding the SOURCE fixed and changing only the
# RECEIVER is what makes the pair a controlled comparison: the illuminating
# wavefield is identical, so any difference between the two gradients is a
# property of what the receiver is sensitive to and nothing else.
#
#   case_fx_hydro     src_type=6  comp=_rp              The pressure is BLIND
#                                                       to S (div u = 0 for a
#                                                       plane S wave), so the
#                                                       residual holds only the
#                                                       P arrival even though
#                                                       Fx radiates both.  All
#                                                       of g_vs is therefore
#                                                       reached through P, and
#                                                       the Vs site is informed
#                                                       only by P/S crosstalk
#                                                       and by the P leg that
#                                                       passes through it.
#   case_fx_geo       src_type=6  comp=_rvx,_rvz        Both modes radiated AND
#                                                       both recorded, so the
#                                                       S arrival enters the
#                                                       residual and the Vs
#                                                       site is illuminated
#                                                       directly.
#
# An Fx source radiates P as cos and S as sin of the angle from the x axis, so
# a horizontal path would give max P and zero S and a vertical path the
# reverse.  Only a DIAGONAL path excites both from one force, which is why the
# geometry is dipping and not cosmetic.
#
# The two cases SHARE their observed data, since only comp= differs.  Both
# record vx, vz, Txx, Tzz from the true model regardless, and both keep
# rec_type_txx/tzz on in the gradient runs, because the SYNTHETIC hydrophone is
# built inside test_gradient_visual by computeHydrophone.  comp= is the ONLY
# parameter that differs between the two gradient runs.
#
# No data weighting (data_weight=0, misfit=0), and here that is safe rather
# than merely convenient: each case carries a SINGLE unit -- pressure in Pa for
# the hydrophone, velocity in m/s for the two geophone components -- so nothing
# is ever summed across units and no weight is needed to make a sum meaningful.
#
# An earlier version of this script had a third case, Fx + OBN
# (comp=_rvx,_rvz,_rp), and that case was NOT safe at data_weight=0.  The
# gradient is exactly linear in the adjoint source and the components add, so
# g_OBN = g_geophone + g_hydrophone; the run measured max|g_vp| = 1.28e-24 for
# the geophone against 8.39e-12 for the OBN total, i.e. the geophone entered at
# 1.5e-13 and was annihilated.  "OBN" at data_weight=0 is the hydrophone alone
# with the geophone rounded away.  Mixing units in one misfit needs
# data_weight=2 (the energy norm, Q = diag(rho I, S)); that case is dropped
# here rather than left in a form that silently means something else.
#
# The consequence to keep in mind: the two surviving cases still cannot be
# compared in ABSOLUTE amplitude against each other, because Pa and m/s are
# not commensurable.  Compare the SHAPE of g(x,tau).  Every figure is clipped
# per panel for exactly this reason.
#
# Figures
# -------
# Once, in the work directory, from plot_lag_models.py:
#   models.png             true and background model and the per cent
#                          difference, Vp / Vs / rho
#
# Per case, from plot_lag_gradient.py:
#   lag_dense_panels.png   g_vp and g_vs maps at a series of lags tau
#   lag_dense_cig.png      the (z,tau) gathers at each --xcig column
#   lag_dense_gather.png   the single gather at the focus depth
#   lag_norms.png          ||g(tau)||^2 against tau, where the apex lag shows
# from plot_lag_cube.py, the same cube animated along the lag axis:
#   lag_dense_vp_cube.png      the three faces, static
#   lag_dense_vs_cube.png
#   lag_dense_vp_sweep_tau.gif the x-z map walked through every tau
#   lag_dense_vs_sweep_tau.gif
# from plot_lag_snapshots.py, the two wavefields the gradient correlates:
#   snap_frames_vx.png     forward and adjoint vx at SNAP_TIME
#   snap_frames_vz.png     the same for vz
# from plot_lag_slant.py, the same CIGs turned into an opening-angle gather:
#   slant_dense.png        A(z,theta), the stack anchored at tau=0 at each z
#   slant_dense_power.png  energy vs theta, and the peak theta against depth
# from plot_lag_filter.py, the tomographic kernel EXTRACTED, per parameter:
#   dense_vp_tomo.su / dense_vs_tomo.su   the filtered gradient itself
#   lag_filter_dense_{vp,vs}.png          raw / kept / rejected at tau=0
#   lag_filter_cig_dense_{vp,vs}.png      the filter in both domains at XMID
# plus res_dense_<comp>_trace.png, which is not decoration: it carries the one
# MEASURED number the whole premise rests on, the obs-syn lag in cycles, and so
# is the guard that this setup is not cycle skipped.
# Across the cases, from plot_lag_cases.py: cases_tau0.png and cases_cig.png.
#
# The snapshots are written by the SAME forward and adjoint pass that builds
# the lag gradient -- file_snap/file_snap_adj cost disk and nothing else -- so
# they are the actual wavefields behind the kernel, not a re-run of it.
#
# Usage
#   ./test_lag_diagonal.sh

set -e
set -o pipefail
ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
export PATH=${ROOT}/bin:/rcp3/software/codes/cwp/bin:${PATH}
export OMP_NUM_THREADS=16

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${HERE}/lag_diagonal
rm -rf ${WORK}
mkdir -p ${WORK}
cd ${WORK}

# SIZEX is 5004 and not 5000 so that it is an exact multiple of dx: 834*6.
DX=6; SIZEX=5004; SIZEZ=3000

# Homogeneous background.  Vs0 = 1200 and not the 1150 used elsewhere in this
# directory purely for the dispersion headroom computed in the header.
VP0=2000; VS0=1200; RO0=2000

# Geometry.  dx = 2400, dz = 1800 -> path exactly 3000 m, dip 36.87 deg.  The
# pair is CENTRED on the domain, (2502, 1500), so the two ends have equal room,
# and every coordinate below -- S, R and both anomaly sites -- is a multiple of
# dx = 6, so no site is interpolated onto the grid.
XSRC=1302; ZSRC=600
XRCV=3702; ZRCV=2400

# Anomalies: fractional position along the ray, and the perturbation.
FRAC_A=0.30          # Vp anomaly -> (2022, 1140)
FRAC_B=0.70          # Vs anomaly -> (2982, 1860)
SIGMA=300            # Gaussian sigma [m]
APCT=-5              # per cent, SLOW

TMOD=3.1; DTRCV=0.001
# NTAPER is in GRID POINTS, and it EXTENDS the compute grid rather than eating
# into the model.  Holding it at 200 while halving dx holds the absorbing pad
# at the same thickness in WAVELENGTHS -- 200*6 = 1200 m = 11.1 lambda_S at the
# 11.06 Hz peak, against 200*10 = 2000 m = 11.1 lambda_S at the old 6.64 Hz.
# Same absorption, and the pad does not grow with the model.
NTAPER=200
# Likewise GRAD_TAPER: 30*6 = 180 m = 1.66 lambda_S, the same count as the old
# 30*10 = 300 m at 180.7 m wavelengths.
GRAD_TAPER=30
WAV_FMIN=2; WAV_FMAX=25; WAV_T0=0.15

# Lag axis.  The apex lag of an event is HALF its traveltime residual, so the
# two events sit near tau0 = 9.9 ms (P) and 16.5 ms (S); 2 ms samples put 5.0
# and 8.2 samples between them and zero lag.
LAG_NT=101; LAG_DT=0.002             # tau in [-0.10, +0.10] s

# Display window: the ray runs from z = 600 to z = 2400, so this is centred on
# its mid depth and holds all of it.
ZFOCUS=1500; ZWIN=2000

# Wavefield snapshots.  ONE schedule, shared by the forward and the adjoint
# pass -- file_snap_adj reuses tsnap1/tsnap2/dtsnap rather than taking its own,
# which is what makes the two sets frame-for-frame comparable.
TSNAP1=0.1; TSNAP2=3.0; DTSNAP=0.1
# Frame times for the panel figure, chosen on the physics and snapped onto the
# 0.1 s frame grid:
#   0.6  P through site A   (900 m along the ray:  900/2000 + 0.15 = 0.60)
#   1.0  mid path
#   1.7  P at the receiver  (1.65 s)
#   1.9  S through site B   (2100 m along the ray: 2100/1200 + 0.15 = 1.90)
#   2.7  S at the receiver  (2.65 s)
SNAP_TIME=0.6,1.0,1.7,1.9,2.7

# Frequency set.  The binding constraint is the one lagGradientInit enforces,
# df <= 1/T with T the record length: conj(A)B is a correlation over the whole
# record, so its support in RELATIVE lag is T and anything outside the 1/df
# reconstruction period folds back onto the displayed lags.  T = 3.1 s needs
# df <= 0.3226 Hz; df = 0.2 Hz over 2-36 Hz is 171 frequencies and df*T = 0.62,
# the same margin that measured corr = 0.999999 at tau = 0 in
# test_lag_reflector.sh.  Aliasing on the tau axis itself needs
# |tau| < 1/(4 df) = 1.25 s, against the 0.10 s asked for.  D_FMAX = 36 sits
# above the measured wavelet edge, 1.22*25 + 1.46 = 32.0 Hz.
D_FMIN=2; D_FMAX=36; D_NFREQ=171

# ---------------------------------------------------------------------------
# Derived geometry, delays and the cycle-skip budget
# ---------------------------------------------------------------------------
# Computed from the parameters above rather than written out, so the numbers
# printed cannot drift away from the model actually built.
read XA ZA XB ZB SEP LPATH LEFF T_P_T T_P_B T_S_T T_S_B FPEAK KSIG <<EOF
$(echo "${XSRC} ${ZSRC} ${XRCV} ${ZRCV} ${FRAC_A} ${FRAC_B} ${SIGMA} \
        ${APCT} ${VP0} ${VS0} ${WAV_T0} ${WAV_FMAX}" | \
awk '{sx=$1; sz=$2; rx=$3; rz=$4; fa=$5; fb=$6; sig=$7; a=$8/100;
      vp=$9; vs=$10; t0=$11; fm=$12;
      pi=3.14159265358979;
      L=sqrt((rx-sx)^2 + (rz-sz)^2);
      xa=sx+fa*(rx-sx); za=sz+fa*(rz-sz);
      xb=sx+fb*(rx-sx); zb=sz+fb*(rz-sz);
      leff=sig*sqrt(2*pi);
      dtp=leff*(1/(vp*(1+a)) - 1/vp);
      dts=leff*(1/(vs*(1+a)) - 1/vs);
      fpk=0.4425*fm;
      printf "%.0f %.0f %.0f %.0f %.1f %.1f %.1f %.6f %.6f %.6f %.6f %.4f %.2f",
             xa, za, xb, zb, (fb-fa)*L, L, leff,
             L/vp+t0+dtp, L/vp+t0, L/vs+t0+dts, L/vs+t0,
             fpk, 2*pi*fpk*sig/vs}')
EOF

echo "=========================================================="
echo " test_lag_diagonal.sh -- 1 source / 1 receiver, diagonal"
echo "=========================================================="
echo "--- geometry ---"
printf '  S  (%6.0f, %6.0f) m\n' ${XSRC} ${ZSRC}
printf '  R  (%6.0f, %6.0f) m   path %.1f m, dip %.2f deg\n' \
    ${XRCV} ${ZRCV} ${LPATH} \
    $(echo "${XSRC} ${ZSRC} ${XRCV} ${ZRCV}" | \
      awk '{printf "%.2f", atan2($4-$2, $3-$1)*180/3.14159265358979}')
printf '  Fx radiation along that path:  P ~ cos = %.3f,  S ~ sin = %.3f\n' \
    $(echo "${XSRC} ${ZSRC} ${XRCV} ${ZRCV}" | \
      awk '{d=sqrt(($3-$1)^2+($4-$2)^2); printf "%.3f %.3f", ($3-$1)/d, ($4-$2)/d}')
printf '  Vp anomaly  site A  (%6.0f, %6.0f) m   at %.0f%% of the path\n' \
    ${XA} ${ZA} $(echo ${FRAC_A} | awk '{print 100*$1}')
printf '  Vs anomaly  site B  (%6.0f, %6.0f) m   at %.0f%% of the path\n' \
    ${XB} ${ZB} $(echo ${FRAC_B} | awk '{print 100*$1}')
printf '  A-B separation %.1f m = %.2f sigma   (sigma = %s m)\n' \
    ${SEP} $(echo "${SEP} ${SIGMA}" | awk '{printf "%.2f", $1/$2}') ${SIGMA}

echo ""
echo "--- background model (homogeneous) ---"
makemod sizex=${SIZEX} sizez=${SIZEZ} dx=${DX} dz=${DX} \
    cp0=${VP0} cs0=${VS0} ro0=${RO0} \
    orig=0,0 file_base=bg.su verbose=0
# The true model starts as a copy of it and gets the two Gaussians.
for P in cp cs ro; do cp bg_${P}.su model_${P}.su; done

echo "--- true model: two smooth Gaussian anomalies ---"
python3 - <<PYEOF
import sys
import numpy as np
sys.path.insert(0, '${ROOT}/fdelfwi/tests/kh_misfit')
import su_io

DX = ${DX}
AMP = ${APCT} / 100.0
SIG = ${SIGMA}

# (file, label, x, z) -- one Gaussian per PARAMETER, at its own place.
SITES = [('model_cp.su', 'Vp', ${XA}, ${ZA}),
         ('model_cs.su', 'Vs', ${XB}, ${ZB})]

print('  %-4s %10s %10s %10s %10s %12s'
      % ('par', 'x [m]', 'z [m]', 'v0', 'v_min', 'peak pert'))
for fn, lab, xa, za in SITES:
    d, h = su_io.read(fn)
    nx, nz = d.shape
    x = (np.arange(nx) * DX)[:, None]
    z = (np.arange(nz) * DX)[None, :]
    g = np.exp(-((x - xa) ** 2 + (z - za) ** 2) / (2.0 * SIG ** 2))
    v0 = float(d.mean())
    out = d.astype(np.float64) * (1.0 + AMP * g)
    su_io.write(fn, out.astype(np.float32), h)
    print('  %-4s %10.0f %10.0f %10.1f %10.1f %11.2f%%'
          % (lab, xa, za, v0, out.min(), 100.0 * (out.min() / v0 - 1.0)))

# cmin over BOTH models, which is what the dispersion guard tests, and the
# measured wavelet edge it is tested against.
cmin = min(float(su_io.read(f)[0].min())
           for f in ('model_cs.su', 'bg_cs.su'))
fdisp = cmin / (5.0 * DX)
fmeas = 1.22 * ${WAV_FMAX} + 1.46
print('  dispersion guard: cmin = %.1f m/s -> fmax < %.2f Hz;  measured '
      'wavelet edge %.2f Hz  (%.0f%% headroom)'
      % (cmin, fdisp, fmeas, 100.0 * (fdisp / fmeas - 1.0)))
if fmeas >= fdisp:
    raise SystemExit('the wavelet edge is above the dispersion guard -- '
                     'lower WAV_FMAX or |APCT|')

# Backscatter from a Gaussian: the transform is exp(-k^2 sig^2/2) and
# backscatter samples 2k, so the amplitude goes as exp(-2 k^2 sig^2).
for lab, f, v in (('peak  %5.2f Hz' % ${FPEAK}, ${FPEAK}, cmin),
                  ('edge  %5.2f Hz' % fmeas,    fmeas,    cmin)):
    ks = 2.0 * np.pi * f * SIG / v
    print('  backscatter at the %s:  k*sigma = %5.1f -> amplitude ~ e^%-6.0f'
          % (lab, ks, -2.0 * ks ** 2))
PYEOF

echo "  grid $(surange < model_cp.su | \
    awk '/traces:/{nx=$1} $1=="ns"{nz=$2} END{print nx" x "nz}')  (nx x nz)"

# Both cases share these models, so the figure is made ONCE here rather than
# per case.  Rows 1-2 share a colour scale per column, which is the only way
# "the background is homogeneous" is visible rather than asserted.
python3 ${HERE}/plot_lag_models.py --dx=${DX} \
    --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} \
    --xa=${XA} --za=${ZA} --xb=${XB} --zb=${ZB}

echo ""
echo "--- arrivals and the cycle-skip budget ---"
printf '  %-28s %9s %9s %9s %8s\n' event true bg 'dt [ms]' cycles
echo "${T_P_T} ${T_P_B} ${T_S_T} ${T_S_B} ${FPEAK}" | awk \
    '{T=1/$5;
      printf "  %-28s %8.3f %8.3f %8.1f %8.3f\n", "direct P  (site A only)",
             $1, $2, 1e3*($1-$2), ($1-$2)/T;
      printf "  %-28s %8.3f %8.3f %8.1f %8.3f\n", "direct S  (site B only)",
             $3, $4, 1e3*($3-$4), ($3-$4)/T;
      printf "  f_peak = %.2f Hz, half period = %.1f ms -> budget 0.500 cycles\n",
             $5, 500*T;
      printf "  (ray theory; the MEASURED shift is smaller -- the anomaly is\n";
      printf "   narrower than the Fresnel zone, so the front partly heals)\n"}'
echo "  effective path through a Gaussian, sigma*sqrt(2 pi) = ${LEFF} m"

echo ""
echo "--- wavelet ---"
makewave fmin=${WAV_FMIN} fmax=${WAV_FMAX} \
    dt=0.001 nt=4096 file_out=wave.su t0=${WAV_T0} verbose=0

# ---------------------------------------------------------------------------
# Observed data, one set per SOURCE type
# ---------------------------------------------------------------------------
# A single receiver has to be given as an explicit position list: the
# xrcv1/xrcv2 path cannot express it, because recvPar.c resets dxrcv back to 0
# when xrcv1 == xrcv2 and then errors out.
echo ""
echo "--- observed data (TRUE model) ---"
for SRC in 6; do
    ${FDELFWI}/fdelfwi_fwd \
        file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
        file_src=wave.su file_rcv=obsrc${SRC} \
        ischeme=3 iorder=4 src_type=${SRC} \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=0 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} left=4 right=4 top=4 bottom=4 > /dev/null
    # The hydrophone, P = (Txx + Tzz)/2.  fdelfwi_fwd does not write _rp.
    suop2 obsrc${SRC}_000_rtxx.su obsrc${SRC}_000_rtzz.su op=sum | \
        sugain scale=0.5 > obsrc${SRC}_000_rp.su
    echo "  src_type=${SRC}:  obsrc${SRC}_000_{rvx,rvz,rtxx,rtzz,rp}.su"
done

# ---------------------------------------------------------------------------
# The two cases
# ---------------------------------------------------------------------------
# name : src_type : comp : label.  Each runs in its own directory so that the
# plot scripts, which look for obs_000_* and bg_c?.su in the current
# directory, need no changes.
CASES=(
    "case_fx_hydro:6:_rp:Fx source + hydrophone"
    "case_fx_geo:6:_rvx,_rvz:Fx source + 2C geophone"
)


XMID=$(( (XSRC + XRCV) / 2 ))
XCIG=1500,${XA},${XMID},${XB},3500

for SPEC in "${CASES[@]}"; do
    IFS=: read -r NAME SRC COMP LABEL <<< "${SPEC}"
    echo ""
    echo "=========================================================="
    echo " ${NAME}:  ${LABEL}"
    echo "   src_type=${SRC}  comp=${COMP}"
    echo "=========================================================="
    mkdir -p ${NAME}
    cd ${NAME}
    for F in model_cp model_cs model_ro bg_cp bg_cs bg_ro wave; do
        ln -sf ../${F}.su .
    done
    for C in rvx rvz rtxx rtzz rp; do
        ln -sf ../obsrc${SRC}_000_${C}.su obs_000_${C}.su
    done

    # Both cases use an Fx source, which radiates P as cos and S as sin of the
    # angle from the x axis, so the diagonal path carries both arrivals and
    # both markers apply.
    ARR="P:${T_P_T}:${T_P_B},S:${T_S_T}:${T_S_B}"

    echo "--- lag gradient (nfreq=${D_NFREQ}, ${D_FMIN}-${D_FMAX} Hz) ---"
    ${FDELFWI}/test_gradient_visual \
        file_cp=bg_cp.su file_cs=bg_cs.su file_den=bg_ro.su \
        file_src=wave.su file_rcv=syn_dense \
        ischeme=3 iorder=4 src_type=${SRC} \
        rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
        dtrcv=${DTRCV} tmod=${TMOD} verbose=2 \
        xrcva=${XRCV} zrcva=${ZRCV} \
        xsrc=${XSRC} zsrc=${ZSRC} nshot=1 \
        ntaper=${NTAPER} grad_taper=${GRAD_TAPER} res_taper=0 aper_taper=0 \
        left=4 right=4 top=4 bottom=4 \
        chk_skipdt=50 chk_base=chk_dense \
        param=2 scaling=0 precond=0 smooth_grad=0 \
        vp_min=1000 vp_max=4500 vs_min=500 vs_max=2500 \
        rho_min=1500 rho_max=2500 \
        data_weight=0 misfit=0 \
        comp=${COMP} \
        file_snap=snap_syn file_snap_adj=snap_adj \
        tsnap1=${TSNAP1} tsnap2=${TSNAP2} dtsnap=${DTSNAP} \
        sna_type_vx=1 sna_type_vz=1 \
        lag_nfreq=${D_NFREQ} lag_fmin=${D_FMIN} lag_fmax=${D_FMAX} \
        lag_nt=${LAG_NT} lag_dt=${LAG_DT} \
        file_obs=obs out_base=dense > log_dense.txt 2>&1
    grep -E "lagGradientInit" log_dense.txt | head -3 || true
    grep -E "Total misfit" log_dense.txt | tail -1

    echo ""
    echo "--- figures: ${NAME} ---"
    # The residual, and the one MEASURED number in the whole script: the
    # cross-correlation lag between obs and syn around each arrival, in cycles.
    python3 ../../plot_lag_residual.py --offset=${LPATH} \
        --comp=${COMP} --arrivals="${ARR}" --fpeak=${FPEAK} dense

    echo ""
    # THE deliverable: g_vp and g_vs at a series of lags, plus the (z,tau)
    # gathers at each --xcig.  Site A sits at x=${XA} and site B at x=${XB},
    # so two of the CIG columns pass through the anomalies themselves.
    python3 ../../plot_lag_gradient.py \
        --zfocus=${ZFOCUS} --zwin=${ZWIN} --xgather=${XMID} \
        --xcig=${XCIG} dense

    echo ""
    # The two wavefields the gradient cross-correlates, at the SAME physical
    # instants.  adj_shot marches from t=T down to t=0 and relabels the headers
    # on the way (fake_it, adj_shot.c:546), so frame 1 of snap_adj is at TSNAP2
    # and the last is at TSNAP1; the plot script flips them back, so every panel
    # is real physical time.  There is no true-model pass here, so the figure
    # has two rows, forward and adjoint, and no difference row.
    python3 ../../plot_lag_snapshots.py --dx=${DX} \
        --tsnap1=${TSNAP1} --dtsnap=${DTSNAP} \
        --xsrc=${XSRC} --zsrc=${ZSRC} \
        --field=vx,vz --vpbg=${VP0} --t0=${WAV_T0} \
        --times=${SNAP_TIME}

    echo ""
    # The lag cube itself, animated: the x-z map walked along tau, one GIF per
    # parameter.  nframe=${LAG_NT} so the stride is 1 and every lag is a frame
    # -- the default 60 would silently drop every other one.  The clip is fixed
    # for the whole animation, so brightening IS amplitude.
    python3 ../../plot_lag_cube.py --par=vp,vs --sweep=tau \
        --xslice=${XMID} --zslice=${ZFOCUS} \
        --zfocus=${ZFOCUS} --zwin=${ZWIN} \
        --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} \
        --nframe=${LAG_NT} --fps=10 dense

    echo ""
    # The same CIGs read as an OPENING-ANGLE gather.  At each depth the stack
    # is anchored at tau = 0 -- the only lag that IS the gradient -- and swept
    # over slopes p = -cos(theta)/v across a short aperture, so the straight-
    # line assumption stays LOCAL and the curvature of the band does not
    # defeat it.  (A single pivot for the whole column does not work here: the
    # bands are DSR moveout curves whose apex depth marches with x, so one
    # long straight line is a tangent, not a fit.)
    #
    # theta = 90 deg is transmission -> TOMOGRAPHIC, 0 and 180 are
    # backscatter -> MIGRATION, the same convention as energy_ic=.  Both
    # anomalies here are smooth Gaussians with k*sigma ~ 18, so backscatter is
    # down by e^-669 and there is no reflector anywhere in either model: the
    # energy HAS to sit on 90, and the waist of the bowtie has to follow the
    # ray.  That is the check this figure is for.
    #
    # The aperture is Hann tapered (--taper=1, the default).  A boxcar
    # aperture is a rectangle in z and convolves the stack with a sinc:
    # measured on g_vp at x=${XMID}, the sidelobe at the waist row is 0.0275
    # of the peak boxcar against 0.0026 Hann, a 10x drop for a main lobe 13 ->
    # 20 deg wide.  Partial Tukey is WORSE than either (0.04-0.05), so this is
    # all-or-nothing.  The bowtie ARMS are not touched by any of this -- they
    # are the tangency locus of the transform, not a window artifact.
    python3 ../../plot_lag_slant.py --par=vp,vs \
        --xcig=${XCIG} dense

    echo ""
    # EXTRACT the tomographic kernel instead of just displaying its angle.
    # Same slant stack, but built as a least-squares operator in the lag-
    # frequency domain so it can be INVERTED: a straight line in (tau,z) is a
    # pure shift along tau, hence one phase multiply per lag frequency, no
    # interpolation.  A Gaussian on the origin (p,tau0) = (0,0) keeps the
    # background's own arrival -- whose apex is at tau = 0 by construction and
    # which at tau = 0 IS the straight S-R ray path -- and rejects any band
    # whose apex sits elsewhere.  The weighted panel is then transformed BACK,
    # so the output dense_{vp,vs}_tomo.su is a gradient, not a Radon panel.
    #
    # --lwin uses the same depth-swept pivot as plot_lag_slant.py above, with
    # z_ref = the output depth, so no apex depth is assumed.  That matters:
    # zref=${ZFOCUS} would be an ORACLE here, since the ray depth is only
    # knowable because we placed S and R ourselves.
    #
    # Each run prints its own round-trip check with W=1 -- if that correlation
    # is not ~1 the operator is not invertible and the separation is
    # meaningless, so read it before reading the figures.
    # No --xcig here: that figure needs a slant panel at every listed x and
    # the local path only forms one, at --xgather.  The script says so when
    # asked anyway, rather than skipping it silently.
    python3 ../../plot_lag_filter.py --mode=gauss --par=vp,vs \
        --sigtheta=15 --sigtau=40 --rmax=3 --lwin=800 \
        --xgather=${XMID} \
        --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} dense

    # -----------------------------------------------------------------------
    # Drop what this experiment does not use.
    #
    # test_gradient_visual ALWAYS builds and writes the Yang pseudo-Hessian
    # preconditioner, the scaled gradient and the preconditioned gradient:
    # the build/apply/write block has no "if" around it, so scaling=0 and
    # precond=0 change only the CONTENTS of those arrays, not whether the
    # files appear.  With scaling=0 the scaled gradient is bit-identical to
    # the raw one (m0 = [1,1,1]), and none of these files is read by anything
    # downstream -- the figures come from *_raw.su and *_lag{vp,vs}.su alone.
    #
    # The per-shot cubes are a separate case: nshot=1 here, so the shot stack
    # IS the single shot and <base>_shot000_lag*.su duplicates <base>_lag*.su
    # byte for byte, at 58 MB each.  The plot scripts read only the stack.
    rm -f dense_vp_scaled.su dense_vs_scaled.su dense_rho_scaled.su \
          dense_vp_preco.su  dense_vs_preco.su  dense_rho_preco.su \
          precond_vp.su precond_vs.su precond_rho.su \
          precond_vp_vs.su precond_vp_rho.su precond_vs_rho.su \
          hess_raw_lam.su hess_raw_mu.su hess_raw_rho.su \
          hess_prebeta_vp.su hess_prebeta_vs.su hess_prebeta_rho.su \
          dense_shot000_lagvp.su dense_shot000_lagvs.su \
          dense_shot000_lagaxis.txt

    echo ""
    echo "  kept in ${NAME}:"
    du -sh . | awk '{printf "    %s total\n", $1}'
    ls *.su *.txt *.png 2>/dev/null | sed 's/^/    /'

    cd ${WORK}
done

# ---------------------------------------------------------------------------
# The two cases side by side
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo " the two acquisitions side by side"
echo "=========================================================="
python3 ${HERE}/plot_lag_cases.py \
    --cases=case_fx_hydro,case_fx_geo \
    --labels="Fx + hydrophone,Fx + geophone" \
    --xsrc=${XSRC} --zsrc=${ZSRC} --xrcv=${XRCV} --zrcv=${ZRCV} \
    --xa=${XA} --za=${ZA} --xb=${XB} --zb=${ZB} \
    --zfocus=${ZFOCUS} --zwin=${ZWIN}

echo ""
echo "Figures and numbers are in ${WORK}"
