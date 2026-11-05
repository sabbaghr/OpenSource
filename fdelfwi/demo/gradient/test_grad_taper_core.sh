#!/bin/bash
#
# test_grad_taper_core.sh -- regression + correctness test for grad_taper_core=.
#
# WHAT IS BEING TESTED.  `grad_taper=R` used to apply w = r^2/R^2 around every
# source and receiver.  That is exactly 0 only AT the cell and 1/R^2 one cell
# away, while the imaging-condition singularity decays like ~r^-6.5 over about
# 4 cells -- so the taper RESCALED the source spike instead of removing it, and
# a larger R could not fix it (inside the old radius two masks differ only by a
# constant, so the spike keeps its dominance over its own neighbours).
# `grad_taper_core=c` adds a hard-zero core:
#
#     w = 0                         r <= c
#     w = (r^2-c^2)/(R^2-c^2)       c < r < R
#     w = 1                         r >= R
#
# THE MASK HAS TWO INDEPENDENT IMPLEMENTATIONS and they must not drift:
#   * taperGradientSrcRcv()  in fwi_inversion.c    -> conventional gradient
#   * buildGradTaperMask()   in grad_taper_mask.c  -> time-lag cubes, per lag
# A lag_tomo=1 vs lag_tomo=0 comparison is meaningless unless both see the
# SAME mask, so check 5 below compares the two implementations against each
# other on the same geometry.  That is the real reason this script exists.
#
# WHY IT CAN BE TRUSTED AS A REGRESSION TEST.  The checks do not compare
# against a stored reference (which would only say "it changed").  They divide
# the written gradient by the ANALYTIC mask and require that both runs recover
# the SAME underlying gradient.  core=0's analytic mask IS the old r^2/R^2
# formula, so check 2 passing is a proof that core=0 reproduces the previous
# behaviour, and check 3 is a proof that the new branch is the intended one.
#
# This script does not generate data.  It reuses the models, wavelet and
# observed gathers of gradient_visual_velocity/ by symlink, so run
#     ./test_gradient_visual_velocity.sh
# first (or just its data steps).  Nothing here writes into that directory.
#
# Exit status is 0 only if every check passes.

set -e

ROOT=/rcp3/software/codes/OpenSource_SL10
FDELFWI=${ROOT}/fdelfwi
BIN=${ROOT}/bin
CWP=/rcp3/software/codes/cwp/bin
export PATH=${BIN}:${CWP}:${PATH}

REF=gradient_visual_velocity          # data donor, read-only
WORKDIR=grad_taper_core
RADIUS=20                             # R, in grid points
CORE=5                                # c, in grid points

# Geometry -- must match the donor's observed data exactly or the residual is
# nonzero at the true model.  (Irrelevant to the mask itself, which is purely
# geometric, but it keeps the gradient physical and the spike real.)
NSHOT=5                               # uses obs_000..obs_004
XSRC=0
DXSHOT=100
ZSRC=10
XRCV1=100; XRCV2=1900; DXRCV=10; ZRCV=50
DX=5
COMP=_rp

echo "========================================================="
echo " grad_taper_core= test   R=${RADIUS}  core=${CORE}"
echo "========================================================="

if [ ! -d "${REF}" ]; then
    echo "FAIL: donor directory ${REF}/ not found."
    echo "      Run ./test_gradient_visual_velocity.sh first."
    exit 1
fi

mkdir -p ${WORKDIR}
cd ${WORKDIR}

# ---------------------------------------------------------------- input links
for f in model_bg_cp.su model_bg_cs.su model_bg_ro.su wave.su; do
    [ -r "../${REF}/${f}" ] || { echo "FAIL: ../${REF}/${f} missing"; exit 1; }
    ln -sfn ../${REF}/${f} ${f}
done
nobs=0
for ((i=0; i<NSHOT; i++)); do
    s=$(printf "%03d" ${i})
    [ -r "../${REF}/obs_${s}${COMP}.su" ] || {
        echo "FAIL: ../${REF}/obs_${s}${COMP}.su missing"; exit 1; }
    ln -sfn ../${REF}/obs_${s}${COMP}.su obs_${s}${COMP}.su
    nobs=$((nobs+1))
done
echo "  linked ${nobs} observed gathers + models + wavelet from ../${REF}"

export OMP_NUM_THREADS=8

# Everything that could move or smear a zero is OFF, so that what lands in the
# output is the taper and nothing else:
#   smooth_grad/smooth_hess=0  a Gaussian would blur the hard zero outward
#   precond=0                  Plessix-Mulder divides by illumination
#   srt_radius=0               taper_source.c has its OWN zero core (per shot)
#   scaling=1                  kept: it is a constant per parameter, so it
#                              cancels in the ratio tests and keeps the run
#                              representative of production
COMMON="file_cp=model_bg_cp.su file_cs=model_bg_cs.su file_den=model_bg_ro.su \
    file_src=wave.su \
    ischeme=3 iorder=4 src_type=1 \
    rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
    dtrcv=0.004 tmod=1.5 \
    xrcv1=${XRCV1} xrcv2=${XRCV2} zrcv1=${ZRCV} zrcv2=${ZRCV} dxrcv=${DXRCV} \
    xsrc=${XSRC} zsrc=${ZSRC} nshot=${NSHOT} dxshot=${DXSHOT} \
    ntaper=100 left=4 right=4 top=1 bottom=4 \
    param=2 scaling=1 precond=0 srt_radius=0 \
    smooth_grad=0 smooth_hess=0 \
    vp_min=1500 vp_max=4500 vs_min=500 vs_max=2500 \
    rho_min=1500 rho_max=2500 \
    comp=${COMP} file_obs=obs"

# ------------------------------------------------- 1. fwi_inversion, core=0/5
# g_*_initial_*.su is written at fwi_inversion.c:2169, AFTER the taper at
# :2108, so these files carry taperGradientSrcRcv's mask.
for c in 0 ${CORE}; do
    echo ""
    echo "--- fwi_inversion  grad_taper=${RADIUS} grad_taper_core=${c} ---"
    ${FDELFWI}/fwi_inversion ${COMMON} \
        file_rcv=syn_fi${c} chk_base=chk_fi${c} \
        grad_taper=${RADIUS} grad_taper_core=${c} \
        res_taper=0 aper_taper=0 \
        algorithm=1 lbfgs_mem=5 niter=1 nls_max=4 conv=1e-6 \
        file_grad=g_fi${c} verbose=1 > log_fi${c}.txt 2>&1 || {
            echo "FAIL: fwi_inversion core=${c} returned $?; tail of log:"
            tail -25 log_fi${c}.txt; exit 1; }
    grep -E "Gradient taper" log_fi${c}.txt | sed 's/^/      /'
done

# ------------------------------------------- 2. test_gradient_visual, core=0/5
# Independent implementation: this driver calls buildGradTaperMask() from
# grad_taper_mask.c, the same function the lag cubes use.
for c in 0 ${CORE}; do
    echo ""
    echo "--- test_gradient_visual  grad_taper=${RADIUS} grad_taper_core=${c} ---"
    ${FDELFWI}/test_gradient_visual ${COMMON} \
        file_rcv=syn_gv${c} chk_base=chk_gv${c} \
        grad_taper=${RADIUS} grad_taper_core=${c} \
        out_base=gv${c} verbose=1 > log_gv${c}.txt 2>&1 || {
            echo "FAIL: test_gradient_visual core=${c} returned $?; tail of log:"
            tail -25 log_gv${c}.txt; exit 1; }
    grep -E "Applying gradient taper" log_gv${c}.txt | sed 's/^/      /'
done

rm -f chk_fi*_*.bin chk_gv*_*.bin syn_*.su res_*.su residual*.su

# ------------------------------------------------------------------- 3. checks
echo ""
echo "========================================================="
echo " CHECKS"
echo "========================================================="

python3 - <<PYEOF
import sys
import numpy as np

RADIUS = ${RADIUS}
CORE   = ${CORE}
DX     = ${DX}

def readsu(fn):
    """Return [n2, n1] float32 from an SU file of equal-length traces."""
    raw = np.fromfile(fn, dtype=np.uint8)
    ns  = int(np.frombuffer(raw[114:116].tobytes(), dtype='<u2')[0])
    rec = 240 + 4*ns
    if raw.size % rec:
        sys.exit("FAIL: %s is not a multiple of %d bytes" % (fn, rec))
    n2  = raw.size // rec
    out = np.empty((n2, ns), dtype=np.float32)
    for i in range(n2):
        o = i*rec + 240
        out[i] = np.frombuffer(raw[o:o+4*ns].tobytes(), dtype='<f4')
    return out                      # [nx, nz]

def analytic_mask(nx, nz, pts, R, c):
    """w = min over pts of the three-branch ramp.  Independent of the C code."""
    m = np.ones((nx, nz), dtype=np.float64)
    r2max, r2core = float(R*R), float(c*c)
    span = r2max - r2core
    ix = np.arange(nx)[:, None]
    iz = np.arange(nz)[None, :]
    for (px, pz) in pts:
        r2 = (ix - px)**2 + (iz - pz)**2
        w  = np.where(r2 <= r2core, 0.0, (r2 - r2core)/span)
        m  = np.minimum(m, np.where(r2 < r2max, w, 1.0))
    return m

# ---- geometry on the interior grid (x0 = z0 = 0) ----
src = [(int(round((${XSRC} + i*${DXSHOT})/DX)), int(round(${ZSRC}/DX)))
       for i in range(${NSHOT})]
rcv = [(int(round(x/DX)), int(round(${ZRCV}/DX)))
       for x in range(${XRCV1}, ${XRCV2}+1, ${DXRCV})]
pts = src + rcv

g_fi0 = readsu("g_fi0_initial_vs.su")
g_fi5 = readsu("g_fi${CORE}_initial_vs.su")
g_gv0 = readsu("gv0_vs_raw.su")
g_gv5 = readsu("gv${CORE}_vs_raw.su")
nx, nz = g_fi0.shape
for nm, a in (("g_fi${CORE}", g_fi5), ("gv0", g_gv0), ("gv${CORE}", g_gv5)):
    if a.shape != (nx, nz):
        sys.exit("FAIL: %s is %s, expected %s" % (nm, a.shape, (nx, nz)))
print("  grid %d x %d (nx x nz), %d sources, %d receivers"
      % (nx, nz, len(src), len(rcv)))

m0 = analytic_mask(nx, nz, pts, RADIUS, 0)
m5 = analytic_mask(nx, nz, pts, RADIUS, CORE)

# nearest-point distance map, for reporting
ix = np.arange(nx)[:, None]; iz = np.arange(nz)[None, :]
dist = np.full((nx, nz), 1e9)
for (px, pz) in pts:
    dist = np.minimum(dist, np.sqrt((ix-px)**2 + (iz-pz)**2))

fails = []
def check(ok, label, detail=""):
    print("  %-4s %s%s" % ("OK" if ok else "FAIL", label,
                           ("\n          " + detail) if detail else ""))
    if not ok: fails.append(label)

# -- 1. the core is EXACTLY zero, and it is the only thing that is --
incore = (m5 == 0.0)
check(np.all(g_fi5[incore] == 0.0),
      "core is exactly 0 in fwi_inversion (%d cells)" % incore.sum(),
      "max|g| in core = %.3e" % (np.abs(g_fi5[incore]).max() if incore.any() else 0.0))
check(np.all(g_gv5[incore] == 0.0),
      "core is exactly 0 in test_gradient_visual",
      "max|g| in core = %.3e" % (np.abs(g_gv5[incore]).max() if incore.any() else 0.0))

# -- 2. REGRESSION: core=0 still is the old w = r^2/R^2 --
# Dividing by the analytic old mask must recover the same field that the
# core=5 run recovers through its own (different) mask.
both = (m0 > 1e-6) & (m5 > 1e-6)
raw0 = g_fi0[both] / m0[both]
raw5 = g_fi5[both] / m5[both]
scale = np.abs(raw0).max()
rel = np.abs(raw0 - raw5).max() / scale
check(rel < 2e-5,
      "core=0 mask IS r^2/R^2 and core=%d mask IS (r^2-c^2)/(R^2-c^2)" % CORE,
      "both recover the same underlying gradient: max rel diff %.2e over %d cells"
      % (rel, both.sum()))

# -- 3. outside R nothing was touched at all --
far = (dist >= RADIUS)
check(np.array_equal(g_fi0[far], g_fi5[far]),
      "outside R the two runs are bit-identical (%d cells)" % far.sum())

# -- 4. what the fix actually removed --
killed = (m0 > 0.0) & (m5 == 0.0)
if killed.any():
    worst = np.abs(g_fi0[killed]).max()
    gmax0 = np.abs(g_fi0).max()
    check(True, "newly zeroed zone: %d cells, 0 < r <= %d" % (killed.sum(), CORE),
          "max|g| there was %.3e (core=0), i.e. %.1f%% of that run's global max %.3e"
          % (worst, 100.0*worst/gmax0, gmax0))
else:
    check(False, "newly zeroed zone is empty -- core had no effect")

# -- 5. THE STAGE-IDENTITY CHECK: the two implementations agree --
# taperGradientSrcRcv (fwi_inversion.c) vs buildGradTaperMask
# (grad_taper_mask.c).  The two drivers do not produce the same gradient
# (test_gradient_visual applies its own scaling), so compare the ZERO SET,
# which is determined by the mask alone.
z_fi = (g_fi5 == 0.0)
z_gv = (g_gv5 == 0.0)
d = int((z_fi != z_gv).sum())
check(d == 0,
      "taperGradientSrcRcv and buildGradTaperMask zero sets agree",
      "%d cells differ (fwi=%d zeros, gv=%d zeros)" % (d, z_fi.sum(), z_gv.sum()))
# and the mask must be the ONLY thing creating zeros.  A cell can also be
# zero because the gradient itself vanishes there -- at iz=0 the free surface
# makes the whole row identically zero -- so the test is not "zero set ==
# core" but "every zero outside the core was already zero at core=0".
extra = z_fi & ~incore
bystander = bool(np.all(g_fi0[extra] == 0.0)) if extra.any() else True
rows = np.unique(np.where(extra)[1]) if extra.any() else []
check(bystander,
      "the core is the only source of new zeros",
      "%d cells are zero outside the core, all of them already zero at "
      "core=0 (rows iz=%s)" % (int(extra.sum()), list(map(int, rows)))
      if extra.any() else "no zeros outside the core")

# -- 6. did the spike actually go away? --
print("")
print("  --- source-spike suppression (g_vs, fwi_inversion) ---")
print("  %-8s %-12s %-10s %-9s %s" % ("core", "max|g_vs|", "at (ix,iz)", "r_nearest", "top-50m energy"))
for tag, g in (("0", g_fi0), (str(CORE), g_fi5)):
    k = np.unravel_index(np.abs(g).argmax(), g.shape)
    e_all = float((g.astype(np.float64)**2).sum())
    nz50 = max(1, int(round(50.0/DX)))
    e_top = float((g[:, :nz50].astype(np.float64)**2).sum())
    print("  %-8s %-12.4e (%3d,%3d)   %-9.2f %.2f%%"
          % (tag, np.abs(g).max(), k[0], k[1], dist[k],
             100.0*e_top/e_all if e_all > 0 else 0.0))

# radial profile away from the most isolated source
sx, sz = src[-1]
print("")
print("  radial profile of |g_vs| along +x from the source at ix=%d iz=%d:" % (sx, sz))
for tag, g in (("core=0 ", g_fi0), ("core=%-2d" % CORE, g_fi5)):
    row = "  %s " % tag
    for r in range(0, CORE+4):
        row += "%d:%.2e " % (r, abs(g[min(sx+r, nx-1), sz]))
    print(row)

print("")
if fails:
    print("  %d CHECK(S) FAILED: %s" % (len(fails), "; ".join(fails)))
    sys.exit(1)
print("  ALL CHECKS PASSED")
sys.exit(0)
PYEOF
rc=$?

echo ""
if [ ${rc} -eq 0 ]; then
    echo "========================================================="
    echo " PASS -- grad_taper_core works and core=0 is unchanged"
    echo "========================================================="
else
    echo "========================================================="
    echo " FAIL -- see checks above.  Outputs kept in $(pwd)"
    echo "========================================================="
fi
exit ${rc}
