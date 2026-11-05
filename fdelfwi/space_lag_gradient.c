#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"

/**
 * space_lag_gradient.c -- Space-lag extended imaging condition for the
 *                         elastic FWI gradient (lambda, mu).
 *
 * WHAT THIS COMPUTES
 * ------------------
 * The conventional FWI gradient is a ZERO-LAG correlation in SPACE as well as
 * in time: both wavefields are read at the same grid point (see accumGradient
 * in fwi_gradient.c).  This module evaluates the SAME kernels with the two
 * fields separated by a space lag h = (hx, hz), applied SYMMETRICALLY -- the
 * forward field at x - h, the adjoint field at x + h:
 *
 *   g_lam(x,h) = dt sum_t div(v_f)(x-h,t) * (psi_txx + psi_tzz)(x+h,t)
 *   g_mu (x,h) = dt sum_t 2[ Dx(vx_f)(x-h) psi_txx(x+h)
 *                          + Dz(vz_f)(x-h) psi_tzz(x+h) ]
 *              + dt sum_t (Dz(vx_f)+Dx(vz_f))(x-h) psi_txz(x+h)
 *
 * so h is the ONE-WAY shift each wavefield carries and 2h is the RELATIVE
 * separation between them.  This is the same convention lag_gradient.c uses on
 * the time axis, and it carries the same factor 2 into every statement about
 * the lag axis below.  h = 0 reproduces the conventional gradient.
 *
 * WHY THE TIME DOMAIN -- THE OPPOSITE CHOICE FROM lag_gradient.c
 * --------------------------------------------------------------
 * lag_gradient.c had to go to the frequency domain, and says why in its own
 * header: in the backward loop of adj_shot the forward and adjoint states are
 * only ever co-resident AT THE SAME TIME INDEX, so a TIME lag would need ring
 * buffers and cross-segment carry buffers.  A SPACE lag has no such obstacle.
 * At step `it` the entire padded grid of both fields is in memory, so the
 * product can be formed directly, lag by lag, and summed over t in place.
 *
 * Everything the frequency domain forced on the time-lag module therefore
 * disappears here:
 *
 *   - no band limiting.  The h = 0 slice is formed by exactly the arithmetic
 *     accumGradient uses, over exactly its loop bounds, in the same order, so
 *     it must reproduce the conventional gradient to float rounding -- a far
 *     sharper check than the time-lag cube can offer, where the tau = 0 slice
 *     is only as good as the frequency set is dense.
 *   - no aliasing period, no df <= 1/T density requirement, no reconstruction
 *     weight.
 *
 * and the costs invert:
 *
 *   - MEMORY IS 2 * nlag * sizem floats and the lag axis must be fixed BEFORE
 *     the run.  For time lags memory scaled with nfreq and the lag axis was
 *     free at write time; here each lag is its own accumulator.
 *   - LAGS ARE INTEGER MULTIPLES OF THE GRID SPACING.  The shift is an array
 *     index offset applied during accumulation, not a phase rotation applied
 *     at write time, so there is no sub-sample freedom.  dhx/dhz are strides
 *     in GRID POINTS; the physical lag is ihx*dhx*dx.
 *
 * WHAT THE h AXIS MEASURES
 * ------------------------
 * For local plane waves A ~ exp(i k_A.x) (forward) and B ~ exp(i k_B.x)
 * (adjoint), the symmetric lag gives
 *
 *   conj(A(x-h)) B(x+h) ~ exp(i (k_B - k_A).x) exp(i (k_B + k_A).h)
 *
 * so the conjugate variable of h is the SUM k_A + k_B, exactly as the
 * conjugate variable of tau in lag_gradient.c is the sum of the two
 * frequencies, 2w.  With d_s the incident direction at x (source -> x) and
 * d_out the outgoing one (x -> receiver),
 *
 *   k_A + k_B = (w/v)(d_s - d_out) = (2 w cos(theta) / v) N
 *
 * N the unit reflector normal and theta the half opening angle: the MIGRATION
 * WAVENUMBER.  So the h-gather at a single image point carries the reflector
 * normal DIRECTION -- which needs no velocity -- and the magnitude
 * 2 w cos(theta)/v.  That is the point of this module: it is a local,
 * pointwise alternative to differentiating the time-lag cube in x, and it
 * removes the vertical-bisector assumption of the 1D slant stack in
 * demo/gradient/plot_lag_slant.py, which measures only dtau/dz and therefore
 * reports cos(theta)cos(phi) with phi the bisector dip.
 *
 * CAUTION ON THAT SIGN.  Whether the stored adjoint array behaves locally like
 * exp(-i w t) or exp(+i w t) decides whether the conjugate of h is the SUM
 * k_A + k_B or the DIFFERENCE k_B - k_A.  This is the same time-reversal sign
 * that silently inverted the energy_ic angle filters (see the psi_v note in
 * energy_imaging.c), it is invisible to the h = 0 check, and it is best fixed
 * empirically on a geometry whose reflector normal is known rather than
 * asserted here.  slag_shift is the calibration that reaches the h axis.
 *
 * CAVEATS (read before interpreting output)
 * -----------------------------------------
 *  - THE JOINT CUBE g(x,h,tau) IS NOT AVAILABLE and is not a small extension:
 *    it would need nfreq * nlag * sizem complex accumulators.  This module and
 *    lag_gradient.c produce the two MARGINALS -- space lags at tau = 0 and
 *    time lags at h = 0.  They can be combined per image point (|k_h| from
 *    here, the effective w from there, giving cos(theta)/v) but not as one
 *    array.
 *  - THE SUPPORT SHRINKS WITH |h|.  Both x-h and x+h must lie inside
 *    accumGradient's valid region, so each lag is accumulated over bounds
 *    inset by its own |h| and a rim of that width is left at zero.  The
 *    alternative -- one h-independent inset by hmax -- was rejected because it
 *    would break the h = 0 identity, which is the only exact check available.
 *    The rim lies in the absorbing taper region that accumGradient already
 *    excludes, so the interior is unaffected; what it does mean is that
 *    apparent h-moveout within |h| grid points of the edge is a bounds effect,
 *    not physics.
 *  - OUTPUT IS (Vp, Vs), converted per lag by the same pointwise chain rule
 *    convertGradientToVelocity applies to the conventional gradient,
 *        g_Vp = 2 rho Vp  g_lam,   g_Vs = 2 rho Vs (g_mu - 2 g_lam),
 *    with rho, Vp, Vs read at the MIDPOINT x.  The map is pointwise in the
 *    output coordinate and lag-independent, so it commutes with the lag axis
 *    and the h = 0 slice is directly comparable to <base>_vp_raw.su from a
 *    param=2 run.  There is NO g_rho cube: only lambda and mu are
 *    accumulated, and the density chain rule additionally needs the direct rho
 *    kernel.
 *  - Accumulate in Lame (adj_shot param=1) and convert ONCE.  Running this
 *    output through extractGradientVector with param=2 is the known
 *    double-chain-rule bug.
 *  - Only the elastic scheme (ischeme = 3 or 5) and iorder 4/6/8 are handled.
 *    Accumulation is a no-op otherwise.
 *
 * Loop bounds and FD stencils are duplicated from accumGradient on purpose, so
 * that the h = 0 slice is formed over exactly the same grid points.
 */

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

static void slagFreeAccum(void);

/* ------------------------------------------------------------------ *
 * Module state.  Inactive until slagGradientInit is called with
 * nhx*nhz > 0, so every entry point is a cheap no-op by default.
 * ------------------------------------------------------------------ */
static int    sl_on   = 0;
static int    sl_nhx  = 0, sl_nhz  = 0;   /* lags per axis, both odd    */
static int    sl_nlag = 0;                /* nhx * nhz                  */
static int    sl_dhx  = 1, sl_dhz  = 1;   /* lag stride in grid points  */
static int    sl_shx  = 0, sl_shz  = 0;   /* calibration shift, points  */
static int    sl_checked = 0;             /* reach check done this shot */
static size_t sl_sizem = 0;

/* Lag accumulators, [nlag][sizem] with the lag SLOWEST so that each lag is a
 * contiguous plane: the accumulation loop walks one plane at a time, which
 * keeps the working set at 2 planes rather than 2*nlag interleaved values. */
static float *sl_glam = NULL;
static float *sl_gmuu = NULL;

/* Strains of the current time step, computed ONCE per step and then read by
 * every lag.  Without these the FD stencils would be re-evaluated nlag times. */
static float *sl_exx = NULL;   /* forward D+x(vx)         , P   grid */
static float *sl_ezz = NULL;   /* forward D+z(vz)         , P   grid */
static float *sl_shr = NULL;   /* forward D-z(vx)+D-x(vz) , Txz grid */

/* Shot stack.  The FWI gradient is a sum over shots, so the cube that
 * corresponds to <base>_vp_raw.su is the SUM of the per-shot cubes.  Lives
 * across slagGradientInit (which only clears the per-shot accumulators) and is
 * released by slagGradientFree. */
static float *sl_stk_vp = NULL, *sl_stk_vs = NULL;
static int    sl_stk_nlag = 0, sl_stk_nmodel = 0, sl_stk_nshot = 0;
static int    sl_stk_nhx = 0, sl_stk_nhz = 0;


/***********************************************************************
 * slagBounds -- Loop bounds for the P and Txz grids.
 *
 * Reproduces the bounds logic of accumGradient (fwi_gradient.c) so the
 * h = 0 slice is accumulated over identical grid points: absorbing taper
 * exclusion, free-surface row exclusion, and FD stencil margins.
 ***********************************************************************/
static void slagBounds(modPar *mod, bndPar *bnd,
                       int *ibPx, int *iePx, int *ibPz, int *iePz,
                       int *ibTx, int *ieTx, int *ibTz, int *ieTz)
{
	int n1   = mod->naz;
	int nax  = mod->nax;
	int half = mod->iorder / 2;

	/* --- P grid (txx, tzz live here) --- */
	*ibPx = mod->ioPx;  *iePx = mod->iePx;
	*ibPz = mod->ioPz;  *iePz = mod->iePz;

	if (bnd->lef == 4 || bnd->lef == 2) *ibPx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) *iePx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) *ibPz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) *iePz -= bnd->ntap;

	if (bnd->top == 1) *ibPz = MAX(*ibPz, mod->ioPz + 1);

	*ibPx = MAX(*ibPx, half - 1);
	*iePx = MAX(*ibPx, *iePx);
	if (*iePx > nax - half) *iePx = nax - half;
	*ibPz = MAX(*ibPz, half - 1);
	*iePz = MAX(*ibPz, *iePz);
	if (*iePz > n1 - half) *iePz = n1 - half;

	/* --- Txz grid (shear) --- */
	*ibTx = mod->ioTx;  *ieTx = mod->ieTx;
	*ibTz = mod->ioTz;  *ieTz = mod->ieTz;

	if (bnd->lef == 4 || bnd->lef == 2) *ibTx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) *ieTx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) *ibTz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) *ieTz -= bnd->ntap;

	if (bnd->top == 1) *ibTz = MAX(*ibTz, mod->ioTz + 1);

	*ibTx = MAX(*ibTx, half);
	*ieTx = MAX(*ibTx, *ieTx);
	if (*ieTx > nax - half) *ieTx = nax - half;
	*ibTz = MAX(*ibTz, half);
	*ieTz = MAX(*ibTz, *ieTz);
	if (*ieTz > n1 - half) *ieTz = n1 - half;
}


/***********************************************************************
 * slagCoef -- FD coefficients matching the forward stress update.
 ***********************************************************************/
static void slagCoef(modPar *mod, float *c1, float *c2, float *c3, float *c4)
{
	*c3 = *c4 = 0.0f;
	switch (mod->iorder) {
		case 6:
			*c1 = 75.0f/64.0f;    *c2 = -25.0f/384.0f;
			*c3 = 3.0f/640.0f;
			break;
		case 8:
			*c1 = 1225.0f/1024.0f; *c2 = -245.0f/3072.0f;
			*c3 = 49.0f/5120.0f;   *c4 = -5.0f/7168.0f;
			break;
		case 4:
		default:
			*c1 = 9.0f/8.0f;      *c2 = -1.0f/24.0f;
			break;
	}
}


/***********************************************************************
 * slagLagOffsets -- The two index offsets of lag number `ilag`.
 *
 * Returns the A (forward) and B (adjoint) grid offsets in x and z.  The
 * symmetric convention puts the forward field at -h and the adjoint at +h; the
 * calibration shift is added to the ADJOINT offset only, which is what makes
 * the identity in slagGradientInit's header hold.
 ***********************************************************************/
static void slagLagOffsets(int ilag, int *oax, int *oaz, int *obx, int *obz)
{
	int ihx = ilag % sl_nhx;
	int ihz = ilag / sl_nhx;
	int ox  = (ihx - (sl_nhx - 1) / 2) * sl_dhx;
	int oz  = (ihz - (sl_nhz - 1) / 2) * sl_dhz;

	*oax = -ox;            *oaz = -oz;
	*obx =  ox + sl_shx;   *obz =  oz + sl_shz;
}


/***********************************************************************
 * slagGradientInit -- Allocate the lag accumulators.
 *
 *   nhx, nhz   number of lags along x and z; forced odd so h = 0 is sampled
 *              exactly.  Either may be 1: (nhx, 1) gives a horizontal-lag
 *              cube and (1, nhz) a vertical one, which between them give both
 *              components of k_A + k_B at cost nhx + nhz instead of nhx*nhz.
 *              (1, 1) is the conventional gradient and is accepted -- it is
 *              the cheapest way to exercise the h = 0 identity.
 *   dhx, dhz   lag stride in GRID POINTS (>= 1).  Physical lag ihx*dhx*dx.
 *   shiftx,    CALIBRATION ONLY.  A constant extra grid shift applied to the
 *   shiftz     ADJOINT field.  Replacing B(x) by B(x+D) in
 *                  g(x,h) = sum_t A(x-h) B(x+h)
 *              gives g_new(x,h) = g_old(x + D/2, h + D/2), a DIAGONAL
 *              translation in (x, h).  Running with shift = 0 and shift = D
 *              and cross-correlating the two cubes therefore validates the
 *              SCALE and the SIGN of the h axis at once -- neither of which
 *              the h = 0 check can see, since h = 0 is invariant under a
 *              factor-2 lag-scale error and under a sign flip.  Use EVEN D so
 *              that D/2 lands on a grid point.
 *
 * Call once per shot: this zeroes the accumulators.  It does NOT touch the
 * shot stack, so a multi-shot loop keeps accumulating.
 ***********************************************************************/
void slagGradientInit(modPar *mod, int nhx, int nhz, int dhx, int dhz,
                      int shiftx, int shiftz, int verbose)
{
	size_t nbytes;

	slagFreeAccum();

	if (nhx <= 0 || nhz <= 0) { sl_on = 0; return; }

	if (mod->ischeme <= 2)
		verr("slagGradientInit: the space-lag gradient needs the elastic "
		     "scheme (ischeme=3); got ischeme=%d.", mod->ischeme);
	if (dhx < 1 || dhz < 1)
		verr("slagGradientInit: lag strides are in GRID POINTS and must be "
		     ">= 1; got slag_dhx=%d slag_dhz=%d.  Unlike a time lag, a space "
		     "lag is an array index offset applied during accumulation, so "
		     "there is no sub-sample freedom.", dhx, dhz);

	if (nhx % 2 == 0) {
		nhx += 1;
		vwarn("slagGradientInit: slag_nhx must be odd so that hx=0 is "
		      "sampled; using %d.", nhx);
	}
	if (nhz % 2 == 0) {
		nhz += 1;
		vwarn("slagGradientInit: slag_nhz must be odd so that hz=0 is "
		      "sampled; using %d.", nhz);
	}

	sl_nhx   = nhx;
	sl_nhz   = nhz;
	sl_nlag  = nhx * nhz;
	sl_dhx   = dhx;
	sl_dhz   = dhz;
	sl_shx   = shiftx;
	sl_shz   = shiftz;
	sl_sizem = (size_t)mod->nax * mod->naz;

	/* The lag reach is checked against the ACCUMULATION BOUNDS, not here:
	 * those need bnd, which this function does not take, and the honest test
	 * is the one slagGradientAccum can make against its own px0/px1.  See the
	 * reach warning there. */

	nbytes = sl_sizem * (size_t)sl_nlag * sizeof(float);
	sl_glam = (float *)calloc(1, nbytes);
	sl_gmuu = (float *)calloc(1, nbytes);
	sl_exx  = (float *)calloc(sl_sizem, sizeof(float));
	sl_ezz  = (float *)calloc(sl_sizem, sizeof(float));
	sl_shr  = (float *)calloc(sl_sizem, sizeof(float));
	if (!sl_glam || !sl_gmuu || !sl_exx || !sl_ezz || !sl_shr)
		verr("slagGradientInit: cannot allocate %.1f MB of lag accumulators "
		     "(2 fields x %d lags x %zu points).  Space lags cost memory per "
		     "LAG, unlike time lags which cost memory per FREQUENCY -- either "
		     "reduce slag_nhx/slag_nhz, or run the x and z lag axes as two "
		     "separate (nhx,1) and (1,nhz) jobs.",
		     2.0 * nbytes / (1024.0*1024.0), sl_nlag, sl_sizem);

	sl_on = 1;

	if (verbose) {
		vmess("slagGradientInit: %d x %d lags (hx stride %d, hz stride %d pts) "
			"= %d planes", sl_nhx, sl_nhz, sl_dhx, sl_dhz, sl_nlag);
		vmess("slagGradientInit: |hx| <= %g m, |hz| <= %g m  (one-way; the "
			"RELATIVE separation is 2h)",
			((sl_nhx - 1) / 2) * sl_dhx * mod->dx,
			((sl_nhz - 1) / 2) * sl_dhz * mod->dz);
		vmess("slagGradientInit: accumulators %.1f MB",
			2.0 * nbytes / (1024.0*1024.0));
		if (sl_shx || sl_shz)
			vmess("slagGradientInit: CALIBRATION shift of the adjoint field "
				"(%d, %d) grid points -> the whole cube must translate by "
				"(%+g, %+g) m in x AND by (%+g, %+g) m in h",
				sl_shx, sl_shz,
				0.5f*sl_shx*mod->dx, 0.5f*sl_shz*mod->dz,
				0.5f*sl_shx*mod->dx, 0.5f*sl_shz*mod->dz);
	}
}


/***********************************************************************
 * slagGradientAccum -- Add one time step into every lag accumulator.
 *
 * Called from the backward loop of adj_shot at the point where psi_sigma is
 * the valid Lagrange multiplier (before callAdjKernel), with the forward
 * velocities of the SAME time index.  Unlike lagGradientAccum there is no `it`
 * argument: nothing here depends on absolute time, because the correlation is
 * formed in place rather than through a pair of transforms.
 *
 * Two passes: the forward strains of this step are evaluated ONCE into
 * sl_exx/sl_ezz/sl_shr, then each lag plane is a pure pointwise multiply-add.
 * The lag loop is the parallel one, so each thread owns a whole plane and the
 * Txz scatter cannot race -- and, since the scatter still runs ix/iz ascending
 * within a plane, the accumulation order is the same as accumGradient's, which
 * is what makes the h = 0 identity exact rather than approximate.
 ***********************************************************************/
void slagGradientAccum(modPar *mod, bndPar *bnd,
                       float *fwd_vx, float *fwd_vz,
                       wflPar *wfl_adj, float dt)
{
	int ix, iz, ilag, n1;
	int ibPx, iePx, ibPz, iePz;
	int ibTx, ieTx, ibTz, ieTz;
	float sdx, sdz, c1, c2, c3, c4;

	if (!sl_on) return;
	if (mod->ischeme <= 2) return;

	n1  = mod->naz;
	sdx = 1.0f / mod->dx;
	sdz = 1.0f / mod->dz;

	slagBounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz,
	                     &ibTx, &ieTx, &ibTz, &ieTz);
	slagCoef(mod, &c1, &c2, &c3, &c4);

	/* Reach check, once: each lag is inset by its own |h| on BOTH sides, so a
	 * lag reaching half the valid region accumulates nothing at all and the
	 * cube would taper to zero at its own edges for purely geometric reasons.
	 * Here and not in slagGradientInit because this is where the bounds are. */
	if (!sl_checked) {
		int hxmax = ((sl_nhx - 1) / 2) * sl_dhx + abs(sl_shx);
		int hzmax = ((sl_nhz - 1) / 2) * sl_dhz + abs(sl_shz);
		sl_checked = 1;
		if (2*hxmax >= iePx - ibPx || 2*hzmax >= iePz - ibPz)
			vwarn("slagGradientAccum: the largest lag reaches %d x %d grid "
			      "points but the valid region is only %d x %d, so the outer "
			      "lags are empty by construction.  Shrink slag_nhx/slag_nhz.",
			      hxmax, hzmax, iePx - ibPx, iePz - ibPz);
	}

	/* ---------------- 1. Strains of this step, once ------------------- */
#pragma omp parallel for private(ix, iz) schedule(static)
	for (ix = ibPx; ix < iePx; ix++) {
		for (iz = ibPz; iz < iePz; iz++) {
			int ig = ix*n1 + iz;
			float dvxdx, dvzdz;

			dvxdx = sdx*(c1*(fwd_vx[(ix+1)*n1+iz]-fwd_vx[ig])
			            +c2*(fwd_vx[(ix+2)*n1+iz]-fwd_vx[(ix-1)*n1+iz]));
			dvzdz = sdz*(c1*(fwd_vz[ig+1]-fwd_vz[ig])
			            +c2*(fwd_vz[ig+2]-fwd_vz[ig-1]));
			if (mod->iorder >= 6) {
				dvxdx += sdx*c3*(fwd_vx[(ix+3)*n1+iz]-fwd_vx[(ix-2)*n1+iz]);
				dvzdz += sdz*c3*(fwd_vz[ig+3]-fwd_vz[ig-2]);
			}
			if (mod->iorder >= 8) {
				dvxdx += sdx*c4*(fwd_vx[(ix+4)*n1+iz]-fwd_vx[(ix-3)*n1+iz]);
				dvzdz += sdz*c4*(fwd_vz[ig+4]-fwd_vz[ig-3]);
			}

			sl_exx[ig] = dvxdx;
			sl_ezz[ig] = dvzdz;
		}
	}

#pragma omp parallel for private(ix, iz) schedule(static)
	for (ix = ibTx; ix < ieTx; ix++) {
		for (iz = ibTz; iz < ieTz; iz++) {
			int ig = ix*n1 + iz;
			float dvxdz, dvzdx;

			dvxdz = sdz*(c1*(fwd_vx[ig]-fwd_vx[ig-1])
			            +c2*(fwd_vx[ig+1]-fwd_vx[ig-2]));
			dvzdx = sdx*(c1*(fwd_vz[ig]-fwd_vz[(ix-1)*n1+iz])
			            +c2*(fwd_vz[(ix+1)*n1+iz]-fwd_vz[(ix-2)*n1+iz]));
			if (mod->iorder >= 6) {
				dvxdz += sdz*c3*(fwd_vx[ig+2]-fwd_vx[ig-3]);
				dvzdx += sdx*c3*(fwd_vz[(ix+2)*n1+iz]-fwd_vz[(ix-3)*n1+iz]);
			}
			if (mod->iorder >= 8) {
				dvxdz += sdz*c4*(fwd_vx[ig+3]-fwd_vx[ig-4]);
				dvzdx += sdx*c4*(fwd_vz[(ix+3)*n1+iz]-fwd_vz[(ix-4)*n1+iz]);
			}

			sl_shr[ig] = dvxdz + dvzdx;
		}
	}

	/* ---------------- 2. One plane per lag ---------------------------- */
#pragma omp parallel for private(ilag, ix, iz) schedule(static)
	for (ilag = 0; ilag < sl_nlag; ilag++) {
		int oax, oaz, obx, obz;
		int px0, px1, pz0, pz1, tx0, tx1, tz0, tz1;
		long offa, offb;
		float *gl = sl_glam + (size_t)ilag * sl_sizem;
		float *gm = sl_gmuu + (size_t)ilag * sl_sizem;

		slagLagOffsets(ilag, &oax, &oaz, &obx, &obz);
		offa = (long)oax * n1 + oaz;
		offb = (long)obx * n1 + obz;

		/* Both x+oa and x+ob must land inside accumGradient's valid region.
		 * With no calibration shift this collapses to an inset of |h|. */
		px0 = ibPx - MIN(oax, obx);  px1 = iePx - MAX(oax, obx);
		pz0 = ibPz - MIN(oaz, obz);  pz1 = iePz - MAX(oaz, obz);
		tx0 = ibTx - MIN(oax, obx);  tx1 = ieTx - MAX(oax, obx);
		tz0 = ibTz - MIN(oaz, obz);  tz1 = ieTz - MAX(oaz, obz);

		/* --- lambda, and the normal-stress half of mu, on the P grid.
		 * The expressions are accumGradient's verbatim, including the
		 * multiplication order, so that at h = 0 the result is bit-for-bit
		 * the conventional gradient. */
		for (ix = px0; ix < px1; ix++) {
			for (iz = pz0; iz < pz1; iz++) {
				int  ig = ix*n1 + iz;
				long ia = ig + offa;
				long ib = ig + offb;
				float exx = sl_exx[ia], ezz = sl_ezz[ia];
				float div = exx + ezz;

				gl[ig] += dt*(wfl_adj->txx[ib] + wfl_adj->tzz[ib]) * div;
				gm[ig] += dt*2.0f*(wfl_adj->txx[ib]*exx
				                 + wfl_adj->tzz[ib]*ezz);
			}
		}

		/* --- shear half of mu, formed on the Txz grid and scattered to the
		 * 4 surrounding P points with weight 0.25, exactly as accumGradient
		 * does.  The scatter target is the MIDPOINT cell, unmoved by the
		 * symmetric lag. */
		for (ix = tx0; ix < tx1; ix++) {
			for (iz = tz0; iz < tz1; iz++) {
				int  ig = ix*n1 + iz;
				float shear = dt*wfl_adj->txz[ig + offb] * sl_shr[ig + offa];

				gm[(ix-1)*n1+iz-1] += 0.25f * shear;
				gm[ix*n1+iz-1]     += 0.25f * shear;
				gm[(ix-1)*n1+iz]   += 0.25f * shear;
				gm[ig]             += 0.25f * shear;
			}
		}
	}
}


/***********************************************************************
 * slagStrip -- Copy the interior of a padded array into a dense one.
 ***********************************************************************/
static void slagStrip(float *dst, float *src, modPar *mod, bndPar *bnd)
{
	int ix, iz;
	int ibndx = mod->ioPx;
	int ibndz = mod->ioPz;
	if (bnd->lef == 4 || bnd->lef == 2) ibndx += bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibndz += bnd->ntap;
	for (ix = 0; ix < mod->nx; ix++)
		for (iz = 0; iz < mod->nz; iz++)
			dst[ix * mod->nz + iz] =
				src[(ix + ibndx) * mod->naz + iz + ibndz];
}


/***********************************************************************
 * slagEmit -- Write one pair of lag cubes plus their axis description.
 *
 * `prefix` is the full stem, e.g. "slag_shot002" or "slag"; the files are
 * <prefix>_slagvp.su, <prefix>_slagvs.su and <prefix>_slagaxis.txt.
 ***********************************************************************/
static void slagEmit(modPar *mod, const char *prefix,
                     float *cube_vp, float *cube_vs, int verbose)
{
	int ihx, ihz;
	char fname[1024];
	FILE *fp;

	snprintf(fname, sizeof(fname), "%s_slagvp.su", prefix);
	writesufile(fname, cube_vp, (size_t)mod->nz,
		(size_t)mod->nx * sl_nlag, mod->z0, mod->x0, mod->dz, mod->dx);
	snprintf(fname, sizeof(fname), "%s_slagvs.su", prefix);
	writesufile(fname, cube_vs, (size_t)mod->nz,
		(size_t)mod->nx * sl_nlag, mod->z0, mod->x0, mod->dz, mod->dx);

	snprintf(fname, sizeof(fname), "%s_slagaxis.txt", prefix);
	fp = fopen(fname, "w");
	if (fp) {
		fprintf(fp, "# space-lag gradient axis description\n");
		fprintf(fp, "nhx   %d\n", sl_nhx);
		fprintf(fp, "nhz   %d\n", sl_nhz);
		fprintf(fp, "dhx   %.9g\n", sl_dhx * mod->dx);
		fprintf(fp, "dhz   %.9g\n", sl_dhz * mod->dz);
		fprintf(fp, "dhx_pts %d\n", sl_dhx);
		fprintf(fp, "dhz_pts %d\n", sl_dhz);
		fprintf(fp, "shiftx_pts %d\n", sl_shx);
		fprintf(fp, "shiftz_pts %d\n", sl_shz);
		fprintf(fp, "nx    %d\n", mod->nx);
		fprintf(fp, "nz    %d\n", mod->nz);
		fprintf(fp, "dx    %.9g\n", mod->dx);
		fprintf(fp, "dz    %.9g\n", mod->dz);
		fprintf(fp, "x0    %.9g\n", mod->x0);
		fprintf(fp, "z0    %.9g\n", mod->z0);
		fprintf(fp, "# cube order: trace = (ihz*nhx + ihx)*nx + ix, "
			"nz samples per trace\n");
		fprintf(fp, "# lags are ONE-WAY: the relative separation of the two "
			"wavefields is 2h\n");
		fprintf(fp, "# hx (m):\n");
		for (ihx = 0; ihx < sl_nhx; ihx++)
			fprintf(fp, "%.9g\n",
				(float)(ihx - (sl_nhx-1)/2) * sl_dhx * mod->dx);
		fprintf(fp, "# hz (m):\n");
		for (ihz = 0; ihz < sl_nhz; ihz++)
			fprintf(fp, "%.9g\n",
				(float)(ihz - (sl_nhz-1)/2) * sl_dhz * mod->dz);
		fclose(fp);
	}
	else vwarn("slagEmit: cannot write %s", fname);

	if (verbose)
		vmess("slagGradient: wrote %s_slag{vp,vs}.su "
			"(%d x %d x %d x %d, lag slowest)",
			prefix, sl_nhz, sl_nhx, mod->nx, mod->nz);
}


/***********************************************************************
 * slagBuildCubes -- Convert the accumulators to (Vp, Vs) cubes.
 *
 * cube_vp, cube_vs must each hold nlag * mod->nx * mod->nz floats, ordered
 * (ihz, ihx, x, z) with z fastest.
 ***********************************************************************/
static void slagBuildCubes(modPar *mod, bndPar *bnd, float *mask,
                           float *cube_vp, float *cube_vs, int verbose)
{
	int ilag, i, nmodel;
	float *slice;

	nmodel = mod->nx * mod->nz;
	slice  = (float *)malloc(sl_sizem * sizeof(float));
	if (!slice) verr("slagBuildCubes: cannot allocate slice scratch.");

	for (ilag = 0; ilag < sl_nlag; ilag++) {
		float *glam = sl_glam + (size_t)ilag * sl_sizem;
		float *gmuu = sl_gmuu + (size_t)ilag * sl_sizem;
		float *pvp  = cube_vp + (size_t)ilag * nmodel;
		float *pvs  = cube_vs + (size_t)ilag * nmodel;
		double n_vp = 0.0, n_vs = 0.0;

		/* Chain rule Lame -> (Vp, Vs) with exactly the factors
		 * convertGradientToVelocity uses, evaluated at the MIDPOINT:
		 *     g_Vp = 2 rho Vp  g_lam
		 *     g_Vs = 2 rho Vs (g_mu - 2 g_lam)
		 * Pointwise and lag-independent, so it commutes with the lag axis --
		 * which is what keeps the h = 0 check against <base>_vp_raw.su exact.
		 * No g_rho cube: the accumulators carry only lam and mu, and the
		 * density term needs the direct rho kernel too. */
		for (i = 0; i < (int)sl_sizem; i++)
			slice[i] = 2.0f * mod->rho[i] * mod->cp[i] * glam[i];
		slagStrip(pvp, slice, mod, bnd);

		for (i = 0; i < (int)sl_sizem; i++)
			slice[i] = 2.0f * mod->rho[i] * mod->cs[i]
			           * (gmuu[i] - 2.0f * glam[i]);
		slagStrip(pvs, slice, mod, bnd);

		/* Same taper as the conventional gradient, on every lag. */
		if (mask != NULL) {
			for (i = 0; i < nmodel; i++) { pvp[i] *= mask[i]; pvs[i] *= mask[i]; }
		}

		if (verbose > 1) {
			int ihx = ilag % sl_nhx, ihz = ilag / sl_nhx;
			for (i = 0; i < nmodel; i++) {
				n_vp += (double)pvp[i]*pvp[i];
				n_vs += (double)pvs[i]*pvs[i];
			}
			vmess("  hx=%+8.2f m  hz=%+8.2f m   ||g_vp||=%.6e   ||g_vs||=%.6e",
				(float)(ihx - (sl_nhx-1)/2) * sl_dhx * mod->dx,
				(float)(ihz - (sl_nhz-1)/2) * sl_dhz * mod->dz,
				sqrt(n_vp), sqrt(n_vs));
		}
	}

	free(slice);
}


/***********************************************************************
 * slagGradientWrite -- Build and write this shot's (Vp, Vs) lag cubes.
 *
 *   mask   optional [nx*nz] multiplier on the interior grid, ix*nz+iz, or
 *          NULL.  Pass the SAME grad_taper mask the conventional gradient
 *          gets: without it the near-source singularity holds essentially all
 *          the energy and swamps every display, and the taper cannot be
 *          applied downstream because adj_shot tapers g*_shot, not these
 *          accumulators.  Using one mask on both also keeps the h = 0 check
 *          like for like.
 *
 * Output, two SU files per shot:
 *   <base>_shotNNN_slagvp.su
 *   <base>_shotNNN_slagvs.su
 * each nz samples by (nx * nhx * nhz) traces, with the LAG as the SLOWEST
 * axis: trace = (ihz*nhx + ihx)*nx + ix.  Reshape to (nhz, nhx, nx, nz).
 *
 * A companion <base>_shotNNN_slagaxis.txt records the lag and grid axes, so
 * plotting cannot silently disagree with what was run.
 *
 * Each call also adds into the shot stack; see slagGradientWriteStack.
 ***********************************************************************/
void slagGradientWrite(modPar *mod, bndPar *bnd, const char *base, int ishot,
                       float *mask, int verbose)
{
	int i, nmodel;
	float *cube_vp, *cube_vs;
	char prefix[1024];

	if (!sl_on) return;

	nmodel  = mod->nx * mod->nz;
	cube_vp = (float *)malloc((size_t)nmodel * sl_nlag * sizeof(float));
	cube_vs = (float *)malloc((size_t)nmodel * sl_nlag * sizeof(float));
	if (!cube_vp || !cube_vs)
		verr("slagGradientWrite: cannot allocate the lag cubes "
		     "(%d lags x %d points).", sl_nlag, nmodel);

	if (verbose)
		vmess("slagGradientWrite: shot %d, %d x %d lags, "
			"|hx|<=%g m |hz|<=%g m", ishot, sl_nhx, sl_nhz,
			((sl_nhx - 1) / 2) * sl_dhx * mod->dx,
			((sl_nhz - 1) / 2) * sl_dhz * mod->dz);

	slagBuildCubes(mod, bnd, mask, cube_vp, cube_vs, verbose);

	snprintf(prefix, sizeof(prefix), "%s_shot%03d", base, ishot);
	slagEmit(mod, prefix, cube_vp, cube_vs, verbose);

	/* ---- add into the shot stack ---- */
	if (sl_stk_vp == NULL) {
		sl_stk_vp = (float *)calloc((size_t)nmodel * sl_nlag, sizeof(float));
		sl_stk_vs = (float *)calloc((size_t)nmodel * sl_nlag, sizeof(float));
		if (!sl_stk_vp || !sl_stk_vs)
			verr("slagGradientWrite: cannot allocate the shot stack.");
		sl_stk_nlag = sl_nlag; sl_stk_nmodel = nmodel;
		sl_stk_nhx  = sl_nhx;  sl_stk_nhz    = sl_nhz;
	}
	else if (sl_stk_nlag != sl_nlag || sl_stk_nmodel != nmodel ||
	         sl_stk_nhx != sl_nhx || sl_stk_nhz != sl_nhz) {
		verr("slagGradientWrite: shot %d has a lag axis (%d x %d, %d points) "
		     "that differs from the stack (%d x %d, %d points).  All shots "
		     "must share one lag axis to be summed.",
		     ishot, sl_nhx, sl_nhz, nmodel,
		     sl_stk_nhx, sl_stk_nhz, sl_stk_nmodel);
	}
	for (i = 0; i < nmodel * sl_nlag; i++) {
		sl_stk_vp[i] += cube_vp[i];
		sl_stk_vs[i] += cube_vs[i];
	}
	sl_stk_nshot++;

	free(cube_vp); free(cube_vs);
}


/***********************************************************************
 * slagGradientWriteStack -- Write the sum of all shots written so far.
 *
 * The FWI gradient is a sum over shots, so this -- not any single shot -- is
 * the cube whose h = 0 slice must reproduce <base>_vp_raw.su and
 * <base>_vs_raw.su.
 *
 * Output: <base>_slagvp.su, <base>_slagvs.su, <base>_slagaxis.txt.
 * Call once after the shot loop, before slagGradientFree.
 ***********************************************************************/
void slagGradientWriteStack(modPar *mod, const char *base, int verbose)
{
	if (!sl_on || sl_stk_vp == NULL) return;

	if (verbose)
		vmess("slagGradientWriteStack: stacking %d shot%s onto one lag cube",
			sl_stk_nshot, sl_stk_nshot == 1 ? "" : "s");

	slagEmit(mod, base, sl_stk_vp, sl_stk_vs, verbose);
}


/***********************************************************************
 * slagGradientStackZeroLag -- Pointer to the h = 0 plane of the shot stack.
 *
 *   ivar  0 for g_Vp, 1 for g_Vs
 *
 * Returns NULL if nothing has been stacked.  Exists so the driver can make
 * the h = 0 comparison against the conventional gradient numerically, in the
 * same run, instead of reading its own SU output back.  That comparison is the
 * exact check this module is built around, so it should not depend on a
 * separate plotting step to be noticed.
 ***********************************************************************/
float *slagGradientStackZeroLag(int ivar)
{
	int mid;

	if (!sl_on || sl_stk_vp == NULL) return NULL;

	mid = ((sl_stk_nhz - 1) / 2) * sl_stk_nhx + (sl_stk_nhx - 1) / 2;

	return ((ivar == 0) ? sl_stk_vp : sl_stk_vs)
	       + (size_t)mid * sl_stk_nmodel;
}


/***********************************************************************
 * slagFreeAccum -- Release the per-shot accumulators, keep the shot stack.
 ***********************************************************************/
static void slagFreeAccum(void)
{
	if (sl_glam) { free(sl_glam); sl_glam = NULL; }
	if (sl_gmuu) { free(sl_gmuu); sl_gmuu = NULL; }
	if (sl_exx)  { free(sl_exx);  sl_exx  = NULL; }
	if (sl_ezz)  { free(sl_ezz);  sl_ezz  = NULL; }
	if (sl_shr)  { free(sl_shr);  sl_shr  = NULL; }
	sl_on      = 0;
	sl_nhx     = sl_nhz = sl_nlag = 0;
	sl_dhx     = sl_dhz = 1;
	sl_shx     = sl_shz = 0;
	sl_checked = 0;
	sl_sizem   = 0;
}


/***********************************************************************
 * slagGradientFree -- Release everything, including the shot stack.
 *
 * Call once after the shot loop.  Calling it between shots would discard the
 * stack; slagGradientInit already clears the per-shot accumulators.
 ***********************************************************************/
void slagGradientFree(void)
{
	slagFreeAccum();
	if (sl_stk_vp) { free(sl_stk_vp); sl_stk_vp = NULL; }
	if (sl_stk_vs) { free(sl_stk_vs); sl_stk_vs = NULL; }
	sl_stk_nlag = sl_stk_nmodel = sl_stk_nshot = 0;
	sl_stk_nhx  = sl_stk_nhz = 0;
}
