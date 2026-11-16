#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"

/**
 * lag_gradient.c -- Time-lag extended imaging condition for the elastic
 *                   FWI gradient (lambda, mu).
 *
 * WHAT THIS COMPUTES
 * ------------------
 * The conventional FWI gradient is a ZERO-LAG correlation in time between
 * the forward strain and the adjoint stress (see accumGradient in
 * fwi_gradient.c):
 *
 *   g_lam = dt sum_t  div(v_f) * (psi_txx + psi_tzz)
 *   g_mu  = dt sum_t  2[ Dx(vx_f) psi_txx + Dz(vz_f) psi_tzz ]
 *                   + (Dz(vx_f) + Dx(vz_f)) psi_txz
 *
 * This module evaluates the SAME kernels at non-zero time lag tau.  The lag
 * is applied SYMMETRICALLY, the forward field shifted back by tau and the
 * adjoint field forward by tau:
 *
 *   g_lam(x,tau) = dt sum_t div(v_f)(x,t-tau) * (psi_txx + psi_tzz)(x,t+tau)
 *
 * so tau is the ONE-WAY shift carried by each wavefield and 2*tau is the
 * relative lag between them.  Because the sum over t is complete, only the
 * relative shift survives,
 *
 *   g_sym(tau) = g_oneside(2 tau),
 *
 * i.e. this convention is the one-sided correlation with the lag axis
 * stretched by two.  It is NOT a cosmetic relabelling: every statement about
 * the lag axis below -- the aliasing period, and the slope-to-angle map used
 * by demo/gradient/plot_lag_slant.py -- carries the factor 2.
 *
 * tau = 0 reproduces the conventional gradient and tau != 0 exposes
 * the time shift between the forward wavefield and the backpropagated
 * residual.  The adjoint source is the DATA RESIDUAL, so this is a genuine
 * lag-extension of dJ/dm: at the correct model the residual vanishes and the
 * gather is zero at every lag.  The diagnostic is therefore the SHAPE of
 * g(x,tau) in tau -- a lag-asymmetric kernel means the modelled arrival is
 * early or late relative to the observed one at that image point.
 *
 * WHY THE FREQUENCY DOMAIN
 * ------------------------
 * A lag correlation is a product of two independent transforms:
 *
 *   g(x,tau) = (dw/pi) sum_k Re{ conj(A^)(x,w_k) B^(x,w_k) exp(-2i w_k tau) }
 *
 * with A^(w) = dt sum_t A(t) exp(+i w t), likewise B^.  The 2 is the
 * symmetric convention: shifting A by -tau multiplies A^ by exp(+i w tau)
 * and shifting B by +tau multiplies B^ by exp(-i w tau), and conjugating the
 * first makes the two phases ADD rather than cancel.  The decisive point
 * for this code base is that A^ and B^ are accumulated INDEPENDENTLY, each
 * at its own time step.  adj_shot.c reconstructs the forward wavefield one
 * checkpoint segment at a time while marching the adjoint backward, so the
 * forward and adjoint states are only ever co-resident AT THE SAME TIME
 * INDEX.  A time-domain lag loop would need states from different times
 * simultaneously, forcing ring buffers and cross-segment carry buffers.
 * Here nothing of the sort is required: at step `it` we fold both fields
 * into their own DFT bins and never look at another time step.
 *
 * Consequences:
 *   - The lag axis (range, sampling, even sub-sample lags) is chosen at
 *     WRITE time, not at run time: exp(-i w tau) does not care whether tau
 *     is a multiple of dt.  Memory depends on nfreq, never on nlag.
 *   - The absolute time origin is irrelevant.  Shifting t -> t + c
 *     multiplies both A^ and B^ by exp(i w c), which cancels in conj(A^)B^.
 *     So mod->t0 needs no bookkeeping here.
 *
 * TEN ACCUMULATORS
 * ----------------
 * Six carry the lambda and mu kernels, four the density kernel; the OUTPUT is
 * g_Vp, g_Vs and g_rho, converted per lag by the same pointwise chain rule
 * convertGradientToVelocity applies to the conventional gradient,
 *
 *     g_Vp  = 2 rho Vp  g_lam
 *     g_Vs  = 2 rho Vs (g_mu - 2 g_lam)
 *     g_rho = g_rho_direct + g_lam (Vp^2 - 2 Vs^2) + g_mu Vs^2
 *
 * The map is pointwise and lag-independent, so it commutes with the lag
 * reconstruction: converting each lag is identical to reconstructing the lag
 * axis of the already-converted kernel.  That is what makes the tau = 0 slice
 * directly comparable to <out_base>_vp_raw.su from a param=2 run.
 *
 * The ten transformed fields are
 *
 *   forward  (P   grid):  Dx(vx), Dz(vz)              -> div = sum of the two
 *   forward  (Txz grid):  Dz(vx) + Dx(vz)
 *   adjoint  (P   grid):  psi_txx, psi_tzz            -> psi sum derived
 *   adjoint  (Txz grid):  psi_txz
 *   forward  (Vx  grid): -rox Dsig_vx                 \  density, see below
 *   adjoint  (Vx  grid):  psi_vx                      |
 *   forward  (Vz  grid): -roz Dsig_vz                 |
 *   adjoint  (Vz  grid):  psi_vz                      /
 *
 * THE DENSITY KERNEL
 * ------------------
 * g_rho_direct is the transpose of the F1 (velocity) update, and
 * accumGradient_rho_Dsig forms it as
 *
 *   g_rho[P] += sum_t 0.5 (-psi_vx rox Dsig_vx)/rho   scattered to the two
 *   g_rho[P] += sum_t 0.5 (-psi_vz roz Dsig_vz)/rho   P points each v straddles
 *
 * with Dsig_vx = D-x(txx) + D+z(txz) on the Vx grid and
 * Dsig_vz = D+x(txz) + D-z(tzz) on the Vz grid.  That is again a product of
 * one forward-only and one adjoint-only field at a single time index, so it
 * lag-extends exactly like the others: transform each factor, multiply
 * conj(A)B, and scatter the reconstructed correlation with the same 0.5/rho
 * weights.  rox/roz/rho are time independent and are folded into the forward
 * factor.  The loop bounds are accumGradient_rho_Dsig's (ioXx..ieXx,
 * ioZx..ieZx), which unlike the lambda/mu bounds exclude neither the taper
 * nor the free-surface row -- mirrored deliberately, so the tau = 0 slice is
 * formed over the same points as the conventional g_rho.
 *
 * TWO BOOKKEEPING POINTS, both of which would otherwise bias the rho cube
 * against the lambda/mu cubes:
 *
 *  - dt.  The lambda/mu kernels carry one dt from the time integral and the
 *    2*df reconstruction of two dt-weighted transforms supplies exactly that.
 *    accumGradient_rho_Dsig carries NO dt (the old dv/dt form had
 *    dt*(dvx/dt) = dvx, and the dt cancelled).  Both rho factors are still
 *    accumulated dt-weighted, for one code path, and the rho reconstruction
 *    weight is 2*df/dt instead of 2*df.
 *
 *  - THE TIME STAMP.  The rho pair is formed at a different point of the
 *    backward loop from the lambda/mu pair -- forward stress sigma^{n-1} with
 *    psi_v AFTER callAdjKernel, against forward strain(v^n) with psi_sigma
 *    BEFORE it -- so the obvious move is to stamp the stress with (it-1)*dt.
 *    That is wrong.  On the staggered grid v^n sits at n dt and sigma^n at
 *    (n+1/2) dt, so the TRUE intra-pair offsets are
 *        lambda/mu:  psi_sigma at (n+1/2) dt  minus  strain at  n     dt
 *        rho:        psi_v     at  n     dt   minus  sigma  at (n-1/2) dt
 *    i.e. +dt/2 for both.  Stamping every field with its own integer step
 *    `it` therefore gives BOTH pairs the same residual bias of -dt/2, which
 *    is a common shift of the whole lag axis by -dt/4 in tau (the relative
 *    lag is 2 tau) and cancels between the three cubes.  Stamping the stress
 *    at it-1 would make the rho cube exact in isolation but shift it by dt/2
 *    in the relative lag -- dt/4 in tau -- RELATIVE to g_Vp and g_Vs, which
 *    is the one error that matters when the cubes are read together.
 *    So lagGradientAccumRho takes the same `it` as lagGradientAccum.
 *
 * CAVEATS (read before interpreting output)
 * -----------------------------------------
 *  - A sparse frequency set ALIASES the lag axis.  The reconstruction is
 *    periodic with period 1/df in the RELATIVE lag, df the frequency
 *    spacing, hence period 1/(2 df) in tau under this symmetric convention.
 *    Pick too few frequencies and the wrap-around looks exactly like lag
 *    moveout, which is the one thing being measured.  lagGradientWrite
 *    refuses to run when |tau| reaches half that period, i.e. 1/(4 df).
 *  - The tau = 0 slice is NOT bit-identical to accumGradient's output: it is
 *    band limited to [lag_fmin, lag_fmax] over nfreq discrete lines.  The
 *    ratio g_dft(0)/g_time(0) is a useful quantitative measure of how much
 *    of the kernel the frequency set captures -- compare against
 *    <out_base>_vp_raw.su and <out_base>_vs_raw.su from a param=2 run with
 *    grad_taper=0 smooth_grad=0.
 *  - The lag SIGN convention follows g(tau) = dt sum_t A(t-tau) B(t+tau)
 *    with A the forward strain and B the adjoint stress, i.e. positive tau
 *    pairs the adjoint at a LATER time with the forward at an earlier one,
 *    separated by 2*tau.  Which
 *    physical velocity error that corresponds to is best fixed empirically
 *    on a deliberately mis-scaled background rather than asserted here.
 *  - Only the elastic scheme (ischeme = 3 or 5) and iorder 4/6/8 are
 *    handled.  Accumulation is a no-op otherwise.
 *
 * Loop bounds and FD stencils are duplicated from accumGradient on purpose,
 * so that the tau = 0 slice is formed over exactly the same grid points.
 */

#define MAX(x,y) ((x) > (y) ? (x) : (y))

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

static void lagFreeAccum(void);

/* ------------------------------------------------------------------ *
 * Module state.  Inactive until lagGradientInit is called with
 * nfreq > 0, so every entry point is a cheap no-op by default.
 * ------------------------------------------------------------------ */
static int    lg_on    = 0;
static int    lg_nfreq = 0;
static size_t lg_sizem = 0;
static float *lg_omega = NULL;   /* [nfreq] angular frequency, rad/s     */
static float  lg_df    = 0.0f;   /* frequency spacing in Hz (0 if nfreq=1) */

/* Complex DFT accumulators.  Layout is FREQUENCY-FASTEST with re/im
 * interleaved: element (ig,k) lives at ((size_t)ig*nfreq + k)*2.  The time
 * loop touches every frequency of one grid point in succession, so this
 * keeps each point's nfreq complex values on the same few cache lines. */
static float *lg_exx = NULL;   /* forward D+x(vx)            , P   grid */
static float *lg_ezz = NULL;   /* forward D+z(vz)            , P   grid */
static float *lg_shr = NULL;   /* forward D-z(vx)+D-x(vz)    , Txz grid */
static float *lg_pxx = NULL;   /* adjoint psi_txx            , P   grid */
static float *lg_pzz = NULL;   /* adjoint psi_tzz            , P   grid */
static float *lg_pxz = NULL;   /* adjoint psi_txz            , Txz grid */

/* Density accumulators.  Allocated only when lagGradientInit is given
 * do_rho != 0; they are 4 more complex fields, +67% on the memory above. */
static int    lg_rho_on   = 0;
static int    lg_rho_nacc = 0; /* steps folded in, to catch a silent zero cube */
static float  lg_dt       = 0.0f;  /* dt seen by the accumulators            */
static float *lg_dsx = NULL;   /* forward -rox*Dsig_vx       , Vx  grid */
static float *lg_pvx = NULL;   /* adjoint psi_vx             , Vx  grid */
static float *lg_dsz = NULL;   /* forward -roz*Dsig_vz       , Vz  grid */
static float *lg_pvz = NULL;   /* adjoint psi_vz             , Vz  grid */

/* Scratch for the per-step phase factors, sized nfreq. */
static float *lg_cos = NULL, *lg_sin = NULL;

/* Shot stack.  The FWI gradient is a sum over shots, so the lag cube that
 * corresponds to <out_base>_vp_raw.su is the SUM of the per-shot cubes, not
 * any one of them.  These buffers live across lagGradientInit (which only
 * clears the per-shot DFT fields) and are released by lagGradientFree. */
static float *lg_stk_vp = NULL, *lg_stk_vs = NULL, *lg_stk_rho = NULL;
static int    lg_stk_nlag = 0, lg_stk_nmodel = 0, lg_stk_nshot = 0;
static float  lg_stk_dlag = 0.0f;


/***********************************************************************
 * lagBounds -- Loop bounds for the P and Txz grids.
 *
 * Reproduces the bounds logic of accumGradient (fwi_gradient.c) so the
 * tau = 0 slice is accumulated over identical grid points: absorbing
 * taper exclusion, free-surface row exclusion, and FD stencil margins.
 ***********************************************************************/
static void lagBounds(modPar *mod, bndPar *bnd,
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
 * lagCoef -- FD coefficients matching the forward stress update.
 ***********************************************************************/
static void lagCoef(modPar *mod, float *c1, float *c2, float *c3, float *c4)
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
 * lagGradientInit -- Allocate the DFT accumulators and frequency grid.
 *
 *   nfreq  number of frequencies; <= 0 disables the module entirely
 *   fmin   lowest frequency in Hz (must be > 0; DC carries no signal)
 *   fmax   highest frequency in Hz (ignored when nfreq == 1)
 *   do_rho allocate the four density accumulators as well (+67% memory).
 *          Required for the g_rho lag cube; pass 0 when only g_Vp and g_Vs
 *          are wanted, as lagGradientTomo does.
 *
 * Frequencies are placed on a uniform grid, because the aliasing period
 * of the lag reconstruction is 1/df and that only has meaning for even
 * spacing.  Call once per shot: this zeroes the accumulators.  It does NOT
 * touch the shot stack, so a multi-shot loop keeps accumulating.
 ***********************************************************************/
void lagGradientInit(modPar *mod, int nfreq, float fmin, float fmax,
                     int do_rho, int verbose)
{
	int k;
	size_t nbytes;
	int nfld;

	lagFreeAccum();

	if (nfreq <= 0) { lg_on = 0; return; }

	if (mod->ischeme <= 2)
		verr("lagGradientInit: time-lag gradient needs the elastic scheme "
		     "(ischeme=3); got ischeme=%d.", mod->ischeme);
	if (fmin <= 0.0f)
		verr("lagGradientInit: lag_fmin=%g must be > 0 Hz.", fmin);
	if (nfreq > 1 && fmax <= fmin)
		verr("lagGradientInit: need lag_fmax (%g) > lag_fmin (%g).",
		     fmax, fmin);

	lg_nfreq = nfreq;
	lg_sizem = (size_t)mod->nax * mod->naz;
	lg_df    = (nfreq > 1) ? (fmax - fmin) / (float)(nfreq - 1) : 0.0f;

	/* DENSITY REQUIREMENT -- df <= 1/T, T the record length.
	 *
	 * This is stricter than the aliasing guard in lagGradientWrite, and it is
	 * the one that actually matters.  The discrete sum replicates g with
	 * period 1/df in the RELATIVE lag.  The aliasing guard only asks that the
	 * *displayed* lag range fit inside one period; it says nothing about the
	 * correlation energy OUTSIDE that range, which the replication folds back
	 * in.  conj(A)B is a correlation over the whole record, so its support in
	 * relative lag is the record length T -- hence 1/df >= T, i.e. the sum
	 * must have at least the natural frequency resolution of the record.
	 *
	 * Measured on the 1-shot _rp setup, T=2.5 s, band [2,24] Hz:
	 *   nfreq=89 (df=0.25, 1/df=4.0 s > T)  -> tau=0 corr 0.99999, ratio 1.000
	 *   nfreq=9  (df=3.00, 1/df=0.33 s < T) -> tau=0 corr 0.22,    ratio 0.69
	 * Both passed the aliasing guard.  Only the first is a gradient. */
	if (lg_df > 0.0f) {
		float trec   = (float)mod->nt * mod->dt;
		int   nf_min = (int)ceilf((fmax - fmin) * trec) + 1;
		if (lg_df * trec > 1.0f)
			verr("lagGradientInit: lag_nfreq=%d gives df=%.4f Hz over "
			     "[%g, %g] Hz, but the record is T=%.3f s long and the lag "
			     "reconstruction replicates with period 1/df=%.4f s in the "
			     "relative lag.  With 1/df < T the replication folds real "
			     "correlation energy back into the lag window and the cube is "
			     "NOT a gradient (the tau=0 slice stops matching the "
			     "conventional gradient).  Use lag_nfreq>=%d so that "
			     "df<=1/T=%.4f Hz.",
			     nfreq, lg_df, fmin, fmax, trec, 1.0f/lg_df,
			     nf_min, 1.0f/trec);
	}

	lg_omega = (float *)malloc((size_t)nfreq * sizeof(float));
	lg_cos   = (float *)malloc((size_t)nfreq * sizeof(float));
	lg_sin   = (float *)malloc((size_t)nfreq * sizeof(float));
	if (!lg_omega || !lg_cos || !lg_sin)
		verr("lagGradientInit: cannot allocate frequency scratch.");

	for (k = 0; k < nfreq; k++)
		lg_omega[k] = 2.0f * M_PI * (fmin + lg_df * (float)k);

	/* Six (ten with rho) complex fields: 2 floats per (point, frequency). */
	nbytes = lg_sizem * (size_t)nfreq * 2 * sizeof(float);
	nfld   = do_rho ? 10 : 6;
	lg_exx = (float *)calloc(1, nbytes);
	lg_ezz = (float *)calloc(1, nbytes);
	lg_shr = (float *)calloc(1, nbytes);
	lg_pxx = (float *)calloc(1, nbytes);
	lg_pzz = (float *)calloc(1, nbytes);
	lg_pxz = (float *)calloc(1, nbytes);
	if (!lg_exx || !lg_ezz || !lg_shr || !lg_pxx || !lg_pzz || !lg_pxz)
		verr("lagGradientInit: cannot allocate %.1f MB of DFT accumulators "
		     "(%d complex fields x %d frequencies x %zu points).",
		     nfld * nbytes / (1024.0*1024.0), nfld, nfreq, lg_sizem);

	if (do_rho) {
		lg_dsx = (float *)calloc(1, nbytes);
		lg_pvx = (float *)calloc(1, nbytes);
		lg_dsz = (float *)calloc(1, nbytes);
		lg_pvz = (float *)calloc(1, nbytes);
		if (!lg_dsx || !lg_pvx || !lg_dsz || !lg_pvz)
			verr("lagGradientInit: cannot allocate %.1f MB of DFT accumulators "
			     "(%d complex fields x %d frequencies x %zu points).  The last "
			     "four are the density kernel; drop it to fit in 6/10 of this.",
			     nfld * nbytes / (1024.0*1024.0), nfld, nfreq, lg_sizem);
		lg_rho_on   = 1;
		lg_rho_nacc = 0;
	}

	lg_on = 1;

	if (verbose) {
		vmess("lagGradientInit: %d frequencies, %.3f - %.3f Hz, df=%.4f Hz",
			nfreq, fmin, fmin + lg_df * (nfreq - 1), lg_df);
		vmess("lagGradientInit: %d accumulators %.1f MB (independent of lag "
			"count), density kernel %s",
			nfld, nfld * nbytes / (1024.0*1024.0),
			do_rho ? "INCLUDED" : "off");
		if (lg_df > 0.0f)
			vmess("lagGradientInit: aliasing period 1/df = %.4f s in the "
				"relative lag = %.4f s in tau -> usable |tau| < %.4f s",
				1.0f/lg_df, 0.5f/lg_df, 0.25f/lg_df);
	}
}


/***********************************************************************
 * lagGradientAccum -- Fold one time step into the running DFTs.
 *
 * Called from the backward loop of adj_shot at the point where psi_sigma
 * is the valid Lagrange multiplier (before callAdjKernel), with the
 * forward velocities of the SAME time index `it`.
 *
 * The forward strains are computed once per grid point and then folded
 * into all nfreq bins, so the FD stencils are evaluated once regardless
 * of how many frequencies are requested.
 ***********************************************************************/
void lagGradientAccum(modPar *mod, bndPar *bnd,
                      float *fwd_vx, float *fwd_vz,
                      wflPar *wfl_adj, int it, float dt)
{
	int ix, iz, k, n1, nfreq;
	int ibPx, iePx, ibPz, iePz;
	int ibTx, ieTx, ibTz, ieTz;
	float sdx, sdz, c1, c2, c3, c4, t;

	if (!lg_on) return;
	if (mod->ischeme <= 2) return;

	n1    = mod->naz;
	nfreq = lg_nfreq;
	sdx   = 1.0f / mod->dx;
	sdz   = 1.0f / mod->dz;
	lg_dt = dt;

	lagBounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz,
	                    &ibTx, &ieTx, &ibTz, &ieTz);
	lagCoef(mod, &c1, &c2, &c3, &c4);

	/* Phase factors exp(+i w t) with the dt of the time integral folded
	 * in.  Two trig calls per frequency per step -- negligible next to
	 * the nfreq x sizem field work, and far more robust than advancing a
	 * recursion that would silently depend on the segment loop visiting
	 * every time index exactly once. */
	t = (float)it * dt;
	for (k = 0; k < nfreq; k++) {
		lg_cos[k] = cosf(lg_omega[k] * t) * dt;
		lg_sin[k] = sinf(lg_omega[k] * t) * dt;
	}

	/* ---------------- P grid: normal strains and psi_txx/psi_tzz ------- */
#pragma omp parallel for private(ix, iz, k) schedule(static)
	for (ix = ibPx; ix < iePx; ix++) {
		for (iz = ibPz; iz < iePz; iz++) {
			int ig = ix*n1 + iz;
			size_t o0 = (size_t)ig * nfreq * 2;
			float dvxdx, dvzdz, axx, azz;

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

			axx = wfl_adj->txx[ig];
			azz = wfl_adj->tzz[ig];

			for (k = 0; k < nfreq; k++) {
				size_t o = o0 + (size_t)k*2;
				float cr = lg_cos[k], ci = lg_sin[k];
				lg_exx[o  ] += dvxdx*cr;  lg_exx[o+1] += dvxdx*ci;
				lg_ezz[o  ] += dvzdz*cr;  lg_ezz[o+1] += dvzdz*ci;
				lg_pxx[o  ] += axx  *cr;  lg_pxx[o+1] += axx  *ci;
				lg_pzz[o  ] += azz  *cr;  lg_pzz[o+1] += azz  *ci;
			}
		}
	}

	/* ---------------- Txz grid: shear strain and psi_txz -------------- */
#pragma omp parallel for private(ix, iz, k) schedule(static)
	for (ix = ibTx; ix < ieTx; ix++) {
		for (iz = ibTz; iz < ieTz; iz++) {
			int ig = ix*n1 + iz;
			size_t o0 = (size_t)ig * nfreq * 2;
			float dvxdz, dvzdx, shear, axz;

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

			shear = dvxdz + dvzdx;
			axz   = wfl_adj->txz[ig];

			for (k = 0; k < nfreq; k++) {
				size_t o = o0 + (size_t)k*2;
				float cr = lg_cos[k], ci = lg_sin[k];
				lg_shr[o  ] += shear*cr;  lg_shr[o+1] += shear*ci;
				lg_pxz[o  ] += axz  *cr;  lg_pxz[o+1] += axz  *ci;
			}
		}
	}
}


/***********************************************************************
 * lagGradientAccumRho -- Fold one time step into the density DFTs.
 *
 * Called from the backward loop of adj_shot at the accumGradient_rho_Dsig
 * call site, i.e. AFTER callAdjKernel, where psi_v is the valid multiplier
 * for the F1 (velocity) update, and with the SAME forward stress that call
 * is given: sigma^{n-1}, the stress before forward step n.  Mirror its
 * gating exactly, or the tau = 0 slice stops reproducing grad_rho.
 *
 * `it` is the integer step of the adjoint state, NOT of the stress.  See
 * "THE TIME STAMP" in the file header: stamping both factors with `it`
 * reproduces the lambda/mu pair's residual half-step bias instead of adding
 * a different one, so the three cubes share one lag reference.
 *
 * A no-op unless lagGradientInit was called with do_rho != 0.
 ***********************************************************************/
void lagGradientAccumRho(modPar *mod,
                         float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                         wflPar *wfl_adj, int it, float dt)
{
	int ix, iz, k, n1, nfreq;
	float c1, c2, c3, c4, t;

	if (!lg_on || !lg_rho_on) return;
	if (mod->ischeme <= 2) return;
	if (!fwd_txx || !fwd_tzz || !fwd_txz) return;

	n1    = mod->naz;
	nfreq = lg_nfreq;
	lg_dt = dt;
	lg_rho_nacc++;

	lagCoef(mod, &c1, &c2, &c3, &c4);

	t = (float)it * dt;
	for (k = 0; k < nfreq; k++) {
		lg_cos[k] = cosf(lg_omega[k] * t) * dt;
		lg_sin[k] = sinf(lg_omega[k] * t) * dt;
	}

	/* ---------------- Vx grid: -rox*Dsig_vx and psi_vx ---------------- *
	 * Stencil, bounds and sign are accumGradient_rho_Dsig's.  Note there
	 * is no 1/dx: the forward kernel is vx -= rox*Dsig with rox carrying
	 * the grid spacing, and the gradient is the transpose of exactly that
	 * discrete update. */
#pragma omp parallel for private(ix, iz, k) schedule(static)
	for (ix = mod->ioXx; ix < mod->ieXx; ix++) {
		for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
			int ig = ix*n1 + iz;
			size_t o0 = (size_t)ig * nfreq * 2;
			float Dsig, fwd, adj;

			Dsig = c1*(fwd_txx[ig]             - fwd_txx[(ix-1)*n1+iz] +
			           fwd_txz[ig+1]           - fwd_txz[ig])          +
			       c2*(fwd_txx[(ix+1)*n1+iz]   - fwd_txx[(ix-2)*n1+iz] +
			           fwd_txz[ig+2]           - fwd_txz[ig-1]);
			if (mod->iorder >= 6)
				Dsig += c3*(fwd_txx[(ix+2)*n1+iz] - fwd_txx[(ix-3)*n1+iz] +
				            fwd_txz[ig+3]         - fwd_txz[ig-2]);
			if (mod->iorder >= 8)
				Dsig += c4*(fwd_txx[(ix+3)*n1+iz] - fwd_txx[(ix-4)*n1+iz] +
				            fwd_txz[ig+4]         - fwd_txz[ig-3]);

			fwd = -mod->rox[ig] * Dsig;
			adj =  wfl_adj->vx[ig];

			for (k = 0; k < nfreq; k++) {
				size_t o = o0 + (size_t)k*2;
				float cr = lg_cos[k], ci = lg_sin[k];
				lg_dsx[o  ] += fwd*cr;  lg_dsx[o+1] += fwd*ci;
				lg_pvx[o  ] += adj*cr;  lg_pvx[o+1] += adj*ci;
			}
		}
	}

	/* ---------------- Vz grid: -roz*Dsig_vz and psi_vz ---------------- */
#pragma omp parallel for private(ix, iz, k) schedule(static)
	for (ix = mod->ioZx; ix < mod->ieZx; ix++) {
		for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
			int ig = ix*n1 + iz;
			size_t o0 = (size_t)ig * nfreq * 2;
			float Dsig, fwd, adj;

			Dsig = c1*(fwd_tzz[ig]             - fwd_tzz[ig-1]          +
			           fwd_txz[(ix+1)*n1+iz]   - fwd_txz[ig])           +
			       c2*(fwd_tzz[ig+1]           - fwd_tzz[ig-2]          +
			           fwd_txz[(ix+2)*n1+iz]   - fwd_txz[(ix-1)*n1+iz]);
			if (mod->iorder >= 6)
				Dsig += c3*(fwd_tzz[ig+2]         - fwd_tzz[ig-3]          +
				            fwd_txz[(ix+3)*n1+iz] - fwd_txz[(ix-2)*n1+iz]);
			if (mod->iorder >= 8)
				Dsig += c4*(fwd_tzz[ig+3]         - fwd_tzz[ig-4]          +
				            fwd_txz[(ix+4)*n1+iz] - fwd_txz[(ix-3)*n1+iz]);

			fwd = -mod->roz[ig] * Dsig;
			adj =  wfl_adj->vz[ig];

			for (k = 0; k < nfreq; k++) {
				size_t o = o0 + (size_t)k*2;
				float cr = lg_cos[k], ci = lg_sin[k];
				lg_dsz[o  ] += fwd*cr;  lg_dsz[o+1] += fwd*ci;
				lg_pvz[o  ] += adj*cr;  lg_pvz[o+1] += adj*ci;
			}
		}
	}
}


/***********************************************************************
 * lagSlice -- Reconstruct g_lam and g_mu at one time lag.
 *
 * For each frequency, X = conj(A^) B^ = (Ar Br + Ai Bi) + i(Ar Bi - Ai Br),
 * and Re{X exp(-i w tau)} = Xr cos(w tau) + Xi sin(w tau).
 *
 * The reconstruction weight follows from
 *   g(tau) = (dw/pi) sum_k Re{...},  dw = 2 pi df  ->  weight = 2 df.
 * With a single frequency there is no spacing, so the weight is set to 1
 * and only the lag SHAPE is meaningful (lagGradientWrite warns).
 *
 * glam/gmuu are padded-grid arrays of lg_sizem, overwritten here.
 ***********************************************************************/
static void lagSlice(modPar *mod, bndPar *bnd, float tau,
                     float *glam, float *gmuu)
{
	int ix, iz, k, n1, nfreq;
	int ibPx, iePx, ibPz, iePz;
	int ibTx, ieTx, ibTz, ieTz;
	float w, *ct, *st;

	n1    = mod->naz;
	nfreq = lg_nfreq;
	w     = (lg_df > 0.0f) ? 2.0f * lg_df : 1.0f;

	lagBounds(mod, bnd, &ibPx, &iePx, &ibPz, &iePz,
	                    &ibTx, &ieTx, &ibTz, &ieTz);

	memset(glam, 0, lg_sizem * sizeof(float));
	memset(gmuu, 0, lg_sizem * sizeof(float));

	/* exp(-2i w tau) per frequency: the symmetric convention shifts the
	 * forward field by -tau and the adjoint by +tau, so the two phases add
	 * (see the header).  The relative lag is 2*tau. */
	ct = (float *)malloc((size_t)nfreq * sizeof(float));
	st = (float *)malloc((size_t)nfreq * sizeof(float));
	if (!ct || !st) verr("lagSlice: cannot allocate phase scratch.");
	for (k = 0; k < nfreq; k++) {
		ct[k] = cosf(2.0f * lg_omega[k] * tau);
		st[k] = sinf(2.0f * lg_omega[k] * tau);
	}

	/* --- P grid: lambda kernel and the normal-stress half of mu --- */
#pragma omp parallel for private(ix, iz, k) schedule(static)
	for (ix = ibPx; ix < iePx; ix++) {
		for (iz = ibPz; iz < iePz; iz++) {
			int ig = ix*n1 + iz;
			size_t o0 = (size_t)ig * nfreq * 2;
			float acc_lam = 0.0f, acc_mu = 0.0f;

			for (k = 0; k < nfreq; k++) {
				size_t o = o0 + (size_t)k*2;
				float exr = lg_exx[o], exi = lg_exx[o+1];
				float ezr = lg_ezz[o], ezi = lg_ezz[o+1];
				float pxr = lg_pxx[o], pxi = lg_pxx[o+1];
				float pzr = lg_pzz[o], pzi = lg_pzz[o+1];

				/* lambda: conj(div^) (psi_txx^ + psi_tzz^) */
				float dr = exr + ezr, di = exi + ezi;
				float sr = pxr + pzr, si = pxi + pzi;
				float Xr = dr*sr + di*si;
				float Xi = dr*si - di*sr;
				acc_lam += Xr*ct[k] + Xi*st[k];

				/* mu, normal part: 2[ conj(exx^) pxx^ + conj(ezz^) pzz^ ] */
				float Xr1 = exr*pxr + exi*pxi;
				float Xi1 = exr*pxi - exi*pxr;
				float Xr2 = ezr*pzr + ezi*pzi;
				float Xi2 = ezr*pzi - ezi*pzr;
				acc_mu += 2.0f*((Xr1 + Xr2)*ct[k] + (Xi1 + Xi2)*st[k]);
			}

			glam[ig] = w * acc_lam;
			gmuu[ig] = w * acc_mu;
		}
	}

	/* --- Txz grid: shear half of mu, scattered to the 4 surrounding P
	 * points with weight 0.25, exactly as accumGradient does.  Kept
	 * serial because neighbouring ix columns write overlapping P points. */
	for (ix = ibTx; ix < ieTx; ix++) {
		for (iz = ibTz; iz < ieTz; iz++) {
			int ig = ix*n1 + iz;
			size_t o0 = (size_t)ig * nfreq * 2;
			float acc = 0.0f, shear;

			for (k = 0; k < nfreq; k++) {
				size_t o = o0 + (size_t)k*2;
				float sr = lg_shr[o], si = lg_shr[o+1];
				float pr = lg_pxz[o], pi = lg_pxz[o+1];
				float Xr = sr*pr + si*pi;
				float Xi = sr*pi - si*pr;
				acc += Xr*ct[k] + Xi*st[k];
			}

			shear = w * acc;
			gmuu[(ix-1)*n1+iz-1] += 0.25f * shear;
			gmuu[ix*n1+iz-1]     += 0.25f * shear;
			gmuu[(ix-1)*n1+iz]   += 0.25f * shear;
			gmuu[ig]             += 0.25f * shear;
		}
	}

	free(ct);
	free(st);
}


/***********************************************************************
 * lagSliceRho -- Reconstruct the DIRECT density kernel at one time lag.
 *
 * Same complex algebra as lagSlice, but on the Vx and Vz grids, and the
 * reconstructed correlation is scattered to the two P points each velocity
 * component straddles with the 0.5/rho weights of accumGradient_rho_Dsig.
 *
 * The weight is 2*df/dt, not 2*df: both factors were accumulated
 * dt-weighted but the density kernel carries no dt of its own (see the
 * file header).  Kept serial, like lagSlice's Txz loop, because
 * neighbouring points scatter into overlapping P points.
 *
 * grho is a padded-grid array of lg_sizem, overwritten here.  This is
 * g_rho_DIRECT; the lambda and mu terms of the chain rule are added in
 * lagBuildCubes.
 ***********************************************************************/
static void lagSliceRho(modPar *mod, float tau, float *grho)
{
	int ix, iz, k, n1, nfreq;
	float w, *ct, *st;

	memset(grho, 0, lg_sizem * sizeof(float));
	if (!lg_rho_on) return;

	n1    = mod->naz;
	nfreq = lg_nfreq;
	w     = (lg_df > 0.0f) ? 2.0f * lg_df : 1.0f;
	if (lg_dt > 0.0f) w /= lg_dt;

	ct = (float *)malloc((size_t)nfreq * sizeof(float));
	st = (float *)malloc((size_t)nfreq * sizeof(float));
	if (!ct || !st) verr("lagSliceRho: cannot allocate phase scratch.");
	for (k = 0; k < nfreq; k++) {
		ct[k] = cosf(2.0f * lg_omega[k] * tau);
		st[k] = sinf(2.0f * lg_omega[k] * tau);
	}

	/* --- Vx grid --- */
	for (ix = mod->ioXx; ix < mod->ieXx; ix++) {
		for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
			int ig = ix*n1 + iz;
			size_t o0 = (size_t)ig * nfreq * 2;
			float acc = 0.0f, contrib;

			for (k = 0; k < nfreq; k++) {
				size_t o = o0 + (size_t)k*2;
				float ar = lg_dsx[o], ai = lg_dsx[o+1];
				float br = lg_pvx[o], bi = lg_pvx[o+1];
				float Xr = ar*br + ai*bi;
				float Xi = ar*bi - ai*br;
				acc += Xr*ct[k] + Xi*st[k];
			}

			contrib = w * acc;
			grho[(ix-1)*n1+iz] += 0.5f*contrib / mod->rho[(ix-1)*n1+iz];
			grho[ig]           += 0.5f*contrib / mod->rho[ig];
		}
	}

	/* --- Vz grid --- */
	for (ix = mod->ioZx; ix < mod->ieZx; ix++) {
		for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
			int ig = ix*n1 + iz;
			size_t o0 = (size_t)ig * nfreq * 2;
			float acc = 0.0f, contrib;

			for (k = 0; k < nfreq; k++) {
				size_t o = o0 + (size_t)k*2;
				float ar = lg_dsz[o], ai = lg_dsz[o+1];
				float br = lg_pvz[o], bi = lg_pvz[o+1];
				float Xr = ar*br + ai*bi;
				float Xi = ar*bi - ai*br;
				acc += Xr*ct[k] + Xi*st[k];
			}

			contrib = w * acc;
			grho[ig-1] += 0.5f*contrib / mod->rho[ig-1];
			grho[ig]   += 0.5f*contrib / mod->rho[ig];
		}
	}

	free(ct);
	free(st);
}


/***********************************************************************
 * lagStrip -- Copy the interior of a padded array into a dense one.
 ***********************************************************************/
static void lagStrip(float *dst, float *src, modPar *mod, bndPar *bnd)
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
 * lagEmit -- Write one pair of lag cubes plus their axis description.
 *
 * `prefix` is the full stem, e.g. "grad_shot002" or "grad"; the files are
 * <prefix>_lagvp.su, <prefix>_lagvs.su, <prefix>_lagrho.su (only when the
 * density accumulators are on) and <prefix>_lagaxis.txt.
 ***********************************************************************/
static void lagEmit(modPar *mod, const char *prefix,
                    float *cube_vp, float *cube_vs, float *cube_rho,
                    int nlag, float dlag, int verbose)
{
	int ilag, mid = (nlag - 1) / 2;
	char fname[1024];
	FILE *fp;

	snprintf(fname, sizeof(fname), "%s_lagvp.su", prefix);
	writesufile(fname, cube_vp, (size_t)mod->nz, (size_t)mod->nx * nlag,
		mod->z0, mod->x0, mod->dz, mod->dx);
	snprintf(fname, sizeof(fname), "%s_lagvs.su", prefix);
	writesufile(fname, cube_vs, (size_t)mod->nz, (size_t)mod->nx * nlag,
		mod->z0, mod->x0, mod->dz, mod->dx);
	if (cube_rho) {
		snprintf(fname, sizeof(fname), "%s_lagrho.su", prefix);
		writesufile(fname, cube_rho, (size_t)mod->nz, (size_t)mod->nx * nlag,
			mod->z0, mod->x0, mod->dz, mod->dx);
	}

	snprintf(fname, sizeof(fname), "%s_lagaxis.txt", prefix);
	fp = fopen(fname, "w");
	if (fp) {
		fprintf(fp, "# time-lag gradient axis description\n");
		fprintf(fp, "nlag  %d\n", nlag);
		fprintf(fp, "dlag  %.9g\n", dlag);
		fprintf(fp, "nx    %d\n", mod->nx);
		fprintf(fp, "nz    %d\n", mod->nz);
		fprintf(fp, "dx    %.9g\n", mod->dx);
		fprintf(fp, "dz    %.9g\n", mod->dz);
		fprintf(fp, "x0    %.9g\n", mod->x0);
		fprintf(fp, "z0    %.9g\n", mod->z0);
		fprintf(fp, "nfreq %d\n", lg_nfreq);
		fprintf(fp, "df    %.9g\n", lg_df);
		fprintf(fp, "# cube order: trace = ilag*nx + ix, nz samples per trace\n");
		fprintf(fp, "# lags (s):\n");
		for (ilag = 0; ilag < nlag; ilag++)
			fprintf(fp, "%.9g\n", (float)(ilag - mid) * dlag);
		fclose(fp);
	}
	else vwarn("lagEmit: cannot write %s", fname);

	if (verbose)
		vmess("lagGradient: wrote %s_lag{vp,vs%s}.su (%d x %d x %d, lag slowest)",
			prefix, cube_rho ? ",rho" : "", nlag, mod->nx, mod->nz);
}


/***********************************************************************
 * lagGradientWrite -- Build and write the Vp and Vs lag cubes.
 *
 *   nlag   number of lags; forced odd so tau = 0 is sampled exactly
 *   dlag   lag increment in seconds (need not be a multiple of mod->dt)
 *   mask   optional [nx*nz] multiplier on the interior grid, ix*nz+iz, or
 *          NULL.  Pass the SAME grad_taper mask that is applied to the
 *          conventional gradient: with grad_taper off, the near-source
 *          singularity holds essentially all the energy and swamps every
 *          display, and the taper cannot be applied downstream because
 *          adj_shot tapers g*_shot, not these accumulators.  Using the same
 *          mask on both also keeps the tau = 0 check against
 *          <base>_vp_raw.su valid, since both sides are then tapered alike.
 *
 * Lags are symmetric: tau_i = (i - (nlag-1)/2) * dlag.
 *
 * Output, two SU files per shot:
 *   <base>_shotNNN_lagvp.su
 *   <base>_shotNNN_lagvs.su
 * each nz samples by (nx * nlag) traces, with the LAG as the SLOWEST
 * axis: trace index = ilag*nx + ix.  Reshape to (nlag, nx, nz).
 *
 * A companion <base>_shotNNN_lagaxis.txt records nlag/dlag/nx/nz and the
 * lag values, so plotting cannot silently disagree with what was run.
 *
 * Each call also adds into the shot stack; see lagGradientWriteStack.
 ***********************************************************************/
/***********************************************************************
 * lagAxisCheck -- Validate the requested lag axis, return nlag forced odd.
 *
 * Shared by every consumer of the accumulators so that the aliasing guard
 * cannot be bypassed by whichever entry point is used.  `who` only names the
 * caller in the messages.
 ***********************************************************************/
static int lagAxisCheck(const char *who, int nlag, float dlag)
{
	float tau_max;

	if (nlag < 1) return 0;
	if (nlag % 2 == 0) {
		nlag += 1;
		vwarn("%s: nlag must be odd so that tau=0 is sampled; using nlag=%d.",
		      who, nlag);
	}
	if (dlag <= 0.0f)
		verr("%s: lag_dt=%g must be > 0 s.", who, dlag);

	tau_max = dlag * (float)((nlag - 1) / 2);

	/* Guard the aliasing period.  The reconstruction repeats every 1/df in
	 * the RELATIVE lag, and the symmetric convention makes the relative lag
	 * 2*tau, so the period on the tau axis is 1/(2 df) and the usable half
	 * range is 1/(4 df). */
	if (lg_df > 0.0f && tau_max >= 0.25f / lg_df)
		verr("%s: requested |tau| up to %.4f s but the "
		     "frequency spacing df=%.4f Hz aliases the lag axis with period "
		     "1/(2 df)=%.4f s (the lag is applied symmetrically, so the "
		     "relative shift is 2*tau).  Wrap-around would be "
		     "indistinguishable from lag moveout.  Either shrink the lag "
		     "range below %.4f s or reduce df (more frequencies over the "
		     "same band).",
		     who, tau_max, lg_df, 0.5f/lg_df, 0.25f/lg_df);
	if (lg_df <= 0.0f)
		vwarn("%s: nfreq=1, so the reconstruction weight is "
		      "arbitrary and only the SHAPE of g(tau) is meaningful.", who);

	return nlag;
}


/***********************************************************************
 * lagBuildCubes -- Reconstruct the (Vp, Vs, rho) lag cubes from the DFTs.
 *
 * cube_vp, cube_vs must each hold nlag * mod->nx * mod->nz floats, ordered
 * (lag, x, z) with z fastest.  nlag must already have passed lagAxisCheck.
 * cube_rho is optional: pass NULL to skip the density reconstruction (it is
 * also skipped, with a warning from the caller, when the density
 * accumulators were never allocated).
 ***********************************************************************/
static void lagBuildCubes(modPar *mod, bndPar *bnd, int nlag, float dlag,
                          float *mask, float *cube_vp, float *cube_vs,
                          float *cube_rho, int verbose)
{
	int ilag, i, nmodel, mid;
	float *glam, *gmuu, *grho, *slice;

	nmodel = mod->nx * mod->nz;
	mid    = (nlag - 1) / 2;
	if (!lg_rho_on) cube_rho = NULL;

	glam  = (float *)malloc(lg_sizem * sizeof(float));
	gmuu  = (float *)malloc(lg_sizem * sizeof(float));
	slice = (float *)malloc(lg_sizem * sizeof(float));
	grho  = cube_rho ? (float *)malloc(lg_sizem * sizeof(float)) : NULL;
	if (!glam || !gmuu || !slice || (cube_rho && !grho))
		verr("lagBuildCubes: cannot allocate slice scratch.");

	for (ilag = 0; ilag < nlag; ilag++) {
		float tau = (float)(ilag - mid) * dlag;
		float *pvp = cube_vp + (size_t)ilag * nmodel;
		float *pvs = cube_vs + (size_t)ilag * nmodel;
		float *pro = cube_rho ? cube_rho + (size_t)ilag * nmodel : NULL;
		double n_vp = 0.0, n_vs = 0.0, n_ro = 0.0;

		lagSlice(mod, bnd, tau, glam, gmuu);
		if (pro) lagSliceRho(mod, tau, grho);

		/* Chain rule Lame -> (Vp, Vs, rho), applied at EVERY lag with exactly
		 * the same pointwise factors convertGradientToVelocity uses on the
		 * conventional gradient:
		 *     g_Vp  = 2 rho Vp  g_lam
		 *     g_Vs  = 2 rho Vs (g_mu - 2 g_lam)
		 *     g_rho = g_rho_direct + g_lam (Vp^2 - 2 Vs^2) + g_mu Vs^2
		 * Because the map is pointwise and lag-independent, it commutes with
		 * the lag reconstruction -- which is what keeps the tau=0 check
		 * against <base>_vp_raw.su exact.  Note the density cube needs all
		 * three kernels, so it is the one that would silently go wrong if
		 * the lam/mu and rho accumulators ever drifted apart in lag
		 * reference; see "THE TIME STAMP" in the file header. */
		for (i = 0; i < (int)lg_sizem; i++)
			slice[i] = 2.0f * mod->rho[i] * mod->cp[i] * glam[i];
		lagStrip(pvp, slice, mod, bnd);

		for (i = 0; i < (int)lg_sizem; i++)
			slice[i] = 2.0f * mod->rho[i] * mod->cs[i]
			           * (gmuu[i] - 2.0f * glam[i]);
		lagStrip(pvs, slice, mod, bnd);

		if (pro) {
			for (i = 0; i < (int)lg_sizem; i++) {
				float vp2 = mod->cp[i] * mod->cp[i];
				float vs2 = mod->cs[i] * mod->cs[i];
				slice[i] = grho[i] + glam[i]*(vp2 - 2.0f*vs2) + gmuu[i]*vs2;
			}
			lagStrip(pro, slice, mod, bnd);
		}

		/* Same taper as the conventional gradient, on every lag. */
		if (mask != NULL) {
			for (i = 0; i < nmodel; i++) { pvp[i] *= mask[i]; pvs[i] *= mask[i]; }
			if (pro) for (i = 0; i < nmodel; i++) pro[i] *= mask[i];
		}

		if (verbose > 1) {
			for (i = 0; i < nmodel; i++) {
				n_vp += (double)pvp[i]*pvp[i];
				n_vs += (double)pvs[i]*pvs[i];
				if (pro) n_ro += (double)pro[i]*pro[i];
			}
			if (pro)
				vmess("  tau=%+8.5f s   ||g_vp||=%.6e   ||g_vs||=%.6e   "
					"||g_rho||=%.6e", tau, sqrt(n_vp), sqrt(n_vs), sqrt(n_ro));
			else
				vmess("  tau=%+8.5f s   ||g_vp||=%.6e   ||g_vs||=%.6e",
					tau, sqrt(n_vp), sqrt(n_vs));
		}
	}

	free(glam); free(gmuu); free(slice);
	if (grho) free(grho);
}


void lagGradientWrite(modPar *mod, bndPar *bnd, const char *base, int ishot,
                      int nlag, float dlag, float *mask, int verbose)
{
	int i, nmodel;
	float *cube_vp, *cube_vs, *cube_rho;
	char prefix[1024];

	if (!lg_on) return;

	nlag = lagAxisCheck("lagGradientWrite", nlag, dlag);
	if (nlag == 0) {
		vwarn("lagGradientWrite: nlag=0, nothing to do.");
		return;
	}

	/* The density accumulators exist but nothing ever fed them: adj_shot only
	 * calls lagGradientAccumRho where it has the forward stress, which it
	 * buffers only when a density gradient or the pseudo-Hessian was asked
	 * for.  Writing a silently-zero rho cube would be worse than saying so. */
	if (lg_rho_on && lg_rho_nacc == 0)
		vwarn("lagGradientWrite: shot %d, the density accumulators were "
		      "enabled but no time step was folded into them -- adj_shot "
		      "buffers the forward stress only when it is given a grad_rho "
		      "array.  <%s_shot%03d_lagrho.su> will be zero.",
		      ishot, base, ishot);

	nmodel   = mod->nx * mod->nz;
	cube_vp  = (float *)malloc((size_t)nmodel * nlag * sizeof(float));
	cube_vs  = (float *)malloc((size_t)nmodel * nlag * sizeof(float));
	cube_rho = lg_rho_on
	         ? (float *)malloc((size_t)nmodel * nlag * sizeof(float)) : NULL;
	if (!cube_vp || !cube_vs || (lg_rho_on && !cube_rho))
		verr("lagGradientWrite: cannot allocate lag cubes (%d lags x %d points).",
		     nlag, nmodel);

	if (verbose)
		vmess("lagGradientWrite: shot %d, %d lags, dtau=%.5f s, "
			"|tau|<=%.5f s", ishot, nlag, dlag,
			dlag * (float)((nlag - 1) / 2));

	lagBuildCubes(mod, bnd, nlag, dlag, mask, cube_vp, cube_vs, cube_rho,
		verbose);

	snprintf(prefix, sizeof(prefix), "%s_shot%03d", base, ishot);
	lagEmit(mod, prefix, cube_vp, cube_vs, cube_rho, nlag, dlag, verbose);

	/* ---- add into the shot stack ---- */
	if (lg_stk_vp == NULL) {
		lg_stk_vp = (float *)calloc((size_t)nmodel * nlag, sizeof(float));
		lg_stk_vs = (float *)calloc((size_t)nmodel * nlag, sizeof(float));
		if (cube_rho)
			lg_stk_rho = (float *)calloc((size_t)nmodel * nlag, sizeof(float));
		if (!lg_stk_vp || !lg_stk_vs || (cube_rho && !lg_stk_rho))
			verr("lagGradientWrite: cannot allocate the shot stack.");
		lg_stk_nlag = nlag; lg_stk_dlag = dlag; lg_stk_nmodel = nmodel;
	}
	else if (lg_stk_nlag != nlag || lg_stk_dlag != dlag ||
	         lg_stk_nmodel != nmodel) {
		verr("lagGradientWrite: shot %d has a lag axis (%d x %g s, %d points) "
		     "that differs from the stack (%d x %g s, %d points).  All shots "
		     "must share one lag axis to be summed.",
		     ishot, nlag, dlag, nmodel,
		     lg_stk_nlag, lg_stk_dlag, lg_stk_nmodel);
	}
	for (i = 0; i < nmodel * nlag; i++) {
		lg_stk_vp[i] += cube_vp[i];
		lg_stk_vs[i] += cube_vs[i];
	}
	if (lg_stk_rho && cube_rho)
		for (i = 0; i < nmodel * nlag; i++) lg_stk_rho[i] += cube_rho[i];
	lg_stk_nshot++;

	free(cube_vp); free(cube_vs);
	if (cube_rho) free(cube_rho);
}


/***********************************************************************
 * lagGradientTomo -- The shot's gradient, angle-filtered in the lag domain.
 *
 * Builds this shot's (Vp, Vs) lag cubes, applies the cone filter, and
 * returns the tau = 0 slice of the PASSED cube as the shot's gradient
 * contribution.  With `p->s_pass` small this keeps the wide-angle
 * transmission (tomographic) kernel and discards the specular reflection
 * imprint; swap which half is returned by raising s_pass above the grid's
 * largest slope and subtracting, or see lagConeFilter's `tapered` output.
 *
 *   gvp, gvs   [mod->nx * mod->nz] OUTPUT, ix*nz+iz, already in (Vp, Vs).
 *              ADDED INTO, not overwritten, so the caller can accumulate
 *              over shots.  Zero them before the shot loop.
 *
 * WHY tau = 0.  The gradient is by definition the zero-lag correlation; the
 * lag axis is only the vehicle that lets the cone filter see the local
 * slope.  So the filter is applied in 3D and then evaluated at tau = 0.
 *
 * WHY (Vp, Vs) AND NOT LAME.  The cone filter is a spatial convolution and
 * the Lame->velocity chain rule is a multiplication by a spatially VARYING
 * factor, so the two do not commute: filtering lambda/mu cubes and
 * converting afterwards is a different operator from filtering the converted
 * cubes.  The validated reference filters the converted cubes, so that is
 * what happens here -- and it means the caller must NOT convert again.
 * Running this result through extractGradientVector with param=2 would be
 * the double-chain-rule bug.
 *
 * There is no rho output: this path asks lagBuildCubes for the (Vp, Vs)
 * cubes only, and its caller should pass do_rho=0 to lagGradientInit so the
 * four density accumulators are never allocated.  Freeze rho when using it.
 *
 * WHY lagConePspiTau0 AND NOT lagConeFilter.  Two reasons, and the second is
 * the one that matters physically:
 *
 *  - COST.  Only the tau = 0 sample is ever used, so lagConeFilter's inverse
 *    3D transform throws ~99.7% of its own output away.  The projection-slice
 *    form takes the sum over nu BEFORE the spatial inverse, which makes the
 *    inverse 2D.  Even with p->nv reference velocities it is cheaper than the
 *    single 3D inverse it replaces.
 *
 *  - CORRECTNESS OFF THE REFERENCE VELOCITY.  A slope is cos(theta)/v, so a
 *    cone cut at one fixed slope enforces a DIFFERENT opening angle wherever
 *    the model is not the reference velocity.  With p->nv > 1 the cutoff
 *    follows v(x,z): the Vp cube is filtered against mod->cp and the Vs cube
 *    against mod->cs, each stripped to the interior.  p->nv <= 1 reproduces
 *    lagConeFilter's tau = 0 slice exactly (lagConePspiSelfTest, 3.7e-07).
 ***********************************************************************/
void lagGradientTomo(modPar *mod, bndPar *bnd, int nlag, float dlag,
                     float *mask, const lagConePar *p,
                     float *gvp, float *gvs, int verbose)
{
	int i, nmodel;
	float *cube_vp, *cube_vs, *tau0, *vp, *vs;
	lagConeStats st;

	if (!lg_on) return;

	nlag = lagAxisCheck("lagGradientTomo", nlag, dlag);
	if (nlag < 3)
		verr("lagGradientTomo: the cone filter needs a lag AXIS to measure a "
		     "slope against; nlag=%d is not enough (use >= 3, in practice "
		     "15-31).", nlag);

	nmodel = mod->nx * mod->nz;

	cube_vp = (float *)malloc((size_t)nmodel * nlag * sizeof(float));
	cube_vs = (float *)malloc((size_t)nmodel * nlag * sizeof(float));
	tau0    = (float *)malloc((size_t)nmodel * sizeof(float));
	if (!cube_vp || !cube_vs || !tau0)
		verr("lagGradientTomo: cannot allocate 2 cubes of %d lags x %d points "
		     "(%.2f GB total).", nlag, nmodel,
		     2.0 * (double)nmodel * nlag * sizeof(float) / 1073741824.0);

	/* The cone cuts on the MODE velocity: the Vp cube's slopes are
	 * cos(theta)/Vp, the Vs cube's are cos(theta)/Vs.  Sharing one velocity
	 * between them would mis-cut both by the Vp/Vs ratio. */
	vp = vs = NULL;
	if (p->nv > 1) {
		vp = (float *)malloc((size_t)nmodel * sizeof(float));
		vs = (float *)malloc((size_t)nmodel * sizeof(float));
		if (!vp || !vs)
			verr("lagGradientTomo: cannot allocate the interior velocities.");
		lagStrip(vp, mod->cp, mod, bnd);
		lagStrip(vs, mod->cs, mod, bnd);
	}

	lagBuildCubes(mod, bnd, nlag, dlag, mask, cube_vp, cube_vs, NULL, verbose);

	/* Vp */
	lagConePspiTau0(cube_vp, nlag, mod->nx, mod->nz, mod->dx, mod->dz, dlag,
	                p, vp, tau0, &st, verbose);
	for (i = 0; i < nmodel; i++)
		gvp[i] += tau0[i];
	if (verbose)
		vmess("lagGradientTomo: g_Vp kept %.2f%% of the spectral energy",
		      100.0 * st.energy_frac_passed);

	/* Vs */
	lagConePspiTau0(cube_vs, nlag, mod->nx, mod->nz, mod->dx, mod->dz, dlag,
	                p, vs, tau0, &st, verbose);
	for (i = 0; i < nmodel; i++)
		gvs[i] += tau0[i];
	if (verbose)
		vmess("lagGradientTomo: g_Vs kept %.2f%% of the spectral energy",
		      100.0 * st.energy_frac_passed);

	free(cube_vp); free(cube_vs); free(tau0);
	free(vp); free(vs);
}


/***********************************************************************
 * lagGradientWriteStack -- Write the sum of all shots written so far.
 *
 * The FWI gradient is a sum over shots, so this -- not any single shot --
 * is the cube whose tau = 0 slice must reproduce <base>_vp_raw.su and
 * <base>_vs_raw.su.
 *
 * Output: <base>_lagvp.su, <base>_lagvs.su, <base>_lagrho.su (when the
 * density accumulators are on) and <base>_lagaxis.txt.
 * Call once after the shot loop, before lagGradientFree.
 ***********************************************************************/
void lagGradientWriteStack(modPar *mod, const char *base, int verbose)
{
	if (!lg_on || lg_stk_vp == NULL) return;

	if (verbose)
		vmess("lagGradientWriteStack: stacking %d shot%s onto one lag cube",
			lg_stk_nshot, lg_stk_nshot == 1 ? "" : "s");

	lagEmit(mod, base, lg_stk_vp, lg_stk_vs, lg_stk_rho,
		lg_stk_nlag, lg_stk_dlag, verbose);
}


/***********************************************************************
 * lagFreeAccum -- Release the per-shot DFT fields, keep the shot stack.
 ***********************************************************************/
static void lagFreeAccum(void)
{
	if (lg_exx)   { free(lg_exx);   lg_exx   = NULL; }
	if (lg_ezz)   { free(lg_ezz);   lg_ezz   = NULL; }
	if (lg_shr)   { free(lg_shr);   lg_shr   = NULL; }
	if (lg_pxx)   { free(lg_pxx);   lg_pxx   = NULL; }
	if (lg_pzz)   { free(lg_pzz);   lg_pzz   = NULL; }
	if (lg_pxz)   { free(lg_pxz);   lg_pxz   = NULL; }
	if (lg_dsx)   { free(lg_dsx);   lg_dsx   = NULL; }
	if (lg_pvx)   { free(lg_pvx);   lg_pvx   = NULL; }
	if (lg_dsz)   { free(lg_dsz);   lg_dsz   = NULL; }
	if (lg_pvz)   { free(lg_pvz);   lg_pvz   = NULL; }
	if (lg_omega) { free(lg_omega); lg_omega = NULL; }
	if (lg_cos)   { free(lg_cos);   lg_cos   = NULL; }
	if (lg_sin)   { free(lg_sin);   lg_sin   = NULL; }
	lg_on       = 0;
	lg_nfreq    = 0;
	lg_sizem    = 0;
	lg_df       = 0.0f;
	lg_dt       = 0.0f;
	lg_rho_on   = 0;
	lg_rho_nacc = 0;
}


/***********************************************************************
 * lagGradientFree -- Release everything, including the shot stack.
 *
 * Call once after the shot loop.  Calling it between shots would discard
 * the stack; lagGradientInit already clears the per-shot accumulators.
 ***********************************************************************/
void lagGradientFree(void)
{
	lagFreeAccum();
	if (lg_stk_vp)  { free(lg_stk_vp);  lg_stk_vp  = NULL; }
	if (lg_stk_vs)  { free(lg_stk_vs);  lg_stk_vs  = NULL; }
	if (lg_stk_rho) { free(lg_stk_rho); lg_stk_rho = NULL; }
	lg_stk_nlag = lg_stk_nmodel = lg_stk_nshot = 0;
	lg_stk_dlag = 0.0f;
}
