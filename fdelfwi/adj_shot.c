#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"

/**
 * adj_shot.c  --  Adjoint backpropagation with re-propagation checkpointing.
 *
 * For each segment between consecutive checkpoints:
 *   1. Load the forward checkpoint and re-propagate forward (with source),
 *      storing vx/vz at every intermediate time step in memory.
 *   2. Propagate the adjoint wavefield backward through the segment,
 *      cross-correlating with the stored forward states at each step
 *      to accumulate the exact gradient.
 *
 * AUTHOR:
 *   Based on backpropagation.md design and fdacrtmc RTM architecture.
 */

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

double wallclock_time(void);

/* Forward FD kernels (for re-propagation) */
int acoustic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose);
int acoustic6(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *p, float *rox, float *roz, float *l2m, int verbose);
int elastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int verbose);
int elastic6(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int verbose);

/* Forward FD kernel (8th order) */
int elastic8(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int verbose);

/* Adjoint FD kernels (true discrete adjoint with material params inside derivatives) */
int elastic4_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int rec_delay, int rec_skipdt, int verbose);
int elastic6_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int rec_delay, int rec_skipdt, int verbose);
int elastic8_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int rec_delay, int rec_skipdt, int verbose);

/* Checkpoint I/O (checkpoint.c) */
int readCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl);

/* Snapshot output (writeSnapTimes.c) */
int writeSnapTimes(modPar mod, snaPar sna, bndPar bnd, wavPar wav,
                   int ixsrc, int izsrc, int itime,
                   float *vx, float *vz, float *tzz, float *txx, float *txz,
                   int verbose);
void writeSnapTimesReset(void);

/* Gradient cross-correlation and parameterization (fwi_gradient.c) */
void accumGradient(modPar *mod, bndPar *bnd,
                   float *fwd_vx, float *fwd_vz,
                   float *fwd_vx_prev, float *fwd_vz_prev,
                   wflPar *wfl_adj,
                   float dt,
                   float *grad_lam, float *grad_muu, float *grad_rho);
void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho,
                               size_t sizem);
void accumGradient_rho_Dsig(modPar *mod, bndPar *bnd,
                            float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                            wflPar *wfl_adj,
                            float dt,
                            float *grad_rho);


/***********************************************************************
 * callKernel -- Dispatch to the appropriate FD kernel.
 ***********************************************************************/
static void callKernel(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                       int it, int ixsrc, int izsrc, float **src_nwav,
                       wflPar *wfl, int verbose)
{
	switch (mod->ischeme) {
		case 1:
			if (mod->iorder == 4)
				acoustic4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, mod->rox, mod->roz, mod->l2m, verbose);
			else if (mod->iorder == 6)
				acoustic6(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, mod->rox, mod->roz, mod->l2m, verbose);
			break;
		case 3:
		case 5:
			if (mod->iorder == 4)
				elastic4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			else if (mod->iorder == 6)
				elastic6(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			else if (mod->iorder == 8)
				elastic8(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			break;
	}
}


/***********************************************************************
 * callAdjKernel -- Dispatch to the appropriate adjoint FD kernel.
 *
 * Uses dedicated adjoint kernels that inject multicomponent residuals
 * at the physically correct points in the staggered time step:
 *   - Force residuals (Fx, Fz) after velocity update
 *   - Stress residuals (P, Txx, Tzz, Txz) after stress update
 ***********************************************************************/
static void callAdjKernel(modPar *mod, adjSrcPar *adj, bndPar *bnd,
                          int it, wflPar *wfl,
                          int rec_delay, int rec_skipdt, int verbose)
{
	switch (mod->ischeme) {
		case 1:
			/* TODO: acoustic4_adj / acoustic6_adj */
			break;
		case 3:
		case 5:
			if (mod->iorder == 4)
				elastic4_adj(*mod, *adj, *bnd, it,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu,
					rec_delay, rec_skipdt, verbose);
			else if (mod->iorder == 6)
				elastic6_adj(*mod, *adj, *bnd, it,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu,
					rec_delay, rec_skipdt, verbose);
			else if (mod->iorder == 8)
				elastic8_adj(*mod, *adj, *bnd, it,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu,
					rec_delay, rec_skipdt, verbose);
			break;
	}
}


/***********************************************************************
 * adj_shot -- Adjoint backpropagation with re-propagation for one shot.
 *
 * Computes the FWI gradient contribution by processing checkpoint
 * segments in reverse order.  For each segment the forward wavefield
 * is reconstructed from the checkpoint and cross-correlated with the
 * backward-propagated adjoint wavefield at every time step.
 *
 * Parameters:
 *   mod, bnd              - model and boundary parameters
 *   src, wav, src_nwav,   - forward source (needed for re-propagation)
 *     ixsrc, izsrc
 *   rec                   - receiver timing (delay, skipdt)
 *   adj                   - adjoint source traces (residuals)
 *   chk                   - checkpoint metadata
 *   sna                   - snapshot parameters (NULL to disable)
 *   grad1/grad2/grad3     - OUTPUT gradient arrays (pre-allocated, may
 *                           already contain accumulated values from
 *                           previous shots; pass NULL to skip)
 *   param                 - Parameterization choice:
 *                           1 = Lamé (λ, μ, ρ_direct)
 *                           2 = Velocity (Vp, Vs, ρ_full with chain rule)
 *   verbose               - verbosity level
 *
 * Output interpretation based on param:
 *   param=1: grad1=g_λ, grad2=g_μ, grad3=g_ρ(direct)
 *   param=2: grad1=g_Vp, grad2=g_Vs, grad3=g_ρ(full)
 *
 * Chain rule for velocity parameterization (param=2):
 *   λ = ρ(Vp² - 2Vs²),  μ = ρVs²
 *   g_Vp = g_λ × 2ρVp
 *   g_Vs = 2ρVs(g_μ - 2g_λ)
 *   g_ρ  = g_ρ(direct) + g_λ(Vp² - 2Vs²) + g_μ(Vs²)
 ***********************************************************************/
int adj_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
             recPar *rec, adjSrcPar *adj,
             int ixsrc, int izsrc, float **src_nwav,
             checkpointPar *chk, snaPar *sna,
             float *grad1, float *grad2, float *grad3,
             int param, int verbose)
{
	wflPar wfl_fwd, wfl_adj;
	float *buf_vx, *buf_vz;
	float *buf_txx, *buf_tzz, *buf_txz;     /* Forward stress buffer for D_σ density gradient */
	float *grad_lam, *grad_muu, *grad_rho;  /* Internal Lamé gradient pointers */
	int k, j, it, seg_start, seg_end, nsteps, max_nsteps;
	size_t sizem;
	int it1;
	float dt;
	double t0_wall;

	int adj_snap_count;

	it1   = mod->nt + NINT(-mod->t0 / mod->dt);
	dt    = mod->dt;
	sizem = (size_t)mod->nax * mod->naz;

	/* ------------------------------------------------------------ */
	/* Setup: Always accumulate in Lamé space, convert at end       */
	/* ------------------------------------------------------------ */
	if (param == 1) {
		/* Lamé parameterization: accumulate directly into output */
		grad_lam = grad1;
		grad_muu = grad2;
		grad_rho = grad3;
	}
	else if (param == 2) {
		/* Velocity parameterization: accumulate into output arrays
		 * (treated as Lamé during accumulation), convert at end.
		 * NOTE: For multi-shot accumulation with param=2, the caller
		 * should use param=1, accumulate Lamé, then convert once at
		 * the end using convertGradientToVelocity(). */
		grad_lam = grad1;
		grad_muu = grad2;
		grad_rho = grad3;
	}
	else {
		verr("adj_shot: Invalid param=%d (must be 1=Lamé or 2=velocity)", param);
		return -1;
	}

	/* ------------------------------------------------------------ */
	/* 1. Allocate forward re-propagation wavefield                 */
	/* ------------------------------------------------------------ */
	memset(&wfl_fwd, 0, sizeof(wflPar));
	wfl_fwd.vx  = (float *)calloc(sizem, sizeof(float));
	wfl_fwd.vz  = (float *)calloc(sizem, sizeof(float));
	wfl_fwd.tzz = (float *)calloc(sizem, sizeof(float));
	if (mod->ischeme > 2) {
		wfl_fwd.txz = (float *)calloc(sizem, sizeof(float));
		wfl_fwd.txx = (float *)calloc(sizem, sizeof(float));
	}

	/* ------------------------------------------------------------ */
	/* 2. Allocate adjoint wavefield (zero initial condition)        */
	/* ------------------------------------------------------------ */
	memset(&wfl_adj, 0, sizeof(wflPar));
	wfl_adj.vx  = (float *)calloc(sizem, sizeof(float));
	wfl_adj.vz  = (float *)calloc(sizem, sizeof(float));
	wfl_adj.tzz = (float *)calloc(sizem, sizeof(float));
	if (mod->ischeme > 2) {
		wfl_adj.txz = (float *)calloc(sizem, sizeof(float));
		wfl_adj.txx = (float *)calloc(sizem, sizeof(float));
	}

	/* ------------------------------------------------------------ */
	/* 3. Allocate forward vx/vz buffer for one segment             */
	/* ------------------------------------------------------------ */
	max_nsteps = chk->skipdt;
	{
		int last_start = chk->delay + (chk->nsnap - 1) * chk->skipdt;
		int last_n = it1 - last_start;
		if (last_n > max_nsteps) max_nsteps = last_n;
	}

	buf_vx = (float *)malloc((size_t)max_nsteps * sizem * sizeof(float));
	buf_vz = (float *)malloc((size_t)max_nsteps * sizem * sizeof(float));
	if (!buf_vx || !buf_vz)
		verr("adj_shot: Cannot allocate forward buffer (%d steps x %zu)", max_nsteps, sizem);

	/* Stress buffer for exact D_σ density gradient (eliminates source
	 * injection contamination in the dv/dt approximation).
	 * Stores σ at each forward time step so that during the backward
	 * pass we can compute D_σ(σ^{j-1}) directly. */
	buf_txx = buf_tzz = buf_txz = NULL;
	if (mod->ischeme > 2 && grad3) {
		buf_txx = (float *)malloc((size_t)max_nsteps * sizem * sizeof(float));
		buf_tzz = (float *)malloc((size_t)max_nsteps * sizem * sizeof(float));
		buf_txz = (float *)malloc((size_t)max_nsteps * sizem * sizeof(float));
		if (!buf_txx || !buf_tzz || !buf_txz)
			verr("adj_shot: Cannot allocate stress buffer (%d steps x %zu)", max_nsteps, sizem);
	}

	/* ------------------------------------------------------------ */
	/* Pre-pass: compute end-of-segment forward stress for j=0      */
	/* density gradient at segment boundaries.                       */
	/*                                                               */
	/* born_shot injects density virtual source at j=0 of segment k */
	/* using forward stress from the END of segment k-1 (saved_txx).*/
	/* The adjoint (accumGradient_rho_Dsig) must use the same stress */
	/* at j=0.  Since we process segments backward (k=nsnap-1 to 0),*/
	/* the end-of-segment-(k-1) stress is not available when we      */
	/* reach j=0 of segment k.  So we pre-compute it here.          */
	/*                                                               */
	/* seg_end_txx[k] = forward stress at END of segment k           */
	/* Used for j=0 of segment k+1.  k=0,j=0 uses zero (IC).       */
	/* ------------------------------------------------------------ */
	float *seg_end_txx = NULL, *seg_end_tzz = NULL, *seg_end_txz = NULL;
	if (buf_txx && chk->nsnap > 1) {
		size_t seg_buf_size = (size_t)(chk->nsnap - 1) * sizem;
		seg_end_txx = (float *)calloc(seg_buf_size, sizeof(float));
		seg_end_tzz = (float *)calloc(seg_buf_size, sizeof(float));
		seg_end_txz = (float *)calloc(seg_buf_size, sizeof(float));
		if (!seg_end_txx || !seg_end_tzz || !seg_end_txz)
			verr("adj_shot: Cannot allocate seg_end stress buffer");

		for (k = 0; k < chk->nsnap - 1; k++) {
			seg_start = chk->delay + k * chk->skipdt;
			seg_end   = chk->delay + (k + 1) * chk->skipdt;
			nsteps    = seg_end - seg_start;

			readCheckpoint(chk, k, &wfl_fwd);
			for (j = 1; j < nsteps; j++) {
				it = seg_start + j;
#pragma omp parallel default(shared)
{
				callKernel(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav, &wfl_fwd, verbose);
}
			}
			memcpy(seg_end_txx + (size_t)k * sizem, wfl_fwd.txx, sizem * sizeof(float));
			memcpy(seg_end_tzz + (size_t)k * sizem, wfl_fwd.tzz, sizem * sizeof(float));
			memcpy(seg_end_txz + (size_t)k * sizem, wfl_fwd.txz, sizem * sizeof(float));
		}

		if (verbose)
			vmess("adj_shot: Pre-pass computed %d end-of-segment stresses for density j=0",
				chk->nsnap - 1);
	}

	if (verbose) {
		float buf_mb = (buf_txx ? 5.0f : 2.0f) * max_nsteps * sizem * sizeof(float) / (1024.0*1024.0);
		if (seg_end_txx)
			buf_mb += 3.0f * (chk->nsnap - 1) * sizem * sizeof(float) / (1024.0*1024.0);
		vmess("adj_shot: %d segments, max_nsteps=%d, buffer=%.1f MB",
			chk->nsnap, max_nsteps, buf_mb);
		t0_wall = wallclock_time();
	}

	/* ------------------------------------------------------------ */
	/* 4. Negate residual: ψ is driven by −residual (Lagrangian    */
	/*    convention ψ = −∂J/∂u).  applyAdjointSource uses +=,     */
	/*    so we negate the data here to get −residual injection.    */
	/* ------------------------------------------------------------ */
	{
		size_t i, ntotal = (size_t)adj->nsrc * adj->nt;
		for (i = 0; i < ntotal; i++)
			adj->wav[i] = -adj->wav[i];
	}

	/* ------------------------------------------------------------ */
	/* 5. Process segments in reverse order                         */
	/* ------------------------------------------------------------ */

	/* Reset snapshot file state so adjoint snapshots get their own files */
	adj_snap_count = 0;
	if (sna && sna->nsnap > 0)
		writeSnapTimesReset();

	for (k = chk->nsnap - 1; k >= 0; k--) {

		seg_start = chk->delay + k * chk->skipdt;
		seg_end   = (k == chk->nsnap - 1) ? it1 : chk->delay + (k + 1) * chk->skipdt;
		nsteps    = seg_end - seg_start;

		if (verbose > 1)
			vmess("adj_shot: Segment %d/%d  it=[%d,%d)  nsteps=%d",
				k, chk->nsnap, seg_start, seg_end, nsteps);

		/* ---- 4a. Forward re-propagation ---- */
		readCheckpoint(chk, k, &wfl_fwd);

		/* Store checkpoint state as buf[0] */
		memcpy(buf_vx, wfl_fwd.vx, sizem * sizeof(float));
		memcpy(buf_vz, wfl_fwd.vz, sizem * sizeof(float));
		if (buf_txx) {
			memcpy(buf_txx, wfl_fwd.txx, sizem * sizeof(float));
			memcpy(buf_tzz, wfl_fwd.tzz, sizem * sizeof(float));
			memcpy(buf_txz, wfl_fwd.txz, sizem * sizeof(float));
		}

		/* Re-propagate forward through rest of segment */
		for (j = 1; j < nsteps; j++) {
			it = seg_start + j;
#pragma omp parallel default(shared)
{
			callKernel(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav, &wfl_fwd, verbose);
}
			memcpy(buf_vx + (size_t)j * sizem, wfl_fwd.vx, sizem * sizeof(float));
			memcpy(buf_vz + (size_t)j * sizem, wfl_fwd.vz, sizem * sizeof(float));
			if (buf_txx) {
				memcpy(buf_txx + (size_t)j * sizem, wfl_fwd.txx, sizem * sizeof(float));
				memcpy(buf_tzz + (size_t)j * sizem, wfl_fwd.tzz, sizem * sizeof(float));
				memcpy(buf_txz + (size_t)j * sizem, wfl_fwd.txz, sizem * sizeof(float));
			}
		}

		/* ---- 4b. Adjoint backward through segment ---- */
		for (j = nsteps - 1; j >= 0; j--) {
			it = seg_start + j;

			/* Lambda/mu gradient: needs ψ_σ BEFORE callAdjKernel.
			 * This is ψ^{n+1}_σ (adjoint stress at the output of
			 * forward step n), which is the correct Lagrangian
			 * multiplier for the F2 (stress update) constraint.
			 * Pass NULL for density (accumulated after adjoint step). */
			accumGradient(mod, bnd,
				buf_vx + (size_t)j * sizem,
				buf_vz + (size_t)j * sizem,
				NULL, NULL,  /* no density here */
				&wfl_adj, dt,
				grad_lam, grad_muu, NULL);

			/* Advance adjoint wavefield with multicomponent source injection.
			 * The adjoint kernel injects force residuals (Fx,Fz) at the
			 * velocity-update point and stress residuals (P,Txx,Tzz,Txz)
			 * at the stress-update point — physically correct timing. */
#pragma omp parallel default(shared)
{
			callAdjKernel(mod, adj, bnd, it, &wfl_adj,
				rec->delay, rec->skipdt, verbose);
}

			/* Density gradient: needs ψ_v AFTER callAdjKernel.
			 * Phase A2 (stress backward) inside callAdjKernel does NOT
			 * modify ψ_v, so ψ_v after callAdjKernel = ψ_v_mid
			 * (after Phase A1 + velocity source injection), which is
			 * the correct multiplier for the F1 (velocity update).
			 *
			 * Uses D_σ(stress) directly instead of dv/dt to avoid
			 * source injection contamination at the shot point.
			 * For step j, the density virtual source uses σ^{j-1}
			 * (stress BEFORE step seg_start+j = buf_stress[j-1]).
			 * For j=0 at k>0: uses pre-computed end-of-segment-(k-1)
			 * stress from the pre-pass.
			 * For j=0 at k=0: zero IC → zero contribution, skip. */
			if (buf_txx && j > 0) {
				accumGradient_rho_Dsig(mod, bnd,
					buf_txx + (size_t)(j-1) * sizem,
					buf_tzz + (size_t)(j-1) * sizem,
					buf_txz + (size_t)(j-1) * sizem,
					&wfl_adj, dt,
					grad_rho);
			}
			else if (seg_end_txx && j == 0 && k > 0) {
				/* At j=0 of segment k>0: use end-of-segment-(k-1) stress.
				 * Matches born_shot's saved_txx at segment boundaries. */
				accumGradient_rho_Dsig(mod, bnd,
					seg_end_txx + (size_t)(k-1) * sizem,
					seg_end_tzz + (size_t)(k-1) * sizem,
					seg_end_txz + (size_t)(k-1) * sizem,
					&wfl_adj, dt,
					grad_rho);
			}
			else if (!buf_txx && grad_rho) {
				/* Fallback to dv/dt for acoustic (no stress buffers) */
				float *vx_prev = (j > 0) ? buf_vx + (size_t)(j-1) * sizem : NULL;
				float *vz_prev = (j > 0) ? buf_vz + (size_t)(j-1) * sizem : NULL;

				accumGradient(mod, bnd,
					buf_vx + (size_t)j * sizem,
					buf_vz + (size_t)j * sizem,
					vx_prev, vz_prev,
					&wfl_adj, dt,
					NULL, NULL, grad_rho);
			}

			/* Write adjoint wavefield snapshot if requested. */
			if (sna && sna->nsnap > 0 &&
			    (it >= sna->delay) &&
			    (it <= sna->delay + (sna->nsnap - 1) * sna->skipdt) &&
			    ((it - sna->delay) % sna->skipdt == 0)) {
				int fake_it = sna->delay + adj_snap_count * sna->skipdt;
				writeSnapTimes(*mod, *sna, *bnd, *wav, ixsrc, izsrc, fake_it,
					wfl_adj.vx, wfl_adj.vz, wfl_adj.tzz, wfl_adj.txx, wfl_adj.txz,
					verbose);
				adj_snap_count++;
			}
		}

	} /* end segment loop */

	/* ------------------------------------------------------------ */
	/* 6. Restore residual (undo negation so caller's data is       */
	/*    unchanged).                                                */
	/* ------------------------------------------------------------ */
	{
		size_t i, ntotal = (size_t)adj->nsrc * adj->nt;
		for (i = 0; i < ntotal; i++)
			adj->wav[i] = -adj->wav[i];
	}

	if (verbose)
		vmess("adj_shot: Completed gradient accumulation in %.2f s.", wallclock_time() - t0_wall);

	/* ------------------------------------------------------------ */
	/* 7. Convert to velocity parameterization if param=2           */
	/*                                                               */
	/* Chain rule for velocity parameterization:                     */
	/*   λ = ρ(Vp² - 2Vs²),  μ = ρVs²                               */
	/*   g_Vp = g_λ × 2ρVp                                          */
	/*   g_Vs = 2ρVs(g_μ - 2g_λ)                                    */
	/*   g_ρ  = g_ρ(direct) + g_λ(Vp² - 2Vs²) + g_μ(Vs²)           */
	/* ------------------------------------------------------------ */
	if (param == 2 && mod->ischeme > 2) {
		if (verbose)
			vmess("adj_shot: Converting to velocity parameterization (Vp, Vs, rho)");
		convertGradientToVelocity(grad1, grad2, grad3,
		                          mod->cp, mod->cs, mod->rho, sizem);
	}

	/* ------------------------------------------------------------ */
	/* 8. Free                                                       */
	/* ------------------------------------------------------------ */
	free(buf_vx);
	free(buf_vz);
	if (buf_txx) free(buf_txx);
	if (buf_tzz) free(buf_tzz);
	if (buf_txz) free(buf_txz);
	if (seg_end_txx) free(seg_end_txx);
	if (seg_end_tzz) free(seg_end_tzz);
	if (seg_end_txz) free(seg_end_txz);
	free(wfl_fwd.vx); free(wfl_fwd.vz); free(wfl_fwd.tzz);
	if (wfl_fwd.txx) free(wfl_fwd.txx);
	if (wfl_fwd.txz) free(wfl_fwd.txz);
	free(wfl_adj.vx); free(wfl_adj.vz); free(wfl_adj.tzz);
	if (wfl_adj.txx) free(wfl_adj.txx);
	if (wfl_adj.txz) free(wfl_adj.txz);

	return 0;
}
