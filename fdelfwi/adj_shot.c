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

/* Adjoint FD kernels (multicomponent source injection at correct time-step points) */
int elastic4_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int rec_delay, int rec_skipdt, int verbose);

/* Checkpoint I/O (checkpoint.c) */
int readCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl);

/* Snapshot output (writeSnapTimes.c) */
int writeSnapTimes(modPar mod, snaPar sna, bndPar bnd, wavPar wav,
                   int ixsrc, int izsrc, int itime,
                   float *vx, float *vz, float *tzz, float *txx, float *txz,
                   int verbose);
void writeSnapTimesReset(void);


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
			if (mod->iorder == 4)
				elastic4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			else if (mod->iorder == 6)
				elastic6(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			break;
		case 5:
			if (mod->iorder == 4)
				elastic4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
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
			/* TODO: else if (mod->iorder == 6) elastic6_adj(...) */
			break;
	}
}


/***********************************************************************
 * accumGradient -- Cross-correlate forward/adjoint wavefields (ELASTIC ONLY).
 *
 * Accumulates gradient contributions for one time step using the
 * rigorous formulas from the adjoint-state method (Métivier & Brossier).
 *
 * NOTE: This implementation is for ELASTIC FWI only (ischeme=3).
 *       Acoustic FWI would require different gradient formulas.
 *
 * Gradient formulas (velocity-stress formulation):
 *
 *   Lambda (Eq. 51):  g_λ = ∫ tr(σ_adj)(∇·v_fwd) dt
 *                         = ∫ (σxx_adj + σzz_adj)(∂vx/∂x + ∂vz/∂z) dt
 *                    Computed at P grid (where txx, tzz are defined)
 *
 *   Mu (Eq. 54):      g_μ = ∫ 2 σ_adj : ε̇_fwd dt
 *                         = ∫ 2(σxx_adj·ε̇xx + σzz_adj·ε̇zz + 2·σxz_adj·ε̇xz) dt
 *                    Computed at Txz grid (where muu is defined)
 *                    Diagonal terms interpolated from P grid to Txz grid
 *
 *   Density (Eq. 21): g_ρ = ∫ v_adj · (∂v_fwd/∂t) dt
 *
 * Boundary handling:
 *   - Absorbing boundaries (type 2,4): skip taper zone (ntap points)
 *   - Free surface (type 1): skip 1 point below surface for stress gradients
 *   - FD stencil requirements: 4th-order needs 2 points margin for ix±2, iz±2
 *   - Interpolation in mu gradient needs ix-1, iz-1 access
 *
 ***********************************************************************/
static void accumGradient(modPar *mod, bndPar *bnd,
                          float *fwd_vx, float *fwd_vz,
                          float *fwd_vx_prev, float *fwd_vz_prev,
                          wflPar *wfl_adj,
                          float dt,
                          float *grad_lam, float *grad_muu, float *grad_rho)
{
	int ix, iz, n1, nax;
	int ibPx, iePx, ibPz, iePz;
	int ibTx, ieTx, ibTz, ieTz;
	int ibVx_x, ieVx_x, ibVx_z, ieVx_z;
	int ibVz_x, ieVz_x, ibVz_z, ieVz_z;
	float sdx, sdz;
	float c1, c2;

	/* Only for elastic (ischeme > 2) */
	if (mod->ischeme <= 2) return;

	n1  = mod->naz;
	nax = mod->nax;
	sdx = 1.0f / mod->dx;
	sdz = 1.0f / mod->dz;

	/* FD coefficients for 4th order */
	c1 = 9.0f/8.0f;
	c2 = -1.0f/24.0f;

	/* ================================================================
	 * Compute safe loop bounds for each gradient type.
	 * Start with kernel loop bounds, then adjust for:
	 *   1. Absorbing boundaries: skip ntap points
	 *   2. Free surface: skip 1 point below surface
	 *   3. FD stencil requirements: ensure ix±2, iz±2 are valid
	 *   4. Interpolation margin for mu gradient
	 * ================================================================ */

	/* --- Lambda gradient bounds (P grid) --- */
	ibPx = mod->ioPx;
	iePx = mod->iePx;
	ibPz = mod->ioPz;
	iePz = mod->iePz;

	/* Absorbing boundary adjustments */
	if (bnd->lef == 4 || bnd->lef == 2) ibPx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) iePx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibPz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) iePz -= bnd->ntap;

	/* Free surface: skip gradient at surface row (stress BC artifacts) */
	if (bnd->top == 1) ibPz = MAX(ibPz, mod->ioPz + 1);

	/* FD stencil safety: need ix-1, ix+2, iz-1, iz+2 valid */
	ibPx = MAX(ibPx, 1);
	iePx = MAX(ibPx, iePx);  /* ensure iePx >= ibPx */
	if (iePx > nax - 2) iePx = nax - 2;
	ibPz = MAX(ibPz, 1);
	iePz = MAX(ibPz, iePz);
	if (iePz > n1 - 2) iePz = n1 - 2;

	/* --- Mu gradient bounds (Txz grid) --- */
	ibTx = mod->ioTx;
	ieTx = mod->ieTx;
	ibTz = mod->ioTz;
	ieTz = mod->ieTz;

	/* Absorbing boundary adjustments */
	if (bnd->lef == 4 || bnd->lef == 2) ibTx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) ieTx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibTz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) ieTz -= bnd->ntap;

	/* Free surface: skip gradient near surface (txz=0 BC and interpolation) */
	if (bnd->top == 1) ibTz = MAX(ibTz, mod->ioTz + 1);

	/* FD stencil + interpolation safety:
	 * - Interpolation needs ix-1, iz-1
	 * - Stencil for dvxdx_00 etc needs ix-2, iz-2 and ix+2, iz+2 */
	ibTx = MAX(ibTx, 2);
	ieTx = MAX(ibTx, ieTx);
	if (ieTx > nax - 2) ieTx = nax - 2;
	ibTz = MAX(ibTz, 2);
	ieTz = MAX(ibTz, ieTz);
	if (ieTz > n1 - 2) ieTz = n1 - 2;

	/* --- Density gradient bounds (Vx and Vz grids) ---
	 * Note: mod->ioXx, ioXz, ioZx, ioZz are ALREADY adjusted for ntap in
	 * getParameters.c (lines 488-489, 508-509), so we only need to adjust
	 * the END indices for right/bottom absorbing boundaries.
	 */
	ibVx_x = mod->ioXx;
	ieVx_x = mod->ieXx;
	ibVx_z = mod->ioXz;
	ieVx_z = mod->ieXz;
	ibVz_x = mod->ioZx;
	ieVz_x = mod->ieZx;
	ibVz_z = mod->ioZz;
	ieVz_z = mod->ieZz;

	/* Absorbing boundary adjustments for END indices only (start already adjusted) */
	if (bnd->rig == 4 || bnd->rig == 2) { ieVx_x -= bnd->ntap; ieVz_x -= bnd->ntap; }
	if (bnd->bot == 4 || bnd->bot == 2) { ieVx_z -= bnd->ntap; ieVz_z -= bnd->ntap; }

	/* ================================================================
	 * Lambda gradient at P grid (where txx, tzz are defined)
	 * g_λ = ∫ tr(σ_adj)(∇·v_fwd) dt  (Eq. 51)
	 * ================================================================ */
	if (grad_lam) {
		for (ix = ibPx; ix < iePx; ix++) {
			for (iz = ibPz; iz < iePz; iz++) {
				float dvxdx_f, dvzdz_f;

				/* Forward velocity divergence at P grid */
				dvxdx_f = sdx*(c1*(fwd_vx[(ix+1)*n1+iz]-fwd_vx[ix*n1+iz])
				              +c2*(fwd_vx[(ix+2)*n1+iz]-fwd_vx[(ix-1)*n1+iz]));
				dvzdz_f = sdz*(c1*(fwd_vz[ix*n1+iz+1]-fwd_vz[ix*n1+iz])
				              +c2*(fwd_vz[ix*n1+iz+2]-fwd_vz[ix*n1+iz-1]));

				/* g_λ = tr(σ_adj) * div(v_fwd) */
				grad_lam[ix*n1+iz] -= dt*(wfl_adj->txx[ix*n1+iz] + wfl_adj->tzz[ix*n1+iz])
				                        *(dvxdx_f + dvzdz_f);
			}
		}
	}

	/* ================================================================
	 * Mu gradient at Txz grid (where muu is defined in staggered grid)
	 * g_μ = ∫ 2 σ_adj : ε̇_fwd dt  (Eq. 54)
	 * All terms interpolated to Txz grid position for consistency
	 * ================================================================ */
	if (grad_muu) {
		for (ix = ibTx; ix < ieTx; ix++) {
			for (iz = ibTz; iz < ieTz; iz++) {
				float txx_interp, tzz_interp;
				float dvxdx_interp, dvzdz_interp;
				float dvxdz_f, dvzdx_f;

				/* Interpolate txx, tzz from P grid to Txz grid (4-point average) */
				txx_interp = 0.25f*(wfl_adj->txx[ix*n1+iz] + wfl_adj->txx[(ix-1)*n1+iz]
				                  + wfl_adj->txx[ix*n1+iz-1] + wfl_adj->txx[(ix-1)*n1+iz-1]);
				tzz_interp = 0.25f*(wfl_adj->tzz[ix*n1+iz] + wfl_adj->tzz[(ix-1)*n1+iz]
				                  + wfl_adj->tzz[ix*n1+iz-1] + wfl_adj->tzz[(ix-1)*n1+iz-1]);

				/* Interpolate dvx/dx from P grid to Txz grid */
				{
					float dvxdx_00, dvxdx_10, dvxdx_01, dvxdx_11;
					dvxdx_00 = sdx*(c1*(fwd_vx[ix*n1+iz-1]-fwd_vx[(ix-1)*n1+iz-1])
					              +c2*(fwd_vx[(ix+1)*n1+iz-1]-fwd_vx[(ix-2)*n1+iz-1]));
					dvxdx_10 = sdx*(c1*(fwd_vx[(ix+1)*n1+iz-1]-fwd_vx[ix*n1+iz-1])
					              +c2*(fwd_vx[(ix+2)*n1+iz-1]-fwd_vx[(ix-1)*n1+iz-1]));
					dvxdx_01 = sdx*(c1*(fwd_vx[ix*n1+iz]-fwd_vx[(ix-1)*n1+iz])
					              +c2*(fwd_vx[(ix+1)*n1+iz]-fwd_vx[(ix-2)*n1+iz]));
					dvxdx_11 = sdx*(c1*(fwd_vx[(ix+1)*n1+iz]-fwd_vx[ix*n1+iz])
					              +c2*(fwd_vx[(ix+2)*n1+iz]-fwd_vx[(ix-1)*n1+iz]));
					dvxdx_interp = 0.25f*(dvxdx_00 + dvxdx_10 + dvxdx_01 + dvxdx_11);
				}

				/* Interpolate dvz/dz from P grid to Txz grid */
				{
					float dvzdz_00, dvzdz_10, dvzdz_01, dvzdz_11;
					dvzdz_00 = sdz*(c1*(fwd_vz[(ix-1)*n1+iz]-fwd_vz[(ix-1)*n1+iz-1])
					              +c2*(fwd_vz[(ix-1)*n1+iz+1]-fwd_vz[(ix-1)*n1+iz-2]));
					dvzdz_10 = sdz*(c1*(fwd_vz[ix*n1+iz]-fwd_vz[ix*n1+iz-1])
					              +c2*(fwd_vz[ix*n1+iz+1]-fwd_vz[ix*n1+iz-2]));
					dvzdz_01 = sdz*(c1*(fwd_vz[(ix-1)*n1+iz+1]-fwd_vz[(ix-1)*n1+iz])
					              +c2*(fwd_vz[(ix-1)*n1+iz+2]-fwd_vz[(ix-1)*n1+iz-1]));
					dvzdz_11 = sdz*(c1*(fwd_vz[ix*n1+iz+1]-fwd_vz[ix*n1+iz])
					              +c2*(fwd_vz[ix*n1+iz+2]-fwd_vz[ix*n1+iz-1]));
					dvzdz_interp = 0.25f*(dvzdz_00 + dvzdz_10 + dvzdz_01 + dvzdz_11);
				}

				/* Shear strain rate at Txz grid (native position) */
				dvxdz_f = sdz*(c1*(fwd_vx[ix*n1+iz]-fwd_vx[ix*n1+iz-1])
				              +c2*(fwd_vx[ix*n1+iz+1]-fwd_vx[ix*n1+iz-2]));
				dvzdx_f = sdx*(c1*(fwd_vz[ix*n1+iz]-fwd_vz[(ix-1)*n1+iz])
				              +c2*(fwd_vz[(ix+1)*n1+iz]-fwd_vz[(ix-2)*n1+iz]));

				/* g_μ = 2(σxx·ε̇xx + σzz·ε̇zz + 2·σxz·ε̇xz)
				 * Note: ε̇xz = 0.5*(dvx/dz + dvz/dx), so 2·ε̇xz = (dvx/dz + dvz/dx) */
				grad_muu[ix*n1+iz] -= dt*2.0f*(txx_interp*dvxdx_interp
				                             + tzz_interp*dvzdz_interp
				                             + wfl_adj->txz[ix*n1+iz]*(dvxdz_f + dvzdx_f));
			}
		}
	}

	/* ================================================================
	 * Density gradient (Eq. 21): g_ρ = ∫ v_adj · (∂v_fwd/∂t) dt
	 * Time derivative approximated as (v[t] - v[t-dt]) / dt
	 * ================================================================ */
	if (grad_rho && fwd_vx_prev && fwd_vz_prev) {
		float dvx_dt, dvz_dt;
		float sdt = 1.0f / dt;

		/* Vx contribution */
		for (ix = ibVx_x; ix < ieVx_x; ix++) {
			for (iz = ibVx_z; iz < ieVx_z; iz++) {
				dvx_dt = (fwd_vx[ix*n1+iz] - fwd_vx_prev[ix*n1+iz]) * sdt;
				grad_rho[ix*n1+iz] -= dt * wfl_adj->vx[ix*n1+iz] * dvx_dt;
			}
		}
		/* Vz contribution */
		for (ix = ibVz_x; ix < ieVz_x; ix++) {
			for (iz = ibVz_z; iz < ieVz_z; iz++) {
				dvz_dt = (fwd_vz[ix*n1+iz] - fwd_vz_prev[ix*n1+iz]) * sdt;
				grad_rho[ix*n1+iz] -= dt * wfl_adj->vz[ix*n1+iz] * dvz_dt;
			}
		}
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
	float *grad_lam, *grad_muu, *grad_rho;  /* Internal Lamé gradient pointers */
	int k, j, it, seg_start, seg_end, nsteps, max_nsteps;
	size_t sizem, i;
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

	if (verbose) {
		vmess("adj_shot: %d segments, max_nsteps=%d, buffer=%.1f MB",
			chk->nsnap, max_nsteps,
			2.0 * max_nsteps * sizem * sizeof(float) / (1024.0*1024.0));
		t0_wall = wallclock_time();
	}

	/* ------------------------------------------------------------ */
	/* 4. Process segments in reverse order                         */
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

		/* Re-propagate forward through rest of segment */
		for (j = 1; j < nsteps; j++) {
			it = seg_start + j;
#pragma omp parallel default(shared)
{
			callKernel(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav, &wfl_fwd, verbose);
}
			memcpy(buf_vx + (size_t)j * sizem, wfl_fwd.vx, sizem * sizeof(float));
			memcpy(buf_vz + (size_t)j * sizem, wfl_fwd.vz, sizem * sizeof(float));
		}

		/* ---- 4b. Adjoint backward through segment ---- */
		for (j = nsteps - 1; j >= 0; j--) {
			it = seg_start + j;

			/* Advance adjoint wavefield with multicomponent source injection.
			 * The adjoint kernel injects force residuals (Fx,Fz) at the
			 * velocity-update point and stress residuals (P,Txx,Tzz,Txz)
			 * at the stress-update point — physically correct timing. */
#pragma omp parallel default(shared)
{
			callAdjKernel(mod, adj, bnd, it, &wfl_adj,
				rec->delay, rec->skipdt, verbose);
}

			/* Write adjoint wavefield snapshot if requested.
			 * Check snapshot timing manually and remap to sequential
			 * itime so that writeSnapTimes produces headers with
			 * fldr=1,2,3,... in write order (required for suxmovie). */
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

			/* Cross-correlate for gradient.
			 * For density gradient, we need the time derivative of forward velocity,
			 * which requires the previous time step. For j=0, we don't have it
			 * (would need previous checkpoint), so pass NULL to skip density. */
			{
				float *vx_prev = (j > 0) ? buf_vx + (size_t)(j-1) * sizem : NULL;
				float *vz_prev = (j > 0) ? buf_vz + (size_t)(j-1) * sizem : NULL;

				accumGradient(mod, bnd,
					buf_vx + (size_t)j * sizem,
					buf_vz + (size_t)j * sizem,
					vx_prev, vz_prev,
					&wfl_adj, dt,
					grad_lam, grad_muu, grad_rho);
			}
		}

	} /* end segment loop */

	if (verbose)
		vmess("adj_shot: Completed gradient accumulation in %.2f s.", wallclock_time() - t0_wall);

	/* ------------------------------------------------------------ */
	/* 5. Convert to velocity parameterization if param=2           */
	/*                                                               */
	/* Chain rule for velocity parameterization:                     */
	/*   λ = ρ(Vp² - 2Vs²),  μ = ρVs²                               */
	/*   g_Vp = g_λ × 2ρVp                                          */
	/*   g_Vs = 2ρVs(g_μ - 2g_λ)                                    */
	/*   g_ρ  = g_ρ(direct) + g_λ(Vp² - 2Vs²) + g_μ(Vs²)           */
	/* ------------------------------------------------------------ */
	if (param == 2 && mod->ischeme > 2) {
		float *cp_arr = mod->cp;   /* Vp array */
		float *cs_arr = mod->cs;   /* Vs array */
		float *ro_arr = mod->rho;  /* density array */

		if (verbose)
			vmess("adj_shot: Converting to velocity parameterization (Vp, Vs, rho)");

		/* Convert in-place: grad1,2,3 contain Lamé gradients, will become velocity gradients.
		 * Must do all three simultaneously since they depend on each other. */
		for (i = 0; i < sizem; i++) {
			float g_lam = grad1 ? grad1[i] : 0.0f;
			float g_muu = grad2 ? grad2[i] : 0.0f;
			float g_rho_direct = grad3 ? grad3[i] : 0.0f;

			float rho = ro_arr[i];
			float vp  = cp_arr[i];
			float vs  = cs_arr[i];
			float vp2 = vp * vp;
			float vs2 = vs * vs;

			/* Apply chain rule */
			float g_vp = g_lam * 2.0f * rho * vp;
			float g_vs = 2.0f * rho * vs * (g_muu - 2.0f * g_lam);
			float g_rho_full = g_rho_direct + g_lam * (vp2 - 2.0f * vs2) + g_muu * vs2;

			/* Store converted gradients */
			if (grad1) grad1[i] = g_vp;
			if (grad2) grad2[i] = g_vs;
			if (grad3) grad3[i] = g_rho_full;
		}
	}

	/* ------------------------------------------------------------ */
	/* 6. Free                                                       */
	/* ------------------------------------------------------------ */
	free(buf_vx);
	free(buf_vz);
	free(wfl_fwd.vx); free(wfl_fwd.vz); free(wfl_fwd.tzz);
	if (wfl_fwd.txx) free(wfl_fwd.txx);
	if (wfl_fwd.txz) free(wfl_fwd.txz);
	free(wfl_adj.vx); free(wfl_adj.vz); free(wfl_adj.tzz);
	if (wfl_adj.txx) free(wfl_adj.txx);
	if (wfl_adj.txz) free(wfl_adj.txz);

	return 0;
}
