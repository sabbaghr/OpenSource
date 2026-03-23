#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"fdelfwi.h"

/**
 * fwi_gradient.c -- FWI gradient cross-correlation and parameterization.
 *
 * Contains:
 *   accumGradient()              - per-timestep imaging condition
 *   convertGradientToVelocity()  - Lamé → (Vp, Vs, ρ) chain rule
 *
 * Gradient formulas (elastic velocity-stress formulation):
 *
 *   g_λ  = +∫ (ψ_txx + ψ_tzz)(∂vx/∂x + ∂vz/∂z) dt
 *   g_μ  = +∫ [2(ψ_txx·∂vx/∂x + ψ_tzz·∂vz/∂z) + ψ_txz·(∂vx/∂z + ∂vz/∂x)] dt
 *   g_ρ  = (1/ρ) ∫ (ψ_vx·∂vx/∂t + ψ_vz·∂vz/∂t) dt
 *
 * The mu gradient has factor 2 on the normal-stress terms (from l2m = λ+2μ)
 * but factor 1 on the shear term (from mul = μ).
 *
 * Pseudo-Hessian (Yang et al., 2018, SIAM J. Sci. Comput., eq 45):
 *   Replaces adjoint wavefield with forward wavefield in gradient formula.
 *   K_λ = (txx+tzz)·div(v),  K_μ = 2(txx·exx+tzz·ezz)+txz·exz,
 *   K_ρ = (1/ρ)(vx·ax+vz·az).  H̃_ij = Σ_t K_i·K_j.
 *
 * Spatial derivatives match the forward operator stencils (order 4, 6, or 8).
 */

#define MAX(x,y) ((x) > (y) ? (x) : (y))


/***********************************************************************
 * accumGradient -- Cross-correlate forward/adjoint wavefields.
 *
 * Accumulates gradient contributions for one time step using the
 * exact discrete sensitivities of the forward operator ∂F/∂m.
 * Supports FD orders 4, 6, and 8 with matching stencils.
 *
 * Sign convention: The adjoint wavefield ψ is driven by −residual
 * (see adj_shot.c), matching the Lagrangian convention ψ = −∂J/∂u.
 * Gradient formulas use += (positive sign), directly implementing:
 *   g_λ = +Σ dt·(ψ_txx + ψ_tzz)·div(v)
 *   g_μ = +Σ dt·[2·(ψ_txx·∂vx/∂x + ψ_tzz·∂vz/∂z) + ψ_txz·(∂vx/∂z + ∂vz/∂x)]
 *   g_ρ = +Σ (dt/ρ)·ψ_v·∂v/∂t
 * The result is dJ/dm (true gradient). Optimizer uses m -= α·g.
 *
 * Pseudo-Hessian (Yang et al., 2018):
 *   When fwd_txx/fwd_tzz/fwd_txz are non-NULL and hess_* arrays are
 *   non-NULL, accumulates the Yang pseudo-Hessian H̃_ij = Σ K_i·K_j
 *   where K_i = w†∂A/∂m_i w uses the forward wavefield for both
 *   "source" and "receiver" sides (zero-offset approximation).
 *
 *   Scratch arrays K_lam_tmp/K_muu_tmp/K_rho_tmp (size nax*naz each,
 *   pre-allocated by caller) hold per-timestep kernel values at P grid.
 ***********************************************************************/
void accumGradient(modPar *mod, bndPar *bnd,
                   float *fwd_vx, float *fwd_vz,
                   float *fwd_vx_prev, float *fwd_vz_prev,
                   float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                   wflPar *wfl_adj,
                   float dt,
                   float *grad_lam, float *grad_muu, float *grad_rho,
                   float *hess_lam, float *hess_muu, float *hess_rho,
                   float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
                   float *K_lam_tmp, float *K_muu_tmp, float *K_rho_tmp,
                   float *wfld_energy)
{
	int ix, iz, n1, nax;
	int ibPx, iePx, ibPz, iePz;
	int ibTx, ieTx, ibTz, ieTz;
	int ibVx_x, ieVx_x, ibVx_z, ieVx_z;
	int ibVz_x, ieVz_x, ibVz_z, ieVz_z;
	float sdx, sdz;
	float c1, c2, c3, c4;
	int half;

	int do_hess;   /* Yang pseudo-Hessian accumulation active */

	/* Only for elastic (ischeme > 2) */
	if (mod->ischeme <= 2) return;

	n1  = mod->naz;
	nax = mod->nax;
	sdx = 1.0f / mod->dx;
	sdz = 1.0f / mod->dz;

	/* Pseudo-Hessian accumulation active if any diagonal array is provided.
	 * New formulation (Liu et al. 2022) uses only forward velocities,
	 * not forward stress or scratch arrays. */
	do_hess = (hess_lam != NULL);


	/* ================================================================
	 * Order-dependent FD coefficients (must match forward kernels)
	 * ================================================================ */
	c3 = c4 = 0.0f;
	half = mod->iorder / 2;

	switch (mod->iorder) {
		case 4:
			c1 = 9.0f/8.0f;
			c2 = -1.0f/24.0f;
			break;
		case 6:
			c1 = 75.0f/64.0f;
			c2 = -25.0f/384.0f;
			c3 = 3.0f/640.0f;
			break;
		case 8:
			c1 = 1225.0f/1024.0f;
			c2 = -245.0f/3072.0f;
			c3 = 49.0f/5120.0f;
			c4 = -5.0f/7168.0f;
			break;
		default:
			c1 = 9.0f/8.0f;
			c2 = -1.0f/24.0f;
			break;
	}

	/* ================================================================
	 * Compute safe loop bounds for each gradient type.
	 * Start with kernel loop bounds, then adjust for:
	 *   1. Absorbing boundaries: skip ntap points
	 *   2. Free surface: skip 1 point below surface
	 *   3. FD stencil requirements (order-dependent)
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

	/* FD stencil safety: D+x(vx) accesses ix-(half-1) to ix+half
	 * So need ix >= half-1 and ix+half < nax, i.e. ix < nax-half */
	ibPx = MAX(ibPx, half - 1);
	iePx = MAX(ibPx, iePx);
	if (iePx > nax - half) iePx = nax - half;
	ibPz = MAX(ibPz, half - 1);
	iePz = MAX(ibPz, iePz);
	if (iePz > n1 - half) iePz = n1 - half;

	/* --- Mu shear gradient bounds (Txz grid) ---
	 * For the shear part: D-z(vx) and D-x(vz) at Txz grid
	 * D-z accesses iz-(half) to iz+(half-1), D-x accesses ix-(half) to ix+(half-1) */
	ibTx = mod->ioTx;
	ieTx = mod->ieTx;
	ibTz = mod->ioTz;
	ieTz = mod->ieTz;

	/* Absorbing boundary adjustments */
	if (bnd->lef == 4 || bnd->lef == 2) ibTx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) ieTx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibTz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) ieTz -= bnd->ntap;

	/* Free surface: skip gradient near surface (txz=0 BC) */
	if (bnd->top == 1) ibTz = MAX(ibTz, mod->ioTz + 1);

	/* FD stencil safety for D-z/D-x: need ix >= half, ix < nax - half */
	ibTx = MAX(ibTx, half);
	ieTx = MAX(ibTx, ieTx);
	if (ieTx > nax - half) ieTx = nax - half;
	ibTz = MAX(ibTz, half);
	ieTz = MAX(ibTz, ieTz);
	if (ieTz > n1 - half) ieTz = n1 - half;

	/* --- Density gradient bounds (Vx and Vz grids) ---
	 * In getParameters.c, the velocity grid io/ie indices are BOTH shifted
	 * by ntap for top/left absorbing boundaries.  Unlike the P grid (where
	 * ioPx is NOT shifted but iePx includes ALL ntap shifts), the velocity
	 * grid ie values do NOT include the right/bottom taper padding.
	 * Therefore: io already skips left/top taper, ie already stops at the
	 * physical domain edge.  No further ntap adjustments are needed.
	 */
	ibVx_x = mod->ioXx;
	ieVx_x = mod->ieXx;
	ibVx_z = mod->ioXz;
	ieVx_z = mod->ieXz;
	ibVz_x = mod->ioZx;
	ieVz_x = mod->ieZx;
	ibVz_z = mod->ioZz;
	ieVz_z = mod->ieZz;

	/* ================================================================
	 * Lambda gradient at P grid (where txx, tzz are defined)
	 *
	 *   g_λ[ix,iz] = +(ψ_txx + ψ_tzz) * (D+x(vx) + D+z(vz))
	 *
	 * The D+x/D+z stencils are the same as in the forward stress update.
	 * ================================================================ */
	if (grad_lam || do_hess) {
		for (ix = ibPx; ix < iePx; ix++) {
			for (iz = ibPz; iz < iePz; iz++) {
				float dvxdx_f, dvzdz_f, div_f;
				int ig = ix*n1+iz;

				/* Forward velocity divergence at P grid (order-dependent) */
				dvxdx_f = sdx*(c1*(fwd_vx[(ix+1)*n1+iz]-fwd_vx[ig])
				              +c2*(fwd_vx[(ix+2)*n1+iz]-fwd_vx[(ix-1)*n1+iz]));
				dvzdz_f = sdz*(c1*(fwd_vz[ig+1]-fwd_vz[ig])
				              +c2*(fwd_vz[ig+2]-fwd_vz[ig-1]));
				if (mod->iorder >= 6) {
					dvxdx_f += sdx*c3*(fwd_vx[(ix+3)*n1+iz]-fwd_vx[(ix-2)*n1+iz]);
					dvzdz_f += sdz*c3*(fwd_vz[ig+3]-fwd_vz[ig-2]);
				}
				if (mod->iorder >= 8) {
					dvxdx_f += sdx*c4*(fwd_vx[(ix+4)*n1+iz]-fwd_vx[(ix-3)*n1+iz]);
					dvzdz_f += sdz*c4*(fwd_vz[ig+4]-fwd_vz[ig-3]);
				}

				div_f = dvxdx_f + dvzdz_f;

				if (grad_lam)
					grad_lam[ig] += dt*(wfl_adj->txx[ig] + wfl_adj->tzz[ig])
					                   * div_f;

				/* Pseudo-Hessian diagonal at P grid (Liu et al. 2022, eq A20-A21).
				 * K_λ = (∂C/∂λ)Dv = [div, div, 0] → |K_λ|² = 2·div²
				 * K_μ normal part = [2exx, 2ezz, 0] → 4exx² + 4ezz²
				 * K_λ·K_μ = 2·div² = H_λλ (always identical) */
				if (do_hess) {
					hess_lam[ig]     += 2.0f * div_f * div_f;
					hess_muu[ig]     += 4.0f * (dvxdx_f*dvxdx_f + dvzdz_f*dvzdz_f);
					if (hess_lam_muu)
						hess_lam_muu[ig] += 2.0f * div_f * div_f;
					/* H_λρ = H_μρ = 0 in Lamé space (orthogonal subspaces) */
				}
			}
		}
	}

	/* ================================================================
	 * Mu gradient: two contributions at their native grid positions.
	 *
	 * 1. Normal-stress part at P grid (from l2m = λ+2μ, ∂l2m/∂μ = 2):
	 *    g_μ += 2*(ψ_txx·D+x(vx) + ψ_tzz·D+z(vz))
	 *    ψ_txx/ψ_tzz and D+x(vx)/D+z(vz) are both native at P grid.
	 *
	 * 2. Shear part at Txz grid (from mul = μ, ∂mul/∂μ = 1):
	 *    g_μ += ψ_txz·(D-z(vx) + D-x(vz))
	 *    ψ_txz and D-z(vx)/D-x(vz) are both native at Txz grid.
	 *
	 * No interpolation needed — each cross-correlation uses collocated fields.
	 * ================================================================ */
	if (grad_muu || do_hess) {
		/* --- Part 1: Normal-stress at P grid (same bounds as lambda) ---
		 * Gradient: g_μ += 2*(ψ_txx·dvxdx + ψ_tzz·dvzdz)
		 * K_μ_P already stored in K_muu_tmp by the lambda loop above. */
		if (grad_muu) {
			for (ix = ibPx; ix < iePx; ix++) {
				for (iz = ibPz; iz < iePz; iz++) {
					float dvxdx_f, dvzdz_f;
					int ig = ix*n1+iz;

					dvxdx_f = sdx*(c1*(fwd_vx[(ix+1)*n1+iz]-fwd_vx[ig])
					              +c2*(fwd_vx[(ix+2)*n1+iz]-fwd_vx[(ix-1)*n1+iz]));
					dvzdz_f = sdz*(c1*(fwd_vz[ig+1]-fwd_vz[ig])
					              +c2*(fwd_vz[ig+2]-fwd_vz[ig-1]));
					if (mod->iorder >= 6) {
						dvxdx_f += sdx*c3*(fwd_vx[(ix+3)*n1+iz]-fwd_vx[(ix-2)*n1+iz]);
						dvzdz_f += sdz*c3*(fwd_vz[ig+3]-fwd_vz[ig-2]);
					}
					if (mod->iorder >= 8) {
						dvxdx_f += sdx*c4*(fwd_vx[(ix+4)*n1+iz]-fwd_vx[(ix-3)*n1+iz]);
						dvzdz_f += sdz*c4*(fwd_vz[ig+4]-fwd_vz[ig-3]);
					}

					grad_muu[ig] += dt*2.0f*(wfl_adj->txx[ig]*dvxdx_f
					                        + wfl_adj->tzz[ig]*dvzdz_f);
				}
			}
		}

		/* --- Part 2: Shear at Txz grid, scattered to P grid ---
		 * Compute ψ_txz·(D-z(vx)+D-x(vz)) at native Txz grid, then
		 * scatter to the 4 surrounding P-grid points with weight 0.25.
		 * This is the transpose of the harmonic averaging in readModel.c
		 * (approximated with arithmetic weights).
		 * Index mapping: Txz(ix,iz) → P-grid (ix-1,iz-1),(ix,iz-1),(ix-1,iz),(ix,iz)
		 * since ioTx = ioPx+1, ioTz = ioPz+1.
		 *
		 * For Yang pseudo-Hessian: also scatter K_μ_S = txz·(exz) to P grid
		 * so that K_muu_tmp = K_μ_P + K_μ_S at P grid. */
		for (ix = ibTx; ix < ieTx; ix++) {
			for (iz = ibTz; iz < ieTz; iz++) {
				float dvxdz_f, dvzdx_f;
				int ig = ix*n1+iz;

				dvxdz_f = sdz*(c1*(fwd_vx[ig]-fwd_vx[ig-1])
				              +c2*(fwd_vx[ig+1]-fwd_vx[ig-2]));
				dvzdx_f = sdx*(c1*(fwd_vz[ig]-fwd_vz[(ix-1)*n1+iz])
				              +c2*(fwd_vz[(ix+1)*n1+iz]-fwd_vz[(ix-2)*n1+iz]));
				if (mod->iorder >= 6) {
					dvxdz_f += sdz*c3*(fwd_vx[ig+2]-fwd_vx[ig-3]);
					dvzdx_f += sdx*c3*(fwd_vz[(ix+2)*n1+iz]-fwd_vz[(ix-3)*n1+iz]);
				}
				if (mod->iorder >= 8) {
					dvxdz_f += sdz*c4*(fwd_vx[ig+3]-fwd_vx[ig-4]);
					dvzdx_f += sdx*c4*(fwd_vz[(ix+3)*n1+iz]-fwd_vz[(ix-4)*n1+iz]);
				}

				if (grad_muu) {
					float shear = dt*wfl_adj->txz[ig]*(dvxdz_f + dvzdx_f);
					/* Scatter to 4 surrounding P-grid points */
					grad_muu[(ix-1)*n1+iz-1] += 0.25f * shear;
					grad_muu[ix*n1+iz-1]     += 0.25f * shear;
					grad_muu[(ix-1)*n1+iz]   += 0.25f * shear;
					grad_muu[ig]             += 0.25f * shear;
				}
				/* H_μμ shear part: (dvx/dz + dvz/dx)² scattered to P grid.
				 * K_μ shear component = dvx/dz + dvz/dx
				 * |K_μ_shear|² = (dvx/dz + dvz/dx)² */
				if (do_hess) {
					float dvxdz_dvzdx = dvxdz_f + dvzdx_f;
					float shear_sq = dvxdz_dvzdx * dvxdz_dvzdx;
					hess_muu[(ix-1)*n1+iz-1] += 0.25f * shear_sq;
					hess_muu[ix*n1+iz-1]     += 0.25f * shear_sq;
					hess_muu[(ix-1)*n1+iz]   += 0.25f * shear_sq;
					hess_muu[ig]             += 0.25f * shear_sq;
				}
			}
		}
	}

	/* ================================================================
	 * Density gradient:
	 *
	 *   g_ρ = -(1/ρ) ∫ ψ_v · (∂v_fwd/∂t) dt
	 *
	 * Derivation: rox = dt/(dx·ρ), drox/dρ = -rox/ρ
	 *   → d(vx)/d(ρ) = (drox/dρ) · RHS = -(1/ρ) · Δvx
	 *   (negative sign handled by residual negation in adj_shot)
	 *
	 * Time derivative approximated as (v[t] - v[t-dt]) / dt.
	 *
	 * ψ_vx · dvx/dt is native at Vx grid, which straddles P(ix-1,iz)
	 * and P(ix,iz) in x.  Scatter to those two P points with weight 0.5.
	 *
	 * ψ_vz · dvz/dt is native at Vz grid, which straddles P(ix,iz-1)
	 * and P(ix,iz) in z.  Same scatter pattern.
	 * ================================================================ */
	if ((grad_rho || do_hess) && fwd_vx_prev && fwd_vz_prev) {
		float dvx_dt, dvz_dt;
		float sdt = 1.0f / dt;
		float *rho = mod->rho;

		/* Vx contribution, scattered to P grid */
		for (ix = ibVx_x; ix < ieVx_x; ix++) {
			for (iz = ibVx_z; iz < ieVx_z; iz++) {
				int ig = ix*n1+iz;
				dvx_dt = (fwd_vx[ig] - fwd_vx_prev[ig]) * sdt;
				if (grad_rho) {
					float vx_contrib = dt * wfl_adj->vx[ig] * dvx_dt;
					grad_rho[(ix-1)*n1+iz] += 0.5f * vx_contrib / rho[(ix-1)*n1+iz];
					grad_rho[ig]           += 0.5f * vx_contrib / rho[ig];
				}
				/* H_ρρ: (dvx/dt)² scattered to P grid.
				 * K_ρ = [∂ₜvx, ∂ₜvz] → |K_ρ|² = (∂ₜvx)² + (∂ₜvz)²
				 * (Liu et al. 2022, eq A21) */
				if (do_hess) {
					float dvx_dt_sq = dvx_dt * dvx_dt;
					hess_rho[(ix-1)*n1+iz] += 0.5f * dvx_dt_sq;
					hess_rho[ig]           += 0.5f * dvx_dt_sq;
				}
			}
		}
		/* Vz contribution, scattered to P grid */
		for (ix = ibVz_x; ix < ieVz_x; ix++) {
			for (iz = ibVz_z; iz < ieVz_z; iz++) {
				int ig = ix*n1+iz;
				dvz_dt = (fwd_vz[ig] - fwd_vz_prev[ig]) * sdt;
				if (grad_rho) {
					float vz_contrib = dt * wfl_adj->vz[ig] * dvz_dt;
					grad_rho[ig-1] += 0.5f * vz_contrib / rho[ig-1];
					grad_rho[ig]   += 0.5f * vz_contrib / rho[ig];
				}
				/* H_ρρ: (dvz/dt)² scattered to P grid */
				if (do_hess) {
					float dvz_dt_sq = dvz_dt * dvz_dt;
					hess_rho[ig-1] += 0.5f * dvz_dt_sq;
					hess_rho[ig]   += 0.5f * dvz_dt_sq;
				}
			}
		}
	}

	/* ================================================================
	 * Note: H_λλ, H_μμ, H_ρρ, H_λμ are accumulated DIRECTLY in their
	 * respective loops above (P grid, Txz grid, Vx/Vz grid).
	 * No final K_i·K_j sweep needed.
	 *
	 * H_λρ = H_μρ = 0 in Lamé space (stress-space and velocity-space
	 * virtual sources are orthogonal). These arrays are not touched.
	 * ================================================================ */

	/* ================================================================
	 * Forward wavefield energy: Ws += vx² + vz²
	 *
	 * Particle velocity energy (DENISE convention, eprecond.c).
	 * Used by precond=1 (Plessix & Mulder / DENISE EPRECOND=3).
	 * Averaged to P grid from staggered Vx/Vz grids.
	 * ================================================================ */
	if (wfld_energy && fwd_vx && fwd_vz) {
		for (ix = ibPx; ix < iePx; ix++) {
			for (iz = ibPz; iz < iePz; iz++) {
				int ig = ix*n1+iz;
				float vx_avg = 0.5f*(fwd_vx[ig] + fwd_vx[(ix+1)*n1+iz]);
				float vz_avg = 0.5f*(fwd_vz[ig] + fwd_vz[ig+1]);
				wfld_energy[ig] += vx_avg*vx_avg + vz_avg*vz_avg;
			}
		}
	}
}


/***********************************************************************
 * accumGradient_rho_Dsig -- Density gradient using exact D_σ(stress).
 *
 * Replaces the dv/dt approximation in accumGradient for density with
 * the exact spatial derivative D_σ(forward stress), eliminating source
 * injection contamination that breaks the B/B^T transpose duality
 * needed for Hessian symmetry.
 *
 * The exact transpose of born_vsrc_vel for density is:
 *   g_ρ[P] += Σ_t 0.5 * (ψ_vx · rox · Dsig_vx) / ρ   [scatter]
 *   g_ρ[P] += Σ_t 0.5 * (ψ_vz · roz · Dsig_vz) / ρ   [scatter]
 *
 * where Dsig_vx = D-x(txx) + D+z(txz) at the Vx grid, and
 *       Dsig_vz = D+x(txz) + D-z(tzz) at the Vz grid.
 *
 * This uses the same stencils and loop bounds as born_vsrc_vel.
 *
 * Parameters:
 *   fwd_txx, fwd_tzz, fwd_txz - forward stress at time n-1
 *                                (stress BEFORE step n, same timing
 *                                 as born_vsrc_vel at step n)
 *   wfl_adj     - adjoint wavefield (ψ_vx, ψ_vz after callAdjKernel)
 *   grad_rho    - density gradient array (accumulated, padded grid)
 ***********************************************************************/
void accumGradient_rho_Dsig(modPar *mod, bndPar *bnd,
                            float *fwd_txx, float *fwd_tzz, float *fwd_txz,
                            wflPar *wfl_adj,
                            float dt,
                            float *grad_rho)
{
	int ix, iz, n1;
	float c1, c2, c3, c4;

	(void)dt;  /* dt parameter kept for interface consistency but unused;
	            * the old dv/dt code had dt*(Δvx/dt)=Δvx, which cancels. */

	if (mod->ischeme <= 2) return;
	if (!grad_rho || !fwd_txx || !fwd_tzz || !fwd_txz) return;

	n1  = mod->naz;
	c3 = c4 = 0.0f;

	switch (mod->iorder) {
		case 4:
			c1 = 9.0f/8.0f;
			c2 = -1.0f/24.0f;
			break;
		case 6:
			c1 = 75.0f/64.0f;
			c2 = -25.0f/384.0f;
			c3 = 3.0f/640.0f;
			break;
		case 8:
			c1 = 1225.0f/1024.0f;
			c2 = -245.0f/3072.0f;
			c3 = 49.0f/5120.0f;
			c4 = -5.0f/7168.0f;
			break;
		default:
			c1 = 9.0f/8.0f;
			c2 = -1.0f/24.0f;
			break;
	}

	/* Vx contribution: Dsig_vx = D-x(txx) + D+z(txz)
	 * Same stencil as born_vsrc_vel and elastic4 Phase F1 for vx.
	 * Loop bounds: ioXx..ieXx (same as born_vsrc_vel and accumGradient). */
	for (ix = mod->ioXx; ix < mod->ieXx; ix++) {
		for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
			float Dsig, vx_contrib;
			float rox_val = mod->rox[ix*n1+iz];

			Dsig = c1*(fwd_txx[ix*n1+iz]     - fwd_txx[(ix-1)*n1+iz] +
			           fwd_txz[ix*n1+iz+1]   - fwd_txz[ix*n1+iz])    +
			       c2*(fwd_txx[(ix+1)*n1+iz] - fwd_txx[(ix-2)*n1+iz] +
			           fwd_txz[ix*n1+iz+2]   - fwd_txz[ix*n1+iz-1]);
			if (mod->iorder >= 6)
				Dsig += c3*(fwd_txx[(ix+2)*n1+iz] - fwd_txx[(ix-3)*n1+iz] +
				            fwd_txz[ix*n1+iz+3]   - fwd_txz[ix*n1+iz-2]);
			if (mod->iorder >= 8)
				Dsig += c4*(fwd_txx[(ix+3)*n1+iz] - fwd_txx[(ix-4)*n1+iz] +
				            fwd_txz[ix*n1+iz+4]   - fwd_txz[ix*n1+iz-3]);

			/* Forward kernel: vx -= rox * Dsig, so Δvx = -rox * Dsig.
			 * The old dv/dt code: vx_contrib = ψ*Δvx = -ψ*rox*Dsig.
			 * We must match the negative sign for correct -J^T. */
			vx_contrib = -wfl_adj->vx[ix*n1+iz] * rox_val * Dsig;

			/* Scatter to 2 P-grid points straddled by Vx(ix,iz) */
			grad_rho[(ix-1)*n1+iz] += 0.5f * vx_contrib / mod->rho[(ix-1)*n1+iz];
			grad_rho[ix*n1+iz]     += 0.5f * vx_contrib / mod->rho[ix*n1+iz];
		}
	}

	/* Vz contribution: Dsig_vz = D+x(txz) + D-z(tzz)
	 * Same stencil as born_vsrc_vel and elastic4 Phase F1 for vz. */
	for (ix = mod->ioZx; ix < mod->ieZx; ix++) {
		for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
			float Dsig, vz_contrib;
			float roz_val = mod->roz[ix*n1+iz];

			Dsig = c1*(fwd_tzz[ix*n1+iz]     - fwd_tzz[ix*n1+iz-1]    +
			           fwd_txz[(ix+1)*n1+iz] - fwd_txz[ix*n1+iz])     +
			       c2*(fwd_tzz[ix*n1+iz+1]   - fwd_tzz[ix*n1+iz-2]    +
			           fwd_txz[(ix+2)*n1+iz] - fwd_txz[(ix-1)*n1+iz]);
			if (mod->iorder >= 6)
				Dsig += c3*(fwd_tzz[ix*n1+iz+2]   - fwd_tzz[ix*n1+iz-3]    +
				            fwd_txz[(ix+3)*n1+iz] - fwd_txz[(ix-2)*n1+iz]);
			if (mod->iorder >= 8)
				Dsig += c4*(fwd_tzz[ix*n1+iz+3]   - fwd_tzz[ix*n1+iz-4]    +
				            fwd_txz[(ix+4)*n1+iz] - fwd_txz[(ix-3)*n1+iz]);

			/* Forward kernel: vz -= roz * Dsig, so Δvz = -roz * Dsig. */
			vz_contrib = -wfl_adj->vz[ix*n1+iz] * roz_val * Dsig;

			/* Scatter to 2 P-grid points straddled by Vz(ix,iz) */
			grad_rho[ix*n1+(iz-1)] += 0.5f * vz_contrib / mod->rho[ix*n1+(iz-1)];
			grad_rho[ix*n1+iz]     += 0.5f * vz_contrib / mod->rho[ix*n1+iz];
		}
	}
}


/***********************************************************************
 * convertGradientToVelocity -- Chain rule: Lamé → (Vp, Vs, ρ).
 *
 * Converts Lamé-parameter gradients (g_λ, g_μ, g_ρ_direct) to
 * velocity-parameter gradients (g_Vp, g_Vs, g_ρ_full) in-place.
 *
 * Chain rule:
 *   λ = ρ(Vp² - 2Vs²),  μ = ρVs²
 *   g_Vp = g_λ × 2ρVp
 *   g_Vs = 2ρVs(g_μ - 2g_λ)
 *   g_ρ  = g_ρ(direct) + g_λ(Vp² - 2Vs²) + g_μ(Vs²)
 *
 * Parameters:
 *   grad1  - INPUT: g_λ,  OUTPUT: g_Vp   (in-place, may be NULL)
 *   grad2  - INPUT: g_μ,  OUTPUT: g_Vs   (in-place, may be NULL)
 *   grad3  - INPUT: g_ρ,  OUTPUT: g_ρ    (in-place, may be NULL)
 *   cp     - Vp array (padded grid, sizem elements)
 *   cs     - Vs array
 *   rho    - density array
 *   sizem  - total padded grid size (nax*naz)
 ***********************************************************************/
void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho,
                               size_t sizem)
{
	size_t i;

	for (i = 0; i < sizem; i++) {
		float g_lam = grad1 ? grad1[i] : 0.0f;
		float g_muu = grad2 ? grad2[i] : 0.0f;
		float g_rho_direct = grad3 ? grad3[i] : 0.0f;

		float rho_val = rho[i];
		float vp  = cp[i];
		float vs  = cs[i];
		float vp2 = vp * vp;
		float vs2 = vs * vs;

		/* Apply chain rule */
		float g_vp = g_lam * 2.0f * rho_val * vp;
		float g_vs = 2.0f * rho_val * vs * (g_muu - 2.0f * g_lam);
		float g_rho_full = g_rho_direct + g_lam * (vp2 - 2.0f * vs2) + g_muu * vs2;

		/* Store converted gradients */
		if (grad1) grad1[i] = g_vp;
		if (grad2) grad2[i] = g_vs;
		if (grad3) grad3[i] = g_rho_full;
	}
}


/***********************************************************************
 * accumHessianP4 -- Cross-correlate forward × receiver wavefield kernels
 *                   for preconditioner P4 (Liu et al. 2022, eq A22).
 *
 * At each timestep, computes:
 *   H_λλ += 2 · div(v_fwd) · div(v_rcv)
 *   H_μμ += 4·exx_fwd·exx_rcv + 4·ezz_fwd·ezz_rcv  (normal, P grid)
 *           + (dvxdz+dvzdx)_fwd · (dvxdz+dvzdx)_rcv  (shear, Txz→P)
 *   H_ρρ += ∂ₜvx_fwd·∂ₜvx_rcv + ∂ₜvz_fwd·∂ₜvz_rcv  (Vx/Vz→P)
 *
 * Off-diagonal cross-products (all 6) also computed when arrays provided.
 *
 * Absolute value is taken by the caller AFTER all timesteps.
 ***********************************************************************/
void accumHessianP4(modPar *mod, bndPar *bnd,
                    float *fwd_vx, float *fwd_vz,
                    float *fwd_vx_prev, float *fwd_vz_prev,
                    float *rcv_vx, float *rcv_vz,
                    float *rcv_vx_prev, float *rcv_vz_prev,
                    float dt,
                    float *hess_lam, float *hess_muu, float *hess_rho,
                    float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho)
{
	int ix, iz, n1;
	int ibPx, iePx, ibPz, iePz;
	int ibTx, ieTx, ibTz, ieTz;
	int ibVx_x, ieVx_x, ibVx_z, ieVx_z;
	int ibVz_x, ieVz_x, ibVz_z, ieVz_z;
	float sdx, sdz, sdt;
	float c1, c2, c3, c4;

	if (mod->ischeme <= 2) return;
	if (!fwd_vx || !fwd_vz || !rcv_vx || !rcv_vz) return;

	n1  = mod->naz;
	sdx = 1.0f / mod->dx;
	sdz = 1.0f / mod->dz;
	sdt = 1.0f / dt;

	/* FD coefficients (same as accumGradient) */
	c3 = c4 = 0.0f;
	switch (mod->iorder) {
		case 4:  c1 = 9.0f/8.0f;    c2 = -1.0f/24.0f;   break;
		case 6:  c1 = 75.0f/64.0f;  c2 = -25.0f/384.0f;  c3 = 3.0f/640.0f; break;
		case 8:  c1 = 1225.0f/1024.0f; c2 = -245.0f/3072.0f;
		         c3 = 49.0f/5120.0f; c4 = -5.0f/7168.0f;  break;
		default: c1 = 9.0f/8.0f;    c2 = -1.0f/24.0f;   break;
	}

	ibPx = mod->ioPx; iePx = mod->iePx;
	ibPz = mod->ioPz; iePz = mod->iePz;
	ibTx = mod->ioTx; ieTx = mod->ieTx;
	ibTz = mod->ioTz; ieTz = mod->ieTz;
	ibVx_x = mod->ioXx; ieVx_x = mod->ieXx;
	ibVx_z = mod->ioXz; ieVx_z = mod->ieXz;
	ibVz_x = mod->ioZx; ieVz_x = mod->ieZx;
	ibVz_z = mod->ioZz; ieVz_z = mod->ieZz;

	/* ================================================================
	 * Lambda and Mu (normal) at P grid.
	 * K_λ = div(v),  K̃_λ = div(ṽ)
	 * K_μ_normal = [2·dvx/dx, 2·dvz/dz] (for both fwd and rcv)
	 * ================================================================ */
	if (hess_lam || hess_muu) {
		for (ix = ibPx; ix < iePx; ix++) {
			for (iz = ibPz; iz < iePz; iz++) {
				int ig = ix*n1+iz;

				/* Forward velocity derivatives */
				float dvxdx_f = sdx*(c1*(fwd_vx[(ix+1)*n1+iz]-fwd_vx[ig])
				                    +c2*(fwd_vx[(ix+2)*n1+iz]-fwd_vx[(ix-1)*n1+iz]));
				float dvzdz_f = sdz*(c1*(fwd_vz[ig+1]-fwd_vz[ig])
				                    +c2*(fwd_vz[ig+2]-fwd_vz[ig-1]));
				if (mod->iorder >= 6) {
					dvxdx_f += sdx*c3*(fwd_vx[(ix+3)*n1+iz]-fwd_vx[(ix-2)*n1+iz]);
					dvzdz_f += sdz*c3*(fwd_vz[ig+3]-fwd_vz[ig-2]);
				}
				if (mod->iorder >= 8) {
					dvxdx_f += sdx*c4*(fwd_vx[(ix+4)*n1+iz]-fwd_vx[(ix-3)*n1+iz]);
					dvzdz_f += sdz*c4*(fwd_vz[ig+4]-fwd_vz[ig-3]);
				}

				/* Receiver velocity derivatives */
				float dvxdx_r = sdx*(c1*(rcv_vx[(ix+1)*n1+iz]-rcv_vx[ig])
				                    +c2*(rcv_vx[(ix+2)*n1+iz]-rcv_vx[(ix-1)*n1+iz]));
				float dvzdz_r = sdz*(c1*(rcv_vz[ig+1]-rcv_vz[ig])
				                    +c2*(rcv_vz[ig+2]-rcv_vz[ig-1]));
				if (mod->iorder >= 6) {
					dvxdx_r += sdx*c3*(rcv_vx[(ix+3)*n1+iz]-rcv_vx[(ix-2)*n1+iz]);
					dvzdz_r += sdz*c3*(rcv_vz[ig+3]-rcv_vz[ig-2]);
				}
				if (mod->iorder >= 8) {
					dvxdx_r += sdx*c4*(rcv_vx[(ix+4)*n1+iz]-rcv_vx[(ix-3)*n1+iz]);
					dvzdz_r += sdz*c4*(rcv_vz[ig+4]-rcv_vz[ig-3]);
				}

				float div_f = dvxdx_f + dvzdz_f;
				float div_r = dvxdx_r + dvzdz_r;

				if (hess_lam)
					hess_lam[ig] += 2.0f * div_f * div_r;
				if (hess_muu)
					hess_muu[ig] += 4.0f * (dvxdx_f*dvxdx_r + dvzdz_f*dvzdz_r);

				/* Off-diagonals (non-trivial for P4 since fwd ≠ rcv) */
				if (hess_lam_muu)
					hess_lam_muu[ig] += div_f * (2.0f*dvxdx_r + 2.0f*dvzdz_r)
					                  + div_r * (2.0f*dvxdx_f + 2.0f*dvzdz_f);
				/* H_λρ and H_μρ cross stress×velocity: computed below
				 * after we have the time derivatives */
			}
		}
	}

	/* ================================================================
	 * Mu shear part at Txz grid, scattered to P grid.
	 * exz_fwd = dvx/dz + dvz/dx (forward),  exz_rcv (receiver)
	 * ================================================================ */
	if (hess_muu) {
		for (ix = ibTx; ix < ieTx; ix++) {
			for (iz = ibTz; iz < ieTz; iz++) {
				int ig = ix*n1+iz;

				float dvxdz_f = sdz*(c1*(fwd_vx[ig]-fwd_vx[ig-1])
				                    +c2*(fwd_vx[ig+1]-fwd_vx[ig-2]));
				float dvzdx_f = sdx*(c1*(fwd_vz[ig]-fwd_vz[(ix-1)*n1+iz])
				                    +c2*(fwd_vz[(ix+1)*n1+iz]-fwd_vz[(ix-2)*n1+iz]));
				if (mod->iorder >= 6) {
					dvxdz_f += sdz*c3*(fwd_vx[ig+2]-fwd_vx[ig-3]);
					dvzdx_f += sdx*c3*(fwd_vz[(ix+2)*n1+iz]-fwd_vz[(ix-3)*n1+iz]);
				}

				float dvxdz_r = sdz*(c1*(rcv_vx[ig]-rcv_vx[ig-1])
				                    +c2*(rcv_vx[ig+1]-rcv_vx[ig-2]));
				float dvzdx_r = sdx*(c1*(rcv_vz[ig]-rcv_vz[(ix-1)*n1+iz])
				                    +c2*(rcv_vz[(ix+1)*n1+iz]-rcv_vz[(ix-2)*n1+iz]));
				if (mod->iorder >= 6) {
					dvxdz_r += sdz*c3*(rcv_vx[ig+2]-rcv_vx[ig-3]);
					dvzdx_r += sdx*c3*(rcv_vz[(ix+2)*n1+iz]-rcv_vz[(ix-3)*n1+iz]);
				}

				float exz_f = dvxdz_f + dvzdx_f;
				float exz_r = dvxdz_r + dvzdx_r;
				float cross = exz_f * exz_r;

				hess_muu[(ix-1)*n1+iz-1] += 0.25f * cross;
				hess_muu[ix*n1+iz-1]     += 0.25f * cross;
				hess_muu[(ix-1)*n1+iz]   += 0.25f * cross;
				hess_muu[ig]             += 0.25f * cross;
			}
		}
	}

	/* ================================================================
	 * Density at Vx/Vz grids, scattered to P grid.
	 * K_ρ = ∂ₜv,  K̃_ρ = ∂ₜṽ
	 * H_ρρ += ∂ₜvx_fwd·∂ₜvx_rcv + ∂ₜvz_fwd·∂ₜvz_rcv
	 * ================================================================ */
	if (hess_rho && fwd_vx_prev && fwd_vz_prev && rcv_vx_prev && rcv_vz_prev) {
		/* Vx contribution */
		for (ix = ibVx_x; ix < ieVx_x; ix++) {
			for (iz = ibVx_z; iz < ieVx_z; iz++) {
				int ig = ix*n1+iz;
				float dvx_dt_f = (fwd_vx[ig] - fwd_vx_prev[ig]) * sdt;
				float dvx_dt_r = (rcv_vx[ig] - rcv_vx_prev[ig]) * sdt;
				float cross = dvx_dt_f * dvx_dt_r;
				hess_rho[(ix-1)*n1+iz] += 0.5f * cross;
				hess_rho[ig]           += 0.5f * cross;
			}
		}
		/* Vz contribution */
		for (ix = ibVz_x; ix < ieVz_x; ix++) {
			for (iz = ibVz_z; iz < ieVz_z; iz++) {
				int ig = ix*n1+iz;
				float dvz_dt_f = (fwd_vz[ig] - fwd_vz_prev[ig]) * sdt;
				float dvz_dt_r = (rcv_vz[ig] - rcv_vz_prev[ig]) * sdt;
				float cross = dvz_dt_f * dvz_dt_r;
				hess_rho[ig-1] += 0.5f * cross;
				hess_rho[ig]   += 0.5f * cross;
			}
		}
	}
}
