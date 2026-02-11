#include<stdlib.h>
#include<stdio.h>
#include<math.h>
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
 ***********************************************************************/
void accumGradient(modPar *mod, bndPar *bnd,
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
	float c1, c2, c3, c4;
	int half;

	/* Only for elastic (ischeme > 2) */
	if (mod->ischeme <= 2) return;

	n1  = mod->naz;
	nax = mod->nax;
	sdx = 1.0f / mod->dx;
	sdz = 1.0f / mod->dz;

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
	if (grad_lam) {
		for (ix = ibPx; ix < iePx; ix++) {
			for (iz = ibPz; iz < iePz; iz++) {
				float dvxdx_f, dvzdz_f;

				/* Forward velocity divergence at P grid (order-dependent) */
				dvxdx_f = sdx*(c1*(fwd_vx[(ix+1)*n1+iz]-fwd_vx[ix*n1+iz])
				              +c2*(fwd_vx[(ix+2)*n1+iz]-fwd_vx[(ix-1)*n1+iz]));
				dvzdz_f = sdz*(c1*(fwd_vz[ix*n1+iz+1]-fwd_vz[ix*n1+iz])
				              +c2*(fwd_vz[ix*n1+iz+2]-fwd_vz[ix*n1+iz-1]));
				if (mod->iorder >= 6) {
					dvxdx_f += sdx*c3*(fwd_vx[(ix+3)*n1+iz]-fwd_vx[(ix-2)*n1+iz]);
					dvzdz_f += sdz*c3*(fwd_vz[ix*n1+iz+3]-fwd_vz[ix*n1+iz-2]);
				}
				if (mod->iorder >= 8) {
					dvxdx_f += sdx*c4*(fwd_vx[(ix+4)*n1+iz]-fwd_vx[(ix-3)*n1+iz]);
					dvzdz_f += sdz*c4*(fwd_vz[ix*n1+iz+4]-fwd_vz[ix*n1+iz-3]);
				}

				grad_lam[ix*n1+iz] += dt*(wfl_adj->txx[ix*n1+iz] + wfl_adj->tzz[ix*n1+iz])
				                        *(dvxdx_f + dvzdz_f);
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
	if (grad_muu) {
		/* --- Part 1: Normal-stress at P grid (same bounds as lambda) --- */
		for (ix = ibPx; ix < iePx; ix++) {
			for (iz = ibPz; iz < iePz; iz++) {
				float dvxdx_f, dvzdz_f;

				dvxdx_f = sdx*(c1*(fwd_vx[(ix+1)*n1+iz]-fwd_vx[ix*n1+iz])
				              +c2*(fwd_vx[(ix+2)*n1+iz]-fwd_vx[(ix-1)*n1+iz]));
				dvzdz_f = sdz*(c1*(fwd_vz[ix*n1+iz+1]-fwd_vz[ix*n1+iz])
				              +c2*(fwd_vz[ix*n1+iz+2]-fwd_vz[ix*n1+iz-1]));
				if (mod->iorder >= 6) {
					dvxdx_f += sdx*c3*(fwd_vx[(ix+3)*n1+iz]-fwd_vx[(ix-2)*n1+iz]);
					dvzdz_f += sdz*c3*(fwd_vz[ix*n1+iz+3]-fwd_vz[ix*n1+iz-2]);
				}
				if (mod->iorder >= 8) {
					dvxdx_f += sdx*c4*(fwd_vx[(ix+4)*n1+iz]-fwd_vx[(ix-3)*n1+iz]);
					dvzdz_f += sdz*c4*(fwd_vz[ix*n1+iz+4]-fwd_vz[ix*n1+iz-3]);
				}

				grad_muu[ix*n1+iz] += dt*2.0f*(wfl_adj->txx[ix*n1+iz]*dvxdx_f
				                              + wfl_adj->tzz[ix*n1+iz]*dvzdz_f);
			}
		}

		/* --- Part 2: Shear at Txz grid, scattered to P grid ---
		 * Compute ψ_txz·(D-z(vx)+D-x(vz)) at native Txz grid, then
		 * scatter to the 4 surrounding P-grid points with weight 0.25.
		 * This is the transpose of the harmonic averaging in readModel.c
		 * (approximated with arithmetic weights).
		 * Index mapping: Txz(ix,iz) → P-grid (ix-1,iz-1),(ix,iz-1),(ix-1,iz),(ix,iz)
		 * since ioTx = ioPx+1, ioTz = ioPz+1. */
		for (ix = ibTx; ix < ieTx; ix++) {
			for (iz = ibTz; iz < ieTz; iz++) {
				float dvxdz_f, dvzdx_f, shear;

				dvxdz_f = sdz*(c1*(fwd_vx[ix*n1+iz]-fwd_vx[ix*n1+iz-1])
				              +c2*(fwd_vx[ix*n1+iz+1]-fwd_vx[ix*n1+iz-2]));
				dvzdx_f = sdx*(c1*(fwd_vz[ix*n1+iz]-fwd_vz[(ix-1)*n1+iz])
				              +c2*(fwd_vz[(ix+1)*n1+iz]-fwd_vz[(ix-2)*n1+iz]));
				if (mod->iorder >= 6) {
					dvxdz_f += sdz*c3*(fwd_vx[ix*n1+iz+2]-fwd_vx[ix*n1+iz-3]);
					dvzdx_f += sdx*c3*(fwd_vz[(ix+2)*n1+iz]-fwd_vz[(ix-3)*n1+iz]);
				}
				if (mod->iorder >= 8) {
					dvxdz_f += sdz*c4*(fwd_vx[ix*n1+iz+3]-fwd_vx[ix*n1+iz-4]);
					dvzdx_f += sdx*c4*(fwd_vz[(ix+3)*n1+iz]-fwd_vz[(ix-4)*n1+iz]);
				}

				shear = dt*wfl_adj->txz[ix*n1+iz]*(dvxdz_f + dvzdx_f);

				/* Scatter to 4 surrounding P-grid points */
				grad_muu[(ix-1)*n1+iz-1] += 0.25f * shear;
				grad_muu[ix*n1+iz-1]     += 0.25f * shear;
				grad_muu[(ix-1)*n1+iz]   += 0.25f * shear;
				grad_muu[ix*n1+iz]       += 0.25f * shear;
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
	if (grad_rho && fwd_vx_prev && fwd_vz_prev) {
		float dvx_dt, dvz_dt;
		float sdt = 1.0f / dt;
		float *rho = mod->rho;

		/* Vx contribution, scattered to P grid */
		for (ix = ibVx_x; ix < ieVx_x; ix++) {
			for (iz = ibVx_z; iz < ieVx_z; iz++) {
				float vx_contrib;
				dvx_dt = (fwd_vx[ix*n1+iz] - fwd_vx_prev[ix*n1+iz]) * sdt;
				vx_contrib = dt * wfl_adj->vx[ix*n1+iz] * dvx_dt;
				/* Vx(ix,iz) straddles P(ix-1,iz) and P(ix,iz) */
				grad_rho[(ix-1)*n1+iz] += 0.5f * vx_contrib / rho[(ix-1)*n1+iz];
				grad_rho[ix*n1+iz]     += 0.5f * vx_contrib / rho[ix*n1+iz];
			}
		}
		/* Vz contribution, scattered to P grid */
		for (ix = ibVz_x; ix < ieVz_x; ix++) {
			for (iz = ibVz_z; iz < ieVz_z; iz++) {
				float vz_contrib;
				dvz_dt = (fwd_vz[ix*n1+iz] - fwd_vz_prev[ix*n1+iz]) * sdt;
				vz_contrib = dt * wfl_adj->vz[ix*n1+iz] * dvz_dt;
				/* Vz(ix,iz) straddles P(ix,iz-1) and P(ix,iz) */
				grad_rho[ix*n1+(iz-1)] += 0.5f * vz_contrib / rho[ix*n1+(iz-1)];
				grad_rho[ix*n1+iz]     += 0.5f * vz_contrib / rho[ix*n1+iz];
			}
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
