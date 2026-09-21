#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"

/**
 * acoustic_gradient.c -- Acoustic FWI gradient imaging condition.
 *
 * Contains:
 *   accumGradientAcoustic()      - per-timestep gradient cross-correlation
 *   convertGradientAcoustic()    - kappa/rho -> Vp/rho chain rule
 *
 * Gradient formulas (acoustic velocity-pressure formulation):
 *
 *   g_kappa = +Integral psi_p * div(v_fwd) dt
 *   g_rho   = +(1/rho) * Integral (psi_vx * dvx/dt + psi_vz * dvz/dt) dt
 *
 * where kappa = l2m = rho * Vp^2 (bulk modulus).
 *
 * The kappa gradient uses the same D+x/D+z stencils as the forward
 * pressure update in acoustic4.c, ensuring discrete consistency.
 *
 * The density gradient uses the time derivative of particle velocity,
 * scattered from staggered Vx/Vz grids to the P grid with weight 0.5.
 *
 * Shin pseudo-Hessian (diagonal approximation):
 *   H_kappa = Sum_src Sum_t [div(v_fwd)]^2
 *   H_rho   = Sum_src Sum_t [(dvx/dt)^2 + (dvz/dt)^2]
 *
 * Chain rule kappa -> Vp:
 *   g_Vp  = g_kappa * 2 * rho * Vp
 *   g_rho = g_rho_direct + g_kappa * Vp^2
 */

#define MAX(x,y) ((x) > (y) ? (x) : (y))


/***********************************************************************
 * accumGradientAcoustic -- Cross-correlate forward/adjoint wavefields.
 *
 * Accumulates gradient contributions for one time step.
 * Called at each time step during adjoint backpropagation in adj_shot.c.
 *
 * The adjoint pressure psi_p is stored in wfl_adj->tzz (field alias
 * for the acoustic case, same as forward where p=tzz).
 *
 * Parameters:
 *   fwd_vx, fwd_vz         - forward velocity at current time step
 *   fwd_vx_prev, fwd_vz_prev - forward velocity at previous time step
 *                               (for dv/dt in density gradient)
 *   wfl_adj                 - adjoint wavefield (psi_p in tzz, psi_vx/vz)
 *   dt                      - time step size
 *   grad_l2m                - kappa gradient (accumulated, padded grid)
 *   grad_rho                - density gradient (accumulated, padded grid)
 *   hess_l2m                - Shin pseudo-Hessian for kappa (may be NULL)
 *   hess_rho                - Shin pseudo-Hessian for rho (may be NULL)
 *   wfld_energy             - forward wavefield energy (may be NULL)
 ***********************************************************************/
void accumGradientAcoustic(modPar *mod, bndPar *bnd,
	float *fwd_vx, float *fwd_vz,
	float *fwd_vx_prev, float *fwd_vz_prev,
	wflPar *wfl_adj, float dt,
	float *grad_l2m, float *grad_rho,
	float *hess_l2m, float *hess_rho,
	float *wfld_energy)
{
	int ix, iz, n1, nax;
	int ibPx, iePx, ibPz, iePz;
	float sdx, sdz;
	float c1, c2;
	int half;

	n1  = mod->naz;
	nax = mod->nax;
	sdx = 1.0f / mod->dx;
	sdz = 1.0f / mod->dz;

	/* L2-optimized FD coefficients (must match acoustic4.c forward) */
	c1 = 1.129042f;
	c2 = -0.04301412f;
	half = 2; /* 4th order / 2 */

	/* ================================================================
	 * Compute safe loop bounds at P grid.
	 * Start with kernel loop bounds, then adjust for:
	 *   1. Absorbing boundaries: skip ntap points
	 *   2. Free surface: skip 1 point below surface
	 *   3. FD stencil requirements
	 * ================================================================ */
	ibPx = mod->ioPx;
	iePx = mod->iePx;
	ibPz = mod->ioPz;
	iePz = mod->iePz;

	/* Absorbing boundary adjustments */
	if (bnd->lef == 4 || bnd->lef == 2) ibPx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) iePx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibPz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) iePz -= bnd->ntap;

	/* Free surface: skip gradient at surface row */
	if (bnd->top == 1) ibPz = MAX(ibPz, mod->ioPz + 1);

	/* FD stencil safety: D+x(vx) accesses ix+1,ix+2,ix,ix-1
	 * Need ibPx >= 1 and iePx <= nax - 2 */
	ibPx = MAX(ibPx, half - 1);
	iePx = MAX(ibPx, iePx);
	if (iePx > nax - half) iePx = nax - half;
	ibPz = MAX(ibPz, half - 1);
	iePz = MAX(ibPz, iePz);
	if (iePz > n1 - half) iePz = n1 - half;

	/* ================================================================
	 * Kappa (l2m) gradient at P grid
	 *
	 *   g_kappa[ix,iz] += dt * psi_p[ix,iz] * div(v_fwd)[ix,iz]
	 *
	 * div(v_fwd) = D+x(vx_fwd) + D+z(vz_fwd)
	 * Same D+x/D+z stencils as forward pressure update in acoustic4.c.
	 * ================================================================ */
	if (grad_l2m || hess_l2m) {
		for (ix = ibPx; ix < iePx; ix++) {
			for (iz = ibPz; iz < iePz; iz++) {
				float dvxdx_f, dvzdz_f, div_f;
				int ig = ix*n1+iz;

				/* Forward velocity divergence (L2-optimized stencil) */
				dvxdx_f = sdx*(c1*(fwd_vx[(ix+1)*n1+iz] - fwd_vx[ig])
				              +c2*(fwd_vx[(ix+2)*n1+iz] - fwd_vx[(ix-1)*n1+iz]));
				dvzdz_f = sdz*(c1*(fwd_vz[ig+1] - fwd_vz[ig])
				              +c2*(fwd_vz[ig+2] - fwd_vz[ig-1]));
				div_f = dvxdx_f + dvzdz_f;

				/* Kappa gradient: psi_p * div(v_fwd) */
				if (grad_l2m)
					grad_l2m[ig] += dt * wfl_adj->tzz[ig] * div_f;

				/* Shin pseudo-Hessian for kappa: |div(v)|^2 */
				if (hess_l2m)
					hess_l2m[ig] += div_f * div_f;
			}
		}
	}

	/* ================================================================
	 * Density gradient
	 *
	 *   g_rho = (1/rho) * Integral psi_v . dv/dt dt
	 *
	 * Time derivative: dv/dt approx (v[t] - v[t-dt]) / dt
	 * Native at Vx and Vz grids, scattered to P grid with weight 0.5.
	 *
	 * Uses same loop bounds as the Vx/Vz arrays.
	 * ================================================================ */
	if ((grad_rho || hess_rho) && fwd_vx_prev && fwd_vz_prev) {
		float sdt = 1.0f / dt;
		float *rho = mod->rho;

		/* Vx contribution, scattered to 2 P-grid neighbors */
		for (ix = mod->ioXx; ix < mod->ieXx; ix++) {
			for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
				int ig = ix*n1+iz;
				float dvx_dt = (fwd_vx[ig] - fwd_vx_prev[ig]) * sdt;

				if (grad_rho) {
					float vx_contrib = dt * wfl_adj->vx[ig] * dvx_dt;
					grad_rho[(ix-1)*n1+iz] += 0.5f * vx_contrib / rho[(ix-1)*n1+iz];
					grad_rho[ig]           += 0.5f * vx_contrib / rho[ig];
				}
				if (hess_rho) {
					float dvx_dt_sq = dvx_dt * dvx_dt;
					hess_rho[(ix-1)*n1+iz] += 0.5f * dvx_dt_sq;
					hess_rho[ig]           += 0.5f * dvx_dt_sq;
				}
			}
		}

		/* Vz contribution, scattered to 2 P-grid neighbors */
		for (ix = mod->ioZx; ix < mod->ieZx; ix++) {
			for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
				int ig = ix*n1+iz;
				float dvz_dt = (fwd_vz[ig] - fwd_vz_prev[ig]) * sdt;

				if (grad_rho) {
					float vz_contrib = dt * wfl_adj->vz[ig] * dvz_dt;
					grad_rho[ig-1] += 0.5f * vz_contrib / rho[ig-1];
					grad_rho[ig]   += 0.5f * vz_contrib / rho[ig];
				}
				if (hess_rho) {
					float dvz_dt_sq = dvz_dt * dvz_dt;
					hess_rho[ig-1] += 0.5f * dvz_dt_sq;
					hess_rho[ig]   += 0.5f * dvz_dt_sq;
				}
			}
		}
	}

	/* ================================================================
	 * Forward wavefield energy: Ws += vx^2 + vz^2
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
 * convertGradientAcoustic -- Chain rule: kappa/rho -> Vp/rho.
 *
 * Converts kappa-parameter gradient to velocity-parameter gradient
 * in-place.
 *
 * Chain rule (kappa = rho * Vp^2):
 *   g_Vp  = g_kappa * 2 * rho * Vp    (d_kappa/d_Vp = 2*rho*Vp)
 *   g_rho = g_rho_direct + g_kappa * Vp^2  (d_kappa/d_rho = Vp^2)
 *
 * Parameters:
 *   grad_l2m - INPUT: g_kappa,  OUTPUT: g_Vp  (in-place)
 *   grad_rho - INPUT: g_rho,    OUTPUT: g_rho_full  (in-place, may be NULL)
 *   cp       - Vp array (padded grid)
 *   rho      - density array (padded grid)
 *   sizem    - total padded grid size (nax * naz)
 ***********************************************************************/
void convertGradientAcoustic(float *grad_l2m, float *grad_rho,
	float *cp, float *rho, size_t sizem)
{
	size_t i;

	for (i = 0; i < sizem; i++) {
		float g_kappa = grad_l2m ? grad_l2m[i] : 0.0f;
		float vp  = cp[i];
		float vp2 = vp * vp;
		float rho_val = rho[i];

		/* g_Vp = g_kappa * 2*rho*Vp */
		if (grad_l2m)
			grad_l2m[i] = g_kappa * 2.0f * rho_val * vp;

		/* g_rho_full = g_rho_direct + g_kappa * Vp^2 */
		if (grad_rho)
			grad_rho[i] += g_kappa * vp2;
	}
}
