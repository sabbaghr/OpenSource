/*
 * born_vsrc.c - Virtual source injection for Born (mu_1) wavefield.
 *
 * Implements the (dA(m)/dm . dm) operator applied to the forward
 * wavefield u, generating the distributed source for the Born/scattered
 * wavefield mu_1 used in the Gauss-Newton Hessian-vector product.
 *
 * Two injection points per time step:
 *
 *   1. Phase F1 virtual source (velocity, from density perturbation):
 *      born_vx -= delta_rox * [D-x(txx_fwd) + D+z(txz_fwd)]
 *      born_vz -= delta_roz * [D+x(txz_fwd) + D-z(tzz_fwd)]
 *
 *   2. Phase F2 virtual source (stress, from stiffness perturbation):
 *      born_txx -= delta_l2m * D+x(vx_fwd) + delta_lam * D+z(vz_fwd)
 *      born_tzz -= delta_lam * D+x(vx_fwd) + delta_l2m * D+z(vz_fwd)
 *      born_txz -= delta_mul * [D-z(vx_fwd) + D-x(vz_fwd)]
 *
 * Stencils, loop bounds, and coefficients match elastic4/6/8.c exactly.
 *
 * Time-step ordering in born_shot:
 *   1. inject_born_vsrc_vel(sigma_fwd^n -> born_v)    [BEFORE callKernel]
 *   2. callKernel(fwd, with source)                    [advance forward]
 *   3. callKernel(born, no source)                     [advance born]
 *   4. inject_born_vsrc_stress(v_fwd^{n+1} -> born_s) [AFTER callKernel]
 *
 * Step 1 must use sigma_fwd^n (before forward step), so Phase F2 of the
 * born step (inside callKernel) sees the correct born_v with the virtual
 * source contribution.
 *
 * Step 4 uses v_fwd^{n+1} (after forward step) for the Phase F2 virtual
 * source, injected after the born step to avoid second-order cross-terms.
 *
 * See HESSIAN_MATH.md for the full derivation.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fdelfwi.h"


/***********************************************************************
 * inject_born_vsrc_vel -- Phase F1 virtual source (velocity).
 *
 * Injects the density-perturbation virtual source into the Born
 * velocity fields.  Uses the same stencils as elastic4/6/8 Phase F1,
 * applied to the FORWARD stress field, scaled by delta_rox/delta_roz.
 *
 * Call BEFORE callKernel(born) and BEFORE callKernel(fwd), so that:
 *   - Forward stress is still at time n (not yet advanced)
 *   - Born Phase F2 (inside callKernel) sees the correct born_v
 *
 * Must be called inside an OMP parallel region.
 *
 * Parameters:
 *   mod        - model parameters (grid dims, loop bounds, iorder)
 *   txx_fwd, tzz_fwd, txz_fwd - forward stress at current time n
 *   delta_rox, delta_roz       - perturbed buoyancy FD coefficients
 *   born_vx, born_vz           - Born velocity (modified in-place)
 ***********************************************************************/
void inject_born_vsrc_vel(modPar *mod,
                          float *txx_fwd, float *tzz_fwd, float *txz_fwd,
                          float *delta_rox, float *delta_roz,
                          float *born_vx, float *born_vz)
{
	float c1, c2, c3, c4;
	int   ix, iz, n1;

	n1 = mod->naz;
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

	/* born_vx -= delta_rox * [D-x(txx_fwd) + D+z(txz_fwd)] */
#pragma omp for private(ix, iz) nowait schedule(guided,1)
	for (ix = mod->ioXx; ix < mod->ieXx; ix++) {
#pragma simd
		for (iz = mod->ioXz; iz < mod->ieXz; iz++) {
			float Dsig;
			Dsig = c1*(txx_fwd[ix*n1+iz]     - txx_fwd[(ix-1)*n1+iz] +
			           txz_fwd[ix*n1+iz+1]   - txz_fwd[ix*n1+iz])    +
			       c2*(txx_fwd[(ix+1)*n1+iz] - txx_fwd[(ix-2)*n1+iz] +
			           txz_fwd[ix*n1+iz+2]   - txz_fwd[ix*n1+iz-1]);
			if (mod->iorder >= 6)
				Dsig += c3*(txx_fwd[(ix+2)*n1+iz] - txx_fwd[(ix-3)*n1+iz] +
				            txz_fwd[ix*n1+iz+3]   - txz_fwd[ix*n1+iz-2]);
			if (mod->iorder >= 8)
				Dsig += c4*(txx_fwd[(ix+3)*n1+iz] - txx_fwd[(ix-4)*n1+iz] +
				            txz_fwd[ix*n1+iz+4]   - txz_fwd[ix*n1+iz-3]);

			born_vx[ix*n1+iz] -= delta_rox[ix*n1+iz] * Dsig;
		}
	}

	/* born_vz -= delta_roz * [D+x(txz_fwd) + D-z(tzz_fwd)] */
#pragma omp for private(ix, iz) schedule(guided,1)
	for (ix = mod->ioZx; ix < mod->ieZx; ix++) {
#pragma simd
		for (iz = mod->ioZz; iz < mod->ieZz; iz++) {
			float Dsig;
			Dsig = c1*(tzz_fwd[ix*n1+iz]     - tzz_fwd[ix*n1+iz-1]    +
			           txz_fwd[(ix+1)*n1+iz] - txz_fwd[ix*n1+iz])     +
			       c2*(tzz_fwd[ix*n1+iz+1]   - tzz_fwd[ix*n1+iz-2]    +
			           txz_fwd[(ix+2)*n1+iz] - txz_fwd[(ix-1)*n1+iz]);
			if (mod->iorder >= 6)
				Dsig += c3*(tzz_fwd[ix*n1+iz+2]   - tzz_fwd[ix*n1+iz-3]    +
				            txz_fwd[(ix+3)*n1+iz] - txz_fwd[(ix-2)*n1+iz]);
			if (mod->iorder >= 8)
				Dsig += c4*(tzz_fwd[ix*n1+iz+3]   - tzz_fwd[ix*n1+iz-4]    +
				            txz_fwd[(ix+4)*n1+iz] - txz_fwd[(ix-3)*n1+iz]);

			born_vz[ix*n1+iz] -= delta_roz[ix*n1+iz] * Dsig;
		}
	}
}


/***********************************************************************
 * inject_born_vsrc_stress -- Phase F2 virtual source (stress).
 *
 * Injects the stiffness-perturbation virtual source into the Born
 * stress fields.  Uses the same stencils as elastic4/6/8 Phase F2,
 * applied to the FORWARD velocity field, scaled by delta_l2m/lam/mul.
 *
 * Call AFTER callKernel(fwd) and AFTER callKernel(born), so that:
 *   - Forward velocity v_fwd is at time n+1 (already advanced)
 *   - Born stress has completed its elastic4 step
 *
 * Must be called inside an OMP parallel region.
 *
 * Parameters:
 *   mod        - model parameters
 *   vx_fwd, vz_fwd              - forward velocity at time n+1
 *   delta_l2m, delta_lam, delta_mul - perturbed stiffness FD coefficients
 *   born_txx, born_tzz, born_txz    - Born stress (modified in-place)
 ***********************************************************************/
void inject_born_vsrc_stress(modPar *mod, bndPar *bnd,
                             float *vx_fwd, float *vz_fwd,
                             float *delta_l2m, float *delta_lam,
                             float *delta_mul,
                             float *born_txx, float *born_tzz,
                             float *born_txz)
{
	float c1, c2, c3, c4;
	float dvx, dvz;
	int   ix, iz, n1, nax;
	int   ibPx, iePx, ibPz, iePz;
	int   ibTx, ieTx, ibTz, ieTz;
	int   half;

	n1  = mod->naz;
	nax = mod->nax;
	half = mod->iorder / 2;
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

	/* ---- Compute interior loop bounds matching accumGradient ----
	 * For B/B^T duality, the Born virtual source spatial domain must
	 * match the gradient accumulation domain exactly.  accumGradient
	 * excludes the absorbing taper zone and applies FD stencil safety. */

	/* P grid bounds (for txx/tzz, matching lambda/mu-normal gradient) */
	ibPx = mod->ioPx;  iePx = mod->iePx;
	ibPz = mod->ioPz;  iePz = mod->iePz;
	if (bnd->lef == 4 || bnd->lef == 2) ibPx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) iePx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibPz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) iePz -= bnd->ntap;
	if (bnd->top == 1) ibPz = (ibPz > mod->ioPz + 1) ? ibPz : mod->ioPz + 1;
	/* FD stencil safety: D+x accesses ix-(half-1) to ix+half */
	ibPx = (ibPx > half - 1) ? ibPx : half - 1;
	if (iePx > nax - half) iePx = nax - half;
	ibPz = (ibPz > half - 1) ? ibPz : half - 1;
	if (iePz > n1 - half)  iePz = n1 - half;

	/* Txz grid bounds (for txz, matching mu-shear gradient) */
	ibTx = mod->ioTx;  ieTx = mod->ieTx;
	ibTz = mod->ioTz;  ieTz = mod->ieTz;
	if (bnd->lef == 4 || bnd->lef == 2) ibTx += bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) ieTx -= bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibTz += bnd->ntap;
	if (bnd->bot == 4 || bnd->bot == 2) ieTz -= bnd->ntap;
	if (bnd->top == 1) ibTz = (ibTz > mod->ioTz + 1) ? ibTz : mod->ioTz + 1;
	/* FD stencil safety: D-z/D-x access ix-half to ix+(half-1) */
	ibTx = (ibTx > half) ? ibTx : half;
	if (ieTx > nax - half) ieTx = nax - half;
	ibTz = (ibTz > half) ? ibTz : half;
	if (ieTz > n1 - half)  ieTz = n1 - half;

	/* born_txx -= delta_l2m * D+x(vx_fwd) + delta_lam * D+z(vz_fwd)
	 * born_tzz -= delta_lam * D+x(vx_fwd) + delta_l2m * D+z(vz_fwd) */
#pragma omp for private(ix, iz, dvx, dvz) nowait schedule(guided,1)
	for (ix = ibPx; ix < iePx; ix++) {
#pragma simd
		for (iz = ibPz; iz < iePz; iz++) {
			dvx = c1*(vx_fwd[(ix+1)*n1+iz] - vx_fwd[ix*n1+iz]) +
			      c2*(vx_fwd[(ix+2)*n1+iz] - vx_fwd[(ix-1)*n1+iz]);
			dvz = c1*(vz_fwd[ix*n1+iz+1]   - vz_fwd[ix*n1+iz]) +
			      c2*(vz_fwd[ix*n1+iz+2]   - vz_fwd[ix*n1+iz-1]);
			if (mod->iorder >= 6) {
				dvx += c3*(vx_fwd[(ix+3)*n1+iz] - vx_fwd[(ix-2)*n1+iz]);
				dvz += c3*(vz_fwd[ix*n1+iz+3]   - vz_fwd[ix*n1+iz-2]);
			}
			if (mod->iorder >= 8) {
				dvx += c4*(vx_fwd[(ix+4)*n1+iz] - vx_fwd[(ix-3)*n1+iz]);
				dvz += c4*(vz_fwd[ix*n1+iz+4]   - vz_fwd[ix*n1+iz-3]);
			}

			born_txx[ix*n1+iz] -= delta_l2m[ix*n1+iz]*dvx
			                    + delta_lam[ix*n1+iz]*dvz;
			born_tzz[ix*n1+iz] -= delta_lam[ix*n1+iz]*dvx
			                    + delta_l2m[ix*n1+iz]*dvz;
		}
	}

	/* born_txz -= delta_mul * [D-z(vx_fwd) + D-x(vz_fwd)] */
#pragma omp for private(ix, iz) schedule(guided,1)
	for (ix = ibTx; ix < ieTx; ix++) {
#pragma simd
		for (iz = ibTz; iz < ieTz; iz++) {
			float shear;
			shear = c1*(vx_fwd[ix*n1+iz]     - vx_fwd[ix*n1+iz-1] +
			            vz_fwd[ix*n1+iz]     - vz_fwd[(ix-1)*n1+iz]) +
			        c2*(vx_fwd[ix*n1+iz+1]   - vx_fwd[ix*n1+iz-2] +
			            vz_fwd[(ix+1)*n1+iz] - vz_fwd[(ix-2)*n1+iz]);
			if (mod->iorder >= 6)
				shear += c3*(vx_fwd[ix*n1+iz+2]   - vx_fwd[ix*n1+iz-3] +
				             vz_fwd[(ix+2)*n1+iz] - vz_fwd[(ix-3)*n1+iz]);
			if (mod->iorder >= 8)
				shear += c4*(vx_fwd[ix*n1+iz+3]   - vx_fwd[ix*n1+iz-4] +
				             vz_fwd[(ix+3)*n1+iz] - vz_fwd[(ix-4)*n1+iz]);

			born_txz[ix*n1+iz] -= delta_mul[ix*n1+iz] * shear;
		}
	}
}
