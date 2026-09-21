#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"

/*********************************************************************
 *
 * acoustic4_adj - TRUE 4th-order acoustic adjoint FD kernel.
 *
 * Implements the mathematically correct discrete adjoint of acoustic4.c.
 * Derived by transposing the discrete spatial operator using the
 * Summation By Parts (SBP) property: (D-)^T = D+, (D+)^T = D-.
 *
 * KEY DIFFERENCES from the forward operator:
 *
 *   1. All updates use += instead of -=
 *   2. Material parameters are INSIDE the spatial derivative
 *      (pre-multiply the field before differentiating)
 *   3. Phase ordering is reversed (velocity update first, then pressure)
 *
 * Forward (acoustic4.c):
 *   Phase F1:  vx -= rox * D-x(p)              velocity update
 *              vz -= roz * D-z(p)
 *   Phase F2:  p  -= l2m * [D+x(vx) + D+z(vz)] pressure update
 *
 * TRUE ADJOINT (this file):
 *   Phase (F2)^T:  vx += D-x(l2m * p)          adjoint velocity update
 *                  vz += D-z(l2m * p)
 *   Phase (F1)^T:  p  += D+x(rox * vx) + D+z(roz * vz)  adjoint pressure
 *
 * The field 'p' here is stored in the tzz slot of the wavefield
 * structure (wflPar->tzz) for consistency with the elastic code.
 *
 * FD coefficients: L2-optimized (same as acoustic4.c forward):
 *   c1 = 1.129042,  c2 = -0.04301412
 *
 * AUTHOR:
 *   Discrete adjoint of acoustic4.c by Jan Thorbecke (TU Delft).
 *   Adjoint derivation for acoustic FWI dot product test.
 *
 **********************************************************************/

int applyAdjointSource(modPar mod, adjSrcPar adj, int itime,
	float *vx, float *vz, float *tzz, float *txx, float *txz,
	float *mul, int rec_delay, int rec_skipdt, int phase, int verbose);

int boundariesP_adj(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz,
	float *txx, float *txz, float *rox, float *roz, float *l2m,
	float *lam, float *mul, int itime, int verbose);

int boundariesV_adj(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz,
	float *txx, float *txz, float *rox, float *roz, float *l2m,
	float *lam, float *mul, int itime, int verbose);


int acoustic4_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
	float *vx, float *vz, float *p,
	float *rox, float *roz, float *l2m,
	int rec_delay, int rec_skipdt, int verbose)
{
	float c1, c2;
	int   ix, iz;
	int   n1;

	/* L2-optimized coefficients (must match forward acoustic4.c) */
	c1 = 1.129042;
	c2 = -0.04301412;
	n1 = mod.naz;

	/* ============================================================ */
	/*  Step 6^T: Adjoint free surface (boundariesV_adj)            */
	/*  Must act BEFORE the stencil reads p at surface points.      */
	/*  Pass NULL for txx, txz, lam, mul (acoustic only uses p).    */
	/* ============================================================ */
	boundariesV_adj(mod, bnd, vx, vz, p, NULL, NULL,
		rox, roz, l2m, NULL, NULL, itime, verbose);

	/* ============================================================ */
	/*  Step 5^T: Inject adjoint stress/pressure sources            */
	/*  Phase 2: pressure residuals injected into p (=tzz slot).    */
	/* ============================================================ */
	applyAdjointSource(mod, adj, itime, vx, vz, p, NULL, NULL,
		NULL, rec_delay, rec_skipdt, /*phase=*/2, verbose);

	/* ============================================================ */
	/*  Phase (F2)^T: Adjoint velocity update                       */
	/*                                                               */
	/*  Forward: p -= l2m * (D+x(vx) + D+z(vz))                    */
	/*  Adjoint: vx += D-x(l2m * p)                                 */
	/*           vz += D-z(l2m * p)                                 */
	/*                                                               */
	/*  l2m is INSIDE the D- derivative (true discrete adjoint).    */
	/*  D-x at Vx grid: c1*(f[ix]-f[ix-1]) + c2*(f[ix+1]-f[ix-2]) */
	/*  where f = l2m * p at P grid.                                */
	/* ============================================================ */
#pragma omp for private (ix, iz) nowait schedule(guided,1)
	for (ix=mod.ioXx; ix<mod.ieXx; ix++) {
#pragma simd
		for (iz=mod.ioXz; iz<mod.ieXz; iz++) {
			vx[ix*n1+iz] +=
				c1*(l2m[ix*n1+iz]    *p[ix*n1+iz]     - l2m[(ix-1)*n1+iz]*p[(ix-1)*n1+iz]) +
				c2*(l2m[(ix+1)*n1+iz]*p[(ix+1)*n1+iz]  - l2m[(ix-2)*n1+iz]*p[(ix-2)*n1+iz]);
		}
	}

	/* vz += D-z(l2m * p) */
#pragma omp for private (ix, iz) schedule(guided,1)
	for (ix=mod.ioZx; ix<mod.ieZx; ix++) {
#pragma simd
		for (iz=mod.ioZz; iz<mod.ieZz; iz++) {
			vz[ix*n1+iz] +=
				c1*(l2m[ix*n1+iz]  *p[ix*n1+iz]     - l2m[ix*n1+iz-1]*p[ix*n1+iz-1]) +
				c2*(l2m[ix*n1+iz+1]*p[ix*n1+iz+1]   - l2m[ix*n1+iz-2]*p[ix*n1+iz-2]);
		}
	}

	/* ============================================================ */
	/*  Step 3^T: Adjoint velocity boundaries (boundariesP_adj)     */
	/* ============================================================ */
	boundariesP_adj(mod, bnd, vx, vz, p, NULL, NULL,
		rox, roz, l2m, NULL, NULL, itime, verbose);

	/* ============================================================ */
	/*  Step 2^T: Inject adjoint force sources                      */
	/*  Phase 1: Fx/Fz residuals injected into vx/vz.              */
	/* ============================================================ */
	applyAdjointSource(mod, adj, itime, vx, vz, p, NULL, NULL,
		NULL, rec_delay, rec_skipdt, /*phase=*/1, verbose);

	/* ============================================================ */
	/*  Phase (F1)^T: Adjoint pressure update                       */
	/*                                                               */
	/*  Forward: vx -= rox * D-x(p),  vz -= roz * D-z(p)           */
	/*  Adjoint: p += D+x(rox * vx) + D+z(roz * vz)               */
	/*                                                               */
	/*  Buoyancy (rox, roz) INSIDE the D+ derivative.               */
	/*  D+x at P grid: c1*(f[ix+1]-f[ix]) + c2*(f[ix+2]-f[ix-1])  */
	/*  where f = rox * vx at Vx grid.                             */
	/* ============================================================ */
#pragma omp for private (ix, iz) schedule(guided,1)
	for (ix=mod.ioPx; ix<mod.iePx; ix++) {
#pragma simd
		for (iz=mod.ioPz; iz<mod.iePz; iz++) {
			p[ix*n1+iz] +=
				/* D+x(rox * vx) */
				c1*(rox[(ix+1)*n1+iz]*vx[(ix+1)*n1+iz] - rox[ix*n1+iz]*vx[ix*n1+iz]) +
				c2*(rox[(ix+2)*n1+iz]*vx[(ix+2)*n1+iz] - rox[(ix-1)*n1+iz]*vx[(ix-1)*n1+iz]) +
				/* D+z(roz * vz) */
				c1*(roz[ix*n1+iz+1]*vz[ix*n1+iz+1]     - roz[ix*n1+iz]*vz[ix*n1+iz]) +
				c2*(roz[ix*n1+iz+2]*vz[ix*n1+iz+2]     - roz[ix*n1+iz-1]*vz[ix*n1+iz-1]);
		}
	}

	return 0;
}
