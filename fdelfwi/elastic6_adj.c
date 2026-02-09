#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"

/*********************************************************************
 *
 * elastic6_adj - TRUE 6th-order elastic adjoint FD kernel.
 *
 * Implements the mathematically correct discrete adjoint of elastic6.c.
 * See elastic4_adj.c for the derivation and sign conventions.
 *
 * Forward (elastic6.c):
 *   vx -= rox * [D-x(txx) + D+z(txz)]           Step 1 / Phase F1
 *   vz -= roz * [D+x(txz) + D-z(tzz)]           Step 1 / Phase F1
 *   txx -= l2m * D+x(vx) + lam * D+z(vz)        Step 4 / Phase F2
 *   tzz -= lam * D+x(vx) + l2m * D+z(vz)        Step 4 / Phase F2
 *   txz -= mul * [D-z(vx) + D-x(vz)]             Step 4 / Phase F2
 *
 * TRUE ADJOINT (this file):
 *   vx += D-x(l2m*txx) + D-x(lam*tzz) + D+z(mul*txz)   Step 4^T / (F2)^T
 *   vz += D-z(lam*txx) + D-z(l2m*tzz) + D+x(mul*txz)   Step 4^T / (F2)^T
 *   txx += D+x(rox*vx)                                    Step 1^T / (F1)^T
 *   tzz += D+z(roz*vz)                                    Step 1^T / (F1)^T
 *   txz += D-z(rox*vx) + D-x(roz*vz)                     Step 1^T / (F1)^T
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


int elastic6_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
	float *vx, float *vz, float *tzz, float *txx, float *txz,
	float *rox, float *roz, float *l2m, float *lam, float *mul,
	int rec_delay, int rec_skipdt, int verbose)
{
	float c1, c2, c3;
	int   ix, iz;
	int   n1;

	c1 = 75.0/64.0;
	c2 = -25.0/384.0;
	c3 = 3.0/640.0;
	n1 = mod.naz;

	/* ============================================================ */
	/*  Step 6^T: Adjoint free surface (boundariesV_adj)            */
	/*  Must act BEFORE the stencil reads stress at surface points. */
	/* ============================================================ */
	boundariesV_adj(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);

	/* ============================================================ */
	/*  Step 5^T: Inject adjoint stress sources                     */
	/* ============================================================ */
	applyAdjointSource(mod, adj, itime, vx, vz, tzz, txx, txz,
		mul, rec_delay, rec_skipdt, /*phase=*/2, verbose);

	/* ============================================================ */
	/*  Step 4^T / Phase A1 = (F2)^T: Adjoint velocity update      */
	/*  vx += D-x(l2m*txx) + D-x(lam*tzz) + D+z(mul*txz)         */
	/*                                                               */
	/*  Material params (l2m, lam, mul) INSIDE the derivative --    */
	/*  transpose of fwd where they multiply OUTSIDE.                */
	/* ============================================================ */
#pragma omp for private (ix, iz) nowait schedule(guided,1)
	for (ix=mod.ioXx; ix<mod.ieXx; ix++) {
#pragma simd
		for (iz=mod.ioXz; iz<mod.ieXz; iz++) {
			vx[ix*n1+iz] +=
				c1*(l2m[ix*n1+iz]*txx[ix*n1+iz]         - l2m[(ix-1)*n1+iz]*txx[(ix-1)*n1+iz] +
				    lam[ix*n1+iz]*tzz[ix*n1+iz]         - lam[(ix-1)*n1+iz]*tzz[(ix-1)*n1+iz] +
				    mul[ix*n1+iz+1]*txz[ix*n1+iz+1]     - mul[ix*n1+iz]*txz[ix*n1+iz]) +
				c2*(l2m[(ix+1)*n1+iz]*txx[(ix+1)*n1+iz] - l2m[(ix-2)*n1+iz]*txx[(ix-2)*n1+iz] +
				    lam[(ix+1)*n1+iz]*tzz[(ix+1)*n1+iz] - lam[(ix-2)*n1+iz]*tzz[(ix-2)*n1+iz] +
				    mul[ix*n1+iz+2]*txz[ix*n1+iz+2]     - mul[ix*n1+iz-1]*txz[ix*n1+iz-1]) +
				c3*(l2m[(ix+2)*n1+iz]*txx[(ix+2)*n1+iz] - l2m[(ix-3)*n1+iz]*txx[(ix-3)*n1+iz] +
				    lam[(ix+2)*n1+iz]*tzz[(ix+2)*n1+iz] - lam[(ix-3)*n1+iz]*tzz[(ix-3)*n1+iz] +
				    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]);
		}
	}

	/* vz += D-z(lam*txx) + D-z(l2m*tzz) + D+x(mul*txz) */
#pragma omp for private (ix, iz) schedule(guided,1)
	for (ix=mod.ioZx; ix<mod.ieZx; ix++) {
#pragma simd
		for (iz=mod.ioZz; iz<mod.ieZz; iz++) {
			vz[ix*n1+iz] +=
				c1*(lam[ix*n1+iz]*txx[ix*n1+iz]         - lam[ix*n1+iz-1]*txx[ix*n1+iz-1] +
				    l2m[ix*n1+iz]*tzz[ix*n1+iz]         - l2m[ix*n1+iz-1]*tzz[ix*n1+iz-1] +
				    mul[(ix+1)*n1+iz]*txz[(ix+1)*n1+iz] - mul[ix*n1+iz]*txz[ix*n1+iz]) +
				c2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1]     - lam[ix*n1+iz-2]*txx[ix*n1+iz-2] +
				    l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]     - l2m[ix*n1+iz-2]*tzz[ix*n1+iz-2] +
				    mul[(ix+2)*n1+iz]*txz[(ix+2)*n1+iz] - mul[(ix-1)*n1+iz]*txz[(ix-1)*n1+iz]) +
				c3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2]     - lam[ix*n1+iz-3]*txx[ix*n1+iz-3] +
				    l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]     - l2m[ix*n1+iz-3]*tzz[ix*n1+iz-3] +
				    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]);
		}
	}

	/* ============================================================ */
	/*  Step 3^T: Adjoint velocity boundaries (boundariesP_adj)     */
	/* ============================================================ */
	boundariesP_adj(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);

	/* ============================================================ */
	/*  Step 2^T: Inject adjoint force sources                      */
	/* ============================================================ */
	applyAdjointSource(mod, adj, itime, vx, vz, tzz, txx, txz,
		mul, rec_delay, rec_skipdt, /*phase=*/1, verbose);

	/* ============================================================ */
	/*  Step 1^T / Phase A2 = (F1)^T: Adjoint stress update        */
	/*  txx += D+x(rox*vx),  tzz += D+z(roz*vz)                   */
	/*                                                               */
	/*  Buoyancy (rox, roz) INSIDE the derivative -- transpose of   */
	/*  fwd where they multiply OUTSIDE.                             */
	/* ============================================================ */
#pragma omp for private (ix, iz) nowait schedule(guided,1)
	for (ix=mod.ioPx; ix<mod.iePx; ix++) {
#pragma simd
		for (iz=mod.ioPz; iz<mod.iePz; iz++) {
			txx[ix*n1+iz] +=
				c1*(rox[(ix+1)*n1+iz]*vx[(ix+1)*n1+iz] - rox[ix*n1+iz]*vx[ix*n1+iz]) +
				c2*(rox[(ix+2)*n1+iz]*vx[(ix+2)*n1+iz] - rox[(ix-1)*n1+iz]*vx[(ix-1)*n1+iz]) +
				c3*(rox[(ix+3)*n1+iz]*vx[(ix+3)*n1+iz] - rox[(ix-2)*n1+iz]*vx[(ix-2)*n1+iz]);
			tzz[ix*n1+iz] +=
				c1*(roz[ix*n1+iz+1]*vz[ix*n1+iz+1]     - roz[ix*n1+iz]*vz[ix*n1+iz]) +
				c2*(roz[ix*n1+iz+2]*vz[ix*n1+iz+2]     - roz[ix*n1+iz-1]*vz[ix*n1+iz-1]) +
				c3*(roz[ix*n1+iz+3]*vz[ix*n1+iz+3]     - roz[ix*n1+iz-2]*vz[ix*n1+iz-2]);
		}
	}

	/* txz += D-z(rox*vx) + D-x(roz*vz) */
#pragma omp for private (ix, iz) schedule(guided,1)
	for (ix=mod.ioTx; ix<mod.ieTx; ix++) {
#pragma simd
		for (iz=mod.ioTz; iz<mod.ieTz; iz++) {
			txz[ix*n1+iz] +=
				c1*(rox[ix*n1+iz]*vx[ix*n1+iz]         - rox[ix*n1+iz-1]*vx[ix*n1+iz-1] +
				    roz[ix*n1+iz]*vz[ix*n1+iz]         - roz[(ix-1)*n1+iz]*vz[(ix-1)*n1+iz]) +
				c2*(rox[ix*n1+iz+1]*vx[ix*n1+iz+1]     - rox[ix*n1+iz-2]*vx[ix*n1+iz-2] +
				    roz[(ix+1)*n1+iz]*vz[(ix+1)*n1+iz] - roz[(ix-2)*n1+iz]*vz[(ix-2)*n1+iz]) +
				c3*(rox[ix*n1+iz+2]*vx[ix*n1+iz+2]     - rox[ix*n1+iz-3]*vx[ix*n1+iz-3] +
				    roz[(ix+2)*n1+iz]*vz[(ix+2)*n1+iz] - roz[(ix-3)*n1+iz]*vz[(ix-3)*n1+iz]);
		}
	}

	/* ============================================================ */
	/*  Free surface txz correction (supplement to Step 1^T)        */
	/*                                                               */
	/*  Phase F1 vx at iz=surface reads txz[surface] and            */
	/*  txz[surface-1] through D+z, but Phase A2 txz starts at     */
	/*  ioTz=surface+1 and misses these transpose contributions.    */
	/*  Without them, BV_adj's txz scatter has nothing to propagate */
	/*  and the antisymmetric mirror coupling is lost.              */
	/* ============================================================ */
	if (bnd.top == 1) {
		int izs = bnd.surface[mod.ioPx];
#pragma omp for private (ix) schedule(guided,1)
		for (ix=mod.ioTx; ix<mod.ieTx; ix++) {
			txz[ix*n1+izs] +=
				c1*rox[ix*n1+izs]*vx[ix*n1+izs] +
				c2*rox[ix*n1+izs+1]*vx[ix*n1+izs+1] +
				c3*rox[ix*n1+izs+2]*vx[ix*n1+izs+2];
			txz[ix*n1+izs-1] +=
				c2*rox[ix*n1+izs]*vx[ix*n1+izs] +
				c3*rox[ix*n1+izs+1]*vx[ix*n1+izs+1];
			txz[ix*n1+izs-2] +=
				c3*rox[ix*n1+izs]*vx[ix*n1+izs];
		}
	}

	return 0;
}
