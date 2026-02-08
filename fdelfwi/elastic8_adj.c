#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"

/*********************************************************************
 *
 * elastic8_adj - TRUE 8th-order elastic adjoint FD kernel.
 *
 * Implements the mathematically correct discrete adjoint of elastic8.c.
 * See elastic4_adj.c for the derivation and sign conventions.
 *
 * TRUE ADJOINT:
 *   vx += D-x(l2m*txx) + D-x(lam*tzz) + D+z(mul*txz)
 *   vz += D-z(lam*txx) + D-z(l2m*tzz) + D+x(mul*txz)
 *   txx += D+x(bx*vx)
 *   tzz += D+z(bz*vz)
 *   txz += D-z(bx*vx) + D-x(bz*vz)
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


int elastic8_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
	float *vx, float *vz, float *tzz, float *txx, float *txz,
	float *rox, float *roz, float *l2m, float *lam, float *mul,
	int rec_delay, int rec_skipdt, int verbose)
{
	float c1, c2, c3, c4;
	int   ix, iz;
	int   n1;

	c1 = 1225.0/1024.0;
	c2 = -245.0/3072.0;
	c3 = 49.0/5120.0;
	c4 = -5.0/7168.0;
	n1 = mod.naz;

	/* ============================================================ */
	/*  STEP 1: Adjoint Free Surface (reverse of forward Step 6)    */
	/*  Must act BEFORE the stencil reads stress at surface points. */
	/* ============================================================ */
	boundariesV_adj(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);

	/* ============================================================ */
	/*  STEP 2: Inject adjoint STRESS sources (reverse of fwd Step 5)*/
	/* ============================================================ */
	applyAdjointSource(mod, adj, itime, vx, vz, tzz, txx, txz,
		mul, rec_delay, rec_skipdt, /*phase=*/2, verbose);

	/* ============================================================ */
	/*  Phase 1: Adjoint velocity update                            */
	/*  vx += D-x(l2m*txx) + D-x(lam*tzz) + D+z(mul*txz)          */
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
				    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]) +
				c4*(l2m[(ix+3)*n1+iz]*txx[(ix+3)*n1+iz] - l2m[(ix-4)*n1+iz]*txx[(ix-4)*n1+iz] +
				    lam[(ix+3)*n1+iz]*tzz[(ix+3)*n1+iz] - lam[(ix-4)*n1+iz]*tzz[(ix-4)*n1+iz] +
				    mul[ix*n1+iz+4]*txz[ix*n1+iz+4]     - mul[ix*n1+iz-3]*txz[ix*n1+iz-3]);
		}
	}

	/* ============================================================ */
	/*  vz += D-z(lam*txx) + D-z(l2m*tzz) + D+x(mul*txz)          */
	/* ============================================================ */
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
				    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]) +
				c4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3]     - lam[ix*n1+iz-4]*txx[ix*n1+iz-4] +
				    l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]     - l2m[ix*n1+iz-4]*tzz[ix*n1+iz-4] +
				    mul[(ix+4)*n1+iz]*txz[(ix+4)*n1+iz] - mul[(ix-3)*n1+iz]*txz[(ix-3)*n1+iz]);
		}
	}

	/* STEP 4: Adjoint velocity boundaries (reverse of fwd Step 3) */
	boundariesP_adj(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);

	/* STEP 5: Inject adjoint FORCE sources (reverse of fwd Step 2) */
	applyAdjointSource(mod, adj, itime, vx, vz, tzz, txx, txz,
		mul, rec_delay, rec_skipdt, /*phase=*/1, verbose);

	/* ============================================================ */
	/*  Phase 2: Adjoint stress update                              */
	/*  txx += D+x(rox*vx),  tzz += D+z(roz*vz)                   */
	/* ============================================================ */
#pragma omp for private (ix, iz) nowait schedule(guided,1)
	for (ix=mod.ioPx; ix<mod.iePx; ix++) {
#pragma simd
		for (iz=mod.ioPz; iz<mod.iePz; iz++) {
			txx[ix*n1+iz] +=
				c1*(rox[(ix+1)*n1+iz]*vx[(ix+1)*n1+iz] - rox[ix*n1+iz]*vx[ix*n1+iz]) +
				c2*(rox[(ix+2)*n1+iz]*vx[(ix+2)*n1+iz] - rox[(ix-1)*n1+iz]*vx[(ix-1)*n1+iz]) +
				c3*(rox[(ix+3)*n1+iz]*vx[(ix+3)*n1+iz] - rox[(ix-2)*n1+iz]*vx[(ix-2)*n1+iz]) +
				c4*(rox[(ix+4)*n1+iz]*vx[(ix+4)*n1+iz] - rox[(ix-3)*n1+iz]*vx[(ix-3)*n1+iz]);
			tzz[ix*n1+iz] +=
				c1*(roz[ix*n1+iz+1]*vz[ix*n1+iz+1]     - roz[ix*n1+iz]*vz[ix*n1+iz]) +
				c2*(roz[ix*n1+iz+2]*vz[ix*n1+iz+2]     - roz[ix*n1+iz-1]*vz[ix*n1+iz-1]) +
				c3*(roz[ix*n1+iz+3]*vz[ix*n1+iz+3]     - roz[ix*n1+iz-2]*vz[ix*n1+iz-2]) +
				c4*(roz[ix*n1+iz+4]*vz[ix*n1+iz+4]     - roz[ix*n1+iz-3]*vz[ix*n1+iz-3]);
		}
	}

	/* ============================================================ */
	/*  txz += D-z(rox*vx) + D-x(roz*vz)                           */
	/* ============================================================ */
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
				    roz[(ix+2)*n1+iz]*vz[(ix+2)*n1+iz] - roz[(ix-3)*n1+iz]*vz[(ix-3)*n1+iz]) +
				c4*(rox[ix*n1+iz+3]*vx[ix*n1+iz+3]     - rox[ix*n1+iz-4]*vx[ix*n1+iz-4] +
				    roz[(ix+3)*n1+iz]*vz[(ix+3)*n1+iz] - roz[(ix-4)*n1+iz]*vz[(ix-4)*n1+iz]);
		}
	}

	/* ============================================================ */
	/*  Free surface txz correction: missing F1^T contributions     */
	/* ============================================================ */
	if (bnd.top == 1) {
		int izs = bnd.surface[mod.ioPx];
#pragma omp for private (ix) schedule(guided,1)
		for (ix=mod.ioTx; ix<mod.ieTx; ix++) {
			txz[ix*n1+izs] +=
				c1*rox[ix*n1+izs]*vx[ix*n1+izs] +
				c2*rox[ix*n1+izs+1]*vx[ix*n1+izs+1] +
				c3*rox[ix*n1+izs+2]*vx[ix*n1+izs+2] +
				c4*rox[ix*n1+izs+3]*vx[ix*n1+izs+3];
			txz[ix*n1+izs-1] +=
				c2*rox[ix*n1+izs]*vx[ix*n1+izs] +
				c3*rox[ix*n1+izs+1]*vx[ix*n1+izs+1] +
				c4*rox[ix*n1+izs+2]*vx[ix*n1+izs+2];
			txz[ix*n1+izs-2] +=
				c3*rox[ix*n1+izs]*vx[ix*n1+izs] +
				c4*rox[ix*n1+izs+1]*vx[ix*n1+izs+1];
			txz[ix*n1+izs-3] +=
				c4*rox[ix*n1+izs]*vx[ix*n1+izs];
		}
	}

	return 0;
}
