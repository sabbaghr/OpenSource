#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"

/*********************************************************************
 *
 * elastic4_adj - 4th-order elastic adjoint FD kernel.
 *
 * Same staggered-grid FD stencils as elastic4.c, but replaces
 * applySource with applyAdjointSource for multicomponent FWI:
 *
 *   1. Velocity update (vx, vz)       -- identical stencils
 *   2. Inject force adjoint sources    -- Fx (type 6), Fz (type 7)
 *   3. Boundary conditions             -- boundariesP
 *   4. Stress update (txx, tzz, txz)   -- identical stencils
 *   5. Inject stress adjoint sources   -- P (1), Txz (2), Tzz (3), Txx (4)
 *   6. Boundary conditions             -- boundariesV
 *
 * The forward-modeling kernels (elastic4.c) are NOT modified.
 *
 *   AUTHOR:
 *           FD stencils from elastic4.c by Jan Thorbecke
 *           Adjoint source injection for FWI backpropagation.
 *
 **********************************************************************/

int applyAdjointSource(modPar mod, adjSrcPar adj, int itime,
	float *vx, float *vz, float *tzz, float *txx, float *txz,
	float *mul, int rec_delay, int rec_skipdt, int phase, int verbose);

int boundariesP(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz,
	float *txx, float *txz, float *rox, float *roz, float *l2m,
	float *lam, float *mul, int itime, int verbose);

int boundariesV(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz,
	float *txx, float *txz, float *rox, float *roz, float *l2m,
	float *lam, float *mul, int itime, int verbose);


int elastic4_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
	float *vx, float *vz, float *tzz, float *txx, float *txz,
	float *rox, float *roz, float *l2m, float *lam, float *mul,
	int rec_delay, int rec_skipdt, int verbose)
{
/*********************************************************************
       COMPUTATIONAL OVERVIEW OF THE 4th ORDER STAGGERED GRID:

  The captial symbols T (=Txx,Tzz) Txz,Vx,Vz represent the actual grid
  The indices ix,iz are related to the T grid, so the capital T
  symbols represent the actual modelled grid.

  one cel (iz,ix)
       |
       V                              extra column of vx,txz
                                                      |
    -------                                           V
   | txz vz| txz vz  txz vz  txz vz  txz vz  txz vz txz
   |       |
   | vx  t | vx  t   vx  t   vx  t   vx  t   vx  t  vx
    -------
     txz vz  txz vz  txz vz  txz vz  txz vz  txz vz  txz

     vx  t   vx  T---Vx--T---Vx--T---Vx--T   vx  t   vx
                 |   |   |   |   |   |   |
     txz vz  txz Vz--Txz-Vz--Txz-Vz  Txz-Vz  txz vz  txz
                 |   |   |   |   |   |   |
     vx  t   vx  T---Vx--T---Vx--T---Vx--T   vx  t   vx

***********************************************************************/

	float c1, c2;
	float dvx, dvz;
	int   ix, iz;
	int   n1;

	c1 = 9.0/8.0;
	c2 = -1.0/24.0;
	n1 = mod.naz;

	/* ============================================================ */
	/*  Velocity update: vx                                         */
	/* ============================================================ */
#pragma omp for private (ix, iz) nowait schedule(guided,1)
	for (ix=mod.ioXx; ix<mod.ieXx; ix++) {
#pragma simd
		for (iz=mod.ioXz; iz<mod.ieXz; iz++) {
			vx[ix*n1+iz] -= rox[ix*n1+iz]*(
						c1*(txx[ix*n1+iz]     - txx[(ix-1)*n1+iz] +
							txz[ix*n1+iz+1]   - txz[ix*n1+iz])    +
						c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
							txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
		}
	}

	/* ============================================================ */
	/*  Velocity update: vz                                         */
	/* ============================================================ */
#pragma omp for private (ix, iz) schedule(guided,1)
	for (ix=mod.ioZx; ix<mod.ieZx; ix++) {
#pragma simd
		for (iz=mod.ioZz; iz<mod.ieZz; iz++) {
			vz[ix*n1+iz] -= roz[ix*n1+iz]*(
						c1*(tzz[ix*n1+iz]     - tzz[ix*n1+iz-1] +
							txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
						c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
							txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
		}
	}

	/* ============================================================ */
	/*  Inject adjoint FORCE sources (Fx, Fz) at velocity point     */
	/* ============================================================ */
	applyAdjointSource(mod, adj, itime, vx, vz, tzz, txx, txz,
		mul, rec_delay, rec_skipdt, /*phase=*/1, verbose);

	/* ============================================================ */
	/*  Boundary conditions (velocity)                              */
	/* ============================================================ */
	boundariesP(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);

	/* ============================================================ */
	/*  Stress update: Txx, Tzz                                     */
	/* ============================================================ */
#pragma omp for private (ix, iz, dvx, dvz) nowait schedule(guided,1)
	for (ix=mod.ioPx; ix<mod.iePx; ix++) {
#pragma simd
		for (iz=mod.ioPz; iz<mod.iePz; iz++) {
			dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
			      c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
			dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
			      c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
			txx[ix*n1+iz] -= l2m[ix*n1+iz]*dvx + lam[ix*n1+iz]*dvz;
			tzz[ix*n1+iz] -= l2m[ix*n1+iz]*dvz + lam[ix*n1+iz]*dvx;
		}
	}

	/* ============================================================ */
	/*  Stress update: Txz                                          */
	/* ============================================================ */
#pragma omp for private (ix, iz) schedule(guided,1)
	for (ix=mod.ioTx; ix<mod.ieTx; ix++) {
#pragma simd
		for (iz=mod.ioTz; iz<mod.ieTz; iz++) {
			txz[ix*n1+iz] -= mul[ix*n1+iz]*(
					c1*(vx[ix*n1+iz]     - vx[ix*n1+iz-1] +
						vz[ix*n1+iz]     - vz[(ix-1)*n1+iz]) +
					c2*(vx[ix*n1+iz+1]   - vx[ix*n1+iz-2] +
						vz[(ix+1)*n1+iz] - vz[(ix-2)*n1+iz]) );
		}
	}

	/* ============================================================ */
	/*  Inject adjoint STRESS sources (P, Txx, Tzz, Txz) at stress */
	/* ============================================================ */
	applyAdjointSource(mod, adj, itime, vx, vz, tzz, txx, txz,
		mul, rec_delay, rec_skipdt, /*phase=*/2, verbose);

	/* ============================================================ */
	/*  Free surface boundary conditions for stresses               */
	/* ============================================================ */
	boundariesV(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);

	return 0;
}
