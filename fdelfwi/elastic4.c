#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include"fdelfwi.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))

int applySource(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float *vx, float *vz, float *tzz,
float *txx, float *txz, float *rox, float *roz, float *l2m, float **src_nwav, int verbose);

int storeSourceOnSurface(modPar mod, srcPar src, bndPar bnd, int ixsrc, int izsrc, float *vx, float *vz, float *tzz, float *txx, float *txz, int verbose);

int reStoreSourceOnSurface(modPar mod, srcPar src, bndPar bnd, int ixsrc, int izsrc, float *vx, float *vz, float *tzz, float *txx, float *txz, int verbose);

int boundariesP(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int itime, int verbose);

int boundariesV(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int itime, int verbose);

int elastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime, int ixsrc, int izsrc, float **src_nwav, float *vx,
float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float
*l2m, float *lam, float *mul, int verbose)
{
/*********************************************************************

  Forward 4th-order elastic staggered-grid FD kernel.

  Forward operator (this file):
    vx -= rox * [D-x(txx) + D+z(txz)]           Step 1 / Phase F1
    vz -= roz * [D+x(txz) + D-z(tzz)]           Step 1 / Phase F1
    txx -= l2m * D+x(vx) + lam * D+z(vz)        Step 4 / Phase F2
    tzz -= lam * D+x(vx) + l2m * D+z(vz)        Step 4 / Phase F2
    txz -= mul * [D-z(vx) + D-x(vz)]             Step 4 / Phase F2

  TRUE ADJOINT (see elastic4_adj.c):
    vx += D-x(l2m*txx) + D-x(lam*tzz) + D+z(mul*txz)   Step 4^T / (F2)^T
    vz += D-z(lam*txx) + D-z(l2m*tzz) + D+x(mul*txz)   Step 4^T / (F2)^T
    txx += D+x(rox*vx)                                    Step 1^T / (F1)^T
    tzz += D+z(roz*vz)                                    Step 1^T / (F1)^T
    txz += D-z(rox*vx) + D-x(roz*vz)                     Step 1^T / (F1)^T

  Staggered grid layout:

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
                 |   |   |   |   |   |   |
     txz vz  txz Vz  Txz-Vz  Txz-Vz  Txz-Vz  txz vz  txz
                 |   |   |   |   |   |   |
     vx  t   vx  T---Vx--T---Vx--T---Vx--T   vx  t   vx
                 |   |   |   |   |   |   |
     txz vz  txz Vz  Txz-Vz  Txz-Vz  Txz-Vz  txz vz  txz
                 |   |   |   |   |   |   |
     vx  t   vx  T---Vx--T---Vx--T---Vx--T   vx  t   vx

     txz vz  txz vz  txz vz  txz vz  txz vz  txz vz  txz

     vx  t   vx  t   vx  t   vx  t   vx  t   vx  t  vx

     txz vz  txz vz  txz vz  txz vz  txz vz  txz vz  txz  <--|
                                                             |
                                         extra row of txz/vz |

   AUTHOR:
           Jan Thorbecke (janth@xs4all.nl)
           The Netherlands

***********************************************************************/

	float c1, c2;
	float dvx, dvz;
	int   ix, iz;
	int   n1;

	c1 = 9.0/8.0;
	c2 = -1.0/24.0;
	n1  = mod.naz;

	/* ============================================================ */
	/*  Step 1 / Phase F1: Velocity update                          */
	/*  vx -= rox * [D-x(txx) + D+z(txz)]                         */
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

	/* vz -= roz * [D+x(txz) + D-z(tzz)] */
#pragma omp for private (ix, iz)  schedule(guided,1)
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
	/*  Step 2: Force source injection (src_type > 5)               */
	/* ============================================================ */
	if (src.type > 5 && src.type!=20) {
		 applySource(mod, src, wav, bnd, itime, ixsrc, izsrc, vx, vz, tzz, txx, txz, rox, roz, l2m, src_nwav, verbose);
	}

	/* ============================================================ */
	/*  Step 3: Velocity boundary conditions (boundariesP / taper)  */
	/* ============================================================ */
	boundariesP(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);

	/* ============================================================ */
	/*  Step 4 / Phase F2: Stress update                            */
	/*  txx -= l2m * D+x(vx) + lam * D+z(vz)                      */
	/*  tzz -= lam * D+x(vx) + l2m * D+z(vz)                      */
	/* ============================================================ */
#pragma omp	for private (ix, iz, dvx, dvz) nowait schedule(guided,1)
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

	/* txz -= mul * [D-z(vx) + D-x(vz)] */
#pragma omp	for private (ix, iz)  schedule(guided,1)
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
	/*  Step 5: Stress source injection (src_type < 6)              */
	/* ============================================================ */
	if (src.type < 6) {
		 applySource(mod, src, wav, bnd, itime, ixsrc, izsrc, vx, vz, tzz, txx, txz, rox, roz, l2m, src_nwav, verbose);
	}

	/* ============================================================ */
	/*  Step 6: Free surface boundary conditions                    */
	/*  storeSourceOnSurface -> boundariesV -> reStoreSource        */
	/* ============================================================ */
    storeSourceOnSurface(mod, src, bnd, ixsrc, izsrc, vx, vz, tzz, txx, txz, verbose);
    boundariesV(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);
	reStoreSourceOnSurface(mod, src, bnd, ixsrc, izsrc, vx, vz, tzz, txx, txz, verbose);

    return 0;
}
