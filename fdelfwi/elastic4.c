#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include"fdelfwi.h"
#include"fdelfwi_offload.h"

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
	int   ix, iz;
	int   n1;
	int   pml;
	int   ioPx, iePx, ioPz, iePz;
	int   ioTx, ieTx, ioTz, ieTz;
	int   ioXx, ieXx, ioXz, ieXz;
	int   ioZx, ieZx, ioZz, ieZz;
	size_t sizem;   /* full padded grid; used by the slice-1a update macros */


	sizem = (size_t)mod.nax * mod.naz;
	c1 = 9.0/8.0;
	c2 = -1.0/24.0;
	n1  = mod.naz;
	pml = (bnd.top==2 || bnd.bot==2 || bnd.lef==2 || bnd.rig==2);

	/* Hoist every grid bound out of the struct BEFORE the kernels.
	 * A target region must not touch modPar: it is passed by value and
	 * carries host pointers, which are meaningless on the device.  As
	 * plain scalars these are implicitly firstprivate, so the kernels
	 * below need no map clause of any kind. */
	ioXx = mod.ioXx; ieXx = mod.ieXx; ioXz = mod.ioXz; ieXz = mod.ieXz;
	ioZx = mod.ioZx; ieZx = mod.ieZx; ioZz = mod.ioZz; ieZz = mod.ieZz;

	/* ============================================================ */
	/*  Step 1 / Phase F1: Velocity update                          */
	/*  vx -= rox * [D-x(txx) + D+z(txz)]                         */
	/* ============================================================ */
	FD_LOOP_IJ_NW
	for (ix=ioXx; ix<ieXx; ix++) {
		FD_SIMD
		for (iz=ioXz; iz<ieXz; iz++) {
			vx[ix*n1+iz] -= rox[ix*n1+iz]*(
						c1*(txx[ix*n1+iz]     - txx[(ix-1)*n1+iz] +
							txz[ix*n1+iz+1]   - txz[ix*n1+iz])    +
						c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
							txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
		}
	}

	/* vz -= roz * [D+x(txz) + D-z(tzz)] */
	FD_LOOP_IJ
	for (ix=ioZx; ix<ieZx; ix++) {
		FD_SIMD
		for (iz=ioZz; iz<ieZz; iz++) {
			vz[ix*n1+iz] -= roz[ix*n1+iz]*(
						c1*(tzz[ix*n1+iz]     - tzz[ix*n1+iz-1] +
							txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
						c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
							txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
		}
	}

	/* ============================================================ */
	/*  Step 2: Force source injection (src_type > 5)               */
	/*  Step 3: Velocity boundary conditions (boundariesP / taper)  */
	/* ============================================================ */
	/* SCAFFOLDING (slice 1a).  applySource and boundariesP still run on
	 * the host, so the wavefields are pulled back and pushed again around
	 * them.  This is CORRECT but slow -- it is the very host round-trip
	 * the offload exists to remove, and it disappears in slice 1b once
	 * the taper path and point-source injection are themselves kernels.
	 * Kept explicit rather than hidden so the cost is impossible to
	 * forget about. */
	OFFLOAD_WFL_FROM_DEVICE;

	if (src.type > 5 && src.type!=20) {
		 applySource(mod, src, wav, bnd, itime, ixsrc, izsrc, vx, vz, tzz, txx, txz, rox, roz, l2m, src_nwav, verbose);
	}

	boundariesP(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);

	OFFLOAD_WFL_TO_DEVICE;

	/* ============================================================ */
	/*  Step 4 / Phase F2: Stress update                            */
	/*  txx -= l2m * D+x(vx) + lam * D+z(vz)                      */
	/*  tzz -= lam * D+x(vx) + l2m * D+z(vz)                      */
	/* ============================================================ */
	/* Under CFS-CPML the stress in the PML zone is computed by boundariesV,
	 * so the kernel must skip it. fdelfwi's mod.ioPx/ioPz/ioTx/ioTz do NOT
	 * include the absorbing pad (getParameters.c leaves them unshifted, and
	 * ~60 call sites across the FWI files rely on that), so the pad is removed
	 * here in a locally scoped way -- the same trick acoustic4.c:119-142
	 * already uses for the acoustic PML.
	 *
	 * The resulting ranges reproduce fdelmodc's elastic4.c exactly: BOTH stress
	 * loops there run over [ioTx,ieTx) x [ioPz,iePz) of its own shifted indices,
	 * which is the complement of the four PML blocks in boundaries.c -- no gap,
	 * no double update. Note the txx/tzz loop uses the T-range in x, one column
	 * narrower than fdelfwi's usual ioPx..iePx; that is deliberate, not a typo. */
	if (pml) {
		ioPx = mod.ioTx + (bnd.lef==2 ? bnd.npml : 0);
		iePx = mod.ieTx - (bnd.rig==2 ? bnd.npml : 0);
		ioPz = mod.ioPz + (bnd.top==2 ? bnd.npml : 0);
		iePz = mod.iePz - (bnd.bot==2 ? bnd.npml : 0);
		ioTx = ioPx; ieTx = iePx;
		ioTz = ioPz; ieTz = iePz;
	}
	else {
		ioPx = mod.ioPx; iePx = mod.iePx;
		ioPz = mod.ioPz; iePz = mod.iePz;
		ioTx = mod.ioTx; ieTx = mod.ieTx;
		ioTz = mod.ioTz; ieTz = mod.ieTz;
	}

	FD_LOOP_IJ_NW
	for (ix=ioPx; ix<iePx; ix++) {
		FD_SIMD
		for (iz=ioPz; iz<iePz; iz++) {
			float dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
			            c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
			float dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
			            c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
			txx[ix*n1+iz] -= l2m[ix*n1+iz]*dvx + lam[ix*n1+iz]*dvz;
			tzz[ix*n1+iz] -= l2m[ix*n1+iz]*dvz + lam[ix*n1+iz]*dvx;
		}
	}

	/* txz -= mul * [D-z(vx) + D-x(vz)] */
	FD_LOOP_IJ
	for (ix=ioTx; ix<ieTx; ix++) {
		FD_SIMD
		for (iz=ioTz; iz<ieTz; iz++) {
			txz[ix*n1+iz] -= mul[ix*n1+iz]*(
					c1*(vx[ix*n1+iz]     - vx[ix*n1+iz-1] +
						vz[ix*n1+iz]     - vz[(ix-1)*n1+iz]) +
					c2*(vx[ix*n1+iz+1]   - vx[ix*n1+iz-2] +
						vz[(ix+1)*n1+iz] - vz[(ix-2)*n1+iz]) );
		}
	}

	/* ============================================================ */
	/*  Step 5: Stress source injection (src_type < 6)              */
	/*  Step 6: Free surface boundary conditions                    */
	/*  storeSourceOnSurface -> boundariesV -> reStoreSource        */
	/* ============================================================ */
	OFFLOAD_WFL_FROM_DEVICE;        /* slice 1a scaffolding, see above */

	if (src.type < 6) {
		 applySource(mod, src, wav, bnd, itime, ixsrc, izsrc, vx, vz, tzz, txx, txz, rox, roz, l2m, src_nwav, verbose);
	}

    storeSourceOnSurface(mod, src, bnd, ixsrc, izsrc, vx, vz, tzz, txx, txz, verbose);
    boundariesV(mod, bnd, vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, itime, verbose);
	reStoreSourceOnSurface(mod, src, bnd, ixsrc, izsrc, vx, vz, tzz, txx, txz, verbose);

	OFFLOAD_WFL_TO_DEVICE;

    return 0;
}
