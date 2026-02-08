#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include"fdelfwi.h"

/*********************************************************************
 *
 * boundariesP_adj - TRUE ADJOINT taper boundary conditions for velocity.
 *
 * This is the adjoint of boundariesP for elastic taper boundaries (bnd==4).
 * The forward boundariesP extends velocity FD into the taper zone using
 * forward stencils (vx -= rox * D(sigma)) then damps with taper.
 *
 * The correct adjoint of this combined operation is:
 *   1. TRUE ADJOINT FD stencil (material INSIDE derivatives, += sign)
 *   2. Taper damping (self-adjoint: taper *= coeff)
 *
 * This function mirrors the exact loop structure and taper coefficient
 * handling of boundariesP in boundaries.c, but uses the adjoint stencils
 * from elastic4_adj.c Phase A1:
 *
 *   Forward:  vx -= rox * [D-x(txx) + D+z(txz)]
 *   Adjoint:  vx += D-x(l2m*txx) + D-x(lam*tzz) + D+z(mul*txz)
 *
 *   Forward:  vz -= roz * [D-z(tzz) + D+x(txz)]
 *   Adjoint:  vz += D-z(lam*txx) + D-z(l2m*tzz) + D+x(mul*txz)
 *
 * Uses 4th-order stencil coefficients (same as boundariesP).
 *
 * For rigid (bnd==3) and free surface (bnd==1) boundaries, the operations
 * are self-adjoint (mirror/zero conditions), so the same code as the
 * forward is used.
 *
 *   AUTHOR:
 *           Adjoint boundary derivation for FWI dot product test.
 *           Based on boundariesP by Jan Thorbecke.
 *
 **********************************************************************/

int boundariesP_adj(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz,
	float *txx, float *txz, float *rox, float *roz, float *l2m,
	float *lam, float *mul, int itime, int verbose)
{
	float c1, c2, c3, c4;
	int   ix, iz, ib, ibx, ibz;
	int   nx, nz, n1, n2;
	int   ixo, ixe, izo, ize;
	int   ibnd;

	if (mod.iorder <= 4) {
		c1 = 9.0/8.0;  c2 = -1.0/24.0;  c3 = 0.0;  c4 = 0.0;
	} else if (mod.iorder == 6) {
		c1 = 75.0/64.0;  c2 = -25.0/384.0;  c3 = 3.0/640.0;  c4 = 0.0;
	} else { /* iorder >= 8 */
		c1 = 1225.0/1024.0;  c2 = -245.0/3072.0;  c3 = 49.0/5120.0;  c4 = -5.0/7168.0;
	}
	nx  = mod.nx;
	nz  = mod.nz;
	n1  = mod.naz;
	n2  = mod.nax;
	ibnd = mod.iorder/2-1;

	/* For non-elastic schemes, no adjoint boundary needed */
	if (mod.ischeme <= 2) return 0;

/************************************************************/
/* Rigid boundary condition (self-adjoint, same as forward) */
/************************************************************/

	if (bnd.top==3) { /* rigid surface at top */
#pragma omp for private (ix, iz) nowait
#pragma simd
		for (ix=1; ix<=nx; ix++) {
			vz[ix*n1+ibnd] = -vz[ix*n1+ibnd+1];
			for (ib=1; ib<=ibnd; ib++) {
				vz[ix*n1+ibnd-ib] = -vz[ix*n1+ibnd+1+ib];
			}
		}
	}
	if (bnd.rig==3) { /* rigid surface at right */
#pragma omp for private (ix, iz) nowait
#pragma simd
		for (iz=1; iz<=nz; iz++) {
			vx[(nx+ibnd)*n1+iz]   = -vx[(nx+ibnd-1)*n1+iz];
			for (ib=1; ib<=ibnd; ib++) {
				vx[(nx+ibnd+ib)*n1+iz] = -vx[(nx+ibnd-1-ib)*n1+iz];
			}
		}
	}
	if (bnd.bot==3) { /* rigid surface at bottom */
#pragma omp for private (ix, iz) nowait
#pragma simd
		for (ix=1; ix<=nx; ix++) {
			vz[ix*n1+nz+ibnd]   = -vz[ix*n1+nz+ibnd-1];
			for (ib=1; ib<=ibnd; ib++) {
				vz[ix*n1+nz+ibnd+ib] = -vz[ix*n1+nz+ibnd-1-ib];
			}
		}
	}
	if (bnd.lef==3) { /* rigid surface at left */
#pragma omp for private (ix, iz) nowait
#pragma simd
		for (iz=1; iz<=nz; iz++) {
			vx[ibnd*n1+iz] = -vx[(ibnd+1)*n1+iz];
			for (ib=1; ib<=ibnd; ib++) {
				vx[(ibnd-ib)*n1+iz] = -vx[(ibnd+1+ib)*n1+iz];
			}
		}
	}

/************************************************************/
/* Free surface velocity adjoint (bnd==1)                    */
/* Forward boundariesP copies vz into virtual boundary:     */
/*   vz[iz_s] = vz[iz_s+1], vz[iz_s-1] = vz[iz_s+2]      */
/* Adjoint: scatter back and zero overwritten positions.    */
/*                                                           */
/* ACOUSTIC ONLY: the forward vz copy at the free surface   */
/* is only done for acoustic schemes (ischeme <= 2).         */
/* For elastic schemes, the free surface is handled entirely */
/* by boundariesV (stress conditions), not by vz copy.       */
/************************************************************/

	if (mod.ischeme <= 2 && bnd.top==1) {
#pragma omp for private (ix, iz) nowait
		for (ix=mod.ioPx; ix<mod.iePx; ix++) {
			iz = bnd.surface[ix];
			vz[ix*n1+iz+1] += vz[ix*n1+iz];
			vz[ix*n1+iz+2] += vz[ix*n1+iz-1];
			vz[ix*n1+iz]   = 0.0;
			vz[ix*n1+iz-1] = 0.0;
		}
	}

/************************************************************/
/* Taper boundaries: TRUE ADJOINT stencils                  */
/* Elastic scheme only (ischeme > 2)                        */
/************************************************************/

	/*********/
	/* Top   */
	/*********/
	if (bnd.top==4) {

		/* Vx field: adjoint stencil + taper */
		ixo = mod.ioXx;
		ixe = mod.ieXx;
		izo = mod.ioXz-bnd.ntap;
		ize = mod.ioXz;

		ib = (bnd.ntap+izo-1);
#pragma omp for private(ix,iz)
		for (ix=ixo; ix<ixe; ix++) {
#pragma simd
			for (iz=izo; iz<ize; iz++) {
				vx[ix*n1+iz] +=
					c1*(l2m[ix*n1+iz]*txx[ix*n1+iz]         - l2m[(ix-1)*n1+iz]*txx[(ix-1)*n1+iz] +
					    lam[ix*n1+iz]*tzz[ix*n1+iz]         - lam[(ix-1)*n1+iz]*tzz[(ix-1)*n1+iz] +
					    mul[ix*n1+iz+1]*txz[ix*n1+iz+1]     - mul[ix*n1+iz]*txz[ix*n1+iz]) +
					c2*(l2m[(ix+1)*n1+iz]*txx[(ix+1)*n1+iz] - l2m[(ix-2)*n1+iz]*txx[(ix-2)*n1+iz] +
					    lam[(ix+1)*n1+iz]*tzz[(ix+1)*n1+iz] - lam[(ix-2)*n1+iz]*tzz[(ix-2)*n1+iz] +
					    mul[ix*n1+iz+2]*txz[ix*n1+iz+2]     - mul[ix*n1+iz-1]*txz[ix*n1+iz-1]);
				if (mod.iorder >= 6) vx[ix*n1+iz] +=
					c3*(l2m[(ix+2)*n1+iz]*txx[(ix+2)*n1+iz] - l2m[(ix-3)*n1+iz]*txx[(ix-3)*n1+iz] +
					    lam[(ix+2)*n1+iz]*tzz[(ix+2)*n1+iz] - lam[(ix-3)*n1+iz]*tzz[(ix-3)*n1+iz] +
					    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]);
				if (mod.iorder >= 8) vx[ix*n1+iz] +=
					c4*(l2m[(ix+3)*n1+iz]*txx[(ix+3)*n1+iz] - l2m[(ix-4)*n1+iz]*txx[(ix-4)*n1+iz] +
					    lam[(ix+3)*n1+iz]*tzz[(ix+3)*n1+iz] - lam[(ix-4)*n1+iz]*tzz[(ix-4)*n1+iz] +
					    mul[ix*n1+iz+4]*txz[ix*n1+iz+4]     - mul[ix*n1+iz-3]*txz[ix*n1+iz-3]);

				vx[ix*n1+iz]   *= bnd.tapx[ib-iz];
			}
		}
		/* right top corner */
		if (bnd.rig==4) {
			ixo = mod.ieXx;
			ixe = ixo+bnd.ntap;
			ibz = (bnd.ntap+izo-1);
			ibx = (ixo);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] +=
						c1*(l2m[ix*n1+iz]*txx[ix*n1+iz]         - l2m[(ix-1)*n1+iz]*txx[(ix-1)*n1+iz] +
						    lam[ix*n1+iz]*tzz[ix*n1+iz]         - lam[(ix-1)*n1+iz]*tzz[(ix-1)*n1+iz] +
						    mul[ix*n1+iz+1]*txz[ix*n1+iz+1]     - mul[ix*n1+iz]*txz[ix*n1+iz]) +
						c2*(l2m[(ix+1)*n1+iz]*txx[(ix+1)*n1+iz] - l2m[(ix-2)*n1+iz]*txx[(ix-2)*n1+iz] +
						    lam[(ix+1)*n1+iz]*tzz[(ix+1)*n1+iz] - lam[(ix-2)*n1+iz]*tzz[(ix-2)*n1+iz] +
						    mul[ix*n1+iz+2]*txz[ix*n1+iz+2]     - mul[ix*n1+iz-1]*txz[ix*n1+iz-1]);
					if (mod.iorder >= 6) vx[ix*n1+iz] +=
						c3*(l2m[(ix+2)*n1+iz]*txx[(ix+2)*n1+iz] - l2m[(ix-3)*n1+iz]*txx[(ix-3)*n1+iz] +
						    lam[(ix+2)*n1+iz]*tzz[(ix+2)*n1+iz] - lam[(ix-3)*n1+iz]*tzz[(ix-3)*n1+iz] +
						    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]);
					if (mod.iorder >= 8) vx[ix*n1+iz] +=
						c4*(l2m[(ix+3)*n1+iz]*txx[(ix+3)*n1+iz] - l2m[(ix-4)*n1+iz]*txx[(ix-4)*n1+iz] +
						    lam[(ix+3)*n1+iz]*tzz[(ix+3)*n1+iz] - lam[(ix-4)*n1+iz]*tzz[(ix-4)*n1+iz] +
						    mul[ix*n1+iz+4]*txz[ix*n1+iz+4]     - mul[ix*n1+iz-3]*txz[ix*n1+iz-3]);

					vx[ix*n1+iz]   *= bnd.tapxz[(ix-ibx)*bnd.ntap+(ibz-iz)];
				}
			}
		}
		/* left top corner */
		if (bnd.lef==4) {
			ixo = mod.ioXx-bnd.ntap;
			ixe = mod.ioXx;
			ibz = (bnd.ntap+izo-1);
			ibx = (bnd.ntap+ixo-1);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] +=
						c1*(l2m[ix*n1+iz]*txx[ix*n1+iz]         - l2m[(ix-1)*n1+iz]*txx[(ix-1)*n1+iz] +
						    lam[ix*n1+iz]*tzz[ix*n1+iz]         - lam[(ix-1)*n1+iz]*tzz[(ix-1)*n1+iz] +
						    mul[ix*n1+iz+1]*txz[ix*n1+iz+1]     - mul[ix*n1+iz]*txz[ix*n1+iz]) +
						c2*(l2m[(ix+1)*n1+iz]*txx[(ix+1)*n1+iz] - l2m[(ix-2)*n1+iz]*txx[(ix-2)*n1+iz] +
						    lam[(ix+1)*n1+iz]*tzz[(ix+1)*n1+iz] - lam[(ix-2)*n1+iz]*tzz[(ix-2)*n1+iz] +
						    mul[ix*n1+iz+2]*txz[ix*n1+iz+2]     - mul[ix*n1+iz-1]*txz[ix*n1+iz-1]);
					if (mod.iorder >= 6) vx[ix*n1+iz] +=
						c3*(l2m[(ix+2)*n1+iz]*txx[(ix+2)*n1+iz] - l2m[(ix-3)*n1+iz]*txx[(ix-3)*n1+iz] +
						    lam[(ix+2)*n1+iz]*tzz[(ix+2)*n1+iz] - lam[(ix-3)*n1+iz]*tzz[(ix-3)*n1+iz] +
						    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]);
					if (mod.iorder >= 8) vx[ix*n1+iz] +=
						c4*(l2m[(ix+3)*n1+iz]*txx[(ix+3)*n1+iz] - l2m[(ix-4)*n1+iz]*txx[(ix-4)*n1+iz] +
						    lam[(ix+3)*n1+iz]*tzz[(ix+3)*n1+iz] - lam[(ix-4)*n1+iz]*tzz[(ix-4)*n1+iz] +
						    mul[ix*n1+iz+4]*txz[ix*n1+iz+4]     - mul[ix*n1+iz-3]*txz[ix*n1+iz-3]);

					vx[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(ibz-iz)];
				}
			}
		}

		/* Vz field: adjoint stencil + taper */
		ixo = mod.ioZx;
		ixe = mod.ieZx;
		izo = mod.ioZz-bnd.ntap;
		ize = mod.ioZz;

		ib = (bnd.ntap+izo-1);
#pragma omp for private (ix, iz)
		for (ix=ixo; ix<ixe; ix++) {
#pragma simd
			for (iz=izo; iz<ize; iz++) {
				vz[ix*n1+iz] +=
					c1*(lam[ix*n1+iz]*txx[ix*n1+iz]         - lam[ix*n1+iz-1]*txx[ix*n1+iz-1] +
					    l2m[ix*n1+iz]*tzz[ix*n1+iz]         - l2m[ix*n1+iz-1]*tzz[ix*n1+iz-1] +
					    mul[(ix+1)*n1+iz]*txz[(ix+1)*n1+iz] - mul[ix*n1+iz]*txz[ix*n1+iz]) +
					c2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1]     - lam[ix*n1+iz-2]*txx[ix*n1+iz-2] +
					    l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]     - l2m[ix*n1+iz-2]*tzz[ix*n1+iz-2] +
					    mul[(ix+2)*n1+iz]*txz[(ix+2)*n1+iz] - mul[(ix-1)*n1+iz]*txz[(ix-1)*n1+iz]);
				if (mod.iorder >= 6) vz[ix*n1+iz] +=
					c3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2]     - lam[ix*n1+iz-3]*txx[ix*n1+iz-3] +
					    l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]     - l2m[ix*n1+iz-3]*tzz[ix*n1+iz-3] +
					    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]);
				if (mod.iorder >= 8) vz[ix*n1+iz] +=
					c4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3]     - lam[ix*n1+iz-4]*txx[ix*n1+iz-4] +
					    l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]     - l2m[ix*n1+iz-4]*tzz[ix*n1+iz-4] +
					    mul[(ix+4)*n1+iz]*txz[(ix+4)*n1+iz] - mul[(ix-3)*n1+iz]*txz[(ix-3)*n1+iz]);

				vz[ix*n1+iz]   *= bnd.tapz[ib-iz];
			}
		}
		/* right top corner */
		if (bnd.rig==4) {
			ixo = mod.ieZx;
			ixe = ixo+bnd.ntap;
			ibz = (bnd.ntap+izo-1);
			ibx = (ixo);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] +=
						c1*(lam[ix*n1+iz]*txx[ix*n1+iz]         - lam[ix*n1+iz-1]*txx[ix*n1+iz-1] +
						    l2m[ix*n1+iz]*tzz[ix*n1+iz]         - l2m[ix*n1+iz-1]*tzz[ix*n1+iz-1] +
						    mul[(ix+1)*n1+iz]*txz[(ix+1)*n1+iz] - mul[ix*n1+iz]*txz[ix*n1+iz]) +
						c2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1]     - lam[ix*n1+iz-2]*txx[ix*n1+iz-2] +
						    l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]     - l2m[ix*n1+iz-2]*tzz[ix*n1+iz-2] +
						    mul[(ix+2)*n1+iz]*txz[(ix+2)*n1+iz] - mul[(ix-1)*n1+iz]*txz[(ix-1)*n1+iz]);
					if (mod.iorder >= 6) vz[ix*n1+iz] +=
						c3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2]     - lam[ix*n1+iz-3]*txx[ix*n1+iz-3] +
						    l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]     - l2m[ix*n1+iz-3]*tzz[ix*n1+iz-3] +
						    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]);
					if (mod.iorder >= 8) vz[ix*n1+iz] +=
						c4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3]     - lam[ix*n1+iz-4]*txx[ix*n1+iz-4] +
						    l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]     - l2m[ix*n1+iz-4]*tzz[ix*n1+iz-4] +
						    mul[(ix+4)*n1+iz]*txz[(ix+4)*n1+iz] - mul[(ix-3)*n1+iz]*txz[(ix-3)*n1+iz]);

					vz[ix*n1+iz]   *= bnd.tapxz[(ix-ibx)*bnd.ntap+(ibz-iz)];
				}
			}
		}
		/* left top corner */
		if (bnd.lef==4) {
			ixo = mod.ioZx-bnd.ntap;
			ixe = mod.ioZx;
			ibz = (bnd.ntap+izo-1);
			ibx = (bnd.ntap+ixo-1);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] +=
						c1*(lam[ix*n1+iz]*txx[ix*n1+iz]         - lam[ix*n1+iz-1]*txx[ix*n1+iz-1] +
						    l2m[ix*n1+iz]*tzz[ix*n1+iz]         - l2m[ix*n1+iz-1]*tzz[ix*n1+iz-1] +
						    mul[(ix+1)*n1+iz]*txz[(ix+1)*n1+iz] - mul[ix*n1+iz]*txz[ix*n1+iz]) +
						c2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1]     - lam[ix*n1+iz-2]*txx[ix*n1+iz-2] +
						    l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]     - l2m[ix*n1+iz-2]*tzz[ix*n1+iz-2] +
						    mul[(ix+2)*n1+iz]*txz[(ix+2)*n1+iz] - mul[(ix-1)*n1+iz]*txz[(ix-1)*n1+iz]);
					if (mod.iorder >= 6) vz[ix*n1+iz] +=
						c3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2]     - lam[ix*n1+iz-3]*txx[ix*n1+iz-3] +
						    l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]     - l2m[ix*n1+iz-3]*tzz[ix*n1+iz-3] +
						    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]);
					if (mod.iorder >= 8) vz[ix*n1+iz] +=
						c4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3]     - lam[ix*n1+iz-4]*txx[ix*n1+iz-4] +
						    l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]     - l2m[ix*n1+iz-4]*tzz[ix*n1+iz-4] +
						    mul[(ix+4)*n1+iz]*txz[(ix+4)*n1+iz] - mul[(ix-3)*n1+iz]*txz[(ix-3)*n1+iz]);

					vz[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(ibz-iz)];
				}
			}
		}

	}

	/*********/
	/* Bottom */
	/*********/
	if (bnd.bot==4) {

		/* Vx field: adjoint stencil + taper */
		ixo = mod.ioXx;
		ixe = mod.ieXx;
		izo = mod.ieXz;
		ize = mod.ieXz+bnd.ntap;

		ib = (ize-bnd.ntap);
#pragma omp for private(ix,iz)
		for (ix=ixo; ix<ixe; ix++) {
#pragma simd
			for (iz=izo; iz<ize; iz++) {
				vx[ix*n1+iz] +=
					c1*(l2m[ix*n1+iz]*txx[ix*n1+iz]         - l2m[(ix-1)*n1+iz]*txx[(ix-1)*n1+iz] +
					    lam[ix*n1+iz]*tzz[ix*n1+iz]         - lam[(ix-1)*n1+iz]*tzz[(ix-1)*n1+iz] +
					    mul[ix*n1+iz+1]*txz[ix*n1+iz+1]     - mul[ix*n1+iz]*txz[ix*n1+iz]) +
					c2*(l2m[(ix+1)*n1+iz]*txx[(ix+1)*n1+iz] - l2m[(ix-2)*n1+iz]*txx[(ix-2)*n1+iz] +
					    lam[(ix+1)*n1+iz]*tzz[(ix+1)*n1+iz] - lam[(ix-2)*n1+iz]*tzz[(ix-2)*n1+iz] +
					    mul[ix*n1+iz+2]*txz[ix*n1+iz+2]     - mul[ix*n1+iz-1]*txz[ix*n1+iz-1]);
				if (mod.iorder >= 6) vx[ix*n1+iz] +=
					c3*(l2m[(ix+2)*n1+iz]*txx[(ix+2)*n1+iz] - l2m[(ix-3)*n1+iz]*txx[(ix-3)*n1+iz] +
					    lam[(ix+2)*n1+iz]*tzz[(ix+2)*n1+iz] - lam[(ix-3)*n1+iz]*tzz[(ix-3)*n1+iz] +
					    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]);
				if (mod.iorder >= 8) vx[ix*n1+iz] +=
					c4*(l2m[(ix+3)*n1+iz]*txx[(ix+3)*n1+iz] - l2m[(ix-4)*n1+iz]*txx[(ix-4)*n1+iz] +
					    lam[(ix+3)*n1+iz]*tzz[(ix+3)*n1+iz] - lam[(ix-4)*n1+iz]*tzz[(ix-4)*n1+iz] +
					    mul[ix*n1+iz+4]*txz[ix*n1+iz+4]     - mul[ix*n1+iz-3]*txz[ix*n1+iz-3]);

				vx[ix*n1+iz]   *= bnd.tapx[iz-ib];
			}
		}
		/* right bottom corner */
		if (bnd.rig==4) {
			ixo = mod.ieXx;
			ixe = ixo+bnd.ntap;
			ibz = (izo);
			ibx = (ixo);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] +=
						c1*(l2m[ix*n1+iz]*txx[ix*n1+iz]         - l2m[(ix-1)*n1+iz]*txx[(ix-1)*n1+iz] +
						    lam[ix*n1+iz]*tzz[ix*n1+iz]         - lam[(ix-1)*n1+iz]*tzz[(ix-1)*n1+iz] +
						    mul[ix*n1+iz+1]*txz[ix*n1+iz+1]     - mul[ix*n1+iz]*txz[ix*n1+iz]) +
						c2*(l2m[(ix+1)*n1+iz]*txx[(ix+1)*n1+iz] - l2m[(ix-2)*n1+iz]*txx[(ix-2)*n1+iz] +
						    lam[(ix+1)*n1+iz]*tzz[(ix+1)*n1+iz] - lam[(ix-2)*n1+iz]*tzz[(ix-2)*n1+iz] +
						    mul[ix*n1+iz+2]*txz[ix*n1+iz+2]     - mul[ix*n1+iz-1]*txz[ix*n1+iz-1]);
					if (mod.iorder >= 6) vx[ix*n1+iz] +=
						c3*(l2m[(ix+2)*n1+iz]*txx[(ix+2)*n1+iz] - l2m[(ix-3)*n1+iz]*txx[(ix-3)*n1+iz] +
						    lam[(ix+2)*n1+iz]*tzz[(ix+2)*n1+iz] - lam[(ix-3)*n1+iz]*tzz[(ix-3)*n1+iz] +
						    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]);
					if (mod.iorder >= 8) vx[ix*n1+iz] +=
						c4*(l2m[(ix+3)*n1+iz]*txx[(ix+3)*n1+iz] - l2m[(ix-4)*n1+iz]*txx[(ix-4)*n1+iz] +
						    lam[(ix+3)*n1+iz]*tzz[(ix+3)*n1+iz] - lam[(ix-4)*n1+iz]*tzz[(ix-4)*n1+iz] +
						    mul[ix*n1+iz+4]*txz[ix*n1+iz+4]     - mul[ix*n1+iz-3]*txz[ix*n1+iz-3]);

					vx[ix*n1+iz]   *= bnd.tapxz[(ix-ibx)*bnd.ntap+(iz-ibz)];
				}
			}
		}
		/* left bottom corner */
		if (bnd.lef==4) {
			ixo = mod.ioXx-bnd.ntap;
			ixe = mod.ioXx;
			ibz = (izo);
			ibx = (bnd.ntap+ixo-1);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] +=
						c1*(l2m[ix*n1+iz]*txx[ix*n1+iz]         - l2m[(ix-1)*n1+iz]*txx[(ix-1)*n1+iz] +
						    lam[ix*n1+iz]*tzz[ix*n1+iz]         - lam[(ix-1)*n1+iz]*tzz[(ix-1)*n1+iz] +
						    mul[ix*n1+iz+1]*txz[ix*n1+iz+1]     - mul[ix*n1+iz]*txz[ix*n1+iz]) +
						c2*(l2m[(ix+1)*n1+iz]*txx[(ix+1)*n1+iz] - l2m[(ix-2)*n1+iz]*txx[(ix-2)*n1+iz] +
						    lam[(ix+1)*n1+iz]*tzz[(ix+1)*n1+iz] - lam[(ix-2)*n1+iz]*tzz[(ix-2)*n1+iz] +
						    mul[ix*n1+iz+2]*txz[ix*n1+iz+2]     - mul[ix*n1+iz-1]*txz[ix*n1+iz-1]);
					if (mod.iorder >= 6) vx[ix*n1+iz] +=
						c3*(l2m[(ix+2)*n1+iz]*txx[(ix+2)*n1+iz] - l2m[(ix-3)*n1+iz]*txx[(ix-3)*n1+iz] +
						    lam[(ix+2)*n1+iz]*tzz[(ix+2)*n1+iz] - lam[(ix-3)*n1+iz]*tzz[(ix-3)*n1+iz] +
						    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]);
					if (mod.iorder >= 8) vx[ix*n1+iz] +=
						c4*(l2m[(ix+3)*n1+iz]*txx[(ix+3)*n1+iz] - l2m[(ix-4)*n1+iz]*txx[(ix-4)*n1+iz] +
						    lam[(ix+3)*n1+iz]*tzz[(ix+3)*n1+iz] - lam[(ix-4)*n1+iz]*tzz[(ix-4)*n1+iz] +
						    mul[ix*n1+iz+4]*txz[ix*n1+iz+4]     - mul[ix*n1+iz-3]*txz[ix*n1+iz-3]);

					vx[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(iz-ibz)];
				}
			}
		}

		/* Vz field: adjoint stencil + taper */
		ixo = mod.ioZx;
		ixe = mod.ieZx;
		izo = mod.ieZz;
		ize = mod.ieZz+bnd.ntap;

		ib = (ize-bnd.ntap);
#pragma omp for private (ix, iz)
		for (ix=ixo; ix<ixe; ix++) {
#pragma simd
			for (iz=izo; iz<ize; iz++) {
				vz[ix*n1+iz] +=
					c1*(lam[ix*n1+iz]*txx[ix*n1+iz]         - lam[ix*n1+iz-1]*txx[ix*n1+iz-1] +
					    l2m[ix*n1+iz]*tzz[ix*n1+iz]         - l2m[ix*n1+iz-1]*tzz[ix*n1+iz-1] +
					    mul[(ix+1)*n1+iz]*txz[(ix+1)*n1+iz] - mul[ix*n1+iz]*txz[ix*n1+iz]) +
					c2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1]     - lam[ix*n1+iz-2]*txx[ix*n1+iz-2] +
					    l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]     - l2m[ix*n1+iz-2]*tzz[ix*n1+iz-2] +
					    mul[(ix+2)*n1+iz]*txz[(ix+2)*n1+iz] - mul[(ix-1)*n1+iz]*txz[(ix-1)*n1+iz]);
				if (mod.iorder >= 6) vz[ix*n1+iz] +=
					c3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2]     - lam[ix*n1+iz-3]*txx[ix*n1+iz-3] +
					    l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]     - l2m[ix*n1+iz-3]*tzz[ix*n1+iz-3] +
					    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]);
				if (mod.iorder >= 8) vz[ix*n1+iz] +=
					c4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3]     - lam[ix*n1+iz-4]*txx[ix*n1+iz-4] +
					    l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]     - l2m[ix*n1+iz-4]*tzz[ix*n1+iz-4] +
					    mul[(ix+4)*n1+iz]*txz[(ix+4)*n1+iz] - mul[(ix-3)*n1+iz]*txz[(ix-3)*n1+iz]);

				vz[ix*n1+iz]   *= bnd.tapz[iz-ib];
			}
		}
		/* right bottom corner */
		if (bnd.rig==4) {
			ixo = mod.ieZx;
			ixe = ixo+bnd.ntap;
			ibz = (izo);
			ibx = (ixo);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] +=
						c1*(lam[ix*n1+iz]*txx[ix*n1+iz]         - lam[ix*n1+iz-1]*txx[ix*n1+iz-1] +
						    l2m[ix*n1+iz]*tzz[ix*n1+iz]         - l2m[ix*n1+iz-1]*tzz[ix*n1+iz-1] +
						    mul[(ix+1)*n1+iz]*txz[(ix+1)*n1+iz] - mul[ix*n1+iz]*txz[ix*n1+iz]) +
						c2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1]     - lam[ix*n1+iz-2]*txx[ix*n1+iz-2] +
						    l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]     - l2m[ix*n1+iz-2]*tzz[ix*n1+iz-2] +
						    mul[(ix+2)*n1+iz]*txz[(ix+2)*n1+iz] - mul[(ix-1)*n1+iz]*txz[(ix-1)*n1+iz]);
					if (mod.iorder >= 6) vz[ix*n1+iz] +=
						c3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2]     - lam[ix*n1+iz-3]*txx[ix*n1+iz-3] +
						    l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]     - l2m[ix*n1+iz-3]*tzz[ix*n1+iz-3] +
						    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]);
					if (mod.iorder >= 8) vz[ix*n1+iz] +=
						c4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3]     - lam[ix*n1+iz-4]*txx[ix*n1+iz-4] +
						    l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]     - l2m[ix*n1+iz-4]*tzz[ix*n1+iz-4] +
						    mul[(ix+4)*n1+iz]*txz[(ix+4)*n1+iz] - mul[(ix-3)*n1+iz]*txz[(ix-3)*n1+iz]);

					vz[ix*n1+iz]   *= bnd.tapxz[(ix-ibx)*bnd.ntap+(iz-ibz)];
				}
			}
		}
		/* left bottom corner */
		if (bnd.lef==4) {
			ixo = mod.ioZx-bnd.ntap;
			ixe = mod.ioZx;
			ibz = (izo);
			ibx = (bnd.ntap+ixo-1);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] +=
						c1*(lam[ix*n1+iz]*txx[ix*n1+iz]         - lam[ix*n1+iz-1]*txx[ix*n1+iz-1] +
						    l2m[ix*n1+iz]*tzz[ix*n1+iz]         - l2m[ix*n1+iz-1]*tzz[ix*n1+iz-1] +
						    mul[(ix+1)*n1+iz]*txz[(ix+1)*n1+iz] - mul[ix*n1+iz]*txz[ix*n1+iz]) +
						c2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1]     - lam[ix*n1+iz-2]*txx[ix*n1+iz-2] +
						    l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]     - l2m[ix*n1+iz-2]*tzz[ix*n1+iz-2] +
						    mul[(ix+2)*n1+iz]*txz[(ix+2)*n1+iz] - mul[(ix-1)*n1+iz]*txz[(ix-1)*n1+iz]);
					if (mod.iorder >= 6) vz[ix*n1+iz] +=
						c3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2]     - lam[ix*n1+iz-3]*txx[ix*n1+iz-3] +
						    l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]     - l2m[ix*n1+iz-3]*tzz[ix*n1+iz-3] +
						    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]);
					if (mod.iorder >= 8) vz[ix*n1+iz] +=
						c4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3]     - lam[ix*n1+iz-4]*txx[ix*n1+iz-4] +
						    l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]     - l2m[ix*n1+iz-4]*tzz[ix*n1+iz-4] +
						    mul[(ix+4)*n1+iz]*txz[(ix+4)*n1+iz] - mul[(ix-3)*n1+iz]*txz[(ix-3)*n1+iz]);

					vz[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(iz-ibz)];
				}
			}
		}

	}

	/*********/
	/* Left  */
	/*********/
	if (bnd.lef==4) {

		/* Vx field: adjoint stencil + taper */
		ixo = mod.ioXx-bnd.ntap;
		ixe = mod.ioXx;
		izo = mod.ioXz;
		ize = mod.ieXz;

		ib = (bnd.ntap+ixo-1);
#pragma omp for private(ix,iz)
		for (ix=ixo; ix<ixe; ix++) {
#pragma simd
			for (iz=izo; iz<ize; iz++) {
				vx[ix*n1+iz] +=
					c1*(l2m[ix*n1+iz]*txx[ix*n1+iz]         - l2m[(ix-1)*n1+iz]*txx[(ix-1)*n1+iz] +
					    lam[ix*n1+iz]*tzz[ix*n1+iz]         - lam[(ix-1)*n1+iz]*tzz[(ix-1)*n1+iz] +
					    mul[ix*n1+iz+1]*txz[ix*n1+iz+1]     - mul[ix*n1+iz]*txz[ix*n1+iz]) +
					c2*(l2m[(ix+1)*n1+iz]*txx[(ix+1)*n1+iz] - l2m[(ix-2)*n1+iz]*txx[(ix-2)*n1+iz] +
					    lam[(ix+1)*n1+iz]*tzz[(ix+1)*n1+iz] - lam[(ix-2)*n1+iz]*tzz[(ix-2)*n1+iz] +
					    mul[ix*n1+iz+2]*txz[ix*n1+iz+2]     - mul[ix*n1+iz-1]*txz[ix*n1+iz-1]);
				if (mod.iorder >= 6) vx[ix*n1+iz] +=
					c3*(l2m[(ix+2)*n1+iz]*txx[(ix+2)*n1+iz] - l2m[(ix-3)*n1+iz]*txx[(ix-3)*n1+iz] +
					    lam[(ix+2)*n1+iz]*tzz[(ix+2)*n1+iz] - lam[(ix-3)*n1+iz]*tzz[(ix-3)*n1+iz] +
					    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]);
				if (mod.iorder >= 8) vx[ix*n1+iz] +=
					c4*(l2m[(ix+3)*n1+iz]*txx[(ix+3)*n1+iz] - l2m[(ix-4)*n1+iz]*txx[(ix-4)*n1+iz] +
					    lam[(ix+3)*n1+iz]*tzz[(ix+3)*n1+iz] - lam[(ix-4)*n1+iz]*tzz[(ix-4)*n1+iz] +
					    mul[ix*n1+iz+4]*txz[ix*n1+iz+4]     - mul[ix*n1+iz-3]*txz[ix*n1+iz-3]);

				vx[ix*n1+iz]   *= bnd.tapx[ib-ix];
			}
		}

		/* Vz field: adjoint stencil + taper */
		ixo = mod.ioZx-bnd.ntap;
		ixe = mod.ioZx;
		izo = mod.ioZz;
		ize = mod.ieZz;

		ib = (bnd.ntap+ixo-1);
#pragma omp for private (ix, iz)
		for (ix=ixo; ix<ixe; ix++) {
#pragma simd
			for (iz=izo; iz<ize; iz++) {
				vz[ix*n1+iz] +=
					c1*(lam[ix*n1+iz]*txx[ix*n1+iz]         - lam[ix*n1+iz-1]*txx[ix*n1+iz-1] +
					    l2m[ix*n1+iz]*tzz[ix*n1+iz]         - l2m[ix*n1+iz-1]*tzz[ix*n1+iz-1] +
					    mul[(ix+1)*n1+iz]*txz[(ix+1)*n1+iz] - mul[ix*n1+iz]*txz[ix*n1+iz]) +
					c2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1]     - lam[ix*n1+iz-2]*txx[ix*n1+iz-2] +
					    l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]     - l2m[ix*n1+iz-2]*tzz[ix*n1+iz-2] +
					    mul[(ix+2)*n1+iz]*txz[(ix+2)*n1+iz] - mul[(ix-1)*n1+iz]*txz[(ix-1)*n1+iz]);
				if (mod.iorder >= 6) vz[ix*n1+iz] +=
					c3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2]     - lam[ix*n1+iz-3]*txx[ix*n1+iz-3] +
					    l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]     - l2m[ix*n1+iz-3]*tzz[ix*n1+iz-3] +
					    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]);
				if (mod.iorder >= 8) vz[ix*n1+iz] +=
					c4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3]     - lam[ix*n1+iz-4]*txx[ix*n1+iz-4] +
					    l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]     - l2m[ix*n1+iz-4]*tzz[ix*n1+iz-4] +
					    mul[(ix+4)*n1+iz]*txz[(ix+4)*n1+iz] - mul[(ix-3)*n1+iz]*txz[(ix-3)*n1+iz]);

				vz[ix*n1+iz]   *= bnd.tapz[ib-ix];
			}
		}
	}

	/*********/
	/* Right */
	/*********/
	if (bnd.rig==4) {

		/* Vx field: adjoint stencil + taper */
		ixo = mod.ieXx;
		ixe = mod.ieXx+bnd.ntap;
		izo = mod.ioXz;
		ize = mod.ieXz;

		ib = (ixe-bnd.ntap);
#pragma omp for private(ix,iz)
		for (ix=ixo; ix<ixe; ix++) {
#pragma simd
			for (iz=izo; iz<ize; iz++) {
				vx[ix*n1+iz] +=
					c1*(l2m[ix*n1+iz]*txx[ix*n1+iz]         - l2m[(ix-1)*n1+iz]*txx[(ix-1)*n1+iz] +
					    lam[ix*n1+iz]*tzz[ix*n1+iz]         - lam[(ix-1)*n1+iz]*tzz[(ix-1)*n1+iz] +
					    mul[ix*n1+iz+1]*txz[ix*n1+iz+1]     - mul[ix*n1+iz]*txz[ix*n1+iz]) +
					c2*(l2m[(ix+1)*n1+iz]*txx[(ix+1)*n1+iz] - l2m[(ix-2)*n1+iz]*txx[(ix-2)*n1+iz] +
					    lam[(ix+1)*n1+iz]*tzz[(ix+1)*n1+iz] - lam[(ix-2)*n1+iz]*tzz[(ix-2)*n1+iz] +
					    mul[ix*n1+iz+2]*txz[ix*n1+iz+2]     - mul[ix*n1+iz-1]*txz[ix*n1+iz-1]);
				if (mod.iorder >= 6) vx[ix*n1+iz] +=
					c3*(l2m[(ix+2)*n1+iz]*txx[(ix+2)*n1+iz] - l2m[(ix-3)*n1+iz]*txx[(ix-3)*n1+iz] +
					    lam[(ix+2)*n1+iz]*tzz[(ix+2)*n1+iz] - lam[(ix-3)*n1+iz]*tzz[(ix-3)*n1+iz] +
					    mul[ix*n1+iz+3]*txz[ix*n1+iz+3]     - mul[ix*n1+iz-2]*txz[ix*n1+iz-2]);
				if (mod.iorder >= 8) vx[ix*n1+iz] +=
					c4*(l2m[(ix+3)*n1+iz]*txx[(ix+3)*n1+iz] - l2m[(ix-4)*n1+iz]*txx[(ix-4)*n1+iz] +
					    lam[(ix+3)*n1+iz]*tzz[(ix+3)*n1+iz] - lam[(ix-4)*n1+iz]*tzz[(ix-4)*n1+iz] +
					    mul[ix*n1+iz+4]*txz[ix*n1+iz+4]     - mul[ix*n1+iz-3]*txz[ix*n1+iz-3]);

				vx[ix*n1+iz]   *= bnd.tapx[ix-ib];
			}
		}

		/* Vz field: adjoint stencil + taper */
		ixo = mod.ieZx;
		ixe = mod.ieZx+bnd.ntap;
		izo = mod.ioZz;
		ize = mod.ieZz;
		ib = (ixe-bnd.ntap);
#pragma omp for private (ix, iz)
		for (ix=ixo; ix<ixe; ix++) {
#pragma simd
			for (iz=izo; iz<ize; iz++) {
				vz[ix*n1+iz] +=
					c1*(lam[ix*n1+iz]*txx[ix*n1+iz]         - lam[ix*n1+iz-1]*txx[ix*n1+iz-1] +
					    l2m[ix*n1+iz]*tzz[ix*n1+iz]         - l2m[ix*n1+iz-1]*tzz[ix*n1+iz-1] +
					    mul[(ix+1)*n1+iz]*txz[(ix+1)*n1+iz] - mul[ix*n1+iz]*txz[ix*n1+iz]) +
					c2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1]     - lam[ix*n1+iz-2]*txx[ix*n1+iz-2] +
					    l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]     - l2m[ix*n1+iz-2]*tzz[ix*n1+iz-2] +
					    mul[(ix+2)*n1+iz]*txz[(ix+2)*n1+iz] - mul[(ix-1)*n1+iz]*txz[(ix-1)*n1+iz]);
				if (mod.iorder >= 6) vz[ix*n1+iz] +=
					c3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2]     - lam[ix*n1+iz-3]*txx[ix*n1+iz-3] +
					    l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]     - l2m[ix*n1+iz-3]*tzz[ix*n1+iz-3] +
					    mul[(ix+3)*n1+iz]*txz[(ix+3)*n1+iz] - mul[(ix-2)*n1+iz]*txz[(ix-2)*n1+iz]);
				if (mod.iorder >= 8) vz[ix*n1+iz] +=
					c4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3]     - lam[ix*n1+iz-4]*txx[ix*n1+iz-4] +
					    l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]     - l2m[ix*n1+iz-4]*tzz[ix*n1+iz-4] +
					    mul[(ix+4)*n1+iz]*txz[(ix+4)*n1+iz] - mul[(ix-3)*n1+iz]*txz[(ix-3)*n1+iz]);

				vz[ix*n1+iz]   *= bnd.tapz[ix-ib];
			}
		}
	}

	return 0;
}


/*********************************************************************
 *
 * boundariesV_adj - TRUE ADJOINT of free surface boundary conditions.
 *
 * The forward boundariesV applies elastic free surface conditions to
 * the stress field AFTER the stress update (Phase F2). For each free
 * surface side, it performs three operations:
 *
 *   1. Zero normal stress (tzz for top/bottom, txx for left/right)
 *   2. Antisymmetric extrapolation of shear stress (txz) into virtual
 *      boundary to enforce txz=0 at the surface
 *   3. Overwrite the tangential normal stress from velocity derivative:
 *      txx = -dp * D+x(vx)  for top/bottom free surface
 *      tzz = -dp * D+z(vz)  for left/right free surface
 *      where dp = l2m - lam^2/l2m = 4*mu*(lam+mu)/(lam+2*mu)
 *
 * The adjoint of each operation:
 *
 *   1'. Zero normal stress: self-adjoint (projection)
 *       tzz_adj = 0  (same as forward)
 *
 *   2'. Antisymmetric extrapolation adjoint:
 *       Forward: txz[s] = -txz[s+1], txz[s-1] = -txz[s+2]
 *       Adjoint: txz[s+1] -= adj_s, txz[s+2] -= adj_s1,
 *                txz[s] = 0, txz[s-1] = 0
 *       (scatter from overwritten points to source points, then zero)
 *
 *   3'. Stress-from-velocity adjoint:
 *       Forward: txx = -dp * D+x(vx)
 *       Adjoint: vx += D-x(dp * adj_txx), then txx = 0
 *       (D+x^T = -D-x by SBP, so adjoint of -dp*D+x is +D-x(dp*...))
 *       Implemented as scatter: for each ix, scatter adj_txx to
 *       neighboring vx positions with transpose stencil coefficients.
 *
 *   AUTHOR:
 *           Adjoint free surface derivation for FWI dot product test.
 *           Based on boundariesV by Jan Thorbecke.
 *
 **********************************************************************/

int boundariesV_adj(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz,
	float *txx, float *txz, float *rox, float *roz, float *l2m,
	float *lam, float *mul, int itime, int verbose)
{
	float c1, c2;
	float dp;
	int   ix, iz, n1;
	int   ixo, izp;
	float adj_val, adj_val2;

	c1 = 9.0/8.0;
	c2 = -1.0/24.0;
	n1 = mod.naz;

	ixo = mod.ioPx;

	if (mod.ischeme <= 2) { /* Acoustic scheme: tzz=0 is self-adjoint */
		if (bnd.top==1) {
#pragma omp for private (ix, iz) nowait
			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
				iz = bnd.surface[ix];
				tzz[ix*n1+iz] = 0.0;
			}
		}
		if (bnd.rig==1) {
#pragma omp for private (iz) nowait
			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
				tzz[(mod.iePx-1)*n1+iz] = 0.0;
			}
		}
		if (bnd.bot==1) {
#pragma omp for private (ix) nowait
			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
				tzz[ix*n1+mod.iePz-1] = 0.0;
			}
		}
		if (bnd.lef==1) {
#pragma omp for private (iz) nowait
			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
				tzz[(mod.ioPx-1)*n1+iz] = 0.0;
			}
		}
		return 0;
	}

	/* ================================================================ */
	/* Elastic free surface: TRUE ADJOINT                                */
	/* ================================================================ */

	/*** TOP free surface ***/
	if (bnd.top==1) {

		/* ----- Adjoint of txx = -dp * D+x(vx) ----- */
		/* Scatter: vx += (-dp * stencil_coeff) * adj_txx for each ix.
		 * This has race conditions on vx (neighboring ix write to same
		 * vx column), so use #pragma omp single. The loop is O(nx). */
#pragma omp single
{
		izp = bnd.surface[ixo];
		for (ix=mod.ioPx; ix<mod.iePx; ix++) {
			iz = bnd.surface[ix];
			if (izp == iz) {
				if (l2m[ix*n1+iz] != 0.0) {
					dp = l2m[ix*n1+iz] - lam[ix*n1+iz]*lam[ix*n1+iz]/l2m[ix*n1+iz];
					adj_val = txx[ix*n1+iz];

					/* Transpose of D+x gather: scatter to vx */
					vx[(ix+1)*n1+iz] += (-dp * c1) * adj_val;
					vx[ix*n1+iz]     += ( dp * c1) * adj_val;
					vx[(ix+2)*n1+iz] += (-dp * c2) * adj_val;
					vx[(ix-1)*n1+iz] += ( dp * c2) * adj_val;

					txx[ix*n1+iz] = 0.0;
				}
			}
			izp = iz;
		}
}

		/* ----- Adjoint of txz antisymmetric extrapolation ----- */
		/* Forward: txz[iz_s] = -txz[iz_s+1], txz[iz_s-1] = -txz[iz_s+2]
		 * Each ix column is independent (no cross-ix writes). */
		izp = bnd.surface[ixo];
#pragma omp for private (ix, iz, adj_val, adj_val2)
		for (ix=mod.ioTx; ix<mod.ieTx; ix++) {
			iz = bnd.surface[ix];
			if (izp == iz) {
				adj_val  = txz[ix*n1+iz];
				adj_val2 = txz[ix*n1+iz-1];

				txz[ix*n1+iz+1] -= adj_val;
				txz[ix*n1+iz+2] -= adj_val2;

				txz[ix*n1+iz]   = 0.0;
				txz[ix*n1+iz-1] = 0.0;
			}
		}

		/* ----- Adjoint of tzz = 0 (self-adjoint projection) ----- */
		izp = bnd.surface[ixo];
#pragma omp for private (ix, iz)
		for (ix=mod.ioPx; ix<mod.iePx; ix++) {
			iz = bnd.surface[ix];
			if (izp == iz) {
				tzz[ix*n1+iz] = 0.0;
			}
		}

		/* ----- Boundary correction: missing adj_vz at free surface ----- */
		/* ACOUSTIC ONLY: The adjoint Vz stencil (Phase A1) runs from
		 * ioZz = surface+1, so it never writes to adj_vz at iz=surface.
		 * For ACOUSTIC schemes, boundariesP copies vz[surface] = vz[surface+1],
		 * creating an aliasing that must be handled in the adjoint:
		 *   adj_vz[s] += D-z(lam*txx + l2m*tzz)|_{iz=s}
		 * BpT then scatters adj_vz[s] to adj_vz[s+1].
		 *
		 * For ELASTIC schemes, boundariesP does NOT copy vz at the
		 * free surface. vz[surface] is a constrained variable (always 0,
		 * never updated by Phase F1). No correction is needed because
		 * Phase A2 should not read adj_vz at iz < ioZz. */
		if (mod.ischeme <= 2) {
			float cc2, cc3, cc4;
			int ibnd = mod.iorder/2 - 1;

			if (mod.iorder <= 4) {
				cc2 = -1.0/24.0;  cc3 = 0.0;  cc4 = 0.0;
			} else if (mod.iorder == 6) {
				cc2 = -25.0/384.0;  cc3 = 3.0/640.0;  cc4 = 0.0;
			} else { /* iorder >= 8 */
				cc2 = -245.0/3072.0;  cc3 = 49.0/5120.0;  cc4 = -5.0/7168.0;
			}

			izp = bnd.surface[ixo];
#pragma omp for private (ix, iz)
			for (ix=mod.ioZx; ix<mod.ieZx; ix++) {
				iz = bnd.surface[ix];
				if (izp == iz) {
					/* adj_vz at iz=surface: only c2+ terms survive */
					vz[ix*n1+iz] +=
						cc2*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1] + l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]);

					if (mod.iorder >= 6) {
						vz[ix*n1+iz] +=
							cc3*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2] + l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]);

						/* adj_vz at iz=surface-1 (for 6th+ order) */
						vz[ix*n1+iz-1] +=
							cc3*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1] + l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]);
					}
					if (mod.iorder >= 8) {
						vz[ix*n1+iz] +=
							cc4*(lam[ix*n1+iz+3]*txx[ix*n1+iz+3] + l2m[ix*n1+iz+3]*tzz[ix*n1+iz+3]);

						vz[ix*n1+iz-1] +=
							cc4*(lam[ix*n1+iz+2]*txx[ix*n1+iz+2] + l2m[ix*n1+iz+2]*tzz[ix*n1+iz+2]);

						/* adj_vz at iz=surface-2 (for 8th order) */
						vz[ix*n1+iz-2] +=
							cc4*(lam[ix*n1+iz+1]*txx[ix*n1+iz+1] + l2m[ix*n1+iz+1]*tzz[ix*n1+iz+1]);
					}
				}
			}
		}
	}

	/*** RIGHT free surface ***/
	if (bnd.rig==1) {
		ix = mod.iePx;

		/* ----- Adjoint of tzz = -dp * D+z(vz) ----- */
		/* Loop over iz with scatter to vz: race condition, use single */
#pragma omp single
{
		for (iz=mod.ioPz; iz<mod.iePz; iz++) {
			if (l2m[ix*n1+iz] != 0.0) {
				dp = l2m[ix*n1+iz] - lam[ix*n1+iz]*lam[ix*n1+iz]/l2m[ix*n1+iz];
				adj_val = tzz[ix*n1+iz];

				vz[ix*n1+iz+1] += (-dp * c1) * adj_val;
				vz[ix*n1+iz]   += ( dp * c1) * adj_val;
				vz[ix*n1+iz+2] += (-dp * c2) * adj_val;
				vz[ix*n1+iz-1] += ( dp * c2) * adj_val;

				tzz[ix*n1+iz] = 0.0;
			}
		}
}

		/* ----- Adjoint of txz antisymmetric ----- */
		/* Forward: txz[(ix+1)*n1+iz] = -txz[ix*n1+iz]
		 *          txz[(ix+2)*n1+iz] = -txz[(ix-1)*n1+iz]
		 * Each iz is independent (different rows in same ix-columns). */
#pragma omp for private (iz, adj_val, adj_val2)
		for (iz=mod.ioTz; iz<mod.ieTz; iz++) {
			adj_val  = txz[(ix+1)*n1+iz];
			adj_val2 = txz[(ix+2)*n1+iz];

			txz[ix*n1+iz]     -= adj_val;
			txz[(ix-1)*n1+iz] -= adj_val2;

			txz[(ix+1)*n1+iz] = 0.0;
			txz[(ix+2)*n1+iz] = 0.0;
		}

		/* ----- Adjoint of txx = 0 ----- */
#pragma omp for private (iz)
		for (iz=mod.ioPz; iz<mod.iePz; iz++) {
			txx[ix*n1+iz] = 0.0;
		}
	}

	/*** BOTTOM free surface ***/
	if (bnd.bot==1) {
		iz = mod.iePz;

		/* ----- Adjoint of txx = -dp * D+x(vx) ----- */
#pragma omp single
{
		for (ix=mod.ioPx; ix<mod.iePx; ix++) {
			if (l2m[ix*n1+iz] != 0.0) {
				dp = l2m[ix*n1+iz] - lam[ix*n1+iz]*lam[ix*n1+iz]/l2m[ix*n1+iz];
				adj_val = txx[ix*n1+iz];

				vx[(ix+1)*n1+iz] += (-dp * c1) * adj_val;
				vx[ix*n1+iz]     += ( dp * c1) * adj_val;
				vx[(ix+2)*n1+iz] += (-dp * c2) * adj_val;
				vx[(ix-1)*n1+iz] += ( dp * c2) * adj_val;

				txx[ix*n1+iz] = 0.0;
			}
		}
}

		/* ----- Adjoint of txz antisymmetric ----- */
		/* Forward: txz[ix*n1+iz+1] = -txz[ix*n1+iz]
		 *          txz[ix*n1+iz+2] = -txz[ix*n1+iz-1] */
#pragma omp for private (ix, adj_val, adj_val2)
		for (ix=mod.ioTx; ix<mod.ieTx; ix++) {
			adj_val  = txz[ix*n1+iz+1];
			adj_val2 = txz[ix*n1+iz+2];

			txz[ix*n1+iz]   -= adj_val;
			txz[ix*n1+iz-1] -= adj_val2;

			txz[ix*n1+iz+1] = 0.0;
			txz[ix*n1+iz+2] = 0.0;
		}

		/* ----- Adjoint of tzz = 0 ----- */
#pragma omp for private (ix)
		for (ix=mod.ioPx; ix<mod.iePx; ix++) {
			tzz[ix*n1+iz] = 0.0;
		}
	}

	/*** LEFT free surface ***/
	if (bnd.lef==1) {
		ix = mod.ioPx;

		/* ----- Adjoint of tzz = -dp * D+z(vz) ----- */
#pragma omp single
{
		for (iz=mod.ioPz; iz<mod.iePz; iz++) {
			if (l2m[ix*n1+iz] != 0.0) {
				dp = l2m[ix*n1+iz] - lam[ix*n1+iz]*lam[ix*n1+iz]/l2m[ix*n1+iz];
				adj_val = tzz[ix*n1+iz];

				vz[ix*n1+iz+1] += (-dp * c1) * adj_val;
				vz[ix*n1+iz]   += ( dp * c1) * adj_val;
				vz[ix*n1+iz+2] += (-dp * c2) * adj_val;
				vz[ix*n1+iz-1] += ( dp * c2) * adj_val;

				tzz[ix*n1+iz] = 0.0;
			}
		}
}

		/* ----- Adjoint of txz antisymmetric ----- */
		/* Forward: txz[ix*n1+iz] = -txz[(ix+1)*n1+iz]
		 *          txz[(ix-1)*n1+iz] = -txz[(ix+2)*n1+iz] */
#pragma omp for private (iz, adj_val, adj_val2)
		for (iz=mod.ioTz; iz<mod.ieTz; iz++) {
			adj_val  = txz[ix*n1+iz];
			adj_val2 = txz[(ix-1)*n1+iz];

			txz[(ix+1)*n1+iz] -= adj_val;
			txz[(ix+2)*n1+iz] -= adj_val2;

			txz[ix*n1+iz]     = 0.0;
			txz[(ix-1)*n1+iz] = 0.0;
		}

		/* ----- Adjoint of txx = 0 ----- */
#pragma omp for private (iz)
		for (iz=mod.ioPz; iz<mod.iePz; iz++) {
			txx[ix*n1+iz] = 0.0;
		}
	}

	/* ================================================================ */
	/* Zero adj_vx/adj_vz at virtual boundary positions.                */
	/*                                                                  */
	/* The free surface txx scatter (adjoint of txx = -dp*D+x(vx))     */
	/* writes to adj_vx at ix < ibnd and ix >= nax-ibnd, where          */
	/* ibnd = iorder/2. These positions are never updated by the        */
	/* forward operator (they lie outside the grid padding), so the     */
	/* adjoint values must be discarded to prevent Phase A2 from        */
	/* reading spurious values and corrupting the valid domain.         */
	/*                                                                  */
	/* Similarly for adj_vz at virtual z-boundary positions when        */
	/* left/right free surfaces or top free surface with iorder>=8.     */
	/* ================================================================ */
	{
		int ibnd = mod.iorder/2;

		/* adj_vx: virtual x-boundary positions from TOP/BOTTOM free surface scatter */
		if (bnd.top==1 || bnd.bot==1) {
#pragma omp for private (ix, iz) nowait
			for (ix=0; ix<ibnd; ix++)
				for (iz=0; iz<n1; iz++)
					vx[ix*n1+iz] = 0.0;
#pragma omp for private (ix, iz) nowait
			for (ix=mod.nax-ibnd; ix<mod.nax; ix++)
				for (iz=0; iz<n1; iz++)
					vx[ix*n1+iz] = 0.0;
		}

		/* adj_vz: virtual z-boundary positions from LEFT/RIGHT free surface scatter */
		if (bnd.lef==1 || bnd.rig==1) {
#pragma omp for private (ix, iz) nowait
			for (ix=0; ix<mod.nax; ix++)
				for (iz=0; iz<ibnd-1; iz++)
					vz[ix*n1+iz] = 0.0;
#pragma omp for private (ix, iz) nowait
			for (ix=0; ix<mod.nax; ix++)
				for (iz=mod.naz-ibnd+1; iz<mod.naz; iz++)
					vz[ix*n1+iz] = 0.0;
		}

		/* Top free surface with iorder>=8: BV_adj c4 correction writes
		 * adj_vz at iz=surface-2 which is a virtual z position */
		if (bnd.top==1 && mod.iorder >= 8) {
#pragma omp for private (ix, iz)
			for (ix=0; ix<mod.nax; ix++)
				for (iz=0; iz<ibnd-1; iz++)
					vz[ix*n1+iz] = 0.0;
		}
	}

	return 0;
}
