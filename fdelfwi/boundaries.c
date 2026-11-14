#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<assert.h>
#include"fdelfwi.h"

void vmess(char *fmt, ...);

int boundariesP(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int itime, int verbose)
{
/*********************************************************************

   AUTHOR:
		   Jan Thorbecke (janth@xs4all.nl)
		   The Netherlands 

***********************************************************************/

	float c1, c2, c3, c4;
	float dp, dvx, dvz;
	int   ix, iz, ixs, izs, ibnd, ib, ibx, ibz;
	int   nx, nz, n1, n2;
	int   is0, isrc;
	int   ixo, ixe, izo, ize;
    int   npml, pml;
    int   has_lef_pml, has_rig_pml, has_top_pml, has_bot_pml;
    int   zmin, zmax, zlo, zhi;
    float dpx, dpz, *p;
    static float *psi_vx_x = NULL, *psi_vz_z = NULL;
    static float *psi_vx_z = NULL, *psi_vz_x = NULL;
    static float *b_xf = NULL, *c_xf = NULL, *ik_xf = NULL;
    static float *b_zf = NULL, *c_zf = NULL, *ik_zf = NULL;
	static int allocated=0;

	nx  = mod.nx;
    nz  = mod.nz;
    n1  = mod.naz;
    n2  = mod.nax;
    has_lef_pml = (bnd.lef==2);
    has_rig_pml = (bnd.rig==2);
    has_top_pml = (bnd.top==2);
    has_bot_pml = (bnd.bot==2);
    pml = (has_top_pml || has_bot_pml || has_lef_pml || has_rig_pml);

	ibnd = mod.iorder/2-1;

	if (mod.iorder <= 4) {
		c1 = 9.0/8.0;  c2 = -1.0/24.0;  c3 = 0.0;  c4 = 0.0;
	} else if (mod.iorder == 6) {
		c1 = 75.0/64.0;  c2 = -25.0/384.0;  c3 = 3.0/640.0;  c4 = 0.0;
	} else { /* iorder >= 8 */
		c1 = 1225.0/1024.0;  c2 = -245.0/3072.0;  c3 = 49.0/5120.0;  c4 = -5.0/7168.0;
	}

	if (mod.ischeme <= 2) { /* Acoustic scheme */
		if (bnd.top==1) { /* free surface at top */
#pragma omp	for private (ix) nowait
			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
				iz = bnd.surface[ix];
				//fprintf(stderr,"free iz=%d\n", iz);
				vz[ix*n1+iz]   = vz[ix*n1+iz+1];
				vz[ix*n1+iz-1] = vz[ix*n1+iz+2];
			}
		}
//		if (bnd.rig==1) { /* free surface at right */
//#pragma omp	for private (iz) nowait
//			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
//				tzz[(mod.iePx-1)*n1+iz] = 0.0;
//			}
//		}
//		if (bnd.bot==1) { /* free surface at bottom */
//#pragma omp	for private (ix) nowait
//			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
//				tzz[ix*n1+mod.iePz-1] = 0.0;
//			}
//		}
//		if (bnd.lef==1) { /* free surface at left */
//#pragma omp	for private (iz) nowait
//			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
//				tzz[(mod.ioPx-1)*n1+iz] = 0.0;
//			}
//		}
	}

/************************************************************/
/* rigid boundary condition clears velocities on boundaries */
/************************************************************/

	if (bnd.top==3) { /* rigid surface at top */
#pragma omp for private (ix, iz) nowait
#pragma simd
		for (ix=1; ix<=nx; ix++) {
			//vx[ix*n1+ibnd] = 0.0;
			vz[ix*n1+ibnd] = -vz[ix*n1+ibnd+1];
            for (ib=1; ib<=ibnd; ib++) { 
                vz[ix*n1+ibnd-ib] = -vz[ix*n1+ibnd+1+ib];
            }
            /*
			if (mod.iorder >= 4) vz[ix*n1+ibnd-1] = -vz[ix*n1+ibnd+2];
			if (mod.iorder >= 6) vz[ix*n1+ibnd-2] = -vz[ix*n1+ibnd+3];
            */
		}
	}
	if (bnd.rig==3) { /* rigid surface at right */
#pragma omp for private (ix, iz) nowait
#pragma simd
		for (iz=1; iz<=nz; iz++) {
			//vz[(nx+ibnd-1)*n1+iz] = 0.0;
			vx[(nx+ibnd)*n1+iz]   = -vx[(nx+ibnd-1)*n1+iz];
            for (ib=1; ib<=ibnd; ib++) { 
                vx[(nx+ibnd+ib)*n1+iz] = -vx[(nx-ib)*n1+iz];
            }
			/*if (mod.iorder == 4) vx[(nx+2)*n1+iz] = -vx[(nx-1)*n1+iz];
			if (mod.iorder == 6) {
				vx[(nx+1)*n1+iz] = -vx[(nx)*n1+iz];
				vx[(nx+3)*n1+iz] = -vx[(nx-2)*n1+iz];
			}*/
		}
	}
	if (bnd.bot==3) { /* rigid surface at bottom */
#pragma omp for private (ix, iz) nowait
#pragma simd
		for (ix=1; ix<=nx; ix++) {
			//vx[ix*n1+nz+ibnd-1] = 0.0;
			vz[ix*n1+nz+ibnd]   = -vz[ix*n1+nz+ibnd-1];
            for (ib=1; ib<=ibnd; ib++) { 
                vz[ix*n1+nz+ibnd+ib] = -vz[ix*n1+nz-ib];
            }
			/*if (mod.iorder == 4) vz[ix*n1+nz+2] = -vz[ix*n1+nz-1];
			if (mod.iorder == 6) {
				vz[ix*n1+nz+1] = -vz[ix*n1+nz];
				vz[ix*n1+nz+3] = -vz[ix*n1+nz-2];
			}*/
		}
	}
	if (bnd.lef==3) { /* rigid surface at left */
#pragma omp for private (ix, iz) nowait
#pragma simd
		for (iz=1; iz<=nz; iz++) {
			//vz[ibnd*n1+iz] = 0.0;
			vx[ibnd*n1+iz] = -vx[(ibnd+1)*n1+iz];
            for (ib=1; ib<=ibnd; ib++) { 
                vx[(ib-ibnd)*n1+iz] = -vx[(ibnd+1+ib)*n1+iz];
            }
			/*if (mod.iorder == 4) vx[0*n1+iz] = -vx[3*n1+iz];
			if (mod.iorder == 6) {
				vx[1*n1+iz] = -vx[4*n1+iz];
				vx[0*n1+iz] = -vx[5*n1+iz];
			}*/
		}
	}

    

/************************************************************/
/* PML boundaries : CFS-CPML for acoustic and elastic     */
/************************************************************/

    npml=bnd.npml; /* length of pml in grid-points */
    if ( (npml != 0) && (allocated==0) && pml) {
#pragma omp master
{
        if (b_xf) free(b_xf); if (c_xf) free(c_xf); if (ik_xf) free(ik_xf);
        if (b_zf) free(b_zf); if (c_zf) free(c_zf); if (ik_zf) free(ik_zf);

        /* psi is the FORWARD STATE and is owned by cpml_state.c, not by this
         * function, so that checkpoint.c can save and restore it.  The call
         * is idempotent: it keeps the existing buffers when the grid size is
         * unchanged, which is what lets a restored checkpoint survive until
         * the first time step of the reconstruction. */
        {
            float **psi = cpmlStateAlloc((size_t)n2*n1);
            psi_vx_x = psi[CPML_PSI_VX_X];
            psi_vz_z = psi[CPML_PSI_VZ_Z];
            psi_vx_z = psi[CPML_PSI_VX_Z];
            psi_vz_x = psi[CPML_PSI_VZ_X];
        }
        b_xf  = (float *)calloc(n2, sizeof(float));
        c_xf  = (float *)calloc(n2, sizeof(float));
        ik_xf = (float *)calloc(n2, sizeof(float));
        b_zf  = (float *)calloc(n1, sizeof(float));
        c_zf  = (float *)calloc(n1, sizeof(float));
        ik_zf = (float *)calloc(n1, sizeof(float));
        allocated = 1;

        /* Face-staggered CFS-CPML profile. Shared with the adjoint via
         * cpml_coef.c -- the discrete transpose is only a transpose if it
         * uses bit-identical b/c/ik, so there must be ONE builder. */
        cpmlCoefVel(mod, bnd, b_xf, c_xf, ik_xf, b_zf, c_zf, ik_zf);

        if (verbose>=4) vmess("CFS-CPML boundP: cp_min=%e npml=%d kmax=%e amax=%e",
                              mod.cp_min, npml, bnd.kmax, bnd.amax);
}
    }
#pragma omp barrier

	if (mod.ischeme == 1 && pml) { /* Acoustic CFS-CPML */
        p = tzz;

        if (itime == 0) {
#pragma omp master
{
            memset(psi_vx_x, 0, (size_t)n2*n1*sizeof(float));
            memset(psi_vz_z, 0, (size_t)n2*n1*sizeof(float));
            memset(psi_vx_z, 0, (size_t)n2*n1*sizeof(float));
            memset(psi_vz_x, 0, (size_t)n2*n1*sizeof(float));
}
#pragma omp barrier
        }

        /* unified-mask CPML update: one loop per velocity component */
        zmin = ((mod.ioXz-npml)>0?(mod.ioXz-npml):0);
        zmax = ((mod.ieXz+npml)<n1?(mod.ieXz+npml):n1);
#pragma omp for private(ix, iz, dpx) schedule(static)
        for (ix=((mod.ioXx-npml)>0?(mod.ioXx-npml):0); ix<((mod.ieXx+npml)<n2?(mod.ieXx+npml):n2); ix++) {
            int in_x = ((has_lef_pml && ix < mod.ioXx) || (has_rig_pml && ix >= mod.ieXx));

            if (in_x) {
#pragma omp simd
                for (iz=zmin; iz<zmax; iz++) {
                    int i = ix*n1+iz;
                    dpx = c1*(p[i]        - p[i-n1]) +
                          c2*(p[i+n1]     - p[i-2*n1]);
                    psi_vx_x[i] = b_xf[ix]*psi_vx_x[i] + c_xf[ix]*dpx;
                    dpx = ik_xf[ix]*dpx + psi_vx_x[i];
                    vx[i] -= rox[i]*dpx;
                }
            }
            else {
                if (has_top_pml) {
#pragma omp simd
                    for (iz=zmin; iz<mod.ioXz; iz++) {
                        int i = ix*n1+iz;
                        dpx = c1*(p[i]        - p[i-n1]) +
                              c2*(p[i+n1]     - p[i-2*n1]);
                        vx[i] -= rox[i]*dpx;
                    }
                }
                if (has_bot_pml) {
#pragma omp simd
                    for (iz=mod.ieXz; iz<zmax; iz++) {
                        int i = ix*n1+iz;
                        dpx = c1*(p[i]        - p[i-n1]) +
                              c2*(p[i+n1]     - p[i-2*n1]);
                        vx[i] -= rox[i]*dpx;
                    }
                }
            }
        }

        zmin = ((mod.ioZz-npml)>0?(mod.ioZz-npml):0);
        zmax = ((mod.ieZz+npml)<n1?(mod.ieZz+npml):n1);
#pragma omp for private(ix, iz, dpz) schedule(static)
        for (ix=((mod.ioZx-npml)>0?(mod.ioZx-npml):0); ix<((mod.ieZx+npml)<n2?(mod.ieZx+npml):n2); ix++) {
            int in_x = ((has_lef_pml && ix < mod.ioZx) || (has_rig_pml && ix >= mod.ieZx));

            if (in_x) {
                for (iz=zmin; iz<zmax; iz++) {
                    int i = ix*n1+iz;
                    dpz = c1*(p[i]     - p[i-1]) +
                          c2*(p[i+1]   - p[i-2]);
                    if (has_top_pml && iz < mod.ioZz) {
                        psi_vz_z[i] = b_zf[iz]*psi_vz_z[i] + c_zf[iz]*dpz;
                        dpz = ik_zf[iz]*dpz + psi_vz_z[i];
                    }
                    else if (has_bot_pml && iz >= mod.ieZz) {
                        psi_vz_z[i] = b_zf[iz]*psi_vz_z[i] + c_zf[iz]*dpz;
                        dpz = ik_zf[iz]*dpz + psi_vz_z[i];
                    }
                    vz[i] -= roz[i]*dpz;
                }
            }
            else {
                if (has_top_pml) {
#pragma omp simd
                    for (iz=zmin; iz<mod.ioZz; iz++) {
                        int i = ix*n1+iz;
                        dpz = c1*(p[i]     - p[i-1]) +
                              c2*(p[i+1]   - p[i-2]);
                        psi_vz_z[i] = b_zf[iz]*psi_vz_z[i] + c_zf[iz]*dpz;
                        dpz = ik_zf[iz]*dpz + psi_vz_z[i];
                        vz[i] -= roz[i]*dpz;
                    }
                }
                if (has_bot_pml) {
#pragma omp simd
                    for (iz=mod.ieZz; iz<zmax; iz++) {
                        int i = ix*n1+iz;
                        dpz = c1*(p[i]     - p[i-1]) +
                              c2*(p[i+1]   - p[i-2]);
                        psi_vz_z[i] = b_zf[iz]*psi_vz_z[i] + c_zf[iz]*dpz;
                        dpz = ik_zf[iz]*dpz + psi_vz_z[i];
                        vz[i] -= roz[i]*dpz;
                    }
                }
            }
        }

        /* Fill ghost cells before pressure update.
         * Keep 4th-order pressure stencils symmetric on both low/high-index faces. */
        if (bnd.lef == 2) {
#pragma omp for private(iz)
            for (iz=0; iz<n1; iz++) {
                vx[1*n1+iz] = vx[2*n1+iz];
                vx[0*n1+iz] = vx[2*n1+iz];
                vz[1*n1+iz] = vz[2*n1+iz];
                vz[0*n1+iz] = vz[2*n1+iz];
            }
        }
        if (bnd.top == 2) {
#pragma omp for private(ix)
            for (ix=0; ix<n2; ix++) {
                vx[ix*n1+1] = vx[ix*n1+2];
                vx[ix*n1+0] = vx[ix*n1+2];
                vz[ix*n1+1] = vz[ix*n1+2];
                vz[ix*n1+0] = vz[ix*n1+2];
            }
        }

	} /* end acoustic CFS-CPML */

    /* Elastic velocity CFS-CPML. Ported VERBATIM from fdelmodc/boundaries.c:385-568
     * (modPar/bndPar are byte-identical between the two headers, and this block uses
     * only mod.ioXx/ieXx/ioXz/ieXz/ioZx/ieZx/ioZz/ieZz, whose values agree in both
     * codes -- unlike the P/T ranges, so NO index translation is needed here.
     * Keep diffable against fdelmodc. */
    if (mod.ischeme == 3 && pml) { /* Elastic CFS-CPML: full velocity update in PML */
        if (itime == 0) {
#pragma omp master
{
            memset(psi_vx_x, 0, (size_t)n2*n1*sizeof(float));
            memset(psi_vz_z, 0, (size_t)n2*n1*sizeof(float));
            memset(psi_vx_z, 0, (size_t)n2*n1*sizeof(float));
            memset(psi_vz_x, 0, (size_t)n2*n1*sizeof(float));
}
#pragma omp barrier
        }

        /* vx: update all PML cells skipped by elastic4 main kernel */
        zmin = ((mod.ioXz-npml)>0?(mod.ioXz-npml):0);
        zmax = ((mod.ieXz+npml)<n1?(mod.ieXz+npml):n1);
//fprintf(stderr,"ixo=%d ixe=%d zmin=%d zmax=%d\n", ((mod.ioXx-npml)>0?(mod.ioXx-npml):0), ((mod.ieXx+npml)<n2?(mod.ieXx+npml):n2), zmin, zmax);


       if (has_lef_pml) {
#pragma omp for private(ix, iz, dpx, dpz) schedule(static)
            for (ix=mod.ioXx-npml; ix<mod.ioXx; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    int i = ix*n1+iz;
                    dpx = c1*(txx[i]     - txx[i-n1]) +
                          c2*(txx[i+n1]  - txx[i-2*n1]);
                    dpz = c1*(txz[i+1]   - txz[i]) +
                          c2*(txz[i+2]   - txz[i-1]);
                    psi_vx_x[i] = b_xf[ix]*psi_vx_x[i] + c_xf[ix]*dpx;
                    dpx = ik_xf[ix]*dpx + psi_vx_x[i];
                    psi_vx_z[i] = b_zf[iz]*psi_vx_z[i] + c_zf[iz]*dpz;
                    dpz = ik_zf[iz]*dpz + psi_vx_z[i];
                    vx[i] -= rox[i]*(dpx + dpz);
                }
            }
       }
       if (has_rig_pml) {
#pragma omp for private(ix, iz, dpx, dpz) schedule(static)
            for (ix=mod.ieXx; ix<mod.ieXx+npml; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    int i = ix*n1+iz;
                    dpx = c1*(txx[i]     - txx[i-n1]) +
                          c2*(txx[i+n1]  - txx[i-2*n1]);
                    dpz = c1*(txz[i+1]   - txz[i]) +
                          c2*(txz[i+2]   - txz[i-1]);
                    psi_vx_x[i] = b_xf[ix]*psi_vx_x[i] + c_xf[ix]*dpx;
                    dpx = ik_xf[ix]*dpx + psi_vx_x[i];
                    psi_vx_z[i] = b_zf[iz]*psi_vx_z[i] + c_zf[iz]*dpz;
                    dpz = ik_zf[iz]*dpz + psi_vx_z[i];
                    vx[i] -= rox[i]*(dpx + dpz);
                }
            }
       }
       if (has_top_pml) {
#pragma omp for private(ix, iz, dpx, dpz) schedule(static)
            for (ix=mod.ioXx; ix<mod.ieXx; ix++) {
                for (iz=mod.ioXz-npml; iz<mod.ioXz; iz++) {
                    int i = ix*n1+iz;
                    dpx = c1*(txx[i]     - txx[i-n1]) +
                          c2*(txx[i+n1]  - txx[i-2*n1]);
                    dpz = c1*(txz[i+1]   - txz[i]) +
                          c2*(txz[i+2]   - txz[i-1]);
                    psi_vx_x[i] = b_xf[ix]*psi_vx_x[i] + c_xf[ix]*dpx;
                    dpx = ik_xf[ix]*dpx + psi_vx_x[i];
                    psi_vx_z[i] = b_zf[iz]*psi_vx_z[i] + c_zf[iz]*dpz;
                    dpz = ik_zf[iz]*dpz + psi_vx_z[i];
                    vx[i] -= rox[i]*(dpx + dpz);
                }
            }
       }
       if (has_bot_pml) {
#pragma omp for private(ix, iz, dpx, dpz) schedule(static)
            for (ix=mod.ioXx; ix<mod.ieXx; ix++) {
                for (iz=mod.ieXz; iz<mod.ieXz+npml; iz++) {
                    int i = ix*n1+iz;
                    dpx = c1*(txx[i]     - txx[i-n1]) +
                          c2*(txx[i+n1]  - txx[i-2*n1]);
                    dpz = c1*(txz[i+1]   - txz[i]) +
                          c2*(txz[i+2]   - txz[i-1]);
                    psi_vx_x[i] = b_xf[ix]*psi_vx_x[i] + c_xf[ix]*dpx;
                    dpx = ik_xf[ix]*dpx + psi_vx_x[i];
                    psi_vx_z[i] = b_zf[iz]*psi_vx_z[i] + c_zf[iz]*dpz;
                    dpz = ik_zf[iz]*dpz + psi_vx_z[i];
                    vx[i] -= rox[i]*(dpx + dpz);
                }
            }
       }

        /* vz: update all PML cells skipped by elastic4 main kernel */
        zmin = ((mod.ioZz-npml)>0?(mod.ioZz-npml):0);
        zmax = ((mod.ieZz+npml)<n1?(mod.ieZz+npml):n1);
        if (has_lef_pml) {
#pragma omp for private(ix, iz, dpx, dpz) schedule(static)
            for (ix=mod.ioZx-npml; ix<mod.ioZx; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    int i = ix*n1+iz;
                    dpz = c1*(tzz[i]   - tzz[i-1]) +
                          c2*(tzz[i+1] - tzz[i-2]);
                    psi_vz_z[i] = b_zf[iz]*psi_vz_z[i] + c_zf[iz]*dpz;
                    dpz = ik_zf[iz]*dpz + psi_vz_z[i];
    
                    dpx = c1*(txz[i+n1]   - txz[i]) +
                          c2*(txz[i+2*n1] - txz[i-n1]);
                    psi_vz_x[i] = b_xf[ix]*psi_vz_x[i] + c_xf[ix]*dpx;
                    dpx = ik_xf[ix]*dpx + psi_vz_x[i];
                    vz[i] -= roz[i]*(dpz + dpx);
                }
            }
        }
        if (has_rig_pml) {
#pragma omp for private(ix, iz, dpx, dpz) schedule(static)
            for (ix=mod.ieZx; ix<mod.ieZx+npml; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    int i = ix*n1+iz;
                    dpz = c1*(tzz[i]   - tzz[i-1]) +
                          c2*(tzz[i+1] - tzz[i-2]);
                    psi_vz_z[i] = b_zf[iz]*psi_vz_z[i] + c_zf[iz]*dpz;
                    dpz = ik_zf[iz]*dpz + psi_vz_z[i];
    
                    dpx = c1*(txz[i+n1]   - txz[i]) +
                          c2*(txz[i+2*n1] - txz[i-n1]);
                    psi_vz_x[i] = b_xf[ix]*psi_vz_x[i] + c_xf[ix]*dpx;
                    dpx = ik_xf[ix]*dpx + psi_vz_x[i];
                    vz[i] -= roz[i]*(dpz + dpx);
                }
            }
        }
        if (has_top_pml) {
#pragma omp for private(ix, iz, dpx, dpz) schedule(static)
            for (ix=mod.ioZx; ix<mod.ieZx; ix++) {
                for (iz=mod.ioZz-npml; iz<mod.ioZz; iz++) {
                    int i = ix*n1+iz;
                    dpz = c1*(tzz[i]   - tzz[i-1]) +
                          c2*(tzz[i+1] - tzz[i-2]);
                    psi_vz_z[i] = b_zf[iz]*psi_vz_z[i] + c_zf[iz]*dpz;
                    dpz = ik_zf[iz]*dpz + psi_vz_z[i];
    
                    dpx = c1*(txz[i+n1]   - txz[i]) +
                          c2*(txz[i+2*n1] - txz[i-n1]);
                    psi_vz_x[i] = b_xf[ix]*psi_vz_x[i] + c_xf[ix]*dpx;
                    dpx = ik_xf[ix]*dpx + psi_vz_x[i];
                    vz[i] -= roz[i]*(dpz + dpx);
                }
            }
        }
        if (has_bot_pml) {
#pragma omp for private(ix, iz, dpx, dpz) schedule(static)
            for (ix=mod.ioZx; ix<mod.ieZx; ix++) {
                for (iz=mod.ieZz; iz<mod.ieZz+npml; iz++) {
                    int i = ix*n1+iz;
                    dpz = c1*(tzz[i]   - tzz[i-1]) +
                          c2*(tzz[i+1] - tzz[i-2]);
                    psi_vz_z[i] = b_zf[iz]*psi_vz_z[i] + c_zf[iz]*dpz;
                    dpz = ik_zf[iz]*dpz + psi_vz_z[i];
    
                    dpx = c1*(txz[i+n1]   - txz[i]) +
                          c2*(txz[i+2*n1] - txz[i-n1]);
                    psi_vz_x[i] = b_xf[ix]*psi_vz_x[i] + c_xf[ix]*dpx;
                    dpx = ik_xf[ix]*dpx + psi_vz_x[i];
                    vz[i] -= roz[i]*(dpz + dpx);
                }
            }
        }

        /* Fill outer ghost cells with linear extrapolation so 4th-order stress
         * stencils in boundariesV have symmetric closure at all four sides. */
        if (bnd.lef == 2) {
#pragma omp for private(iz)
            for (iz=0; iz<n1; iz++) {
                vx[1*n1+iz] = vx[2*n1+iz];
                vx[0*n1+iz] = vx[2*n1+iz];
                vz[1*n1+iz] = vz[2*n1+iz];
                vz[0*n1+iz] = vz[2*n1+iz];
            }
        }
        if (bnd.top == 2) {
#pragma omp for private(ix)
            for (ix=0; ix<n2; ix++) {
                vx[ix*n1+1] = vx[ix*n1+2];
                vx[ix*n1+0] = vx[ix*n1+2];
                vz[ix*n1+1] = vz[ix*n1+2];
                vz[ix*n1+0] = vz[ix*n1+2];
            }
        }
    } /* end elastic CFS-CPML */
  
    
    
    
/************************************************************/
/* Tapered boundaries for both elastic and acoustic schemes */
/* compute all field values in tapered areas				*/
/************************************************************/

	/*********/
	/*  Top  */
	/*********/
	if (bnd.top==4) {
		
		if (mod.ischeme <= 2) { /* Acoustic scheme */
			
			/* Vx field */
			ixo = mod.ioXx;
			ixe = mod.ieXx;
			izo = mod.ioXz-bnd.ntap;
			ize = mod.ioXz;
	
			ib = (bnd.ntap+izo-1);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c1*(tzz[ix*n1+iz]	 - tzz[(ix-1)*n1+iz]) +
									c2*(tzz[(ix+1)*n1+iz] - tzz[(ix-2)*n1+iz]));

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
						vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c1*(tzz[ix*n1+iz]	 - tzz[(ix-1)*n1+iz]) +
									c2*(tzz[(ix+1)*n1+iz] - tzz[(ix-2)*n1+iz]));
	
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
						vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c1*(tzz[ix*n1+iz]	 - tzz[(ix-1)*n1+iz]) +
									c2*(tzz[(ix+1)*n1+iz] - tzz[(ix-2)*n1+iz]));
						
						vx[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(ibz-iz)];
					}
				}
			}

			
			/* Vz field */
			ixo = mod.ioZx;
			ixe = mod.ieZx;
			izo = mod.ioZz-bnd.ntap;
			ize = mod.ioZz;
	
			ib = (bnd.ntap+izo-1);
#pragma omp for private (ix, iz) 
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]   - tzz[ix*n1+iz-1]) +
								c2*(tzz[ix*n1+iz+1] - tzz[ix*n1+iz-2]));

					vz[ix*n1+iz] *= bnd.tapz[ib-iz];
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
						vz[ix*n1+iz] -= roz[ix*n1+iz]*(
									c1*(tzz[ix*n1+iz]   - tzz[ix*n1+iz-1]) +
									c2*(tzz[ix*n1+iz+1] - tzz[ix*n1+iz-2]));
	
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
						vz[ix*n1+iz] -= roz[ix*n1+iz]*(
									c1*(tzz[ix*n1+iz]   - tzz[ix*n1+iz-1]) +
									c2*(tzz[ix*n1+iz+1] - tzz[ix*n1+iz-2]));
						
						vz[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(ibz-iz)];
					}
				}
			}

		}
		else { /* Elastic scheme */
			
			/* Vx field */
			ixo = mod.ioXx;
			ixe = mod.ieXx;
			izo = mod.ioXz-bnd.ntap;
			ize = mod.ioXz;

			ib = (bnd.ntap+izo-1);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c1*(txx[ix*n1+iz]	 - txx[(ix-1)*n1+iz] +
									txz[ix*n1+iz+1]   - txz[ix*n1+iz])	+
								c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
									txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
					if (mod.iorder >= 6) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c3*(txx[(ix+2)*n1+iz] - txx[(ix-3)*n1+iz] +
								    txz[ix*n1+iz+3]   - txz[ix*n1+iz-2])  );
					if (mod.iorder >= 8) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c4*(txx[(ix+3)*n1+iz] - txx[(ix-4)*n1+iz] +
								    txz[ix*n1+iz+4]   - txz[ix*n1+iz-3])  );

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
						vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c1*(txx[ix*n1+iz]	 - txx[(ix-1)*n1+iz] +
										txz[ix*n1+iz+1]   - txz[ix*n1+iz])	+
									c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
										txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
						if (mod.iorder >= 6) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c3*(txx[(ix+2)*n1+iz] - txx[(ix-3)*n1+iz] +
									    txz[ix*n1+iz+3]   - txz[ix*n1+iz-2])  );
						if (mod.iorder >= 8) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c4*(txx[(ix+3)*n1+iz] - txx[(ix-4)*n1+iz] +
									    txz[ix*n1+iz+4]   - txz[ix*n1+iz-3])  );
	
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
						vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c1*(txx[ix*n1+iz]	 - txx[(ix-1)*n1+iz] +
										txz[ix*n1+iz+1]   - txz[ix*n1+iz])	+
									c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
										txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
						if (mod.iorder >= 6) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c3*(txx[(ix+2)*n1+iz] - txx[(ix-3)*n1+iz] +
									    txz[ix*n1+iz+3]   - txz[ix*n1+iz-2])  );
						if (mod.iorder >= 8) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c4*(txx[(ix+3)*n1+iz] - txx[(ix-4)*n1+iz] +
									    txz[ix*n1+iz+4]   - txz[ix*n1+iz-3])  );
						
						vx[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(ibz-iz)];
					}
				}
			}

			/* Vz field */
			ixo = mod.ioZx;
			ixe = mod.ieZx;
			izo = mod.ioZz-bnd.ntap;
			ize = mod.ioZz;
	
			ib = (bnd.ntap+izo-1);
#pragma omp for private (ix, iz) 
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[ix*n1+iz-1] +
									txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
								c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
									txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
					if (mod.iorder >= 6) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c3*(tzz[ix*n1+iz+2]   - tzz[ix*n1+iz-3] +
								    txz[(ix+3)*n1+iz] - txz[(ix-2)*n1+iz])  );
					if (mod.iorder >= 8) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c4*(tzz[ix*n1+iz+3]   - tzz[ix*n1+iz-4] +
								    txz[(ix+4)*n1+iz] - txz[(ix-3)*n1+iz])  );

					vz[ix*n1+iz] *= bnd.tapz[ib-iz];
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
					vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[ix*n1+iz-1] +
									txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
								c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
									txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
					if (mod.iorder >= 6) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c3*(tzz[ix*n1+iz+2]   - tzz[ix*n1+iz-3] +
								    txz[(ix+3)*n1+iz] - txz[(ix-2)*n1+iz])  );
					if (mod.iorder >= 8) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c4*(tzz[ix*n1+iz+3]   - tzz[ix*n1+iz-4] +
								    txz[(ix+4)*n1+iz] - txz[(ix-3)*n1+iz])  );
	
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
						vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[ix*n1+iz-1] +
									txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
								c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
									txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
						if (mod.iorder >= 6) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c3*(tzz[ix*n1+iz+2]   - tzz[ix*n1+iz-3] +
								    txz[(ix+3)*n1+iz] - txz[(ix-2)*n1+iz])  );
						if (mod.iorder >= 8) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c4*(tzz[ix*n1+iz+3]   - tzz[ix*n1+iz-4] +
								    txz[(ix+4)*n1+iz] - txz[(ix-3)*n1+iz])  );
						
						vz[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(ibz-iz)];
					}
				}
			}

		
		} /* end elastic scheme */
	}
	
	/*********/
	/* Bottom */
	/*********/
	if (bnd.bot==4) {
		
		if (mod.ischeme <= 2) { /* Acoustic scheme */
			
			/* Vx field */
			ixo = mod.ioXx;
			ixe = mod.ieXx;
			izo = mod.ieXz;
			ize = mod.ieXz+bnd.ntap;
			
			ib = (ize-bnd.ntap);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[(ix-1)*n1+iz]) +
								c2*(tzz[(ix+1)*n1+iz] - tzz[(ix-2)*n1+iz]));
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
						vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c1*(tzz[ix*n1+iz]	 - tzz[(ix-1)*n1+iz]) +
									c2*(tzz[(ix+1)*n1+iz] - tzz[(ix-2)*n1+iz]));
	
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
						vx[ix*n1+iz] -= rox[ix*n1+iz]*(
									c1*(tzz[ix*n1+iz]	 - tzz[(ix-1)*n1+iz]) +
									c2*(tzz[(ix+1)*n1+iz] - tzz[(ix-2)*n1+iz]));
						
						vx[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(iz-ibz)];
					}
				}
			}


			/* Vz field */
			ixo = mod.ioZx;
			ixe = mod.ieZx;
			izo = mod.ieZz;
			ize = mod.ieZz+bnd.ntap;
			
			ib = (ize-bnd.ntap);
#pragma omp for private (ix, iz) 
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]   - tzz[ix*n1+iz-1]) +
								c2*(tzz[ix*n1+iz+1] - tzz[ix*n1+iz-2]));
					vz[ix*n1+iz] *= bnd.tapz[iz-ib];
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
						vz[ix*n1+iz] -= roz[ix*n1+iz]*(
									c1*(tzz[ix*n1+iz]   - tzz[ix*n1+iz-1]) +
									c2*(tzz[ix*n1+iz+1] - tzz[ix*n1+iz-2]));
						
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
						vz[ix*n1+iz] -= roz[ix*n1+iz]*(
									c1*(tzz[ix*n1+iz]   - tzz[ix*n1+iz-1]) +
									c2*(tzz[ix*n1+iz+1] - tzz[ix*n1+iz-2]));
						
						vz[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(iz-ibz)];
					}
				}
			}
			
  
		}
		else { /* Elastic scheme */

			/* Vx field */
			ixo = mod.ioXx;
			ixe = mod.ieXx;
			izo = mod.ieXz;
			ize = mod.ieXz+bnd.ntap;
			
			ib = (ize-bnd.ntap);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c1*(txx[ix*n1+iz]	 - txx[(ix-1)*n1+iz] +
									txz[ix*n1+iz+1]   - txz[ix*n1+iz])	+
								c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
									txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
					if (mod.iorder >= 6) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c3*(txx[(ix+2)*n1+iz] - txx[(ix-3)*n1+iz] +
								    txz[ix*n1+iz+3]   - txz[ix*n1+iz-2])  );
					if (mod.iorder >= 8) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c4*(txx[(ix+3)*n1+iz] - txx[(ix-4)*n1+iz] +
								    txz[ix*n1+iz+4]   - txz[ix*n1+iz-3])  );

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
						vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c1*(txx[ix*n1+iz]	 - txx[(ix-1)*n1+iz] +
									txz[ix*n1+iz+1]   - txz[ix*n1+iz])	+
								c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
									txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
						if (mod.iorder >= 6) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c3*(txx[(ix+2)*n1+iz] - txx[(ix-3)*n1+iz] +
								    txz[ix*n1+iz+3]   - txz[ix*n1+iz-2])  );
						if (mod.iorder >= 8) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c4*(txx[(ix+3)*n1+iz] - txx[(ix-4)*n1+iz] +
								    txz[ix*n1+iz+4]   - txz[ix*n1+iz-3])  );
	
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
						vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c1*(txx[ix*n1+iz]	 - txx[(ix-1)*n1+iz] +
									txz[ix*n1+iz+1]   - txz[ix*n1+iz])	+
								c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
									txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
						if (mod.iorder >= 6) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c3*(txx[(ix+2)*n1+iz] - txx[(ix-3)*n1+iz] +
								    txz[ix*n1+iz+3]   - txz[ix*n1+iz-2])  );
						if (mod.iorder >= 8) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c4*(txx[(ix+3)*n1+iz] - txx[(ix-4)*n1+iz] +
								    txz[ix*n1+iz+4]   - txz[ix*n1+iz-3])  );
						
						vx[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(iz-ibz)];
					}
				}
			}
	
			/* Vz field */
			ixo = mod.ioZx;
			ixe = mod.ieZx;
			izo = mod.ieZz;
			ize = mod.ieZz+bnd.ntap;
			
			ib = (ize-bnd.ntap);
#pragma omp for private (ix, iz) 
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[ix*n1+iz-1] +
									txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
								c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
									txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
					if (mod.iorder >= 6) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c3*(tzz[ix*n1+iz+2]   - tzz[ix*n1+iz-3] +
								    txz[(ix+3)*n1+iz] - txz[(ix-2)*n1+iz])  );
					if (mod.iorder >= 8) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c4*(tzz[ix*n1+iz+3]   - tzz[ix*n1+iz-4] +
								    txz[(ix+4)*n1+iz] - txz[(ix-3)*n1+iz])  );

					vz[ix*n1+iz] *= bnd.tapz[iz-ib];
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
						vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[ix*n1+iz-1] +
									txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
								c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
									txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
						if (mod.iorder >= 6) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c3*(tzz[ix*n1+iz+2]   - tzz[ix*n1+iz-3] +
								    txz[(ix+3)*n1+iz] - txz[(ix-2)*n1+iz])  );
						if (mod.iorder >= 8) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c4*(tzz[ix*n1+iz+3]   - tzz[ix*n1+iz-4] +
								    txz[(ix+4)*n1+iz] - txz[(ix-3)*n1+iz])  );
						
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
						vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[ix*n1+iz-1] +
									txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
								c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
									txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
						if (mod.iorder >= 6) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c3*(tzz[ix*n1+iz+2]   - tzz[ix*n1+iz-3] +
								    txz[(ix+3)*n1+iz] - txz[(ix-2)*n1+iz])  );
						if (mod.iorder >= 8) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c4*(tzz[ix*n1+iz+3]   - tzz[ix*n1+iz-4] +
								    txz[(ix+4)*n1+iz] - txz[(ix-3)*n1+iz])  );
						
						vz[ix*n1+iz]   *= bnd.tapxz[(ibx-ix)*bnd.ntap+(iz-ibz)];
					}
				}
			}
 
			
		} /* end elastic scheme */
		
	}
	
	/*********/
	/* Left  */
	/*********/
	if (bnd.lef==4) {
		
		if (mod.ischeme <= 2) { /* Acoustic scheme */
			
			/* Vx field */
			ixo = mod.ioXx-bnd.ntap;
			ixe = mod.ioXx;
			izo = mod.ioXz;
			ize = mod.ieXz;
			
			ib = (bnd.ntap+ixo-1);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[(ix-1)*n1+iz]) +
								c2*(tzz[(ix+1)*n1+iz] - tzz[(ix-2)*n1+iz]));
					
					vx[ix*n1+iz]   *= bnd.tapx[ib-ix];
				}
			}
			
			/* Vz field */
			ixo = mod.ioZx-bnd.ntap;
			ixe = mod.ioZx;
			izo = mod.ioZz;
			ize = mod.ieZz;

			ib = (bnd.ntap+ixo-1);
#pragma omp for private (ix, iz) 
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]   - tzz[ix*n1+iz-1]) +
								c2*(tzz[ix*n1+iz+1] - tzz[ix*n1+iz-2]));
					
					vz[ix*n1+iz] *= bnd.tapz[ib-ix];
				}
			}

		}
		else { /* Elastic scheme */
			
			/* Vx field */
			ixo = mod.ioXx-bnd.ntap;
			ixe = mod.ioXx;
			izo = mod.ioXz;
			ize = mod.ieXz;
			
			ib = (bnd.ntap+ixo-1);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c1*(txx[ix*n1+iz]	 - txx[(ix-1)*n1+iz] +
									txz[ix*n1+iz+1]   - txz[ix*n1+iz])	+
								c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
									txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
					if (mod.iorder >= 6) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c3*(txx[(ix+2)*n1+iz] - txx[(ix-3)*n1+iz] +
								    txz[ix*n1+iz+3]   - txz[ix*n1+iz-2])  );
					if (mod.iorder >= 8) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c4*(txx[(ix+3)*n1+iz] - txx[(ix-4)*n1+iz] +
								    txz[ix*n1+iz+4]   - txz[ix*n1+iz-3])  );
					
					vx[ix*n1+iz]   *= bnd.tapx[ib-ix];
				}
			}
			
			/* Vz field */
			ixo = mod.ioZx-bnd.ntap;
			ixe = mod.ioZx;
			izo = mod.ioZz;
			ize = mod.ieZz;
			
			ib = (bnd.ntap+ixo-1);
#pragma omp for private (ix, iz) 
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[ix*n1+iz-1] +
									txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
								c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
									txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
					if (mod.iorder >= 6) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c3*(tzz[ix*n1+iz+2]   - tzz[ix*n1+iz-3] +
								    txz[(ix+3)*n1+iz] - txz[(ix-2)*n1+iz])  );
					if (mod.iorder >= 8) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c4*(tzz[ix*n1+iz+3]   - tzz[ix*n1+iz-4] +
								    txz[(ix+4)*n1+iz] - txz[(ix-3)*n1+iz])  );
					
					vz[ix*n1+iz] *= bnd.tapz[ib-ix];
				}
			}
		} /* end elastic scheme */
		
	}

	/*********/
	/* Right */
	/*********/
	if (bnd.rig==4) {
		
		if (mod.ischeme <= 2) { /* Acoustic scheme */
			
			/* Vx field */
			ixo = mod.ieXx;
			ixe = mod.ieXx+bnd.ntap;
			izo = mod.ioXz;
			ize = mod.ieXz;
		
			ib = (ixe-bnd.ntap);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[(ix-1)*n1+iz]) +
								c2*(tzz[(ix+1)*n1+iz] - tzz[(ix-2)*n1+iz]));
	
					vx[ix*n1+iz]   *= bnd.tapx[ix-ib];
				}
			}
		
			/* Vz field */
			ixo = mod.ieZx;
			ixe = mod.ieZx+bnd.ntap;
			izo = mod.ioZz;
			ize = mod.ieZz;
			
			ib = (ixe-bnd.ntap);
#pragma omp for private (ix, iz) 
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]   - tzz[ix*n1+iz-1]) +
								c2*(tzz[ix*n1+iz+1] - tzz[ix*n1+iz-2]));
	
					vz[ix*n1+iz] *= bnd.tapz[ix-ib];
				}
			}
		
		}
		else { /* Elastic scheme */
			
			/* Vx field */
			ixo = mod.ieXx;
			ixe = mod.ieXx+bnd.ntap;
			izo = mod.ioXz;
			ize = mod.ieXz;
			
			ib = (ixe-bnd.ntap);
#pragma omp for private(ix,iz)
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c1*(txx[ix*n1+iz]	 - txx[(ix-1)*n1+iz] +
									txz[ix*n1+iz+1]   - txz[ix*n1+iz])	+
								c2*(txx[(ix+1)*n1+iz] - txx[(ix-2)*n1+iz] +
									txz[ix*n1+iz+2]   - txz[ix*n1+iz-1])  );
					if (mod.iorder >= 6) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c3*(txx[(ix+2)*n1+iz] - txx[(ix-3)*n1+iz] +
								    txz[ix*n1+iz+3]   - txz[ix*n1+iz-2])  );
					if (mod.iorder >= 8) vx[ix*n1+iz] -= rox[ix*n1+iz]*(
								c4*(txx[(ix+3)*n1+iz] - txx[(ix-4)*n1+iz] +
								    txz[ix*n1+iz+4]   - txz[ix*n1+iz-3])  );
	
					vx[ix*n1+iz]   *= bnd.tapx[ix-ib];
				}
			}
			
			/* Vz field */
			ixo = mod.ieZx;
			ixe = mod.ieZx+bnd.ntap;
			izo = mod.ioZz;
			ize = mod.ieZz;
			ib = (ixe-bnd.ntap);
#pragma omp for private (ix, iz) 
			for (ix=ixo; ix<ixe; ix++) {
#pragma simd
				for (iz=izo; iz<ize; iz++) {
					vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c1*(tzz[ix*n1+iz]	 - tzz[ix*n1+iz-1] +
									txz[(ix+1)*n1+iz] - txz[ix*n1+iz])  +
								c2*(tzz[ix*n1+iz+1]   - tzz[ix*n1+iz-2] +
									txz[(ix+2)*n1+iz] - txz[(ix-1)*n1+iz])  );
					if (mod.iorder >= 6) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c3*(tzz[ix*n1+iz+2]   - tzz[ix*n1+iz-3] +
								    txz[(ix+3)*n1+iz] - txz[(ix-2)*n1+iz])  );
					if (mod.iorder >= 8) vz[ix*n1+iz] -= roz[ix*n1+iz]*(
								c4*(tzz[ix*n1+iz+3]   - tzz[ix*n1+iz-4] +
								    txz[(ix+4)*n1+iz] - txz[(ix-3)*n1+iz])  );
	
					vz[ix*n1+iz] *= bnd.tapz[ix-ib];
				}
			}
/*
			for (ix=ixo-5; ix<ixo+5; ix++) {
				for (iz=0; iz<5; iz++) {
			fprintf(stderr,"edge ix=%d iz=%d vz=%e roz=%e tzz=%e txz=%e txx=%e lam=%e l2m=%e\n", ix, iz, vz[ix*n1+iz], roz[ix*n1+iz],
tzz[ix*n1+iz], txz[ix*n1+iz], txx[ix*n1+iz], lam[ix*n1+iz], l2m[ix*n1+iz]);
				}
			}
*/
		
		} /* end elastic scheme */

	}

    if ( (npml != 0) && (itime==mod.nt-1) && pml) {
#pragma omp master
{
		if (allocated) {
            /* Drop the references but do NOT free: psi belongs to
             * cpml_state.c and must outlive this shot so the adjoint can
             * restore it from a checkpoint.  The next shot gets the same
             * buffers back and zeroes them at itime==0. */
            psi_vx_x = psi_vz_z = psi_vx_z = psi_vz_x = NULL;
            if (b_xf)  { free(b_xf);  b_xf  = NULL; }
            if (c_xf)  { free(c_xf);  c_xf  = NULL; }
            if (ik_xf) { free(ik_xf); ik_xf = NULL; }
            if (b_zf)  { free(b_zf);  b_zf  = NULL; }
            if (c_zf)  { free(c_zf);  c_zf  = NULL; }
            if (ik_zf) { free(ik_zf); ik_zf = NULL; }
			allocated=0;
		}
}
	}

	return 0;
} 
	
int boundariesV(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz, float *txx, float *txz, float *rox, float *roz, float *l2m, float *lam, float *mul, int itime, int verbose)
{
/*********************************************************************
	 
	AUTHOR:
	Jan Thorbecke (janth@xs4all.nl)
	 The Netherlands 
	 
***********************************************************************/

	float c1, c2;
	float dp, dvx, dvz;
	int   ix, iz, izp, ib;
    int   nx, nz, n1, n2;
	int   ixo, ixe, izo, ize;
    int   npml, pml;
    int   has_lef_pml, has_rig_pml, has_top_pml, has_bot_pml;
    float *p;
    static float *psi_p_x = NULL, *psi_p_z = NULL;
    static float *psi_txz_x = NULL, *psi_txz_z = NULL;
    static float *b_xc = NULL, *c_xc = NULL, *ik_xc = NULL;
    static float *b_zc = NULL, *c_zc = NULL, *ik_zc = NULL;
    static float *b_xf = NULL, *c_xf = NULL, *ik_xf = NULL;
    static float *b_zf = NULL, *c_zf = NULL, *ik_zf = NULL;
	static int allocated=0;
    c1 = 9.0/8.0;
    c2 = -1.0/24.0;
    nx  = mod.nx;
    nz  = mod.nz;
    n1  = mod.naz;
    n2  = mod.nax;
    has_lef_pml = (bnd.lef==2);
    has_rig_pml = (bnd.rig==2);
    has_top_pml = (bnd.top==2);
    has_bot_pml = (bnd.bot==2);
    pml = (has_top_pml || has_bot_pml || has_lef_pml || has_rig_pml);

/************************************************************/
/* CFS-CPML for acoustic and elastic stress update        */
/************************************************************/

    npml=bnd.npml; /* length of pml in grid-points */
    if ( (npml != 0) && (allocated==0) && pml) {
#pragma omp master
{
        if (b_xc) free(b_xc); if (c_xc) free(c_xc); if (ik_xc) free(ik_xc);
        if (b_zc) free(b_zc); if (c_zc) free(c_zc); if (ik_zc) free(ik_zc);
        if (b_xf) free(b_xf); if (c_xf) free(c_xf); if (ik_xf) free(ik_xf);
        if (b_zf) free(b_zf); if (c_zf) free(c_zf); if (ik_zf) free(ik_zf);

        /* psi is owned by cpml_state.c -- see the matching block in
         * boundariesP.  Same buffers, different four slots. */
        {
            float **psi = cpmlStateAlloc((size_t)n2*n1);
            psi_p_x   = psi[CPML_PSI_P_X];
            psi_p_z   = psi[CPML_PSI_P_Z];
            psi_txz_x = psi[CPML_PSI_TXZ_X];
            psi_txz_z = psi[CPML_PSI_TXZ_Z];
        }
        b_xc  = (float *)calloc(n2, sizeof(float));
        c_xc  = (float *)calloc(n2, sizeof(float));
        ik_xc = (float *)calloc(n2, sizeof(float));
        b_zc  = (float *)calloc(n1, sizeof(float));
        c_zc  = (float *)calloc(n1, sizeof(float));
        ik_zc = (float *)calloc(n1, sizeof(float));
        b_xf  = (float *)calloc(n2, sizeof(float));
        c_xf  = (float *)calloc(n2, sizeof(float));
        ik_xf = (float *)calloc(n2, sizeof(float));
        b_zf  = (float *)calloc(n1, sizeof(float));
        c_zf  = (float *)calloc(n1, sizeof(float));
        ik_zf = (float *)calloc(n1, sizeof(float));
        allocated = 1;

        /* Cell-centred (txx/tzz) and face-staggered (txz) CFS-CPML profiles.
         * Shared with the adjoint via cpml_coef.c. NOTE the face set built
         * here is NOT the one cpmlCoefVel builds -- the stress update samples
         * xi half a cell off. See cpml_coef.c for why that matters. */
        cpmlCoefStr(mod, bnd, b_xc, c_xc, ik_xc, b_zc, c_zc, ik_zc,
                              b_xf, c_xf, ik_xf, b_zf, c_zf, ik_zf);

        if (verbose>=4) vmess("CFS-CPML boundV: cp_min=%e npml=%d kmax=%e amax=%e",
                              mod.cp_min, npml, bnd.kmax, bnd.amax);
}
    }

#pragma omp barrier
    if (mod.ischeme == 1 && pml) { /* Acoustic CFS-CPML */
        p = tzz;

        if (itime == 0) {
#pragma omp master
{
            memset(psi_p_x,   0, (size_t)n2*n1*sizeof(float));
            memset(psi_p_z,   0, (size_t)n2*n1*sizeof(float));
            memset(psi_txz_x, 0, (size_t)n2*n1*sizeof(float));
            memset(psi_txz_z, 0, (size_t)n2*n1*sizeof(float));
}
#pragma omp barrier
        }

        /* unified-mask CPML update for pressure on P-grid.
         * ioPx_a..iePz_a are the fdelmodc-equivalent (pad-excluded) stress
         * limits; see the translation comment in the coefficient setup above. */
        int ioPx_a = mod.ioPx + (bnd.lef==2 ? npml : 0);
        int iePx_a = mod.iePx - (bnd.rig==2 ? npml : 0);
        int ioPz_a = mod.ioPz + (bnd.top==2 ? npml : 0);
        int iePz_a = mod.iePz - (bnd.bot==2 ? npml : 0);
        int zmin = ((ioPz_a-npml)>0?(ioPz_a-npml):ioPz_a);
        int zmax = ((iePz_a+npml)<n1?(iePz_a+npml):iePz_a);
        if (has_lef_pml) {
            for (ix=ioPx_a-npml; ix<ioPx_a; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
                          c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
                    dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
                          c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
                    psi_p_x[ix*n1+iz] = b_xc[ix]*psi_p_x[ix*n1+iz] + c_xc[ix]*dvx;
                    dvx = ik_xc[ix]*dvx + psi_p_x[ix*n1+iz];
                    psi_p_z[ix*n1+iz] = b_zc[iz]*psi_p_z[ix*n1+iz] + c_zc[iz]*dvz;
                    dvz = ik_zc[iz]*dvz + psi_p_z[ix*n1+iz];
                    p[ix*n1+iz] -= l2m[ix*n1+iz]*(dvx + dvz);
                }
            }
        }
        if (has_rig_pml) {
            for (ix=iePx_a; ix<iePx_a+npml; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
                          c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
                    dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
                          c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
                    psi_p_x[ix*n1+iz] = b_xc[ix]*psi_p_x[ix*n1+iz] + c_xc[ix]*dvx;
                    dvx = ik_xc[ix]*dvx + psi_p_x[ix*n1+iz];
                    psi_p_z[ix*n1+iz] = b_zc[iz]*psi_p_z[ix*n1+iz] + c_zc[iz]*dvz;
                    dvz = ik_zc[iz]*dvz + psi_p_z[ix*n1+iz];
                    p[ix*n1+iz] -= l2m[ix*n1+iz]*(dvx + dvz);
                }
            }
        }
        if (has_top_pml) {
            for (ix=ioPx_a; ix<iePx_a; ix++) {
                for (iz=ioPz_a-npml; iz<ioPz_a; iz++) {
                    dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
                          c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
                    dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
                          c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
                    psi_p_x[ix*n1+iz] = b_xc[ix]*psi_p_x[ix*n1+iz] + c_xc[ix]*dvx;
                    dvx = ik_xc[ix]*dvx + psi_p_x[ix*n1+iz];
                    psi_p_z[ix*n1+iz] = b_zc[iz]*psi_p_z[ix*n1+iz] + c_zc[iz]*dvz;
                    dvz = ik_zc[iz]*dvz + psi_p_z[ix*n1+iz];
                    p[ix*n1+iz] -= l2m[ix*n1+iz]*(dvx + dvz);
                }
            }
        }
        if (has_bot_pml) {
            for (ix=ioPx_a; ix<iePx_a; ix++) {
                for (iz=iePz_a; iz<iePz_a+npml; iz++) {
                    dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
                          c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
                    dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
                          c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
                    psi_p_x[ix*n1+iz] = b_xc[ix]*psi_p_x[ix*n1+iz] + c_xc[ix]*dvx;
                    dvx = ik_xc[ix]*dvx + psi_p_x[ix*n1+iz];
                    psi_p_z[ix*n1+iz] = b_zc[iz]*psi_p_z[ix*n1+iz] + c_zc[iz]*dvz;
                    dvz = ik_zc[iz]*dvz + psi_p_z[ix*n1+iz];
                    p[ix*n1+iz] -= l2m[ix*n1+iz]*(dvx + dvz);
                }
            }
        }

    } /* end acoustic CFS-CPML */

    if (mod.ischeme == 3 && pml) { /* Elastic CFS-CPML: direct full stress update for PML cells.
         * elastic4 skips PML cells, so txx/tzz/txz there still hold the previous
         * timestep value and the CPML update is applied directly here.
         *
         * Ported verbatim from fdelmodc/boundaries.c:1595-1769 (modPar and bndPar
         * are byte-identical between fdelmodc.h and fdelfwi.h, so the physics code
         * transfers unchanged). The ONLY edit is the index convention: fdelmodc
         * shifts ioPx/ioPz/ioTx/ioTz by npml in getParameters.c and fdelfwi does
         * not, so the pad-excluded equivalents are recomputed here as *_e. The
         * velocity ranges ioXx/ieXx/ioXz/ieXz are already identical in both codes
         * and are used as-is. Keep this block diffable against fdelmodc. */
        int ioPx_e = mod.ioPx + (bnd.lef==2 ? npml : 0);
        int iePx_e = mod.iePx - (bnd.rig==2 ? npml : 0);
        int ioPz_e = mod.ioPz + (bnd.top==2 ? npml : 0);
        int iePz_e = mod.iePz - (bnd.bot==2 ? npml : 0);
        int ioTx_e = mod.ioTx + (bnd.lef==2 ? npml : 0);
        int ieTx_e = mod.ieTx - (bnd.rig==2 ? npml : 0);
        int ioTz_e = mod.ioTz + (bnd.top==2 ? npml : 0);
        int ieTz_e = mod.ieTz - (bnd.bot==2 ? npml : 0);
        int ize_pml, zmin, zmax;

        if (itime == 0) {
#pragma omp master
{
            memset(psi_p_x,   0, (size_t)n2*n1*sizeof(float));
            memset(psi_p_z,   0, (size_t)n2*n1*sizeof(float));
            memset(psi_txz_x, 0, (size_t)n2*n1*sizeof(float));
            memset(psi_txz_z, 0, (size_t)n2*n1*sizeof(float));
}
#pragma omp barrier
        }

        /* Direct CPML update for txx/tzz on P-grid PML cells. The loop must cover
         * the full PML extent including the right and bottom zones, which lie
         * beyond iePx_e/iePz_e (those are the interior limits). */
        ize_pml = bnd.bot==2 ? (mod.ieXz+npml<n1 ? mod.ieXz+npml : n1) : iePz_e;
        zmin    = has_top_pml ? ioPz_e-npml : mod.ioXz;
        zmax    = has_bot_pml ? ize_pml     : mod.ieXz;

        if (has_lef_pml) {
#pragma omp for private(ix, iz, dvx, dvz) schedule(static)
            for (ix=ioPx_e-npml; ix<mod.ioXx; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
                          c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
                    dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
                          c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
                    psi_p_x[ix*n1+iz] = b_xc[ix]*psi_p_x[ix*n1+iz] + c_xc[ix]*dvx;
                    dvx = ik_xc[ix]*dvx + psi_p_x[ix*n1+iz];
                    psi_p_z[ix*n1+iz] = b_zc[iz]*psi_p_z[ix*n1+iz] + c_zc[iz]*dvz;
                    dvz = ik_zc[iz]*dvz + psi_p_z[ix*n1+iz];
                    txx[ix*n1+iz] -= l2m[ix*n1+iz]*dvx + lam[ix*n1+iz]*dvz;
                    tzz[ix*n1+iz] -= l2m[ix*n1+iz]*dvz + lam[ix*n1+iz]*dvx;
                }
            }
        }
        if (has_rig_pml) {
#pragma omp for private(ix, iz, dvx, dvz) schedule(static)
            for (ix=mod.ieXx; ix<mod.ieXx+npml; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
                          c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
                    dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
                          c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
                    psi_p_x[ix*n1+iz] = b_xc[ix]*psi_p_x[ix*n1+iz] + c_xc[ix]*dvx;
                    dvx = ik_xc[ix]*dvx + psi_p_x[ix*n1+iz];
                    psi_p_z[ix*n1+iz] = b_zc[iz]*psi_p_z[ix*n1+iz] + c_zc[iz]*dvz;
                    dvz = ik_zc[iz]*dvz + psi_p_z[ix*n1+iz];
                    txx[ix*n1+iz] -= l2m[ix*n1+iz]*dvx + lam[ix*n1+iz]*dvz;
                    tzz[ix*n1+iz] -= l2m[ix*n1+iz]*dvz + lam[ix*n1+iz]*dvx;
                }
            }
        }
        if (has_top_pml) {
#pragma omp for private(ix, iz, dvx, dvz) schedule(static)
            for (ix=mod.ioXx; ix<mod.ieXx; ix++) {
                for (iz=ioPz_e-npml; iz<mod.ioXz; iz++) {
                    dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
                          c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
                    dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
                          c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
                    psi_p_x[ix*n1+iz] = b_xc[ix]*psi_p_x[ix*n1+iz] + c_xc[ix]*dvx;
                    dvx = ik_xc[ix]*dvx + psi_p_x[ix*n1+iz];
                    psi_p_z[ix*n1+iz] = b_zc[iz]*psi_p_z[ix*n1+iz] + c_zc[iz]*dvz;
                    dvz = ik_zc[iz]*dvz + psi_p_z[ix*n1+iz];
                    txx[ix*n1+iz] -= l2m[ix*n1+iz]*dvx + lam[ix*n1+iz]*dvz;
                    tzz[ix*n1+iz] -= l2m[ix*n1+iz]*dvz + lam[ix*n1+iz]*dvx;
                }
            }
        }
        if (has_bot_pml) {
#pragma omp for private(ix, iz, dvx, dvz) schedule(static)
            for (ix=mod.ioXx; ix<mod.ieXx; ix++) {
                for (iz=iePz_e; iz<mod.ieXz+npml; iz++) {
                    dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
                          c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
                    dvz = c1*(vz[ix*n1+iz+1]   - vz[ix*n1+iz]) +
                          c2*(vz[ix*n1+iz+2]   - vz[ix*n1+iz-1]);
                    psi_p_x[ix*n1+iz] = b_xc[ix]*psi_p_x[ix*n1+iz] + c_xc[ix]*dvx;
                    dvx = ik_xc[ix]*dvx + psi_p_x[ix*n1+iz];
                    psi_p_z[ix*n1+iz] = b_zc[iz]*psi_p_z[ix*n1+iz] + c_zc[iz]*dvz;
                    dvz = ik_zc[iz]*dvz + psi_p_z[ix*n1+iz];
                    txx[ix*n1+iz] -= l2m[ix*n1+iz]*dvx + lam[ix*n1+iz]*dvz;
                    tzz[ix*n1+iz] -= l2m[ix*n1+iz]*dvz + lam[ix*n1+iz]*dvx;
                }
            }
        }

        /* Direct CPML update for txz on T-grid PML cells */
        ize_pml = bnd.bot==2 ? (mod.ieXz+npml<n1 ? mod.ieXz+npml : n1) : ieTz_e;
        zmin    = has_top_pml ? ioTz_e-npml : mod.ioXz;
        zmax    = has_bot_pml ? ize_pml     : mod.ieXz;

        if (has_lef_pml) {
#pragma omp for private(ix, iz, dvx, dvz) schedule(static)
            for (ix=ioTx_e-npml; ix<ioTx_e; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    /* txz stencil: dvz/dx uses vz[ix..ix-1], dvx/dz uses vx[iz..iz-1] */
                    dvz = c1*(vz[ix*n1+iz]     - vz[(ix-1)*n1+iz]) +
                          c2*(vz[(ix+1)*n1+iz] - vz[(ix-2)*n1+iz]);
                    dvx = c1*(vx[ix*n1+iz]   - vx[ix*n1+iz-1]) +
                          c2*(vx[ix*n1+iz+1] - vx[ix*n1+iz-2]);
                    psi_txz_x[ix*n1+iz] = b_xf[ix]*psi_txz_x[ix*n1+iz] + c_xf[ix]*dvz;
                    dvz = ik_xf[ix]*dvz + psi_txz_x[ix*n1+iz];
                    psi_txz_z[ix*n1+iz] = b_zf[iz]*psi_txz_z[ix*n1+iz] + c_zf[iz]*dvx;
                    dvx = ik_zf[iz]*dvx + psi_txz_z[ix*n1+iz];
                    txz[ix*n1+iz] -= mul[ix*n1+iz]*(dvx + dvz);
                }
            }
        }
        if (has_rig_pml) {
#pragma omp for private(ix, iz, dvx, dvz) schedule(static)
            for (ix=ieTx_e; ix<mod.ieXx+npml; ix++) {
                for (iz=zmin; iz<zmax; iz++) {
                    dvz = c1*(vz[ix*n1+iz]     - vz[(ix-1)*n1+iz]) +
                          c2*(vz[(ix+1)*n1+iz] - vz[(ix-2)*n1+iz]);
                    dvx = c1*(vx[ix*n1+iz]   - vx[ix*n1+iz-1]) +
                          c2*(vx[ix*n1+iz+1] - vx[ix*n1+iz-2]);
                    psi_txz_x[ix*n1+iz] = b_xf[ix]*psi_txz_x[ix*n1+iz] + c_xf[ix]*dvz;
                    dvz = ik_xf[ix]*dvz + psi_txz_x[ix*n1+iz];
                    psi_txz_z[ix*n1+iz] = b_zf[iz]*psi_txz_z[ix*n1+iz] + c_zf[iz]*dvx;
                    dvx = ik_zf[iz]*dvx + psi_txz_z[ix*n1+iz];
                    txz[ix*n1+iz] -= mul[ix*n1+iz]*(dvx + dvz);
                }
            }
        }
        if (has_top_pml) {
#pragma omp for private(ix, iz, dvx, dvz) schedule(static)
            for (ix=ioTx_e; ix<ieTx_e; ix++) {
                for (iz=ioTz_e-npml; iz<mod.ioXz; iz++) {
                    dvz = c1*(vz[ix*n1+iz]     - vz[(ix-1)*n1+iz]) +
                          c2*(vz[(ix+1)*n1+iz] - vz[(ix-2)*n1+iz]);
                    dvx = c1*(vx[ix*n1+iz]   - vx[ix*n1+iz-1]) +
                          c2*(vx[ix*n1+iz+1] - vx[ix*n1+iz-2]);
                    psi_txz_x[ix*n1+iz] = b_xf[ix]*psi_txz_x[ix*n1+iz] + c_xf[ix]*dvz;
                    dvz = ik_xf[ix]*dvz + psi_txz_x[ix*n1+iz];
                    psi_txz_z[ix*n1+iz] = b_zf[iz]*psi_txz_z[ix*n1+iz] + c_zf[iz]*dvx;
                    dvx = ik_zf[iz]*dvx + psi_txz_z[ix*n1+iz];
                    txz[ix*n1+iz] -= mul[ix*n1+iz]*(dvx + dvz);
                }
            }
        }
        if (has_bot_pml) {
#pragma omp for private(ix, iz, dvx, dvz) schedule(static)
            for (ix=ioTx_e; ix<ieTx_e; ix++) {
                for (iz=mod.ieXz; iz<mod.ieXz+npml; iz++) {
                    dvz = c1*(vz[ix*n1+iz]     - vz[(ix-1)*n1+iz]) +
                          c2*(vz[(ix+1)*n1+iz] - vz[(ix-2)*n1+iz]);
                    dvx = c1*(vx[ix*n1+iz]   - vx[ix*n1+iz-1]) +
                          c2*(vx[ix*n1+iz+1] - vx[ix*n1+iz-2]);
                    psi_txz_x[ix*n1+iz] = b_xf[ix]*psi_txz_x[ix*n1+iz] + c_xf[ix]*dvz;
                    dvz = ik_xf[ix]*dvz + psi_txz_x[ix*n1+iz];
                    psi_txz_z[ix*n1+iz] = b_zf[iz]*psi_txz_z[ix*n1+iz] + c_zf[iz]*dvx;
                    dvx = ik_zf[iz]*dvx + psi_txz_z[ix*n1+iz];
                    txz[ix*n1+iz] -= mul[ix*n1+iz]*(dvx + dvz);
                }
            }
        }

    } /* end elastic CFS-CPML */


/****************************************************************/
/* Free surface: calculate free surface conditions for stresses */
/****************************************************************/

	
	ixo = mod.ioPx;
	ixe = mod.iePx;
	izo = mod.ioPz;
	ize = mod.iePz;

	if (mod.ischeme <= 2) { /* Acoustic scheme */
		if (bnd.top==1) { /* free surface at top */
#pragma omp	for private (ix) nowait
			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
				iz = bnd.surface[ix];
				tzz[ix*n1+iz] = 0.0;
                //vz[ix*n1+iz] = -vz[ix*n1+iz+1];
                //vz[ix*n1+iz-1] = -vz[ix*n1+iz+2];

			}
		}
		if (bnd.rig==1) { /* free surface at right */
#pragma omp	for private (iz) nowait
			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
				tzz[(mod.iePx-1)*n1+iz] = 0.0;
			}
		}
		if (bnd.bot==1) { /* free surface at bottom */
#pragma omp	for private (ix) nowait
			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
				tzz[ix*n1+mod.iePz-1] = 0.0;
			}
		}
		if (bnd.lef==1) { /* free surface at left */
#pragma omp	for private (iz) nowait
			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
				tzz[(mod.ioPx-1)*n1+iz] = 0.0;
			}
		}
	}
	else { /* Elastic scheme */
/* The implementation for a topgraphy surface is not yet correct */
		
		/* Free surface: calculate free surface conditions for stresses 
		 *	 Conditions (for upper boundary):
		 *	 1. Tzz = 0
		 *	 2. Txz = 0
		 *	 3. Txx: remove term with dVz/dz, computed in e2/e4 routines
		 *			 and add extra term with dVx/dx,
		 *			 corresponding to free-surface condition for Txx.
		 *			 In this way, dVz/dz is not needed in computing Txx
		 *			 on the upper stress free boundary. Other boundaries
		 *			 are treated similar.
		 *			 For the 4th order schemes, the whole virtual boundary
		 *			 must be taken into account in the removal terms, 
		 *			 because the algorithm sets
		 *			 velocities on this boundary!
		 *
		 *	Compute the velocities on the virtual boundary to make interpolation
		 *	possible for receivers. 
		 */
		
		if (bnd.top==1) { /* free surface at top */
			izp = bnd.surface[ixo];
#pragma omp for private (ix, iz) 
			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
				iz = bnd.surface[ix];
				if ( izp==iz ) {
					/* clear normal pressure */
					tzz[ix*n1+iz] = 0.0;

					/* This update to Vz might become unstable (2nd order scheme) */
//					vz[ix*n1+iz] = vz[ix*n1+iz+1] - (vx[(ix+1)*n1+iz]-vx[ix*n1+iz])*
//					lam[ix*n1+iz]/l2m[ix*n1+iz];
				}
				izp=iz;
			}

			izp = bnd.surface[ixo];
#pragma omp for private (ix, iz) 
			for (ix=mod.ioTx; ix<mod.ieTx; ix++) {
				iz = bnd.surface[ix];
				if ( izp==iz ) {
					/* assure that txz=0 on boundary by filling virtual boundary */
					txz[ix*n1+iz] = -txz[ix*n1+iz+1];
					/* extra line of txz has to be copied */
					txz[ix*n1+iz-1] = -txz[ix*n1+iz+2];
				}
				izp=iz;
			}

			/* calculate txx on top stress-free boundary */
			izp = bnd.surface[ixo];
#pragma omp for private (ix, iz, dp, dvx) 
			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
				iz = bnd.surface[ix];
				if ( izp==iz ) {
					if (l2m[ix*n1+iz]!=0.0) {
						dp = l2m[ix*n1+iz]-lam[ix*n1+iz]*lam[ix*n1+iz]/l2m[ix*n1+iz];
						dvx = c1*(vx[(ix+1)*n1+iz] - vx[(ix)*n1+iz]) +
						  	c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
						txx[ix*n1+iz] = -dvx*dp;
					}
				}
				izp=iz;
			}
			
			/* if surface has also left or right edges */
			izp = bnd.surface[ixo];
#pragma omp for private (ix, iz, dp, dvz) 
			for (ix=mod.ioPx+1; ix<mod.iePx; ix++) {
				iz = bnd.surface[ix-1];
				if ( izp < iz ) { /* right boundary */
					/* clear normal pressure */
					txx[ix*n1+iz] = 0.0;
					if ( (iz-izp) >= 2 ) { /* VR point */
						/* assure that txz=0 on boundary */
						txz[(ix+1)*n1+iz] = -txz[ix*n1+iz];
						txz[(ix+2)*n1+iz] = -txz[(ix-1)*n1+iz] ;
						/* calculate tzz on right stress-free boundary */
						if (l2m[ix*n1+iz]!=0.0) {
							dvz = c1*(vz[ix*n1+iz+1] - vz[ix*n1+iz]) +
							c2*(vz[ix*n1+iz+2] - vz[ix*n1+iz-1]);
							dp = l2m[ix*n1+iz]-lam[ix*n1+iz]*lam[ix*n1+iz]/l2m[ix*n1+iz];
							tzz[ix*n1+iz] = -dvz*dp;
						}
					}
					else {
							if (izp) { /* IR point */   
//											  txz[ix*n1+iz] = -txz[ix*n1+iz+1] ;
//											  txz[ix*n1+iz-1] = -txz[ix*n1+iz+2];
//						txz[(ix+1)*n1+iz] = -txz[ix*n1+iz];
//						txz[(ix+2)*n1+iz] = -txz[(ix-1)*n1+iz] ;
//						tzz[ix*n1+iz] = 0.0;
						}
							else { /* OR point */
//						txz[(ix-1)*n1+iz] = 0.0;
//						txz[(ix+1)*n1+iz] = -txz[ix*n1+iz];
//						txz[(ix+2)*n1+iz] = -txz[(ix-1)*n1+iz] ;
//						if (l2m[ix*n1+iz]!=0.0) {
//							vz[ix*n1+iz] = vz[ix*n1+iz+1] - (vx[(ix+1)*n1+iz]-vx[ix*n1+iz])*
//							   lam[ix*n1+iz]/l2m[ix*n1+iz];
//						}
							}
					}
				} /* end if right */
				if ( izp > iz ) { /* left boundary */
					/* clear normal pressure */
					txx[ix*n1+iz] = 0.0;
					/* assure that txz=0 on boundary */
					txz[(ix-1)*n1+iz] = -txz[ix*n1+iz];
					/* extra line of txz has to be copied */
					txz[(ix-2)*n1+iz] = -txz[(ix+1)*n1+iz] ;
					/* calculate tzz on left stress-free boundary */
					dvz = c1*(vz[ix*n1+iz+1] - vz[ix*n1+iz]) +
					c2*(vz[ix*n1+iz+2] - vz[ix*n1+iz-1]);
					if (l2m[ix*n1+iz]!=0.0) {
						dp = l2m[ix*n1+iz]-lam[ix*n1+iz]*lam[ix*n1+iz]/l2m[ix*n1+iz];
				   		tzz[ix*n1+iz] = -dvz*dp;
					}
				} /* end if left */
				izp=iz;
//		fprintf(stderr,"V4 ix=2123 iz=1 tzz=%e\n", tzz[2123*n1+1]);
				//		  izp=bnd.surface[MAX(ix-2,0)];;
			} /* end ix loop */
		}
		
		
		if (bnd.rig==1) { /* free surface at right */
			ix = mod.iePx;
#pragma omp for private (iz) 
			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
				/* clear normal pressure */
				txx[ix*n1+iz] = 0.0;
			}
#pragma omp for private (iz) 
			for (iz=mod.ioTz; iz<mod.ieTz; iz++) {
				/* assure that txz=0 on boundary by filling virtual boundary */
				txz[(ix+1)*n1+iz] = -txz[(ix)*n1+iz];
				/* extra line of txz has to be copied */
				txz[(ix+2)*n1+iz] = -txz[(ix-1)*n1+iz] ;
			}
			/* calculate tzz on right stress-free boundary */
#pragma omp for private (iz) 
			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
				dvz = c1*(vz[(ix)*n1+iz+1] - vz[(ix)*n1+iz]) +
					  c2*(vz[(ix)*n1+iz+2] - vz[(ix)*n1+iz-1]);
				if (l2m[ix*n1+iz]!=0.0) {
					dp = l2m[(ix)*n1+iz]-lam[(ix)*n1+iz]*lam[(ix)*n1+iz]/l2m[(ix)*n1+iz];
					tzz[(ix)*n1+iz] = -dvz*dp;
				}
			}
		}
		
		
		if (bnd.bot==1) { /* free surface at bottom */
			iz = mod.iePz;
#pragma omp for private (ix) 
			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
				/* clear normal pressure */
				tzz[ix*n1+iz] = 0.0;
			}
#pragma omp for private (ix) 
			for (ix=mod.ioTx; ix<mod.ieTx; ix++) {
				/* assure that txz=0 on boundary by filling virtual boundary */
				txz[ix*n1+iz+1] = -txz[ix*n1+iz];
				/* extra line of txz has to be copied */
				txz[ix*n1+iz+2] = -txz[ix*n1+iz-1];
			}
			/* calculate txx on bottom stress-free boundary */
#pragma omp for private (ix) 
			for (ix=mod.ioPx; ix<mod.iePx; ix++) {
				dvx = c1*(vx[(ix+1)*n1+iz] - vx[ix*n1+iz]) +
					  c2*(vx[(ix+2)*n1+iz] - vx[(ix-1)*n1+iz]);
				if (l2m[ix*n1+iz]!=0.0) {
					dp = l2m[ix*n1+iz]-lam[ix*n1+iz]*lam[ix*n1+iz]/l2m[ix*n1+iz];
					txx[ix*n1+iz] = -dvx*dp;
				}
			}
		}
		
		if (bnd.lef==1) { /* free surface at left */
			ix = mod.ioPx;
#pragma omp for private (iz) 
			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
				/* clear normal pressure */
				txx[ix*n1+iz] = 0.0;
			}
#pragma omp for private (iz) 
			for (iz=mod.ioTz; iz<mod.ieTz; iz++) {
				/* assure that txz=0 on boundary by filling virtual boundary */
				txz[(ix)*n1+iz] = -txz[(ix+1)*n1+iz];
				/* extra line of txz has to be copied */
				txz[(ix-1)*n1+iz] = -txz[(ix+2)*n1+iz] ;
			}
			/* calculate tzz on left stress-free boundary */
#pragma omp for private (iz) 
			for (iz=mod.ioPz; iz<mod.iePz; iz++) {
				dvz = c1*(vz[ix*n1+iz+1] - vz[ix*n1+iz]) +
					  c2*(vz[ix*n1+iz+2] - vz[ix*n1+iz-1]);
				if (l2m[ix*n1+iz]!=0.0) {
					dp = l2m[ix*n1+iz]-lam[ix*n1+iz]*lam[ix*n1+iz]/l2m[ix*n1+iz];
					tzz[ix*n1+iz] = -dvz*dp;
				}
			}
		}
	}
	
    if ( (npml != 0) && (itime==mod.nt-1) && pml) {
#pragma omp master
{
		if (allocated) {
            /* Drop the references but do NOT free -- see boundariesP. */
            psi_p_x = psi_p_z = psi_txz_x = psi_txz_z = NULL;
            if (b_xc)  { free(b_xc);  b_xc  = NULL; }
            if (c_xc)  { free(c_xc);  c_xc  = NULL; }
            if (ik_xc) { free(ik_xc); ik_xc = NULL; }
            if (b_zc)  { free(b_zc);  b_zc  = NULL; }
            if (c_zc)  { free(c_zc);  c_zc  = NULL; }
            if (ik_zc) { free(ik_zc); ik_zc = NULL; }
            if (b_xf)  { free(b_xf);  b_xf  = NULL; }
            if (c_xf)  { free(c_xf);  c_xf  = NULL; }
            if (ik_xf) { free(ik_xf); ik_xf = NULL; }
            if (b_zf)  { free(b_zf);  b_zf  = NULL; }
            if (c_zf)  { free(c_zf);  c_zf  = NULL; }
            if (ik_zf) { free(ik_zf); ik_zf = NULL; }
            allocated=0;
		}
}
	}

	return 0;
}
