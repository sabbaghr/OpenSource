#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include "segy.h"
#include "par.h"
#include "fdelfwi.h" // Includes the corrected modPar struct

#define     MAX(x,y) ((x) > (y) ? (x) : (y))
#define     MIN(x,y) ((x) < (y) ? (x) : (y))
#define     NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

/*
 * Reads gridded model files and calculates FD parameters (l2m, rox, lam, muu).
 * Adapts to Acoustic (1), Visco (2/4), or Elastic (3/4) schemes.
 * Allocates memory directly into modPar structure.
 */
int readModel(modPar *mod, bndPar *bnd)
{
    FILE    *fpcp, *fpcs=NULL, *fpro;
    FILE    *fpqp=NULL, *fpqs=NULL;
    size_t  nread, sizem;
    int     i, tracesToDo;
    int     n1, ix, iz, nz, nx;
    int     ioXx, ioXz, ioZz, ioZx, ioPx, ioPz, ioTx, ioTz;
    int     ixo, ixe, izo, ize; // Loop limits for boundaries
    
    float   cp2, cs2, cs11, cs12, cs21, cs22, mul, mu, lamda2mu, lamda;
    float   cs2c, cs2b, cs2a, bx, bz, fac;
    float   *cp, *cs=NULL, *ro, *qp=NULL, *qs=NULL; // Temp buffers
    float   a, b;
    segy    hdr;

    /* 1. Setup Dimensions & Offsets */
    nz = mod->nz;
    nx = mod->nx;
    n1 = mod->naz;
    sizem = mod->nax * mod->naz;
    fac = mod->dt / mod->dx;

    /* Offsets */
    ioXx=mod->ioXx; ioXz=mod->ioXz;
    ioZz=mod->ioZz; ioZx=mod->ioZx;
    ioPx=mod->ioPx; ioPz=mod->ioPz;
    ioTx=mod->ioTx; ioTz=mod->ioTz;

    /* Adjust for Taper/Boundary */
    if (bnd->lef==4 || bnd->lef==2) { ioPx += bnd->ntap; ioTx += bnd->ntap; }
    if (bnd->top==4 || bnd->top==2) { ioPz += bnd->ntap; ioTz += bnd->ntap; }

    /* 2. Allocate Struct Arrays */
    /* Always needed: Buoyancy & P-modulus */
    mod->rox = (float *)calloc(sizem, sizeof(float));
    mod->roz = (float *)calloc(sizem, sizeof(float));
    mod->l2m = (float *)calloc(sizem, sizeof(float));

    /* Elastic: Lambda & Mu */
    if (mod->ischeme > 2) {
        mod->lam = (float *)calloc(sizem, sizeof(float));
        mod->muu = (float *)calloc(sizem, sizeof(float));
    }

    /* Store raw velocity/density for FWI chain rule (velocity parameterization) */
    mod->cp  = (float *)calloc(sizem, sizeof(float));
    mod->rho = (float *)calloc(sizem, sizeof(float));
    if (mod->ischeme > 2) {
        mod->cs = (float *)calloc(sizem, sizeof(float));
    } else {
        mod->cs = NULL;
    }
    
    /* Visco-Acoustic/Elastic: Memory Variables P */
    if (mod->ischeme == 2 || mod->ischeme == 4) {
        mod->tss = (float *)calloc(sizem, sizeof(float));
        mod->tep = (float *)calloc(sizem, sizeof(float));
    }
    
    /* Visco-Elastic: Memory Variables S */
    if (mod->ischeme == 4) {
        mod->tes = (float *)calloc(sizem, sizeof(float));
    }

    /* 3. Allocate Temp Buffers & Open Files */
    cp = (float *)malloc(nz*nx*sizeof(float));
    ro = (float *)malloc(nz*nx*sizeof(float));
    
    fpcp = fopen(mod->file_cp, "r");
    fpro = fopen(mod->file_ro, "r");
    if (!fpcp || !fpro) { 
        fprintf(stderr, "Error: Missing Cp or Rho file.\n"); 
        return -1; 
    }
    fread(&hdr, 1, TRCBYTES, fpcp);
    fread(&hdr, 1, TRCBYTES, fpro);

    if (mod->ischeme > 2 && mod->ischeme != 5) {
        cs = (float *)calloc(nz*nx, sizeof(float));
        fpcs = fopen(mod->file_cs, "r");
        if (!fpcs) { fprintf(stderr, "Error: Missing Cs file for elastic scheme.\n"); return -1; }
        fread(&hdr, 1, TRCBYTES, fpcs);
    }

    /* Q-files (Optional) */
    if (mod->file_qp && (mod->ischeme==2 || mod->ischeme==4)) {
        qp = (float *)malloc(nz*sizeof(float)); // Reading 1D Q-logs as example, adapt if 2D
        fpqp = fopen(mod->file_qp, "r");
        fread(&hdr, 1, TRCBYTES, fpqp);
    }
    if (mod->file_qs && mod->ischeme==4) {
        qs = (float *)malloc(nz*sizeof(float));
        fpqs = fopen(mod->file_qs, "r");
        fread(&hdr, 1, TRCBYTES, fpqs);
    }

/* 4. Read Data Loop */
    tracesToDo = mod->nx;
    i = 0;
    while (tracesToDo) {
        /* Read Standard Files */
        nread = fread(&cp[i*nz], sizeof(float), hdr.ns, fpcp);
        assert(nread == hdr.ns);
        
        nread = fread(&ro[i*nz], sizeof(float), hdr.ns, fpro);
        assert(nread == hdr.ns);

        if (mod->ischeme > 2 && mod->ischeme != 5) {
            nread = fread(&cs[i*nz], sizeof(float), hdr.ns, fpcs);
            assert(nread == hdr.ns);
        }

        /* ========================================================= */
        /* VISCO-ACOUSTIC LOGIC (Scheme 2 or 4)                 */
        /* Converts Qp -> Tau-Sigma & Tau-EpsilonP              */
        /* ========================================================= */
        if (mod->ischeme == 2 || mod->ischeme == 4) {
            if (mod->file_qp != NULL) {
                /* Case A: Qp is varying (read from file) */
                nread = fread(&qp[0], sizeof(float), nz, fpqp);
                assert(nread == hdr.ns);
                
                for (iz = 0; iz < nz; iz++) {
                    /* The math: standard linear solid model conversion */
                    a = (sqrt(1.0 + (1.0 / (qp[iz] * qp[iz]))) - (1.0 / qp[iz])) / mod->fw;
                    b = 1.0 / (mod->fw * mod->fw * a);
                    
                    /* Write to struct arrays */
                    mod->tss[(i + ioPx) * n1 + iz + ioPz] = 1.0 / a;
                    mod->tep[(i + ioPx) * n1 + iz + ioPz] = b;
                }
            } 
            else {
                /* Case B: Qp is constant (from parameters) */
                /* Pre-calculation optimization could be done here, but we keep original logic */
                for (iz = 0; iz < nz; iz++) {
                    a = (sqrt(1.0 + (1.0 / (mod->Qp * mod->Qp))) - (1.0 / mod->Qp)) / mod->fw;
                    b = 1.0 / (mod->fw * mod->fw * a);
                    
                    mod->tss[(i + ioPx) * n1 + iz + ioPz] = 1.0 / a;
                    mod->tep[(i + ioPx) * n1 + iz + ioPz] = b;
                }
            }
        }

        /* ========================================================= */
        /* VISCO-ELASTIC LOGIC (Scheme 4 Only)                  */
        /* Converts Qs -> Tau-EpsilonS                          */
        /* ========================================================= */
        if (mod->ischeme == 4) {
            if (mod->file_qs != NULL) {
                /* Case A: Qs is varying */
                nread = fread(&qs[0], sizeof(float), hdr.ns, fpqs);
                assert(nread == hdr.ns);
                
                for (iz = 0; iz < nz; iz++) {
                    // We reuse 'tss' calculated above
                    a = 1.0 / mod->tss[(i + ioPx) * n1 + iz + ioPz];
                    float q_val = qs[iz];
                    
                    mod->tes[(i + ioPx) * n1 + iz + ioPz] = 
                        (1.0 + (mod->fw * q_val * a)) / (mod->fw * q_val - (mod->fw * mod->fw * a));
                }
            } 
            else {
                /* Case B: Qs is constant */
                for (iz = 0; iz < nz; iz++) {
                    a = 1.0 / mod->tss[(i + ioPx) * n1 + iz + ioPz];
                    float q_val = mod->Qs;
                    
                    mod->tes[(i + ioPx) * n1 + iz + ioPz] = 
                        (1.0 + (mod->fw * q_val * a)) / (mod->fw * q_val - (mod->fw * mod->fw * a));
                }
            }
        }

        /* Advance File Pointers / Read Next Headers */
        nread = fread(&hdr, 1, TRCBYTES, fpcp);
        if (nread == 0) break;
        
        nread = fread(&hdr, 1, TRCBYTES, fpro);
        if (nread == 0) break;
        
        if (mod->ischeme > 2 && mod->ischeme != 5) {
            nread = fread(&hdr, 1, TRCBYTES, fpcs);
            if (nread == 0) break;
        }
        
        if (mod->file_qp != NULL && (mod->ischeme == 2 || mod->ischeme == 4)) {
            nread = fread(&hdr, 1, TRCBYTES, fpqp);
            if (nread == 0) break;
        }
        
        if (mod->file_qs != NULL && mod->ischeme == 4) {
            nread = fread(&hdr, 1, TRCBYTES, fpqs);
            if (nread == 0) break;
        }
        
        i++;
    }
    
    fclose(fpcp); fclose(fpro);
    if (fpcs) fclose(fpcs);
    if (fpqp) fclose(fpqp); if (fpqs) fclose(fpqs);

    /* 5. Check for zero densities */
    for (i=0; i<nz*nx; i++) {
        if (ro[i] == 0.0) {
            fprintf(stderr, "ERROR: Zero density at trace=%d sample=%d\n", i/nz, i%nz);
            free(cp); free(ro); if(cs) free(cs);
            return -1;
        }
    }

    /* 6. Interpolate & Calculate Physics */

    /* --- ELASTIC SCHEME --- */
    if (mod->ischeme > 2) {

        /* Bottom edge (iz = nz-1): no iz+1 neighbor */
        iz = nz - 1;
        for (ix = 0; ix < nx - 1; ix++) {
            cp2  = cp[ix*nz+iz]*cp[ix*nz+iz];
            cs2  = cs[ix*nz+iz]*cs[ix*nz+iz];
            cs2a = cs[(ix+1)*nz+iz]*cs[(ix+1)*nz+iz];
            cs11 = cs2*ro[ix*nz+iz];
            cs12 = cs2*ro[ix*nz+iz];
            cs21 = cs2a*ro[(ix+1)*nz+iz];
            cs22 = cs2a*ro[(ix+1)*nz+iz];

            if (cs11 > 0.0) mul = 4.0/(1.0/cs11+1.0/cs12+1.0/cs21+1.0/cs22);
            else mul = 0.0;

            mu = cs2*ro[ix*nz+iz];
            lamda2mu = cp2*ro[ix*nz+iz];
            lamda = lamda2mu - 2*mu;

            bx = 0.5*(ro[ix*nz+iz]+ro[(ix+1)*nz+iz]);
            bz = ro[ix*nz+iz];

            mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/bx;
            mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/bz;
            mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
            mod->lam[(ix+ioPx)*n1+iz+ioPz] = fac*lamda;
            mod->muu[(ix+ioTx)*n1+iz+ioTz] = fac*mul;
        }

        /* Right edge (ix = nx-1): no ix+1 neighbor */
        ix = nx - 1;
        for (iz = 0; iz < nz - 1; iz++) {
            cp2  = cp[ix*nz+iz]*cp[ix*nz+iz];
            cs2  = cs[ix*nz+iz]*cs[ix*nz+iz];
            cs2b = cs[ix*nz+iz+1]*cs[ix*nz+iz+1];
            cs11 = cs2*ro[ix*nz+iz];
            cs12 = cs2b*ro[ix*nz+iz+1];
            cs21 = cs2*ro[ix*nz+iz];
            cs22 = cs2b*ro[ix*nz+iz+1];

            if (cs11 > 0.0) mul = 4.0/(1.0/cs11+1.0/cs12+1.0/cs21+1.0/cs22);
            else mul = 0.0;

            mu = cs2*ro[ix*nz+iz];
            lamda2mu = cp2*ro[ix*nz+iz];
            lamda = lamda2mu - 2*mu;

            bx = ro[ix*nz+iz];
            bz = 0.5*(ro[ix*nz+iz]+ro[ix*nz+iz+1]);

            mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/bx;
            mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/bz;
            mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
            mod->lam[(ix+ioPx)*n1+iz+ioPz] = fac*lamda;
            mod->muu[(ix+ioTx)*n1+iz+ioTz] = fac*mul;
        }

        /* Bottom-right corner */
        ix = nx - 1;
        iz = nz - 1;
        cp2 = cp[ix*nz+iz]*cp[ix*nz+iz];
        cs2 = cs[ix*nz+iz]*cs[ix*nz+iz];
        mu  = cs2*ro[ix*nz+iz];
        lamda2mu = cp2*ro[ix*nz+iz];
        lamda = lamda2mu - 2*mu;
        bx = ro[ix*nz+iz];
        bz = ro[ix*nz+iz];

        mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/bx;
        mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/bz;
        mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
        mod->lam[(ix+ioPx)*n1+iz+ioPz] = fac*lamda;
        mod->muu[(ix+ioTx)*n1+iz+ioTz] = fac*mu;

        /* Interior loop */
        for (ix = 0; ix < nx - 1; ix++) {
            for (iz = 0; iz < nz - 1; iz++) {
                cp2  = cp[ix*nz+iz]*cp[ix*nz+iz];
                cs2  = cs[ix*nz+iz]*cs[ix*nz+iz];
                cs2a = cs[(ix+1)*nz+iz]*cs[(ix+1)*nz+iz];
                cs2b = cs[ix*nz+iz+1]*cs[ix*nz+iz+1];
                cs2c = cs[(ix+1)*nz+iz+1]*cs[(ix+1)*nz+iz+1];

                cs11 = cs2*ro[ix*nz+iz];
                cs12 = cs2b*ro[ix*nz+iz+1];
                cs21 = cs2a*ro[(ix+1)*nz+iz];
                cs22 = cs2c*ro[(ix+1)*nz+iz+1];

                if (cs11 > 0.0) mul = 4.0/(1.0/cs11+1.0/cs12+1.0/cs21+1.0/cs22);
                else mul = 0.0;

                mu = cs2*ro[ix*nz+iz];
                lamda2mu = cp2*ro[ix*nz+iz];
                lamda = lamda2mu - 2*mu;

                bx = 0.5*(ro[ix*nz+iz]+ro[(ix+1)*nz+iz]);
                bz = 0.5*(ro[ix*nz+iz]+ro[ix*nz+iz+1]);

                mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/bx;
                mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/bz;
                mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
                mod->lam[(ix+ioPx)*n1+iz+ioPz] = fac*lamda;
                mod->muu[(ix+ioTx)*n1+iz+ioTz] = fac*mul;
            }
        }
    }
    /* --- ACOUSTIC SCHEME --- */
    else {
        /* Bottom edge (iz = nz-1) */
        iz = nz - 1;
        for (ix = 0; ix < nx - 1; ix++) {
            cp2 = cp[ix*nz+iz]*cp[ix*nz+iz];
            lamda2mu = cp2*ro[ix*nz+iz];
            bx = 0.5*(ro[ix*nz+iz]+ro[(ix+1)*nz+iz]);
            bz = ro[ix*nz+iz];
            mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/bx;
            mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/bz;
            mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
        }

        /* Right edge (ix = nx-1) */
        ix = nx - 1;
        for (iz = 0; iz < nz - 1; iz++) {
            cp2 = cp[ix*nz+iz]*cp[ix*nz+iz];
            lamda2mu = cp2*ro[ix*nz+iz];
            bx = ro[ix*nz+iz];
            bz = 0.5*(ro[ix*nz+iz]+ro[ix*nz+iz+1]);
            mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/bx;
            mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/bz;
            mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
        }

        /* Bottom-right corner */
        ix = nx - 1;
        iz = nz - 1;
        cp2 = cp[ix*nz+iz]*cp[ix*nz+iz];
        lamda2mu = cp2*ro[ix*nz+iz];
        bx = ro[ix*nz+iz];
        bz = ro[ix*nz+iz];
        mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/bx;
        mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/bz;
        mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;

        /* Interior loop */
        for (ix = 0; ix < nx - 1; ix++) {
            for (iz = 0; iz < nz - 1; iz++) {
                cp2 = cp[ix*nz+iz]*cp[ix*nz+iz];
                lamda2mu = cp2*ro[ix*nz+iz];
                bx = 0.5*(ro[ix*nz+iz]+ro[(ix+1)*nz+iz]);
                bz = 0.5*(ro[ix*nz+iz]+ro[ix*nz+iz+1]);
                mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/bx;
                mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/bz;
                mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
            }
        }
    }

    /* 6b. Store raw velocity/density for FWI chain rule */
    for (ix = 0; ix < nx; ix++) {
        for (iz = 0; iz < nz; iz++) {
            mod->cp[(ix+ioPx)*n1+iz+ioPz]  = cp[ix*nz+iz];
            mod->rho[(ix+ioPx)*n1+iz+ioPz] = ro[ix*nz+iz];
            if (mod->cs)
                mod->cs[(ix+ioPx)*n1+iz+ioPz] = cs[ix*nz+iz];
        }
    }

    /* 7. Zero-velocity topography check: set rox/roz to zero where l2m is zero */
    for (ix = 0; ix < nx; ix++) {
        for (iz = 0; iz < nz; iz++) {
            if (mod->l2m[(ix+ioPx)*n1+iz+ioPz] == 0.0) {
                mod->rox[(ix+ioXx)*n1+iz+ioXz] = 0.0;
                mod->roz[(ix+ioZx)*n1+iz+ioZz] = 0.0;
            }
        }
    }

    /*****************************************************/
    /* 8. Boundary extension for tapered/PML boundaries  */
    /*****************************************************/

    /* Left */
    if (bnd->lef==4 || bnd->lef==2) {

        /* rox field */
        ixo = mod->ioXx-bnd->ntap;
        ixe = mod->ioXx;
        izo = mod->ioXz;
        ize = mod->ieXz;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->rox[ix*n1+iz] = mod->rox[ixe*n1+iz];
            }
        }

        /* roz field */
        ixo = mod->ioZx-bnd->ntap;
        ixe = mod->ioZx;
        izo = mod->ioZz;
        ize = mod->ieZz;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->roz[ix*n1+iz] = mod->roz[ixe*n1+iz];
            }
        }

        /* l2m field */
        ixo = mod->ioPx;
        ixe = mod->ioPx+bnd->ntap;
        izo = mod->ioPz;
        ize = mod->iePz;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->l2m[ix*n1+iz] = mod->l2m[ixe*n1+iz];
            }
        }

        if (mod->ischeme > 2) {
            /* lam field */
            ixo = mod->ioPx;
            ixe = mod->ioPx+bnd->ntap;
            izo = mod->ioPz;
            ize = mod->iePz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->lam[ix*n1+iz] = mod->lam[ixe*n1+iz];
                }
            }
            /* muu field */
            ixo = mod->ioTx;
            ixe = mod->ioTx+bnd->ntap;
            izo = mod->ioTz;
            ize = mod->ieTz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->muu[ix*n1+iz] = mod->muu[ixe*n1+iz];
                }
            }
        }
        if (mod->ischeme==2 || mod->ischeme==4) {
            ixo = mod->ioPx;
            ixe = mod->ioPx+bnd->ntap;
            izo = mod->ioPz;
            ize = mod->iePz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->tss[ix*n1+iz] = mod->tss[ixe*n1+iz];
                    mod->tep[ix*n1+iz] = mod->tep[ixe*n1+iz];
                }
            }
        }
        if (mod->ischeme==4) {
            ixo = mod->ioPx;
            ixe = mod->ioPx+bnd->ntap;
            izo = mod->ioPz;
            ize = mod->iePz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->tes[ix*n1+iz] = mod->tes[ixe*n1+iz];
                }
            }
        }
    }

    /* Right */
    if (bnd->rig==4 || bnd->rig==2) {

        /* rox field */
        ixo = mod->ieXx;
        ixe = mod->ieXx+bnd->ntap;
        izo = mod->ioXz;
        ize = mod->ieXz;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->rox[ix*n1+iz] = mod->rox[(ixo-1)*n1+iz];
            }
        }

        /* roz field */
        ixo = mod->ieZx;
        ixe = mod->ieZx+bnd->ntap;
        izo = mod->ioZz;
        ize = mod->ieZz;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->roz[ix*n1+iz] = mod->roz[(ixo-1)*n1+iz];
            }
        }

        /* l2m field */
        ixo = mod->iePx-bnd->ntap;
        ixe = mod->iePx;
        izo = mod->ioPz;
        ize = mod->iePz;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->l2m[ix*n1+iz] = mod->l2m[(ixo-1)*n1+iz];
            }
        }

        if (mod->ischeme > 2) {
            /* lam field */
            ixo = mod->iePx-bnd->ntap;
            ixe = mod->iePx;
            izo = mod->ioPz;
            ize = mod->iePz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->lam[ix*n1+iz] = mod->lam[(ixo-1)*n1+iz];
                }
            }
            /* muu field */
            ixo = mod->ieTx-bnd->ntap;
            ixe = mod->ieTx;
            izo = mod->ioTz;
            ize = mod->ieTz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->muu[ix*n1+iz] = mod->muu[(ixo-1)*n1+iz];
                }
            }
        }
        if (mod->ischeme==2 || mod->ischeme==4) {
            ixo = mod->iePx-bnd->ntap;
            ixe = mod->iePx;
            izo = mod->ioPz;
            ize = mod->iePz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->tss[ix*n1+iz] = mod->tss[(ixo-1)*n1+iz];
                    mod->tep[ix*n1+iz] = mod->tep[(ixo-1)*n1+iz];
                }
            }
        }
        if (mod->ischeme==4) {
            ixo = mod->iePx-bnd->ntap;
            ixe = mod->iePx;
            izo = mod->ioPz;
            ize = mod->iePz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->tes[ix*n1+iz] = mod->tes[(ixo-1)*n1+iz];
                }
            }
        }
    }

    /* Top */
    if (bnd->top==4 || bnd->top==2) {

        /* rox field */
        ixo = mod->ioXx;
        ixe = mod->ieXx;
        if (bnd->lef==4 || bnd->lef==2) ixo -= bnd->ntap;
        if (bnd->rig==4 || bnd->rig==2) ixe += bnd->ntap;
        izo = mod->ioXz-bnd->ntap;
        ize = mod->ioXz;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->rox[ix*n1+iz] = mod->rox[ix*n1+ize];
            }
        }

        /* roz field */
        ixo = mod->ioZx;
        ixe = mod->ieZx;
        if (bnd->lef==4 || bnd->lef==2) ixo -= bnd->ntap;
        if (bnd->rig==4 || bnd->rig==2) ixe += bnd->ntap;
        izo = mod->ioZz-bnd->ntap;
        ize = mod->ioZz;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->roz[ix*n1+iz] = mod->roz[ix*n1+ize];
            }
        }

        /* l2m field */
        ixo = mod->ioPx;
        ixe = mod->iePx;
        izo = mod->ioPz;
        ize = mod->ioPz+bnd->ntap;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->l2m[ix*n1+iz] = mod->l2m[ix*n1+ize];
            }
        }

        if (mod->ischeme > 2) {
            /* lam field */
            ixo = mod->ioPx;
            ixe = mod->iePx;
            izo = mod->ioPz;
            ize = mod->ioPz+bnd->ntap;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->lam[ix*n1+iz] = mod->lam[ix*n1+ize];
                }
            }
            /* muu field */
            ixo = mod->ioTx;
            ixe = mod->ieTx;
            izo = mod->ioTz;
            ize = mod->ioTz+bnd->ntap;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->muu[ix*n1+iz] = mod->muu[ix*n1+ize];
                }
            }
        }
        if (mod->ischeme==2 || mod->ischeme==4) {
            ixo = mod->ioPx;
            ixe = mod->iePx;
            izo = mod->ioPz;
            ize = mod->ioPz+bnd->ntap;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->tss[ix*n1+iz] = mod->tss[ix*n1+ize];
                    mod->tep[ix*n1+iz] = mod->tep[ix*n1+ize];
                }
            }
        }
        if (mod->ischeme==4) {
            ixo = mod->ioPx;
            ixe = mod->iePx;
            izo = mod->ioPz;
            ize = mod->ioPz+bnd->ntap;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->tes[ix*n1+iz] = mod->tes[ix*n1+ize];
                }
            }
        }
    }

    /* Bottom */
    if (bnd->bot==4 || bnd->bot==2) {

        /* rox field */
        ixo = mod->ioXx;
        ixe = mod->ieXx;
        if (bnd->lef==4 || bnd->lef==2) ixo -= bnd->ntap;
        if (bnd->rig==4 || bnd->rig==2) ixe += bnd->ntap;
        izo = mod->ieXz;
        ize = mod->ieXz+bnd->ntap;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->rox[ix*n1+iz] = mod->rox[ix*n1+izo-1];
            }
        }

        /* roz field */
        ixo = mod->ioZx;
        ixe = mod->ieZx;
        if (bnd->lef==4 || bnd->lef==2) ixo -= bnd->ntap;
        if (bnd->rig==4 || bnd->rig==2) ixe += bnd->ntap;
        izo = mod->ieZz;
        ize = mod->ieZz+bnd->ntap;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->roz[ix*n1+iz] = mod->roz[ix*n1+izo-1];
            }
        }

        /* l2m field */
        ixo = mod->ioPx;
        ixe = mod->iePx;
        izo = mod->iePz-bnd->ntap;
        ize = mod->iePz;
        for (ix=ixo; ix<ixe; ix++) {
            for (iz=izo; iz<ize; iz++) {
                mod->l2m[ix*n1+iz] = mod->l2m[ix*n1+izo-1];
            }
        }

        if (mod->ischeme > 2) {
            /* lam field */
            ixo = mod->ioPx;
            ixe = mod->iePx;
            izo = mod->iePz-bnd->ntap;
            ize = mod->iePz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->lam[ix*n1+iz] = mod->lam[ix*n1+izo-1];
                }
            }
            /* muu field */
            ixo = mod->ioTx;
            ixe = mod->ieTx;
            izo = mod->ieTz-bnd->ntap;
            ize = mod->ieTz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->muu[ix*n1+iz] = mod->muu[ix*n1+izo-1];
                }
            }
        }
        if (mod->ischeme==2 || mod->ischeme==4) {
            ixo = mod->ioPx;
            ixe = mod->iePx;
            izo = mod->iePz-bnd->ntap;
            ize = mod->iePz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->tss[ix*n1+iz] = mod->tss[ix*n1+izo-1];
                    mod->tep[ix*n1+iz] = mod->tep[ix*n1+izo-1];
                }
            }
        }
        if (mod->ischeme==4) {
            ixo = mod->ioPx;
            ixe = mod->iePx;
            izo = mod->iePz-bnd->ntap;
            ize = mod->iePz;
            for (ix=ixo; ix<ixe; ix++) {
                for (iz=izo; iz<ize; iz++) {
                    mod->tes[ix*n1+iz] = mod->tes[ix*n1+izo-1];
                }
            }
        }
    }

    /* 9. Cleanup */
    free(cp); free(ro);
    if(cs) free(cs);
    if(qp) free(qp); if(qs) free(qs);

    return 0;
}