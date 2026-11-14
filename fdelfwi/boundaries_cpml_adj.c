#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include"fdelfwi.h"
#include"par.h"

/*********************************************************************
 *
 * boundaries_cpml_adj.c -- EXACT discrete transpose of the CFS-CPML.
 *
 * WHAT IS BEING TRANSPOSED
 * ------------------------
 * One forward elastic time step (elastic4.c + boundaries.c) is, as a
 * linear operator on the state (v, t, psi):
 *
 *   A1  v += M1 t          elastic4.c main velocity loops   (no psi)
 *   A2  (v,psi_v) += ...   boundariesP CFS-CPML blocks
 *   G   v  = G v           boundariesP ghost-column fill
 *   A3  t += M2 v          elastic4.c main stress loops     (no psi)
 *   A4  (t,psi_t) += ...   boundariesV CFS-CPML blocks
 *   FS  free surface on t
 *
 * A1 and A2 write DISJOINT sets of v and both read only t, so they are
 * one linear map v += M t over the union of their index sets; likewise
 * A3/A4.  That is the key simplification exploited here: this file
 * implements (A1+A2)^T as cpmlAdjF1T, (A3+A4)^T as cpmlAdjF2T and G^T
 * as cpmlAdjGhostT -- so cpmlAdjF1T/F2T REPLACE the corresponding main
 * loops of elastic4_adj.c rather than supplementing them.
 *
 * Reversing the order gives the adjoint sweep
 *
 *   FS^T  ->  cpmlAdjF2T  ->  cpmlAdjGhostT  ->  cpmlAdjF1T
 *
 * which is how elastic4_adj.c is wired.
 *
 *
 * WHY A MASK AND NOT A UNIFORM LOOP
 * ---------------------------------
 * The obvious implementation -- extend b/c/ik over the whole grid with
 * the no-op defaults b=0, c=0, ik=1 and run ONE loop -- is WRONG, and
 * silently so.  boundariesP builds its z profile on the vz face grid
 * (xi = (ioZz-iz)/npml) while the vx main kernel starts at ioXz = ioZz-1.
 * So b_zf[ioXz] != 0 even though the forward applies NO psi correction
 * at iz = ioXz.  A uniform loop would apply a correction the forward
 * never applied, and the dot-product test would fail by a thin line of
 * cells that is easy to mistake for a stencil-edge effect.
 *
 * Instead each field carries a per-point byte mask recording what the
 * forward actually did there:
 *
 *   0  forward does not update this field at this point
 *   1  forward updates it with the PLAIN stencil   (main kernel)
 *   2  forward updates it with the CPML stencil    (boundaries block)
 *
 * The masks are built by replaying the forward's loop bounds verbatim,
 * so they cannot drift from it by arithmetic reasoning -- only by an
 * edit to boundaries.c that is not mirrored here.  cpmlAdjBuild counts
 * overlaps between the main set and the PML blocks and complains: the
 * forward relies on them being disjoint, and so does this transpose.
 *
 * Only ONE mask per field is needed, not one per direction: inside a
 * PML block BOTH directions take their coefficients from the same 1D
 * arrays, and in the main region NEITHER does.
 *
 *
 * THE TRANSPOSE ITSELF
 * --------------------
 * The forward stretched derivative, at a point where the CPML is on,
 *
 *   psi' = b psi + c D(u)
 *   D_s  = ik D(u) + psi' = (ik+c) D(u) + b psi
 *
 * is linear in (D(u), psi).  With the forward update  f -= rho * D_s,
 * write g = rho * f_a.  Transposing the 2x2 block gives
 *
 *   w     = (ik+c) g - c psi_a          -> feeds the transposed stencil
 *   psi_a = b (psi_a - g)               -> the backward psi recursion
 *
 * and the transposed stencil follows from SBP, (D-)^T = -D+:
 *
 *   forward D-x  ->  adjoint  += D+x(w)
 *   forward D+x  ->  adjoint  += D-x(w)
 *
 * ORDERING TRAP: w must be formed from the INCOMING psi_a, before the
 * recursion overwrites it.  Both lines are written together below for
 * that reason.
 *
 * Every line degenerates, at ik=1 and b=c=0, to exactly the stencil in
 * elastic4_adj.c.  That is the built-in correctness check: with no PML
 * these routines reproduce the existing adjoint bit for bit.
 *
 *
 * SCATTER -> GATHER
 * -----------------
 * The transposed stencils are scatters (txx_a[i-n1] += ...), which race
 * under OpenMP.  Each routine therefore runs in two passes: a fill pass
 * that stages w over the forward's update set, then a gather pass that
 * applies the transposed stencil.  Each pass writes only its own column,
 * so parallelising over ix is race free.
 *
 * The work arrays carry a 2-cell halo in BOTH directions (stride nzp =
 * naz+4, origin offset 2*nzp+2).  The fill pass never writes the halo,
 * so it stays at its calloc'd zero and the gather can read i +/- 2 and
 * i +/- 2*nzp unguarded and still see a true zero outside the update
 * set.  Without the z halo the iz=0 and iz=naz-1 rows would wrap into
 * the neighbouring column and pick up a real value.
 *
 *   AUTHOR:
 *           True CFS-CPML adjoint for the fdelfwi FWI gradient.
 *           Forward stencils by Jan Thorbecke.
 *
 **********************************************************************/

/* --- state, shared by the three entry points ------------------------ */
static int    alloc_n1 = 0, alloc_n2 = 0, alloc_npml = -1;
static int    nzp = 0;                 /* padded z stride of the work arrays */

static unsigned char *msk_vx = NULL, *msk_vz = NULL;
static unsigned char *msk_p  = NULL, *msk_tz = NULL;

static float *psi_vx_x_a = NULL, *psi_vx_z_a = NULL;
static float *psi_vz_x_a = NULL, *psi_vz_z_a = NULL;
static float *psi_p_x_a  = NULL, *psi_p_z_a  = NULL;
static float *psi_tz_x_a = NULL, *psi_tz_z_a = NULL;

/* work arrays, halo'd; wbase is what gets free()d */
static float *wbase1 = NULL, *wbase2 = NULL, *wbase3 = NULL, *wbase4 = NULL;
static float *wrk1 = NULL, *wrk2 = NULL, *wrk3 = NULL, *wrk4 = NULL;

/* 1D CFS-CPML profiles: vel = velocity faces, p = stress cell centres,
 * t = stress faces.  See cpml_coef.c -- these three sets differ. */
static float *bvx=NULL, *cvx=NULL, *ivx=NULL, *bvz=NULL, *cvz=NULL, *ivz=NULL;
static float *bpx=NULL, *cpx=NULL, *ipx=NULL, *bpz=NULL, *cpz=NULL, *ipz=NULL;
static float *btx=NULL, *ctx=NULL, *itx=NULL, *btz=NULL, *ctz=NULL, *itz=NULL;


static void mark(unsigned char *m, int n1, int n2,
                 int ix0, int ix1, int iz0, int iz1,
                 unsigned char val, int *clash)
{
    int ix, iz;

    if (ix0 < 0)  ix0 = 0;
    if (iz0 < 0)  iz0 = 0;
    if (ix1 > n2) ix1 = n2;
    if (iz1 > n1) iz1 = n1;

    for (ix=ix0; ix<ix1; ix++) {
        for (iz=iz0; iz<iz1; iz++) {
            if (m[ix*n1+iz] != 0) (*clash)++;
            m[ix*n1+iz] = val;
        }
    }
}


/*---------------------------------------------------------------------
 * cpmlAdjBuild -- allocate and fill the masks and profiles.
 *
 * Every loop bound below is a verbatim copy of the matching forward
 * loop; the comment names the source.  Keep them diffable.
 *--------------------------------------------------------------------*/
static void cpmlAdjBuild(modPar mod, bndPar bnd, int verbose)
{
    int n1, n2, npml, nxp, clash;
    int zmin, zmax, ioPx_e, iePx_e, ioPz_e, iePz_e;
    int ioTx_e, ieTx_e, ioTz_e, ieTz_e, ize_pml;
    int ioPx_s, iePx_s, ioPz_s, iePz_s;
    size_t sizef, sizeb, sizew;

    n1   = mod.naz;
    n2   = mod.nax;
    npml = bnd.npml;

    if (alloc_n1 == n1 && alloc_n2 == n2 && alloc_npml == npml) return;

    nzp   = n1 + 4;
    nxp   = n2 + 4;
    sizef = (size_t)n2*n1*sizeof(float);
    sizeb = (size_t)n2*n1*sizeof(unsigned char);
    sizew = (size_t)nxp*nzp*sizeof(float);

    free(msk_vx); free(msk_vz); free(msk_p); free(msk_tz);
    free(psi_vx_x_a); free(psi_vx_z_a); free(psi_vz_x_a); free(psi_vz_z_a);
    free(psi_p_x_a);  free(psi_p_z_a);  free(psi_tz_x_a); free(psi_tz_z_a);
    free(wbase1); free(wbase2); free(wbase3); free(wbase4);
    free(bvx); free(cvx); free(ivx); free(bvz); free(cvz); free(ivz);
    free(bpx); free(cpx); free(ipx); free(bpz); free(cpz); free(ipz);
    free(btx); free(ctx); free(itx); free(btz); free(ctz); free(itz);

    msk_vx = (unsigned char *)calloc(sizeb,1); msk_vz = (unsigned char *)calloc(sizeb,1);
    msk_p  = (unsigned char *)calloc(sizeb,1); msk_tz = (unsigned char *)calloc(sizeb,1);

    psi_vx_x_a = (float *)calloc(sizef,1); psi_vx_z_a = (float *)calloc(sizef,1);
    psi_vz_x_a = (float *)calloc(sizef,1); psi_vz_z_a = (float *)calloc(sizef,1);
    psi_p_x_a  = (float *)calloc(sizef,1); psi_p_z_a  = (float *)calloc(sizef,1);
    psi_tz_x_a = (float *)calloc(sizef,1); psi_tz_z_a = (float *)calloc(sizef,1);

    wbase1 = (float *)calloc(sizew,1); wbase2 = (float *)calloc(sizew,1);
    wbase3 = (float *)calloc(sizew,1); wbase4 = (float *)calloc(sizew,1);
    wrk1 = wbase1 + 2*nzp + 2;  wrk2 = wbase2 + 2*nzp + 2;
    wrk3 = wbase3 + 2*nzp + 2;  wrk4 = wbase4 + 2*nzp + 2;

    bvx=(float*)calloc(n2,sizeof(float)); cvx=(float*)calloc(n2,sizeof(float));
    ivx=(float*)calloc(n2,sizeof(float)); bvz=(float*)calloc(n1,sizeof(float));
    cvz=(float*)calloc(n1,sizeof(float)); ivz=(float*)calloc(n1,sizeof(float));
    bpx=(float*)calloc(n2,sizeof(float)); cpx=(float*)calloc(n2,sizeof(float));
    ipx=(float*)calloc(n2,sizeof(float)); bpz=(float*)calloc(n1,sizeof(float));
    cpz=(float*)calloc(n1,sizeof(float)); ipz=(float*)calloc(n1,sizeof(float));
    btx=(float*)calloc(n2,sizeof(float)); ctx=(float*)calloc(n2,sizeof(float));
    itx=(float*)calloc(n2,sizeof(float)); btz=(float*)calloc(n1,sizeof(float));
    ctz=(float*)calloc(n1,sizeof(float)); itz=(float*)calloc(n1,sizeof(float));

    /* Identical builders to the forward -- see cpml_coef.c. */
    cpmlCoefVel(mod, bnd, bvx, cvx, ivx, bvz, cvz, ivz);
    cpmlCoefStr(mod, bnd, bpx, cpx, ipx, bpz, cpz, ipz,
                          btx, ctx, itx, btz, ctz, itz);

    clash = 0;

    /* ---- vx -------------------------------------------------------- */
    /* main: elastic4.c velocity loop */
    mark(msk_vx, n1, n2, mod.ioXx, mod.ieXx, mod.ioXz, mod.ieXz, 1, &clash);
    /* blocks: boundaries.c "vx: update all PML cells skipped by elastic4" */
    zmin = ((mod.ioXz-npml)>0?(mod.ioXz-npml):0);
    zmax = ((mod.ieXz+npml)<n1?(mod.ieXz+npml):n1);
    if (bnd.lef==2) mark(msk_vx,n1,n2, mod.ioXx-npml, mod.ioXx,      zmin, zmax, 2,&clash);
    if (bnd.rig==2) mark(msk_vx,n1,n2, mod.ieXx,      mod.ieXx+npml, zmin, zmax, 2,&clash);
    if (bnd.top==2) mark(msk_vx,n1,n2, mod.ioXx, mod.ieXx, mod.ioXz-npml, mod.ioXz,      2,&clash);
    if (bnd.bot==2) mark(msk_vx,n1,n2, mod.ioXx, mod.ieXx, mod.ieXz,      mod.ieXz+npml, 2,&clash);

    /* ---- vz -------------------------------------------------------- */
    mark(msk_vz, n1, n2, mod.ioZx, mod.ieZx, mod.ioZz, mod.ieZz, 1, &clash);
    zmin = ((mod.ioZz-npml)>0?(mod.ioZz-npml):0);
    zmax = ((mod.ieZz+npml)<n1?(mod.ieZz+npml):n1);
    if (bnd.lef==2) mark(msk_vz,n1,n2, mod.ioZx-npml, mod.ioZx,      zmin, zmax, 2,&clash);
    if (bnd.rig==2) mark(msk_vz,n1,n2, mod.ieZx,      mod.ieZx+npml, zmin, zmax, 2,&clash);
    if (bnd.top==2) mark(msk_vz,n1,n2, mod.ioZx, mod.ieZx, mod.ioZz-npml, mod.ioZz,      2,&clash);
    if (bnd.bot==2) mark(msk_vz,n1,n2, mod.ioZx, mod.ieZx, mod.ieZz,      mod.ieZz+npml, 2,&clash);

    /* ---- txx / tzz ------------------------------------------------- */
    /* main: elastic4.c scoped stress range (ioPx/ioPz shifted inward) */
    ioPx_s = mod.ioTx + (bnd.lef==2 ? npml : 0);
    iePx_s = mod.ieTx - (bnd.rig==2 ? npml : 0);
    ioPz_s = mod.ioPz + (bnd.top==2 ? npml : 0);
    iePz_s = mod.iePz - (bnd.bot==2 ? npml : 0);
    mark(msk_p, n1, n2, ioPx_s, iePx_s, ioPz_s, iePz_s, 1, &clash);
    /* blocks: boundaries.c "Direct CPML update for txx/tzz on P-grid" */
    ioPx_e  = mod.ioPx + (bnd.lef==2 ? npml : 0);
    iePx_e  = mod.iePx - (bnd.rig==2 ? npml : 0);
    ioPz_e  = mod.ioPz + (bnd.top==2 ? npml : 0);
    iePz_e  = mod.iePz - (bnd.bot==2 ? npml : 0);
    ize_pml = bnd.bot==2 ? (mod.ieXz+npml<n1 ? mod.ieXz+npml : n1) : iePz_e;
    zmin    = (bnd.top==2) ? ioPz_e-npml : mod.ioXz;
    zmax    = (bnd.bot==2) ? ize_pml     : mod.ieXz;
    if (bnd.lef==2) mark(msk_p,n1,n2, ioPx_e-npml, mod.ioXx,      zmin, zmax, 2,&clash);
    if (bnd.rig==2) mark(msk_p,n1,n2, mod.ieXx,    mod.ieXx+npml, zmin, zmax, 2,&clash);
    if (bnd.top==2) mark(msk_p,n1,n2, mod.ioXx, mod.ieXx, ioPz_e-npml, mod.ioXz,      2,&clash);
    if (bnd.bot==2) mark(msk_p,n1,n2, mod.ioXx, mod.ieXx, iePz_e,      mod.ieXz+npml, 2,&clash);

    /* ---- txz ------------------------------------------------------- */
    /* main: elastic4.c sets ioTx=ioPx_s etc. under PML */
    mark(msk_tz, n1, n2, ioPx_s, iePx_s, ioPz_s, iePz_s, 1, &clash);
    /* blocks: boundaries.c "Direct CPML update for txz on T-grid" */
    ioTx_e  = mod.ioTx + (bnd.lef==2 ? npml : 0);
    ieTx_e  = mod.ieTx - (bnd.rig==2 ? npml : 0);
    ioTz_e  = mod.ioTz + (bnd.top==2 ? npml : 0);
    ieTz_e  = mod.ieTz - (bnd.bot==2 ? npml : 0);
    ize_pml = bnd.bot==2 ? (mod.ieXz+npml<n1 ? mod.ieXz+npml : n1) : ieTz_e;
    zmin    = (bnd.top==2) ? ioTz_e-npml : mod.ioXz;
    zmax    = (bnd.bot==2) ? ize_pml     : mod.ieXz;
    if (bnd.lef==2) mark(msk_tz,n1,n2, ioTx_e-npml, ioTx_e,        zmin, zmax, 2,&clash);
    if (bnd.rig==2) mark(msk_tz,n1,n2, ieTx_e,      mod.ieXx+npml, zmin, zmax, 2,&clash);
    if (bnd.top==2) mark(msk_tz,n1,n2, ioTx_e, ieTx_e, ioTz_e-npml, mod.ioXz,      2,&clash);
    if (bnd.bot==2) mark(msk_tz,n1,n2, ioTx_e, ieTx_e, mod.ieXz,    mod.ieXz+npml, 2,&clash);

    if (clash) {
        vwarn("cpmlAdjBuild: %d cells are updated TWICE by the forward "
              "(main loop and a PML block overlap). The transpose assumes "
              "they are disjoint; the gradient in the PML will be wrong.",
              clash);
    }
    if (verbose>=3) {
        vmess("cpmlAdjBuild: CPML transpose active, npml=%d grid %dx%d", npml, n2, n1);
    }

    alloc_n1 = n1; alloc_n2 = n2; alloc_npml = npml;
}


/*---------------------------------------------------------------------
 * cpmlAdjReset -- zero the adjoint psi memory.
 *
 * The forward psi starts at zero at it=0, so its transpose must start
 * at zero at the END of the adjoint sweep -- i.e. the adjoint sweep,
 * which runs backwards, starts from zero too.  Call this once before
 * each backward loop.  Serial: not inside an OpenMP parallel region.
 *--------------------------------------------------------------------*/
void cpmlAdjReset(void)
{
    size_t sizef;

    if (!alloc_n1) return;                   /* nothing allocated yet */
    sizef = (size_t)alloc_n2*alloc_n1*sizeof(float);

    memset(psi_vx_x_a, 0, sizef); memset(psi_vx_z_a, 0, sizef);
    memset(psi_vz_x_a, 0, sizef); memset(psi_vz_z_a, 0, sizef);
    memset(psi_p_x_a,  0, sizef); memset(psi_p_z_a,  0, sizef);
    memset(psi_tz_x_a, 0, sizef); memset(psi_tz_z_a, 0, sizef);
}


/*---------------------------------------------------------------------
 * cpmlAdjF1T -- (A1 + A2)^T : adjoint STRESS update from adjoint vx,vz.
 *
 * Forward (per point, CPML coefficients only where the mask says 2):
 *   vx -= rox * [ S_x(D-x txx) + S_z(D+z txz) ]
 *   vz -= roz * [ S_z(D-z tzz) + S_x(D+x txz) ]
 *
 * Transpose:
 *   txx += D+x(w_x)
 *   tzz += D+z(w_zz)
 *   txz += D-z(w_z) + D-x(w_zx)
 *
 * With ik=1, b=c=0 this is exactly elastic4_adj.c:135-159.
 *--------------------------------------------------------------------*/
void cpmlAdjF1T(modPar mod, bndPar bnd, float *vx, float *vz,
    float *tzz, float *txx, float *txz, float *rox, float *roz)
{
    float c1, c2;
    int   ix, iz, n1, n2;
    float *wx, *wz, *wzz, *wzx;

#pragma omp master
{
    cpmlAdjBuild(mod, bnd, 0);
}
#pragma omp barrier

    c1 = 9.0/8.0;
    c2 = -1.0/24.0;
    n1 = mod.naz;
    n2 = mod.nax;
    wx = wrk1; wz = wrk2; wzz = wrk3; wzx = wrk4;

    /* ---- fill ------------------------------------------------------ */
#pragma omp for private(ix, iz) schedule(static)
    for (ix=0; ix<n2; ix++) {
#pragma simd
        for (iz=0; iz<n1; iz++) {
            int   i  = ix*n1  + iz;
            int   iw = ix*nzp + iz;
            float act, pml, ik, cc, bb, g, h;

            /* vx half */
            act = (msk_vx[i] != 0) ? 1.0f : 0.0f;
            pml = (msk_vx[i] == 2) ? 1.0f : 0.0f;
            g   = act*rox[i]*vx[i];
            ik  = 1.0f + pml*(ivx[ix]-1.0f);  cc = pml*cvx[ix];  bb = pml*bvx[ix];
            wx[iw]        = (ik+cc)*g - cc*psi_vx_x_a[i];
            psi_vx_x_a[i] = bb*(psi_vx_x_a[i] - g);
            ik  = 1.0f + pml*(ivz[iz]-1.0f);  cc = pml*cvz[iz];  bb = pml*bvz[iz];
            wz[iw]        = (ik+cc)*g - cc*psi_vx_z_a[i];
            psi_vx_z_a[i] = bb*(psi_vx_z_a[i] - g);

            /* vz half */
            act = (msk_vz[i] != 0) ? 1.0f : 0.0f;
            pml = (msk_vz[i] == 2) ? 1.0f : 0.0f;
            h   = act*roz[i]*vz[i];
            ik  = 1.0f + pml*(ivz[iz]-1.0f);  cc = pml*cvz[iz];  bb = pml*bvz[iz];
            wzz[iw]       = (ik+cc)*h - cc*psi_vz_z_a[i];
            psi_vz_z_a[i] = bb*(psi_vz_z_a[i] - h);
            ik  = 1.0f + pml*(ivx[ix]-1.0f);  cc = pml*cvx[ix];  bb = pml*bvx[ix];
            wzx[iw]       = (ik+cc)*h - cc*psi_vz_x_a[i];
            psi_vz_x_a[i] = bb*(psi_vz_x_a[i] - h);
        }
    }

    /* ---- gather ---------------------------------------------------- */
#pragma omp for private(ix, iz) schedule(static)
    for (ix=0; ix<n2; ix++) {
#pragma simd
        for (iz=0; iz<n1; iz++) {
            int i  = ix*n1  + iz;
            int iw = ix*nzp + iz;

            txx[i] += c1*(wx[iw+nzp]   - wx[iw])
                    + c2*(wx[iw+2*nzp] - wx[iw-nzp]);
            tzz[i] += c1*(wzz[iw+1] - wzz[iw])
                    + c2*(wzz[iw+2] - wzz[iw-1]);
            txz[i] += c1*(wz[iw]   - wz[iw-1])
                    + c2*(wz[iw+1] - wz[iw-2])
                    + c1*(wzx[iw]     - wzx[iw-nzp])
                    + c2*(wzx[iw+nzp] - wzx[iw-2*nzp]);
        }
    }
}


/*---------------------------------------------------------------------
 * cpmlAdjF2T -- (A3 + A4)^T : adjoint VELOCITY update from adjoint
 *               txx, tzz, txz.
 *
 * Forward:
 *   txx -= l2m*S_x(D+x vx) + lam*S_z(D+z vz)
 *   tzz -= lam*S_x(D+x vx) + l2m*S_z(D+z vz)
 *   txz -= mul*[ S_z(D-z vx) + S_x(D-x vz) ]
 *
 * Transpose:
 *   vx += D-x(u_x) + D+z(q_z)
 *   vz += D-z(u_z) + D+x(q_x)
 *
 * With ik=1, b=c=0 this is exactly elastic4_adj.c:88-115.
 *
 * NOTE txx and tzz share one psi pair in the forward (a single
 * stretched dvx feeds both), so the transpose feeds back the combined
 * sensitivity l2m*txx_a + lam*tzz_a -- not one psi per component.
 *--------------------------------------------------------------------*/
void cpmlAdjF2T(modPar mod, bndPar bnd, float *vx, float *vz,
    float *tzz, float *txx, float *txz,
    float *l2m, float *lam, float *mul)
{
    float c1, c2;
    int   ix, iz, n1, n2;
    float *ux, *uz, *qx, *qz;

#pragma omp master
{
    cpmlAdjBuild(mod, bnd, 0);
}
#pragma omp barrier

    c1 = 9.0/8.0;
    c2 = -1.0/24.0;
    n1 = mod.naz;
    n2 = mod.nax;
    ux = wrk1; uz = wrk2; qx = wrk3; qz = wrk4;

    /* ---- fill ------------------------------------------------------ */
#pragma omp for private(ix, iz) schedule(static)
    for (ix=0; ix<n2; ix++) {
#pragma simd
        for (iz=0; iz<n1; iz++) {
            int   i  = ix*n1  + iz;
            int   iw = ix*nzp + iz;
            float act, pml, ik, cc, bb, gx, gz, hh;

            /* txx/tzz half, cell-centred profile */
            act = (msk_p[i] != 0) ? 1.0f : 0.0f;
            pml = (msk_p[i] == 2) ? 1.0f : 0.0f;
            gx  = act*(l2m[i]*txx[i] + lam[i]*tzz[i]);
            gz  = act*(lam[i]*txx[i] + l2m[i]*tzz[i]);
            ik  = 1.0f + pml*(ipx[ix]-1.0f);  cc = pml*cpx[ix];  bb = pml*bpx[ix];
            ux[iw]       = (ik+cc)*gx - cc*psi_p_x_a[i];
            psi_p_x_a[i] = bb*(psi_p_x_a[i] - gx);
            ik  = 1.0f + pml*(ipz[iz]-1.0f);  cc = pml*cpz[iz];  bb = pml*bpz[iz];
            uz[iw]       = (ik+cc)*gz - cc*psi_p_z_a[i];
            psi_p_z_a[i] = bb*(psi_p_z_a[i] - gz);

            /* txz half, face profile (NOT the velocity one) */
            act = (msk_tz[i] != 0) ? 1.0f : 0.0f;
            pml = (msk_tz[i] == 2) ? 1.0f : 0.0f;
            hh  = act*mul[i]*txz[i];
            ik  = 1.0f + pml*(itx[ix]-1.0f);  cc = pml*ctx[ix];  bb = pml*btx[ix];
            qx[iw]        = (ik+cc)*hh - cc*psi_tz_x_a[i];
            psi_tz_x_a[i] = bb*(psi_tz_x_a[i] - hh);
            ik  = 1.0f + pml*(itz[iz]-1.0f);  cc = pml*ctz[iz];  bb = pml*btz[iz];
            qz[iw]        = (ik+cc)*hh - cc*psi_tz_z_a[i];
            psi_tz_z_a[i] = bb*(psi_tz_z_a[i] - hh);
        }
    }

    /* ---- gather ---------------------------------------------------- */
#pragma omp for private(ix, iz) schedule(static)
    for (ix=0; ix<n2; ix++) {
#pragma simd
        for (iz=0; iz<n1; iz++) {
            int i  = ix*n1  + iz;
            int iw = ix*nzp + iz;

            vx[i] += c1*(ux[iw]     - ux[iw-nzp])
                   + c2*(ux[iw+nzp] - ux[iw-2*nzp])
                   + c1*(qz[iw+1] - qz[iw])
                   + c2*(qz[iw+2] - qz[iw-1]);
            vz[i] += c1*(uz[iw]   - uz[iw-1])
                   + c2*(uz[iw+1] - uz[iw-2])
                   + c1*(qx[iw+nzp]   - qx[iw])
                   + c2*(qx[iw+2*nzp] - qx[iw-nzp]);
        }
    }
}


/*---------------------------------------------------------------------
 * cpmlAdjGhostT -- G^T, transpose of the ghost-column fill.
 *
 * Forward (boundaries.c, right after the velocity CPML blocks):
 *   lef==2:  v[1][iz] = v[2][iz];  v[0][iz] = v[2][iz]
 *   top==2:  v[ix][1] = v[ix][2];  v[ix][0] = v[ix][2]
 *
 * G = G_top . G_lef, so G^T = G_lef^T . G_top^T -- TOP FIRST here,
 * the opposite order to the forward.  Each overwrite transposes to a
 * scatter back onto its source followed by a zero of the destination.
 *--------------------------------------------------------------------*/
void cpmlAdjGhostT(modPar mod, bndPar bnd, float *vx, float *vz)
{
    int ix, iz, n1, n2;

    n1 = mod.naz;
    n2 = mod.nax;

    if (bnd.top == 2) {
#pragma omp for private(ix) schedule(static)
        for (ix=0; ix<n2; ix++) {
            vx[ix*n1+2] += vx[ix*n1+1] + vx[ix*n1+0];
            vx[ix*n1+1]  = 0.0f;
            vx[ix*n1+0]  = 0.0f;
            vz[ix*n1+2] += vz[ix*n1+1] + vz[ix*n1+0];
            vz[ix*n1+1]  = 0.0f;
            vz[ix*n1+0]  = 0.0f;
        }
    }
    if (bnd.lef == 2) {
#pragma omp for private(iz) schedule(static)
        for (iz=0; iz<n1; iz++) {
            vx[2*n1+iz] += vx[1*n1+iz] + vx[0*n1+iz];
            vx[1*n1+iz]  = 0.0f;
            vx[0*n1+iz]  = 0.0f;
            vz[2*n1+iz] += vz[1*n1+iz] + vz[0*n1+iz];
            vz[1*n1+iz]  = 0.0f;
            vz[0*n1+iz]  = 0.0f;
        }
    }
}
