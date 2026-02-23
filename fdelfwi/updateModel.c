/*
 * updateModel.c - Model vector utilities for FWI optimization.
 *
 * Bridges the optimizer's flat model vector (interior-only, physical
 * parameters) with the FD code's padded grid arrays (with FD scaling,
 * staggered-grid averaging, and boundary extensions).
 *
 * Supports both parameterizations:
 *   param=1 (Lame):     x = [lambda | mu     | rho]  (physical units)
 *   param=2 (velocity): x = [Vp     | Vs     | rho]
 *   acoustic:           x = [Vp     | rho]  or  [kappa | rho]
 *
 * Each block is nx*nz floats in column-major order (ix outer, iz inner).
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "fdelfwi.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))


/*--------------------------------------------------------------------
 * getInteriorOffsets -- Get padded grid offsets for interior domain.
 *
 * The "interior" starts at (ibndx, ibndz) in the padded array:
 *   ibndx = ioPx + ntap  (if left taper)
 *   ibndz = ioPz + ntap  (if top taper)
 *--------------------------------------------------------------------*/
static void getInteriorOffsets(modPar *mod, bndPar *bnd,
                               int *ibndx, int *ibndz)
{
	*ibndx = mod->ioPx;
	*ibndz = mod->ioPz;
	if (bnd->lef == 4 || bnd->lef == 2) *ibndx += bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) *ibndz += bnd->ntap;
}


/*--------------------------------------------------------------------
 * extendField -- Extend a single padded-grid array into taper zones.
 *
 * Copies the nearest interior value into absorbing/rigid boundary
 * padding for each side. Used for l2m, lam, muu, rox, roz.
 *
 * Parameters:
 *   arr   - padded array [nax * naz]
 *   n1    - naz (fast dimension)
 *   io/ie - interior loop bounds (origin, end) for this field
 *   bnd   - boundary parameters
 *   side  - which fields' offsets to use ('P', 'X', 'Z', 'T')
 *--------------------------------------------------------------------*/
static void extendFieldLR(float *arr, modPar *mod, bndPar *bnd,
                          int ioX, int ieX, int ioZ, int ieZ)
{
	int ix, iz, n1 = mod->naz;

	/* Left */
	if (bnd->lef == 4 || bnd->lef == 2) {
		for (ix = ioX - bnd->ntap; ix < ioX; ix++)
			for (iz = ioZ; iz < ieZ; iz++)
				arr[ix*n1+iz] = arr[ioX*n1+iz];
	}

	/* Right */
	if (bnd->rig == 4 || bnd->rig == 2) {
		for (ix = ieX; ix < ieX + bnd->ntap; ix++)
			for (iz = ioZ; iz < ieZ; iz++)
				arr[ix*n1+iz] = arr[(ieX-1)*n1+iz];
	}
}

static void extendFieldTB(float *arr, modPar *mod, bndPar *bnd,
                          int ioX, int ieX, int ioZ, int ieZ)
{
	int ix, iz, n1 = mod->naz;

	/* Extend x-range to include already-extended L/R zones */
	if (bnd->lef == 4 || bnd->lef == 2) ioX -= bnd->ntap;
	if (bnd->rig == 4 || bnd->rig == 2) ieX += bnd->ntap;

	/* Top */
	if (bnd->top == 4 || bnd->top == 2) {
		for (ix = ioX; ix < ieX; ix++)
			for (iz = ioZ - bnd->ntap; iz < ioZ; iz++)
				arr[ix*n1+iz] = arr[ix*n1+ioZ];
	}

	/* Bottom */
	if (bnd->bot == 4 || bnd->bot == 2) {
		for (ix = ioX; ix < ieX; ix++)
			for (iz = ieZ; iz < ieZ + bnd->ntap; iz++)
				arr[ix*n1+iz] = arr[ix*n1+(ieZ-1)];
	}
}


/*--------------------------------------------------------------------
 * recomputeFDcoefficients -- Recompute FD arrays from cp/cs/rho.
 *
 * Reads mod->cp, mod->cs, mod->rho (stored at P-grid interior
 * positions in the padded grid) and recomputes:
 *   - mod->l2m  (P grid)
 *   - mod->lam  (P grid, elastic only)
 *   - mod->muu  (Txz grid, elastic only)
 *   - mod->rox  (Vx grid)
 *   - mod->roz  (Vz grid)
 *
 * Then extends all arrays into absorbing boundary taper zones.
 *
 * This is the in-memory equivalent of readModel.c sections 5-8.
 *--------------------------------------------------------------------*/
void recomputeFDcoefficients(modPar *mod, bndPar *bnd)
{
	int ix, iz, n1, nx, nz;
	int ioXx, ioXz, ioZx, ioZz, ioPx, ioPz, ioTx, ioTz;
	float cp2, cs2, cs2a, cs2b, cs2c, cs11, cs12, cs21, cs22;
	float mul, mu, lamda2mu, lamda, bx, bz, fac;
	float *cp, *cs, *ro;
	size_t sizem;

	nx = mod->nx;
	nz = mod->nz;
	n1 = mod->naz;
	sizem = (size_t)mod->nax * mod->naz;
	fac = mod->dt / mod->dx;

	/* Grid offsets (before taper adjustment) */
	ioXx = mod->ioXx; ioXz = mod->ioXz;
	ioZx = mod->ioZx; ioZz = mod->ioZz;
	ioPx = mod->ioPx; ioPz = mod->ioPz;
	ioTx = mod->ioTx; ioTz = mod->ioTz;

	/* Adjust P and T offsets for taper (same as readModel.c) */
	if (bnd->lef == 4 || bnd->lef == 2) { ioPx += bnd->ntap; ioTx += bnd->ntap; }
	if (bnd->top == 4 || bnd->top == 2) { ioPz += bnd->ntap; ioTz += bnd->ntap; }

	/* Extract interior to temp arrays (simplifies edge-case indexing) */
	cp = (float *)malloc(nx * nz * sizeof(float));
	ro = (float *)malloc(nx * nz * sizeof(float));
	cs = NULL;
	if (mod->ischeme > 2)
		cs = (float *)malloc(nx * nz * sizeof(float));

	for (ix = 0; ix < nx; ix++) {
		for (iz = 0; iz < nz; iz++) {
			cp[ix*nz+iz] = mod->cp[(ix+ioPx)*n1+iz+ioPz];
			ro[ix*nz+iz] = mod->rho[(ix+ioPx)*n1+iz+ioPz];
			if (cs)
				cs[ix*nz+iz] = mod->cs[(ix+ioPx)*n1+iz+ioPz];
		}
	}

	/* Zero out FD coefficient arrays */
	memset(mod->l2m, 0, sizem * sizeof(float));
	memset(mod->rox, 0, sizem * sizeof(float));
	memset(mod->roz, 0, sizem * sizeof(float));
	if (mod->lam) memset(mod->lam, 0, sizem * sizeof(float));
	if (mod->muu) memset(mod->muu, 0, sizem * sizeof(float));

	/* ============================================================ */
	/* Compute FD coefficients (same logic as readModel.c)          */
	/* ============================================================ */

	if (mod->ischeme > 2) {
		/* ---- ELASTIC ---- */

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

		mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/ro[ix*nz+iz];
		mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/ro[ix*nz+iz];
		mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
		mod->lam[(ix+ioPx)*n1+iz+ioPz] = fac*lamda;
		mod->muu[(ix+ioTx)*n1+iz+ioTz] = fac*mu;

		/* Interior */
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
	} else {
		/* ---- ACOUSTIC ---- */

		/* Bottom edge */
		iz = nz - 1;
		for (ix = 0; ix < nx - 1; ix++) {
			cp2 = cp[ix*nz+iz]*cp[ix*nz+iz];
			lamda2mu = cp2*ro[ix*nz+iz];
			bx = 0.5*(ro[ix*nz+iz]+ro[(ix+1)*nz+iz]);
			mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/bx;
			mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/ro[ix*nz+iz];
			mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
		}

		/* Right edge */
		ix = nx - 1;
		for (iz = 0; iz < nz - 1; iz++) {
			cp2 = cp[ix*nz+iz]*cp[ix*nz+iz];
			lamda2mu = cp2*ro[ix*nz+iz];
			bz = 0.5*(ro[ix*nz+iz]+ro[ix*nz+iz+1]);
			mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/ro[ix*nz+iz];
			mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/bz;
			mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;
		}

		/* Bottom-right corner */
		ix = nx - 1; iz = nz - 1;
		cp2 = cp[ix*nz+iz]*cp[ix*nz+iz];
		lamda2mu = cp2*ro[ix*nz+iz];
		mod->rox[(ix+ioXx)*n1+iz+ioXz] = fac/ro[ix*nz+iz];
		mod->roz[(ix+ioZx)*n1+iz+ioZz] = fac/ro[ix*nz+iz];
		mod->l2m[(ix+ioPx)*n1+iz+ioPz] = fac*lamda2mu;

		/* Interior */
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

	/* Zero-velocity topography: zero buoyancy where l2m is zero */
	for (ix = 0; ix < nx; ix++) {
		for (iz = 0; iz < nz; iz++) {
			if (mod->l2m[(ix+ioPx)*n1+iz+ioPz] == 0.0) {
				mod->rox[(ix+ioXx)*n1+iz+ioXz] = 0.0;
				mod->roz[(ix+ioZx)*n1+iz+ioZz] = 0.0;
			}
		}
	}

	/* ============================================================ */
	/* Boundary extension into taper zones                          */
	/* ============================================================ */

	/* rox: Vx grid offsets */
	extendFieldLR(mod->rox, mod, bnd,
	              mod->ioXx, mod->ieXx, mod->ioXz, mod->ieXz);
	extendFieldTB(mod->rox, mod, bnd,
	              mod->ioXx, mod->ieXx, mod->ioXz, mod->ieXz);

	/* roz: Vz grid offsets */
	extendFieldLR(mod->roz, mod, bnd,
	              mod->ioZx, mod->ieZx, mod->ioZz, mod->ieZz);
	extendFieldTB(mod->roz, mod, bnd,
	              mod->ioZx, mod->ieZx, mod->ioZz, mod->ieZz);

	/* l2m: P grid -- note: readModel.c uses shifted offsets for L/R */
	{
		int l2m_ioX = mod->ioPx;
		int l2m_ieX = mod->iePx;
		int l2m_ioZ = mod->ioPz;
		int l2m_ieZ = mod->iePz;
		if (bnd->lef == 4 || bnd->lef == 2) l2m_ioX += bnd->ntap;
		if (bnd->rig == 4 || bnd->rig == 2) l2m_ieX -= bnd->ntap;
		if (bnd->top == 4 || bnd->top == 2) l2m_ioZ += bnd->ntap;
		if (bnd->bot == 4 || bnd->bot == 2) l2m_ieZ -= bnd->ntap;
		extendFieldLR(mod->l2m, mod, bnd,
		              l2m_ioX, l2m_ieX, mod->ioPz, mod->iePz);
		extendFieldTB(mod->l2m, mod, bnd,
		              mod->ioPx, mod->iePx, l2m_ioZ, l2m_ieZ);
	}

	if (mod->ischeme > 2) {
		/* lam: same bounds as l2m */
		int lam_ioX = mod->ioPx;
		int lam_ieX = mod->iePx;
		int lam_ioZ = mod->ioPz;
		int lam_ieZ = mod->iePz;
		if (bnd->lef == 4 || bnd->lef == 2) lam_ioX += bnd->ntap;
		if (bnd->rig == 4 || bnd->rig == 2) lam_ieX -= bnd->ntap;
		if (bnd->top == 4 || bnd->top == 2) lam_ioZ += bnd->ntap;
		if (bnd->bot == 4 || bnd->bot == 2) lam_ieZ -= bnd->ntap;
		extendFieldLR(mod->lam, mod, bnd,
		              lam_ioX, lam_ieX, mod->ioPz, mod->iePz);
		extendFieldTB(mod->lam, mod, bnd,
		              mod->ioPx, mod->iePx, lam_ioZ, lam_ieZ);

		/* muu: Txz grid offsets */
		int muu_ioX = mod->ioTx;
		int muu_ieX = mod->ieTx;
		int muu_ioZ = mod->ioTz;
		int muu_ieZ = mod->ieTz;
		if (bnd->lef == 4 || bnd->lef == 2) muu_ioX += bnd->ntap;
		if (bnd->rig == 4 || bnd->rig == 2) muu_ieX -= bnd->ntap;
		if (bnd->top == 4 || bnd->top == 2) muu_ioZ += bnd->ntap;
		if (bnd->bot == 4 || bnd->bot == 2) muu_ieZ -= bnd->ntap;
		extendFieldLR(mod->muu, mod, bnd,
		              muu_ioX, muu_ieX, mod->ioTz, mod->ieTz);
		extendFieldTB(mod->muu, mod, bnd,
		              mod->ioTx, mod->ieTx, muu_ioZ, muu_ieZ);
	}

	free(cp); free(ro);
	if (cs) free(cs);
}


/*--------------------------------------------------------------------
 * extractModelVector -- Extract interior model into flat vector.
 *
 * param=1 (Lame):
 *   x = [lambda(nx*nz) | mu(nx*nz) | rho(nx*nz)]
 *   lambda = rho*(Vp^2 - 2*Vs^2),  mu = rho*Vs^2
 *
 * param=2 (velocity):
 *   x = [Vp(nx*nz) | Vs(nx*nz) | rho(nx*nz)]
 *
 * For acoustic (ischeme <= 2), only 2 parameters:
 *   param=1: x = [kappa(nx*nz) | rho(nx*nz)]   where kappa = rho*Vp^2
 *   param=2: x = [Vp(nx*nz) | rho(nx*nz)]
 *--------------------------------------------------------------------*/
void extractModelVector(float *x, modPar *mod, bndPar *bnd, int param)
{
	int ix, iz, ibndx, ibndz, n1;
	int nx = mod->nx, nz = mod->nz;
	int nmodel = nx * nz;
	int elastic = (mod->ischeme > 2);

	getInteriorOffsets(mod, bnd, &ibndx, &ibndz);
	n1 = mod->naz;

	for (ix = 0; ix < nx; ix++) {
		for (iz = 0; iz < nz; iz++) {
			int ig = (ix + ibndx) * n1 + iz + ibndz;
			int il = ix * nz + iz;
			float vp  = mod->cp[ig];
			float rho_val = mod->rho[ig];

			if (param == 1) {
				/* Lame parameterization */
				float vp2 = vp * vp;
				if (elastic) {
					float vs  = mod->cs[ig];
					float vs2 = vs * vs;
					x[il]            = rho_val * (vp2 - 2.0f * vs2); /* lambda */
					x[nmodel + il]   = rho_val * vs2;                 /* mu */
					x[2*nmodel + il] = rho_val;                       /* rho */
				} else {
					x[il]          = rho_val * vp2; /* kappa = lambda+2mu */
					x[nmodel + il] = rho_val;       /* rho */
				}
			} else {
				/* Velocity parameterization */
				if (elastic) {
					x[il]            = vp;                 /* Vp */
					x[nmodel + il]   = mod->cs[ig];        /* Vs */
					x[2*nmodel + il] = rho_val;            /* rho */
				} else {
					x[il]          = vp;      /* Vp */
					x[nmodel + il] = rho_val; /* rho */
				}
			}
		}
	}
}


/*--------------------------------------------------------------------
 * injectModelVector -- Inject flat vector back into FD arrays.
 *
 * Updates mod->cp, mod->cs, mod->rho at interior positions, then
 * calls recomputeFDcoefficients() to rebuild all FD arrays.
 *--------------------------------------------------------------------*/
void injectModelVector(float *x, modPar *mod, bndPar *bnd, int param)
{
	int ix, iz, ibndx, ibndz, n1;
	int nx = mod->nx, nz = mod->nz;
	int nmodel = nx * nz;
	int elastic = (mod->ischeme > 2);

	getInteriorOffsets(mod, bnd, &ibndx, &ibndz);
	n1 = mod->naz;

	for (ix = 0; ix < nx; ix++) {
		for (iz = 0; iz < nz; iz++) {
			int ig = (ix + ibndx) * n1 + iz + ibndz;
			int il = ix * nz + iz;

			if (param == 1) {
				/* Lame -> velocity */
				float rho_val, lam_val, mu_val;
				if (elastic) {
					lam_val = x[il];
					mu_val  = x[nmodel + il];
					rho_val = x[2*nmodel + il];
					mod->cp[ig]  = sqrtf((lam_val + 2.0f * mu_val) / rho_val);
					mod->cs[ig]  = sqrtf(mu_val / rho_val);
					mod->rho[ig] = rho_val;
				} else {
					float kappa = x[il];
					rho_val = x[nmodel + il];
					mod->cp[ig]  = sqrtf(kappa / rho_val);
					mod->rho[ig] = rho_val;
				}
			} else {
				/* Velocity: direct assignment */
				if (elastic) {
					mod->cp[ig]  = x[il];
					mod->cs[ig]  = x[nmodel + il];
					mod->rho[ig] = x[2*nmodel + il];
				} else {
					mod->cp[ig]  = x[il];
					mod->rho[ig] = x[nmodel + il];
				}
			}
		}
	}

	/* Rebuild all FD coefficient arrays and boundary extensions */
	recomputeFDcoefficients(mod, bnd);
}


/*--------------------------------------------------------------------
 * extractGradientVector -- Strip boundary and concatenate gradients.
 *
 * Extracts interior (nx*nz) portions from the padded gradient arrays
 * and concatenates them into the flat optimizer gradient vector.
 *
 * If param=2 (velocity), applies chain rule conversion first.
 *
 * grad1: g_lambda or g_Vp  (padded, sizem)
 * grad2: g_mu or g_Vs      (padded, sizem, NULL for acoustic)
 * grad3: g_rho             (padded, sizem)
 * g:     output flat vector [nparam * nx * nz]
 *--------------------------------------------------------------------*/
void extractGradientVector(float *g, float *grad1, float *grad2, float *grad3,
                           modPar *mod, bndPar *bnd, int param)
{
	int ix, iz, ibndx, ibndz, n1;
	int nx = mod->nx, nz = mod->nz;
	int nmodel = nx * nz;
	int elastic = (mod->ischeme > 2);
	size_t sizem = (size_t)mod->nax * mod->naz;

	getInteriorOffsets(mod, bnd, &ibndx, &ibndz);
	n1 = mod->naz;

	/* Apply velocity chain rule if needed (in-place on padded arrays) */
	if (param == 2 && elastic) {
		convertGradientToVelocity(grad1, grad2, grad3,
		                          mod->cp, mod->cs, mod->rho, sizem);
	}

	/* Strip boundary: copy interior to flat vector */
	for (ix = 0; ix < nx; ix++) {
		for (iz = 0; iz < nz; iz++) {
			int ig = (ix + ibndx) * n1 + iz + ibndz;
			int il = ix * nz + iz;

			g[il] = grad1 ? grad1[ig] : 0.0f;
			if (elastic)
				g[nmodel + il] = grad2 ? grad2[ig] : 0.0f;
			g[(elastic ? 2 : 1) * nmodel + il] = grad3 ? grad3[ig] : 0.0f;
		}
	}
}
