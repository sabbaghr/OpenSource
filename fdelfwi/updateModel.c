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

	/* NOTE: caller must pass the FULL x-range including L/R taper zones.
	 *
	 * For Vx/Vz fields (ioXx, ioZx): these offsets already include
	 * the taper adjustment (getParameters.c lines 512-513), so
	 * callers subtract ntap to get the L/R taper start.
	 *
	 * For P/Txz fields (ioPx, ioTx): these offsets do NOT include
	 * taper (getParameters.c lines 518-519, commented out), so the
	 * L/R taper zone sits within [ioPx, iePx]. Callers should pass
	 * ioPx/iePx directly — no subtraction needed.
	 */

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

	/* rox: Vx grid offsets
	 * ioXx/ieXx include taper offset (getParameters.c line 512).
	 * L/R fills from ioXx-ntap to ioXx. T/B must span the same
	 * x-range including those L/R taper columns. */
	extendFieldLR(mod->rox, mod, bnd,
	              mod->ioXx, mod->ieXx, mod->ioXz, mod->ieXz);
	{
		int rox_tbx = mod->ioXx, rox_tbxe = mod->ieXx;
		if (bnd->lef == 4 || bnd->lef == 2) rox_tbx -= bnd->ntap;
		if (bnd->rig == 4 || bnd->rig == 2) rox_tbxe += bnd->ntap;
		extendFieldTB(mod->rox, mod, bnd,
		              rox_tbx, rox_tbxe, mod->ioXz, mod->ieXz);
	}

	/* roz: Vz grid offsets (same pattern as rox) */
	extendFieldLR(mod->roz, mod, bnd,
	              mod->ioZx, mod->ieZx, mod->ioZz, mod->ieZz);
	{
		int roz_tbx = mod->ioZx, roz_tbxe = mod->ieZx;
		if (bnd->lef == 4 || bnd->lef == 2) roz_tbx -= bnd->ntap;
		if (bnd->rig == 4 || bnd->rig == 2) roz_tbxe += bnd->ntap;
		extendFieldTB(mod->roz, mod, bnd,
		              roz_tbx, roz_tbxe, mod->ioZz, mod->ieZz);
	}

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


/*--------------------------------------------------------------------
 * perturbFDcoefficients -- Linearized FD coefficient perturbation.
 *
 * Converts the optimizer's perturbation vector (e.g. opt->d from
 * the TRN inner CG) into padded-grid FD coefficient perturbations
 * used by the Born virtual source injection (born_vsrc.c).
 *
 * This is the linearized (Jacobian) version of recomputeFDcoefficients:
 *   rox = fac/bx          -> delta_rox = -(rox^2/fac) * delta_bx
 *   l2m = fac*(lambda+2mu)-> delta_l2m = fac*(delta_lam + 2*delta_mu)
 *   lam = fac*lambda       -> delta_lam_out = fac*delta_lam
 *   muu = fac*H(mu_4corners) -> delta_mul = fac*dH/df * delta_f
 *
 * where H is the harmonic mean of mu = rho*Vs^2 at 4 corners.
 *
 * The perturbation vector dpert has the same layout as the model
 * vector from extractModelVector:
 *   param=1 (Lame):     [delta_lam | delta_mu | delta_rho]
 *   param=2 (velocity): [delta_Vp  | delta_Vs | delta_rho]
 *
 * For param=2, the chain rule converts to Lame perturbations first:
 *   delta_lam = 2*rho*Vp*dVp - 4*rho*Vs*dVs + (Vp^2-2Vs^2)*drho
 *   delta_mu  = 2*rho*Vs*dVs + Vs^2*drho
 *
 * All output arrays must be pre-allocated to sizem = nax*naz.
 *
 * See HESSIAN_MATH.md Section 1.5 for full derivation.
 *--------------------------------------------------------------------*/
void perturbFDcoefficients(modPar *mod, bndPar *bnd,
                           float *dpert, int param,
                           float *delta_rox, float *delta_roz,
                           float *delta_l2m, float *delta_lam,
                           float *delta_mul)
{
	int ix, iz, n1, nx, nz;
	int ioXx, ioXz, ioZx, ioZz, ioPx, ioPz, ioTx, ioTz;
	float fac;
	float *cp, *cs, *ro;
	float *dlam_i, *dmu_i, *drho_i;
	size_t sizem;
	int nmodel, elastic;

	nx = mod->nx;
	nz = mod->nz;
	n1 = mod->naz;
	nmodel = nx * nz;
	elastic = (mod->ischeme > 2);
	sizem = (size_t)mod->nax * mod->naz;
	fac = mod->dt / mod->dx;

	/* Grid offsets (same as recomputeFDcoefficients) */
	ioXx = mod->ioXx; ioXz = mod->ioXz;
	ioZx = mod->ioZx; ioZz = mod->ioZz;
	ioPx = mod->ioPx; ioPz = mod->ioPz;
	ioTx = mod->ioTx; ioTz = mod->ioTz;

	/* Adjust P and T offsets for taper */
	if (bnd->lef == 4 || bnd->lef == 2) { ioPx += bnd->ntap; ioTx += bnd->ntap; }
	if (bnd->top == 4 || bnd->top == 2) { ioPz += bnd->ntap; ioTz += bnd->ntap; }

	/* ============================================================ */
	/* 1. Extract current model to interior arrays                  */
	/* ============================================================ */
	cp = (float *)malloc(nmodel * sizeof(float));
	ro = (float *)malloc(nmodel * sizeof(float));
	cs = NULL;
	if (elastic)
		cs = (float *)malloc(nmodel * sizeof(float));

	for (ix = 0; ix < nx; ix++) {
		for (iz = 0; iz < nz; iz++) {
			int ig = (ix + ioPx) * n1 + iz + ioPz;
			cp[ix*nz+iz] = mod->cp[ig];
			ro[ix*nz+iz] = mod->rho[ig];
			if (cs)
				cs[ix*nz+iz] = mod->cs[ig];
		}
	}

	/* ============================================================ */
	/* 2. Convert perturbation to Lame space on interior grid       */
	/* ============================================================ */
	dlam_i = (float *)calloc(nmodel, sizeof(float));
	dmu_i  = elastic ? (float *)calloc(nmodel, sizeof(float)) : NULL;
	drho_i = (float *)calloc(nmodel, sizeof(float));

	if (param == 1) {
		/* Already Lame: dpert = [delta_lam | delta_mu | delta_rho] */
		memcpy(dlam_i, dpert, nmodel * sizeof(float));
		if (elastic)
			memcpy(dmu_i, dpert + nmodel, nmodel * sizeof(float));
		memcpy(drho_i, dpert + (elastic ? 2 : 1) * nmodel,
		       nmodel * sizeof(float));
	} else {
		/* Velocity: dpert = [dVp | dVs | drho] -> convert to Lame */
		float *dvp = dpert;
		float *dvs = elastic ? dpert + nmodel : NULL;
		float *drp = dpert + (elastic ? 2 : 1) * nmodel;

		for (ix = 0; ix < nx; ix++) {
			for (iz = 0; iz < nz; iz++) {
				int il = ix * nz + iz;
				float vp  = cp[il];
				float rho_val = ro[il];
				float dvp_val = dvp[il];
				float drho_val = drp[il];

				if (elastic) {
					float vs  = cs[il];
					float dvs_val = dvs[il];

					/* delta_lam = d(rho*(Vp^2-2Vs^2))
					 *           = 2*rho*Vp*dVp - 4*rho*Vs*dVs
					 *             + (Vp^2-2Vs^2)*drho */
					dlam_i[il] = 2.0f*rho_val*vp*dvp_val
					           - 4.0f*rho_val*vs*dvs_val
					           + (vp*vp - 2.0f*vs*vs)*drho_val;

					/* delta_mu = d(rho*Vs^2)
					 *          = 2*rho*Vs*dVs + Vs^2*drho */
					dmu_i[il] = 2.0f*rho_val*vs*dvs_val
					          + vs*vs*drho_val;
				} else {
					/* Acoustic: delta_kappa = d(rho*Vp^2)
					 *                       = 2*rho*Vp*dVp + Vp^2*drho */
					dlam_i[il] = 2.0f*rho_val*vp*dvp_val
					           + vp*vp*drho_val;
				}
				drho_i[il] = drho_val;
			}
		}
	}

	/* ============================================================ */
	/* 3. Zero output arrays                                        */
	/* ============================================================ */
	memset(delta_rox, 0, sizem * sizeof(float));
	memset(delta_roz, 0, sizem * sizeof(float));
	memset(delta_l2m, 0, sizem * sizeof(float));
	if (delta_lam)
		memset(delta_lam, 0, sizem * sizeof(float));
	if (delta_mul)
		memset(delta_mul, 0, sizem * sizeof(float));

	/* ============================================================ */
	/* 4. Compute FD coefficient perturbations at interior points   */
	/*                                                               */
	/* Uses clamped indices for stagger averaging at edges, which    */
	/* matches the edge-case handling in recomputeFDcoefficients     */
	/* (where boundary neighbors are replicated).                    */
	/* ============================================================ */

	for (ix = 0; ix < nx; ix++) {
		int ixp1 = (ix < nx - 1) ? ix + 1 : ix;

		for (iz = 0; iz < nz; iz++) {
			int izp1 = (iz < nz - 1) ? iz + 1 : iz;
			int il = ix * nz + iz;
			float dl = dlam_i[il];
			float dm = dmu_i ? dmu_i[il] : 0.0f;
			float dr = drho_i[il];

			/* ---- delta_l2m = fac * (delta_lam + 2*delta_mu) ---- */
			delta_l2m[(ix+ioPx)*n1 + iz+ioPz] = fac * (dl + 2.0f*dm);

			/* ---- delta_lam = fac * delta_lam ---- */
			if (delta_lam && elastic)
				delta_lam[(ix+ioPx)*n1 + iz+ioPz] = fac * dl;

			/* ---- delta_rox ----
			 * rox = fac/bx where bx = 0.5*(rho[ix]+rho[ix+1])
			 * d(rox) = -fac/bx^2 * d(bx) = -(rox^2/fac) * d(bx)
			 * d(bx) = 0.5*(drho[ix] + drho[ix+1])
			 * At right edge: ix+1 clamped to ix -> d(bx) = drho[ix] */
			{
				float rox_cur = mod->rox[(ix+ioXx)*n1 + iz+ioXz];
				if (rox_cur != 0.0f) {
					float dbx = 0.5f * (dr + drho_i[ixp1*nz + iz]);
					delta_rox[(ix+ioXx)*n1 + iz+ioXz] =
						-(rox_cur * rox_cur / fac) * dbx;
				}
			}

			/* ---- delta_roz ----
			 * roz = fac/bz where bz = 0.5*(rho[ix,iz]+rho[ix,iz+1])
			 * d(roz) = -(roz^2/fac) * d(bz)
			 * d(bz) = 0.5*(drho[ix,iz] + drho[ix,iz+1])
			 * At bottom edge: iz+1 clamped -> d(bz) = drho[ix,iz] */
			{
				float roz_cur = mod->roz[(ix+ioZx)*n1 + iz+ioZz];
				if (roz_cur != 0.0f) {
					float dbz = 0.5f * (dr + drho_i[ix*nz + izp1]);
					delta_roz[(ix+ioZx)*n1 + iz+ioZz] =
						-(roz_cur * roz_cur / fac) * dbz;
				}
			}

			/* ---- delta_mul (harmonic avg linearization) ----
			 * muu = fac * H(f1,f2,f3,f4)
			 * where f_i = rho_i * Vs_i^2  at 4 Txz corners
			 * H = 4 / (1/f1 + 1/f2 + 1/f3 + 1/f4)
			 *
			 * dH/df_i = H^2 / (4 * f_i^2)
			 * delta_H = sum_i [ H^2/(4*f_i^2) * delta_f_i ]
			 * delta_mul = fac * delta_H
			 *
			 * delta_f_i = delta_mu_i  (in Lame space) */
			if (elastic && delta_mul) {
				float f[4], df[4];
				int k;

				f[0] = ro[ix*nz+iz]     * cs[ix*nz+iz]     * cs[ix*nz+iz];
				f[1] = ro[ix*nz+izp1]   * cs[ix*nz+izp1]   * cs[ix*nz+izp1];
				f[2] = ro[ixp1*nz+iz]   * cs[ixp1*nz+iz]   * cs[ixp1*nz+iz];
				f[3] = ro[ixp1*nz+izp1] * cs[ixp1*nz+izp1] * cs[ixp1*nz+izp1];

				df[0] = dmu_i[ix*nz+iz];
				df[1] = dmu_i[ix*nz+izp1];
				df[2] = dmu_i[ixp1*nz+iz];
				df[3] = dmu_i[ixp1*nz+izp1];

				if (f[0] > 0.0f) {
					float sum_inv = 1.0f/f[0] + 1.0f/f[1]
					              + 1.0f/f[2] + 1.0f/f[3];
					float h = 4.0f / sum_inv;
					float h2_over4 = h * h * 0.25f;
					float dh = 0.0f;
					for (k = 0; k < 4; k++)
						dh += (h2_over4 / (f[k] * f[k])) * df[k];
					delta_mul[(ix+ioTx)*n1 + iz+ioTz] = fac * dh;
				}
			}
		}
	}

	/* ============================================================ */
	/* 5. Zero-velocity topography guard                            */
	/*    Where l2m=0 (no material), all perturbations must be 0.   */
	/* ============================================================ */
	for (ix = 0; ix < nx; ix++) {
		for (iz = 0; iz < nz; iz++) {
			if (mod->l2m[(ix+ioPx)*n1 + iz+ioPz] == 0.0f) {
				delta_rox[(ix+ioXx)*n1 + iz+ioXz] = 0.0f;
				delta_roz[(ix+ioZx)*n1 + iz+ioZz] = 0.0f;
				delta_l2m[(ix+ioPx)*n1 + iz+ioPz] = 0.0f;
				if (delta_lam)
					delta_lam[(ix+ioPx)*n1 + iz+ioPz] = 0.0f;
				if (delta_mul)
					delta_mul[(ix+ioTx)*n1 + iz+ioTz] = 0.0f;
			}
		}
	}

	/* ============================================================ */
	/* 6. Boundary extension into taper zones                       */
	/*    Same pattern as recomputeFDcoefficients.                   */
	/* ============================================================ */

	/* delta_rox: Vx grid offsets (ioXx includes taper) */
	extendFieldLR(delta_rox, mod, bnd,
	              mod->ioXx, mod->ieXx, mod->ioXz, mod->ieXz);
	{
		int rox_tbx = mod->ioXx, rox_tbxe = mod->ieXx;
		if (bnd->lef == 4 || bnd->lef == 2) rox_tbx -= bnd->ntap;
		if (bnd->rig == 4 || bnd->rig == 2) rox_tbxe += bnd->ntap;
		extendFieldTB(delta_rox, mod, bnd,
		              rox_tbx, rox_tbxe, mod->ioXz, mod->ieXz);
	}

	/* delta_roz: Vz grid offsets (ioZx includes taper) */
	extendFieldLR(delta_roz, mod, bnd,
	              mod->ioZx, mod->ieZx, mod->ioZz, mod->ieZz);
	{
		int roz_tbx = mod->ioZx, roz_tbxe = mod->ieZx;
		if (bnd->lef == 4 || bnd->lef == 2) roz_tbx -= bnd->ntap;
		if (bnd->rig == 4 || bnd->rig == 2) roz_tbxe += bnd->ntap;
		extendFieldTB(delta_roz, mod, bnd,
		              roz_tbx, roz_tbxe, mod->ioZz, mod->ieZz);
	}

	/* delta_l2m: P grid */
	{
		int l2m_ioX = mod->ioPx, l2m_ieX = mod->iePx;
		int l2m_ioZ = mod->ioPz, l2m_ieZ = mod->iePz;
		if (bnd->lef == 4 || bnd->lef == 2) l2m_ioX += bnd->ntap;
		if (bnd->rig == 4 || bnd->rig == 2) l2m_ieX -= bnd->ntap;
		if (bnd->top == 4 || bnd->top == 2) l2m_ioZ += bnd->ntap;
		if (bnd->bot == 4 || bnd->bot == 2) l2m_ieZ -= bnd->ntap;
		extendFieldLR(delta_l2m, mod, bnd,
		              l2m_ioX, l2m_ieX, mod->ioPz, mod->iePz);
		extendFieldTB(delta_l2m, mod, bnd,
		              mod->ioPx, mod->iePx, l2m_ioZ, l2m_ieZ);
	}

	if (elastic) {
		/* delta_lam: same bounds as delta_l2m */
		if (delta_lam) {
			int lam_ioX = mod->ioPx, lam_ieX = mod->iePx;
			int lam_ioZ = mod->ioPz, lam_ieZ = mod->iePz;
			if (bnd->lef == 4 || bnd->lef == 2) lam_ioX += bnd->ntap;
			if (bnd->rig == 4 || bnd->rig == 2) lam_ieX -= bnd->ntap;
			if (bnd->top == 4 || bnd->top == 2) lam_ioZ += bnd->ntap;
			if (bnd->bot == 4 || bnd->bot == 2) lam_ieZ -= bnd->ntap;
			extendFieldLR(delta_lam, mod, bnd,
			              lam_ioX, lam_ieX, mod->ioPz, mod->iePz);
			extendFieldTB(delta_lam, mod, bnd,
			              mod->ioPx, mod->iePx, lam_ioZ, lam_ieZ);
		}

		/* delta_mul: Txz grid offsets */
		if (delta_mul) {
			int muu_ioX = mod->ioTx, muu_ieX = mod->ieTx;
			int muu_ioZ = mod->ioTz, muu_ieZ = mod->ieTz;
			if (bnd->lef == 4 || bnd->lef == 2) muu_ioX += bnd->ntap;
			if (bnd->rig == 4 || bnd->rig == 2) muu_ieX -= bnd->ntap;
			if (bnd->top == 4 || bnd->top == 2) muu_ioZ += bnd->ntap;
			if (bnd->bot == 4 || bnd->bot == 2) muu_ieZ -= bnd->ntap;
			extendFieldLR(delta_mul, mod, bnd,
			              muu_ioX, muu_ieX, mod->ioTz, mod->ieTz);
			extendFieldTB(delta_mul, mod, bnd,
			              mod->ioTx, mod->ieTx, muu_ioZ, muu_ieZ);
		}
	}

	/* ============================================================ */
	/* Cleanup                                                       */
	/* ============================================================ */
	free(cp); free(ro);
	if (cs) free(cs);
	free(dlam_i); free(drho_i);
	if (dmu_i) free(dmu_i);
}
