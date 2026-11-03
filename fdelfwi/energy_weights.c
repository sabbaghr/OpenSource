#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"
#include"par.h"

/**
 * energy_weights.c -- Energy-norm data weighting W_r.
 *
 * Multi-component data mixes physical units: particle velocity is in m/s
 * while stresses are in Pa.  A plain L2 misfit over such a vector is
 * dimensionally inhomogeneous and is dominated by whichever component
 * happens to carry the larger numerical amplitude.
 *
 * The energy norm fixes this by measuring the residual in the natural
 * metric of the elastic wave equation,
 *
 *     W_r = diag( rho^(1/2) I , S^(1/2) ),    Q = W_r^T W_r = diag( rho I, S )
 *
 *     J = 0.5 * |W_r dd|^2 = 0.5 * dd^T Q dd          [J/m^3]
 *
 * with S the elastic compliance.  Each channel then contributes an energy
 * density: rho*dv^2 is kinetic, dsigma:S:dsigma is strain energy, and the
 * two are equipartitioned for a plane wave (dP = rho*Vp*dv).
 *
 * Velocity channels (Vx, Vz):
 *     Q_vv = rho
 *
 * Hydrophone channel (P = 0.5*(Txx+Tzz) = (lam+mu)*div u in 2D):
 *     Q_pp = 1/(lam+mu)
 *
 * Stress channels (Txx, Tzz, Txz).  Inverting the 2D plane-strain
 * stiffness [[lam+2mu, lam],[lam, lam+2mu]] (determinant D = 4*mu*(lam+mu))
 * gives the compliance, so that 0.5*sigma:S:sigma is the strain energy:
 *
 *     Q = | (lam+2mu)/D    -lam/D       0   |
 *         |   -lam/D     (lam+2mu)/D    0   |
 *         |     0            0        1/mu  |
 *
 * Note the off-diagonal -lam/D: Txx and Tzz are coupled, so their residuals
 * cannot be weighted independently.  The eigenbasis is volumetric/deviatoric,
 * with eigenvalue 1/(2*(lam+mu)) for (Txx+Tzz)/sqrt(2) and 1/(2*mu) for
 * (Txx-Tzz)/sqrt(2).  Substituting dTxx = dTzz = dP collapses the block to
 * dP^2/(lam+mu), recovering the hydrophone weight exactly -- which is the
 * basis of the seafloor degeneracy check (where Txz=0 and Txx=Tzz).
 *
 * The weights are sampled from the model at the receiver positions ONCE,
 * from the starting model, and then held fixed.  rho, Vp and Vs at the
 * receivers are themselves inversion parameters; recomputing Q every
 * iteration would make J a moving target (invalidating the Wolfe tests in
 * the linesearch) and would drop the dQ/dm term from the gradient.
 *
 * AUTHOR:
 *   fdelfwi -- elastic FWI
 */

/* Relative threshold below which a receiver is treated as being in a fluid.
 * In a fluid mu -> 0, so both 1/mu (Txz) and D = 4*mu*(lam+mu) are singular. */
#define EW_MU_REL_EPS 1.0e-6f


/***********************************************************************
 * energyWeightsBuild -- Sample rho, lambda, mu at each receiver.
 *
 * rec->x[]/rec->z[] are interior grid indices; the padded model arrays
 * are indexed (ix+ibndx)*naz + iz+ibndz, matching getRecTimes.c:50-51.
 ***********************************************************************/
energyWeight *energyWeightsBuild(modPar *mod, bndPar *bnd, recPar *rec, int verbose)
{
	energyWeight *ew;
	int irec, ibndx, ibndz, n1;
	int nfluid = 0;
	float rho_min = 0.0f, rho_max = 0.0f;
	float wv_min = 0.0f, wv_max = 0.0f, wp_min = 0.0f, wp_max = 0.0f;

	if (!mod || !bnd || !rec) verr("energyWeightsBuild: NULL argument.");
	if (rec->n < 1) verr("energyWeightsBuild: rec->n = %d, need at least 1 receiver.", rec->n);
	if (!mod->rho || !mod->cp)
		verr("energyWeightsBuild: model arrays not allocated (call readModel first).");

	ew = (energyWeight *)malloc(sizeof(energyWeight));
	if (!ew) verr("energyWeightsBuild: allocation failed.");

	ew->nrec = rec->n;
	ew->rho  = (float *)malloc(rec->n * sizeof(float));
	ew->lam  = (float *)malloc(rec->n * sizeof(float));
	ew->muu  = (float *)malloc(rec->n * sizeof(float));
	if (!ew->rho || !ew->lam || !ew->muu)
		verr("energyWeightsBuild: allocation failed for nrec=%d.", rec->n);

	ibndx = mod->ioPx;
	ibndz = mod->ioPz;
	if (bnd->lef == 4 || bnd->lef == 2) ibndx += bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibndz += bnd->ntap;
	n1 = mod->naz;

	for (irec = 0; irec < rec->n; irec++) {
		int ix = rec->x[irec] + ibndx;
		int iz = rec->z[irec] + ibndz;
		int ig = ix * n1 + iz;
		float vp, vs, rho_r, lam_r, mu_r;

		if (ix < 0 || ix >= mod->nax || iz < 0 || iz >= mod->naz)
			verr("energyWeightsBuild: receiver %d maps outside padded grid "
			     "(ix=%d/%d, iz=%d/%d).", irec, ix, mod->nax, iz, mod->naz);

		vp    = mod->cp[ig];
		vs    = (mod->ischeme > 2 && mod->cs) ? mod->cs[ig] : 0.0f;
		rho_r = mod->rho[ig];

		if (rho_r <= 0.0f)
			verr("energyWeightsBuild: non-positive density rho=%g at receiver %d.",
			     rho_r, irec);
		if (vp <= 0.0f)
			verr("energyWeightsBuild: non-positive Vp=%g at receiver %d.", vp, irec);
		if (vs < 0.0f)
			verr("energyWeightsBuild: negative Vs=%g at receiver %d.", vs, irec);
		if (vp * vp <= 2.0f * vs * vs - 1.0e-6f)
			vwarn("energyWeightsBuild: receiver %d has Vp=%g, Vs=%g "
			      "(Vp < sqrt(2)*Vs implies negative lambda).", irec, vp, vs);

		mu_r  = rho_r * vs * vs;
		lam_r = rho_r * (vp * vp - 2.0f * vs * vs);

		/* lam+mu = rho*(Vp^2 - Vs^2) is the 2D bulk modulus; it must be
		 * positive for the volumetric channel to be well posed. */
		if (lam_r + mu_r <= 0.0f)
			verr("energyWeightsBuild: non-positive bulk modulus lam+mu=%g at "
			     "receiver %d (Vp=%g, Vs=%g).", lam_r + mu_r, irec, vp, vs);

		ew->rho[irec] = rho_r;
		ew->lam[irec] = lam_r;
		ew->muu[irec] = mu_r;

		if (energyWeightIsFluid(mu_r, lam_r)) nfluid++;

		{
			float wv = sqrtf(rho_r);
			float wp = 1.0f / sqrtf(lam_r + mu_r);
			if (irec == 0) {
				rho_min = rho_max = rho_r;
				wv_min = wv_max = wv;
				wp_min = wp_max = wp;
			}
			else {
				if (rho_r < rho_min) rho_min = rho_r;
				if (rho_r > rho_max) rho_max = rho_r;
				if (wv < wv_min) wv_min = wv;
				if (wv > wv_max) wv_max = wv;
				if (wp < wp_min) wp_min = wp;
				if (wp > wp_max) wp_max = wp;
			}
		}
	}

	if (verbose) {
		vmess("Energy-norm data weights (W_r) from %d receivers:", ew->nrec);
		vmess("  rho       : %.4e .. %.4e kg/m^3", rho_min, rho_max);
		vmess("  sqrt(rho) : %.4e .. %.4e  (velocity channels)", wv_min, wv_max);
		vmess("  sqrt(S)   : %.4e .. %.4e  (volumetric stress channel)", wp_min, wp_max);
		if (nfluid > 0)
			vmess("  %d of %d receivers are in a fluid (mu ~ 0): deviatoric "
			      "stress channels unavailable there.", nfluid, ew->nrec);
	}

	return ew;
}


/***********************************************************************
 * energyWeightIsFluid -- True when mu is negligible against the bulk
 * modulus, i.e. the receiver sits in a fluid and the deviatoric stress
 * channels (1/mu) are singular.
 ***********************************************************************/
int energyWeightIsFluid(float mu_r, float lam_r)
{
	return (mu_r <= EW_MU_REL_EPS * (lam_r + mu_r));
}


/***********************************************************************
 * energyWeightApply -- Apply Q = W_r^T W_r to one receiver's residual.
 *
 * Maps the residual vector r[k] (one entry per component, all for the
 * SAME receiver irec) to psi[k] = (Q r)[k].  The caller obtains the
 * misfit contribution as 0.5 * sum_k r[k]*psi[k] = 0.5 * r^T Q r.
 *
 * chan[] holds the physical channel of each component, using the
 * readResidual typ convention (1=P, 2=Txz, 3=Tzz, 4=Txx, 6=Vx, 7=Vz).
 *
 * Diagonal for the velocity and hydrophone channels; the Txx/Tzz pair is
 * coupled through the -lam/D off-diagonal of the compliance.
 * idx_txx/idx_tzz are component indices, or -1 when absent.
 ***********************************************************************/
void energyWeightApply(int ncomp, const int *chan, const float *r, float *psi,
                       const energyWeight *ew, int irec,
                       int idx_txx, int idx_tzz)
{
	float rho_r = ew->rho[irec];
	float lam_r = ew->lam[irec];
	float mu_r  = ew->muu[irec];
	float bulk  = lam_r + mu_r;          /* 2D bulk modulus, = rho*(Vp^2-Vs^2) */
	int   fluid = energyWeightIsFluid(mu_r, lam_r);
	int   k;

	for (k = 0; k < ncomp; k++) {
		switch (chan[k]) {
		case 6:  /* Vx */
		case 7:  /* Vz */
			/* Kinetic energy density: Q_vv = rho */
			psi[k] = rho_r * r[k];
			break;

		case 1:  /* P = 0.5*(Txx+Tzz) -- volumetric strain energy */
			psi[k] = r[k] / bulk;
			break;

		case 2:  /* Txz -- pure deviatoric, Q = 1/mu */
			if (fluid)
				verr("energyWeightApply: receiver %d is in a fluid (mu=%g) but a "
				     "Txz component (_rtxz) was requested. The energy norm for "
				     "Txz is 1/mu, which is singular there. Drop _rtxz or move "
				     "the receivers into the solid.", irec, mu_r);
			psi[k] = r[k] / mu_r;
			break;

		case 3:  /* Tzz */
		case 4:  /* Txx */
			if (idx_txx >= 0 && idx_tzz >= 0) {
				/* Coupled pair -- handled once, in the Txx slot */
				if (chan[k] != 4) break;
				if (fluid) {
					/* mu -> 0: the deviatoric eigenvalue 1/(2mu) diverges and
					 * the pair carries only volumetric energy.  Project onto
					 * P = 0.5*(Txx+Tzz) so that dTxx=dTzz=dP reproduces the
					 * hydrophone weight dP^2/bulk exactly. */
					float p = 0.5f * (r[idx_txx] + r[idx_tzz]);
					psi[idx_txx] = 0.5f * p / bulk;
					psi[idx_tzz] = 0.5f * p / bulk;
				}
				else {
					float D   = 4.0f * mu_r * bulk;
					float l2m = lam_r + 2.0f * mu_r;
					psi[idx_txx] = (l2m * r[idx_txx] - lam_r * r[idx_tzz]) / D;
					psi[idx_tzz] = (l2m * r[idx_tzz] - lam_r * r[idx_txx]) / D;
				}
			}
			else {
				/* Only one of the pair supplied: the companion residual is
				 * taken as zero, so the correct quadratic form is the
				 * diagonal entry (lam+2mu)/D. */
				if (fluid)
					verr("energyWeightApply: receiver %d is in a fluid (mu=%g) but "
					     "a single normal-stress component was requested without "
					     "its companion. Use _rp, or supply both _rtxx and _rtzz.",
					     irec, mu_r);
				psi[k] = (lam_r + 2.0f * mu_r) * r[k] / (4.0f * mu_r * bulk);
			}
			break;

		default:
			verr("energyWeightApply: energy-norm weighting is not defined for "
			     "channel %d (component %d).", chan[k], k);
		}
	}
}


void energyWeightsFree(energyWeight *ew)
{
	if (!ew) return;
	if (ew->rho) free(ew->rho);
	if (ew->lam) free(ew->lam);
	if (ew->muu) free(ew->muu);
	free(ew);
}
