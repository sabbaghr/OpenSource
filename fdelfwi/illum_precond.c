#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"

void vmess(char *fmt, ...);
void verr(char *fmt, ...);

/**
 * illum_precond.c -- diagonal SOURCE-ILLUMINATION preconditioner
 *                    (illum_precond=1 in fwi_inversion).
 *
 * WHAT IT IS
 * ----------
 *     I(x) = sum_shots sum_t ( vx_f(x,t)^2 + vz_f(x,t)^2 )
 *
 * the forward particle-velocity energy, already accumulated by accumGradient
 * (fwi_gradient.c, the `wfld_energy` array that precond=1 also uses) and
 * shot/MPI-summed by compute_fwi_gradient.  No receiver-side term: this is
 * deliberately the source illumination only (precond=1 multiplies sqrt(I) by
 * the Plessix & Mulder asinh receiver-aperture term; this does not).
 *
 *     P(x) = nu / ( I(x) + beta ),     beta = illum_eps * max_{mask==1} I
 *
 * HOW IT IS USED
 * --------------
 * As the initial inverse Hessian H0 of PLBFGS (algorithm=2): it is applied to
 * q_plb between the two loops of the recursion (OPT_PREC), and to the first
 * gradient (grad_preco, which PLBFGS uses only at its first iteration).  The
 * (s,y) pairs stay the plain gradients, so rebuilding P every outer iteration
 * is legitimate -- unlike pre-dividing the gradient, which would mix metrics
 * inside the curvature pairs.
 *
 * WHY IT IS ALLOWED UNDER lag_tomo=1 (where precond=1..5 is refused)
 * -----------------------------------------------------------------
 * The Yang pseudo-Hessian is a zero-lag, Lame-space operator and does not
 * match the cone-filtered (Vp,Vs) gradient.  I(x) is not a kernel: it is a
 * positive diagonal weight that measures how much energy reached x, with no
 * lag axis and no parameterisation.  P is applied AFTER the cone filter, to
 * the already-filtered velocity gradient, so the filter is untouched.  (P F
 * is not symmetric, but neither is the lag_tomo "gradient" a gradient of J;
 * PLBFGS only needs P > 0, which it is.)
 *
 * beta
 * ----
 * The maximum is taken only where the source/receiver taper mask is 1 (the
 * same buildGradTaperMask the gradient taper uses), so the near-source
 * singularity -- where the gradient is zeroed anyway -- cannot set beta.
 * Otherwise beta would be fixed by one cell beside a source and P would be
 * flat everywhere else.
 *
 * nu (scale)
 * ----------
 * PLBFGS scales P q by gamma = s'y / y'y, so the absolute size of P sets the
 * step.  nu is fixed once per outer iteration from the GRADIENT,
 * nu = ||g|| / ||P0 g||, and then held for every q_plb of that iteration --
 * the same norm-preserving convention as the Yang path (getPrecondNu).  At
 * the first iteration this keeps alpha = 1/||g|| a sensible first step.
 *
 * Frozen parameters: P is applied to every parameter block, and the caller
 * zeroes frozen blocks and the water layer afterwards (applyParamMask /
 * applyWaterMask), exactly as for the gradient.
 */

/* Interior (nx*nz, ix*nz+iz) illumination from the padded wfld_energy. */
static void illumExtract(const float *wfld_energy, modPar *mod, bndPar *bnd,
                         float *I)
{
	int ix, iz, n1 = mod->naz;
	int ibx = mod->ioPx, ibz = mod->ioPz;
	if (bnd->lef == 4 || bnd->lef == 2) ibx += bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibz += bnd->ntap;

	for (ix = 0; ix < mod->nx; ix++)
		for (iz = 0; iz < mod->nz; iz++) {
			float e = wfld_energy[(size_t)(ix + ibx) * n1 + iz + ibz];
			I[ix * mod->nz + iz] = (e > 0.0f) ? e : 0.0f;
		}
}

/***********************************************************************
 * illumPrecondBuild -- P0 = 1/(I + beta) on the interior grid.
 *
 *   wfld_energy : padded (nax*naz), already summed over shots and ranks
 *   mask        : interior taper mask (nx*nz) or NULL (max over all cells)
 *   eps         : beta = eps * max_{mask>=1} I
 *   P           : out, nmodel = nx*nz
 *   I_out       : optional out, the interior illumination (for QC), or NULL
 ***********************************************************************/
void illumPrecondBuild(const float *wfld_energy, modPar *mod, bndPar *bnd,
                       const float *mask, float eps, float *P, float *I_out,
                       int verbose)
{
	int i, nmodel = mod->nx * mod->nz;
	float *I = I_out ? I_out : (float *)malloc((size_t)nmodel * sizeof(float));
	double imax = 0.0, imin = 1e300;
	int npass = 0;
	float beta;

	if (!I) verr("illumPrecondBuild: cannot allocate %d floats.", nmodel);
	illumExtract(wfld_energy, mod, bnd, I);

	for (i = 0; i < nmodel; i++) {
		if (mask && mask[i] < 0.999f) continue;
		npass++;
		if (I[i] > imax) imax = I[i];
		if (I[i] < imin) imin = I[i];
	}
	if (!(imax > 0.0))
		verr("illum_precond: the source illumination is zero everywhere "
		     "outside the taper mask -- wfld_energy was not accumulated "
		     "(is adj_shot being given the array?).");

	beta = (float)(eps * imax);
	for (i = 0; i < nmodel; i++)
		P[i] = 1.0f / (I[i] + beta);

	if (verbose >= 1)
		vmess("illum_precond: I max=%.4e min=%.4e (over %d unmasked cells, "
		      "range %.1f dB)  beta=%.4e (eps=%.2e)  P range %.1f dB",
		      imax, imin, npass,
		      10.0 * log10(imax / (imin > 0.0 ? imin : imax * 1e-30)),
		      beta, eps,
		      10.0 * log10((imax + beta) / ((imin > 0.0 ? imin : 0.0) + beta)));

	if (!I_out) free(I);
}

/***********************************************************************
 * illumPrecondNu -- norm-preserving scale nu = ||g|| / ||P0 g||.
 * Call once per outer iteration with the (scaled, masked) gradient.
 ***********************************************************************/
float illumPrecondNu(const float *P, const float *g, int nmodel, int nparam)
{
	int ip, i;
	double g2 = 0.0, pg2 = 0.0;

	for (ip = 0; ip < nparam; ip++)
		for (i = 0; i < nmodel; i++) {
			double gi = g[(size_t)ip * nmodel + i];
			g2  += gi * gi;
			pg2 += (P[i] * gi) * (P[i] * gi);
		}
	if (!(pg2 > 0.0)) return 1.0f;
	return (float)sqrt(g2 / pg2);
}

/***********************************************************************
 * illumPrecondApply -- out = nu * P0 * in, per parameter block.
 * in and out may alias.
 ***********************************************************************/
void illumPrecondApply(float *out, const float *in, const float *P,
                       float nu, int nmodel, int nparam)
{
	int ip, i;
	for (ip = 0; ip < nparam; ip++)
		for (i = 0; i < nmodel; i++) {
			size_t k = (size_t)ip * nmodel + i;
			out[k] = nu * P[i] * in[k];
		}
}
