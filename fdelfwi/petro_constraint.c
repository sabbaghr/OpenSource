#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "par.h"
#include "fdelfwi.h"

/*--------------------------------------------------------------------
 * petro_constraint.c -- Probabilistic petrophysical model constraint.
 *
 * Aragao, O., and P. Sava, 2020, Elastic full-waveform inversion with
 * probabilistic petrophysical model constraints: GEOPHYSICS 85(2),
 * R101-R111, doi:10.1190/geo2019-0285.1.  Their equations 16-20.
 *
 * The constraint confines the inverted parameter pair at every model
 * point to the region of the crossplot plane that is populated by a
 * petrophysical probability density f(p) -- in practice a crossplot of
 * well-log samples.  It is a *statistical* attraction to the whole
 * distribution, not a fit to any single trend, so it works when the
 * petrophysics is multi-modal or nonlinear (their Marmousi 2 example,
 * Figure 10) where the two-straight-line logarithmic barrier J_L of
 * Duan & Sava (2016) fails.
 *
 * For a model point m = [a1 a2] and a PDF cell p = [a1_p a2_p]:
 *
 *     d(m,p) = ||m - p||_2                                     (eq 16)
 *     1/D(m) = sum_p f(p)/d(m,p)                               (eq 17)
 *     J_P    = a * sum_x D(m)                                  (eq 18)
 *
 *     dJ_P/da1 = a D^2(m) sum_p f(p)(a1 - a1_p)/d^3(m,p)       (eq 19)
 *     dJ_P/da2 = a D^2(m) sum_p f(p)(a2 - a2_p)/d^3(m,p)       (eq 20)
 *
 * D(m) is a harmonic-mean distance: every PDF cell pulls on every model
 * point, weighted by its probability and inversely by its distance
 * (their Figure 2).  Inside the high-probability region D and its
 * gradient are SMALL, not zero: a harmonic mean over a distribution of
 * finite spread has a floor of order that spread, and the other clusters
 * of a multi-modal crossplot keep pulling.  On the three-cluster demo
 * crossplot min_m D = 0.72 GPa (~11 eps) and |g| at a true-rock point is
 * ~13% of its value at the initial model.  So J_P adds a constant floor to
 * the cost and a small residual bias even at the correct model; 'a' sets
 * how much.  Far outside, the J_P gradient dominates and drags the point
 * back.
 *
 * CROSSPLOT SPACE.  The paper works exclusively in (lambda, mu), which is
 * where the elastic radiation patterns make the crosstalk, and that is
 * PETRO_SPACE_LAMMU here.  But nothing in equations 16-20 is specific to
 * that plane, and classical rock-physics trends (Castagna's mudrock line,
 * Gardner) are quoted in velocity, so PETRO_SPACE_VPVS puts the same
 * machinery in the (Vp, Vs) plane.  Choosing the crossplot space to match
 * the inversion parameterization makes the mapping an identity, so the
 * penalty acts directly on the inverted variables and its notion of
 * distance is the one the optimizer actually sees.  The four cases:
 *
 *                 param=1 (lam,mu,rho)      param=2 (Vp,Vs,rho)
 *   lam-mu        identity                  forward chain rule
 *   Vp-Vs         inverse chain rule        identity
 *
 * In both off-diagonal cases a DENSITY gradient appears even though J_P
 * has no explicit rho dependence, because each pair carries rho through
 * lambda = rho(Vp^2-2Vs^2), mu = rho Vs^2.  Dropping it would make J_P
 * and its gradient inconsistent and break the linesearch.
 *
 * UNITS.  d(m,p) is a *distance in the crossplot plane*, so it mixes the
 * two axes and the axis unit is part of the physics, not cosmetics: it
 * sets the relative weight of the two parameters in the penalty and ties
 * the numerical value of 'a' to that choice.  The units here are the
 * ones that put both axes on a comparable O(1-10) scale, which is also
 * what the papers plot -- GPa for (lambda, mu), km/s for (Vp, Vs).  All
 * conversion to the per-Pa / per-(m/s) units of the FWI gradient happens
 * in petroAccumGradient.
 *
 * COST.  O(nmodel * ncell).  The PDF is stored as a sparse list of
 * non-empty cells, which for a real well-log crossplot is a few hundred
 * out of nlam*nmu, so this is a small fraction of one forward model.
 *
 * NOT IMPLEMENTED (deliberate):
 *  - J_L, the logarithmic barrier of their equations 10-15.  The paper
 *    shows it distorts the background and does not recover the
 *    anomalies (their Figures 6, 8e and 14); J_P supersedes it.
 *  - Their empirical requirement that the PDF be re-estimated as the
 *    inversion proceeds.  f(p) is prior information and is held fixed,
 *    exactly as the paper does.
 *--------------------------------------------------------------------*/

#define PETRO_MAXLINE 512

/* Cells below this fraction of the peak probability are dropped from the
 * sparse list.  They contribute O(1e-4) to 1/D while costing the same as
 * every other cell, and the list is renormalised afterwards. */
#define PETRO_FTHRESH 1.0e-4f


/*--------------------------------------------------------------------
 * petroSmoothHist -- separable Gaussian smoothing of the 2D histogram.
 *
 * A raw histogram of a few thousand log samples is spiky, and a spiky
 * f(p) makes the J_P gradient spiky in the same way.  Smoothing turns
 * the sample cloud into the smooth density the method assumes (the
 * contours of their Figures 7 and 15-17).
 *
 * sigma is in bins.  Kernel is truncated at 3 sigma.
 *--------------------------------------------------------------------*/
static void petroSmoothHist(float *h, int nlam, int nmu, float sigma)
{
	int    half, k, i, j;
	float *kern, *tmp;
	float  sum = 0.0f;

	if (sigma <= 0.0f) return;

	half = (int)(3.0f * sigma + 0.5f);
	if (half < 1) return;

	kern = (float *)malloc((size_t)(2*half+1) * sizeof(float));
	tmp  = (float *)calloc((size_t)nlam * nmu, sizeof(float));

	for (k = -half; k <= half; k++) {
		kern[k+half] = expf(-0.5f * (float)(k*k) / (sigma*sigma));
		sum += kern[k+half];
	}
	for (k = 0; k <= 2*half; k++) kern[k] /= sum;

	/* along mu (fast axis) */
	for (i = 0; i < nlam; i++) {
		for (j = 0; j < nmu; j++) {
			float acc = 0.0f;
			for (k = -half; k <= half; k++) {
				int jj = j + k;
				if (jj < 0 || jj >= nmu) continue;
				acc += kern[k+half] * h[i*nmu + jj];
			}
			tmp[i*nmu + j] = acc;
		}
	}

	/* along lambda (slow axis) */
	for (i = 0; i < nlam; i++) {
		for (j = 0; j < nmu; j++) {
			float acc = 0.0f;
			for (k = -half; k <= half; k++) {
				int ii = i + k;
				if (ii < 0 || ii >= nlam) continue;
				acc += kern[k+half] * tmp[ii*nmu + j];
			}
			h[i*nmu + j] = acc;
		}
	}

	free(tmp);
	free(kern);
}


/*--------------------------------------------------------------------
 * petroBuild -- build the sparse PDF from an ASCII sample file.
 *
 * file   - text file, one sample pair per line.  Blank lines and lines
 *          starting with '#' are ignored.  This is the raw well-log
 *          crossplot (their Figures 1a, 10).  Units follow 'space':
 *          GPa for PETRO_SPACE_LAMMU, km/s for PETRO_SPACE_VPVS.
 * space   - PETRO_SPACE_LAMMU or PETRO_SPACE_VPVS
 * nlam   - histogram bins along axis 1 (lambda or Vp)
 * nmu    - histogram bins along axis 2 (mu or Vs)
 * smooth - Gaussian smoothing of the histogram, in bins (0 = none)
 * eps    - softening length on d(m,p), in the crossplot unit, or <=0 for
 *          the automatic value.  The paper "explicitly excludes
 *          situations where the model space distance from m to a given p
 *          is equal to zero"; eps does that continuously instead of by
 *          rejection, which keeps J_P differentiable.
 *
 *          It enters as a SOFTENING, not a floor:
 *
 *              d(m,p) = sqrt( (a1-a1p)^2 + (a2-a2p)^2 + eps^2 )
 *
 *          not as d = max(||m-p||, eps).  The two agree far from every
 *          cell and differ by O(eps) near one, but only the softened
 *          form is differentiable: with a max() clamp, d is frozen at
 *          eps inside the clamp radius, so the cost stops changing while
 *          equations 19/20 keep returning (a_k - a_k,p)/eps^3 -- a
 *          gradient the cost does not have.  That is invisible in the far
 *          field (where nothing is clamped) and invisible to a sign or
 *          monotonicity test, but it corrupts exactly the near-cloud
 *          cells the constraint is supposed to act on.  The softened form
 *          has derivative -(a_k - a_k,p)/d^3 exactly, so the same inner
 *          loop serves both the cost and the gradient and (J_P, g_P)
 *          stays a consistent pair.  test_petro check 1 catches this:
 *          with the clamp, its Taylor residual converged to 0.59 of the
 *          predicted slope instead of 1.
 *
 *          The automatic value is half the bin diagonal.  A softening
 *          below the bin size is pointless: f(p) is a histogram, so it
 *          carries no information at sub-bin separations, yet 1/d^3 in
 *          equations 19/20 would still produce gradient spikes on that
 *          scale whenever a model point drifts onto a cell centre.
 *
 * Returns 0 on success, non-zero on failure.
 *
 * Does NOT touch pp->a, pp->balance or pp->calibrated: those are the
 * caller's trade-off policy, not part of the PDF, and the caller is free
 * to set them either side of this call.
 *--------------------------------------------------------------------*/
int petroBuild(const char *file, int space, int nlam, int nmu, float smooth,
               float eps, petroPar *pp, int verbose)
{
	const char *an1 = (space == PETRO_SPACE_VPVS) ? "Vp" : "lambda";
	const char *an2 = (space == PETRO_SPACE_VPVS) ? "Vs" : "mu";
	const char *aun = (space == PETRO_SPACE_VPVS) ? "km/s" : "GPa";
	FILE  *fp;
	char   line[PETRO_MAXLINE];
	float *slam = NULL, *smuu = NULL, *hist = NULL;
	int    nsamp = 0, ncap = 0;
	int    i, j, n;
	float  lmin, lmax, mmin, mmax, dlam, dmu, fmax, fsum;
	float  lpad, mpad;

	/* Zero only the fields petroBuild OWNS -- deliberately not a blanket
	 * memset(pp,0,sizeof *pp).  'a', 'balance' and 'calibrated' are the
	 * CALLER's policy, parsed from the command line before we are called,
	 * and a blanket memset silently wipes them: petro_balance became 0,
	 * so 'calibrated' was set to 1, auto-calibration never ran, 'a' stayed
	 * 0 and the constraint was inert while still reporting a perfectly
	 * healthy sum_x D.  Nothing errored, because the mutual-exclusion
	 * check in the driver runs BEFORE this call.
	 * The point of the zeroing is only that c1/c2/f must be NULL if we
	 * bail out below, so petroFree stays safe. */
	pp->on      = 0;
	pp->space   = 0;
	pp->ncell   = 0;
	pp->c1      = NULL;
	pp->c2      = NULL;
	pp->f       = NULL;
	pp->eps     = 0.0f;
	pp->nsample = 0;
	pp->c1_min  = pp->c1_max = 0.0f;
	pp->c2_min  = pp->c2_max = 0.0f;

	fp = fopen(file, "r");
	if (fp == NULL) {
		fprintf(stderr, "petroBuild: cannot open petro_file=%s\n", file);
		return 1;
	}

	/* --- read the samples --- */
	while (fgets(line, sizeof(line), fp) != NULL) {
		float l, m;
		char *s = line;
		while (*s == ' ' || *s == '\t') s++;
		if (*s == '#' || *s == '\n' || *s == '\0') continue;
		if (sscanf(s, "%f %f", &l, &m) != 2) continue;

		if (nsamp == ncap) {
			ncap = ncap ? 2*ncap : 1024;
			slam = (float *)realloc(slam, (size_t)ncap * sizeof(float));
			smuu = (float *)realloc(smuu, (size_t)ncap * sizeof(float));
		}
		slam[nsamp] = l;
		smuu[nsamp] = m;
		nsamp++;
	}
	fclose(fp);

	if (nsamp < 2) {
		fprintf(stderr, "petroBuild: %s has %d usable (%s,%s) samples, "
		        "need at least 2\n", file, nsamp, an1, an2);
		free(slam); free(smuu);
		return 1;
	}

	/* --- histogram axes, padded by one bin so edge samples are interior --- */
	lmin = lmax = slam[0];
	mmin = mmax = smuu[0];
	for (n = 1; n < nsamp; n++) {
		if (slam[n] < lmin) lmin = slam[n];
		if (slam[n] > lmax) lmax = slam[n];
		if (smuu[n] < mmin) mmin = smuu[n];
		if (smuu[n] > mmax) mmax = smuu[n];
	}
	lpad = (lmax - lmin) / (float)nlam;
	mpad = (mmax - mmin) / (float)nmu;
	if (lpad <= 0.0f) lpad = 1.0f;   /* degenerate: all samples share a lambda */
	if (mpad <= 0.0f) mpad = 1.0f;
	lmin -= lpad; lmax += lpad;
	mmin -= mpad; mmax += mpad;

	dlam = (lmax - lmin) / (float)nlam;
	dmu  = (mmax - mmin) / (float)nmu;

	/* --- bin --- */
	hist = (float *)calloc((size_t)nlam * nmu, sizeof(float));
	for (n = 0; n < nsamp; n++) {
		i = (int)((slam[n] - lmin) / dlam);
		j = (int)((smuu[n] - mmin) / dmu);
		if (i < 0) i = 0;
		if (i >= nlam) i = nlam - 1;
		if (j < 0) j = 0;
		if (j >= nmu) j = nmu - 1;
		hist[i*nmu + j] += 1.0f;
	}

	petroSmoothHist(hist, nlam, nmu, smooth);

	/* --- compress to the sparse list of significant cells --- */
	fmax = 0.0f;
	for (i = 0; i < nlam*nmu; i++) if (hist[i] > fmax) fmax = hist[i];

	pp->ncell = 0;
	for (i = 0; i < nlam*nmu; i++) if (hist[i] > PETRO_FTHRESH * fmax) pp->ncell++;

	pp->c1 = (float *)malloc((size_t)pp->ncell * sizeof(float));
	pp->c2 = (float *)malloc((size_t)pp->ncell * sizeof(float));
	pp->f   = (float *)malloc((size_t)pp->ncell * sizeof(float));

	n = 0;
	fsum = 0.0f;
	for (i = 0; i < nlam; i++) {
		for (j = 0; j < nmu; j++) {
			float v = hist[i*nmu + j];
			if (v <= PETRO_FTHRESH * fmax) continue;
			pp->c1[n] = lmin + ((float)i + 0.5f) * dlam;  /* cell centre */
			pp->c2[n] = mmin + ((float)j + 0.5f) * dmu;
			pp->f[n]   = v;
			fsum += v;
			n++;
		}
	}
	for (n = 0; n < pp->ncell; n++) pp->f[n] /= fsum;   /* sum_p f(p) = 1 */

	/* Default the singularity floor to half a bin diagonal (see above) */
	pp->eps     = (eps > 0.0f) ? eps
	                           : 0.5f * sqrtf(dlam*dlam + dmu*dmu);
	pp->nsample = nsamp;
	pp->c1_min = lmin;  pp->c1_max = lmax;
	pp->c2_min = mmin;  pp->c2_max = mmax;
	pp->space   = space;
	pp->on      = 1;

	free(hist);
	free(slam);
	free(smuu);

	if (verbose) {
		vmess("Petrophysical constraint (Aragao & Sava 2020, eq 16-20):");
		vmess("  file=%s  samples=%d  crossplot=(%s,%s) in %s",
		      file, pp->nsample, an1, an2, aun);
		vmess("  histogram %dx%d -> %d active cells (%.1f%% fill), smooth=%.1f bins",
		      nlam, nmu, pp->ncell,
		      100.0f*(float)pp->ncell/(float)(nlam*nmu), smooth);
		vmess("  %-6s range [%.3f, %.3f] %s", an1, lmin, lmax, aun);
		vmess("  %-6s range [%.3f, %.3f] %s", an2, mmin, mmax, aun);
		vmess("  eps=%.3e %s%s", pp->eps, aun,
		      (eps > 0.0f) ? "" : " (auto: half a bin diagonal)");
	}

	return 0;
}


/*--------------------------------------------------------------------
 * petroCostGrad -- J_P / a and its gradient, for the whole model.
 *
 * xlame  - [3*nmodel] flat model as produced by extractModelVector with
 *          param=1: [lambda | mu | rho], in Pa.  Always Lame-in, whatever
 *          the crossplot space: the mapping to the crossplot axes is done
 *          here, per point, and costs two sqrt against an inner loop of
 *          ncell, so it is not worth caching.
 * wmask  - [nmodel] water mask, 1 = water cell, or NULL.  Water has
 *          mu == 0 and sits nowhere near a rock crossplot, so those
 *          cells are skipped entirely rather than dragged onto the
 *          distribution.
 * g1     - [nmodel] OUT: d(sum_x D)/d a1, per GPa or per km/s
 * g2     - [nmodel] OUT: d(sum_x D)/d a2, per GPa or per km/s
 *
 * Returns sum_x D(m), in the crossplot axis unit.  The gradients come
 * back in crossplot units; petroAccumGradient converts them and applies
 * the parameterization mapping.  The caller multiplies both the returned
 * cost and the gradients by 'a'; keeping 'a' out of here is what lets the
 * caller calibrate it against ||g_data|| (petro_balance).
 *--------------------------------------------------------------------*/
double petroCostGrad(const float *xlame, int nmodel, const petroPar *pp,
                     const int *wmask, float *g1, float *g2)
{
	const float *lam_pa = xlame;
	const float *muu_pa = xlame + nmodel;
	const float *rho_in = xlame + 2*nmodel;
	int          vpvs   = (pp->space == PETRO_SPACE_VPVS);
	double       dsum   = 0.0;
	int          i;

	memset(g1, 0, (size_t)nmodel * sizeof(float));
	memset(g2, 0, (size_t)nmodel * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel for reduction(+:dsum) schedule(static)
#endif
	for (i = 0; i < nmodel; i++) {
		float  a1, a2, D;
		double s = 0.0, s1 = 0.0, s2 = 0.0;
		int    p;

		if (wmask && wmask[i]) continue;

		if (vpvs) {
			float r = rho_in[i];
			a1 = sqrtf((lam_pa[i] + 2.0f*muu_pa[i]) / r) * 1.0e-3f;  /* Vp, km/s */
			a2 = sqrtf(muu_pa[i] / r) * 1.0e-3f;                     /* Vs, km/s */
		} else {
			a1 = lam_pa[i] * 1.0e-9f;    /* lambda, GPa */
			a2 = muu_pa[i] * 1.0e-9f;    /* mu,     GPa */
		}

		for (p = 0; p < pp->ncell; p++) {
			float d1 = a1 - pp->c1[p];
			float d2 = a2 - pp->c2[p];
			float fp = pp->f[p];
			float d, id3;

			/* eq 16, with the singularity softened rather than clamped:
			 * d = sqrt(dist^2 + eps^2) has derivative -d_k/d^3 exactly,
			 * so the s1/s2 sums below are the true gradient of s. */
			d = sqrtf(d1*d1 + d2*d2 + pp->eps*pp->eps);

			s  += (double)(fp / d);            /* eq 17: 1/D = sum f/d */
			id3 = fp / (d*d*d);
			s1 += (double)(id3 * d1);          /* eq 19 inner sum */
			s2 += (double)(id3 * d2);          /* eq 20 inner sum */
		}

		if (s <= 0.0) continue;                /* cannot happen: f>0, d finite */

		D = (float)(1.0 / s);
		dsum += (double)D;

		g1[i] = (float)(D * D * s1);           /* eq 19 */
		g2[i] = (float)(D * D * s2);           /* eq 20 */
	}

	return dsum;
}


/*--------------------------------------------------------------------
 * petroAccumGradient -- add a*g_petro into the optimizer gradient.
 *
 * g1/g2 come out of petroCostGrad in crossplot units (per GPa or per
 * km/s) with respect to the crossplot axes.  The optimizer vector is in
 * whatever 'param' selects, so there are four cases.
 *
 * MATCHED (crossplot space == inversion parameterization): identity, and
 * the density block gets nothing, because J_P genuinely does not depend
 * on rho once the other two are fixed.
 *
 * lam-mu crossplot, param=2.  Same chain rule as
 * convertGradientToVelocity (fwi_gradient.c):
 *
 *     lambda = rho(Vp^2 - 2Vs^2),  mu = rho Vs^2
 *     g_Vp  = g_lam * 2 rho Vp
 *     g_Vs  = 2 rho Vs (g_mu - 2 g_lam)
 *     g_rho = g_lam (Vp^2 - 2Vs^2) + g_mu Vs^2
 *
 * Vp-Vs crossplot, param=1.  The inverse map, from
 * Vp = sqrt((lambda+2mu)/rho), Vs = sqrt(mu/rho):
 *
 *     g_lam = g_Vp/(2 rho Vp)
 *     g_mu  = g_Vp/(rho Vp) + g_Vs/(2 rho Vs)
 *     g_rho = -(g_Vp Vp + g_Vs Vs)/(2 rho)
 *
 * In BOTH off-diagonal cases a density gradient appears even though J_P
 * has no explicit rho dependence, because the pair being constrained
 * carries rho.  Dropping that term would make J_P and its gradient
 * inconsistent and break the linesearch.  test_petro check 3 is there to
 * catch exactly that, in both directions.
 *
 * g      - [nparam*nmodel] optimizer gradient, accumulated in place
 * xlame  - [3*nmodel] model in Pa, [lambda | mu | rho]
 * a      - trade-off weight
 *--------------------------------------------------------------------*/
void petroAccumGradient(float *g, int nmodel, int param, const float *xlame,
                        const float *g1, const float *g2, float a,
                        const petroPar *pp)
{
	const float *lam_pa = xlame;
	const float *muu_pa = xlame + nmodel;
	const float *rho    = xlame + 2*nmodel;
	int i;

	if (pp->space == PETRO_SPACE_VPVS) {
		/* g1,g2 are per km/s -> per m/s */
		if (param == 2) {
			for (i = 0; i < nmodel; i++) {
				g[i]          += a * g1[i] * 1.0e-3f;
				g[nmodel + i] += a * g2[i] * 1.0e-3f;
			}
			return;
		}
		for (i = 0; i < nmodel; i++) {
			float g_vp = a * g1[i] * 1.0e-3f;
			float g_vs = a * g2[i] * 1.0e-3f;
			float r    = rho[i];
			float vp   = sqrtf((lam_pa[i] + 2.0f*muu_pa[i]) / r);
			float vs   = sqrtf(muu_pa[i] / r);

			/* vs==0 is water; those cells are masked out upstream, but
			 * guard anyway so a stray fluid cell cannot produce a NaN
			 * that would silently poison the whole gradient. */
			if (vp <= 0.0f || vs <= 0.0f) continue;

			g[i]            += g_vp / (2.0f * r * vp);
			g[nmodel + i]   += g_vp / (r * vp) + g_vs / (2.0f * r * vs);
			g[2*nmodel + i] += -(g_vp * vp + g_vs * vs) / (2.0f * r);
		}
		return;
	}

	/* lam-mu crossplot: g1,g2 are per GPa -> per Pa */
	if (param == 1) {
		for (i = 0; i < nmodel; i++) {
			g[i]          += a * g1[i] * 1.0e-9f;
			g[nmodel + i] += a * g2[i] * 1.0e-9f;
		}
		return;
	}

	for (i = 0; i < nmodel; i++) {
		float g_lam = a * g1[i] * 1.0e-9f;
		float g_muu = a * g2[i] * 1.0e-9f;
		float r     = rho[i];
		float vs2   = muu_pa[i] / r;
		float vp2   = (lam_pa[i] + 2.0f*muu_pa[i]) / r;
		float vp    = sqrtf(vp2);
		float vs    = sqrtf(vs2);

		g[i]            += g_lam * 2.0f * r * vp;
		g[nmodel + i]   += 2.0f * r * vs * (g_muu - 2.0f * g_lam);
		g[2*nmodel + i] += g_lam * (vp2 - 2.0f*vs2) + g_muu * vs2;
	}
}


/*--------------------------------------------------------------------
 * petroHessVec -- add a * H_P d into hd, H_P the EXACT Hessian of J_P/a.
 *
 * Needed by the truncated-Newton family (algorithm 4-7): the gradient
 * handed to the optimizer contains g_P, so the Hessian-vector product the
 * inner CG sees must contain H_P d, or the Newton system it solves is not
 * the Hessian of the objective whose gradient it was given.
 *
 * J_P is a sum of independent per-cell terms, so H_P is block diagonal,
 * one 3x3 block per model point, and the product is O(nmodel*ncell) --
 * the same cost as one petroCostGrad.
 *
 * Crossplot-plane Hessian.  With r = a - p, s = sum_p f/d, D = 1/s and
 * d = sqrt(|r|^2 + eps^2) (the softened eq 16):
 *
 *     grad s     = -G,               G = sum_p f r / d^3
 *     hess s     =  S,               S = sum_p f (3 r r^T / d^5 - I / d^3)
 *     grad D     =  D^2 G                                      (eq 19/20)
 *     hess D     =  2 D^3 G G^T - D^2 S
 *
 * hess D is NOT positive definite in general -- the harmonic mean has a
 * saddle between every two clusters of a multi-modal crossplot -- and it is
 * kept exact on purpose.  The TRN/PTRN inner CG already terminates on
 * negative curvature (optimization/trn.c, ptrn.c, enriched.c), which is
 * the correct response to an indefinite Hessian; a PSD surrogate would
 * just hide the curvature from it.
 *
 * Parameterization.  With a(m) the map from the inversion variables m to
 * the crossplot axes (J = da/dm, 2x3; T_k = d^2 a_k/dm^2, 3x3):
 *
 *     H_m = J^T (hess D) J + sum_k (grad D)_k T_k
 *
 * In the two MATCHED cases (lam-mu with param=1, Vp-Vs with param=2) a is
 * linear in m, T_k = 0, and the rho row/column is zero.  In the two mixed
 * cases the second term is essential: dropping it gives a Hessian that is
 * not the derivative of the gradient petroAccumGradient produces, and
 * test_petro check 4 fails.  J and T include the GPa / km/s unit factor,
 * so d and hd are in the physical units of 'param', exactly as
 * compute_hessian_vector_product sees them.
 *
 * hd, d  - [3*nmodel] in the inversion parameterization (param 1 or 2)
 * xlame  - [3*nmodel] model in Pa, [lambda | mu | rho]
 * wmask  - water mask (skipped), or NULL
 *--------------------------------------------------------------------*/
void petroHessVec(float *hd, const float *d, int nmodel, int param,
                  const float *xlame, const petroPar *pp, const int *wmask,
                  float a)
{
	const float *lam_pa = xlame;
	const float *muu_pa = xlame + nmodel;
	const float *rho_in = xlame + 2*nmodel;
	int vpvs  = (pp->space == PETRO_SPACE_VPVS);
	int i;

	if (!pp->on || a == 0.0f) return;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (i = 0; i < nmodel; i++) {
		double lam = lam_pa[i], mu = muu_pa[i], r = rho_in[i];
		double vp, vs, a1, a2, c;
		double J[2][3], T[2][3][3];
		double s = 0.0, G1 = 0.0, G2 = 0.0, S11 = 0.0, S12 = 0.0, S22 = 0.0;
		double D, D2, g1, g2, H11, H12, H22, Jd1, Jd2, dv[3], out[3];
		int    p, k, l;

		if (wmask && wmask[i]) continue;
		if (r <= 0.0 || mu <= 0.0) continue;    /* fluid / unphysical */

		vp = sqrt((lam + 2.0*mu) / r);
		vs = sqrt(mu / r);
		memset(J, 0, sizeof(J));
		memset(T, 0, sizeof(T));

		if (vpvs) {
			c  = 1.0e-3;                         /* m/s -> km/s */
			a1 = c * vp;
			a2 = c * vs;
			if (param == 2) {                    /* identity */
				J[0][0] = c;
				J[1][1] = c;
			} else {                             /* m = (lam, mu, rho) */
				double u = lam + 2.0*mu;
				J[0][0] = c / (2.0*r*vp);
				J[0][1] = c / (r*vp);
				J[0][2] = -c * vp / (2.0*r);
				J[1][1] = c / (2.0*r*vs);
				J[1][2] = -c * vs / (2.0*r);
				/* Vp = u^1/2 rho^-1/2, u = lam + 2 mu */
				T[0][0][0] = -c*vp / (4.0*u*u);
				T[0][0][1] = -c*vp / (2.0*u*u);
				T[0][1][1] = -c*vp / (u*u);
				T[0][0][2] = -c*vp / (4.0*u*r);
				T[0][1][2] = -c*vp / (2.0*u*r);
				T[0][2][2] =  3.0*c*vp / (4.0*r*r);
				/* Vs = mu^1/2 rho^-1/2 */
				T[1][1][1] = -c*vs / (4.0*mu*mu);
				T[1][1][2] = -c*vs / (4.0*mu*r);
				T[1][2][2] =  3.0*c*vs / (4.0*r*r);
			}
		} else {
			c  = 1.0e-9;                         /* Pa -> GPa */
			a1 = c * lam;
			a2 = c * mu;
			if (param == 1) {                    /* identity */
				J[0][0] = c;
				J[1][1] = c;
			} else {                             /* m = (Vp, Vs, rho) */
				/* lam = rho (Vp^2 - 2 Vs^2), mu = rho Vs^2 */
				J[0][0] =  c * 2.0*r*vp;
				J[0][1] = -c * 4.0*r*vs;
				J[0][2] =  c * (vp*vp - 2.0*vs*vs);
				J[1][1] =  c * 2.0*r*vs;
				J[1][2] =  c * vs*vs;
				T[0][0][0] =  c * 2.0*r;
				T[0][0][2] =  c * 2.0*vp;
				T[0][1][1] = -c * 4.0*r;
				T[0][1][2] = -c * 4.0*vs;
				T[1][1][1] =  c * 2.0*r;
				T[1][1][2] =  c * 2.0*vs;
			}
		}
		/* symmetrise the upper-triangle fills */
		for (k = 0; k < 2; k++)
			for (l = 0; l < 3; l++) {
				int m;
				for (m = l+1; m < 3; m++) T[k][m][l] = T[k][l][m];
			}

		for (p = 0; p < pp->ncell; p++) {
			double r1 = a1 - pp->c1[p];
			double r2 = a2 - pp->c2[p];
			double fp = pp->f[p];
			double dd = sqrt(r1*r1 + r2*r2 + (double)pp->eps*pp->eps);
			double id3 = fp / (dd*dd*dd);
			double id5 = 3.0 * id3 / (dd*dd);
			s   += fp / dd;
			G1  += id3 * r1;
			G2  += id3 * r2;
			S11 += id5 * r1*r1 - id3;
			S12 += id5 * r1*r2;
			S22 += id5 * r2*r2 - id3;
		}
		if (s <= 0.0) continue;

		D   = 1.0 / s;
		D2  = D * D;
		g1  = D2 * G1;
		g2  = D2 * G2;
		H11 = 2.0*D*D2*G1*G1 - D2*S11;
		H12 = 2.0*D*D2*G1*G2 - D2*S12;
		H22 = 2.0*D*D2*G2*G2 - D2*S22;

		dv[0] = d[i];
		dv[1] = d[nmodel + i];
		dv[2] = d[2*nmodel + i];

		Jd1 = J[0][0]*dv[0] + J[0][1]*dv[1] + J[0][2]*dv[2];
		Jd2 = J[1][0]*dv[0] + J[1][1]*dv[1] + J[1][2]*dv[2];
		{
			double h1 = H11*Jd1 + H12*Jd2;
			double h2 = H12*Jd1 + H22*Jd2;
			for (l = 0; l < 3; l++) {
				out[l] = J[0][l]*h1 + J[1][l]*h2
				       + g1 * (T[0][l][0]*dv[0] + T[0][l][1]*dv[1] + T[0][l][2]*dv[2])
				       + g2 * (T[1][l][0]*dv[0] + T[1][l][1]*dv[1] + T[1][l][2]*dv[2]);
			}
		}

		hd[i]            += (float)(a * out[0]);
		hd[nmodel + i]   += (float)(a * out[1]);
		hd[2*nmodel + i] += (float)(a * out[2]);
	}
}


/*--------------------------------------------------------------------
 * petroFree -- release the sparse PDF.
 *--------------------------------------------------------------------*/
void petroFree(petroPar *pp)
{
	if (pp == NULL) return;
	free(pp->c1); pp->c1 = NULL;
	free(pp->c2); pp->c2 = NULL;
	free(pp->f);  pp->f  = NULL;
	pp->ncell = 0;
	pp->on    = 0;
}
