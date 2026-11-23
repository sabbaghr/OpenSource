#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "par.h"
#include "fdelfwi.h"

/*--------------------------------------------------------------------
 * test_petro.c -- standalone verification of the probabilistic
 * petrophysical constraint J_P (Aragao & Sava 2020, eq 16-20).
 *
 * No wave equation is involved: J_P is a pure model-domain functional,
 * so its cost and gradient can be checked exactly and in a fraction of
 * a second.  This is worth doing separately because inside the
 * inversion J_P is buried under scaling, masking and a chain rule, and
 * a sign or factor error there would look like a linesearch failure.
 *
 * Everything is driven through the same three entry points the
 * inversion uses, with the same crossplot space, so a mistake in the
 * space-dependent mapping cannot hide:
 *
 *   petroBuild(file, space, ...)   ->  sparse PDF on the chosen axes
 *   petroCostGrad(xlame, ...)      ->  sum_x D and dD/d(a1,a2)
 *   petroAccumGradient(..., param) ->  mapped into the optimizer vector
 *
 * Three checks, each returning non-zero on failure so the program is
 * usable as a regression test:
 *
 *  1. TAYLOR.  J_P and g_P are a CONSISTENT pair -- unlike grad_taper,
 *     smooth_grad or energy_ic, which modify the gradient without a
 *     matching cost.  So
 *         J(m + t dm) - J(m) - t <g,dm>  =  O(t^2)
 *     and the residual must fall by 4x per halving of t.  This is the
 *     check that catches a wrong power of d in equations 19/20.  It is
 *     run in LAME space (param=1) and perturbs all three parameters,
 *     which means that for a Vp-Vs crossplot it also exercises the
 *     inverse chain rule and its rho term.
 *
 *     NOTE on petro_eps.  The Taylor ratios are only clean if the
 *     perturbation is large compared with the eps floor on d(m,p).  An
 *     eps far below the histogram bin size (the old 1e-3 GPa default
 *     against a 0.067 GPa bin) lets 1/d^3 vary on a length scale the
 *     histogram cannot resolve, and the ratios wander between 2.5 and
 *     5.6.  petroBuild now defaults eps to half a bin diagonal; this
 *     check is what keeps that honest.
 *
 *  2. MINIMUM.  D(m) must grow monotonically as m is moved away from
 *     the PDF cloud and the gradient must point back TOWARD it (the
 *     paper's Figure 1c).  This is the check that catches a sign error,
 *     which Taylor alone would not: flipping the sign of both J and g
 *     keeps the Taylor test passing while turning the constraint into a
 *     repulsion.
 *
 *     Both the probe DIRECTION and its DISTANCE are derived from f(p)
 *     rather than hardcoded, because no fixed choice works for every
 *     crossplot:
 *
 *     - DIRECTION: the MINOR principal axis of f(p).  A fixed walk along
 *       the +a1/+a2 diagonal is wrong whenever the clusters are nearly
 *       collinear.  For the three-layer model in
 *       demo/inversion/test_petro_inversion.sh the trend runs at about
 *       30 degrees in the Vp-Vs plane, so a 45-degree walk moves ALONG
 *       the trend and straight at the far cluster -- and the gradient
 *       quite correctly does not push back, because by eq 17 the nearest
 *       rock is then ahead of the probe, not behind it.  That is the
 *       harmonic mean doing its job, and scoring it as a sign error says
 *       more about the cluster geometry than about the code.
 *
 *       The minor axis is the direction in which the PDF falls off
 *       fastest, so "away from the centroid" and "away from the cloud"
 *       coincide.  It is also the direction the constraint exists to
 *       police: perpendicular to the petrophysical trend is exactly the
 *       lambda/mu -- equivalently Vp/Vs ratio -- crosstalk direction.
 *
 *     - DISTANCE: at least a quarter of the histogram box diagonal, so
 *       the probe clears the cloud even when the minor spread is small
 *       (a nearly one-dimensional trend), where 3 sigma of the minor
 *       axis would still leave it sitting on the ridge with an
 *       ill-determined gradient.
 *
 *  3. CHAIN RULE.  petroAccumGradient is checked against a central
 *     finite difference of J_P for BOTH parameterizations, param=1
 *     (lambda,mu,rho) and param=2 (Vp,Vs,rho), whatever the crossplot
 *     space.  One of the two is the identity and one is a chain rule;
 *     which is which depends on petro_space, so running the test with
 *     petro_space=1 and petro_space=2 covers all four cases in
 *     petroAccumGradient.
 *
 *     This is the check that catches a dropped g_rho term.  J_P has no
 *     explicit rho dependence, but whenever the crossplot space and the
 *     parameterization differ, the constrained pair carries rho, so
 *     dJ_P/drho is NOT zero.
 *
 *  4. HESSIAN.  petroHessVec (the exact J_P Hessian the TRN family adds
 *     to every Hd) against a central difference of petroAccumGradient,
 *     both params, far field and at the PDF centroid.  Catches a dropped
 *     second-derivative term of the parameter map in the mixed cases.
 *
 * Usage:
 *   test_petro petro_file=wells.txt [petro_space=] [petro_nlam=]
 *              [petro_nmu=] [petro_smooth=] [petro_eps=] [verbose=]
 *--------------------------------------------------------------------*/

char *sdoc[] = {
"  ",
" test_petro - verify the probabilistic petrophysical constraint J_P",
"  ",
" test_petro petro_file= [optional parameters]",
"  ",
" Required parameters:",
"   petro_file=        ASCII well-log crossplot, one pair per line, in the",
"                      units of petro_space",
"  ",
" Optional parameters:",
"   petro_space=1      crossplot plane: 1=(lambda,mu) in GPa,",
"                      2=(Vp,Vs) in km/s",
"   petro_nlam=100     PDF histogram bins along axis 1",
"   petro_nmu=100      PDF histogram bins along axis 2",
"   petro_smooth=1     Gaussian smoothing of the histogram, in bins",
"   petro_eps=0        floor on d(m,p); 0 = half a histogram bin diagonal",
"   verbose=1          1=print the three check tables",
"  ",
" Runs four checks on J_P (Aragao & Sava 2020, eq 16-20) with no wave",
" propagation: a Taylor test of (J,g) consistency, an attraction test that",
" catches a sign error the Taylor test cannot, and a finite-difference test",
" of petroAccumGradient for both param=1 and param=2, including the",
" dJ_P/drho term that appears when the crossplot space and the inversion",
" parameterization differ.",
"  ",
" Run it twice, with petro_space=1 and petro_space=2, to cover all four",
" mapping cases.  Exit status is 0 if all checks pass, 1 otherwise.",
"  ",
NULL};

#define NTEST 64   /* model points used by the Taylor check */

static float frand(unsigned int *seed)
{
	*seed = (*seed * 1103515245u + 12345u);
	return (float)((*seed >> 16) & 0x7fff) / 32767.0f;
}

/* Crossplot coordinates (a1,a2) + rho  ->  [lambda | mu | rho] in Pa,
 * i.e. the inverse of the mapping petroCostGrad does internally.  Used
 * to place test points by their position relative to the PDF, which is
 * the only frame in which "inside the cloud" and "far field" mean
 * anything. */
static void petroMakeLame(int space, float a1, float a2, float rho, float *t)
{
	if (space == PETRO_SPACE_VPVS) {
		float vp = a1 * 1.0e3f;              /* km/s -> m/s */
		float vs = a2 * 1.0e3f;
		t[0] = rho * (vp*vp - 2.0f*vs*vs);   /* lambda */
		t[1] = rho * vs*vs;                  /* mu     */
	} else {
		t[0] = a1 * 1.0e9f;                  /* GPa -> Pa */
		t[1] = a2 * 1.0e9f;
	}
	t[2] = rho;
}

/* param vector -> [lambda | mu | rho] in Pa, for the finite differences
 * of check 3.  Mirrors injectModelVector/extractModelVector. */
static void petroParamToLame(int param, const float *v, float *t)
{
	if (param == 2) {
		t[0] = v[2] * (v[0]*v[0] - 2.0f*v[1]*v[1]);
		t[1] = v[2] * v[1]*v[1];
		t[2] = v[2];
	} else {
		t[0] = v[0];
		t[1] = v[1];
		t[2] = v[2];
	}
}

int main(int argc, char **argv)
{
	petroPar pp;
	char  *petro_file;
	int    space, nlam, nmu, verbose, i, k, nfail = 0;
	float  smooth, eps;
	float *xlame, *dm, *g1, *g2, *gvec, *xpert;
	float  c1_c = 0.0f, c2_c = 0.0f, extent;
	double J0, gdotdm;
	unsigned int seed = 12345u;

	initargs(argc, argv);
	requestdoc(0);

	if (!getparstring("petro_file", &petro_file))
		verr("petro_file= is required (ASCII crossplot sample pairs)");
	if (!getparint("petro_space", &space)) space = PETRO_SPACE_LAMMU;
	if (!getparint("petro_nlam", &nlam)) nlam = 100;
	if (!getparint("petro_nmu", &nmu)) nmu = 100;
	if (!getparfloat("petro_smooth", &smooth)) smooth = 1.0f;
	if (!getparfloat("petro_eps", &eps)) eps = 0.0f;
	if (!getparint("verbose", &verbose)) verbose = 1;

	if (space != PETRO_SPACE_LAMMU && space != PETRO_SPACE_VPVS)
		verr("petro_space=%d: must be 1 (lambda,mu) or 2 (Vp,Vs)", space);

	if (petroBuild(petro_file, space, nlam, nmu, smooth, eps, &pp, verbose))
		verr("petroBuild failed");

	/* Centroid of the PDF: the attractor the model should fall toward */
	for (k = 0; k < pp.ncell; k++) {
		c1_c += pp.f[k] * pp.c1[k];
		c2_c += pp.f[k] * pp.c2[k];
	}
	extent = 0.5f * ((pp.c1_max - pp.c1_min) + (pp.c2_max - pp.c2_min));
	vmess("PDF centroid: a1=%.4f a2=%.4f, mean extent %.4f (%s)",
	      c1_c, c2_c, extent,
	      (space == PETRO_SPACE_VPVS) ? "km/s" : "GPa");

	xlame = (float *)malloc(3 * NTEST * sizeof(float));
	xpert = (float *)malloc(3 * NTEST * sizeof(float));
	dm    = (float *)malloc(3 * NTEST * sizeof(float));
	gvec  = (float *)malloc(3 * NTEST * sizeof(float));
	g1    = (float *)malloc(NTEST * sizeof(float));
	g2    = (float *)malloc(NTEST * sizeof(float));

	/* Scatter test points over a box larger than the PDF extent, so both
	 * the interior (gradient ~0) and the far field are sampled.  In Vp-Vs
	 * space the axes are clamped positive: a non-positive Vs is not a
	 * rock, and petroCostGrad would be asked to take sqrt of mu<0. */
	for (i = 0; i < NTEST; i++) {
		float fl = 1.8f * frand(&seed) - 0.4f;
		float fm = 1.8f * frand(&seed) - 0.4f;
		float a1 = pp.c1_min + fl*(pp.c1_max - pp.c1_min);
		float a2 = pp.c2_min + fm*(pp.c2_max - pp.c2_min);
		float rho = 2000.0f + 400.0f * frand(&seed);
		float t[3];

		if (space == PETRO_SPACE_VPVS) {
			if (a1 < 0.2f*pp.c1_max) a1 = 0.2f*pp.c1_max;
			if (a2 < 0.2f*pp.c2_max) a2 = 0.2f*pp.c2_max;
		}
		petroMakeLame(space, a1, a2, rho, t);
		xlame[i]           = t[0];
		xlame[NTEST + i]   = t[1];
		xlame[2*NTEST + i] = t[2];
	}

	/* ==================================================================
	 * CHECK 1 -- Taylor, in Lame space so the inverse chain rule is
	 *            exercised too when space=Vp-Vs
	 * ================================================================== */
	vmess("");
	vmess("=== CHECK 1: Taylor, J(m+t dm) - J(m) - t<g,dm> = O(t^2) ===");

	J0 = petroCostGrad(xlame, NTEST, &pp, NULL, g1, g2);
	memset(gvec, 0, 3*NTEST*sizeof(float));
	petroAccumGradient(gvec, NTEST, 1, xlame, g1, g2, 1.0f, &pp);

	/* Perturb lambda, mu and rho together, each by ~1% of a characteristic
	 * magnitude for that parameter.  Scaling dlambda on (lambda+2mu)
	 * rather than on lambda keeps the perturbation meaningful at points
	 * where lambda happens to be near zero. */
	gdotdm = 0.0;
	for (i = 0; i < NTEST; i++) {
		float mscale = 0.01f * fabsf(xlame[i] + 2.0f*xlame[NTEST+i]);
		dm[i]           = mscale * (2.0f*frand(&seed) - 1.0f);
		dm[NTEST + i]   = mscale * (2.0f*frand(&seed) - 1.0f);
		dm[2*NTEST + i] = 0.01f * xlame[2*NTEST+i] * (2.0f*frand(&seed) - 1.0f);
		gdotdm += (double)gvec[i]*dm[i]
		        + (double)gvec[NTEST+i]*dm[NTEST+i]
		        + (double)gvec[2*NTEST+i]*dm[2*NTEST+i];
	}

	{
		double prev = 0.0;
		vmess("    t          J(m+t dm)-J(m)      residual        ratio");
		for (k = 0; k < 6; k++) {
			double t = pow(0.5, (double)k);
			double Jt, res, ratio;
			for (i = 0; i < 3*NTEST; i++)
				xpert[i] = xlame[i] + (float)(t * dm[i]);
			Jt  = petroCostGrad(xpert, NTEST, &pp, NULL, g1, g2);
			res = fabs(Jt - J0 - t*gdotdm);
			ratio = (k > 0 && res > 0.0) ? prev/res : 0.0;
			vmess("  %8.5f    %14.6e   %14.6e   %8.3f",
			      t, Jt - J0, res, ratio);
			/* second order: residual must drop by ~4x per halving */
			if (k > 0 && (ratio < 3.0 || ratio > 5.5)) {
				vmess("    ^ FAIL: ratio %.3f outside [3.0, 5.5]", ratio);
				nfail++;
			}
			prev = res;
		}
	}

	/* ==================================================================
	 * CHECK 2 -- the constraint attracts, it does not repel.
	 *            Probe along the PDF's minor axis; see the header note.
	 * ================================================================== */
	vmess("");
	vmess("=== CHECK 2: D(m) grows away from the PDF, g points back ===");
	{
		double Dprev = -1.0;
		double cxx = 0.0, cxy = 0.0, cyy = 0.0;
		double tr, dsc, lmin, lmaj, ux, uy, un, rstep, box;
		int    nprobe = 6;

		/* Covariance of f(p), then its minor eigenvector. */
		for (k = 0; k < pp.ncell; k++) {
			double d1 = pp.c1[k] - c1_c, d2 = pp.c2[k] - c2_c;
			cxx += pp.f[k]*d1*d1;
			cxy += pp.f[k]*d1*d2;
			cyy += pp.f[k]*d2*d2;
		}
		tr   = cxx + cyy;
		dsc  = sqrt(fmax(0.0, tr*tr - 4.0*(cxx*cyy - cxy*cxy)));
		lmin = 0.5*(tr - dsc);
		lmaj = 0.5*(tr + dsc);

		/* Eigenvector of lmin.  (cxy, lmin-cxx) degenerates when cxy ~ 0,
		 * i.e. when the axes are already principal; fall back to the
		 * other form, and to a plain axis if both collapse. */
		ux = cxy;  uy = lmin - cxx;
		un = sqrt(ux*ux + uy*uy);
		if (un < 1.0e-12*fmax(lmaj, 1.0e-30)) {
			ux = lmin - cyy;  uy = cxy;
			un = sqrt(ux*ux + uy*uy);
		}
		if (un <= 0.0) { ux = 0.0; uy = 1.0; un = 1.0; }
		ux /= un;  uy /= un;

		box   = sqrt((double)(pp.c1_max - pp.c1_min)*(pp.c1_max - pp.c1_min)
		           + (double)(pp.c2_max - pp.c2_min)*(pp.c2_max - pp.c2_min));
		rstep = fmax(3.0*sqrt(fmax(lmin, 0.0)), 0.25*box);

		/* The minor axis has no preferred sign, so pick the one that keeps
		 * the probe physical: in Vp-Vs space a far probe on the wrong side
		 * drives Vs (hence mu) negative, and petroCostGrad would take the
		 * square root of a negative modulus.  Of the two signs, take the
		 * one whose furthest probe has the larger minimum coordinate, then
		 * drop probes until even that one is physical. */
		if (space == PETRO_SPACE_VPVS) {
			double rmax = rstep * (double)nprobe;
			double pmin = fmin(c1_c + rmax*ux, c2_c + rmax*uy);
			double mmin = fmin(c1_c - rmax*ux, c2_c - rmax*uy);
			if (mmin > pmin) { ux = -ux; uy = -uy; }
			while (nprobe > 2 &&
			       fmin(c1_c + rstep*nprobe*ux,
			            c2_c + rstep*nprobe*uy) < 0.1*fmin(c1_c, c2_c))
				nprobe--;
		}

		vmess("  PDF principal axes: sqrt(lmaj)=%.4f sqrt(lmin)=%.4f, "
		      "minor dir (%.4f, %.4f), %d probes", sqrt(fmax(lmaj,0.0)),
		      sqrt(fmax(lmin,0.0)), ux, uy, nprobe);
		vmess("   dist           D(m)       <g, unit_away>  (must be > 0)");

		for (k = 1; k <= nprobe; k++) {
			double r = rstep * (double)k;
			float  one[3];
			double D, proj;
			petroMakeLame(space, (float)(c1_c + r*ux), (float)(c2_c + r*uy),
			              2000.0f, one);
			D = petroCostGrad(one, 1, &pp, NULL, g1, g2);
			/* gradient of J in crossplot coordinates; DESCENT is -g, so
			 * +<g, away> means descending moves back toward the cloud */
			proj = ux * (double)g1[0] + uy * (double)g2[0];
			vmess("  %8.3f    %12.6e    %14.6e", r, D, proj);
			if (proj <= 0.0) {
				vmess("    ^ FAIL: gradient does not point back to the PDF "
				      "(sign error)");
				nfail++;
			}
			if (Dprev >= 0.0 && D <= Dprev) {
				vmess("    ^ FAIL: D(m) did not increase with distance");
				nfail++;
			}
			Dprev = D;
		}
	}

	/* ==================================================================
	 * CHECK 3 -- petroAccumGradient vs finite difference, both params
	 * ================================================================== */
	vmess("");
	vmess("=== CHECK 3: petroAccumGradient vs finite difference ===");
	{
		float one[3];
		int   param;

		/* A point outside the cloud, where the gradient is well away from
		 * zero and the finite difference is meaningful. */
		petroMakeLame(space,
		              pp.c1_max + 0.3f*(pp.c1_max - pp.c1_min),
		              pp.c2_max + 0.3f*(pp.c2_max - pp.c2_min),
		              2200.0f, one);

		for (param = 1; param <= 2; param++) {
			const char *pn1[3] = {"lam", "mu ", "rho"};
			const char *pn2[3] = {"Vp ", "Vs ", "rho"};
			const char **pname = (param == 2) ? pn2 : pn1;
			float g3[3], v0[3], step[3];

			if (param == 2) {
				v0[2] = one[2];
				v0[0] = sqrtf((one[0] + 2.0f*one[1]) / one[2]);
				v0[1] = sqrtf(one[1] / one[2]);
				step[0] = step[1] = 1.0e-3f * v0[0];
				step[2] = 1.0e-3f * v0[2];
			} else {
				v0[0] = one[0]; v0[1] = one[1]; v0[2] = one[2];
				step[0] = step[1] = 1.0e-3f * (one[0] + 2.0f*one[1]);
				step[2] = 1.0e-3f * one[2];
			}

			petroCostGrad(one, 1, &pp, NULL, g1, g2);
			memset(g3, 0, sizeof(g3));
			petroAccumGradient(g3, 1, param, one, g1, g2, 1.0f, &pp);

			vmess("  param=%d  (%s, %s, %s)   test point "
			      "%.4e %.4e %.4e", param,
			      pname[0], pname[1], pname[2], v0[0], v0[1], v0[2]);
			vmess("   par       analytic          FD           rel.err");

			for (k = 0; k < 3; k++) {
				float  v[3], t[3];
				double Jp, Jm, fd, rel, scale;

				memcpy(v, v0, sizeof(v));
				v[k] += step[k];
				petroParamToLame(param, v, t);
				Jp = petroCostGrad(t, 1, &pp, NULL, g1, g2);

				memcpy(v, v0, sizeof(v));
				v[k] -= step[k];
				petroParamToLame(param, v, t);
				Jm = petroCostGrad(t, 1, &pp, NULL, g1, g2);

				fd    = (Jp - Jm) / (2.0 * step[k]);
				scale = fabs(fd) > 0.0 ? fabs(fd) : 1.0;
				rel   = fabs(g3[k] - fd) / scale;
				vmess("   %s   %14.6e  %14.6e   %9.2e",
				      pname[k], g3[k], fd, rel);
				/* An exactly-zero analytic AND FD pair is a pass: that is
				 * the matched case, where J_P really has no rho
				 * dependence. */
				if (g3[k] == 0.0f && fabs(fd) < 1.0e-30) continue;
				if (rel > 1.0e-3) {
					vmess("    ^ FAIL: %s gradient disagrees with finite "
					      "difference", pname[k]);
					nfail++;
				}
			}
			vmess("");
		}
	}

	/* ==================================================================
	 * CHECK 4 -- petroHessVec vs finite difference of the gradient
	 *
	 * H_P d  =?  [g(m + h d) - g(m - h d)] / 2h,  g = petroAccumGradient
	 *
	 * Both params, so both the matched (linear map, T_k = 0) and the mixed
	 * (second-derivative T_k term essential) cases are exercised.  Two
	 * points: the far-field point of check 3, and the PDF centroid, where
	 * hess D is dominated by the S term and can be indefinite.
	 * ================================================================== */
	vmess("");
	vmess("=== CHECK 4: petroHessVec vs finite difference of gradient ===");
	{
		float pts[2][3];
		const char *ptname[2] = {"far field", "centroid "};
		int   ipt, param;
#define NHSTEP 7
		const double hstep[NHSTEP] = {1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5};

		petroMakeLame(space,
		              pp.c1_max + 0.3f*(pp.c1_max - pp.c1_min),
		              pp.c2_max + 0.3f*(pp.c2_max - pp.c2_min),
		              2200.0f, pts[0]);
		petroMakeLame(space, c1_c, c2_c, 2100.0f, pts[1]);

		vmess("   point      param   ||H d||          ||FD||           rel.err");
		for (ipt = 0; ipt < 2; ipt++) {
			for (param = 1; param <= 2; param++) {
				float  v0[3], dv[3], hd[3], gp[3], gm[3], v[3], t[3];
				float  ga[1], gb[1];
				double nh = 0.0, nf = 0.0, rel, hbest;
				int    ih;
				const float *one = pts[ipt];

				if (param == 2) {
					v0[2] = one[2];
					v0[0] = sqrtf((one[0] + 2.0f*one[1]) / one[2]);
					v0[1] = sqrtf(one[1] / one[2]);
					dv[0] =  0.7f * v0[0];
					dv[1] = -0.5f * v0[1];
					dv[2] =  0.4f * v0[2];
				} else {
					v0[0] = one[0]; v0[1] = one[1]; v0[2] = one[2];
					dv[0] =  0.7f * (one[0] + 2.0f*one[1]);
					dv[1] = -0.5f * (one[0] + 2.0f*one[1]);
					dv[2] =  0.4f * one[2];
				}

				memset(hd, 0, sizeof(hd));
				petroHessVec(hd, dv, 1, param, one, &pp, NULL, 1.0f);
				for (k = 0; k < 3; k++) nh += (double)hd[k]*hd[k];

				/* Step sweep, best step reported.  g_P is float32 and, near
				 * the cloud, a small net of large cancelling per-cell terms,
				 * so its FD has a rounding floor ~ ulp(sum|terms|)/h that
				 * grows as h shrinks while truncation shrinks with h.  A
				 * fixed h either under-resolves or drowns in that floor; a
				 * correct Hd shows a clear minimum, a wrong one never
				 * converges at any h. */
				rel = 1.0e30; hbest = 0.0;
				for (ih = 0; ih < NHSTEP; ih++) {
					double h = hstep[ih], nf_h = 0.0, ne_h = 0.0, r_h;

					for (k = 0; k < 3; k++) v[k] = v0[k] + (float)h * dv[k];
					petroParamToLame(param, v, t);
					petroCostGrad(t, 1, &pp, NULL, ga, gb);
					memset(gp, 0, sizeof(gp));
					petroAccumGradient(gp, 1, param, t, ga, gb, 1.0f, &pp);

					for (k = 0; k < 3; k++) v[k] = v0[k] - (float)h * dv[k];
					petroParamToLame(param, v, t);
					petroCostGrad(t, 1, &pp, NULL, ga, gb);
					memset(gm, 0, sizeof(gm));
					petroAccumGradient(gm, 1, param, t, ga, gb, 1.0f, &pp);

					for (k = 0; k < 3; k++) {
						double fd = ((double)gp[k] - (double)gm[k]) / (2.0*h);
						nf_h += fd*fd;
						ne_h += (hd[k]-fd)*(hd[k]-fd);
					}
					r_h = sqrt(ne_h) / (sqrt(nf_h) > 0.0 ? sqrt(nf_h) : 1.0);
					if (verbose >= 2)
						vmess("        h=%8.1e  ||FD||=%14.6e  rel.err=%9.2e",
						      h, sqrt(nf_h), r_h);
					if (r_h < rel) { rel = r_h; nf = nf_h; hbest = h; }
				}
				vmess("   %s    %d    %14.6e  %14.6e   %9.2e  (h=%.0e)",
				      ptname[ipt], param, sqrt(nh), sqrt(nf), rel, hbest);
				if (rel > 5.0e-3) {
					vmess("    ^ FAIL: H_P d disagrees with FD of g_P");
					nfail++;
				}
			}
		}
	}

	if (nfail == 0)
		vmess("test_petro: ALL CHECKS PASSED (petro_space=%d)", space);
	else
		vmess("test_petro: %d CHECK(S) FAILED (petro_space=%d)", nfail, space);

	free(xlame); free(xpert); free(dm); free(gvec); free(g1); free(g2);
	petroFree(&pp);

	return (nfail == 0) ? 0 : 1;
}
