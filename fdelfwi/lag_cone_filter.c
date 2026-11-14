#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"
#include"genfft.h"

/**
 * lag_cone_filter.c -- 3D slope (cone) filter for the lag-extended gradient
 *                      cube g(tau, x, z).
 *
 * This is a C port of demo/gradient/lag_slope_filter.py:slope_filter3d, which
 * is the validated reference.  It is kept deliberately self-contained: it
 * knows nothing about modPar, gradients, or parameterisation, and operates on
 * a plain (nlag, nx, nz) array.  lag_gradient.c calls it; nothing else should
 * need to.
 *
 * WHAT IS FILTERED
 * ----------------
 * A local event in the cube is a plane
 *
 *     tau = tau0 + p . xi,        p = (px, pz)   [s/m]
 *
 * whose 3D Fourier support is the line (kx, kz) = -p nu.  Hence
 * |k|/|nu| = |p| and the filter is a CONE in (kx, kz, nu).  The slope
 * magnitude is quoted in ms/m,
 *
 *     s = 1000 sqrt(kx^2 + kz^2) / |nu|        k in cycles/m, nu in cycles/s
 *
 * and the response is
 *
 *     H = 1                 s <= s_pass
 *     H = cos^2(pi u / 2)   s_pass < s < s_stop,  u = (s-s_pass)/(s_stop-s_pass)
 *     H = 0                 s >= s_stop
 *
 * H(s_pass) = 1, H(midpoint) = 0.5, H(s_stop) = 0, and dH/ds vanishes at both
 * corners, so no corner ringing is introduced.
 *
 * A SMALL slope is a nearly lag-INDEPENDENT event, i.e. a wide opening angle
 * -- the transmission/tomographic kernel.  A LARGE slope is the specular
 * reflection imprint.  So keeping s <= s_pass keeps the TOMOGRAPHIC part.
 * The caller converts an opening angle to a slope through
 * |p| = cos(theta)/v with 2theta the opening angle (see lagConeAngleToSlope).
 *
 * nu = 0 IS HANDLED EXPLICITLY, not with an epsilon.  Taking the limit of the
 * definition:
 *
 *     k != 0, nu -> 0   =>   s -> infinity   =>   H = 0
 *     k  = 0, nu  = 0   =>   s = 0/0, the cube mean; slope undefined, H = 1
 *
 * so the default (nu0_keep = 0) keeps only the (0,0,0) sample of the nu = 0
 * plane.  That plane is the tau-INDEPENDENT part of the cube and it is NOT
 * negligible, so its share of the total energy is reported in lagConeStats.
 * nu0_keep = 1 passes the whole plane instead; it is a sensitivity check, not
 * the limit of the stated formula.
 *
 * EDGES
 * -----
 * A cone filter is a global convolution, so a hard cube edge wraps.  Two
 * things are done about it and both are visible to the caller:
 *
 *   - a raised-cosine taper is applied to the tau, x and z edges BEFORE the
 *     transform.  The TAPERED cube is returned alongside the passed one, and
 *     the caller must define rejected = tapered - passed, so that
 *     passed + rejected is the tapered field EXACTLY and the split neither
 *     creates nor destroys energy.  Comparing a filtered field against the
 *     RAW input would credit the taper to the filter;
 *   - every axis is zero-padded before the transform and cropped afterwards,
 *     so the circular convolution is linear over the original support.
 *
 * The taper is part of the operator, not a cosmetic step.
 *
 * NOTE ON THE SPATIAL TAPER.  tap_x and tap_z attenuate the cube -- and hence
 * the tau = 0 slice that becomes the gradient -- near the model edges.  That
 * is a real modification of the gradient, not just of the diagnostic, so the
 * fractions are caller-supplied and reported.  The tau taper is harmless at
 * tau = 0 (an odd nlag puts tau = 0 at the centre, where the weight is 1) but
 * still acts indirectly, because the filter mixes lags.
 *
 * PADDING LENGTHS
 * ---------------
 * Padded lengths are rounded UP to a power of two at or above pad*n.  Powers
 * of two are valid for every FFTlib backend (the Mayer fallback that this
 * tree currently compiles, as well as MKL/ACML/SCS), which removes any
 * dependence on which backend genfft was built with.  The actual lengths are
 * reported in lagConeStats so a reference run can mirror them exactly.
 *
 * SIGN CONVENTION IS IRRELEVANT HERE, which is worth stating because it
 * removes a whole class of bugs: H depends only on |k| and |nu| and is
 * therefore even in k, and the input is real.  Any pair of opposite
 * forward/inverse signs gives the same answer.  Only the NORMALISATION
 * matters: FFTlib transforms are unscaled, so the round trip is multiplied by
 * Nt*Nx*Nz and we divide it out once at the end.  lagConeSelfTest verifies
 * exactly this.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MS_PER_S 1000.0f


/***********************************************************************
 * nextpow2 -- Smallest power of two >= n.
 ***********************************************************************/
static int nextpow2(int n)
{
	int m = 1;
	while (m < n) m <<= 1;
	return m;
}


/***********************************************************************
 * lagConeSlopeResponse -- H(s), s in ms/m.
 *
 * Non-finite s (an infinitely steep event) returns 0.
 ***********************************************************************/
float lagConeSlopeResponse(float s, float s_pass, float s_stop)
{
	float u, c;

	if (!isfinite(s))   return 0.0f;
	if (s <= s_pass)    return 1.0f;
	if (s >= s_stop)    return 0.0f;

	u = (s - s_pass) / (s_stop - s_pass);
	c = cosf(0.5f * (float)M_PI * u);
	return c * c;
}


/***********************************************************************
 * lagConeAngleToSlope -- Full opening angle 2theta [deg] -> |p| [ms/m].
 *
 * |p| = cos(theta)/v, the same-mode reduction of p = -0.5[n_a/v_a + n_b/v_b].
 * This is the ONE place a velocity enters the filter, and it enters only to
 * turn an angle into a slope.  The conversion is stationary -- a single
 * velocity for the whole cube -- which is the filter's main approximation.
 ***********************************************************************/
float lagConeAngleToSlope(float theta_deg, float v)
{
	if (v <= 0.0f)
		verr("lagConeAngleToSlope: velocity must be > 0, got %g.", v);
	return MS_PER_S * cosf((float)(0.5 * theta_deg * M_PI / 180.0)) / v;
}


/***********************************************************************
 * lagConeTaper1d -- Raised-cosine taper over the outer `frac` of each end.
 *
 * frac = 0 gives a box, frac = 0.5 a full Hann.  Matches
 * lag_slope_filter.py:taper1d sample for sample, including the half-sample
 * offset that keeps the taper strictly inside (0,1).
 *
 * NOT static: lag_cone_pspi.c must taper identically, and one definition is
 * the only way to guarantee that the PSPI path and the stationary path can be
 * compared sample for sample.
 ***********************************************************************/
void lagConeTaper1d(float *w, int n, float frac)
{
	int i, k;

	for (i = 0; i < n; i++) w[i] = 1.0f;

	k = (int)floor((double)frac * (double)n + 0.5);   /* python round() */
	if (k < 1) return;
	if (k > n / 2) k = n / 2;

	for (i = 0; i < k; i++) {
		double r = ((double)i + 0.5) / (double)k;
		float  e = (float)(0.5 * (1.0 - cos(M_PI * r)));
		w[i]         = e;
		w[n - 1 - i] = e;
	}
}


/***********************************************************************
 * lagConeDefaults -- Fill a lagConePar with the reference defaults.
 ***********************************************************************/
void lagConeDefaults(lagConePar *p)
{
	p->s_pass   = 0.20f;    /* ms/m */
	p->s_stop   = 0.30f;    /* ms/m */
	p->tap_tau  = 0.10f;
	p->tap_x    = 0.10f;
	p->tap_z    = 0.10f;
	p->pad      = 2;
	p->nu0_keep = 0;
	p->theta_pass = 0.0f;   /* unset; only lag_cone_pspi.c reads these */
	p->theta_stop = 0.0f;
	p->nv         = 1;      /* 1 = stationary, the shipped behaviour   */
}


/***********************************************************************
 * lagConeFilter -- Apply the cone filter to one cube.
 *
 * cube     [nlag*nx*nz] input, order (lag, x, z) with z fastest -- the
 *          order lag_gradient.c builds and lag_slope_filter.py reads.
 * passed   [nlag*nx*nz] output, the s <= s_pass cone.  Required.
 * tapered  [nlag*nx*nz] output, the input after the edge taper.  May be
 *          NULL, but then the caller cannot form an exact `rejected`.
 * st       statistics, may be NULL.
 *
 * `cube` is not modified.  passed may not alias cube or tapered.
 ***********************************************************************/
void lagConeFilter(const float *cube, int nlag, int nx, int nz,
                   float dx, float dz, float dtau,
                   const lagConePar *p,
                   float *passed, float *tapered,
                   lagConeStats *st, int verbose)
{
	int    ilag, ix, iz, inu, ikx, ikz;
	int    Nt, Nx, Nz, Nnu;
	float  *wt, *wx, *wz, *tap;
	float  *kx, *kz;
	complex *G;
	double e_tot = 0.0, e_nu0 = 0.0, e_pass = 0.0;
	float  scl;
	size_t plane, i, ncube;

	if (nlag < 1 || nx < 1 || nz < 1)
		verr("lagConeFilter: bad cube shape (%d, %d, %d).", nlag, nx, nz);
	if (!(p->s_stop > p->s_pass))
		verr("lagConeFilter: need s_stop > s_pass, got %g and %g ms/m.",
		     p->s_pass, p->s_stop);
	if (dtau <= 0.0f || dx <= 0.0f || dz <= 0.0f)
		verr("lagConeFilter: dx=%g, dz=%g, dtau=%g must all be > 0.",
		     dx, dz, dtau);
	if (p->pad < 1)
		verr("lagConeFilter: pad=%d must be >= 1.", p->pad);
	if (passed == NULL)
		verr("lagConeFilter: `passed` must not be NULL.");

	ncube = (size_t)nlag * nx * nz;

	/* ---------------- edge taper ---------------- */
	wt = (float *)malloc((size_t)nlag * sizeof(float));
	wx = (float *)malloc((size_t)nx   * sizeof(float));
	wz = (float *)malloc((size_t)nz   * sizeof(float));
	if (!wt || !wx || !wz) verr("lagConeFilter: cannot allocate tapers.");
	lagConeTaper1d(wt, nlag, p->tap_tau);
	lagConeTaper1d(wx, nx,   p->tap_x);
	lagConeTaper1d(wz, nz,   p->tap_z);

	tap = (tapered != NULL) ? tapered
	                        : (float *)malloc(ncube * sizeof(float));
	if (!tap) verr("lagConeFilter: cannot allocate the tapered cube.");

#pragma omp parallel for private(ilag, ix, iz) schedule(static)
	for (ilag = 0; ilag < nlag; ilag++) {
		for (ix = 0; ix < nx; ix++) {
			float wtx = wt[ilag] * wx[ix];
			size_t o = ((size_t)ilag * nx + ix) * nz;
			for (iz = 0; iz < nz; iz++)
				tap[o + iz] = cube[o + iz] * wtx * wz[iz];
		}
	}

	/* ---------------- padded transform sizes ---------------- */
	Nt  = nextpow2(p->pad * nlag);
	Nx  = nextpow2(p->pad * nx);
	Nz  = nextpow2(p->pad * nz);
	Nnu = Nt / 2 + 1;
	plane = (size_t)Nx * Nz;

	G = (complex *)calloc((size_t)Nnu * plane, sizeof(complex));
	if (!G)
		verr("lagConeFilter: cannot allocate the %d x %d x %d spectrum "
		     "(%.2f GB).  Reduce pad, the lag count or the model size.",
		     Nnu, Nx, Nz,
		     (double)Nnu * plane * sizeof(complex) / 1073741824.0);

	if (verbose)
		vmess("lagConeFilter: cube (%d,%d,%d) -> padded (%d,%d,%d), "
		      "s_pass=%g s_stop=%g ms/m, pad>=%d, nu0=%s",
		      nlag, nx, nz, Nt, Nx, Nz, p->s_pass, p->s_stop, p->pad,
		      p->nu0_keep ? "keep" : "limit");

	/* ---------------- forward along tau ----------------
	 * One real trace per (ix, iz).  Zero beyond nlag; the calloc above
	 * already zeroed the x, z pad region. */
#pragma omp parallel private(ix, iz, ilag, inu)
	{
		float   *rbuf = (float   *)malloc((size_t)Nt  * sizeof(float));
		complex *cbuf = (complex *)malloc((size_t)Nnu * sizeof(complex));
		if (!rbuf || !cbuf) verr("lagConeFilter: cannot allocate tau scratch.");

#pragma omp for schedule(static)
		for (ix = 0; ix < nx; ix++) {
			for (iz = 0; iz < nz; iz++) {
				memset(rbuf, 0, (size_t)Nt * sizeof(float));
				for (ilag = 0; ilag < nlag; ilag++)
					rbuf[ilag] = tap[((size_t)ilag * nx + ix) * nz + iz];

				rc1fft(rbuf, cbuf, Nt, 1);

				for (inu = 0; inu < Nnu; inu++)
					G[(size_t)inu * plane + (size_t)ix * Nz + iz] = cbuf[inu];
			}
		}
		free(rbuf);
		free(cbuf);
	}

	/* ---------------- forward over (x, z), per nu ----------------
	 * cc2dfft(data, n1_fast, n2_slow, ld1, sign); z is the fast axis. */
	for (inu = 0; inu < Nnu; inu++)
		cc2dfft(G + (size_t)inu * plane, Nz, Nx, Nz, 1);

	/* ---------------- wavenumber axes (cycles/m), fftfreq order ---------- */
	kx = (float *)malloc((size_t)Nx * sizeof(float));
	kz = (float *)malloc((size_t)Nz * sizeof(float));
	if (!kx || !kz) verr("lagConeFilter: cannot allocate wavenumber axes.");
	for (ikx = 0; ikx < Nx; ikx++)
		kx[ikx] = (float)((ikx <= Nx/2 ? ikx : ikx - Nx) / ((double)Nx * dx));
	for (ikz = 0; ikz < Nz; ikz++)
		kz[ikz] = (float)((ikz <= Nz/2 ? ikz : ikz - Nz) / ((double)Nz * dz));

	/* ---------------- energy budget before the filter ---------------- */
	for (inu = 0; inu < Nnu; inu++) {
		const complex *Gp = G + (size_t)inu * plane;
		double e = 0.0;
		for (i = 0; i < plane; i++)
			e += (double)Gp[i].r*Gp[i].r + (double)Gp[i].i*Gp[i].i;
		e_tot += e;
		if (inu == 0) e_nu0 = e;
	}

	/* ---------------- the cone ---------------- */
	/* nu = 0 plane, explicitly: keep only k = 0. */
	if (!p->nu0_keep) {
		complex *Gp = G;
		for (ikx = 0; ikx < Nx; ikx++) {
			for (ikz = 0; ikz < Nz; ikz++) {
				if (kx[ikx] != 0.0f || kz[ikz] != 0.0f) {
					Gp[(size_t)ikx * Nz + ikz].r = 0.0f;
					Gp[(size_t)ikx * Nz + ikz].i = 0.0f;
				}
			}
		}
	}

#pragma omp parallel for private(inu, ikx, ikz) schedule(static)
	for (inu = 1; inu < Nnu; inu++) {
		complex *Gp = G + (size_t)inu * plane;
		double nu = (double)inu / ((double)Nt * dtau);   /* cycles/s */
		for (ikx = 0; ikx < Nx; ikx++) {
			for (ikz = 0; ikz < Nz; ikz++) {
				size_t j = (size_t)ikx * Nz + ikz;
				double K = sqrt((double)kx[ikx]*kx[ikx]
				              + (double)kz[ikz]*kz[ikz]);
				float  H = lagConeSlopeResponse((float)(MS_PER_S * K / nu),
				                                p->s_pass, p->s_stop);
				Gp[j].r *= H;
				Gp[j].i *= H;
			}
		}
	}

	for (inu = 0; inu < Nnu; inu++) {
		const complex *Gp = G + (size_t)inu * plane;
		for (i = 0; i < plane; i++)
			e_pass += (double)Gp[i].r*Gp[i].r + (double)Gp[i].i*Gp[i].i;
	}

	/* ---------------- inverse over (x, z), per nu ---------------- */
	for (inu = 0; inu < Nnu; inu++)
		cc2dfft(G + (size_t)inu * plane, Nz, Nx, Nz, -1);

	/* ---------------- inverse along tau, crop ----------------
	 * FFTlib transforms are unscaled, so undo Nt*Nx*Nz once here. */
	scl = (float)(1.0 / ((double)Nt * (double)Nx * (double)Nz));

#pragma omp parallel private(ix, iz, ilag, inu)
	{
		float   *rbuf = (float   *)malloc((size_t)Nt  * sizeof(float));
		complex *cbuf = (complex *)malloc((size_t)Nnu * sizeof(complex));
		if (!rbuf || !cbuf) verr("lagConeFilter: cannot allocate tau scratch.");

#pragma omp for schedule(static)
		for (ix = 0; ix < nx; ix++) {
			for (iz = 0; iz < nz; iz++) {
				for (inu = 0; inu < Nnu; inu++)
					cbuf[inu] = G[(size_t)inu * plane + (size_t)ix * Nz + iz];

				cr1fft(cbuf, rbuf, Nt, -1);

				for (ilag = 0; ilag < nlag; ilag++)
					passed[((size_t)ilag * nx + ix) * nz + iz]
						= scl * rbuf[ilag];
			}
		}
		free(rbuf);
		free(cbuf);
	}

	/* ---------------- stats ---------------- */
	if (st != NULL) {
		st->nlag = nlag; st->nx = nx; st->nz = nz;
		st->Nt = Nt; st->Nx = Nx; st->Nz = Nz;
		st->nu_max = (float)((Nt/2) / ((double)Nt * dtau));
		st->energy_frac_nu0    = (e_tot > 0.0) ? (float)(e_nu0  / e_tot) : 0.0f;
		st->energy_frac_passed = (e_tot > 0.0) ? (float)(e_pass / e_tot) : 0.0f;
	}

	if (verbose)
		vmess("  nu up to %.2f Hz; energy on the nu=0 plane %.3f%% of the "
		      "total; spectral energy kept %.3f%%",
		      (Nt/2) / ((double)Nt * dtau),
		      100.0 * ((e_tot > 0.0) ? e_nu0 / e_tot : 0.0),
		      100.0 * ((e_tot > 0.0) ? e_pass / e_tot : 0.0));

	free(G);
	free(kx); free(kz);
	free(wt); free(wx); free(wz);
	if (tapered == NULL) free(tap);
}


/***********************************************************************
 * lagConeSelfTest -- Verify the normalisation of the FFT round trip.
 *
 * The one thing in this file that depends on how genfft was built is the
 * scaling of the unscaled forward/inverse pair.  Running the filter with a
 * response that is identically 1 (s_pass above the largest reachable slope),
 * no taper and no padding must return the input exactly.  Returns the max
 * absolute error relative to the input RMS; anything above ~1e-5 means the
 * backend does not use the documented unscaled convention and `scl` in
 * lagConeFilter is wrong.
 *
 * Cheap enough (a 8 x 16 x 16 cube) to call at startup behind a verbose flag.
 ***********************************************************************/
float lagConeSelfTest(int verbose)
{
	const int nlag = 8, nx = 16, nz = 16;
	size_t n = (size_t)nlag * nx * nz, i;
	float *in, *out, *tap;
	double num = 0.0, den = 0.0;
	float err;
	lagConePar p;

	in  = (float *)malloc(n * sizeof(float));
	out = (float *)malloc(n * sizeof(float));
	tap = (float *)malloc(n * sizeof(float));
	if (!in || !out || !tap) verr("lagConeSelfTest: allocation failed.");

	for (i = 0; i < n; i++)
		in[i] = (float)sin(0.1 * (double)i) + 0.3f * (float)cos(0.37 * (double)i);

	lagConeDefaults(&p);
	p.tap_tau = 0.0f; p.tap_x = 0.0f; p.tap_z = 0.0f;
	p.pad     = 1;
	/* H == 1 everywhere: put s_pass above the largest slope the grid can
	 * represent, which is bounded by 1000*|k|max/nu_min. */
	p.s_pass  = 1.0e9f;
	p.s_stop  = 2.0e9f;
	p.nu0_keep = 1;          /* the limit rule would zero the nu=0 plane */

	lagConeFilter(in, nlag, nx, nz, 10.0f, 10.0f, 0.005f,
	              &p, out, tap, NULL, 0);

	for (i = 0; i < n; i++) {
		double d = (double)out[i] - (double)in[i];
		if (fabs(d) > num) num = fabs(d);
		den += (double)in[i] * in[i];
	}
	den = sqrt(den / (double)n);
	err = (den > 0.0) ? (float)(num / den) : (float)num;

	if (verbose)
		vmess("lagConeSelfTest: FFT round-trip max error %.3e (relative to "
		      "input RMS) -- %s", err,
		      err < 1.0e-5f ? "normalisation OK"
		                    : "NORMALISATION WRONG, see lagConeFilter scl");

	free(in); free(out); free(tap);
	return err;
}
