#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"
#include"genfft.h"

/**
 * lag_cone_pspi.c -- cone filter with a cutoff that FOLLOWS v(x,z), returning
 *                    only the tau = 0 slice.
 *
 * THE PROBLEM
 * -----------
 * lagConeFilter converts an opening angle into a slope through
 * |p| = cos(theta)/v using ONE scalar velocity for the whole cube.  Wherever
 * the model is not that velocity the filter enforces a DIFFERENT angle than
 * the one asked for.  Differentiating s = cos(theta)/v at fixed s,
 *
 *     d(2theta) = -2 eps cot(theta),     eps = dv/v
 *
 * so the error is mild at the wide-angle (tomographic) end and unbounded at
 * the narrow-angle (migration) end, where cos(theta) -> 1 and the slope stops
 * being invertible at all.  Measured on a constant-opening-angle probe (the
 * only valid probe -- see THE TRAP below), the spread in the half-power
 * cutoff between a slow region and a fast one is
 *
 *     contrast   stationary   this routine
 *      10%          3.8 deg      0.1 deg
 *      50%         19.3 deg      0.9 deg
 *
 * THE METHOD (Gazdag's PSPI, applied to the cone)
 * -----------------------------------------------
 * Filter the cube once per reference velocity v_i, then blend the results
 * pointwise by the local velocity:
 *
 *     g(x) = (1-a) g_j(x) + a g_{j+1}(x),   a = (v(x)-v_j)/(v_{j+1}-v_j)
 *
 * with v_j <= v(x) <= v_{j+1}.  Naively that is nv times the work.  It is not,
 * because of:
 *
 * THE tau = 0 SHORTCUT
 * --------------------
 * lagGradientTomo uses ONLY the tau = 0 slice -- the gradient is by definition
 * the zero-lag correlation, and the lag axis exists only so the filter can see
 * the slope.  lagConeFilter nevertheless runs a full inverse 3D transform and
 * throws ~99.7% of it away.
 *
 * By the Fourier projection-slice property the single sample at lag index
 * `mid` is a weighted SUM over nu, which can be taken BEFORE the spatial
 * inverse transform.  So the nv inverse transforms are 2D, not 3D.
 *
 * MEASURED cost, cube (21, 512, 256), 8 threads, relative to lagConeFilter
 * (two runs, so the spread is visible):
 *
 *     nv = 1   0.59 / 0.62x        nv = 4   0.61 / 0.62x
 *     nv = 2   0.55 / 0.54x        nv = 8   0.73 / 0.82x
 *
 * i.e. following the velocity with 8 reference bins is still CHEAPER than the
 * ONE stationary cutoff it replaces.  The FFT-count ratio alone predicts far
 * better than this; what is left is the forward transform and the per-sample
 * cone evaluation, which neither path avoids.  Two things were measured to
 * matter and are done accordingly: |k| is hoisted out of the nu loop (it was
 * recomputed Nnu*(1+nv) times per sample, and at nv = 8 that alone pushed the
 * routine to 1.11x -- SLOWER than what it replaces), and the energy pass is
 * parallelised.
 *
 * The weighted sum is exact, not an approximation.  With the cube real and
 * stored as nu >= 0 only:
 *
 *     g[mid] = Re( IFFT2[ G[0] + (-1)^mid G[Nt/2]
 *                         + 2 sum_{n=1}^{Nt/2-1} G[n] e^{-2 pi i n mid / Nt} ] )
 *
 * The factor 2 and the outer Re() are legitimate even though IFFT2 does not
 * commute with Re() in general: the missing nu < 0 half satisfies
 * G[-n](k) = conj(G[n](-k)), and IFFT2 of conj(F(-k)) is conj(IFFT2(F)(x)),
 * so the two halves really do add to twice the real part.  The n = 0 and
 * n = Nt/2 planes are themselves transforms of real fields and need no
 * doubling.  lagConePspiSelfTest checks the whole chain, including the sign
 * convention of the backend's inverse, against lagConeFilter itself.
 *
 * THE TRAP, recorded so it is not walked into again
 * -------------------------------------------------
 * Probing an angle filter with an event of FIXED SLOPE p in a varying-v
 * medium measures nothing: p = cos(theta)/v, so a fixed p means the event's
 * opening angle genuinely varies with position, and the filter is CORRECT to
 * sort it differently.  Two such tests "found" errors that were the filter
 * working as designed.  The valid probe is a constant-OPENING-ANGLE event,
 * which is therefore CURVED,
 *
 *     g(x,tau) = w( tau - int_0^x cos(theta)/v(x') dx' ),
 *
 * and it must be scored LOCALLY -- passed/total energy in a slow window and
 * in a fast window separately.  Probes placed deep inside the pass or stop
 * band do not discriminate either (170 deg and 60 deg probes scored every
 * filter identically); the measurement has to be the half-power cutoff found
 * by sweeping the probe's angle.
 *
 * WHAT IS STILL STATIONARY
 * ------------------------
 * The blend is in the OUTPUT, so each reference field is still a global
 * convolution with a single-velocity cone.  PSPI is exact where v is locally
 * constant over the filter's footprint and interpolates between bins
 * elsewhere; it does not model the scattering a sharp velocity step imposes
 * on the filter itself.  The residual is controlled by nv -- which is a knob,
 * and is reported -- rather than by the model, which is the practical reason
 * to prefer it over stretching the tau axis by v(x) (that alternative leaves
 * a residual zeta * grad(ln v) that no parameter controls).
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MS_PER_S 1000.0f


/***********************************************************************
 * nextpow2 -- Smallest power of two >= n.  Must match lag_cone_filter.c,
 * otherwise the two paths pad differently and cannot be compared.
 ***********************************************************************/
static int nextpow2(int n)
{
	int m = 1;
	while (m < n) m <<= 1;
	return m;
}


/***********************************************************************
 * pspiBins -- the nv reference velocities and the cone slopes at each.
 *
 * Factored out ON PURPOSE: lagConePspiCutoffTest measures the angle this
 * filter actually delivers, and that measurement is worthless if it builds
 * the bins its own way.  One definition, used by both.
 *
 * nv = 1 is the stationary path and takes s_pass / s_stop verbatim; the
 * velocity never enters, which is exactly the limitation being measured.
 ***********************************************************************/
static void pspiBins(const lagConePar *p, float vmin, float vmax, int nv,
                     float *vref, float *sp, float *ss)
{
	int iv;

	if (nv == 1) {
		if (!(p->s_stop > p->s_pass))
			verr("lagCone PSPI: need s_stop > s_pass, got %g and %g ms/m.",
			     p->s_pass, p->s_stop);
		vref[0] = 0.0f;
		sp[0]   = p->s_pass;
		ss[0]   = p->s_stop;
		return;
	}
	if (!(p->theta_pass > 0.0f && p->theta_stop > 0.0f))
		verr("lagCone PSPI: nv=%d needs theta_pass/theta_stop (deg); a per-bin "
		     "slope cannot be rebuilt from s_pass/s_stop, which already have "
		     "one velocity baked in.", nv);
	if (!(p->theta_stop < p->theta_pass))
		verr("lagCone PSPI: need theta_stop < theta_pass (a WIDER angle is a "
		     "SMALLER slope), got %g and %g deg.", p->theta_pass, p->theta_stop);

	for (iv = 0; iv < nv; iv++) {
		vref[iv] = vmin + (vmax - vmin) * (float)iv / (float)(nv - 1);
		sp[iv]   = lagConeAngleToSlope(p->theta_pass, vref[iv]);
		ss[iv]   = lagConeAngleToSlope(p->theta_stop, vref[iv]);
	}
}


/***********************************************************************
 * pspiBlendWeight -- which two bins a velocity falls between, and the weight.
 *
 * Clamps outside [vmin, vmax] to the end bins rather than extrapolating; the
 * bins are built from the model's own min/max so this only fires on round-off.
 ***********************************************************************/
static void pspiBlendWeight(float v, float vmin, float dv, int nv,
                            int *jout, float *aout)
{
	float t = (v - vmin) / dv;
	int   j = (int)t;

	if (j < 0)      { j = 0;      t = 0.0f; }
	if (j > nv - 2) { j = nv - 2; t = (float)(nv - 1); }
	*jout = j;
	*aout = t - (float)j;
}


/***********************************************************************
 * lagConePspiNvRequired -- how many reference bins this cone actually needs.
 *
 * The cone's transition band is a fixed FRACTIONAL width in slope,
 * [cos(theta_pass/2), cos(theta_stop/2)] * 1000/v, while the bins are spaced
 * ds/s = dv/v.  Once the spacing exceeds the band's fractional width the two
 * bracketing bins have DISJOINT bands: the blend then has no 1/2 crossing,
 * only a 1/2 plateau, and the delivered cutoff jumps discontinuously as the
 * blend weight crosses 1/2.  No gap requires
 *
 *     dv/v  <  r,   r = [cos(theta_stop/2) - cos(theta_pass/2)]
 *                        / cos(theta_pass/2)
 *
 * which is 0.048 at 2theta = 40/20 but 0.633 at 150/130 -- a factor 13.  This
 * is why a cone that is fine at the tomographic end can be badly wrong at the
 * migration end with the SAME nv.
 *
 * The factor 2 below is MEASURED, not derived: lagConePspiCutoffTest shows
 * that merely avoiding the gap still leaves several degrees of error, and that
 * doubling the bin count past the no-gap point brings it under one degree.
 ***********************************************************************/
int lagConePspiNvRequired(float theta_pass, float theta_stop,
                          float vmin, float vmax)
{
	double cp, cs, r, need;

	if (!(theta_pass > 0.0f && theta_stop > 0.0f && theta_stop < theta_pass))
		verr("lagConePspiNvRequired: need 0 < theta_stop < theta_pass, "
		     "got %g and %g deg.", theta_pass, theta_stop);
	if (!(vmin > 0.0f) || vmax <= vmin) return 1;

	cp = cos(0.5 * (double)theta_pass * M_PI / 180.0);
	cs = cos(0.5 * (double)theta_stop * M_PI / 180.0);
	r  = (cs - cp) / cp;

	need = 1.0 + 2.0 * ((double)vmax - vmin) / ((double)vmin * r);
	if (need > 1024.0) need = 1024.0;
	return (int)ceil(need);
}


/***********************************************************************
 * lagConePspiTau0 -- Cone filter with a v(x,z)-following cutoff, tau=0 only.
 *
 * cube   [nlag*nx*nz] input, order (lag, x, z) with z fastest.  Not modified.
 * vel    [nx*nz] interior velocity, ix*nz+iz.  NULL = stationary.
 * tau0   [nx*nz] OUTPUT, overwritten (not accumulated).
 *
 * When p->nv <= 1, or vel is NULL, or the model is homogeneous, the cutoff is
 * p->s_pass / p->s_stop and the result equals lagConeFilter's tau = 0 slice.
 * Otherwise p->theta_pass / p->theta_stop are required and the slopes are
 * rebuilt per bin as cos(theta)/v_i.
 ***********************************************************************/
void lagConePspiTau0(const float *cube, int nlag, int nx, int nz,
                     float dx, float dz, float dtau,
                     const lagConePar *p,
                     const float *vel,
                     float *tau0, lagConeStats *st, int verbose)
{
	int    ilag, ix, iz, inu, ikx, ikz, iv, nv, mid;
	int    Nt, Nx, Nz, Nnu;
	float  *wt, *wx, *wz, *tap, *kx, *kz, *kmag;
	float  *vref, *sp, *ss, *acc;
	float  vmin, vmax, cpass, cstop, scl;
	complex *G, *A;
	double e_tot = 0.0, e_nu0 = 0.0, e_pass_min = 1.0, e_pass_max = 0.0;
	double *e_pass;
	size_t plane, i, ncube, nmodel;
	/* `pb` is what pspiBins is built from.  It is `p` except on the
	 * homogeneous fallback below, which needs to rebuild the slopes. */
	lagConePar phom;
	const lagConePar *pb = p;

	if (nlag < 1 || nx < 1 || nz < 1)
		verr("lagConePspiTau0: bad cube shape (%d, %d, %d).", nlag, nx, nz);
	if (dtau <= 0.0f || dx <= 0.0f || dz <= 0.0f)
		verr("lagConePspiTau0: dx=%g, dz=%g, dtau=%g must all be > 0.",
		     dx, dz, dtau);
	if (p->pad < 1)
		verr("lagConePspiTau0: pad=%d must be >= 1.", p->pad);
	if (tau0 == NULL)
		verr("lagConePspiTau0: `tau0` must not be NULL.");

	nmodel = (size_t)nx * nz;
	ncube  = (size_t)nlag * nmodel;
	mid    = (nlag - 1) / 2;          /* tau = 0 sits here, NOT at index 0 */

	/* ---------------- reference velocities ----------------
	 * Decide nv FIRST, because a homogeneous model must collapse to the
	 * stationary path exactly rather than divide by (vmax-vmin) = 0. */
	nv = (p->nv > 1 && vel != NULL) ? p->nv : 1;
	vmin = vmax = 0.0f;
	if (nv > 1) {
		vmin = vmax = vel[0];
		for (i = 1; i < nmodel; i++) {
			if (vel[i] < vmin) vmin = vel[i];
			if (vel[i] > vmax) vmax = vel[i];
		}
		if (vmin <= 0.0f) {
			size_t nbad = 0;
			for (i = 0; i < nmodel; i++) if (vel[i] <= 0.0f) nbad++;
			verr("lagConePspiTau0: the cone cutoff is cos(theta)/v, which is "
			     "undefined where v = 0, and %zu of %zu points (%.2f%%) are "
			     "non-positive.  This is a FLUID in the Vs cube: there is no "
			     "shear mode there, so there is no S opening angle to cut on.  "
			     "Either restrict the model to the solid or run lag_nv=1.",
			     nbad, nmodel, 100.0 * (double)nbad / (double)nmodel);
		}
		if (vmax - vmin <= 1e-6f * vmax) {
			if (verbose)
				vmess("lagConePspiTau0: model is homogeneous (%g m/s); "
				      "falling back to the stationary cutoff.", vmin);
			nv = 1;
			/* ...but NOT to the default slopes.  A caller that asked for
			 * nv>1 is allowed to give the cone as ANGLES ONLY (lag_v= is
			 * optional there, because the model supplies the velocity), in
			 * which case s_pass/s_stop were never set and still hold
			 * lagConeDefaults' 0.20/0.30 ms/m.  Dropping to nv=1 would then
			 * cut at a cone nobody asked for, with nothing in the log to say
			 * so.  The model has exactly one velocity here, so the angle maps
			 * to a slope unambiguously. */
			if (p->theta_pass > 0.0f && p->theta_stop > 0.0f) {
				phom        = *p;
				phom.s_pass = lagConeAngleToSlope(p->theta_pass, vmin);
				phom.s_stop = lagConeAngleToSlope(p->theta_stop, vmin);
				pb          = &phom;
			}
		}
	}

	vref = (float *)malloc((size_t)nv * sizeof(float));
	sp   = (float *)malloc((size_t)nv * sizeof(float));
	ss   = (float *)malloc((size_t)nv * sizeof(float));
	e_pass = (double *)calloc((size_t)nv, sizeof(double));
	if (!vref || !sp || !ss || !e_pass)
		verr("lagConePspiTau0: cannot allocate the %d reference bins.", nv);

	pspiBins(pb, vmin, vmax, nv, vref, sp, ss);

	/* Too few bins does not degrade gracefully: once the bracketing bins'
	 * transition bands stop overlapping the delivered cutoff becomes
	 * DISCONTINUOUS in v, and the filter quietly cuts at an angle nobody
	 * asked for.  Narrow (migration) cones need far more bins than wide
	 * (tomographic) ones -- see lagConePspiNvRequired.  Not gated on
	 * verbose: this is silent corruption otherwise. */
	if (nv > 1) {
		int nvreq = lagConePspiNvRequired(p->theta_pass, p->theta_stop,
		                                  vmin, vmax);
		if (nv < nvreq)
			vwarn("lagConePspiTau0: nv=%d is too few for a %.0f/%.0f deg cone "
			      "over %.0f-%.0f m/s; %d bins are needed to hold the cutoff "
			      "within ~1 deg%s.  The delivered angle will be off, and "
			      "below nv=%d it is also discontinuous in v.",
			      nv, p->theta_pass, p->theta_stop, vmin, vmax, nvreq,
			      (nv < 1 + (int)ceil(0.5*(nvreq-1)))
			          ? " and to keep the bins' bands overlapping" : "",
			      1 + (int)ceil(0.5*(nvreq - 1)));
	}
	/* cos(theta) form, so the per-sample test becomes v*u vs cos(theta) and
	 * the whole-plane early-outs below are just two comparisons. */
	cpass = (nv > 1) ? cosf((float)(0.5*p->theta_pass*M_PI/180.0)) : 0.0f;
	cstop = (nv > 1) ? cosf((float)(0.5*p->theta_stop*M_PI/180.0)) : 0.0f;

	/* ---------------- edge taper (identical to lagConeFilter) ---------- */
	wt = (float *)malloc((size_t)nlag * sizeof(float));
	wx = (float *)malloc((size_t)nx   * sizeof(float));
	wz = (float *)malloc((size_t)nz   * sizeof(float));
	tap = (float *)malloc(ncube * sizeof(float));
	if (!wt || !wx || !wz || !tap)
		verr("lagConePspiTau0: cannot allocate the tapered cube.");
	lagConeTaper1d(wt, nlag, p->tap_tau);
	lagConeTaper1d(wx, nx,   p->tap_x);
	lagConeTaper1d(wz, nz,   p->tap_z);

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
	A = (complex *)calloc((size_t)nv  * plane, sizeof(complex));
	if (!G || !A)
		verr("lagConePspiTau0: cannot allocate the %d x %d x %d spectrum plus "
		     "%d accumulator planes (%.2f GB).", Nnu, Nx, Nz, nv,
		     ((double)Nnu + nv) * plane * sizeof(complex) / 1073741824.0);

	if (verbose) {
		vmess("lagConePspiTau0: cube (%d,%d,%d) -> padded (%d,%d,%d), "
		      "pad>=%d, nu0=%s, nv=%d",
		      nlag, nx, nz, Nt, Nx, Nz, p->pad,
		      p->nu0_keep ? "keep" : "limit", nv);
		if (nv > 1)
			vmess("  cutoff FOLLOWS v: %d bins %.0f-%.0f m/s, dv=%.1f m/s "
			      "(%.1f%%); 2theta pass %.1f stop %.1f deg",
			      nv, vmin, vmax, (vmax - vmin) / (nv - 1),
			      100.0 * (vmax - vmin) / ((nv - 1) * vmin),
			      p->theta_pass, p->theta_stop);
		else
			vmess("  cutoff STATIONARY: s_pass=%g s_stop=%g ms/m",
			      sp[0], ss[0]);
	}

	/* ---------------- forward along tau ---------------- */
#pragma omp parallel private(ix, iz, ilag, inu)
	{
		float   *rbuf = (float   *)malloc((size_t)Nt  * sizeof(float));
		complex *cbuf = (complex *)malloc((size_t)Nnu * sizeof(complex));
		if (!rbuf || !cbuf) verr("lagConePspiTau0: cannot allocate tau scratch.");

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
	free(tap); free(wt); free(wx); free(wz);

	/* ---------------- forward over (x, z), per nu ---------------- */
	for (inu = 0; inu < Nnu; inu++)
		cc2dfft(G + (size_t)inu * plane, Nz, Nx, Nz, 1);

	/* ---------------- wavenumber axes (cycles/m), fftfreq order ---------- */
	kx = (float *)malloc((size_t)Nx * sizeof(float));
	kz = (float *)malloc((size_t)Nz * sizeof(float));
	if (!kx || !kz) verr("lagConePspiTau0: cannot allocate wavenumber axes.");
	for (ikx = 0; ikx < Nx; ikx++)
		kx[ikx] = (float)((ikx <= Nx/2 ? ikx : ikx - Nx) / ((double)Nx * dx));
	for (ikz = 0; ikz < Nz; ikz++)
		kz[ikz] = (float)((ikz <= Nz/2 ? ikz : ikz - Nz) / ((double)Nz * dz));

	/* |k| does not depend on nu, so hoist the sqrt out of BOTH the cone loop
	 * and the energy pass: otherwise it is evaluated Nnu*(1+nv) times per
	 * sample and, measured, that pass alone dominated the routine at nv = 8. */
	kmag = (float *)malloc(plane * sizeof(float));
	if (!kmag) verr("lagConePspiTau0: cannot allocate the |k| plane.");
	for (ikx = 0; ikx < Nx; ikx++)
		for (ikz = 0; ikz < Nz; ikz++)
			kmag[(size_t)ikx * Nz + ikz] =
				(float)sqrt((double)kx[ikx]*kx[ikx] + (double)kz[ikz]*kz[ikz]);

	/* ---------------- the cone, accumulated over nu ----------------
	 * This IS the tau = 0 slice (projection-slice); see the file header for
	 * why the doubling and the deferred Re() are exact. */
	for (inu = 0; inu < Nnu; inu++) {
		const complex *Gp = G + (size_t)inu * plane;
		double nu   = (double)inu / ((double)Nt * dtau);      /* cycles/s */
		/* MEASURED, not assumed: rc1fft(...,1) is sum_j g[j] exp(+2 pi i n j/N)
		 * (a delta at j=m transforms to exp(+2 pi i n m/N)), so the inverse
		 * weight that picks sample `mid` carries the MINUS sign. */
		double ph   = -2.0 * M_PI * (double)inu * (double)mid / (double)Nt;
		float  wr   = (float)cos(ph), wi = (float)sin(ph);
		float  mult = (inu == 0 || inu == Nt/2) ? 1.0f : 2.0f;
		double e = 0.0;

		for (i = 0; i < plane; i++)
			e += (double)Gp[i].r*Gp[i].r + (double)Gp[i].i*Gp[i].i;
		e_tot += e;
		if (inu == 0) e_nu0 = e;

		/* nu = 0: s is infinite for every k != 0, so H = 0 there unless the
		 * caller asked to keep the whole plane.  No epsilon. */
		if (inu == 0) {
			for (ikx = 0; ikx < Nx; ikx++) {
				for (ikz = 0; ikz < Nz; ikz++) {
					size_t j = (size_t)ikx * Nz + ikz;
					if (!p->nu0_keep && (kx[ikx] != 0.0f || kz[ikz] != 0.0f))
						continue;
					for (iv = 0; iv < nv; iv++) {
						A[(size_t)iv*plane + j].r += Gp[j].r;
						A[(size_t)iv*plane + j].i += Gp[j].i;
						e_pass[iv] += (double)Gp[j].r*Gp[j].r
						            + (double)Gp[j].i*Gp[j].i;
					}
				}
			}
			continue;
		}

#pragma omp parallel for private(ikx, ikz, iv) schedule(static)
		for (ikx = 0; ikx < Nx; ikx++) {
			for (ikz = 0; ikz < Nz; ikz++) {
				size_t j = (size_t)ikx * Nz + ikz;
				double u = kmag[j] / nu;            /* s/m, velocity-free */
				float  gr = Gp[j].r, gi = Gp[j].i;
				float  ar = mult * (gr*wr - gi*wi);
				float  ai = mult * (gr*wi + gi*wr);

				/* Early-outs: v_i*u vs cos(theta) is monotone in v_i, so the
				 * extreme bins decide the whole set.  With a few percent of
				 * the spectrum passing, the all-zero branch takes most of
				 * the samples and the per-bin loop is rarely entered. */
				if (nv > 1) {
					if (vmin * u >= cstop) continue;             /* all H=0 */
					if (vmax * u <= cpass) {                     /* all H=1 */
						for (iv = 0; iv < nv; iv++) {
							A[(size_t)iv*plane + j].r += ar;
							A[(size_t)iv*plane + j].i += ai;
						}
						continue;
					}
				}
				for (iv = 0; iv < nv; iv++) {
					float H = lagConeSlopeResponse((float)(MS_PER_S * u),
					                               sp[iv], ss[iv]);
					if (H == 0.0f) continue;
					A[(size_t)iv*plane + j].r += H * ar;
					A[(size_t)iv*plane + j].i += H * ai;
				}
			}
		}

		/* Energy bookkeeping is a separate pass: folding it into the loop
		 * above would need an omp reduction over an array and would change
		 * the branch structure the early-outs rely on.  It is parallelised
		 * one bin at a time, which needs only a scalar reduction. */
		for (iv = 0; iv < nv; iv++) {
			double ev = 0.0;
			float  spv = sp[iv], ssv = ss[iv];
#pragma omp parallel for private(ikx) reduction(+:ev) schedule(static)
			for (ikx = 0; ikx < Nx; ikx++) {
				int ikzl;
				for (ikzl = 0; ikzl < Nz; ikzl++) {
					size_t j = (size_t)ikx * Nz + ikzl;
					float H = lagConeSlopeResponse(
					              (float)(MS_PER_S * kmag[j] / nu), spv, ssv);
					if (H == 0.0f) continue;
					ev += (double)H*H*((double)Gp[j].r*Gp[j].r
					                 + (double)Gp[j].i*Gp[j].i);
				}
			}
			e_pass[iv] += ev;
		}
	}

	free(G); free(kx); free(kz); free(kmag);

	/* ---------------- one 2D inverse per reference velocity ---------- */
	for (iv = 0; iv < nv; iv++)
		cc2dfft(A + (size_t)iv * plane, Nz, Nx, Nz, -1);

	/* FFTlib is unscaled: undo Nt*Nx*Nz once, exactly as lagConeFilter does. */
	scl = (float)(1.0 / ((double)Nt * (double)Nx * (double)Nz));

	/* ---------------- crop and blend ---------------- */
	acc = (float *)malloc((size_t)nv * nmodel * sizeof(float));
	if (!acc) verr("lagConePspiTau0: cannot allocate the blend buffer.");
	for (iv = 0; iv < nv; iv++) {
		const complex *Ap = A + (size_t)iv * plane;
		float *dst = acc + (size_t)iv * nmodel;
		for (ix = 0; ix < nx; ix++)
			for (iz = 0; iz < nz; iz++)
				dst[(size_t)ix*nz + iz] = scl * Ap[(size_t)ix*Nz + iz].r;
	}

	if (nv == 1) {
		memcpy(tau0, acc, nmodel * sizeof(float));
	} else {
		float dv = (vmax - vmin) / (float)(nv - 1);
#pragma omp parallel for private(i) schedule(static)
		for (i = 0; i < nmodel; i++) {
			int   j;
			float a;
			pspiBlendWeight(vel[i], vmin, dv, nv, &j, &a);
			tau0[i] = (1.0f - a) * acc[(size_t)j*nmodel + i]
			        +         a  * acc[(size_t)(j+1)*nmodel + i];
		}
	}
	free(acc); free(A);

	/* ---------------- stats ---------------- */
	for (iv = 0; iv < nv; iv++) {
		double f = (e_tot > 0.0) ? e_pass[iv] / e_tot : 0.0;
		if (f < e_pass_min) e_pass_min = f;
		if (f > e_pass_max) e_pass_max = f;
	}
	if (st != NULL) {
		st->nlag = nlag; st->nx = nx; st->nz = nz;
		st->Nt = Nt; st->Nx = Nx; st->Nz = Nz;
		st->nu_max = (float)((Nt/2) / ((double)Nt * dtau));
		st->energy_frac_nu0 = (e_tot > 0.0) ? (float)(e_nu0 / e_tot) : 0.0f;
		st->energy_frac_passed = (float)(0.5 * (e_pass_min + e_pass_max));
	}
	if (verbose) {
		if (nv > 1)
			vmess("  nu up to %.2f Hz; nu=0 plane %.3f%% of the total; "
			      "spectral energy kept %.3f-%.3f%% across the %d bins",
			      (Nt/2) / ((double)Nt * dtau),
			      100.0 * ((e_tot > 0.0) ? e_nu0 / e_tot : 0.0),
			      100.0 * e_pass_min, 100.0 * e_pass_max, nv);
		else
			vmess("  nu up to %.2f Hz; nu=0 plane %.3f%% of the total; "
			      "spectral energy kept %.3f%%",
			      (Nt/2) / ((double)Nt * dtau),
			      100.0 * ((e_tot > 0.0) ? e_nu0 / e_tot : 0.0),
			      100.0 * e_pass_max);
	}

	free(vref); free(sp); free(ss); free(e_pass);
}


/***********************************************************************
 * lagConePspiSelfTest -- The tau = 0 shortcut against the full 3D inverse.
 *
 * The projection-slice sum replaces lagConeFilter's inverse transform along
 * tau, and it carries a phase weight exp(-2 pi i n mid / Nt) whose SIGN depends
 * on the backend's forward/inverse convention -- the one thing in this file
 * that cannot be settled by reading it.  Running both paths on the same random
 * cube with nv = 1 settles it: they must agree to round-off.
 *
 * Returns max|pspi - cone| / rms(cone) over the tau = 0 slice.
 ***********************************************************************/
float lagConePspiSelfTest(int verbose)
{
	const int nlag = 9, nx = 12, nz = 10;
	const float dx = 6.0f, dz = 6.0f, dtau = 0.002f;
	int i, mid = (nlag - 1) / 2;
	size_t nmodel = (size_t)nx * nz, ncube = (size_t)nlag * nmodel;
	float *cube, *passed, *t0;
	double num = 0.0, den = 0.0;
	float err;
	lagConePar p;

	lagConeDefaults(&p);
	p.s_pass = 0.15f;     /* a real cut, so the test exercises H, not H == 1 */
	p.s_stop = 0.30f;
	p.pad    = 2;
	p.nv     = 1;

	cube   = (float *)malloc(ncube * sizeof(float));
	passed = (float *)malloc(ncube * sizeof(float));
	t0     = (float *)malloc(nmodel * sizeof(float));
	if (!cube || !passed || !t0)
		verr("lagConePspiSelfTest: cannot allocate.");

	srand(20261111);
	for (i = 0; i < (int)ncube; i++)
		cube[i] = (float)rand() / (float)RAND_MAX - 0.5f;

	lagConeFilter(cube, nlag, nx, nz, dx, dz, dtau, &p, passed, NULL, NULL, 0);
	lagConePspiTau0(cube, nlag, nx, nz, dx, dz, dtau, &p, NULL, t0, NULL, 0);

	for (i = 0; i < (int)nmodel; i++) {
		double a = passed[(size_t)mid * nmodel + i];
		double d = t0[i] - a;
		den += a * a;
		if (fabs(d) > num) num = fabs(d);
	}
	den = sqrt(den / (double)nmodel);
	err = (den > 0.0) ? (float)(num / den) : (float)num;

	if (verbose)
		vmess("lagConePspiSelfTest: tau=0 slice vs the full 3D inverse, "
		      "max|diff|/rms = %.3e  (%s)", err,
		      (err < 1e-5f) ? "OK" : "FAILED");

	free(cube); free(passed); free(t0);
	return err;
}


/***********************************************************************
 * lagConePspiVaryTest -- Does the cutoff REALLY follow v(x,z)?
 *
 * The selftest above only proves nv = 1 reproduces the old filter.  It says
 * nothing about the blend, the per-bin slopes or the early-outs, all of which
 * are new.  This one measures them, and it needs no probe design at all.
 *
 * Build a model that is EXACTLY vmin over a left block, EXACTLY vmax over a
 * right block, and ramps in between.  Then by construction of the blend the
 * nv > 1 result must coincide, sample for sample, with
 *
 *     the nv = 1 run at v = vmin   on the left block   (alpha = 0, bin 0)
 *     the nv = 1 run at v = vmax   on the right block  (alpha = 1, bin nv-1)
 *
 * Note this is EXACT, not approximate: each reference field is a global
 * convolution contaminated by the whole model, but the blend picks one of them
 * outright where the local velocity sits on a bin.  So any error in the bin
 * index, the interpolation weight, the per-bin cos(theta)/v_i, or the
 * monotonicity assumed by the all-pass/all-stop early-outs shows up here as a
 * finite number instead of round-off.
 *
 * It also reports the spread between the two reference fields, because if
 * those two were themselves identical the test would pass vacuously.
 *
 * Returns max(|pspi - ref|)/rms(ref) over the two constant-velocity blocks.
 ***********************************************************************/
float lagConePspiVaryTest(int verbose)
{
	const int   nlag = 15, nx = 96, nz = 32;
	const float dx = 6.0f, dz = 6.0f, dtau = 0.004f;
	const float vmin = 2000.0f, vmax = 3000.0f;
	const int   ix_lo = 24, ix_hi = 72;   /* blocks end / start here */
	int   i, ix, iz;
	size_t nmodel = (size_t)nx * nz, ncube = (size_t)nlag * nmodel;
	float *cube, *vel, *slow, *fast, *pspi;
	double num = 0.0, den = 0.0, dnum = 0.0;
	float err, spread;
	lagConePar p;

	lagConeDefaults(&p);
	p.theta_pass = 150.0f;    /* the production cone: wide = tomographic */
	p.theta_stop = 130.0f;
	p.pad        = 2;

	cube = (float *)malloc(ncube * sizeof(float));
	vel  = (float *)malloc(nmodel * sizeof(float));
	slow = (float *)malloc(nmodel * sizeof(float));
	fast = (float *)malloc(nmodel * sizeof(float));
	pspi = (float *)malloc(nmodel * sizeof(float));
	if (!cube || !vel || !slow || !fast || !pspi)
		verr("lagConePspiVaryTest: cannot allocate.");

	srand(20261112);
	for (i = 0; i < (int)ncube; i++)
		cube[i] = (float)rand() / (float)RAND_MAX - 0.5f;

	for (ix = 0; ix < nx; ix++) {
		float v;
		if      (ix <= ix_lo) v = vmin;
		else if (ix >= ix_hi) v = vmax;
		else {
			float u = (float)(ix - ix_lo) / (float)(ix_hi - ix_lo);
			v = vmin + (vmax - vmin) * u * u * (3.0f - 2.0f * u);
		}
		for (iz = 0; iz < nz; iz++) vel[(size_t)ix * nz + iz] = v;
	}

	/* the two end-member stationary runs */
	p.nv = 1;
	p.s_pass = lagConeAngleToSlope(p.theta_pass, vmin);
	p.s_stop = lagConeAngleToSlope(p.theta_stop, vmin);
	lagConePspiTau0(cube, nlag, nx, nz, dx, dz, dtau, &p, NULL, slow, NULL, 0);
	p.s_pass = lagConeAngleToSlope(p.theta_pass, vmax);
	p.s_stop = lagConeAngleToSlope(p.theta_stop, vmax);
	lagConePspiTau0(cube, nlag, nx, nz, dx, dz, dtau, &p, NULL, fast, NULL, 0);

	/* the velocity-following run */
	p.nv = 4;
	lagConePspiTau0(cube, nlag, nx, nz, dx, dz, dtau, &p, vel, pspi, NULL, 0);

	for (ix = 0; ix < nx; ix++) {
		const float *ref;
		if      (ix <= ix_lo) ref = slow;
		else if (ix >= ix_hi) ref = fast;
		else continue;
		for (iz = 0; iz < nz; iz++) {
			size_t j = (size_t)ix * nz + iz;
			double d = pspi[j] - ref[j];
			double s = slow[j] - fast[j];
			den += (double)ref[j] * ref[j];
			if (fabs(d) > num)  num  = fabs(d);
			if (fabs(s) > dnum) dnum = fabs(s);
		}
	}
	den = sqrt(den / (double)((ix_lo + 1 + nx - ix_hi) * nz));
	err    = (den > 0.0) ? (float)(num  / den) : (float)num;
	spread = (den > 0.0) ? (float)(dnum / den) : (float)dnum;

	if (verbose) {
		vmess("lagConePspiVaryTest: v = %.0f -> %.0f m/s, 2theta %.0f/%.0f deg, "
		      "nv=4", vmin, vmax, p.theta_pass, p.theta_stop);
		vmess("  the two stationary cutoffs differ by %.3f rms -- the test is "
		      "not vacuous", spread);
		vmess("  nv=4 vs the matching end member on the constant blocks, "
		      "max|diff|/rms = %.3e  (%s)", err,
		      (err < 1e-5f) ? "OK" : "FAILED");
	}

	free(cube); free(vel); free(slow); free(fast); free(pspi);
	return err;
}


/***********************************************************************
 * pspiHalfSlopeRange -- where the blended response crosses 1/2.
 *
 * The effective response at a point is the blend of the two bracketing bins'
 * responses, R(s) = (1-a) H(s; spj, ssj) + a H(s; spk, ssk).  It is
 * non-increasing, so the 1/2 crossing is found by bisection -- but it can be a
 * PLATEAU rather than a point: if the two bins' transition bands do not
 * overlap and a = 1/2, then R = 1/2 identically between ssk and spj.  That is
 * a genuine ambiguity in the cutoff, not a numerical artifact, and it is the
 * signature of nv being too small for the velocity range.  So both edges are
 * returned and the caller reports the width.
 ***********************************************************************/
static void pspiHalfSlopeRange(float spj, float ssj, float spk, float ssk,
                               float a, double *slo, double *shi)
{
	double lo, hi, mid;
	int it;
	double top = (ssj > ssk) ? (double)ssj : (double)ssk;

	/* lower edge: the largest s still strictly above 1/2 */
	lo = 0.0; hi = top;
	for (it = 0; it < 60; it++) {
		mid = 0.5 * (lo + hi);
		if ((1.0-a)*lagConeSlopeResponse((float)mid, spj, ssj)
		  +       a *lagConeSlopeResponse((float)mid, spk, ssk) > 0.5)
			lo = mid;
		else
			hi = mid;
	}
	*slo = lo;

	/* upper edge: the smallest s strictly below 1/2 */
	lo = 0.0; hi = top;
	for (it = 0; it < 60; it++) {
		mid = 0.5 * (lo + hi);
		if ((1.0-a)*lagConeSlopeResponse((float)mid, spj, ssj)
		  +       a *lagConeSlopeResponse((float)mid, spk, ssk) >= 0.5)
			lo = mid;
		else
			hi = mid;
	}
	*shi = hi;
}


/***********************************************************************
 * pspiEffAngle -- slope [ms/m] at velocity v back to a full opening angle.
 *
 * Inverts |p| = cos(theta)/v.  Returns a NEGATIVE value when the slope is
 * UNREACHABLE at that velocity, i.e. cos(theta) > 1: the stationary filter is
 * then asking for an angle that does not exist in that part of the model, and
 * the cut it applies there has no angular meaning at all.
 ***********************************************************************/
static double pspiEffAngle(double s, float v)
{
	double c = s * (double)v / (double)MS_PER_S;
	if (c >= 1.0) return -1.0;
	if (c <= 0.0) return 180.0;
	return 2.0 * acos(c) * 180.0 / M_PI;
}


/***********************************************************************
 * lagConePspiCutoffTest -- WHICH ANGLE does the filter actually cut at?
 *
 * lagConePspiVaryTest proves the blend is wired correctly; it does not say in
 * DEGREES how wrong the cutoff is, and every number measured so far was at the
 * wide/tomographic end (2theta = 150/130).  The claim that the complement is a
 * migration filter rests on the NARROW end, where
 *
 *     d(2theta) = -2 eps cot(theta)
 *
 * is unbounded -- so that end has to be measured, not inferred.
 *
 * NO PROBE, NO FFT.  A wave-packet probe was sized and rejected: the tau
 * window |tau| <= nlag*dtau/2 and the aperture bound together force a ~15%
 * spectral spread that smears the very cutoff being measured, needing
 * nlag = 81 and nx ~ 768 to beat.  The response is known in closed form
 * instead, so the cutoff can be solved for exactly:
 *
 *     R_x(s) = (1-a) H(s; sp_j, ss_j) + a H(s; sp_{j+1}, ss_{j+1}) = 1/2
 *     2theta_eff(x) = 2 arccos( s* v(x) / 1000 )
 *
 * and compared against what was asked for.  The target is the half-response
 * angle of the design cone,
 *
 *     2theta_design = 2 arccos( [cos(theta_pass) + cos(theta_stop)] / 2 ),
 *
 * which is VELOCITY-FREE: a cutoff that truly followed v would deliver it at
 * every point, so any deviation is exactly the error in question.  This counts
 * BOTH error sources -- the stationary conversion AND the bin interpolation --
 * and it uses pspiBins / pspiBlendWeight, the same two routines the production
 * filter uses, so the two cannot drift apart.
 *
 * The nv = 1 reference is given the BEST stationary choice, v = (vmin+vmax)/2,
 * which biases the comparison in its favour.
 *
 * WHAT IT FOUND, and why nv = 4 is NOT a universal default
 * --------------------------------------------------------
 * The narrow end is harder for a reason that is structural, not numerical.
 * The transition band is a FIXED FRACTIONAL width in slope,
 *
 *     [s_pass, s_stop] = [cos(theta_pass/2), cos(theta_stop/2)] * 1000/v,
 *
 * and at the narrow end both cosines approach 1, so the band becomes
 * relatively thin.  The bins are spaced ds/s = dv/v.  When the spacing exceeds
 * the band's fractional width the two bracketing bins have DISJOINT bands, and
 * the blend then has no 1/2 crossing at all -- it has a 1/2 PLATEAU, and the
 * delivered cutoff jumps discontinuously as alpha crosses 1/2.  The condition
 * for the bands to still overlap is exact:
 *
 *     dv/v  <  [cos(theta_stop/2) - cos(theta_pass/2)] / cos(theta_pass/2)
 *
 *            = 0.048 for 2theta = 40/20   (narrow, migration)
 *            = 0.633 for 2theta = 150/130 (wide, tomographic)
 *
 * -- a factor of THIRTEEN between the two ends.  That is the whole reason
 * every earlier measurement, all taken at 150/130, made nv = 4 look generous.
 *
 * MEASURED on top of that derivation: overlapping bands are necessary but not
 * sufficient; twice the no-gap nv brings the error under a degree, and that
 * empirical factor 2 is what nvRequired() returns and what this test scores.
 *
 * Returns the worst error in DEGREES at the nv the rule prescribes.  Note the
 * unit: this is not a relative residual like the other two self-tests.
 ***********************************************************************/
float lagConePspiCutoffTest(int verbose)
{
	const float cones[3][2] = { {40.0f, 20.0f},     /* narrow = migration  */
	                            {90.0f, 70.0f},
	                            {150.0f, 130.0f} }; /* wide = tomographic  */
	const float contrast[2] = { 0.10f, 0.50f };
	const int   nvset[6] = { 1, 2, 4, 8, 16, 32 };
	const float vmin = 2000.0f;
	const int   nvel = 1001;
	int   ic, jc, iset, i;
	float worst = 0.0f;
	lagConePar p;

	lagConeDefaults(&p);

	if (verbose) {
		vmess("lagConePspiCutoffTest: the angle the filter DELIVERS, solved in "
		      "closed form (no probe, no FFT).");
		vmess("  'gap' = the two bracketing bins' transition bands no longer "
		      "overlap, so the cutoff is discontinuous in v.");
	}

	for (ic = 0; ic < 3; ic++) {
		double cp   = cos(0.5*cones[ic][0]*M_PI/180.0);
		double cs   = cos(0.5*cones[ic][1]*M_PI/180.0);
		double cdes = 0.5 * (cp + cs);
		double tdes = 2.0 * acos(cdes) * 180.0 / M_PI;

		p.theta_pass = cones[ic][0];
		p.theta_stop = cones[ic][1];

		if (verbose)
			vmess("  design cone 2theta %.0f/%.0f deg -> half-response at "
			      "%.2f deg; bands overlap while dv/v < %.4f",
			      cones[ic][0], cones[ic][1], tdes, (cs - cp)/cp);

		for (jc = 0; jc < 2; jc++) {
			float  vmax  = vmin * (1.0f + contrast[jc]);
			int    nvreq = lagConePspiNvRequired(p.theta_pass, p.theta_stop,
			                                     vmin, vmax);
			int    nvuse = nvset[5];
			double errat = 0.0;

			for (iset = 0; iset < 6; iset++)
				if (nvset[iset] >= nvreq) { nvuse = nvset[iset]; break; }

			for (iset = 0; iset < 6; iset++) {
				int    nv = nvset[iset];
				float  vref[32], sp[32], ss[32];
				float  dv = (nv > 1) ? (vmax - vmin) / (float)(nv - 1) : 0.0f;
				double maxerr = 0.0, maxjump = 0.0, prev = -1.0;
				int    nbad = 0, gap = 0;

				/* the stationary path gets the centred velocity, its best case */
				p.s_pass = lagConeAngleToSlope(p.theta_pass, 0.5f*(vmin+vmax));
				p.s_stop = lagConeAngleToSlope(p.theta_stop, 0.5f*(vmin+vmax));
				pspiBins(&p, vmin, vmax, nv, vref, sp, ss);

				if (nv > 1 && ss[1] < sp[0]) gap = 1;

				for (i = 0; i < nvel; i++) {
					float  v = vmin + (vmax - vmin) * (float)i / (float)(nvel-1);
					int    j = 0;
					float  a = 0.0f;
					double slo, shi, ang;

					if (nv > 1) pspiBlendWeight(v, vmin, dv, nv, &j, &a);
					pspiHalfSlopeRange(sp[j], ss[j],
					                   sp[nv > 1 ? j+1 : 0],
					                   ss[nv > 1 ? j+1 : 0],
					                   a, &slo, &shi);

					ang = pspiEffAngle(0.5*(slo + shi), v);
					if (ang < 0.0) { nbad++; prev = -1.0; continue; }

					if (fabs(ang - tdes) > maxerr) maxerr = fabs(ang - tdes);
					if (prev >= 0.0 && fabs(ang - prev) > maxjump)
						maxjump = fabs(ang - prev);
					prev = ang;
				}

				if (nv == nvuse) errat = maxerr;

				if (verbose) {
					char note[64];
					if (nbad > 0)
						snprintf(note, sizeof(note),
						         "UNREACHABLE at %d/%d v", nbad, nvel);
					else if (gap)
						snprintf(note, sizeof(note),
						         "gap, cutoff jumps %.1f deg", maxjump);
					else
						snprintf(note, sizeof(note), "-");
					vmess("    v %.0f-%.0f (%2.0f%%)  nv=%-2d  max err %6.2f deg"
					      "   %s%s", vmin, vmax, 100.0*contrast[jc], nv,
					      maxerr, note, (nv == nvuse) ? "   <- rule" : "");
				}
			}

			if (verbose)
				vmess("    rule: nv >= %d (no-gap x2) -> used nv=%d, "
				      "err %.2f deg", nvreq, nvuse, errat);
			if (errat > worst) worst = (float)errat;
		}
	}

	if (verbose)
		vmess("lagConePspiCutoffTest: worst error at the prescribed nv "
		      "%.3f deg  (%s)", worst, (worst < 1.0f) ? "OK" : "FAILED");

	return worst;
}
