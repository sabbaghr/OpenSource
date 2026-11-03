#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"

void vmess(char *fmt, ...);
void verr(char *fmt, ...);

/*********************************************************************
 *
 * applyApertureTaper - Geometric taper on the adjoint source injection
 *                      at the ends of the receiver array.
 *
 * WHY
 * ---
 * The receiver array stops abruptly at its first and last receiver.
 * The adjoint source distribution injected along that array therefore
 * has a hard spatial truncation, which radiates diffractions from the
 * two array endpoints into the adjoint wavefield.  Those diffractions
 * cross-correlate with the forward wavefield and leave dipping streaks
 * in the gradient that have nothing to do with the model -- the usual
 * "limited aperture" artifact.
 *
 * Ramping the injection amplitude smoothly to zero over the outermost
 * `aper_taper` grid points of the array removes the truncation itself,
 * so the edge diffraction is never generated.
 *
 * This is the same idea as the absorbing taper applied to the
 * wavefield in boundaries.c, but applied to the source distribution
 * rather than to a propagating field: it is a one-off amplitude weight
 * per receiver, not a per-time-step damping zone.
 *
 * WEIGHT DEFINITION
 * -----------------
 * The weight is a function of RECEIVER POSITION, not of trace index:
 *
 *     d_i = min(x_i - x_min, x_max - x_i)        (distance to nearer end)
 *     t   = d_i / W
 *     a_i = 6t^5 - 15t^4 + 10t^3,         d_i < W     (quintic smoothstep)
 *     a_i = 1,                            d_i >= W
 *
 * SMOOTHSTEP, NOT A RAISED COSINE.  The obvious choice here is
 * 0.5*(1-cos(pi*t)), and that is what this file used first, but it is
 * only C1: at t=1, where the ramp meets the flat a=1 interior, its
 * curvature jumps from -0.5*(pi/W)^2 to 0.  A curvature discontinuity
 * in the source amplitude profile radiates, so the cosine ramp leaves a
 * weaker but still visible edge diffraction from each end of the line.
 * The quintic has zero value, first AND second derivative at both ends,
 * so it joins the interior with no curvature jump.  The recon fork hit
 * the same problem and made the same change -- see the [TAPER] comment
 * in OpenSource-recon_SL10/fdelmodc/defineRecBoundary.c:84-90, which is
 * where this form is taken from.
 *
 * This matters most for the Kirchhoff-Helmholtz adjoint source
 * (kh_misfit.c), where the whole point is that the line radiates one way
 * only: there is no spherical-spreading decay to hide a truncation
 * artifact behind.
 *
 * with W = aper_taper * dx.  Working in physical position rather than
 * trace number matters:
 *   - irregular or gapped receiver spacing is handled correctly;
 *   - for multicomponent data every component of a given receiver
 *     shares the same x and so is weighted identically, with no
 *     special-casing of the trace-major layout written by
 *     computeResidual.
 *
 * Only the horizontal extent is tapered.  For a surface or OBN line
 * that is where the aperture is truncated; a vertical array would need
 * the same treatment in z, which is deliberately not implemented
 * rather than guessed at.
 *
 * RELATION TO THE MISFIT  (read this before trusting the gradient)
 * ----------------------------------------------------------------
 * Injection is linear, so scaling the injection at receiver i by a_i is
 * mathematically identical to scaling residual trace i by a_i: this is a
 * diagonal data weight W = diag(a_i), whichever file it is applied in.
 *
 * It is applied HERE and not inside computeResidual, so the misfit J
 * does not see it.  The computed gradient is therefore
 *
 *     g = A^T W dd        while      J = 0.5 * dd^T Q dd
 *
 * i.e. g is NOT the exact gradient of the reported J (whose gradient
 * would be A^T Q dd).  This is a deliberate choice, and it matches what
 * the existing res_taper already does.  The consequences:
 *   - reported misfit values are unaffected, so results stay comparable
 *     with earlier runs;
 *   - the weight stays purely geometric, independent of the data;
 *   - but a strong taper biases the Wolfe line search, because the
 *     search direction no longer derives from the objective being
 *     measured.  Keep aper_taper modest relative to the array length,
 *     and if a run shows the line search stalling, suspect this first.
 *
 * To make it exact instead, the same a_i must multiply the residual in
 * computeResidual.c BEFORE the misfit accumulation and again after it
 * (giving J = 0.5 a^2 dd^T Q dd and psi = a^2 Q dd).
 *
 **********************************************************************/

void applyApertureTaper(adjSrcPar *adj, float dx, int aper_taper, int verbose)
{
	int    isrc;
	float  x_min, x_max, width, d, a, t;
	int    ntapered;

	if (aper_taper <= 0) return;
	if (!adj) verr("applyApertureTaper: NULL adjoint source structure.");
	if (adj->nsrc <= 0 || !adj->wav || !adj->x) return;

	/* Horizontal extent of the receiver array */
	x_min = x_max = adj->x[0];
	for (isrc = 1; isrc < adj->nsrc; isrc++) {
		if (adj->x[isrc] < x_min) x_min = adj->x[isrc];
		if (adj->x[isrc] > x_max) x_max = adj->x[isrc];
	}

	width = (float)aper_taper * dx;

	/* A taper wider than half the array would leave no un-tapered
	 * receiver at all and silently rescale the whole gather. */
	if (2.0f * width > (x_max - x_min)) {
		vmess("applyApertureTaper: aper_taper=%d (%.1f m) exceeds half the "
		      "array length (%.1f m); taper NOT applied.",
		      aper_taper, width, 0.5f * (x_max - x_min));
		return;
	}

	ntapered = 0;
	for (isrc = 0; isrc < adj->nsrc; isrc++) {
		int it;
		float *wav;

		d = adj->x[isrc] - x_min;
		if (x_max - adj->x[isrc] < d) d = x_max - adj->x[isrc];

		if (d >= width) continue;

		/* Quintic smoothstep, NOT a raised cosine.  See WEIGHT DEFINITION. */
		t = d / width;
		a = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);

		wav = &adj->wav[(size_t)isrc * adj->nt];
		for (it = 0; it < adj->nt; it++) wav[it] *= a;
		ntapered++;
	}

	if (verbose)
		vmess("applyApertureTaper: tapered %d of %d adjoint traces over "
		      "%d grid points (%.1f m) at each end of x=[%.1f, %.1f]",
		      ntapered, adj->nsrc, aper_taper, width, x_min, x_max);
}
