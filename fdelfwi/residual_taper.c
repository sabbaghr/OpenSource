#include<math.h>
#include"par.h"
#include"fdelfwi.h"

/**
 * residual_taper.c -- Cosine ramp on the ends of each adjoint source trace
 *                     (`res_taper=`).
 *
 * The adjoint source is the data residual, which generally does not start or
 * end at zero: the record is a window on a longer wavefield.  Injecting a
 * step at t=0 or t=T excites the whole band of the FD grid and shows up in
 * the gradient as ringing anchored on the receivers.  Ramping `taper_len`
 * samples at each end with a raised cosine removes that without touching the
 * interior of the trace.
 *
 * A no-op when taper_len <= 0, or when the trace is too short to hold two
 * ramps without them overlapping (nt <= 2*taper_len) -- in that case tapering
 * would attenuate the entire record rather than just its edges, which is
 * silently destructive, so it is skipped instead.
 *
 * Companion to applyApertureTaper (aperture_taper.c), which tapers across
 * receivers rather than along time.
 */
void applyCosineTaper(adjSrcPar *adj, int taper_len)
{
	int isrc, it;
	float *wav, w;

	if (taper_len <= 0 || adj->nt <= 2 * taper_len) return;

	for (isrc = 0; isrc < adj->nsrc; isrc++) {
		wav = &adj->wav[isrc * adj->nt];
		for (it = 0; it < taper_len; it++) {
			w = 0.5f * (1.0f - cosf(M_PI * it / taper_len));
			wav[it]               *= w;
			wav[adj->nt - 1 - it] *= w;
		}
	}
}
