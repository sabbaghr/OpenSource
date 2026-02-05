#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include"fdelfwi.h"

void vmess(char *fmt, ...);

/*********************************************************************
 *
 * applyAdjointSource - Inject residuals into the adjoint wavefield.
 *
 * Handles per-source types for multicomponent FWI (OBN data with
 * hydrophone, vx, and vz receivers in the same gather).
 *
 * Called twice per time step by the adjoint FD kernel:
 *   phase=1 (after velocity update):  force types (Fx=6, Fz=7)
 *   phase=2 (after stress update):    stress types (P=1, Txz=2, Tzz=3, Txx=4)
 *
 * This ensures correct injection timing in the staggered time step:
 *   velocity residuals enter at the velocity-update point,
 *   stress/pressure residuals enter at the stress-update point.
 *
 * Residuals are injected WITHOUT material property scaling.
 * This is the adjoint (transpose) of the receiver extraction operator:
 *   Forward receiver:  d = field[ix,iz]
 *   Adjoint injection: field[ix,iz] += residual
 *
 * Fluid-solid boundary handling (blended injection):
 *   For pressure sources near fluid-solid interfaces, we use a
 *   blended injection scheme that smoothly transitions from
 *   acoustic-like (Tzz only) at the interface to full elastic
 *   (0.5*Tzz + 0.5*Txx) deeper in the solid:
 *
 *   - At interface (dist=0):     Tzz=100%, Txx=0%   (acoustic)
 *   - Transition zone (dist 1-5): gradual blend
 *   - Deep solid (dist>=5):      Tzz=50%, Txx=50%  (elastic)
 *
 *   This eliminates artifacts caused by abrupt injection scheme
 *   changes at the fluid-solid boundary.
 *
 * Grid positions adj.xi[]/adj.zi[] already include staggered-grid
 * boundary offsets (set by readResidual).
 *
 *   AUTHOR:
 *           Based on applySource.c by Jan Thorbecke
 *           Adapted for FWI adjoint backpropagation.
 *
 **********************************************************************/

#define BLEND_TRANSITION_DEPTH 5  /* grid points for blending transition */

int applyAdjointSource(modPar mod, adjSrcPar adj, int itime,
	float *vx, float *vz, float *tzz, float *txx, float *txz,
	float *mul, int rec_delay, int rec_skipdt, int phase, int verbose)
{
	int isrc, isam, n1;
	float src_ampl;

	/* Check if this time step aligns with receiver sampling */
	if (itime < rec_delay) return 0;
	if ((itime - rec_delay) % rec_skipdt != 0) return 0;
	isam = (itime - rec_delay) / rec_skipdt;
	if (isam < 0 || isam >= adj.nt) return 0;

	n1 = mod.naz;

#pragma omp for private(isrc, src_ampl)
	for (isrc = 0; isrc < adj.nsrc; isrc++) {
		int stype   = adj.typ[isrc];
		int sorient = adj.orient[isrc];
		size_t ix   = adj.xi[isrc];
		size_t iz   = adj.zi[isrc];

		/* Phase filtering:
		 *   phase 1 = force injection point  -> only types > 5
		 *   phase 2 = stress injection point -> only types <= 5
		 */
		if (phase == 1 && stype <= 5) continue;
		if (phase == 2 && stype > 5)  continue;

		src_ampl = adj.wav[isrc * adj.nt + isam];
		if (src_ampl == 0.0f) continue;

		/* ================================================ */
		/*  Force sources (phase 1, after velocity update)  */
		/* ================================================ */

		if (stype == 6) {
			/* Fx -> inject into vx */
			vx[ix * n1 + iz] += src_ampl;
		}
		else if (stype == 7) {
			/* Fz -> inject into vz */
			vz[ix * n1 + iz] += src_ampl;
		}

		/* ================================================ */
		/*  Stress sources (phase 2, after stress update)   */
		/* ================================================ */

		else if (stype == 1) {
			/* P (pressure) -> inject into tzz and txx for elastic.
			 * fdelmodc uses compression-positive convention, so
			 * P = +0.5*(Txx + Tzz) for elastic recording.
			 * The adjoint of extracting P = 0.5*(Txx + Tzz) is
			 * injecting 0.5*residual into each stress component.
			 * For acoustic (ischeme <= 2): P = Tzz, inject full amplitude.
			 *
			 * Blended fluid-solid boundary handling:
			 * Find distance from fluid-solid interface by checking mul.
			 * Use smooth transition from acoustic (Tzz only) to elastic
			 * (Tzz + Txx) over BLEND_TRANSITION_DEPTH grid points. */

			float tzz_scale = 1.0f;
			float txx_scale = 0.0f;

			if (mod.ischeme > 2 && txx) {
				/* Default elastic: 0.5 into each */
				tzz_scale = 0.5f;
				txx_scale = 0.5f;

				/* Check for nearby fluid-solid interface */
				if (mul) {
					int dist_from_fluid = BLEND_TRANSITION_DEPTH + 1; /* assume deep in solid */
					int dz;

					/* Search upward for fluid (mul=0) */
					for (dz = 1; dz <= BLEND_TRANSITION_DEPTH + 1; dz++) {
						if (iz >= (size_t)dz && mul[ix * n1 + iz - dz] == 0.0f) {
							dist_from_fluid = dz - 1; /* distance from first solid point */
							break;
						}
					}

					/* Apply blending if near interface */
					if (dist_from_fluid <= BLEND_TRANSITION_DEPTH) {
						/* blend: 0 at interface -> 1 at BLEND_TRANSITION_DEPTH */
						float blend = (float)dist_from_fluid / (float)BLEND_TRANSITION_DEPTH;

						/* Smoothly transition:
						 *   At interface (blend=0): tzz=1.0, txx=0.0 (acoustic)
						 *   Deep solid (blend=1):   tzz=0.5, txx=0.5 (elastic) */
						tzz_scale = 1.0f - 0.5f * blend;
						txx_scale = 0.5f * blend;
					}
				}
			}
			/* else acoustic: tzz_scale=1.0, txx_scale=0.0 (already set) */

			if (sorient == 1) { /* monopole */
				tzz[ix * n1 + iz] += tzz_scale * src_ampl;
				if (txx && txx_scale > 0.0f)
					txx[ix * n1 + iz] += txx_scale * src_ampl;
			}
			else if (sorient == 2) { /* vertical dipole +/- */
				tzz[ix * n1 + iz]     += tzz_scale * src_ampl;
				tzz[ix * n1 + iz + 1] -= tzz_scale * src_ampl;
				if (txx && txx_scale > 0.0f) {
					txx[ix * n1 + iz]     += txx_scale * src_ampl;
					txx[ix * n1 + iz + 1] -= txx_scale * src_ampl;
				}
			}
			else if (sorient == 3) { /* horizontal dipole -/+ */
				tzz[ix * n1 + iz]         += tzz_scale * src_ampl;
				tzz[(ix - 1) * n1 + iz]   -= tzz_scale * src_ampl;
				if (txx && txx_scale > 0.0f) {
					txx[ix * n1 + iz]       += txx_scale * src_ampl;
					txx[(ix - 1) * n1 + iz] -= txx_scale * src_ampl;
				}
			}
		}
		else if (stype == 2) {
			/* Txz -> inject into txz */
			if (txz) {
				txz[ix * n1 + iz] += src_ampl;
				if (sorient == 2) /* vertical dipole */
					txz[ix * n1 + iz + 1] -= src_ampl;
				else if (sorient == 3) /* horizontal dipole */
					txz[(ix - 1) * n1 + iz] -= src_ampl;
			}
		}
		else if (stype == 3) {
			/* Tzz -> inject into tzz only */
			tzz[ix * n1 + iz] += src_ampl;
		}
		else if (stype == 4) {
			/* Txx -> inject into txx only */
			if (txx)
				txx[ix * n1 + iz] += src_ampl;
		}

	} /* end loop over adjoint sources */

	return 0;
}
