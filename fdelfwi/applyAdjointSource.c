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
 * This is the TRUE adjoint (transpose) of the receiver extraction operator:
 *   Forward receiver:  d = field[ix,iz]
 *   Adjoint injection: field[ix,iz] += residual
 *
 * ALL types use += (true adjoint sign). The gradient descent direction
 * is handled uniformly by the -= in accumGradient (adj_shot.c).
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
 * ------------------------------------------------------------------
 * RECEIVER INTERPOLATION MODE 2 (adj.int_vx / adj.int_vz == 2)
 * ------------------------------------------------------------------
 * With rec_int_vx=2 / rec_int_vz=2 the forward recording in
 * getRecTimes.c is no longer a plain grid-point read.  For vz it is
 *
 *   d_vz = 0.5 * SUM over m in {iz, iz+1} of
 *            ( vz[ix,m] - 0.5*roz[ix,m]*Dz(m) )
 *   Dz(m) = c1*( tzz[ix,m]  -tzz[ix,m-1] +txz[ix+1,m]-txz[ix,m]  )
 *         + c2*( tzz[ix,m+1]-tzz[ix,m-2] +txz[ix+2,m]-txz[ix-1,m])
 *
 * i.e. a 2-node mean onto the Txx/Tzz node (space) PLUS a half-step
 * of the velocity update that advances v by +dt/2 onto the stress
 * time level (time).  roz already carries dt/dx (readModel.c).
 *
 * This matters for any misfit that multiplies a velocity residual by
 * a stress residual at the same receiver (the Kirchhoff-Helmholtz
 * energy-flux objective), because with mode 0 the two factors sit
 * dt/2 apart in time and dz/2 apart in space.
 *
 * The transpose therefore scatters into BOTH the velocity array (the
 * v term, at the force injection point, phase 1) and the stress
 * arrays (the Dz term, at the stress injection point, phase 2).  So
 * under mode 2 a velocity-type trace is active in both phases, not
 * just phase 1.
 *
 * c1/c2 are hardcoded to the 4th-order values here because
 * getRecTimes.c hardcodes them too, for every iorder.  The transpose
 * must match the operator that was actually applied, not the one the
 * scheme order would suggest.
 *
 *   AUTHOR:
 *           Based on applySource.c by Jan Thorbecke
 *           Adapted for FWI adjoint backpropagation.
 *
 **********************************************************************/

#define BLEND_TRANSITION_DEPTH 5  /* grid points for blending transition */

/* Under mode 2 the transpose scatters each trace over a 4th-order
 * stencil, so neighbouring receivers write to the same nodes and the
 * accumulation must be atomic.  (The mode-0 path writes one node per
 * trace and is left as it was.) */
#define ADJ_ADD(a, i, v) do { _Pragma("omp atomic") (a)[i] += (v); } while (0)

int applyAdjointSource(modPar mod, adjSrcPar adj, int itime,
	float *vx, float *vz, float *tzz, float *txx, float *txz,
	float *rox, float *roz, float *mul,
	int rec_delay, int rec_skipdt, int phase, int verbose)
{
	int isrc, isam, n1;
	float src_ampl;
	const float c1 = 9.0f/8.0f, c2 = -1.0f/24.0f;

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
		/* Is THIS trace's component co-located by recording mode 2? */
		int colo     = ((stype == 6 && adj.int_vx == 2) ||
		                (stype == 7 && adj.int_vz == 2));
		/* getRecTimes.c averages txz onto the Txx/Tzz node whenever
		 * EITHER velocity flag is 2, so txz keys off both. */
		int colo_txz = (stype == 2 && sorient == 1 &&
		                (adj.int_vx == 2 || adj.int_vz == 2));

		/* Phase filtering:
		 *   phase 1 = force injection point  -> only types > 5
		 *   phase 2 = stress injection point -> types <= 5, plus the
		 *             stress half of a co-located velocity trace
		 *             (the -0.5*ro*Dsigma half-step term).
		 */
		if (phase == 1 && stype <= 5) continue;
		if (phase == 2 && stype > 5 && !colo) continue;

		src_ampl = adj.wav[isrc * adj.nt + isam];
		if (src_ampl == 0.0f) continue;

		/* ================================================ */
		/*  Force sources (phase 1, after velocity update)  */
		/* ================================================ */

		if (stype == 6) {
			if (!colo) {
				/* Fx -> inject into vx */
				vx[ix * n1 + iz] += src_ampl;
			}
			else {
				/* Transpose of the mode-2 vx recording.  readResidual
				 * registers type 6 at the vx node ix2 = ix+1, so the two
				 * averaged nodes are m = ix-1 (=ix_rec) and m = ix
				 * (=ix2); iz is already the Txx/Tzz row. */
				size_t m;
				for (m = ix - 1; m <= ix; m++) {
					if (phase == 1) {
						ADJ_ADD(vx, m * n1 + iz, 0.5f * src_ampl);
					}
					else {
						float g = -0.25f * rox[m * n1 + iz] * src_ampl;
						ADJ_ADD(txx, m * n1 + iz,        c1 * g);
						ADJ_ADD(txx, (m - 1) * n1 + iz, -c1 * g);
						ADJ_ADD(txx, (m + 1) * n1 + iz,  c2 * g);
						ADJ_ADD(txx, (m - 2) * n1 + iz, -c2 * g);
						ADJ_ADD(txz, m * n1 + iz + 1,    c1 * g);
						ADJ_ADD(txz, m * n1 + iz,       -c1 * g);
						ADJ_ADD(txz, m * n1 + iz + 2,    c2 * g);
						ADJ_ADD(txz, m * n1 + iz - 1,   -c2 * g);
					}
				}
			}
		}
		else if (stype == 7) {
			if (!colo) {
				/* Fz -> inject into vz */
				vz[ix * n1 + iz] += src_ampl;
			}
			else {
				/* Transpose of the mode-2 vz recording.  readResidual
				 * registers type 7 at the vz node iz2 = iz+1, so the two
				 * averaged nodes are m = iz-1 (=iz_rec) and m = iz
				 * (=iz2); ix is already the Txx/Tzz column. */
				size_t m;
				for (m = iz - 1; m <= iz; m++) {
					if (phase == 1) {
						ADJ_ADD(vz, ix * n1 + m, 0.5f * src_ampl);
					}
					else {
						float g = -0.25f * roz[ix * n1 + m] * src_ampl;
						ADJ_ADD(tzz, ix * n1 + m,        c1 * g);
						ADJ_ADD(tzz, ix * n1 + m - 1,   -c1 * g);
						ADJ_ADD(tzz, ix * n1 + m + 1,    c2 * g);
						ADJ_ADD(tzz, ix * n1 + m - 2,   -c2 * g);
						ADJ_ADD(txz, (ix + 1) * n1 + m,  c1 * g);
						ADJ_ADD(txz, ix * n1 + m,       -c1 * g);
						ADJ_ADD(txz, (ix + 2) * n1 + m,  c2 * g);
						ADJ_ADD(txz, (ix - 1) * n1 + m, -c2 * g);
					}
				}
			}
		}

		/* ================================================ */
		/*  Stress sources (phase 2, after stress update)   */
		/* ================================================ */

		else if (stype == 1) {
			/* P (pressure) -> inject into tzz and txx for elastic.
			 * fdelmodc uses compression-positive convention, so
			 * P = +0.5*(Txx + Tzz) for elastic recording.
			 *
			 * TRUE ADJOINT sign convention:
			 * The adjoint of recording d = P(field) is field += y.
			 * The gradient sign (descent direction) is handled uniformly
			 * by the -= in accumGradient (adj_shot.c), so all injection
			 * types use += for consistency with force types (Fx, Fz).
			 *
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

				/* Check for nearby fluid-solid interface.
				 *
				 * EXPERIMENT (2026-10-23): -DDISABLE_P_BLEND removes the
				 * blend so the injection is the exact transpose of the
				 * 0.5*(txx+tzz) recording in getRecTimes.c.  See
				 * tests/kh_misfit/ -- test_recop_dp FAILS by up to 126%
				 * wherever this blend is active. */
#ifndef DISABLE_P_BLEND
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
#endif
			}
			/* else acoustic: tzz_scale=1.0, txx_scale=0.0 (already set) */

			/* TRUE ADJOINT: positive sign for pressure injection */
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
			if (txz && colo_txz) {
				/* Transpose of the mode-2 txz recording, which averages
				 * the 4 txz nodes onto the Txx/Tzz node.  readResidual
				 * registers type 2 at (ix2, iz2) = (ix+1, iz+1), so the
				 * four nodes are (ix-1..ix, iz-1..iz) here. */
				ADJ_ADD(txz, (ix - 1) * n1 + iz - 1, 0.25f * src_ampl);
				ADJ_ADD(txz, ix * n1 + iz - 1,       0.25f * src_ampl);
				ADJ_ADD(txz, (ix - 1) * n1 + iz,     0.25f * src_ampl);
				ADJ_ADD(txz, ix * n1 + iz,           0.25f * src_ampl);
			}
			else if (txz) {
				txz[ix * n1 + iz] += src_ampl;
				if (sorient == 2) /* vertical dipole */
					txz[ix * n1 + iz + 1] -= src_ampl;
				else if (sorient == 3) /* horizontal dipole */
					txz[(ix - 1) * n1 + iz] -= src_ampl;
			}
		}
		else if (stype == 3) {
			/* Tzz -> inject into tzz only (true adjoint: positive sign) */
			tzz[ix * n1 + iz] += src_ampl;
		}
		else if (stype == 4) {
			/* Txx -> inject into txx only (true adjoint: positive sign) */
			if (txx)
				txx[ix * n1 + iz] += src_ampl;
		}

	} /* end loop over adjoint sources */

	return 0;
}
