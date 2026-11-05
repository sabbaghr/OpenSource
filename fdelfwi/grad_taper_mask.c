#include<stdlib.h>
#include"par.h"
#include"fdelfwi.h"

/**
 * grad_taper_mask.c -- Source/receiver hole in the gradient (`grad_taper=`).
 *
 * The imaging condition is singular at the sources and receivers: the
 * wavefield amplitude diverges there, so without a taper the few grid points
 * around each source hold essentially all of |g|^2 and swamp both the display
 * and any line search.  The mask is 0 inside `core` grid points, then ramps
 * parabolically to 1 at `radius`:
 *
 *     w = 0                             r <= core
 *     w = (r^2 - core^2)/(R^2 - core^2) core < r < R
 *     w = 1                             r >= R
 *
 * WHY THE RAMP NEEDS A ZERO CORE, AND WHY A BIGGER `radius` IS NOT A
 * SUBSTITUTE.  With core=0 this is the original w = r^2/R^2, which is exactly
 * 0 only AT the center cell; one cell away it is already 1/R^2.  The source
 * singularity is far steeper than r^2, so the ramp does not remove the spike,
 * it only rescales it.  Measured on the Marmousi lag experiment (24 shots,
 * 201 receivers, both at z=20 m, dx=5 m), dividing the written gradient by
 * the known mask recovers the same raw near-source amplitude at both radii:
 *
 *     cells from source   1       2      3      4        interior ~2.6e-3
 *     raw |g_vs|          729.8   56.3   3.64   0.153
 *
 * a ~r^-6.5 decay confined to 4 cells.  Going from radius=30 to radius=100
 * moved the global max from 8.109e-01 to 7.298e-02 -- exactly (30/100)^2 --
 * at the SAME cell, and the share of ||g_vs||^2 in the top 50 m barely
 * changed, 99.83% -> 99.58%.  That is structural, not bad luck: inside the
 * old radius the two masks differ only by the constant (30/100)^2, so the
 * mask SHAPE over the shallow zone is identical and the spike keeps its
 * dominance over its own neighbours.  Enlarging R only improves shallow
 * relative to deep (beyond R, w=1 either way).  Suppressing 729.8 to the
 * interior level by R alone would need R > 854 cells = 4270 m, wider than
 * the model.  A core of a few cells kills it outright instead.
 *
 * The core applies to sources AND receivers, like the ramp -- this mask has
 * always treated the two alike, and its two consumers are only comparable if
 * it keeps doing so.
 *
 * GEOMETRIC CONSEQUENCE, INTENDED: when the receiver spacing is finer than
 * the core (dxrcv=25 m = 5 cells against core=5), the receiver cores MERGE
 * into a continuous zero band from the surface down to z_rcv + core, not a
 * row of separate holes.  core=5 at zrcv=20 m therefore mutes the top 45 m
 * outright.  That is where 99.58% of the conventional g_vs energy sat, and
 * the target of the lag experiment is the deep background (start-model error
 * centroid ~965 m) -- but no shallow structure can be recovered inside the
 * band, so do not use a core when the shallow section is the objective.
 *
 * Kept in its own file because two different consumers must use the SAME
 * mask or a comparison between them is meaningless: the conventional gradient
 * (tapered in the driver, after scaling and preconditioning) and the time-lag
 * cubes of lag_gradient.c (tapered per lag, from their own accumulators,
 * because the driver's taper never reaches them).  The tau=0 slice of the lag
 * cube is only comparable to the raw gradient when both were tapered alike.
 *
 * Positions are INTERIOR (non-padded) grid indices, and so is the mask,
 * indexed ix*nz + iz.
 */

/* Returns a freshly allocated [nx*nz] mask, or NULL when radius <= 0
 * (meaning "no taper" -- callers test for NULL rather than multiplying by
 * an array of ones).  `core` is clamped to [0, radius-1]; core=0 reproduces
 * the original w = r^2/R^2 ramp exactly. */
float *buildGradTaperMask(int nx, int nz,
	shotPar *shot, recPar *rec, int radius, int core)
{
	int ix, iz, k;
	int nmodel = nx * nz;
	float r2max, r2core, span, *mask;

	if (radius <= 0) return NULL;

	if (core < 0) core = 0;
	if (core > radius - 1) core = radius - 1;

	r2max  = (float)radius * (float)radius;
	r2core = (float)core * (float)core;
	span   = r2max - r2core;
	mask   = (float *)malloc((size_t)nmodel * sizeof(float));
	if (!mask) verr("buildGradTaperMask: cannot allocate a %d point mask.", nmodel);
	for (k = 0; k < nmodel; k++) mask[k] = 1.0f;

	/* Zero core + parabolic ramp around each source */
	for (k = 0; k < shot->n; k++) {
		int sx = shot->x[k];
		int sz = shot->z[k];
		int x0 = sx - radius; if (x0 < 0) x0 = 0;
		int x1 = sx + radius; if (x1 >= nx) x1 = nx - 1;
		int z0 = sz - radius; if (z0 < 0) z0 = 0;
		int z1 = sz + radius; if (z1 >= nz) z1 = nz - 1;
		for (ix = x0; ix <= x1; ix++) {
			for (iz = z0; iz <= z1; iz++) {
				float ddx = (float)(ix - sx);
				float ddz = (float)(iz - sz);
				float r2  = ddx * ddx + ddz * ddz;
				if (r2 < r2max) {
					float w = (r2 <= r2core) ? 0.0f : (r2 - r2core) / span;
					if (w < mask[ix * nz + iz]) mask[ix * nz + iz] = w;
				}
			}
		}
	}

	/* Zero core + parabolic ramp around each receiver */
	for (k = 0; k < rec->n; k++) {
		int rx = rec->x[k];
		int rz = rec->z[k];
		int x0 = rx - radius; if (x0 < 0) x0 = 0;
		int x1 = rx + radius; if (x1 >= nx) x1 = nx - 1;
		int z0 = rz - radius; if (z0 < 0) z0 = 0;
		int z1 = rz + radius; if (z1 >= nz) z1 = nz - 1;
		for (ix = x0; ix <= x1; ix++) {
			for (iz = z0; iz <= z1; iz++) {
				float ddx = (float)(ix - rx);
				float ddz = (float)(iz - rz);
				float r2  = ddx * ddx + ddz * ddz;
				if (r2 < r2max) {
					float w = (r2 <= r2core) ? 0.0f : (r2 - r2core) / span;
					if (w < mask[ix * nz + iz]) mask[ix * nz + iz] = w;
				}
			}
		}
	}

	return mask;
}

/* Multiply a flat [nmodel] interior-grid array by the mask in place. */
void applyGradTaperMask(float *data, const float *mask, int nmodel)
{
	int i;
	if (mask == NULL) return;
	for (i = 0; i < nmodel; i++) data[i] *= mask[i];
}
