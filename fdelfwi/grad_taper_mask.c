#include<stdlib.h>
#include"par.h"
#include"fdelfwi.h"

/**
 * grad_taper_mask.c -- Source/receiver hole in the gradient (`grad_taper=`).
 *
 * The imaging condition is singular at the sources and receivers: the
 * wavefield amplitude diverges there, so without a taper the few grid points
 * around each source hold essentially all of |g|^2 and swamp both the display
 * and any line search.  The mask is 0 at each source/receiver position and
 * ramps parabolically (w = r^2/R^2) to 1 at `radius` grid points.
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
 * an array of ones). */
float *buildGradTaperMask(int nx, int nz,
	shotPar *shot, recPar *rec, int radius)
{
	int ix, iz, k;
	int nmodel = nx * nz;
	float r2max, *mask;

	if (radius <= 0) return NULL;

	r2max = (float)radius * (float)radius;
	mask  = (float *)malloc((size_t)nmodel * sizeof(float));
	if (!mask) verr("buildGradTaperMask: cannot allocate a %d point mask.", nmodel);
	for (k = 0; k < nmodel; k++) mask[k] = 1.0f;

	/* Parabolic hole around each source */
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
					float w = r2 / r2max;
					if (w < mask[ix * nz + iz]) mask[ix * nz + iz] = w;
				}
			}
		}
	}

	/* Parabolic hole around each receiver */
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
					float w = r2 / r2max;
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
