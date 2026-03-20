/*--------------------------------------------------------------------
 * taper_source.c -- Per-shot radial source taper for FWI.
 *
 * Builds and applies an erf-based radial taper centered on a source
 * position to remove singularities from the gradient and pseudo-Hessian
 * before accumulation across shots.
 *
 * The taper mask is:
 *   w(x,z) = erf(sqrt(π) · r / radius)
 * where r = sqrt((x - sx)² + (z - sz)²) · dx is the physical distance.
 *
 * At r=0: w=0 (full suppression at source).
 * At r=radius: w ≈ 0.97.
 * Beyond radius: w → 1.
 *
 * A hard-zero zone of (2·filtsize+1)² grid points around the source
 * completely kills the singularity. The erf then provides a smooth
 * transition from zero to one at the edge of this zone.
 *
 * Reference: DENISE-Black-Edition (Koehn et al.), taper_grad_shot.c
 *--------------------------------------------------------------------*/

#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/*--------------------------------------------------------------------
 * buildSourceTaperMask -- Build a 2D erf-based radial taper mask
 *                         centered on a single source position.
 *
 * Input:
 *   nax, naz  - padded grid dimensions (columns, rows)
 *   sx, sz    - source position in padded grid indices
 *   radius    - taper radius in meters (0 = no taper, returns NULL)
 *   dx        - grid spacing in meters (assumes dx == dz)
 *   filtsize  - half-width of hard-zero zone in grid points.
 *               The source point ± filtsize is set to zero,
 *               giving a (2*filtsize+1)² zero square.
 *               Minimum 1. DENISE default: 25.
 *
 * Returns: newly allocated float[nax * naz] mask, or NULL if radius <= 0.
 *          Caller must free.
 *--------------------------------------------------------------------*/
float *buildSourceTaperMask(int nax, int naz, int sx, int sz,
                            float radius, float dx, int filtsize)
{
	int ix, iz;
	float *mask;
	float a, inv_radius;

	if (radius <= 0.0f || dx <= 0.0f) return NULL;
	if (filtsize < 1) filtsize = 1;

	mask = (float *)malloc((size_t)nax * naz * sizeof(float));
	if (!mask) return NULL;

	a = (float)sqrt(M_PI);  /* erf(sqrt(π)) ≈ 0.97 at r = radius */
	inv_radius = 1.0f / radius;

	for (ix = 0; ix < nax; ix++) {
		for (iz = 0; iz < naz; iz++) {
			float rx = (float)(ix - sx) * dx;
			float rz = (float)(iz - sz) * dx;
			float r = sqrtf(rx * rx + rz * rz);
			mask[ix * naz + iz] = erff(a * r * inv_radius);
		}
	}

	/* Hard-zero zone: circular, radius = filtsize grid points */
	{
		int i0 = sx - filtsize;
		int i1 = sx + filtsize;
		int j0 = sz - filtsize;
		int j1 = sz + filtsize;
		int i, j;
		float r2_max = (float)filtsize * (float)filtsize;
		if (i0 < 0) i0 = 0;
		if (j0 < 0) j0 = 0;
		if (i1 >= nax) i1 = nax - 1;
		if (j1 >= naz) j1 = naz - 1;
		for (i = i0; i <= i1; i++)
			for (j = j0; j <= j1; j++) {
				float di = (float)(i - sx);
				float dj = (float)(j - sz);
				if (di*di + dj*dj <= r2_max)
					mask[i * naz + j] = 0.0f;
			}
	}

	return mask;
}


/*--------------------------------------------------------------------
 * applySourceTaper -- Multiply a padded-grid array by a taper mask.
 *
 * If mask is NULL, does nothing (no-op for radius=0 case).
 *--------------------------------------------------------------------*/
void applySourceTaper(float *arr, const float *mask, size_t sizem)
{
	size_t i;
	if (!mask || !arr) return;
	for (i = 0; i < sizem; i++)
		arr[i] *= mask[i];
}
