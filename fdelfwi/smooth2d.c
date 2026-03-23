/*--------------------------------------------------------------------
 * smooth2d.c -- 2D Gaussian smoothing for gradients and preconditioners.
 *
 * Applies a separable 2D Gaussian filter to a flat array stored in
 * column-major order (nz samples per column, nx columns).
 *
 * The smoothing length is specified in grid points (half-width of the
 * Gaussian kernel). The Gaussian is truncated at 3σ.
 *
 * Usage:
 *   smooth2d(array, nx, nz, smooth_x, smooth_z)
 *
 * where smooth_x and smooth_z are the half-widths in grid points.
 * A value of 0 disables smoothing in that direction.
 *
 * Reference: Liu et al. (2022), Section 2.2 — "smoothing and damping
 * are required to avoid numerical instabilities."
 *--------------------------------------------------------------------*/

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/*--------------------------------------------------------------------
 * smooth1d -- 1D Gaussian convolution along a contiguous array.
 *
 * Convolves n samples (stride 1) with a Gaussian of half-width hw.
 * The kernel is truncated at 3σ where σ = hw / 2.
 * Boundary: zero-padding (values outside the array are 0).
 *--------------------------------------------------------------------*/
static void smooth1d(float *data, int n, int hw)
{
	if (hw <= 0 || n <= 1) return;

	float sigma = (float)hw / 2.0f;
	int kw = (int)(3.0f * sigma + 0.5f);  /* kernel half-width (3σ) */
	if (kw < 1) kw = 1;
	if (kw > n/2) kw = n/2;

	int klen = 2 * kw + 1;
	float *kernel = (float *)malloc(klen * sizeof(float));
	float *tmp    = (float *)malloc(n * sizeof(float));

	/* Build normalized Gaussian kernel */
	float sum = 0.0f;
	for (int k = -kw; k <= kw; k++) {
		float arg = (float)k / sigma;
		kernel[k + kw] = expf(-0.5f * arg * arg);
		sum += kernel[k + kw];
	}
	for (int k = 0; k < klen; k++)
		kernel[k] /= sum;

	/* Convolve */
	for (int i = 0; i < n; i++) {
		float val = 0.0f;
		for (int k = -kw; k <= kw; k++) {
			int j = i + k;
			if (j >= 0 && j < n)
				val += data[j] * kernel[k + kw];
		}
		tmp[i] = val;
	}

	memcpy(data, tmp, n * sizeof(float));
	free(kernel);
	free(tmp);
}


/*--------------------------------------------------------------------
 * smooth2d -- 2D separable Gaussian smoothing.
 *
 * Smooths a flat array of size nx × nz (column-major: data[ix*nz+iz]).
 *
 * Parameters:
 *   data      - array to smooth (modified in-place)
 *   nx, nz    - grid dimensions
 *   smooth_x  - smoothing half-width in x direction (grid points)
 *   smooth_z  - smoothing half-width in z direction (grid points)
 *               0 = no smoothing in that direction
 *--------------------------------------------------------------------*/
void smooth2d(float *data, int nx, int nz, int smooth_x, int smooth_z)
{
	int ix, iz;

	if (!data || (smooth_x <= 0 && smooth_z <= 0)) return;

	/* Smooth along z (fast direction, contiguous in memory) */
	if (smooth_z > 0) {
		for (ix = 0; ix < nx; ix++)
			smooth1d(&data[ix * nz], nz, smooth_z);
	}

	/* Smooth along x (slow direction, need to extract/insert columns) */
	if (smooth_x > 0) {
		float *col = (float *)malloc(nx * sizeof(float));
		for (iz = 0; iz < nz; iz++) {
			/* Extract column */
			for (ix = 0; ix < nx; ix++)
				col[ix] = data[ix * nz + iz];
			/* Smooth */
			smooth1d(col, nx, smooth_x);
			/* Insert back */
			for (ix = 0; ix < nx; ix++)
				data[ix * nz + iz] = col[ix];
		}
		free(col);
	}
}


/*--------------------------------------------------------------------
 * smooth2d_padded -- Smooth a padded-grid array, only the interior.
 *
 * Extracts the interior (nx × nz) from a padded array (nax × naz),
 * smooths it, and writes it back. Boundary padding is not modified.
 *
 * Parameters:
 *   data          - padded array [nax * naz]
 *   nax, naz      - padded grid dimensions
 *   nx, nz        - interior grid dimensions
 *   ibndx, ibndz  - boundary offsets (interior starts at data[ibndx*naz+ibndz])
 *   smooth_x, smooth_z - smoothing half-widths in grid points
 *--------------------------------------------------------------------*/
void smooth2d_padded(float *data, int nax, int naz,
                     int nx, int nz, int ibndx, int ibndz,
                     int smooth_x, int smooth_z)
{
	int ix, iz;

	if (!data || (smooth_x <= 0 && smooth_z <= 0)) return;

	/* Extract interior */
	float *interior = (float *)malloc((size_t)nx * nz * sizeof(float));
	for (ix = 0; ix < nx; ix++)
		for (iz = 0; iz < nz; iz++)
			interior[ix * nz + iz] = data[(ix + ibndx) * naz + (iz + ibndz)];

	/* Smooth */
	smooth2d(interior, nx, nz, smooth_x, smooth_z);

	/* Write back */
	for (ix = 0; ix < nx; ix++)
		for (iz = 0; iz < nz; iz++)
			data[(ix + ibndx) * naz + (iz + ibndz)] = interior[ix * nz + iz];

	free(interior);
}
