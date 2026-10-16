#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"bndsrc.h"

void verr(char *fmt, ...);
void vmess(char *fmt, ...);

/*********************************************************************
*
* readBoundaryWav
*
* Reads back what writeBoundaryWav wrote. Cross-checks the file's own
* point count and geometry against the caller's freshly-built rb (from
* calling defineRecBoundary with what should be the SAME rectangle
* used in the recording run) -- catches a mismatched wfl_x1..wfl_z2
* between the two runs rather than silently misapplying the data.
* Allocates bs_vx..bs_txz (caller frees).
*
**********************************************************************/

int readBoundaryWav(char *file, bndSrcPar rb, int nt,
	float **bs_vx, float **bs_vz, float **bs_txx, float **bs_tzz, float **bs_txz)
{
	FILE *fp;
	int n_file, nt_file, i;
	float *xr, *zr, *nx, *nz;
	size_t nb;

	fp = fopen(file,"r");
	if (fp==NULL) verr("readBoundaryWav: cannot open %s for reading", file);

	fread(&n_file, sizeof(int), 1, fp);
	fread(&nt_file, sizeof(int), 1, fp);
	if (n_file!=rb.n) verr("readBoundaryWav: %s has %d boundary points, but wfl_x1..wfl_z2 for this run produce %d -- rectangle must match the recording run", file, n_file, rb.n);
	if (nt_file!=nt) verr("readBoundaryWav: %s has %d time samples, but this run has nt=%d", file, nt_file, nt);

	xr = (float *)malloc(rb.n*sizeof(float));
	zr = (float *)malloc(rb.n*sizeof(float));
	nx = (float *)malloc(rb.n*sizeof(float));
	nz = (float *)malloc(rb.n*sizeof(float));
	fread(xr, sizeof(float), rb.n, fp);
	fread(zr, sizeof(float), rb.n, fp);
	fread(nx, sizeof(float), rb.n, fp);
	fread(nz, sizeof(float), rb.n, fp);
	for (i=0; i<rb.n; i++) {
		if (fabsf(xr[i]-rb.xr[i])>1e-3 || fabsf(zr[i]-rb.zr[i])>1e-3)
			verr("readBoundaryWav: contour point %d geometry mismatch (file vs this run) -- rectangle must match the recording run", i);
	}
	free(xr); free(zr); free(nx); free(nz);

	nb = (size_t)rb.n*nt;
	*bs_vx  = (float *)malloc(nb*sizeof(float));
	*bs_vz  = (float *)malloc(nb*sizeof(float));
	*bs_txx = (float *)malloc(nb*sizeof(float));
	*bs_tzz = (float *)malloc(nb*sizeof(float));
	*bs_txz = (float *)malloc(nb*sizeof(float));
	fread(*bs_vx,  sizeof(float), nb, fp);
	fread(*bs_vz,  sizeof(float), nb, fp);
	fread(*bs_txx, sizeof(float), nb, fp);
	fread(*bs_tzz, sizeof(float), nb, fp);
	fread(*bs_txz, sizeof(float), nb, fp);

	fclose(fp);
	vmess("Read reconstruction boundary wavefield (%d points x %d samples) from %s", rb.n, nt, file);

	return 0;
}
