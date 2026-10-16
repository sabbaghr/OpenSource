#include<stdio.h>
#include"bndsrc.h"

void verr(char *fmt, ...);
void vmess(char *fmt, ...);

/*********************************************************************
*
* writeBoundaryWav
*
* Writes the recorded closed-boundary wavefield (bs_vx..bs_txz, each
* rb.n*nt floats) plus the contour geometry (xr,zr,nx,nz) to a simple
* raw binary file, to be read back by readBoundaryWav in a subsequent
* reconstruction run. Header: two ints (n, nt), then xr,zr,nx,nz (each
* n floats), then bs_vx,bs_vz,bs_txx,bs_tzz,bs_txz (each n*nt floats).
*
**********************************************************************/

int writeBoundaryWav(char *file, bndSrcPar rb, int nt,
	float *bs_vx, float *bs_vz, float *bs_txx, float *bs_tzz, float *bs_txz)
{
	FILE *fp;
	size_t nb = (size_t)rb.n*nt;

	fp = fopen(file,"w");
	if (fp==NULL) verr("writeBoundaryWav: cannot open %s for writing", file);

	fwrite(&rb.n, sizeof(int), 1, fp);
	fwrite(&nt, sizeof(int), 1, fp);
	fwrite(rb.xr, sizeof(float), rb.n, fp);
	fwrite(rb.zr, sizeof(float), rb.n, fp);
	fwrite(rb.nx, sizeof(float), rb.n, fp);
	fwrite(rb.nz, sizeof(float), rb.n, fp);
	fwrite(bs_vx,  sizeof(float), nb, fp);
	fwrite(bs_vz,  sizeof(float), nb, fp);
	fwrite(bs_txx, sizeof(float), nb, fp);
	fwrite(bs_tzz, sizeof(float), nb, fp);
	fwrite(bs_txz, sizeof(float), nb, fp);

	fclose(fp);
	vmess("Wrote reconstruction boundary wavefield (%d points x %d samples) to %s", rb.n, nt, file);

	return 0;
}
