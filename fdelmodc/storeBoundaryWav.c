#include<stdlib.h>
#include"bndsrc.h"

/*********************************************************************
*
* storeBoundaryWav
*
* Records vx,vz,txx,tzz,txz along the closed reconstruction boundary
* (rb, built by defineRecBoundary) at time sample isam. Each field is
* sampled through bndinterp(inject=0,...) using that field's own
* staggered-offset table -- the exact same tables and the exact same
* function that injectBoundary() will later call with inject=1, so
* recording and reconstruction-injection are guaranteed adjoints of
* each other (see bndinterp.h).
*
* Called from the fdelmodc.c time loop at the same cadence as
* getRecTimes(), right alongside it.
*
**********************************************************************/

int storeBoundaryWav(bndSrcPar rb, int nt, int isam,
	float *vx, float *vz, float *txx, float *tzz, float *txz, int naz,
	float *bs_vx, float *bs_vz, float *bs_txx, float *bs_tzz, float *bs_txz)
{
	float *d;
	int i;

	if (!rb.n) return 0;

	d = (float *)malloc(rb.n*sizeof(float));

	bndinterp(0, 0, vx, naz, d, rb.vx);
	for (i=0; i<rb.n; i++) bs_vx[i*nt+isam] = d[i];

	bndinterp(0, 0, vz, naz, d, rb.vz);
	for (i=0; i<rb.n; i++) bs_vz[i*nt+isam] = d[i];

	bndinterp(0, 0, txx, naz, d, rb.txx);
	for (i=0; i<rb.n; i++) bs_txx[i*nt+isam] = d[i];

	bndinterp(0, 0, tzz, naz, d, rb.tzz);
	for (i=0; i<rb.n; i++) bs_tzz[i*nt+isam] = d[i];

	bndinterp(0, 0, txz, naz, d, rb.txz);
	for (i=0; i<rb.n; i++) bs_txz[i*nt+isam] = d[i];

	free(d);
	return 0;
}
