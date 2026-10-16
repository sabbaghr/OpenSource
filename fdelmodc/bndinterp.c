#include<stdlib.h>
#include<math.h>
#include"bndinterp.h"

/*********************************************************************
*
* bndinterpInit / bndinterp
*
* See bndinterp.h for the design rationale: one bilinear weight table
* per staggered field, one execute function reused for both extraction
* (recording) and injection (reconstruction), so injection is the
* adjoint of extraction by construction.
*
**********************************************************************/

bndInterpPar *bndinterpInit(float *xr, float *zr, int npt,
	float ox, float oz, float dx, float dz,
	int ix0, int ix1, int iz0, int iz1)
{
	bndInterpPar *bi;
	int i, ix, iz;
	float fx, fz, rx, rz;

	bi = (bndInterpPar *)malloc(sizeof(bndInterpPar));
	bi->n = npt;
	bi->ix = (int *)malloc(npt*sizeof(int));
	bi->iz = (int *)malloc(npt*sizeof(int));
	bi->w00 = (float *)malloc(npt*sizeof(float));
	bi->w01 = (float *)malloc(npt*sizeof(float));
	bi->w10 = (float *)malloc(npt*sizeof(float));
	bi->w11 = (float *)malloc(npt*sizeof(float));
	bi->valid = (int *)malloc(npt*sizeof(int));

	for (i=0; i<npt; i++) {
		fx = (xr[i]-ox)/dx;
		fz = (zr[i]-oz)/dz;
		ix = (int)floorf(fx);
		iz = (int)floorf(fz);

		bi->ix[i] = ix;
		bi->iz[i] = iz;
		bi->valid[i] = (ix>=ix0) && (ix+1<ix1) && (iz>=iz0) && (iz+1<iz1);

		if (bi->valid[i]) {
			rx = fx-ix;
			rz = fz-iz;
			bi->w00[i] = (1.0-rx)*(1.0-rz);
			bi->w01[i] = (1.0-rx)*(     rz);
			bi->w10[i] = (     rx)*(1.0-rz);
			bi->w11[i] = (     rx)*(     rz);
		}
		else {
			bi->w00[i] = bi->w01[i] = bi->w10[i] = bi->w11[i] = 0.0;
		}
	}

	return bi;
}

void bndinterpFree(bndInterpPar *bi)
{
	free(bi->ix);
	free(bi->iz);
	free(bi->w00);
	free(bi->w01);
	free(bi->w10);
	free(bi->w11);
	free(bi->valid);
	free(bi);
}

void bndinterp(int inject, int add, float *field, int naz, float *d, bndInterpPar *bi)
{
	int i, ix, iz;

	if (inject) { /* spread d[i] onto field -- always accumulates, never resets field */
		for (i=0; i<bi->n; i++) {
			if (!bi->valid[i]) continue;
			ix = bi->ix[i];
			iz = bi->iz[i];
			field[ ix   *naz+iz  ] += d[i]*bi->w00[i];
			field[ ix   *naz+iz+1] += d[i]*bi->w01[i];
			field[(ix+1)*naz+iz  ] += d[i]*bi->w10[i];
			field[(ix+1)*naz+iz+1] += d[i]*bi->w11[i];
		}
	}
	else { /* sample field into d[i] */
		for (i=0; i<bi->n; i++) {
			if (!bi->valid[i]) {
				if (!add) d[i] = 0.0;
				continue;
			}
			ix = bi->ix[i];
			iz = bi->iz[i];
			if (add) {
				d[i] += field[ ix   *naz+iz  ]*bi->w00[i]
				      + field[ ix   *naz+iz+1]*bi->w01[i]
				      + field[(ix+1)*naz+iz  ]*bi->w10[i]
				      + field[(ix+1)*naz+iz+1]*bi->w11[i];
			}
			else {
				d[i]  = field[ ix   *naz+iz  ]*bi->w00[i]
				      + field[ ix   *naz+iz+1]*bi->w01[i]
				      + field[(ix+1)*naz+iz  ]*bi->w10[i]
				      + field[(ix+1)*naz+iz+1]*bi->w11[i];
			}
		}
	}
}
