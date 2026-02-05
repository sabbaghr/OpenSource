/*
 * test_fdfwimodc.c - Test driver for fdfwimodc()
 *
 * Mirrors fdelmodc.c main() lines 277-832 but delegates per-shot
 * time-stepping to fdfwimodc().  Produces identical output so that
 * results can be diff'd against the reference fdelmodc binary.
 *
 * NOTE: The fdelfwi readModel() has a different signature than fdelmodc's.
 *   fdelfwi:  readModel(modPar *mod, bndPar *bnd)  — allocates arrays into mod->
 *   fdelmodc: readModel(modPar mod, bndPar bnd, float *rox, ...)  — fills external arrays
 */

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

double wallclock_time(void);

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);

/* fdelfwi signature: allocates arrays directly into mod-> */
int readModel(modPar *mod, bndPar *bnd);

int defineSource(wavPar wav, srcPar src, modPar mod, recPar rec,
                 shotPar shot, float **src_nwav, int reverse, int verbose);

int allocStoreSourceOnSurface(srcPar src);
int freeStoreSourceOnSurface(void);

int writeSrcRecPos(modPar *mod, recPar *rec, srcPar *src, shotPar *shot);

int fdfwimodc(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec, snaPar *sna, int ixsrc, int izsrc,
              float **src_nwav, int ishot, int nshots, int fileno,
              checkpointPar *chk, int verbose);

/* Self documentation */
char *sdoc[] = {
" ",
" test_fdfwimodc - test driver that wraps fdfwimodc() ",
"                  (same parameters as fdelmodc) ",
" ",
NULL};


int main(int argc, char **argv)
{
	modPar  mod;
	recPar  rec;
	snaPar  sna;
	wavPar  wav;
	srcPar  src;
	bndPar  bnd;
	shotPar shot;
	float **src_nwav;
	float  sinkvel;
	double t0, t1, tinit;
	size_t nsamp, sizew;
	int    n1, ix, iz, ir, ishot, i;
	int    ioPx, ioPz;
	int    ixsrc, izsrc, fileno;
	int    verbose;

	t0 = wallclock_time();
	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;
	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);

	n1 = mod.naz;

	allocStoreSourceOnSurface(src);

	/* -------------------------------------------------------- */
	/* Read velocity and density files                          */
	/* fdelfwi readModel allocates mod.rox, mod.l2m, etc.       */
	/* -------------------------------------------------------- */
	readModel(&mod, &bnd);

	/* -------------------------------------------------------- */
	/* Read and/or define source wavelet(s)                     */
	/* -------------------------------------------------------- */
	if (wav.random) {
		src_nwav    = (float **)calloc(wav.nx, sizeof(float *));
		src_nwav[0] = (float *)calloc(wav.nst, sizeof(float));
		assert(src_nwav[0] != NULL);
		nsamp = 0;
		for (i = 0; i < wav.nx; i++) {
			src_nwav[i] = (float *)(src_nwav[0] + nsamp);
			nsamp += wav.nsamp[i];
		}
	} else {
		sizew       = wav.nt * wav.nx;
		src_nwav    = (float **)calloc(wav.nx, sizeof(float *));
		src_nwav[0] = (float *)calloc(sizew, sizeof(float));
		assert(src_nwav[0] != NULL);
		for (i = 0; i < wav.nx; i++) {
			src_nwav[i] = (float *)(src_nwav[0] + (size_t)(wav.nt * i));
		}
	}

	defineSource(wav, src, mod, rec, shot, src_nwav, mod.grid_dir, verbose);

	/* Get velocity and density at first receiver location */
	ir      = mod.ioZz + rec.z[0] + (rec.x[0] + mod.ioZx) * n1;
	rec.rho = mod.dt / (mod.dx * mod.roz[ir]);
	rec.cp  = sqrt(mod.l2m[ir] * (mod.roz[ir])) * mod.dx / mod.dt;

	t1 = wallclock_time();
	if (verbose) {
		tinit = t1 - t0;
		vmess("*******************************************");
		vmess("************* runtime info ****************");
		vmess("*******************************************");
		vmess("CPU time for intializing arrays and model = %f", tinit);
	}

	/* -------------------------------------------------------- */
	/* Sinking source and receiver arrays                       */
	/* -------------------------------------------------------- */
	ioPx = mod.ioPx;
	ioPz = mod.ioPz;
	if (bnd.lef == 4 || bnd.lef == 2) ioPx += bnd.ntap;
	if (bnd.top == 4 || bnd.top == 2) ioPz += bnd.ntap;
	if (rec.sinkvel) sinkvel = mod.l2m[(rec.x[0] + ioPx) * n1 + rec.z[0] + ioPz];
	else sinkvel = 0.0;

	/* Sink receivers to value different than sinkvel */
	for (ir = 0; ir < rec.n; ir++) {
		iz = rec.z[ir];
		ix = rec.x[ir];
		while (mod.l2m[(ix + ioPx) * n1 + iz + ioPz] == sinkvel) iz++;
		rec.z[ir]  = iz + rec.sinkdepth;
		rec.zr[ir] = rec.zr[ir] + (rec.z[ir] - iz) * mod.dz;
		if (verbose > 3) vmess("receiver position %d at grid[ix=%d, iz=%d] = (x=%f z=%f)",
			ir, ix + ioPx, rec.z[ir] + ioPz, rec.xr[ir] + mod.x0, rec.zr[ir] + mod.z0);
	}

	/* Sink sources to value different than zero */
	for (ishot = 0; ishot < shot.n; ishot++) {
		iz = shot.z[ishot];
		ix = shot.x[ishot];
		while (mod.l2m[(ix + ioPx) * n1 + iz + ioPz] == 0.0) iz++;
		shot.z[ishot] = iz + src.sinkdepth;
	}

	/* Scan for free surface boundary in case of topography */
	for (ix = 0; ix < mod.nx; ix++) {
		iz = ioPz;
		while (mod.l2m[(ix + ioPx) * n1 + iz] == 0.0) iz++;
		bnd.surface[ix + ioPx] = iz;
		if ((verbose > 3) && (iz != ioPz))
			vmess("Topography surface x=%.2f z=%.2f",
				mod.x0 + mod.dx * ix, mod.z0 + mod.dz * (iz - ioPz));
	}
	for (ix = 0; ix < ioPx; ix++) {
		bnd.surface[ix] = bnd.surface[ioPx];
	}
	for (ix = ioPx + mod.nx; ix < mod.iePx; ix++) {
		bnd.surface[ix] = bnd.surface[mod.iePx - 1];
	}
	if (verbose > 3) writeSrcRecPos(&mod, &rec, &src, &shot);

	/* -------------------------------------------------------- */
	/* Shot loop: delegate each shot to fdfwimodc()             */
	/* -------------------------------------------------------- */
	for (ishot = 0; ishot < shot.n; ishot++) {

		izsrc  = shot.z[ishot];
		ixsrc  = shot.x[ishot];
		fileno = 0;

		if (verbose) {
			if (!src.random) {
				vmess("Modeling source %d at gridpoints ix=%d iz=%d",
					ishot, shot.x[ishot], shot.z[ishot]);
				vmess(" which are actual positions x=%.2f z=%.2f",
					mod.x0 + mod.dx * shot.x[ishot],
					mod.z0 + mod.dz * shot.z[ishot]);
			}
			vmess("Receivers at gridpoint x-range ix=%d - %d", rec.x[0], rec.x[rec.n - 1]);
			vmess(" which are actual positions x=%.2f - %.2f",
				mod.x0 + rec.xr[0], mod.x0 + rec.xr[rec.n - 1]);
			vmess("Receivers at gridpoint z-range iz=%d - %d", rec.z[0], rec.z[rec.n - 1]);
			vmess(" which are actual positions z=%.2f - %.2f",
				mod.z0 + rec.zr[0], mod.z0 + rec.zr[rec.n - 1]);
			vmess("*******************************************");
			vmess("***** FD Propagating Source Wavefield *****");
			vmess("*******************************************");
		}

		fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna,
			ixsrc, izsrc, src_nwav, ishot, shot.n, fileno, NULL, verbose);

	} /* end of loop over number of shots */

	t1 = wallclock_time();
	if (verbose) {
		vmess("Total compute time FD modelling = %.2f s.", t1 - t0);
	}

	/* -------------------------------------------------------- */
	/* Free arrays                                              */
	/* -------------------------------------------------------- */
	initargs(argc, argv); /* frees the arg arrays */
	free(mod.rox);
	free(mod.roz);
	free(mod.l2m);
	free(src_nwav[0]);
	free(src_nwav);
	freeStoreSourceOnSurface();
	if (mod.ischeme == 2) {
		free(mod.tss);
		free(mod.tep);
	}
	if (mod.ischeme > 2) {
		free(mod.lam);
		free(mod.muu);
	}
	if (mod.ischeme == 4) {
		free(mod.tss);
		free(mod.tes);
		free(mod.tep);
	}
	if (bnd.ntap) {
		free(bnd.tapx);
		free(bnd.tapz);
		free(bnd.tapxz);
	}
	free(bnd.surface);
	free(shot.x);
	free(shot.z);
	free(src.x);
	free(src.z);
	free(src.tbeg);
	free(src.tend);
	free(rec.x);
	free(rec.z);
	free(rec.xr);
	free(rec.zr);
	if (wav.nsamp != NULL) free(wav.nsamp);

	return 0;
}
