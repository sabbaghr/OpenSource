/*
 * test_wave_op_dp.c - Wave operator adjoint dot product test.
 *
 * Verifies that the elastic wave propagation operator A and its
 * numerical adjoint A^T satisfy the fundamental identity:
 *
 *   <Ax, y> = <x, A^T y>
 *
 * where:
 *   x = random source wavelet (injected as Fz at source location)
 *   Ax = data recorded at receivers from forward propagation
 *   y = random data at receivers
 *   A^T y = adjoint wavefield extracted at source from backpropagating y
 *
 * This test bypasses fdfwimodc and readResidual to avoid time-sample
 * mapping inconsistencies. It calls elastic4/elastic4_adj directly
 * and records/injects at identical grid positions.
 *
 * elastic{4,6,8}_adj implement the TRUE discrete adjoint with material
 * parameters inside the spatial derivatives (SBP transpose property).
 * Both velocity and stress outputs are the true adjoint fields.
 *
 * Parameters (same interface as fdelmodc):
 *   seed=         random number seed (12345)
 *   comp=_rvz     component suffix for recording (default: _rvz)
 *   rec_comp=vz   recording component: vz, vx, txx, tzz, p (default: vz)
 */

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<string.h>
#include"par.h"
#include"segy.h"
#include"fdelfwi.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

double wallclock_time(void);
void threadAffinity(void);

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);
int readModel(modPar *mod, bndPar *bnd);
int defineSource(wavPar wav, srcPar src, modPar mod, recPar rec,
                 shotPar shot, float **src_nwav, int reverse, int verbose);
int allocStoreSourceOnSurface(srcPar src);
int freeStoreSourceOnSurface(void);

/* Forward FD kernels */
int elastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime,
    int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz,
    float *tzz, float *txx, float *txz, float *rox, float *roz,
    float *l2m, float *lam, float *mul, int verbose);
int elastic6(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime,
    int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz,
    float *tzz, float *txx, float *txz, float *rox, float *roz,
    float *l2m, float *lam, float *mul, int verbose);
int elastic8(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime,
    int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz,
    float *tzz, float *txx, float *txz, float *rox, float *roz,
    float *l2m, float *lam, float *mul, int verbose);

/* True adjoint FD kernels */
int elastic4_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *rox, float *roz, float *l2m, float *lam, float *mul,
    int rec_delay, int rec_skipdt, int verbose);
int elastic6_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *rox, float *roz, float *l2m, float *lam, float *mul,
    int rec_delay, int rec_skipdt, int verbose);
int elastic8_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *rox, float *roz, float *l2m, float *lam, float *mul,
    int rec_delay, int rec_skipdt, int verbose);

/* Snapshot output */
int writeSnapTimes(modPar mod, snaPar sna, bndPar bnd, wavPar wav,
    int ixsrc, int izsrc, int itime,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    int verbose);
void writeSnapTimesReset(void);

char *sdoc[] = {
" ",
" test_wave_op_dp - Wave operator adjoint dot product test",
" ",
" Verifies: <Ax, y> = <x, A^T y>",
"   A  = wave propagation operator (source -> receivers)",
"   A^T = adjoint wave operator (receivers -> source)",
" ",
" Same parameters as fdelmodc, plus:",
"   seed=         random number seed (12345)",
"   comp=_rvz     component suffix for recording",
" ",
NULL};


/*====================================================================
 * Main: Wave Operator Adjoint Dot Product Test
 *====================================================================*/

int main(int argc, char **argv)
{
	modPar  mod;
	recPar  rec;
	snaPar  sna;
	wavPar  wav;
	srcPar  src;
	bndPar  bnd;
	shotPar shot;
	adjSrcPar adj;
	float **src_nwav;
	float  sinkvel;
	double t0, t1;
	size_t nsamp, sizew, sizem;
	int    n1, ix, iz, ir, i;
	int    ioPx, ioPz;
	int    ixsrc, izsrc, fileno;
	int    verbose;

	/* Test parameters */
	char  *comp_str;
	int    seed;

	/* Dot product variables */
	double lhs, rhs, ratio, epsilon;

	t0 = wallclock_time();

	printf("=== Wave Operator Adjoint Dot Product Test ===\n");
	printf("  Verifies: <Ax, y> = <x, A^T y>\n\n");

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;
	if (!getparint("seed", &seed)) seed = 12345;
	if (!getparstring("comp", &comp_str)) comp_str = "_rvz";

	/* Recording component: determines forward extraction and adjoint injection.
	 * 0=vz, 1=vx, 2=txx, 3=tzz, 4=p */
	char *rec_comp_str;
	int rec_comp_id;
	if (!getparstring("rec_comp", &rec_comp_str)) rec_comp_str = "vz";
	if (strcmp(rec_comp_str, "vz") == 0)       rec_comp_id = 0;
	else if (strcmp(rec_comp_str, "vx") == 0)   rec_comp_id = 1;
	else if (strcmp(rec_comp_str, "txx") == 0)  rec_comp_id = 2;
	else if (strcmp(rec_comp_str, "tzz") == 0)  rec_comp_id = 3;
	else if (strcmp(rec_comp_str, "p") == 0)    rec_comp_id = 4;
	else verr("Unknown rec_comp=%s. Use vz, vx, txx, tzz, or p", rec_comp_str);

	/* ============================================================ */
	/* Standard setup (same as test_dotproduct.c)                   */
	/* ============================================================ */
	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);
	n1 = mod.naz;

	allocStoreSourceOnSurface(src);
	readModel(&mod, &bnd);

	/* Source wavelet setup */
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
		for (i = 0; i < wav.nx; i++)
			src_nwav[i] = (float *)(src_nwav[0] + (size_t)(wav.nt * i));
	}

	defineSource(wav, src, mod, rec, shot, src_nwav, mod.grid_dir, verbose);

	/* Sinking */
	ioPx = mod.ioPx;
	ioPz = mod.ioPz;
	if (bnd.lef == 4 || bnd.lef == 2) ioPx += bnd.ntap;
	if (bnd.top == 4 || bnd.top == 2) ioPz += bnd.ntap;

	ir = mod.ioZz + rec.z[0] + (rec.x[0] + mod.ioZx) * n1;
	rec.rho = mod.dt / (mod.dx * mod.roz[ir]);
	rec.cp  = sqrt(mod.l2m[ir] * (mod.roz[ir])) * mod.dx / mod.dt;

	if (rec.sinkvel) sinkvel = mod.l2m[(rec.x[0] + ioPx) * n1 + rec.z[0] + ioPz];
	else sinkvel = 0.0;

	for (ir = 0; ir < rec.n; ir++) {
		iz = rec.z[ir];
		ix = rec.x[ir];
		while (mod.l2m[(ix + ioPx) * n1 + iz + ioPz] == sinkvel) iz++;
		rec.z[ir]  = iz + rec.sinkdepth;
		rec.zr[ir] = rec.zr[ir] + (rec.z[ir] - iz) * mod.dz;
	}

	for (i = 0; i < shot.n; i++) {
		iz = shot.z[i];
		ix = shot.x[i];
		while (mod.l2m[(ix + ioPx) * n1 + iz + ioPz] == 0.0) iz++;
		shot.z[i] = iz + src.sinkdepth;
	}

	for (ix = 0; ix < mod.nx; ix++) {
		iz = ioPz;
		while (mod.l2m[(ix + ioPx) * n1 + iz] == 0.0) iz++;
		bnd.surface[ix + ioPx] = iz;
	}
	for (ix = 0; ix < ioPx; ix++)
		bnd.surface[ix] = bnd.surface[ioPx];
	for (ix = ioPx + mod.nx; ix < mod.iePx; ix++)
		bnd.surface[ix] = bnd.surface[mod.iePx - 1];

	/* Process first shot only */
	izsrc  = shot.z[0];
	ixsrc  = shot.x[0];
	fileno = 0;
	sizem  = (size_t)mod.nax * mod.naz;

	int it1 = mod.nt + NINT(-mod.t0 / mod.dt);
	int nrec = rec.n;

	/* Receiver grid positions (same as getRecTimes with no interpolation) */
	int ibndx = mod.ioPx;
	int ibndz = mod.ioPz;
	if (bnd.lef == 4 || bnd.lef == 2) ibndx += bnd.ntap;
	if (bnd.top == 4 || bnd.top == 2) ibndz += bnd.ntap;

	/* Source grid position for extraction (src_type=7: Vz grid) */
	int src_ibndx, src_ibndz, src_ig;
	if (src.type == 6) {
		src_ibndx = mod.ioXx;
		src_ibndz = mod.ioXz;
	}
	else if (src.type == 7) {
		src_ibndx = mod.ioZx;
		src_ibndz = mod.ioZz;
	}
	else {
		src_ibndx = mod.ioPx;
		src_ibndz = mod.ioPz;
		if (bnd.lef == 4 || bnd.lef == 2) src_ibndx += bnd.ntap;
		if (bnd.top == 4 || bnd.top == 2) src_ibndz += bnd.ntap;
	}
	src_ig = (ixsrc + src_ibndx) * n1 + (izsrc + src_ibndz);

	/* Source extraction scaling (adjoint of source injection).
	 * TRUE adjoint: both velocity and stress are true adjoint fields.
	 *
	 * Forward src_type=7: vz += wavelet * roz/dx  -> alpha = +roz/dx
	 * Forward src_type=1: txx,tzz += wavelet*l2m/dx -> alpha = +l2m/dx
	 * Forward src_type=6: vx += wavelet*0.5*rox/dx -> alpha = +0.5*rox/dx
	 */
	float alpha = 0.0f;
	if (src.type == 7) {
		alpha = mod.roz[src_ig] / mod.dx;
	}
	else if (src.type == 1) {
		alpha = mod.l2m[src_ig] / mod.dx;  /* true adjoint: no sign flip */
	}
	else if (src.type == 6) {
		alpha = 0.5 * mod.rox[src_ig] / mod.dx;  /* 0.5 from applySource line 154 */
	}
	else {
		verr("Wave operator DP test not implemented for src_type=%d", src.type);
	}

	vmess("*********************************************");
	vmess("***** WAVE OPERATOR DOT PRODUCT TEST    *****");
	vmess("*********************************************");
	vmess("Grid: nx=%d nz=%d (padded: nax=%d naz=%d)", mod.nx, mod.nz, mod.nax, mod.naz);
	vmess("Source at grid ix=%d iz=%d (src_type=%d)", ixsrc, izsrc, src.type);
	vmess("Source padded index ig=%d (ibndx=%d ibndz=%d)", src_ig, src_ibndx, src_ibndz);
	vmess("Receivers: n=%d, ibndx=%d ibndz=%d", nrec, ibndx, ibndz);
	vmess("Time steps: it1=%d (nt=%d, t0=%.4f, dt=%.6f)", it1, mod.nt, mod.t0, mod.dt);
	vmess("Alpha (source scaling) = %.6e", alpha);
	vmess("Recording component: rec_comp=%s (id=%d)", rec_comp_str, rec_comp_id);
	vmess("Seed: %d", seed);

	/* ============================================================ */
	/* Step 1: Replace source wavelet with random values            */
	/* ============================================================ */
	vmess("--- Step 1: Generate random source wavelet x ---");

	srand48(seed);
	for (i = 0; i < wav.nt; i++)
		src_nwav[0][i] = (float)(drand48() * 2.0 - 1.0);

	vmess("Filled src_nwav[0] with %d random samples", wav.nt);

	/* ============================================================ */
	/* Step 2: Forward propagation using elastic4 directly          */
	/* ============================================================ */
	vmess("--- Step 2: Forward propagation (record at every time step) ---");

	float *fwd_vx  = (float *)calloc(sizem, sizeof(float));
	float *fwd_vz  = (float *)calloc(sizem, sizeof(float));
	float *fwd_tzz = (float *)calloc(sizem, sizeof(float));
	float *fwd_txx = (float *)calloc(sizem, sizeof(float));
	float *fwd_txz = (float *)calloc(sizem, sizeof(float));

	/* Recording array: d[irec * it1 + it] = Vz at receiver irec, time step it */
	float *d_data = (float *)calloc((size_t)nrec * it1, sizeof(float));

	if (verbose) {
		fprintf(stderr, "    %s: Forward progress: %3d%%", xargv[0], 0);
	}

	int it;
	for (it = 0; it < it1; it++) {

#pragma omp parallel default(shared)
{
		if (it == 0 && verbose > 2) threadAffinity();

		if (mod.iorder == 4)
			elastic4(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav,
				fwd_vx, fwd_vz, fwd_tzz, fwd_txx, fwd_txz,
				mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu, verbose);
		else if (mod.iorder == 6)
			elastic6(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav,
				fwd_vx, fwd_vz, fwd_tzz, fwd_txx, fwd_txz,
				mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu, verbose);
		else if (mod.iorder == 8)
			elastic8(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav,
				fwd_vx, fwd_vz, fwd_tzz, fwd_txx, fwd_txz,
				mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu, verbose);
}

		/* Record wavefield at receiver positions.
		 * Grid positions match getRecTimes (no interpolation):
		 *   vz:  (ibndx+rx, ibndz+rz+1)  - Vz stagger shift +1 in z
		 *   vx:  (ibndx+rx+1, ibndz+rz)  - Vx stagger shift +1 in x
		 *   txx: (ibndx+rx, ibndz+rz)    - P grid, no shift
		 *   tzz: (ibndx+rx, ibndz+rz)    - P grid, no shift */
		for (ir = 0; ir < nrec; ir++) {
			int rx = rec.x[ir] + ibndx;
			int rz = rec.z[ir] + ibndz;
			if (rec_comp_id == 0)       /* vz */
				d_data[ir * it1 + it] = fwd_vz[rx * n1 + rz + 1];
			else if (rec_comp_id == 1)  /* vx */
				d_data[ir * it1 + it] = fwd_vx[(rx + 1) * n1 + rz];
			else if (rec_comp_id == 2)  /* txx */
				d_data[ir * it1 + it] = fwd_txx[rx * n1 + rz];
			else if (rec_comp_id == 3)  /* tzz */
				d_data[ir * it1 + it] = fwd_tzz[rx * n1 + rz];
			else                        /* p = 0.5*(txx+tzz) */
				d_data[ir * it1 + it] = 0.5f * (fwd_txx[rx * n1 + rz] + fwd_tzz[rx * n1 + rz]);
		}

		if (verbose && !((it1 - it) % (it1 / 100 + 1)))
			fprintf(stderr, "\b\b\b\b%3d%%", it * 100 / it1);
	}

	if (verbose) fprintf(stderr, "\b\b\b\b%3d%%\n", 100);

	/* Free forward wavefields */
	free(fwd_vx);
	free(fwd_vz);
	free(fwd_tzz);
	free(fwd_txx);
	free(fwd_txz);

	/* Forward diagnostics */
	{
		double max_d = 0.0;
		for (i = 0; i < nrec * it1; i++) {
			double ad = fabs((double)d_data[i]);
			if (ad > max_d) max_d = ad;
		}
		vmess("Forward done. max|d| = %.6e (%d receivers, %d time steps)", max_d, nrec, it1);
	}

	/* ============================================================ */
	/* Step 3: Generate random receiver data y                      */
	/* ============================================================ */
	vmess("--- Step 3: Generate random receiver data y ---");

	float *y_data = (float *)calloc((size_t)nrec * it1, sizeof(float));
	srand48(seed + 77777);
	for (i = 0; i < nrec * it1; i++)
		y_data[i] = (float)(drand48() * 2.0 - 1.0);

	vmess("Generated %d random samples for y (%d traces x %d steps)",
		nrec * it1, nrec, it1);

	/* ============================================================ */
	/* Step 4: Compute LHS = <Ax, y> = <d, y>                      */
	/* ============================================================ */
	vmess("--- Step 4: Compute <Ax, y> ---");

	lhs = 0.0;
	for (i = 0; i < nrec * it1; i++)
		lhs += (double)d_data[i] * (double)y_data[i];

	vmess("<Ax, y> = %.15e", lhs);

	/* ============================================================ */
	/* Step 5: Set up adjoint source (y at receiver positions)      */
	/* ============================================================ */
	vmess("--- Step 5: Set up adjoint sources ---");

	/* Build adjSrcPar manually so positions exactly match getRecTimes.
	 * This avoids the SU header / readResidual position mapping. */
	memset(&adj, 0, sizeof(adjSrcPar));
	adj.nsrc   = nrec;
	adj.nt     = it1;
	adj.xi     = (size_t *)malloc(nrec * sizeof(size_t));
	adj.zi     = (size_t *)malloc(nrec * sizeof(size_t));
	adj.typ    = (int *)malloc(nrec * sizeof(int));
	adj.orient = (int *)malloc(nrec * sizeof(int));
	adj.x      = (float *)malloc(nrec * sizeof(float));
	adj.z      = (float *)malloc(nrec * sizeof(float));
	adj.wav    = (float *)malloc((size_t)nrec * it1 * sizeof(float));

	/* Wav data stored as +y (no sign flip in the data itself).
	 * applyAdjointSource handles sign conventions:
	 *   force types (6,7): += y  (inject into velocity field)
	 *   stress types (3,4): -= y  (inject into stress field, compression-positive) */

	for (ir = 0; ir < nrec; ir++) {
		/* Grid positions: EXACTLY match getRecTimes recording positions.
		 *   vz:  (ibndx+rx, ibndz+rz+1)   Vz stagger +1 in z
		 *   vx:  (ibndx+rx+1, ibndz+rz)   Vx stagger +1 in x
		 *   txx: (ibndx+rx, ibndz+rz)     P grid
		 *   tzz: (ibndx+rx, ibndz+rz)     P grid */
		int rx_base = ibndx + rec.x[ir];
		int rz_base = ibndz + rec.z[ir];

		if (rec_comp_id == 0) {       /* vz */
			adj.xi[ir] = rx_base;
			adj.zi[ir] = rz_base + 1;   /* +1 for Vz stagger */
			adj.typ[ir] = 7;            /* Fz -> injects into vz */
		}
		else if (rec_comp_id == 1) {  /* vx */
			adj.xi[ir] = rx_base + 1;   /* +1 for Vx stagger */
			adj.zi[ir] = rz_base;
			adj.typ[ir] = 6;            /* Fx -> injects into vx */
		}
		else if (rec_comp_id == 2) {  /* txx */
			adj.xi[ir] = rx_base;
			adj.zi[ir] = rz_base;
			adj.typ[ir] = 4;            /* Txx -> injects into txx */
		}
		else if (rec_comp_id == 3) {  /* tzz */
			adj.xi[ir] = rx_base;
			adj.zi[ir] = rz_base;
			adj.typ[ir] = 3;            /* Tzz -> injects into tzz */
		}
		else {                        /* p = 0.5*(txx+tzz) */
			adj.xi[ir] = rx_base;
			adj.zi[ir] = rz_base;
			adj.typ[ir] = 1;            /* P -> injects into txx+tzz */
		}
		adj.orient[ir] = 1;   /* monopole */
		adj.x[ir] = rec.xr[ir];
		adj.z[ir] = rec.zr[ir];

		/* Copy y data for this receiver.
		 * applyAdjointSource reads: adj.wav[isrc * adj.nt + isam]
		 * where isam = (itime - delay) / skipdt = itime (with delay=0, skipdt=1). */
		for (it = 0; it < it1; it++)
			adj.wav[ir * it1 + it] = y_data[ir * it1 + it];
	}

	vmess("Set up %d adjoint sources (type=%d, rec_comp=%s) with nt=%d samples",
		adj.nsrc, adj.typ[0], rec_comp_str, adj.nt);
	vmess("  Rcv[0] position: xi=%zu zi=%zu",
		adj.xi[0], adj.zi[0]);

	/* ============================================================ */
	/* Step 6: Adjoint propagation using elastic4_adj               */
	/* ============================================================ */
	vmess("--- Step 6: Adjoint propagation (A^T y) ---");

	float *adj_vx  = (float *)calloc(sizem, sizeof(float));
	float *adj_vz  = (float *)calloc(sizem, sizeof(float));
	float *adj_tzz = (float *)calloc(sizem, sizeof(float));
	float *adj_txx = (float *)calloc(sizem, sizeof(float));
	float *adj_txz = (float *)calloc(sizem, sizeof(float));
	float *z_adj   = (float *)calloc(it1, sizeof(float));

	if (verbose) {
		fprintf(stderr, "    %s: Adjoint progress: %3d%%", xargv[0], 0);
	}

	/* Set up snapshot parameters for adjoint wavefield visualization.
	 * Use sna from getParameters -- if dtsnap was specified, snapshots will
	 * be written to adj_snap_svz.su, adj_snap_svx.su, etc. */
	if (sna.nsnap > 0) {
		sna.file_snap = "adj_snap";
		sna.type.vz  = 1;
		sna.type.vx  = 1;
		sna.type.txx = 1;
		sna.type.tzz = 1;
		sna.type.txz = 1;
		sna.type.p   = 0;
		writeSnapTimesReset();
		vmess("Writing adjoint snapshots: %d snaps, dt_snap=%d steps", sna.nsnap, sna.skipdt);
	}

	/* Stress SOURCE extraction (src_type=1) must happen BEFORE elastic4_adj
	 * because Phase A2 modifies txx/tzz.  Phase A1 doesn't modify stress,
	 * so extracting before elastic4_adj = extracting before Phase A2.
	 *
	 * Stress RECORDING injection is now handled correctly by applyAdjointSource
	 * inside the adjoint kernel (phase=2, before Phase A1, with += sign). */

	int adj_snap_count = 0;
	for (it = it1 - 1; it >= 0; it--) {

		/* ---- PRE-KERNEL: stress source extraction ---- */
		/* For stress source (P/explosive): extract BEFORE elastic4_adj.
		 * Phase A1 doesn't modify stress, Phase A2 does. So extracting
		 * here gives the correct value (before Phase A2 contamination). */
		if (src.type == 1)
			z_adj[it] = alpha * (adj_txx[src_ig] + adj_tzz[src_ig]);

		/* ---- KERNEL: elastic adjoint FD step ---- */
		/* applyAdjointSource inside the kernel handles ALL recording types:
		 *   phase=2 (before Phase A1): stress types (P, Txx, Tzz, Txz)
		 *   phase=1 (after Phase A1):  force types (Fx, Fz)
		 * All use += sign (true adjoint). */
#pragma omp parallel default(shared)
{
		if (it == it1 - 1 && verbose > 2) threadAffinity();

		if (mod.iorder == 4)
			elastic4_adj(mod, adj, bnd, it,
				adj_vx, adj_vz, adj_tzz, adj_txx, adj_txz,
				mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu,
				/*rec_delay=*/0, /*rec_skipdt=*/1, verbose);
		else if (mod.iorder == 6)
			elastic6_adj(mod, adj, bnd, it,
				adj_vx, adj_vz, adj_tzz, adj_txx, adj_txz,
				mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu,
				/*rec_delay=*/0, /*rec_skipdt=*/1, verbose);
		else if (mod.iorder == 8)
			elastic8_adj(mod, adj, bnd, it,
				adj_vx, adj_vz, adj_tzz, adj_txx, adj_txz,
				mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu,
				/*rec_delay=*/0, /*rec_skipdt=*/1, verbose);
}

		/* ---- POST-KERNEL: force source extraction ---- */
		/* For force sources (Fz, Fx): extract AFTER elastic4_adj.
		 * Phase A2 updates stress, not velocity, so velocity is
		 * unchanged by Phase A2. Extracting here is equivalent to
		 * extracting between BoundP^T and Phase A2 (correct point). */
		if (src.type == 7)
			z_adj[it] = alpha * adj_vz[src_ig];
		else if (src.type == 6)
			z_adj[it] = alpha * adj_vx[src_ig];

		/* Write adjoint wavefield snapshot if requested */
		if (sna.nsnap > 0 &&
		    (it >= sna.delay) &&
		    (it <= sna.delay + (sna.nsnap - 1) * sna.skipdt) &&
		    ((it - sna.delay) % sna.skipdt == 0)) {
			int fake_it = sna.delay + adj_snap_count * sna.skipdt;
			writeSnapTimes(mod, sna, bnd, wav, ixsrc, izsrc, fake_it,
				adj_vx, adj_vz, adj_tzz, adj_txx, adj_txz, verbose);
			adj_snap_count++;
		}

		/* Diagnostic: print max wavefield values for first few backward steps */
		if (verbose >= 1 && (it >= it1 - 3 || it <= 2)) {
			double mxvx=0, mxvz=0, mxtzz=0, mxtxx=0;
			int ii;
			for (ii=0; ii<(int)sizem; ii++) {
				if (fabs(adj_vx[ii])>mxvx) mxvx=fabs(adj_vx[ii]);
				if (fabs(adj_vz[ii])>mxvz) mxvz=fabs(adj_vz[ii]);
				if (fabs(adj_tzz[ii])>mxtzz) mxtzz=fabs(adj_tzz[ii]);
				if (fabs(adj_txx[ii])>mxtxx) mxtxx=fabs(adj_txx[ii]);
			}
			fprintf(stderr, "  [DIAG it=%d] max|vx|=%.3e max|vz|=%.3e max|txx|=%.3e max|tzz|=%.3e z_adj=%.3e\n",
				it, mxvx, mxvz, mxtxx, mxtzz, z_adj[it]);
		}

		if (verbose && !((it1 - it) % (it1 / 100 + 1)))
			fprintf(stderr, "\b\b\b\b%3d%%", (it1 - it) * 100 / it1);
	}

	if (verbose) fprintf(stderr, "\b\b\b\b%3d%%\n", 100);

	vmess("Adjoint propagation complete.");

	/* Adjoint diagnostics */
	{
		double max_z = 0.0;
		for (it = 0; it < it1; it++) {
			double az = fabs((double)z_adj[it]);
			if (az > max_z) max_z = az;
		}
		vmess("max|z_adj| = %.6e", max_z);
	}

	/* ============================================================ */
	/* Step 7: Compute RHS = <x, A^T y>                            */
	/* ============================================================ */
	vmess("--- Step 7: Compute <x, A^T y> ---");

	/* RHS = sum over time steps of: x_interpolated(it) * z_adj(it)
	 * where x_interpolated uses the SAME interpolation as applySource. */
	rhs = 0.0;
	{
		float dt = mod.dt;

		for (it = 0; it < it1; it++) {
			float time, x_val;
			int id1, id2;

			/* Same time-to-sample mapping as applySource.c (line 101) */
			time = it * dt - src.tbeg[0] + mod.t0;
			id1 = (int)floor(time / dt);
			id2 = id1 + 1;

			/* Skip if outside wavelet window (same checks as applySource line 106) */
			if (time < 0.0f) continue;
			if ((it * dt + mod.t0) >= src.tend[0]) continue;
			if (id2 > wav.nt) continue;
			if (id1 < 0) continue;

			/* Linear interpolation (same as applySource line 111) */
			x_val = src_nwav[0][id1] * (id2 - time / dt)
			      + src_nwav[0][id2] * (time / dt - id1);

			rhs += (double)x_val * (double)z_adj[it];
		}
	}

	vmess("<x, A^T y> = %.15e", rhs);

	/* ============================================================ */
	/* Step 8: Evaluate                                              */
	/* ============================================================ */
	printf("\n--- Wave Operator Dot Product Test Results (src_type=%d, rec_comp=%s) ---\n",
		src.type, rec_comp_str);
	printf("  <Ax, y>     = %+.15e\n", lhs);
	printf("  <x, A^T y>  = %+.15e\n", rhs);

	if (fabs(lhs) > 0.0) {
		ratio = rhs / lhs;
		epsilon = fabs(ratio - 1.0);
	} else {
		ratio = 0.0;
		epsilon = fabs(rhs);
	}

	printf("  ratio        = %.12f\n", ratio);
	printf("  epsilon      = %.2e\n", epsilon);
	printf("  %s (epsilon %s 1e-02)\n",
		epsilon < 0.01 ? "PASS" : "FAIL",
		epsilon < 0.01 ? "<" : ">=");

	t1 = wallclock_time();
	printf("\nTotal wall time: %.2f s\n", t1 - t0);

	/* ============================================================ */
	/* Cleanup                                                       */
	/* ============================================================ */
	free(adj_vx);
	free(adj_vz);
	free(adj_tzz);
	free(adj_txx);
	free(adj_txz);
	free(z_adj);

	free(adj.xi);
	free(adj.zi);
	free(adj.typ);
	free(adj.orient);
	free(adj.x);
	free(adj.z);
	free(adj.wav);

	free(d_data);
	free(y_data);
	free(src_nwav[0]);
	free(src_nwav);

	free(mod.rox);
	free(mod.roz);
	free(mod.l2m);
	if (mod.lam) free(mod.lam);
	if (mod.muu) free(mod.muu);
	if (mod.tss) free(mod.tss);
	if (mod.tep) free(mod.tep);
	if (mod.tes) free(mod.tes);
	if (bnd.surface) free(bnd.surface);
	if (bnd.tapz) free(bnd.tapz);
	if (bnd.tapx) free(bnd.tapx);
	if (bnd.tapxz) free(bnd.tapxz);

	freeStoreSourceOnSurface();

	printf("\n=== Wave operator dot product test completed ===\n");

	return (epsilon < 0.01) ? 0 : 1;
}
