/*
 * test_fwi_gradient.c - End-to-end gradient test for elastic FWI.
 *
 * Runs the full single-shot FWI gradient pipeline:
 *   1. Forward modeling with disk checkpointing (fdfwimodc)
 *   2. Residual computation (computeResidual)
 *   3. Residual parsing (readResidual)
 *   4. Adjoint backpropagation with gradient accumulation (adj_shot)
 *   5. Gradient output to SU files (writesufile)
 *
 * Parameters are the same as test_fdfwimodc / fdelmodc, plus:
 *   file_obs=     base name of observed data (e.g., "obs" -> reads obs_rvz.su)
 *   chk_skipdt=   time steps between checkpoints (default 500)
 *   chk_base=     base path for checkpoint files (default "chk")
 *   file_grad=    base name for gradient output  (default "gradient")
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
#define MAX_COMP 8

double wallclock_time(void);

void threadAffinity(void);

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);

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

int initCheckpoints(checkpointPar *chk, modPar *mod, int skipdt,
                    int delay, const char *file_base);
int cleanCheckpoints(checkpointPar *chk);

float computeResidual(int ncomp, const char **obs_files, const char **syn_files,
                      const char *res_file, misfitType mtype, int verbose);

int readResidual(const char *filename, adjSrcPar *adj, modPar *mod, bndPar *bnd);
void freeResidual(adjSrcPar *adj);

int adj_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
             recPar *rec, adjSrcPar *adj,
             int ixsrc, int izsrc, float **src_nwav,
             checkpointPar *chk, snaPar *sna,
             float *grad1, float *grad2, float *grad3,
             int param, int verbose);

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

char *sdoc[] = {
" ",
" test_fwi_gradient - end-to-end gradient test for elastic FWI",
" ",
" Same parameters as fdelmodc, plus:",
"   file_obs=     observed data base name (required)",
"   chk_skipdt=   checkpoint interval in time steps (500)",
"   chk_base=     checkpoint file base path (chk)",
"   file_grad=    gradient output base name (gradient)",
"   comp=_rvz     comma-separated component suffixes for residual",
"                 e.g., comp=_rvz,_rvx,_rtzz,_rtxx",
"   file_adj_snap= adjoint snapshot base name (adj_snap)",
"   res_taper=100 cosine taper length (samples) applied to residual",
"                  traces at both ends before backpropagation",
"   param=1       parameterization for gradient output:",
"                  1 = Lame (lambda, mu, rho_direct)",
"                  2 = Velocity (Vp, Vs, rho_full with chain rule)",
" ",
NULL};


/*--------------------------------------------------------------------
 * stripBoundary -- Copy interior model from padded array to dense array.
 *
 * src:  padded array [nax * naz]
 * dst:  dense array  [nx  * nz]  (column-major: dst[ix*nz+iz])
 *
 * ioPx/ioPz do NOT include ntap for absorbing boundaries, so we must
 * add ntap when lef/top boundaries are absorbing (type 4 or 2),
 * matching the convention in getRecTimes.c and readResidual.c.
 *--------------------------------------------------------------------*/
static void stripBoundary(float *dst, float *src, modPar *mod, bndPar *bnd)
{
	int ix, iz;
	int ibndx = mod->ioPx;
	int ibndz = mod->ioPz;
	if (bnd->lef == 4 || bnd->lef == 2) ibndx += bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibndz += bnd->ntap;
	for (ix = 0; ix < mod->nx; ix++)
		for (iz = 0; iz < mod->nz; iz++)
			dst[ix * mod->nz + iz] = src[(ix + ibndx) * mod->naz + iz + ibndz];
}


/*--------------------------------------------------------------------
 * computeHydrophone -- Compute pressure P = 0.5*(Tzz + Txx) for elastic.
 *
 * For elastic modeling (ischeme > 2), fdelmodc records Tzz and Txx
 * separately. The hydrophone (pressure) is the mean normal stress:
 *   P = 0.5 * (Txx + Tzz)
 *
 * This function reads the Tzz and Txx SU files, computes the sum,
 * scales by 0.5, and writes the pressure SU file.
 *
 * Returns 0 on success, -1 on error (files not found or size mismatch).
 *--------------------------------------------------------------------*/
static int computeHydrophone(const char *file_tzz, const char *file_txx,
                             const char *file_p, int verbose)
{
	FILE *fp_tzz, *fp_txx, *fp_p;
	segy hdr_tzz, hdr_txx;
	float *data_tzz, *data_txx;
	int ns, ntr_tzz, ntr_txx, itr, is;
	size_t nread;

	fp_tzz = fopen(file_tzz, "r");
	fp_txx = fopen(file_txx, "r");
	if (!fp_tzz || !fp_txx) {
		if (fp_tzz) fclose(fp_tzz);
		if (fp_txx) fclose(fp_txx);
		if (verbose) vmess("computeHydrophone: cannot open Tzz or Txx files");
		return -1;
	}

	fp_p = fopen(file_p, "w");
	if (!fp_p) {
		fclose(fp_tzz);
		fclose(fp_txx);
		if (verbose) vmess("computeHydrophone: cannot create pressure file %s", file_p);
		return -1;
	}

	/* Read first header to get ns */
	nread = fread(&hdr_tzz, 1, TRCBYTES, fp_tzz);
	if (nread != TRCBYTES) {
		fclose(fp_tzz); fclose(fp_txx); fclose(fp_p);
		return -1;
	}
	rewind(fp_tzz);

	ns = hdr_tzz.ns;
	data_tzz = (float *)malloc(ns * sizeof(float));
	data_txx = (float *)malloc(ns * sizeof(float));

	ntr_tzz = 0;
	ntr_txx = 0;
	while (fread(&hdr_tzz, 1, TRCBYTES, fp_tzz) == TRCBYTES &&
	       fread(&hdr_txx, 1, TRCBYTES, fp_txx) == TRCBYTES) {

		if (hdr_tzz.ns != ns || hdr_txx.ns != ns) {
			if (verbose) vmess("computeHydrophone: ns mismatch");
			break;
		}

		nread = fread(data_tzz, sizeof(float), ns, fp_tzz);
		if ((int)nread != ns) break;
		nread = fread(data_txx, sizeof(float), ns, fp_txx);
		if ((int)nread != ns) break;

		/* P = 0.5*(Tzz + Txx) */
		for (is = 0; is < ns; is++)
			data_tzz[is] = 0.5f * (data_tzz[is] + data_txx[is]);

		/* Update header for pressure: trid=11 (pressure) */
		hdr_tzz.trid = 11;

		fwrite(&hdr_tzz, 1, TRCBYTES, fp_p);
		fwrite(data_tzz, sizeof(float), ns, fp_p);
		ntr_tzz++;
	}

	free(data_tzz);
	free(data_txx);
	fclose(fp_tzz);
	fclose(fp_txx);
	fclose(fp_p);

	if (verbose)
		vmess("computeHydrophone: wrote %d traces to %s", ntr_tzz, file_p);

	return 0;
}


int main(int argc, char **argv)
{
	modPar  mod;
	recPar  rec;
	snaPar  sna;
	wavPar  wav;
	srcPar  src;
	bndPar  bnd;
	shotPar shot;
	checkpointPar chk;
	adjSrcPar adj;
	float **src_nwav;
	float  sinkvel;
	double t0, t1, tinit;
	size_t nsamp, sizew, sizem;
	int    n1, ix, iz, ir, ishot, i;
	int    ioPx, ioPz;
	int    ixsrc, izsrc, fileno;
	int    verbose;

	/* FWI-specific parameters */
	char  *file_obs, *file_grad, *chk_base;
	char  *comp_str, *file_adj_snap;
	int    chk_skipdt, res_taper, param;
	float  misfit;
	float *grad1, *grad2, *grad3;  /* Gradient arrays (meaning depends on param) */
	float *grad_interior;

	t0 = wallclock_time();
	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;

	/* FWI parameters */
	if (!getparstring("file_obs", &file_obs))
		verr("file_obs= parameter is required (observed data base name).");
	if (!getparint("chk_skipdt", &chk_skipdt)) chk_skipdt = 500;
	if (!getparstring("chk_base", &chk_base)) chk_base = "chk";
	if (!getparstring("file_grad", &file_grad)) file_grad = "gradient";
	if (!getparstring("comp", &comp_str)) comp_str = "_rvz";
	if (!getparstring("file_adj_snap", &file_adj_snap)) file_adj_snap = "adj_snap";
	if (!getparint("res_taper", &res_taper)) res_taper = 100;
	if (!getparint("param", &param)) param = 1;  /* 1=Lamé (λ,μ,ρ), 2=velocity (Vp,Vs,ρ) */
	if (param < 1 || param > 2)
		verr("param must be 1 (Lame: lambda,mu,rho) or 2 (velocity: Vp,Vs,rho)");

	/* ============================================================ */
	/* Standard setup (same as test_fdfwimodc.c)                    */
	/* ============================================================ */
	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);
	n1 = mod.naz;

	allocStoreSourceOnSurface(src);

	readModel(&mod, &bnd);

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

	ir      = mod.ioZz + rec.z[0] + (rec.x[0] + mod.ioZx) * n1;
	rec.rho = mod.dt / (mod.dx * mod.roz[ir]);
	rec.cp  = sqrt(mod.l2m[ir] * (mod.roz[ir])) * mod.dx / mod.dt;

	t1 = wallclock_time();
	tinit = t1 - t0;

	/* Sinking */
	ioPx = mod.ioPx;
	ioPz = mod.ioPz;
	if (bnd.lef == 4 || bnd.lef == 2) ioPx += bnd.ntap;
	if (bnd.top == 4 || bnd.top == 2) ioPz += bnd.ntap;
	if (rec.sinkvel) sinkvel = mod.l2m[(rec.x[0] + ioPx) * n1 + rec.z[0] + ioPz];
	else sinkvel = 0.0;

	for (ir = 0; ir < rec.n; ir++) {
		iz = rec.z[ir];
		ix = rec.x[ir];
		while (mod.l2m[(ix + ioPx) * n1 + iz + ioPz] == sinkvel) iz++;
		rec.z[ir]  = iz + rec.sinkdepth;
		rec.zr[ir] = rec.zr[ir] + (rec.z[ir] - iz) * mod.dz;
	}

	for (ishot = 0; ishot < shot.n; ishot++) {
		iz = shot.z[ishot];
		ix = shot.x[ishot];
		while (mod.l2m[(ix + ioPx) * n1 + iz + ioPz] == 0.0) iz++;
		shot.z[ishot] = iz + src.sinkdepth;
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

	/* ============================================================ */
	/* Process first shot only for this test                        */
	/* ============================================================ */
	izsrc  = shot.z[0];
	ixsrc  = shot.x[0];
	fileno = 0;
	sizem  = (size_t)mod.nax * mod.naz;

	vmess("*******************************************");
	vmess("***** FWI GRADIENT TEST                *****");
	vmess("*******************************************");
	vmess("Source at grid ix=%d iz=%d (x=%.1f z=%.1f)",
		ixsrc, izsrc,
		mod.x0 + mod.dx * ixsrc, mod.z0 + mod.dz * izsrc);
	vmess("Checkpoint interval: %d time steps", chk_skipdt);
	vmess("Observed data base: %s", file_obs);

	/* ============================================================ */
	/* STEP 1: Forward modeling with checkpointing                  */
	/* ============================================================ */
	vmess("--- Step 1: Forward modeling with checkpointing ---");

	initCheckpoints(&chk, &mod, chk_skipdt, 0, chk_base);

	fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna,
		ixsrc, izsrc, src_nwav, 0, 1, fileno, &chk, verbose);

	t1 = wallclock_time();
	vmess("Forward modeling completed in %.2f s", t1 - t0 - tinit);

	/* Compute synthetic hydrophone (pressure) for elastic modeling.
	 * For elastic (ischeme > 2), pressure is P = 0.5*(Tzz + Txx).
	 * fdelmodc records Tzz and Txx separately, so we compute P here. */
	if (mod.ischeme > 2) {
		char syn_tzz[1024], syn_txx[1024], syn_p[1024];
		snprintf(syn_tzz, sizeof(syn_tzz), "%s_rtzz.su", rec.file_rcv);
		snprintf(syn_txx, sizeof(syn_txx), "%s_rtxx.su", rec.file_rcv);
		snprintf(syn_p, sizeof(syn_p), "%s_rp.su", rec.file_rcv);
		vmess("Computing synthetic hydrophone: P = 0.5*(Tzz + Txx)");
		computeHydrophone(syn_tzz, syn_txx, syn_p, verbose);
	}

	/* ============================================================ */
	/* STEP 2: Compute residuals                                    */
	/* ============================================================ */
	vmess("--- Step 2: Compute residuals ---");

	{
		/* Parse comma-separated component suffixes and build filename arrays */
		char comp_buf[1024];
		char *comp_suffixes[MAX_COMP];
		int ncomp = 0;
		char *token;
		int k;

		strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
		comp_buf[sizeof(comp_buf) - 1] = '\0';
		token = strtok(comp_buf, ",");
		while (token && ncomp < MAX_COMP) {
			comp_suffixes[ncomp++] = token;
			token = strtok(NULL, ",");
		}
		if (ncomp == 0)
			verr("comp= parameter is empty or invalid.");

		vmess("Number of residual components: %d", ncomp);

		{
			char obs_names[MAX_COMP][1024], syn_names[MAX_COMP][1024];
			const char *obs_arr[MAX_COMP], *syn_arr[MAX_COMP];

			for (k = 0; k < ncomp; k++) {
				snprintf(obs_names[k], sizeof(obs_names[k]), "%s%s.su", file_obs, comp_suffixes[k]);
				snprintf(syn_names[k], sizeof(syn_names[k]), "%s%s.su", rec.file_rcv, comp_suffixes[k]);
				obs_arr[k] = obs_names[k];
				syn_arr[k] = syn_names[k];
				vmess("  Component %d: obs=%s  syn=%s", k + 1, obs_arr[k], syn_arr[k]);
			}

			misfit = computeResidual(ncomp, obs_arr, syn_arr,
				"residual.su", MISFIT_L2, verbose);
		}

		vmess("L2 misfit = %.6e", misfit);
	}

	/* ============================================================ */
	/* STEP 3: Read residuals                                       */
	/* ============================================================ */
	vmess("--- Step 3: Read residuals ---");

	memset(&adj, 0, sizeof(adjSrcPar));
	readResidual("residual.su", &adj, &mod, &bnd);
	vmess("Loaded %d adjoint source traces, nt=%d", adj.nsrc, adj.nt);

	/* Apply cosine taper to residual trace ends to suppress ringing
	 * in the adjoint wavefield (same purpose as sutaper tend= in
	 * fdacrtmc examples). */
	if (res_taper > 0 && adj.nsrc > 0) {
		int k, it, ntap;
		ntap = MIN(res_taper, adj.nt / 2);
		for (k = 0; k < adj.nsrc; k++) {
			float *trace = &adj.wav[k * adj.nt];
			for (it = 0; it < ntap; it++) {
				float w = 0.5f * (1.0f - cosf(M_PI * (float)it / (float)ntap));
				trace[it] *= w;
				trace[adj.nt - 1 - it] *= w;
			}
		}
		vmess("Applied cosine taper (%d samples) to %d residual traces", ntap, adj.nsrc);
	}

	/* ============================================================ */
	/* STEP 4: Adjoint backpropagation                              */
	/* ============================================================ */
	vmess("--- Step 4: Adjoint backpropagation ---");
	vmess("Parameterization: param=%d (%s)", param,
		param == 1 ? "Lame: lambda, mu, rho" : "Velocity: Vp, Vs, rho");

	/* Allocate gradient arrays.
	 * param=1: grad1=λ, grad2=μ, grad3=ρ_direct (Lamé)
	 * param=2: grad1=Vp, grad2=Vs, grad3=ρ_full (velocity with chain rule)
	 */
	grad1 = (float *)calloc(sizem, sizeof(float));
	grad2 = NULL;
	grad3 = (float *)calloc(sizem, sizeof(float));

	if (mod.ischeme > 2) {
		grad2 = (float *)calloc(sizem, sizeof(float));
	}

	t1 = wallclock_time();

	{
		/* Override snapshot base name for adjoint wavefield */
		char *orig_snap_name = sna.file_snap;
		sna.file_snap = file_adj_snap;

		adj_shot(&mod, &src, &wav, &bnd, &rec, &adj,
			ixsrc, izsrc, src_nwav, &chk, &sna,
			grad1, grad2, grad3, param, verbose);

		sna.file_snap = orig_snap_name;
	}

	vmess("Adjoint pass completed in %.2f s", wallclock_time() - t1);

	/* ============================================================ */
	/* STEP 5: Write gradient to SU files                           */
	/* ============================================================ */
	vmess("--- Step 5: Write gradient ---");

	{
		char fname[1024];
		float gmax;
		const char *label1, *label2, *label3;
		const char *suffix1, *suffix2, *suffix3;

		/* Set labels and suffixes based on parameterization */
		if (param == 1) {
			label1 = "lam"; label2 = "muu"; label3 = "rho";
			suffix1 = "_lam"; suffix2 = "_muu"; suffix3 = "_rho";
		} else {
			label1 = "Vp";  label2 = "Vs";  label3 = "rho";
			suffix1 = "_vp"; suffix2 = "_vs"; suffix3 = "_rho";
		}

		grad_interior = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));

		/* grad1: λ (param=1) or Vp (param=2) */
		if (grad1) {
			stripBoundary(grad_interior, grad1, &mod, &bnd);
			snprintf(fname, sizeof(fname), "%s%s.su", file_grad, suffix1);
			writesufile(fname, grad_interior, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);

			gmax = 0.0f;
			for (i = 0; i < mod.nx * mod.nz; i++)
				if (fabsf(grad_interior[i]) > gmax) gmax = fabsf(grad_interior[i]);
			vmess("Gradient %s: max|g| = %.6e  -> %s", label1, gmax, fname);
		}

		/* grad2: μ (param=1) or Vs (param=2) */
		if (grad2) {
			stripBoundary(grad_interior, grad2, &mod, &bnd);
			snprintf(fname, sizeof(fname), "%s%s.su", file_grad, suffix2);
			writesufile(fname, grad_interior, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);

			gmax = 0.0f;
			for (i = 0; i < mod.nx * mod.nz; i++)
				if (fabsf(grad_interior[i]) > gmax) gmax = fabsf(grad_interior[i]);
			vmess("Gradient %s: max|g| = %.6e  -> %s", label2, gmax, fname);
		}

		/* grad3: ρ (both parameterizations, but different formulas) */
		if (grad3) {
			stripBoundary(grad_interior, grad3, &mod, &bnd);
			snprintf(fname, sizeof(fname), "%s%s.su", file_grad, suffix3);
			writesufile(fname, grad_interior, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);

			gmax = 0.0f;
			for (i = 0; i < mod.nx * mod.nz; i++)
				if (fabsf(grad_interior[i]) > gmax) gmax = fabsf(grad_interior[i]);
			vmess("Gradient %s: max|g| = %.6e  -> %s", label3, gmax, fname);
		}

		free(grad_interior);
	}

	/* ============================================================ */
	/* Summary                                                       */
	/* ============================================================ */
	vmess("*******************************************");
	vmess("***** GRADIENT TEST SUMMARY            *****");
	vmess("*******************************************");
	vmess("Total L2 misfit:  %.6e", misfit);
	vmess("Total wall time:  %.2f s", wallclock_time() - t0);
	vmess("Checkpoints:      %d (skipdt=%d)", chk.nsnap, chk_skipdt);
	vmess("Adjoint sources:  %d traces", adj.nsrc);
	vmess("*******************************************");

	/* ============================================================ */
	/* Cleanup                                                       */
	/* ============================================================ */
	cleanCheckpoints(&chk);
	freeResidual(&adj);

	if (grad1) free(grad1);
	if (grad2) free(grad2);
	if (grad3) free(grad3);

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
