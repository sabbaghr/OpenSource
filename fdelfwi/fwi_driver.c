/*
 * fwi_driver.c - Driver for elastic Full Waveform Inversion.
 *
 * Can be compiled with or without MPI:
 *   - With MPI (-DUSE_MPI): distributes shots across ranks, reduces gradients
 *   - Without MPI: processes all shots serially on single process
 *
 * Pipeline per shot:
 *   1. Forward modeling with disk checkpointing (fdfwimodc)
 *   2. Residual computation (computeResidual)
 *   3. Residual parsing (readResidual)
 *   4. Adjoint backpropagation with gradient accumulation (adj_shot)
 *
 * Usage:
 *   Serial:    fwi_driver <parameters>
 *   Parallel:  mpirun -np <nproc> fwi_mpi_driver <parameters>
 *
 * Parameters (same as fdelmodc, plus):
 *   file_obs=     base name of observed data (required)
 *   chk_skipdt=   time steps between checkpoints (default 500)
 *   chk_base=     base path for checkpoint files (default "chk")
 *   file_grad=    base name for gradient output (default "gradient")
 *   comp=         comma-separated component suffixes (default "_rvz")
 *   param=        parameterization: 1=Lame, 2=velocity (default 1)
 *
 * Shot distribution (MPI only):
 *   Shots are assigned to ranks in round-robin fashion.
 *   Each rank creates its own working directory for checkpoints.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "par.h"
#include "segy.h"
#include "fdelfwi.h"

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

void vmess(char *fmt, ...);
void verr(char *fmt, ...);


char *sdoc[] = {
" ",
#ifdef USE_MPI
" fwi_mpi_driver - MPI-parallel elastic FWI gradient computation",
" ",
" mpirun -np <nproc> fwi_mpi_driver <parameters>",
#else
" fwi_driver - Serial elastic FWI gradient computation",
" ",
" fwi_driver <parameters>",
#endif
" ",
" Same parameters as fdelmodc, plus:",
"   file_obs=     observed data base name (required)",
"   chk_skipdt=   checkpoint interval in time steps (500)",
"   chk_base=     checkpoint file prefix (chk)",
"   file_grad=    gradient output base name (gradient)",
"   comp=         comma-separated component suffixes (_rvz)",
"   param=1       parameterization: 1=Lame (lambda,mu,rho), 2=velocity (Vp,Vs,rho)",
"   res_taper=100 cosine taper length for residual traces",
" ",
NULL};


/*--------------------------------------------------------------------
 * stripBoundary -- Copy interior model from padded array to dense array.
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
 *--------------------------------------------------------------------*/
static int computeHydrophone(const char *file_tzz, const char *file_txx,
                             const char *file_p, int verbose)
{
	FILE *fp_tzz, *fp_txx, *fp_p;
	segy hdr_tzz, hdr_txx;
	float *data_tzz, *data_txx;
	int ns, ntr, is;
	size_t nread;

	fp_tzz = fopen(file_tzz, "r");
	fp_txx = fopen(file_txx, "r");
	if (!fp_tzz || !fp_txx) {
		if (fp_tzz) fclose(fp_tzz);
		if (fp_txx) fclose(fp_txx);
		return -1;
	}

	fp_p = fopen(file_p, "w");
	if (!fp_p) {
		fclose(fp_tzz);
		fclose(fp_txx);
		return -1;
	}

	nread = fread(&hdr_tzz, 1, TRCBYTES, fp_tzz);
	if (nread != TRCBYTES) {
		fclose(fp_tzz); fclose(fp_txx); fclose(fp_p);
		return -1;
	}
	rewind(fp_tzz);

	ns = hdr_tzz.ns;
	data_tzz = (float *)malloc(ns * sizeof(float));
	data_txx = (float *)malloc(ns * sizeof(float));

	ntr = 0;
	while (fread(&hdr_tzz, 1, TRCBYTES, fp_tzz) == TRCBYTES &&
	       fread(&hdr_txx, 1, TRCBYTES, fp_txx) == TRCBYTES) {

		if (hdr_tzz.ns != ns || hdr_txx.ns != ns) break;

		nread = fread(data_tzz, sizeof(float), ns, fp_tzz);
		if ((int)nread != ns) break;
		nread = fread(data_txx, sizeof(float), ns, fp_txx);
		if ((int)nread != ns) break;

		for (is = 0; is < ns; is++)
			data_tzz[is] = 0.5f * (data_tzz[is] + data_txx[is]);

		hdr_tzz.trid = 11;
		fwrite(&hdr_tzz, 1, TRCBYTES, fp_p);
		fwrite(data_tzz, sizeof(float), ns, fp_p);
		ntr++;
	}

	free(data_tzz);
	free(data_txx);
	fclose(fp_tzz);
	fclose(fp_txx);
	fclose(fp_p);

	return 0;
}


/*--------------------------------------------------------------------
 * applyCosineTaper -- Apply cosine taper to residual trace ends.
 *--------------------------------------------------------------------*/
static void applyCosineTaper(adjSrcPar *adj, int taper_len)
{
	int isrc, it;
	float *wav;
	float w;

	if (taper_len <= 0 || adj->nt <= 2 * taper_len) return;

	for (isrc = 0; isrc < adj->nsrc; isrc++) {
		wav = &adj->wav[isrc * adj->nt];
		for (it = 0; it < taper_len; it++) {
			w = 0.5f * (1.0f - cosf(M_PI * it / taper_len));
			wav[it] *= w;
			wav[adj->nt - 1 - it] *= w;
		}
	}
}


int main(int argc, char **argv)
{
	/* MPI/parallel variables */
	int mpi_rank = 0, mpi_size = 1;
	int my_nshots, *my_shots;
	double t_start, t_shot, t_total;

	/* Model and simulation parameters */
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
	size_t nsamp, sizew, sizem;
	int    n1, ix, iz, ir, ishot, i, k;
	int    ioPx, ioPz;
	int    ixsrc, izsrc, fileno;
	int    verbose;

	/* FWI-specific parameters */
	char  *file_obs, *file_grad, *chk_base;
	char  *comp_str;
	int    chk_skipdt, res_taper, param;
	float  misfit, total_misfit;
	float *grad1_local, *grad2_local, *grad3_local;  /* Local shot gradients */
	float *grad1_global, *grad2_global, *grad3_global; /* Reduced gradients (or same as local for serial) */
	float *grad_interior;
	char   work_dir[256], chk_path[512];

#ifdef USE_MPI
	/* Initialize MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	t_start = MPI_Wtime();
#else
	t_start = wallclock_time();
#endif

	/* Only rank 0 parses arguments initially */
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
	if (!getparint("res_taper", &res_taper)) res_taper = 100;
	if (!getparint("param", &param)) param = 1;
	if (param < 1 || param > 2)
		verr("param must be 1 (Lame) or 2 (velocity)");

	/* ============================================================ */
	/* Standard setup                                                */
	/* ============================================================ */
	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);
	n1 = mod.naz;
	sizem = (size_t)mod.nax * mod.naz;

	allocStoreSourceOnSurface(src);
	readModel(&mod, &bnd);

	/* Allocate and define source wavelets */
	if (wav.nx > 1) {
		nsamp = 0;
		sizew = wav.nt * wav.nx;
		src_nwav = (float **)calloc(wav.nx, sizeof(float *));
		for (i = 0; i < wav.nx; i++) {
			src_nwav[i] = (float *)calloc(wav.nt, sizeof(float));
		}
	} else {
		sizew = wav.nt * wav.nx;
		src_nwav = (float **)calloc(wav.nx, sizeof(float *));
		src_nwav[0] = (float *)calloc(sizew, sizeof(float));
		assert(src_nwav[0] != NULL);
		for (i = 0; i < wav.nx; i++)
			src_nwav[i] = (float *)(src_nwav[0] + (size_t)(wav.nt * i));
	}

	defineSource(wav, src, mod, rec, shot, src_nwav, mod.grid_dir, verbose);
	if (nsamp == 0) nsamp = 1;

	/* Sinking receivers */
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
	/* Distribute shots across MPI ranks (round-robin)               */
	/* ============================================================ */
	my_nshots = 0;
	for (ishot = mpi_rank; ishot < shot.n; ishot += mpi_size)
		my_nshots++;

	my_shots = (int *)malloc(my_nshots * sizeof(int));
	k = 0;
	for (ishot = mpi_rank; ishot < shot.n; ishot += mpi_size)
		my_shots[k++] = ishot;

	if (mpi_rank == 0) {
		vmess("*******************************************");
#ifdef USE_MPI
		vmess("***** MPI FWI GRADIENT DRIVER          *****");
#else
		vmess("***** FWI GRADIENT DRIVER (Serial)     *****");
#endif
		vmess("*******************************************");
		vmess("Total shots: %d", shot.n);
#ifdef USE_MPI
		vmess("MPI ranks: %d", mpi_size);
#endif
		vmess("Parameterization: %s", param == 1 ? "Lame (lambda, mu, rho)" : "Velocity (Vp, Vs, rho)");
	}

	if (verbose)
		vmess("Rank %d: processing %d shots", mpi_rank, my_nshots);

	/* ============================================================ */
	/* Allocate gradient arrays                                      */
	/* ============================================================ */
	grad1_local = (float *)calloc(sizem, sizeof(float));
	grad3_local = (float *)calloc(sizem, sizeof(float));
	grad2_local = NULL;
	if (mod.ischeme > 2)
		grad2_local = (float *)calloc(sizem, sizeof(float));

	grad1_global = (float *)calloc(sizem, sizeof(float));
	grad3_global = (float *)calloc(sizem, sizeof(float));
	grad2_global = NULL;
	if (mod.ischeme > 2)
		grad2_global = (float *)calloc(sizem, sizeof(float));

	/* Create per-rank working directory for checkpoints */
	snprintf(work_dir, sizeof(work_dir), "fwi_rank%03d", mpi_rank);
	mkdir(work_dir, 0755);
	snprintf(chk_path, sizeof(chk_path), "%s/%s", work_dir, chk_base);

	/* ============================================================ */
	/* Process shots assigned to this rank                           */
	/* ============================================================ */
	total_misfit = 0.0f;

	for (k = 0; k < my_nshots; k++) {
		ishot = my_shots[k];
		izsrc = shot.z[ishot];
		ixsrc = shot.x[ishot];
		fileno = ishot;

#ifdef USE_MPI
		t_shot = MPI_Wtime();
#else
		t_shot = wallclock_time();
#endif

		if (verbose)
			vmess("Rank %d: Shot %d/%d (global %d) at x=%.1f z=%.1f",
				mpi_rank, k + 1, my_nshots, ishot,
				mod.x0 + mod.dx * ixsrc, mod.z0 + mod.dz * izsrc);

		/* Temporary arrays for this shot's gradient contribution */
		float *shot_grad1 = (float *)calloc(sizem, sizeof(float));
		float *shot_grad2 = NULL;
		float *shot_grad3 = (float *)calloc(sizem, sizeof(float));
		if (mod.ischeme > 2)
			shot_grad2 = (float *)calloc(sizem, sizeof(float));

		/* -------------------------------------------------------- */
		/* Step 1: Forward modeling with checkpointing               */
		/* -------------------------------------------------------- */
		initCheckpoints(&chk, &mod, chk_skipdt, 0, chk_path);

		/* Disable snapshot output for parallel runs */
		snaPar sna_off = sna;
		sna_off.nsnap = 0;

		fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna_off,
			ixsrc, izsrc, src_nwav, ishot, shot.n, fileno, &chk, verbose);

		/* Compute synthetic hydrophone for elastic */
		if (mod.ischeme > 2) {
			char syn_tzz[512], syn_txx[512], syn_p[512];
			snprintf(syn_tzz, sizeof(syn_tzz), "%s_%03d_rtzz.su", rec.file_rcv, fileno);
			snprintf(syn_txx, sizeof(syn_txx), "%s_%03d_rtxx.su", rec.file_rcv, fileno);
			snprintf(syn_p, sizeof(syn_p), "%s_%03d_rp.su", rec.file_rcv, fileno);
			computeHydrophone(syn_tzz, syn_txx, syn_p, 0);
		}

		/* -------------------------------------------------------- */
		/* Step 2: Compute residuals                                 */
		/* -------------------------------------------------------- */
		{
			char comp_buf[1024];
			char *comp_suffixes[MAX_COMP];
			int ncomp = 0;
			char *token;

			strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
			comp_buf[sizeof(comp_buf) - 1] = '\0';
			token = strtok(comp_buf, ",");
			while (token && ncomp < MAX_COMP) {
				comp_suffixes[ncomp++] = token;
				token = strtok(NULL, ",");
			}

			char obs_names[MAX_COMP][512], syn_names[MAX_COMP][512];
			const char *obs_arr[MAX_COMP], *syn_arr[MAX_COMP];
			char res_file[512];

			for (i = 0; i < ncomp; i++) {
				snprintf(obs_names[i], sizeof(obs_names[i]), "%s_%03d%s.su",
					file_obs, ishot, comp_suffixes[i]);
				snprintf(syn_names[i], sizeof(syn_names[i]), "%s_%03d%s.su",
					rec.file_rcv, fileno, comp_suffixes[i]);
				obs_arr[i] = obs_names[i];
				syn_arr[i] = syn_names[i];
			}

			snprintf(res_file, sizeof(res_file), "%s/residual.su", work_dir);
			misfit = computeResidual(ncomp, obs_arr, syn_arr, res_file, MISFIT_L2, 0);
			total_misfit += misfit;

			/* -------------------------------------------------------- */
			/* Step 3: Read residuals and apply taper                   */
			/* -------------------------------------------------------- */
			memset(&adj, 0, sizeof(adjSrcPar));
			readResidual(res_file, &adj, &mod, &bnd);
			applyCosineTaper(&adj, res_taper);

			/* -------------------------------------------------------- */
			/* Step 4: Adjoint backpropagation                          */
			/* -------------------------------------------------------- */
			adj_shot(&mod, &src, &wav, &bnd, &rec, &adj,
				ixsrc, izsrc, src_nwav, &chk, NULL,
				shot_grad1, shot_grad2, shot_grad3, param, 0);

			freeResidual(&adj);
		}

		/* Accumulate this shot's gradient into local sum */
		for (i = 0; i < (int)sizem; i++) {
			grad1_local[i] += shot_grad1[i];
			grad3_local[i] += shot_grad3[i];
			if (grad2_local && shot_grad2)
				grad2_local[i] += shot_grad2[i];
		}

		/* Clean up shot-specific arrays */
		free(shot_grad1);
		free(shot_grad3);
		if (shot_grad2) free(shot_grad2);
		cleanCheckpoints(&chk);

		if (verbose) {
#ifdef USE_MPI
			vmess("Rank %d: Shot %d completed in %.2f s, misfit=%.6e",
				mpi_rank, ishot, MPI_Wtime() - t_shot, misfit);
#else
			vmess("Shot %d completed in %.2f s, misfit=%.6e",
				ishot, wallclock_time() - t_shot, misfit);
#endif
		}
	}

	/* ============================================================ */
	/* Reduce gradients (MPI) or copy to global (serial)             */
	/* ============================================================ */
	float global_misfit;

#ifdef USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);

	if (mpi_rank == 0 && verbose)
		vmess("Reducing gradients across %d ranks...", mpi_size);

	MPI_Allreduce(grad1_local, grad1_global, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(grad3_local, grad3_global, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	if (grad2_local && grad2_global)
		MPI_Allreduce(grad2_local, grad2_global, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

	MPI_Allreduce(&total_misfit, &global_misfit, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
	/* Serial: local gradients are the global gradients */
	memcpy(grad1_global, grad1_local, sizem * sizeof(float));
	memcpy(grad3_global, grad3_local, sizem * sizeof(float));
	if (grad2_local && grad2_global)
		memcpy(grad2_global, grad2_local, sizem * sizeof(float));
	global_misfit = total_misfit;
#endif

	/* ============================================================ */
	/* Rank 0: Write gradient output                                 */
	/* ============================================================ */
	if (mpi_rank == 0) {
		char fname[512];
		float gmax;

		vmess("Total L2 misfit = %.6e", global_misfit);
		vmess("Writing gradients...");

		grad_interior = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));

		/* Gradient 1: lambda (param=1) or Vp (param=2) */
		stripBoundary(grad_interior, grad1_global, &mod, &bnd);
		if (param == 1)
			snprintf(fname, sizeof(fname), "%s_lam.su", file_grad);
		else
			snprintf(fname, sizeof(fname), "%s_vp.su", file_grad);
		writesufile(fname, grad_interior, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);

		gmax = 0.0f;
		for (i = 0; i < mod.nx * mod.nz; i++)
			if (fabsf(grad_interior[i]) > gmax) gmax = fabsf(grad_interior[i]);
		vmess("  %s: max|g| = %.6e", fname, gmax);

		/* Gradient 2: mu (param=1) or Vs (param=2) */
		if (grad2_global) {
			stripBoundary(grad_interior, grad2_global, &mod, &bnd);
			if (param == 1)
				snprintf(fname, sizeof(fname), "%s_muu.su", file_grad);
			else
				snprintf(fname, sizeof(fname), "%s_vs.su", file_grad);
			writesufile(fname, grad_interior, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);

			gmax = 0.0f;
			for (i = 0; i < mod.nx * mod.nz; i++)
				if (fabsf(grad_interior[i]) > gmax) gmax = fabsf(grad_interior[i]);
			vmess("  %s: max|g| = %.6e", fname, gmax);
		}

		/* Gradient 3: rho */
		stripBoundary(grad_interior, grad3_global, &mod, &bnd);
		snprintf(fname, sizeof(fname), "%s_rho.su", file_grad);
		writesufile(fname, grad_interior, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);

		gmax = 0.0f;
		for (i = 0; i < mod.nx * mod.nz; i++)
			if (fabsf(grad_interior[i]) > gmax) gmax = fabsf(grad_interior[i]);
		vmess("  %s: max|g| = %.6e", fname, gmax);

		free(grad_interior);

#ifdef USE_MPI
		t_total = MPI_Wtime() - t_start;
#else
		t_total = wallclock_time() - t_start;
#endif
		vmess("*******************************************");
		vmess("FWI gradient completed in %.2f s", t_total);
		vmess("  Shots processed: %d", shot.n);
		vmess("  Total misfit: %.6e", global_misfit);
		vmess("*******************************************");
	}

	/* ============================================================ */
	/* Cleanup                                                       */
	/* ============================================================ */
	free(grad1_local); free(grad1_global);
	free(grad3_local); free(grad3_global);
	if (grad2_local) free(grad2_local);
	if (grad2_global) free(grad2_global);
	free(my_shots);

	for (k = 0; k < wav.nx; k++)
		free(src_nwav[k]);
	free(src_nwav);
	freeStoreSourceOnSurface();

#ifdef USE_MPI
	MPI_Finalize();
#endif
	return 0;
}
