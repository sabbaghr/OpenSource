/*
 * fwi_inversion.c - Full Waveform Inversion driver with optimization loop.
 *
 * Wraps the existing FWI gradient pipeline (forward, residual, adjoint)
 * in an iterative optimization loop. Supports all algorithms from the
 * SEISCOPE-translated optimization library:
 *   0 = Steepest Descent
 *   1 = L-BFGS (default)
 *   2 = PLBFGS (preconditioned L-BFGS, identity preconditioner)
 *   3 = PNLCG (Dai-Yuan conjugate gradient)
 *
 * Can be compiled with or without MPI (-DUSE_MPI).
 * MPI parallelization is over shots; the optimizer runs on rank 0 only.
 *
 * Usage:
 *   Serial:    fwi_inversion <parameters>
 *   Parallel:  mpirun -np N fwi_mpi_inversion <parameters>
 *
 * Additional parameters (beyond fwi_driver):
 *   niter=20        max optimization iterations
 *   conv=1e-6       convergence tolerance (fcost/f0)
 *   algorithm=1     0=SD, 1=LBFGS, 2=PLBFGS, 3=PNLCG
 *   lbfgs_mem=20    L-BFGS history pairs (for algorithm 1,2)
 *   nls_max=20      max linesearch iterations per step
 *   vp_min=,vp_max= Vp bounds (m/s, optional)
 *   vs_min=,vs_max= Vs bounds (m/s, optional)
 *   rho_min=,rho_max= density bounds (kg/m3, optional)
 *   write_iter=1    write model every N iterations
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
#include "optim.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))
#define MAX_COMP 8

/* External declarations (same as fwi_driver.c) */
double wallclock_time(void);
void threadAffinity(void);

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);
int readModel(modPar *mod, bndPar *bnd);
int defineSource(wavPar wav, srcPar src, modPar mod, recPar rec,
                 shotPar shot, float **src_nwav, int reverse, int verbose);
int allocStoreSourceOnSurface(srcPar src);
int freeStoreSourceOnSurface(void);

int fdfwimodc(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec, snaPar *sna, int ixsrc, int izsrc,
              float **src_nwav, int ishot, int nshots, int fileno,
              checkpointPar *chk, int verbose);

int initCheckpoints(checkpointPar *chk, modPar *mod, int skipdt,
                    int delay, const char *file_base);
int cleanCheckpoints(checkpointPar *chk);

float computeDataRMS(const char *filename);
float computeResidual(int ncomp, const char **obs_files, const char **syn_files,
                      const char *res_file, misfitType mtype,
                      const float *comp_weights, int verbose);

int readResidual(const char *filename, adjSrcPar *adj, modPar *mod, bndPar *bnd);
void freeResidual(adjSrcPar *adj);

int adj_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
             recPar *rec, adjSrcPar *adj,
             int ixsrc, int izsrc, float **src_nwav,
             checkpointPar *chk, snaPar *sna,
             float *grad1, float *grad2, float *grad3,
             float *hess_lam, float *hess_muu, float *hess_rho,
             float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
             int param, int verbose);

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

void writeSnapTimesReset(void);
void vmess(char *fmt, ...);
void verr(char *fmt, ...);

/* From updateModel.c */
void recomputeFDcoefficients(modPar *mod, bndPar *bnd);
void extractModelVector(float *x, modPar *mod, bndPar *bnd, int param);
void injectModelVector(float *x, modPar *mod, bndPar *bnd, int param);
void extractGradientVector(float *g, float *grad1, float *grad2, float *grad3,
                           modPar *mod, bndPar *bnd, int param);
void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho, size_t sizem);


char *sdoc[] = {
" ",
#ifdef USE_MPI
" fwi_mpi_inversion - MPI-parallel elastic FWI with optimization",
" ",
" mpirun -np <nproc> fwi_mpi_inversion <parameters>",
#else
" fwi_inversion - Serial elastic FWI with optimization",
" ",
" fwi_inversion <parameters>",
#endif
" ",
" Same parameters as fwi_driver, plus:",
"   niter=20           max optimization iterations",
"   conv=1e-6          convergence tolerance (fcost/f0)",
"   algorithm=1        0=SD, 1=LBFGS, 2=PLBFGS, 3=PNLCG, 4=TRN, 5=ENRICHED, 6=PTRN, 7=PENRICHED",
"   niter_cg=5         max inner CG iterations per TRN/Enriched step (algorithm=4,5,7)",
"   enr_l=20           L-BFGS cycle length for Enriched method (algorithm=5,7)",
"   lbfgs_mem=20       L-BFGS history pairs",
"   nls_max=20         max linesearch iterations",
"   vp_min=,vp_max=    Vp bounds (m/s, optional)",
"   vs_min=,vs_max=    Vs bounds (m/s, optional)",
"   rho_min=,rho_max=  density bounds (kg/m3, optional)",
"   write_iter=1       write model every N iterations",
"   scaling=0          parameter scaling: 0=none, 1=Brossier(mean), 2=Yang(max)",
"   precond=0          Yang pseudo-Hessian preconditioner (0=off, 1=on)",
"   precond_eps=1e-3   regularization epsilon for preconditioner",
"   precond_scale_1/2/3=1.0  manual Yang scaling (overridden by scaling>0)",
"   data_weight=auto    Brossier W_d data weighting: 0=off, 1=on (auto=1 when scaling>0)",
" ",
NULL};


/* Yang block preconditioner functions in yang_precond.c */


/*--------------------------------------------------------------------
 * computeHydrophone -- Compute pressure P = 0.5*(Tzz + Txx).
 *--------------------------------------------------------------------*/
static int computeHydrophone(const char *file_tzz, const char *file_txx,
                             const char *file_p, int verbose)
{
	FILE *fp_tzz, *fp_txx, *fp_p;
	segy hdr_tzz, hdr_txx;
	float *data_tzz, *data_txx;
	int ns;
	size_t nread;

	fp_tzz = fopen(file_tzz, "r");
	fp_txx = fopen(file_txx, "r");
	if (!fp_tzz || !fp_txx) {
		if (fp_tzz) fclose(fp_tzz);
		if (fp_txx) fclose(fp_txx);
		return -1;
	}

	fp_p = fopen(file_p, "w");
	if (!fp_p) { fclose(fp_tzz); fclose(fp_txx); return -1; }

	nread = fread(&hdr_tzz, 1, TRCBYTES, fp_tzz);
	if (nread != TRCBYTES) {
		fclose(fp_tzz); fclose(fp_txx); fclose(fp_p);
		return -1;
	}
	rewind(fp_tzz);

	ns = hdr_tzz.ns;
	data_tzz = (float *)malloc(ns * sizeof(float));
	data_txx = (float *)malloc(ns * sizeof(float));

	while (fread(&hdr_tzz, 1, TRCBYTES, fp_tzz) == TRCBYTES &&
	       fread(&hdr_txx, 1, TRCBYTES, fp_txx) == TRCBYTES) {
		if (hdr_tzz.ns != ns || hdr_txx.ns != ns) break;
		nread = fread(data_tzz, sizeof(float), ns, fp_tzz);
		if ((int)nread != ns) break;
		nread = fread(data_txx, sizeof(float), ns, fp_txx);
		if ((int)nread != ns) break;
		for (int is = 0; is < ns; is++)
			data_tzz[is] = 0.5f * (data_tzz[is] + data_txx[is]);
		hdr_tzz.trid = 11;
		fwrite(&hdr_tzz, 1, TRCBYTES, fp_p);
		fwrite(data_tzz, sizeof(float), ns, fp_p);
	}

	free(data_tzz); free(data_txx);
	fclose(fp_tzz); fclose(fp_txx); fclose(fp_p);
	return 0;
}


/*--------------------------------------------------------------------
 * applyCosineTaper -- Apply cosine taper to residual trace ends.
 *--------------------------------------------------------------------*/
static void applyCosineTaper(adjSrcPar *adj, int taper_len)
{
	int isrc, it;
	float *wav, w;

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


/*--------------------------------------------------------------------
 * taperGradientSrcRcv -- Suppress gradient near source/receiver.
 *
 * The FWI gradient (cross-correlation of forward and adjoint wavefields)
 * has artificially large values near sources and receivers where the
 * wavefield amplitudes are highest.  This applies a Gaussian taper that
 * smoothly ramps from 0 at each source/receiver position to 1 at a
 * distance of `radius` grid points.
 *
 * grad: padded 2D array [nax * naz], indexed as grad[ix*n1 + iz]
 * n1:   fast dimension (naz)
 * nax:  total padded x dimension
 * shot: source positions (grid indices, interior-relative)
 * rec:  receiver positions (grid indices, interior-relative)
 * ibndx, ibndz: boundary offsets to convert to padded indices
 * radius: taper radius in grid points (0 = no taper)
 *--------------------------------------------------------------------*/
static void taperGradientSrcRcv(float *grad, int n1, int nax,
	shotPar *shot, recPar *rec,
	int ibndx, int ibndz, int radius)
{
	if (radius <= 0) return;

	int ix, iz, k;
	float r2max = (float)radius * (float)radius;

	/* Build a 1D taper mask (initially all ones) */
	size_t sizem = (size_t)nax * n1;
	float *mask = (float *)malloc(sizem * sizeof(float));
	for (k = 0; k < (int)sizem; k++) mask[k] = 1.0f;

	/* Apply Gaussian hole around each source */
	for (k = 0; k < shot->n; k++) {
		int sx = shot->x[k] + ibndx;
		int sz = shot->z[k] + ibndz;
		int x0 = sx - radius; if (x0 < 0) x0 = 0;
		int x1 = sx + radius; if (x1 >= nax) x1 = nax - 1;
		int z0 = sz - radius; if (z0 < 0) z0 = 0;
		int z1 = sz + radius; if (z1 >= n1) z1 = n1 - 1;
		for (ix = x0; ix <= x1; ix++) {
			for (iz = z0; iz <= z1; iz++) {
				float dx = (float)(ix - sx);
				float dz = (float)(iz - sz);
				float r2 = dx * dx + dz * dz;
				if (r2 < r2max) {
					float w = r2 / r2max; /* 0 at center, 1 at edge */
					if (w < mask[ix * n1 + iz])
						mask[ix * n1 + iz] = w;
				}
			}
		}
	}

	/* Apply Gaussian hole around each receiver */
	for (k = 0; k < rec->n; k++) {
		int rx = rec->x[k] + ibndx;
		int rz = rec->z[k] + ibndz;
		int x0 = rx - radius; if (x0 < 0) x0 = 0;
		int x1 = rx + radius; if (x1 >= nax) x1 = nax - 1;
		int z0 = rz - radius; if (z0 < 0) z0 = 0;
		int z1 = rz + radius; if (z1 >= n1) z1 = n1 - 1;
		for (ix = x0; ix <= x1; ix++) {
			for (iz = z0; iz <= z1; iz++) {
				float dx = (float)(ix - rx);
				float dz = (float)(iz - rz);
				float r2 = dx * dx + dz * dz;
				if (r2 < r2max) {
					float w = r2 / r2max;
					if (w < mask[ix * n1 + iz])
						mask[ix * n1 + iz] = w;
				}
			}
		}
	}

	/* Apply mask to gradient */
	for (k = 0; k < (int)sizem; k++)
		grad[k] *= mask[k];

	free(mask);
}


/*--------------------------------------------------------------------
 * compute_fwi_gradient -- Compute total misfit and gradient.
 *
 * Distributes shots across MPI ranks, runs forward + residual +
 * adjoint for each shot, and reduces gradients via MPI_Allreduce.
 *
 * Returns total L2 misfit. Gradient arrays (padded grid) are filled.
 *--------------------------------------------------------------------*/
static float compute_fwi_gradient(
	modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
	recPar *rec, snaPar *sna, shotPar *shot,
	float **src_nwav,
	const char *file_obs, const char *comp_str,
	int res_taper, int param,
	float *grad1, float *grad2, float *grad3,
	float *hess_lam, float *hess_muu, float *hess_rho,
	float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
	const char *chk_base, int chk_skipdt,
	int mpi_rank, int mpi_size,
	int keep_checkpoints,
	const float *comp_weights,
	int verbose)
{
	int ishot, k, i;
	int n1 = mod->naz;
	size_t sizem = (size_t)mod->nax * mod->naz;
	float misfit, total_misfit = 0.0f, global_misfit;
	char work_dir[256], chk_path[512];
	checkpointPar chk;
	adjSrcPar adj;

	/* Zero gradient and hessian arrays */
	memset(grad1, 0, sizem * sizeof(float));
	memset(grad3, 0, sizem * sizeof(float));
	if (grad2) memset(grad2, 0, sizem * sizeof(float));
	if (hess_lam) memset(hess_lam, 0, sizem * sizeof(float));
	if (hess_muu) memset(hess_muu, 0, sizem * sizeof(float));
	if (hess_rho) memset(hess_rho, 0, sizem * sizeof(float));
	if (hess_lam_muu) memset(hess_lam_muu, 0, sizem * sizeof(float));
	if (hess_lam_rho) memset(hess_lam_rho, 0, sizem * sizeof(float));
	if (hess_muu_rho) memset(hess_muu_rho, 0, sizem * sizeof(float));

	/* Determine shots for this rank */
	int my_nshots = 0;
	for (ishot = mpi_rank; ishot < shot->n; ishot += mpi_size)
		my_nshots++;

	int *my_shots = (int *)malloc(my_nshots * sizeof(int));
	k = 0;
	for (ishot = mpi_rank; ishot < shot->n; ishot += mpi_size)
		my_shots[k++] = ishot;

	/* Working directory for checkpoints */
	snprintf(work_dir, sizeof(work_dir), "fwi_rank%03d", mpi_rank);
	mkdir(work_dir, 0755);

	/* Process assigned shots */
	for (k = 0; k < my_nshots; k++) {
		ishot = my_shots[k];
		int izsrc = shot->z[ishot];
		int ixsrc = shot->x[ishot];
		int fileno = ishot;

		/* Per-shot gradient arrays (zeroed) */
		float *shot_grad1 = (float *)calloc(sizem, sizeof(float));
		float *shot_grad2 = NULL;
		float *shot_grad3 = (float *)calloc(sizem, sizeof(float));
		if (mod->ischeme > 2)
			shot_grad2 = (float *)calloc(sizem, sizeof(float));

		/* Step 1: Forward modeling with checkpointing */
		if (keep_checkpoints)
			snprintf(chk_path, sizeof(chk_path), "%s/%s_shot%03d",
			         work_dir, chk_base, ishot);
		else
			snprintf(chk_path, sizeof(chk_path), "%s/%s", work_dir, chk_base);
		initCheckpoints(&chk, mod, chk_skipdt, 0, chk_path);

		snaPar sna_off = *sna;
		sna_off.nsnap = 0;  /* No snapshots during inversion */

		fdfwimodc(mod, src, wav, bnd, rec, &sna_off,
		          ixsrc, izsrc, src_nwav, ishot, shot->n, fileno,
		          &chk, 0);

		/* Compute synthetic hydrophone for elastic */
		if (mod->ischeme > 2) {
			char syn_tzz[512], syn_txx[512], syn_p[512];
			snprintf(syn_tzz, sizeof(syn_tzz), "%s_%03d_rtzz.su", rec->file_rcv, fileno);
			snprintf(syn_txx, sizeof(syn_txx), "%s_%03d_rtxx.su", rec->file_rcv, fileno);
			snprintf(syn_p, sizeof(syn_p), "%s_%03d_rp.su", rec->file_rcv, fileno);
			computeHydrophone(syn_tzz, syn_txx, syn_p, 0);
		}

		/* Step 2: Compute residuals */
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
				         rec->file_rcv, fileno, comp_suffixes[i]);
				obs_arr[i] = obs_names[i];
				syn_arr[i] = syn_names[i];
			}

			snprintf(res_file, sizeof(res_file), "%s/residual.su", work_dir);
			misfit = computeResidual(ncomp, obs_arr, syn_arr, res_file,
			                         MISFIT_L2, comp_weights, 0);
			total_misfit += misfit;

			/* Step 3: Read residuals and apply taper */
			memset(&adj, 0, sizeof(adjSrcPar));
			readResidual(res_file, &adj, mod, bnd);
			applyCosineTaper(&adj, res_taper);

			/* Step 4: Adjoint backpropagation.
			 * Always accumulate in Lamé space (param=1) for multi-shot.
			 * The chain rule conversion to velocity (if param=2) is done
			 * once at the end by extractGradientVector, avoiding double
			 * conversion. See NOTE in adj_shot.c lines 213-215. */
			adj_shot(mod, src, wav, bnd, rec, &adj,
			         ixsrc, izsrc, src_nwav, &chk, NULL,
			         shot_grad1, shot_grad2, shot_grad3,
			         hess_lam, hess_muu, hess_rho,
			         hess_lam_muu, hess_lam_rho, hess_muu_rho,
			         1, 0);

			freeResidual(&adj);
		}

		/* Accumulate into local gradient sum */
		for (i = 0; i < (int)sizem; i++) {
			grad1[i] += shot_grad1[i];
			grad3[i] += shot_grad3[i];
			if (grad2 && shot_grad2)
				grad2[i] += shot_grad2[i];
		}

		if (verbose > 0) {
			/* Per-shot gradient diagnostic */
			float gmax = 0.0f;
			for (i = 0; i < (int)sizem; i++)
				if (fabsf(shot_grad1[i]) > gmax) gmax = fabsf(shot_grad1[i]);
			fprintf(stderr, "  rank %d: shot %d  misfit=%.4e  max|g_shot|=%.4e\n",
			        mpi_rank, ishot, misfit, gmax);
		}

		free(shot_grad1);
		free(shot_grad3);
		if (shot_grad2) free(shot_grad2);
		if (!keep_checkpoints)
			cleanCheckpoints(&chk);
	}

	/* Reduce gradients and misfit across ranks */
#ifdef USE_MPI
	{
		/* Diagnostic: local gradient norm before reduction */
		if (verbose > 0) {
			float local_gnorm = 0.0f;
			for (i = 0; i < (int)sizem; i++)
				local_gnorm += grad1[i] * grad1[i];
			local_gnorm = sqrtf(local_gnorm);
			fprintf(stderr, "  rank %d: local ||g1||=%.4e  local misfit=%.4e\n",
			        mpi_rank, local_gnorm, total_misfit);
		}

		float *tmp1 = (float *)calloc(sizem, sizeof(float));
		float *tmp3 = (float *)calloc(sizem, sizeof(float));
		float *tmp2 = NULL;
		if (grad2) tmp2 = (float *)calloc(sizem, sizeof(float));

		MPI_Allreduce(grad1, tmp1, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(grad3, tmp3, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		if (grad2 && tmp2)
			MPI_Allreduce(grad2, tmp2, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		memcpy(grad1, tmp1, sizem * sizeof(float));
		memcpy(grad3, tmp3, sizem * sizeof(float));
		if (grad2 && tmp2)
			memcpy(grad2, tmp2, sizem * sizeof(float));

		free(tmp1); free(tmp3);
		if (tmp2) free(tmp2);

		/* Reduce pseudo-Hessian arrays across ranks */
		if (hess_lam) {
			float *tmp_h = (float *)calloc(sizem, sizeof(float));
			MPI_Allreduce(hess_lam, tmp_h, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
			memcpy(hess_lam, tmp_h, sizem * sizeof(float));
			if (hess_muu) {
				MPI_Allreduce(hess_muu, tmp_h, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
				memcpy(hess_muu, tmp_h, sizem * sizeof(float));
			}
			MPI_Allreduce(hess_rho, tmp_h, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
			memcpy(hess_rho, tmp_h, sizem * sizeof(float));
			if (hess_lam_muu) {
				MPI_Allreduce(hess_lam_muu, tmp_h, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
				memcpy(hess_lam_muu, tmp_h, sizem * sizeof(float));
			}
			if (hess_lam_rho) {
				MPI_Allreduce(hess_lam_rho, tmp_h, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
				memcpy(hess_lam_rho, tmp_h, sizem * sizeof(float));
			}
			if (hess_muu_rho) {
				MPI_Allreduce(hess_muu_rho, tmp_h, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
				memcpy(hess_muu_rho, tmp_h, sizem * sizeof(float));
			}
			free(tmp_h);
		}

		MPI_Allreduce(&total_misfit, &global_misfit, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		if (verbose > 0 && mpi_rank == 0) {
			float global_gnorm = 0.0f;
			for (i = 0; i < (int)sizem; i++)
				global_gnorm += grad1[i] * grad1[i];
			global_gnorm = sqrtf(global_gnorm);
			vmess("  MPI reduction: global ||g1||=%.4e  global misfit=%.4e",
			      global_gnorm, global_misfit);
		}
	}
#else
	global_misfit = total_misfit;
#endif

	free(my_shots);
	return global_misfit;
}


/*--------------------------------------------------------------------
 * clean_shot_checkpoints -- Remove per-shot checkpoint files.
 *
 * Called when TRN accepts a new step (OPT_NSTE) or at termination,
 * to free disk space used by per-shot checkpoints.
 *--------------------------------------------------------------------*/
static void clean_shot_checkpoints(
	modPar *mod, shotPar *shot,
	const char *chk_base,
	int mpi_rank, int mpi_size)
{
	int ishot;
	char work_dir[256], chk_path[512];
	checkpointPar chk;

	snprintf(work_dir, sizeof(work_dir), "fwi_rank%03d", mpi_rank);

	for (ishot = mpi_rank; ishot < shot->n; ishot += mpi_size) {
		snprintf(chk_path, sizeof(chk_path), "%s/%s_shot%03d",
		         work_dir, chk_base, ishot);
		initCheckpoints(&chk, mod, 1, 0, chk_path);
		cleanCheckpoints(&chk);
	}
}


/*--------------------------------------------------------------------
 * compute_hessian_vector_product -- Compute H_GN * d for TRN.
 *
 * For each shot assigned to this rank:
 *   1. Reconstruct checkpoint struct (files on disk from gradient step)
 *   2. Call hess_shot (born_shot + adj_shot)
 *   3. Accumulate per-shot Hd
 * Then MPI_Allreduce to get global Hd.
 *
 * Parameters:
 *   dpert      - perturbation vector (opt->d, flat, length nvec)
 *   Hd_flat    - OUTPUT: H*d result (flat, length nvec, on rank 0)
 *   grad_taper - gradient taper radius
 *
 * Other parameters same as compute_fwi_gradient.
 *--------------------------------------------------------------------*/
static void compute_hessian_vector_product(
	modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
	recPar *rec, shotPar *shot,
	float **src_nwav,
	const char *comp_str,
	int res_taper, int param,
	const char *chk_base, int chk_skipdt,
	int mpi_rank, int mpi_size,
	float *dpert, float *Hd_flat, int nvec,
	int grad_taper,
	const float *comp_weights,
	int verbose)
{
	int ishot, k, i;
	int n1 = mod->naz;
	size_t sizem = (size_t)mod->nax * mod->naz;
	char work_dir[256], chk_path[512], born_path[512];
	checkpointPar chk;

	/* Perturbed FD coefficient arrays */
	int elastic = (mod->ischeme > 2);
	float *delta_lam = NULL;
	float *delta_mul = NULL;
	if (elastic) {
		delta_lam = (float *)calloc(sizem, sizeof(float));
		delta_mul = (float *)calloc(sizem, sizeof(float));
	}
	float *delta_rox = (float *)calloc(sizem, sizeof(float));
	float *delta_roz = (float *)calloc(sizem, sizeof(float));
	float *delta_l2m = (float *)calloc(sizem, sizeof(float));

	/* Convert perturbation vector to FD coefficient perturbations */
	perturbFDcoefficients(mod, bnd, dpert, param,
	                      delta_rox, delta_roz, delta_l2m, delta_lam, delta_mul);

	/* Padded Hd gradient arrays */
	float *hd1 = (float *)calloc(sizem, sizeof(float));
	float *hd3 = (float *)calloc(sizem, sizeof(float));
	float *hd2 = NULL;
	if (elastic) hd2 = (float *)calloc(sizem, sizeof(float));

	/* Determine shots for this rank */
	int my_nshots = 0;
	for (ishot = mpi_rank; ishot < shot->n; ishot += mpi_size)
		my_nshots++;

	int *my_shots = (int *)malloc(my_nshots * sizeof(int));
	k = 0;
	for (ishot = mpi_rank; ishot < shot->n; ishot += mpi_size)
		my_shots[k++] = ishot;

	snprintf(work_dir, sizeof(work_dir), "fwi_rank%03d", mpi_rank);

	/* Process assigned shots */
	for (k = 0; k < my_nshots; k++) {
		ishot = my_shots[k];
		int izsrc = shot->z[ishot];
		int ixsrc = shot->x[ishot];
		int fileno = ishot;

		/* Per-shot Hd (zeroed) */
		float *shot_hd1 = (float *)calloc(sizem, sizeof(float));
		float *shot_hd2 = elastic ? (float *)calloc(sizem, sizeof(float)) : NULL;
		float *shot_hd3 = (float *)calloc(sizem, sizeof(float));

		/* Reconstruct checkpoint metadata (files persist on disk
		 * from the gradient computation with keep_checkpoints=1) */
		snprintf(chk_path, sizeof(chk_path), "%s/%s_shot%03d",
		         work_dir, chk_base, ishot);
		initCheckpoints(&chk, mod, chk_skipdt, 0, chk_path);

		/* Born output base name */
		snprintf(born_path, sizeof(born_path), "%s/born", work_dir);

		/* Compute per-shot H*d via born_shot + adj_shot */
		hess_shot(mod, src, wav, bnd, rec,
		          ixsrc, izsrc, src_nwav, &chk,
		          delta_rox, delta_roz, delta_l2m, delta_lam, delta_mul,
		          born_path, comp_str,
		          ishot, shot->n, fileno,
		          res_taper,
		          shot_hd1, shot_hd2, shot_hd3,
		          1,  /* always Lame internally */
		          comp_weights, verbose);

		/* Accumulate into local Hd sum */
		for (i = 0; i < (int)sizem; i++) {
			hd1[i] += shot_hd1[i];
			hd3[i] += shot_hd3[i];
			if (hd2 && shot_hd2)
				hd2[i] += shot_hd2[i];
		}

		if (verbose > 0) {
			float hmax = 0.0f;
			for (i = 0; i < (int)sizem; i++)
				if (fabsf(shot_hd1[i]) > hmax) hmax = fabsf(shot_hd1[i]);
			fprintf(stderr, "  rank %d: shot %d  max|Hd_shot|=%.4e\n",
			        mpi_rank, ishot, hmax);
		}

		free(shot_hd1);
		free(shot_hd3);
		if (shot_hd2) free(shot_hd2);
	}

	/* Reduce Hd across ranks */
#ifdef USE_MPI
	{
		float *tmp1 = (float *)calloc(sizem, sizeof(float));
		float *tmp3 = (float *)calloc(sizem, sizeof(float));
		float *tmp2 = NULL;
		if (hd2) tmp2 = (float *)calloc(sizem, sizeof(float));

		MPI_Allreduce(hd1, tmp1, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(hd3, tmp3, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		if (hd2 && tmp2)
			MPI_Allreduce(hd2, tmp2, (int)sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		memcpy(hd1, tmp1, sizem * sizeof(float));
		memcpy(hd3, tmp3, sizem * sizeof(float));
		if (hd2 && tmp2)
			memcpy(hd2, tmp2, sizem * sizeof(float));

		free(tmp1); free(tmp3);
		if (tmp2) free(tmp2);
	}
#endif

	/* Apply gradient taper (same as gradient) */
	if (grad_taper > 0) {
		int ibx = mod->ioPx, ibz = mod->ioPz;
		if (bnd->lef == 4 || bnd->lef == 2) ibx += bnd->ntap;
		if (bnd->top == 4 || bnd->top == 2) ibz += bnd->ntap;
		taperGradientSrcRcv(hd1, n1, mod->nax, shot, rec, ibx, ibz, grad_taper);
		if (elastic)
			taperGradientSrcRcv(hd2, n1, mod->nax, shot, rec, ibx, ibz, grad_taper);
		taperGradientSrcRcv(hd3, n1, mod->nax, shot, rec, ibx, ibz, grad_taper);
	}

	/* Extract to flat vector (raw, no scaling — matching SEISCOPE
	 * convention: optimizer receives unscaled Hd).
	 * Use param=1 (Lame) since hess_shot accumulated in Lame space. */
	if (mpi_rank == 0) {
		extractGradientVector(Hd_flat, hd1, hd2, hd3, mod, bnd, 1);

		if (verbose > 0) {
			float hnorm = 0.0f;
			for (i = 0; i < nvec; i++)
				hnorm += Hd_flat[i] * Hd_flat[i];
			vmess("  Hessian-vector product: ||Hd||=%.4e", sqrtf(hnorm));
		}
	}

	/* Cleanup */
	free(my_shots);
	free(delta_rox); free(delta_roz); free(delta_l2m);
	if (delta_lam) free(delta_lam);
	if (delta_mul) free(delta_mul);
	free(hd1); free(hd3);
	if (hd2) free(hd2);
}


/*--------------------------------------------------------------------
 * write_iteration_model -- Write model SU files for this iteration.
 *--------------------------------------------------------------------*/
static void write_iteration_model(int iter, float *x, modPar *mod,
                                  bndPar *bnd, int param)
{
	char fname[512];
	int nx = mod->nx, nz = mod->nz;
	int nmodel = nx * nz;
	int elastic = (mod->ischeme > 2);

	if (param == 1) {
		snprintf(fname, sizeof(fname), "model_iter%03d_lam.su", iter);
		writesufile(fname, x, nz, nx, mod->z0, mod->x0, mod->dz, mod->dx);
		if (elastic) {
			snprintf(fname, sizeof(fname), "model_iter%03d_muu.su", iter);
			writesufile(fname, &x[nmodel], nz, nx,
			            mod->z0, mod->x0, mod->dz, mod->dx);
		}
		snprintf(fname, sizeof(fname), "model_iter%03d_rho.su", iter);
		writesufile(fname, &x[(elastic ? 2 : 1) * nmodel], nz, nx,
		            mod->z0, mod->x0, mod->dz, mod->dx);
	} else {
		snprintf(fname, sizeof(fname), "model_iter%03d_vp.su", iter);
		writesufile(fname, x, nz, nx, mod->z0, mod->x0, mod->dz, mod->dx);
		if (elastic) {
			snprintf(fname, sizeof(fname), "model_iter%03d_vs.su", iter);
			writesufile(fname, &x[nmodel], nz, nx,
			            mod->z0, mod->x0, mod->dz, mod->dx);
		}
		snprintf(fname, sizeof(fname), "model_iter%03d_rho.su", iter);
		writesufile(fname, &x[(elastic ? 2 : 1) * nmodel], nz, nx,
		            mod->z0, mod->x0, mod->dz, mod->dx);
	}
}


/*--------------------------------------------------------------------
 * setup_bounds -- Configure optimizer bounds from command-line params.
 *--------------------------------------------------------------------*/
static void setup_bounds(optim_type *opt, modPar *mod, bndPar *bnd,
                         int param, int nvec)
{
	float vp_min, vp_max, vs_min, vs_max, rho_min, rho_max;
	int has_bounds = 0;
	int nmodel = mod->nx * mod->nz;
	int elastic = (mod->ischeme > 2);

	/* Check for bound parameters */
	has_bounds += getparfloat("vp_min", &vp_min);
	has_bounds += getparfloat("vp_max", &vp_max);
	has_bounds += getparfloat("vs_min", &vs_min);
	has_bounds += getparfloat("vs_max", &vs_max);
	has_bounds += getparfloat("rho_min", &rho_min);
	has_bounds += getparfloat("rho_max", &rho_max);

	if (has_bounds == 0) return;

	/* Default bounds if only some are specified */
	if (!getparfloat("vp_min", &vp_min)) vp_min = 100.0f;
	if (!getparfloat("vp_max", &vp_max)) vp_max = 10000.0f;
	if (!getparfloat("vs_min", &vs_min)) vs_min = 0.0f;
	if (!getparfloat("vs_max", &vs_max)) vs_max = 6000.0f;
	if (!getparfloat("rho_min", &rho_min)) rho_min = 500.0f;
	if (!getparfloat("rho_max", &rho_max)) rho_max = 5000.0f;

	opt->bound = 1;
	opt->threshold = 0.0f;
	opt->lb = (float *)malloc(nvec * sizeof(float));
	opt->ub = (float *)malloc(nvec * sizeof(float));

	int i;
	if (param == 2) {
		/* Velocity parameterization: bounds directly on Vp, Vs, rho */
		for (i = 0; i < nmodel; i++) {
			opt->lb[i] = vp_min;
			opt->ub[i] = vp_max;
		}
		if (elastic) {
			for (i = 0; i < nmodel; i++) {
				opt->lb[nmodel + i] = vs_min;
				opt->ub[nmodel + i] = vs_max;
			}
			for (i = 0; i < nmodel; i++) {
				opt->lb[2*nmodel + i] = rho_min;
				opt->ub[2*nmodel + i] = rho_max;
			}
		} else {
			for (i = 0; i < nmodel; i++) {
				opt->lb[nmodel + i] = rho_min;
				opt->ub[nmodel + i] = rho_max;
			}
		}
	} else {
		/* Lame parameterization: convert velocity bounds to Lame bounds */
		float lam_min = rho_min * (vp_min*vp_min - 2.0f*vs_max*vs_max);
		float lam_max = rho_max * (vp_max*vp_max - 2.0f*vs_min*vs_min);
		float mu_min  = rho_min * vs_min * vs_min;
		float mu_max  = rho_max * vs_max * vs_max;

		for (i = 0; i < nmodel; i++) {
			opt->lb[i] = elastic ? lam_min : rho_min * vp_min * vp_min;
			opt->ub[i] = elastic ? lam_max : rho_max * vp_max * vp_max;
		}
		if (elastic) {
			for (i = 0; i < nmodel; i++) {
				opt->lb[nmodel + i] = mu_min;
				opt->ub[nmodel + i] = mu_max;
			}
			for (i = 0; i < nmodel; i++) {
				opt->lb[2*nmodel + i] = rho_min;
				opt->ub[2*nmodel + i] = rho_max;
			}
		} else {
			for (i = 0; i < nmodel; i++) {
				opt->lb[nmodel + i] = rho_min;
				opt->ub[nmodel + i] = rho_max;
			}
		}
	}
}


/*====================================================================
 * main
 *====================================================================*/
int main(int argc, char **argv)
{
	/* MPI variables */
	int mpi_rank = 0, mpi_size = 1;
	double t_start;

	/* Model and simulation parameters */
	modPar  mod;
	recPar  rec;
	snaPar  sna;
	wavPar  wav;
	srcPar  src;
	bndPar  bnd;
	shotPar shot;
	float **src_nwav;
	float  sinkvel;
	size_t nsamp, sizew, sizem;
	int    n1, ix, iz, ir, ishot, i;
	int    ioPx, ioPz;
	int    verbose;

	/* FWI parameters (same as fwi_driver) */
	char  *file_obs, *file_grad, *chk_base, *comp_str;
	int    chk_skipdt, res_taper, grad_taper, param;

	/* Optimization parameters */
	int    niter, algorithm, lbfgs_mem, nls_max, write_iter, niter_cg;
	int    use_precond, scaling;
	float  conv, precond_eps;
	float  s1 = 1.0f, s2 = 1.0f, s3 = 1.0f;  /* Yang scaling (eq 48) */

#ifdef USE_MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	t_start = MPI_Wtime();
#else
	t_start = wallclock_time();
#endif

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;

	/* FWI parameters */
	if (!getparstring("file_obs", &file_obs))
		verr("file_obs= parameter is required");
	if (!getparint("chk_skipdt", &chk_skipdt)) chk_skipdt = 500;
	if (!getparstring("chk_base", &chk_base)) chk_base = "chk";
	if (!getparstring("file_grad", &file_grad)) file_grad = "gradient";
	if (!getparstring("comp", &comp_str)) comp_str = "_rvz";
	if (!getparint("res_taper", &res_taper)) res_taper = 100;
	int grad_taper_set = getparint("grad_taper", &grad_taper);
	if (!grad_taper_set) grad_taper = 5;
	if (!getparint("param", &param)) param = 1;
	if (param < 1 || param > 2)
		verr("param must be 1 (Lame) or 2 (velocity)");

	/* Optimization parameters */
	if (!getparint("niter", &niter)) niter = 20;
	if (!getparfloat("conv", &conv)) conv = 1e-6f;
	if (!getparint("algorithm", &algorithm)) algorithm = 1;
	if (!getparint("lbfgs_mem", &lbfgs_mem)) lbfgs_mem = 20;
	if (!getparint("nls_max", &nls_max)) nls_max = 20;
	if (!getparint("write_iter", &write_iter)) write_iter = 1;
	if (!getparint("niter_cg", &niter_cg)) niter_cg = 5;
	if (!getparint("precond", &use_precond)) use_precond = 0;
	if (!getparfloat("precond_eps", &precond_eps)) precond_eps = 1e-3f;
	if (!getparint("scaling", &scaling)) scaling = 0;
	if (use_precond) {
		getparfloat("precond_scale_1", &s1);
		getparfloat("precond_scale_2", &s2);
		getparfloat("precond_scale_3", &s3);
	}

	/* When illumination preconditioner is active AND algorithm uses it
	 * natively, disable gradient taper — the preconditioner handles
	 * the source singularity. Only preconditioned algorithms (2,3,6,7)
	 * actually apply P^{-1}, so keep taper for unpreconditioned ones.
	 * User can still force taper with explicit grad_taper= on command line. */
	if (use_precond && !grad_taper_set &&
	    (algorithm == 2 || algorithm == 3 || algorithm == 6 || algorithm == 7))
		grad_taper = 0;

	if (algorithm < 0 || algorithm > 7)
		verr("algorithm must be 0-7: SD, LBFGS, PLBFGS, PNLCG, TRN, ENRICHED, PTRN, PENRICHED");

	/* ============================================================ */
	/* Standard setup (same as fwi_driver.c)                        */
	/* ============================================================ */
	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);
	n1 = mod.naz;
	sizem = (size_t)mod.nax * mod.naz;

	allocStoreSourceOnSurface(src);
	readModel(&mod, &bnd);

	/* Allocate source wavelets */
	if (wav.nx > 1) {
		nsamp = 0;
		sizew = wav.nt * wav.nx;
		src_nwav = (float **)calloc(wav.nx, sizeof(float *));
		for (i = 0; i < wav.nx; i++)
			src_nwav[i] = (float *)calloc(wav.nt, sizeof(float));
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
	/* Print inversion summary                                      */
	/* ============================================================ */
	if (mpi_rank == 0) {
		const char *alg_names[] = {"Steepest Descent", "L-BFGS", "PLBFGS", "PNLCG", "TRN", "Enriched", "PTRN", "PEnriched"};
		vmess("*******************************************");
#ifdef USE_MPI
		vmess("***** MPI FWI INVERSION               *****");
#else
		vmess("***** FWI INVERSION (Serial)           *****");
#endif
		vmess("*******************************************");
		vmess("Total shots: %d", shot.n);
#ifdef USE_MPI
		vmess("MPI ranks: %d", mpi_size);
#endif
		vmess("Parameterization: %s", param == 1 ? "Lame" : "Velocity");
		vmess("Algorithm: %s", alg_names[algorithm]);
		vmess("Max iterations: %d", niter);
		vmess("Convergence tolerance: %.2e", conv);
		if (algorithm == 1 || algorithm == 2 || algorithm == 7)
			vmess("L-BFGS memory: %d", lbfgs_mem);
		if (algorithm == 4 || algorithm == 5 || algorithm == 6 || algorithm == 7)
			vmess("Max CG iterations: %d", niter_cg);
		if (algorithm == 5 || algorithm == 7) {
			int enr_l_val;
			if (!getparint("enr_l", &enr_l_val)) enr_l_val = 20;
			vmess("Enriched L-BFGS cycle: %d", enr_l_val);
		}
		if (use_precond) {
			int is_precond_alg = (algorithm == 2 || algorithm == 3 ||
			                      algorithm == 6 || algorithm == 7);
			vmess("Preconditioner: Yang pseudo-Hessian (eps=%.2e, s=[%.2f,%.2f,%.2f])%s",
			      precond_eps, s1, s2, s3,
			      is_precond_alg ? "" : " (not used: algorithm has no native mechanism)");
			vmess("Gradient taper: %s (grad_taper=%d)",
			      grad_taper > 0 ? "enabled" : "disabled (preconditioner active)",
			      grad_taper);
		} else {
			vmess("Gradient taper: %d grid points", grad_taper);
		}
		vmess("*******************************************");
	}

	/* ============================================================ */
	/* Allocate optimizer arrays                                     */
	/* ============================================================ */
	int elastic = (mod.ischeme > 2);
	int nparam = elastic ? 3 : 2;
	int nmodel = mod.nx * mod.nz;
	int nvec = nparam * nmodel;

	float *x = (float *)calloc(nvec, sizeof(float));
	float *grad_vec = (float *)calloc(nvec, sizeof(float));

	/* Padded gradient arrays */
	float *grad1 = (float *)calloc(sizem, sizeof(float));
	float *grad3 = (float *)calloc(sizem, sizeof(float));
	float *grad2 = NULL;
	if (elastic) grad2 = (float *)calloc(sizem, sizeof(float));

	/* Yang pseudo-Hessian arrays (padded grid, accumulated in adj_shot) */
	float *hess_lam = NULL, *hess_muu = NULL, *hess_rho = NULL;
	float *hess_lam_muu = NULL, *hess_lam_rho = NULL, *hess_muu_rho = NULL;
	/* Block preconditioner matrix (6 upper-triangle, flat nmodel each) */
	float *P11 = NULL, *P12 = NULL, *P13 = NULL;
	float *P22 = NULL, *P23 = NULL, *P33 = NULL;
	float *grad_preco_vec = NULL;
	if (use_precond) {
		hess_lam = (float *)calloc(sizem, sizeof(float));
		hess_rho = (float *)calloc(sizem, sizeof(float));
		if (elastic) hess_muu = (float *)calloc(sizem, sizeof(float));
		hess_lam_muu = (float *)calloc(sizem, sizeof(float));
		hess_lam_rho = (float *)calloc(sizem, sizeof(float));
		hess_muu_rho = (float *)calloc(sizem, sizeof(float));
		P11 = (float *)calloc(nmodel, sizeof(float));
		P12 = (float *)calloc(nmodel, sizeof(float));
		P13 = (float *)calloc(nmodel, sizeof(float));
		P22 = (float *)calloc(nmodel, sizeof(float));
		P23 = (float *)calloc(nmodel, sizeof(float));
		P33 = (float *)calloc(nmodel, sizeof(float));
		if (algorithm == 2 || algorithm == 3 || algorithm == 6 || algorithm == 7)
			grad_preco_vec = (float *)calloc(nvec, sizeof(float));
	}

	/* Extract initial model into optimizer vector */
	extractModelVector(x, &mod, &bnd, param);

	/* Parameter scaling (0=none, 1=Brossier mean, 2=Yang max):
	 * m0[p] computed from initial model, x_tilde = x / m0.
	 * When preconditioner is active, s_p = m0_p automatically. */
	float m0[3] = {1.0f, 1.0f, 1.0f};
	float m_shift[3] = {0.0f, 0.0f, 0.0f};
	if (scaling > 0) {
		scaling_compute_m0(scaling, x, nmodel, nparam, m0, m_shift);
		scaling_normalize(x, nmodel, nparam, m0, m_shift);
		if (use_precond) {
			s1 = m0[0];
			if (nparam >= 2) s2 = m0[1];
			if (nparam >= 3) s3 = m0[2];
		}
		if (mpi_rank == 0) {
			const char *sname = (scaling == 1) ? "Brossier (mean)" : "Yang (max)";
			vmess("Scaling=%d (%s): m0 = [%.4e, %.4e, %.4e]",
			      scaling, sname, m0[0],
			      nparam >= 2 ? m0[1] : 0.0f,
			      nparam >= 3 ? m0[2] : 0.0f);
		}
	}

	/* ============================================================ */
	/* Configure optimizer                                          */
	/* ============================================================ */
	/* Verbose linesearch diagnostics (verbose >= 2) */
	int diag_new_iter = 1;
	int diag_ls_trial = 0;
	float diag_gnorm_raw = 0.0f;
	float diag_gnorm_raw_start = 0.0f;

	optim_type opt;
	memset(&opt, 0, sizeof(optim_type));
	opt.niter_max = niter;
	opt.conv = conv;
	opt.l = lbfgs_mem;
	opt.nls_max = nls_max;
	opt.print_flag = (mpi_rank == 0) ? 1 : 0;
	opt.debug = (verbose > 1) ? 1 : 0;

	/* TRN/Enriched-specific: max inner CG iterations */
	opt.niter_max_CG = niter_cg;

	/* Enriched-specific: L-BFGS cycle length and max CG */
	if (algorithm == 5 || algorithm == 7) {
		int enr_l_val;
		if (!getparint("enr_l", &enr_l_val)) enr_l_val = 20;
		opt.enr_l = enr_l_val;
		opt.enr_maxcg = niter_cg;
	}

	/* Setup bounds if specified */
	if (mpi_rank == 0) {
		setup_bounds(&opt, &mod, &bnd, param, nvec);
		if (scaling > 0 && opt.bound) {
			scaling_normalize(opt.lb, nmodel, nparam, m0, m_shift);
			scaling_normalize(opt.ub, nmodel, nparam, m0, m_shift);
		}
	}

	/* For TRN/Enriched: allocate d and Hd on ALL MPI ranks.
	 * The optimizer init (rank 0 only) will re-allocate these,
	 * but non-root ranks need valid buffers for MPI_Bcast. */
	if (algorithm == 4 || algorithm == 5 || algorithm == 6 || algorithm == 7) {
		opt.d  = (float *)calloc(nvec, sizeof(float));
		opt.Hd = (float *)calloc(nvec, sizeof(float));
	}

	/* ============================================================ */
	/* Initial gradient computation                                  */
	/* ============================================================ */
	if (mpi_rank == 0)
		vmess("Computing initial gradient...");

	int keep_chk = (algorithm == 4 || algorithm == 5 || algorithm == 6 || algorithm == 7) ? 1 : 0;

	/* ============================================================ */
	/* Compute Brossier W_d data weights from shot 0 observed data  */
	/* w_c = 1/rms(d_obs_c) per component, so each component        */
	/* contributes ~N/2 to the misfit regardless of amplitude.       */
	/* data_weight=1 enables auto weighting (default=1 when scaling>0) */
	/* ============================================================ */
	int data_weight;
	if (!getparint("data_weight", &data_weight))
		data_weight = (scaling > 0) ? 1 : 0;

	float *comp_weights = NULL;
	int ncomp_w = 0;
	{
		char comp_buf_w[1024];
		char *comp_suffixes_w[MAX_COMP];
		char *token_w;
		strncpy(comp_buf_w, comp_str, sizeof(comp_buf_w) - 1);
		comp_buf_w[sizeof(comp_buf_w) - 1] = '\0';
		token_w = strtok(comp_buf_w, ",");
		while (token_w && ncomp_w < MAX_COMP) {
			comp_suffixes_w[ncomp_w++] = token_w;
			token_w = strtok(NULL, ",");
		}

		if (data_weight > 0) {
			comp_weights = (float *)malloc(ncomp_w * sizeof(float));
			for (i = 0; i < ncomp_w; i++) {
				char obs_w[512];
				snprintf(obs_w, sizeof(obs_w), "%s_%03d%s.su",
				         file_obs, 0, comp_suffixes_w[i]);
				float rms = computeDataRMS(obs_w);
				comp_weights[i] = (rms > 1.0e-30f) ? (1.0f / rms) : 1.0f;
				if (mpi_rank == 0)
					vmess("Data weight: comp %s  rms=%.4e  w=%.4e",
					      comp_suffixes_w[i], rms, comp_weights[i]);
			}
		}
	}

	float fcost = compute_fwi_gradient(
		&mod, &src, &wav, &bnd, &rec, &sna, &shot, src_nwav,
		file_obs, comp_str, res_taper, param,
		grad1, grad2, grad3,
		hess_lam, hess_muu, hess_rho,
		hess_lam_muu, hess_lam_rho, hess_muu_rho,
		chk_base, chk_skipdt, mpi_rank, mpi_size, keep_chk,
		comp_weights, verbose);

	/* Taper gradient near source and receiver positions */
	{
		int ibx = mod.ioPx, ibz = mod.ioPz;
		if (bnd.lef == 4 || bnd.lef == 2) ibx += bnd.ntap;
		if (bnd.top == 4 || bnd.top == 2) ibz += bnd.ntap;
		taperGradientSrcRcv(grad1, n1, mod.nax, &shot, &rec, ibx, ibz, grad_taper);
		if (mod.ischeme > 2)
			taperGradientSrcRcv(grad2, n1, mod.nax, &shot, &rec, ibx, ibz, grad_taper);
		taperGradientSrcRcv(grad3, n1, mod.nax, &shot, &rec, ibx, ibz, grad_taper);
	}

	/* Extract gradient to flat vector, apply preconditioner.
	 *
	 * For PLBFGS/PNLCG: preconditioner is applied through the
	 * optimizer's native mechanism (OPT_PREC / grad_preco), so
	 * grad_vec stays raw and grad_preco_vec gets g/H.
	 *
	 * For SD/LBFGS/TRN/Enriched: grad_vec is pre-divided by H. */

	if (mpi_rank == 0) {
		extractGradientVector(grad_vec, grad1, grad2, grad3,
		                      &mod, &bnd, param);

		/* Capture raw gradient norm before scaling */
		if (verbose >= 2) {
			double g2 = 0.0;
			for (i = 0; i < nvec; i++) g2 += (double)grad_vec[i] * (double)grad_vec[i];
			diag_gnorm_raw_start = (float)sqrt(g2);
		}

		if (scaling > 0)
			scaling_scale_gradient(grad_vec, nmodel, nparam, m0);

		/* Build Yang block preconditioner and apply to gradient */
		if (use_precond && hess_lam) {
			buildBlockPrecond(hess_lam, hess_muu, hess_rho,
			    hess_lam_muu, hess_lam_rho, hess_muu_rho,
			    &mod, &bnd, param, elastic,
			    precond_eps, s1, s2, s3,
			    P11, P12, P13, P22, P23, P33,
			    nmodel, mpi_rank);
			if (grad_preco_vec &&
			    (algorithm == 2 || algorithm == 3 || algorithm == 6 || algorithm == 7))
				applyBlockPrecond(grad_preco_vec, grad_vec,
				    P11, P12, P13, P22, P23, P33, nmodel, nparam);
		}
		vmess("Initial misfit: %.6e", fcost);

		float gnorm = 0.0f;
		for (i = 0; i < nvec; i++)
			gnorm += grad_vec[i] * grad_vec[i];
		gnorm = sqrtf(gnorm);

		float gnorm_preco = 0.0f;
		if (grad_preco_vec) {
			double gp2 = 0.0;
			for (i = 0; i < nvec; i++)
				gp2 += (double)grad_preco_vec[i] * (double)grad_preco_vec[i];
			gnorm_preco = (float)sqrt(gp2);
			vmess("Preconditioned gradient: ||P^{-1}g||=%.4e (max=%.4e)",
			      gnorm_preco, (float)sqrt(gp2/(nvec>0?nvec:1))*sqrt(nvec));
			/* print a few sample values for diagnosis */
			float maxabs = 0.0f;
			for (i = 0; i < nvec; i++)
				if (fabsf(grad_preco_vec[i]) > maxabs) maxabs = fabsf(grad_preco_vec[i]);
			vmess("  P^{-1}g: max|element|=%.4e, ||.||_2=%.4e", maxabs, (float)sqrt(gp2));
		}

		if (gnorm > 0.0f) {
			/* Initial step length for first iteration.
			 * L-BFGS/SD/PLBFGS/PNLCG use descent=-g at iter 0 (no Hessian
			 * scaling), so alpha=1/||g|| gives a unit step in model space.
			 * TRN/Enriched use CG to get Hessian-scaled descent → alpha=1. */
			if (algorithm <= 3) {
				opt.alpha = 1.0f / gnorm;
			} else {
				opt.alpha = 1.0f;
			}
			vmess("Gradient: ||g||=%.4e, alpha=%.4e", gnorm, opt.alpha);
		}
	}

	/* ============================================================ */
	/* Optimization loop                                             */
	/* ============================================================ */
	optFlag flag = OPT_INIT;
	int opt_iter = 0;

	while (flag != OPT_CONV && flag != OPT_FAIL) {

		/* --- Optimizer step (rank 0 only) --- */
		if (mpi_rank == 0) {
			switch (algorithm) {
			case 0:
				steepest_descent_run(nvec, x, fcost, grad_vec,
				                     &opt, &flag);
				break;
			case 1:
				lbfgs_run(nvec, x, fcost, grad_vec, &opt, &flag);
				break;
			case 2:
				plbfgs_run(nvec, x, fcost, grad_vec,
				           grad_preco_vec ? grad_preco_vec : grad_vec,
				           &opt, &flag);
				while (flag == OPT_PREC) {
					/* Apply Yang block preconditioner to q_plb */
					if (P11)
						applyBlockPrecond(opt.q_plb, opt.q_plb,
						    P11, P12, P13, P22, P23, P33, nmodel, nparam);
					plbfgs_run(nvec, x, fcost, grad_vec,
					           grad_preco_vec ? grad_preco_vec : grad_vec,
					           &opt, &flag);
				}
				break;
			case 3:
				pnlcg_run(nvec, x, fcost, grad_vec,
				           grad_preco_vec ? grad_preco_vec : grad_vec,
				           &opt, &flag);
				break;
			case 4:
				trn_run(nvec, x, fcost, grad_vec, &opt, &flag);
				break;
			case 5:
				enriched_run(nvec, x, fcost, grad_vec, &opt, &flag);
				break;
			case 6:
				ptrn_run(nvec, x, fcost, grad_vec,
				         grad_preco_vec ? grad_preco_vec : grad_vec,
				         &opt, &flag);
				break;
			case 7:
				penriched_run(nvec, x, fcost, grad_vec,
				              grad_preco_vec ? grad_preco_vec : grad_vec,
				              &opt, &flag);
				break;
			}
		}

		/* --- Broadcast flag to all ranks --- */
#ifdef USE_MPI
		MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

		if (flag == OPT_GRAD) {
			/* --- Model updated by optimizer, need new gradient --- */

			/* Verbose: print iteration header at start of new linesearch */
			if (verbose >= 2 && mpi_rank == 0 && diag_new_iter) {
				float gnorm_scaled_start = optim_norm_l2(nvec, opt.grad);
				float dnorm = optim_norm_l2(nvec, opt.descent);
				vmess("");
				vmess("Iteration %d:", opt.cpt_iter + 1);
				vmess("  Cost:          %.4e  (%.2f%%)",
				      opt.fk, 100.0 * opt.fk / opt.f0);
				vmess("  ||g||:         %.4e", diag_gnorm_raw_start);
				vmess("  ||g~||:        %.4e", gnorm_scaled_start);
				vmess("  ||d||:         %.4e", dnorm);
				vmess("  <g,d>:        %.4e %s",
				      opt.q0, opt.q0 < 0 ? "(negative = descent OK)" : "(POSITIVE = NOT DESCENT!)");
				vmess("  alpha_0:       %.4e", opt.alpha);
				diag_new_iter = 0;
				diag_ls_trial = 0;
			}

			/* Denormalize x to physical units for forward modeling */
			if (scaling > 0)
				scaling_denormalize(x, nmodel, nparam, m0, m_shift);

			/* Broadcast new model vector from rank 0 */
#ifdef USE_MPI
			MPI_Bcast(x, nvec, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif

			/* All ranks: inject model into FD arrays */
			injectModelVector(x, &mod, &bnd, param);

			/* Re-normalize x back to tilde-space */
			if (scaling > 0)
				scaling_normalize(x, nmodel, nparam, m0, m_shift);

			/* All ranks: compute gradient */
			fcost = compute_fwi_gradient(
				&mod, &src, &wav, &bnd, &rec, &sna, &shot, src_nwav,
				file_obs, comp_str, res_taper, param,
				grad1, grad2, grad3,
				hess_lam, hess_muu, hess_rho,
				hess_lam_muu, hess_lam_rho, hess_muu_rho,
				chk_base, chk_skipdt, mpi_rank, mpi_size, keep_chk,
				comp_weights, verbose);

			/* Taper gradient near source and receiver positions */
			{
				int ibx = mod.ioPx, ibz = mod.ioPz;
				if (bnd.lef == 4 || bnd.lef == 2) ibx += bnd.ntap;
				if (bnd.top == 4 || bnd.top == 2) ibz += bnd.ntap;
				taperGradientSrcRcv(grad1, n1, mod.nax, &shot, &rec, ibx, ibz, grad_taper);
				if (mod.ischeme > 2)
					taperGradientSrcRcv(grad2, n1, mod.nax, &shot, &rec, ibx, ibz, grad_taper);
				taperGradientSrcRcv(grad3, n1, mod.nax, &shot, &rec, ibx, ibz, grad_taper);
			}

			/* Rank 0: extract gradient and apply preconditioner */
			if (mpi_rank == 0) {
				extractGradientVector(grad_vec, grad1, grad2, grad3,
				                      &mod, &bnd, param);

				/* Capture raw gradient norm before scaling */
				if (verbose >= 2) {
					double g2 = 0.0;
					for (i = 0; i < nvec; i++) g2 += (double)grad_vec[i] * (double)grad_vec[i];
					diag_gnorm_raw = (float)sqrt(g2);
				}

				if (scaling > 0)
					scaling_scale_gradient(grad_vec, nmodel, nparam, m0);

				/* Build Yang block preconditioner and apply to gradient */
				if (use_precond && hess_lam) {
					buildBlockPrecond(hess_lam, hess_muu, hess_rho,
					    hess_lam_muu, hess_lam_rho, hess_muu_rho,
					    &mod, &bnd, param, elastic,
					    precond_eps, s1, s2, s3,
					    P11, P12, P13, P22, P23, P33,
					    nmodel, mpi_rank);
					if (grad_preco_vec &&
					    (algorithm == 2 || algorithm == 3 || algorithm == 6 || algorithm == 7))
						applyBlockPrecond(grad_preco_vec, grad_vec,
						    P11, P12, P13, P22, P23, P33, nmodel, nparam);
				}

				/* NaN guard: if forward modeling went unstable,
				 * set huge cost so linesearch rejects this step.
				 * Note: use !(x==x) instead of isnan() for
				 * portability with -ffast-math. */
				int bad = !(fcost == fcost) || fcost > 1.0e20f;
				if (!bad) {
					for (i = 0; i < nvec && !bad; i++)
						bad = !(grad_vec[i] == grad_vec[i]);
				}
				if (bad) {
					vwarn("NaN/Inf detected — rejecting step");
					fcost = 1.0e30f;
					memset(grad_vec, 0, nvec * sizeof(float));
				}

				if (verbose > 0) {
					float gnorm = 0.0f;
					for (i = 0; i < nvec; i++)
						gnorm += grad_vec[i] * grad_vec[i];
					vmess("  gradient ||g||=%.4e", sqrtf(gnorm));
				}

				/* Verbose linesearch trial diagnostics */
				if (verbose >= 2) {
					diag_ls_trial++;
					float armijo_rhs = opt.fk + opt.m1 * opt.alpha * opt.q0;
					float q_curr = optim_dot(nvec, grad_vec, opt.descent);
					float wolfe2_rhs = opt.m2 * opt.q0;
					int wolfe1_pass = (fcost <= armijo_rhs);
					int wolfe2_pass = (q_curr >= wolfe2_rhs);

					vmess("");
					vmess("  Line search trial %d: alpha=%.4e",
					      diag_ls_trial, opt.alpha);
					vmess("    f(x+ad):     %.4e", fcost);
					vmess("    Wolfe 1: f=%.4e %s %.4e ? %s",
					      fcost, wolfe1_pass ? "<=" : ">",
					      armijo_rhs, wolfe1_pass ? "PASS" : "FAIL");
					vmess("    Wolfe 2: <g,d>=%.4e %s %.4e ? %s",
					      q_curr, wolfe2_pass ? ">=" : "<",
					      wolfe2_rhs, wolfe2_pass ? "PASS" : "FAIL");
					vmess("    Outcome: %s",
					      (wolfe1_pass && wolfe2_pass) ? "ACCEPTED" :
					      (!wolfe1_pass ? "ARMIJO FAIL -> bisect" :
					       "CURVATURE FAIL -> expand/bisect"));
				}
			}

		} else if (flag == OPT_HESS) {
			/* --- TRN inner CG needs Hessian-vector product --- */

			/* Denormalize d to physical units: d_phys = m0 * d_tilde */
			if (scaling > 0)
				scaling_denormalize(opt.d, nmodel, nparam, m0, m_shift);

			/* Broadcast CG direction from rank 0 to all ranks */
#ifdef USE_MPI
			MPI_Bcast(opt.d, nvec, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif

			/* Compute H*d = J^T(J*d) using Born + adjoint for all shots */
			compute_hessian_vector_product(
				&mod, &src, &wav, &bnd, &rec, &shot, src_nwav,
				comp_str, res_taper, param,
				chk_base, chk_skipdt, mpi_rank, mpi_size,
				opt.d, opt.Hd, nvec,
				grad_taper,
				comp_weights, verbose);

			/* Re-normalize d and scale Hd to tilde-space */
			if (scaling > 0) {
				scaling_normalize(opt.d, nmodel, nparam, m0, m_shift);
				scaling_scale_hessian_vec(opt.Hd, nmodel, nparam, m0);
			}

#ifdef USE_MPI
			/* Broadcast Hd back to all ranks (rank 0 has the result) */
			MPI_Bcast(opt.Hd, nvec, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif

		} else if (flag == OPT_PREC && algorithm == 6) {
			/* --- PTRN inner CG: apply block preconditioner to residual --- */
			if (mpi_rank == 0 && P11)
				applyBlockPrecond(opt.residual_preco, opt.residual,
				    P11, P12, P13, P22, P23, P33, nmodel, nparam);
			if (mpi_rank == 0) {
				ptrn_run(nvec, x, fcost, grad_vec,
				         grad_preco_vec ? grad_preco_vec : grad_vec,
				         &opt, &flag);
			}
#ifdef USE_MPI
			MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
			continue;

		} else if (flag == OPT_PREC && algorithm == 7) {
			/* --- PEnriched: apply block preconditioner to q_plb --- */
			if (mpi_rank == 0 && P11)
				applyBlockPrecond(opt.q_plb, opt.q_plb,
				    P11, P12, P13, P22, P23, P33, nmodel, nparam);
			if (mpi_rank == 0) {
				penriched_run(nvec, x, fcost, grad_vec,
				              grad_preco_vec ? grad_preco_vec : grad_vec,
				              &opt, &flag);
			}
#ifdef USE_MPI
			MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
			continue;

		} else if (flag == OPT_NSTE) {
			/* --- New step accepted: write intermediate model --- */
			opt_iter++;
			if (mpi_rank == 0) {
				vmess("Iteration %d: misfit = %.6e (%.2f%% of initial)",
				      opt_iter, fcost, 100.0f * fcost / opt.f0);
				if (write_iter > 0 && (opt_iter % write_iter == 0)) {
					if (scaling > 0)
						scaling_denormalize(x, nmodel, nparam, m0, m_shift);
					write_iteration_model(opt_iter, x, &mod, &bnd, param);
					if (scaling > 0)
						scaling_normalize(x, nmodel, nparam, m0, m_shift);
				}

				/* Verbose end-of-iteration diagnostics */
				if (verbose >= 2) {
					float dm_norm = 0.0f;
					for (i = 0; i < nvec; i++) {
						float dx = x[i] - opt.xk[i];
						dm_norm += dx * dx;
					}
					dm_norm = sqrtf(dm_norm);
					vmess("");
					vmess("  Final accepted alpha: %.4e", opt.alpha);
					vmess("  ||delta_m||:   %.4e", dm_norm);
					vmess("  Updated cost:  %.4e (%.2f%% of initial)",
					      fcost, 100.0f * fcost / opt.f0);
					vmess("  =========================================================");
					diag_gnorm_raw_start = diag_gnorm_raw;
					diag_new_iter = 1;
				}
			}

			/* TRN/Enriched: keep checkpoints alive — the next
			 * outer CG iteration needs them for H*v products.
			 * They'll be overwritten on the next OPT_GRAD, and
			 * cleaned up at termination. */
		}
	}

	/* Clean up any remaining TRN/Enriched checkpoints at termination */
	if (algorithm == 4 || algorithm == 5 || algorithm == 6 || algorithm == 7)
		clean_shot_checkpoints(&mod, &shot, chk_base, mpi_rank, mpi_size);

	/* ============================================================ */
	/* Write final results                                           */
	/* ============================================================ */
	if (mpi_rank == 0) {
		double t_total;
#ifdef USE_MPI
		t_total = MPI_Wtime() - t_start;
#else
		t_total = wallclock_time() - t_start;
#endif

		if (flag == OPT_CONV)
			vmess("Optimization CONVERGED in %d iterations", opt.cpt_iter);
		else
			vmess("Optimization FAILED (linesearch failure) at iteration %d",
			      opt.cpt_iter);

		vmess("Final misfit: %.6e (%.4f%% of initial)", fcost,
		      100.0f * fcost / opt.f0);
		vmess("Total forward problems: %d", opt.nfwd_pb);
		vmess("Total time: %.1f s", t_total);

		/* Write final model (denormalize to physical units) */
		if (scaling > 0)
			scaling_denormalize(x, nmodel, nparam, m0, m_shift);
		write_iteration_model(opt.cpt_iter, x, &mod, &bnd, param);

		/* Write final gradient */
		float *grad_interior = (float *)malloc(nmodel * sizeof(float));
		char fname[512];
		int ibndx, ibndz;
		ibndx = mod.ioPx;
		ibndz = mod.ioPz;
		if (bnd.lef == 4 || bnd.lef == 2) ibndx += bnd.ntap;
		if (bnd.top == 4 || bnd.top == 2) ibndz += bnd.ntap;

		for (ix = 0; ix < mod.nx; ix++)
			for (iz = 0; iz < mod.nz; iz++)
				grad_interior[ix*mod.nz+iz] = grad1[(ix+ibndx)*n1+iz+ibndz];

		if (param == 1)
			snprintf(fname, sizeof(fname), "%s_lam.su", file_grad);
		else
			snprintf(fname, sizeof(fname), "%s_vp.su", file_grad);
		writesufile(fname, grad_interior, mod.nz, mod.nx,
		            mod.z0, mod.x0, mod.dz, mod.dx);

		if (grad2) {
			for (ix = 0; ix < mod.nx; ix++)
				for (iz = 0; iz < mod.nz; iz++)
					grad_interior[ix*mod.nz+iz] = grad2[(ix+ibndx)*n1+iz+ibndz];
			if (param == 1)
				snprintf(fname, sizeof(fname), "%s_muu.su", file_grad);
			else
				snprintf(fname, sizeof(fname), "%s_vs.su", file_grad);
			writesufile(fname, grad_interior, mod.nz, mod.nx,
			            mod.z0, mod.x0, mod.dz, mod.dx);
		}

		for (ix = 0; ix < mod.nx; ix++)
			for (iz = 0; iz < mod.nz; iz++)
				grad_interior[ix*mod.nz+iz] = grad3[(ix+ibndx)*n1+iz+ibndz];
		snprintf(fname, sizeof(fname), "%s_rho.su", file_grad);
		writesufile(fname, grad_interior, mod.nz, mod.nx,
		            mod.z0, mod.x0, mod.dz, mod.dx);

		free(grad_interior);

		vmess("*******************************************");
		vmess("FWI inversion completed");
		vmess("*******************************************");
	}

	/* ============================================================ */
	/* Cleanup                                                       */
	/* ============================================================ */
	if (mpi_rank == 0) {
		optim_finalize(&opt);
		if (opt.lb) free(opt.lb);
		if (opt.ub) free(opt.ub);
	}

	free(x);
	free(grad_vec);
	if (grad_preco_vec) free(grad_preco_vec);
	free(grad1); free(grad3);
	if (grad2) free(grad2);
	if (hess_lam) free(hess_lam);
	if (hess_muu) free(hess_muu);
	if (hess_rho) free(hess_rho);
	if (hess_lam_muu) free(hess_lam_muu);
	if (hess_lam_rho) free(hess_lam_rho);
	if (hess_muu_rho) free(hess_muu_rho);
	if (P11) free(P11);
	if (P12) free(P12);
	if (P13) free(P13);
	if (P22) free(P22);
	if (P23) free(P23);
	if (P33) free(P33);

	for (i = 0; i < wav.nx; i++)
		free(src_nwav[i]);
	free(src_nwav);
	freeStoreSourceOnSurface();

#ifdef USE_MPI
	MPI_Finalize();
#endif
	return 0;
}
