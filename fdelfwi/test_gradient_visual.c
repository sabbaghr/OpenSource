/*
 * test_gradient_visual.c - Gradient visualization test for elastic FWI.
 *
 * Computes forward modeling + adjoint for multiple shots, accumulates
 * gradients, and saves raw and Brossier-scaled gradients as SU files
 * for visual inspection (e.g., in SeiSee).
 *
 * Works in both velocity (param=2) and Lamé (param=1) parameterizations.
 *
 * Output files (param=2):
 *   grad_vp_raw.su, grad_vs_raw.su, grad_rho_raw.su
 *   grad_vp_scaled.su, grad_vs_scaled.su, grad_rho_scaled.su
 *   precond_vp.su, precond_vs.su, precond_rho.su           (diagonal of P)
 *   precond_vp_vs.su, precond_vp_rho.su, precond_vs_rho.su (off-diagonal)
 *   grad_vp_preco.su, grad_vs_preco.su, grad_rho_preco.su  (P^{-1} g)
 *
 * Output files (param=1):
 *   grad_lam_raw.su, grad_mu_raw.su, grad_rho_raw.su
 *   grad_lam_scaled.su, grad_mu_scaled.su, grad_rho_scaled.su
 *   precond_lam.su, precond_mu.su, precond_rho.su             (diagonal of P)
 *   precond_lam_mu.su, precond_lam_rho.su, precond_mu_rho.su  (off-diagonal)
 *   grad_lam_preco.su, grad_mu_preco.su, grad_rho_preco.su    (P^{-1} g)
 *
 * Parameters (same as fdelmodc, plus):
 *   chk_skipdt=   checkpoint interval in time steps (100)
 *   chk_base=     checkpoint file base path (chk_gv)
 *   comp=_rvz     component suffix for recording
 *   param=2       parameterization (1=Lamé, 2=velocity)
 *   file_obs=     observed data base name (obs)
 *   out_base=     output gradient file base (grad)
 *   res_taper=    cosine taper length on residual traces (100)
 *   grad_taper=   Gaussian taper radius around src/rcv in grid pts (0=off)
 *   scaling=1     parameter scaling (0=none, 1=Brossier, 2=Yang)
 *   precond_eps=  epsilon damping for pseudo-Hessian preconditioner (1e-3)
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

int fdfwimodc(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec, snaPar *sna, int ixsrc, int izsrc,
              float **src_nwav, int ishot, int nshots, int fileno,
              checkpointPar *chk, int verbose);

int initCheckpoints(checkpointPar *chk, modPar *mod, int skipdt,
                    int delay, const char *file_base);
int cleanCheckpoints(checkpointPar *chk);

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

void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho,
                               size_t sizem);

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

float computeResidual(int ncomp, const char **obs_files, const char **syn_files,
                      const char *res_file, misfitType mtype,
                      const float *comp_weights, int verbose);
float computeDataRMS(const char *filename);

void scaling_compute_m0(int scaling, const float *x, int nmodel, int nparam,
                        float *m0, float *m_shift);
void scaling_scale_gradient(float *g, int nmodel, int nparam, const float *m0);

void extractModelVector(float *x, modPar *mod, bndPar *bnd, int param);
void extractGradientVector(float *g, float *grad1, float *grad2, float *grad3,
                           modPar *mod, bndPar *bnd, int param);


char *sdoc[] = {
" ",
" test_gradient_visual - Gradient visualization for elastic FWI",
" ",
" Multi-shot gradient accumulation with raw and Brossier-scaled SU output.",
" ",
" Same parameters as fdelmodc, plus:",
"   chk_skipdt=   checkpoint interval in time steps (100)",
"   chk_base=     checkpoint file base path (chk_gv)",
"   comp=_rvz     component suffix for recording",
"   param=2       parameterization (1=Lamé, 2=velocity)",
"   file_obs=     observed data base name (obs)",
"   out_base=     output gradient file base (grad)",
"   res_taper=    cosine taper on residual traces in samples (100)",
"   grad_taper=   Gaussian taper radius around src/rcv in grid pts (0=off)",
"   scaling=1     parameter scaling (0=none, 1=Brossier, 2=Yang)",
"   precond_eps=  epsilon damping for pseudo-Hessian preconditioner (1e-3)",
" ",
NULL};


/*====================================================================
 * Helper: stripBoundary -- Copy interior from padded array
 *====================================================================*/
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


/*====================================================================
 * Helper: print gradient statistics (norm, min, max, NaN check)
 *====================================================================*/
static void print_grad_stats(const char *name, float *data, int nmodel)
{
	int i, nan_count = 0;
	double sum2 = 0.0;
	float mn = data[0], mx = data[0];

	for (i = 0; i < nmodel; i++) {
		if (isnan(data[i]) || isinf(data[i])) {
			nan_count++;
			continue;
		}
		sum2 += (double)data[i] * (double)data[i];
		if (data[i] < mn) mn = data[i];
		if (data[i] > mx) mx = data[i];
	}

	vmess("  %s: ||g||=%.6e  min=%.6e  max=%.6e  NaN/Inf=%d",
		name, (float)sqrt(sum2), mn, mx, nan_count);
}


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


/*====================================================================
 * applyCosineTaper -- Cosine taper on residual traces (time domain).
 *
 * Smoothly ramps the first and last taper_len samples of each adjoint
 * source trace to zero, suppressing wrap-around / edge artifacts.
 * (Copied from fwi_inversion.c)
 *====================================================================*/
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


/*====================================================================
 * buildGradTaperMask -- Build parabolic taper mask on interior grid.
 *
 * Returns a [nx * nz] mask that is 0 at each source/receiver position
 * and smoothly ramps to 1 at a distance of `radius` grid points
 * (w = r²/R²). The caller must free the returned pointer.
 *
 * Source/receiver positions are in interior (non-padded) grid indices.
 *====================================================================*/
static float *buildGradTaperMask(int nx, int nz,
	shotPar *shot, recPar *rec, int radius)
{
	if (radius <= 0) return NULL;

	int ix, iz, k;
	int nmodel = nx * nz;
	float r2max = (float)radius * (float)radius;
	float *mask = (float *)malloc(nmodel * sizeof(float));
	for (k = 0; k < nmodel; k++) mask[k] = 1.0f;

	/* Parabolic hole around each source (interior coordinates) */
	for (k = 0; k < shot->n; k++) {
		int sx = shot->x[k];
		int sz = shot->z[k];
		int x0 = sx - radius; if (x0 < 0) x0 = 0;
		int x1 = sx + radius; if (x1 >= nx) x1 = nx - 1;
		int z0 = sz - radius; if (z0 < 0) z0 = 0;
		int z1 = sz + radius; if (z1 >= nz) z1 = nz - 1;
		for (ix = x0; ix <= x1; ix++) {
			for (iz = z0; iz <= z1; iz++) {
				float ddx = (float)(ix - sx);
				float ddz = (float)(iz - sz);
				float r2 = ddx * ddx + ddz * ddz;
				if (r2 < r2max) {
					float w = r2 / r2max;
					if (w < mask[ix * nz + iz])
						mask[ix * nz + iz] = w;
				}
			}
		}
	}

	/* Parabolic hole around each receiver (interior coordinates) */
	for (k = 0; k < rec->n; k++) {
		int rx = rec->x[k];
		int rz = rec->z[k];
		int x0 = rx - radius; if (x0 < 0) x0 = 0;
		int x1 = rx + radius; if (x1 >= nx) x1 = nx - 1;
		int z0 = rz - radius; if (z0 < 0) z0 = 0;
		int z1 = rz + radius; if (z1 >= nz) z1 = nz - 1;
		for (ix = x0; ix <= x1; ix++) {
			for (iz = z0; iz <= z1; iz++) {
				float ddx = (float)(ix - rx);
				float ddz = (float)(iz - rz);
				float r2 = ddx * ddx + ddz * ddz;
				if (r2 < r2max) {
					float w = r2 / r2max;
					if (w < mask[ix * nz + iz])
						mask[ix * nz + iz] = w;
				}
			}
		}
	}

	return mask;
}

/*====================================================================
 * applyMask -- Multiply a flat [nmodel] array by the mask.
 *====================================================================*/
static void applyMask(float *data, const float *mask, int nmodel)
{
	int i;
	for (i = 0; i < nmodel; i++)
		data[i] *= mask[i];
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
	double t0, t1;
	size_t nsamp, sizew, sizem;
	int    n1, ix, iz, ir, i;
	int    ioPx, ioPz;
	int    ixsrc, izsrc, fileno;
	int    verbose;

	/* Test parameters */
	char  *chk_base_str, *comp_str, *file_obs, *out_base;
	int    chk_skipdt, param, res_taper, grad_taper, scaling;

	/* Gradient arrays (padded grid) */
	float *grad1, *grad2, *grad3;
	/* Per-shot gradient arrays */
	float *g1_shot, *g2_shot, *g3_shot;
	/* Interior gradient arrays */
	float *g1_int, *g2_int, *g3_int;
	/* Scaled copies */
	float *g1_sc, *g2_sc, *g3_sc;
	/* Model vector for scaling */
	float *x_model;

	/* Pseudo-Hessian arrays (padded grid, accumulated across shots) */
	float *hess_lam, *hess_muu, *hess_rho;
	float *hess_lam_muu, *hess_lam_rho, *hess_muu_rho;
	/* Preconditioner arrays (flat nmodel, output of buildBlockPrecond) */
	float *P11, *P12, *P13, *P22, *P23, *P33;
	/* Preconditioned gradient (flat concatenated vector [nparam*nmodel]) */
	float *grad_vec, *grad_preco_vec;
	float  precond_eps;

	int    nmodel;
	int    nparam = 3;
	float  m0[3], m_shift[3];
	float  total_misfit;
	int    ishot;

	t0 = wallclock_time();

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;
	if (!getparint("chk_skipdt", &chk_skipdt)) chk_skipdt = 100;
	if (!getparstring("chk_base", &chk_base_str)) chk_base_str = "chk_gv";
	if (!getparstring("comp", &comp_str)) comp_str = "_rvz";
	if (!getparint("param", &param)) param = 2;
	if (!getparstring("file_obs", &file_obs)) file_obs = "obs";
	if (!getparstring("out_base", &out_base)) out_base = "grad";
	if (!getparint("res_taper", &res_taper)) res_taper = 100;
	if (!getparint("grad_taper", &grad_taper)) grad_taper = 0;
	if (!getparint("scaling", &scaling)) scaling = 1;
	if (!getparfloat("precond_eps", &precond_eps)) precond_eps = 1e-3f;

	/* Standard setup */
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

	sizem  = (size_t)mod.nax * mod.naz;
	nmodel = mod.nx * mod.nz;

	vmess("*******************************************");
	vmess("***** GRADIENT VISUALIZATION TEST     *****");
	vmess("*******************************************");
	vmess("Grid: nx=%d nz=%d (padded: nax=%d naz=%d)", mod.nx, mod.nz, mod.nax, mod.naz);
	vmess("Parameterization: %s (param=%d)", param == 1 ? "Lamé" : "Velocity", param);
	vmess("Number of shots: %d", shot.n);
	vmess("Component: %s", comp_str);
	vmess("Residual taper: %d samples, Gradient taper: %d grid pts",
		res_taper, grad_taper);

	/* ============================================================ */
	/* Allocate gradient arrays                                      */
	/* ============================================================ */
	grad1 = (float *)calloc(sizem, sizeof(float));
	grad2 = (float *)calloc(sizem, sizeof(float));
	grad3 = (float *)calloc(sizem, sizeof(float));
	g1_shot = (float *)calloc(sizem, sizeof(float));
	g2_shot = (float *)calloc(sizem, sizeof(float));
	g3_shot = (float *)calloc(sizem, sizeof(float));

	/* Pseudo-Hessian arrays (padded grid) */
	hess_lam     = (float *)calloc(sizem, sizeof(float));
	hess_muu     = (float *)calloc(sizem, sizeof(float));
	hess_rho     = (float *)calloc(sizem, sizeof(float));
	hess_lam_muu = (float *)calloc(sizem, sizeof(float));
	hess_lam_rho = (float *)calloc(sizem, sizeof(float));
	hess_muu_rho = (float *)calloc(sizem, sizeof(float));

	total_misfit = 0.0f;

	/* ============================================================ */
	/* Multi-shot loop: forward + residual + adjoint                 */
	/* ============================================================ */
	for (ishot = 0; ishot < shot.n; ishot++) {

		ixsrc  = shot.x[ishot];
		izsrc  = shot.z[ishot];
		fileno = ishot;

		vmess("--- Shot %d/%d: ix=%d iz=%d ---", ishot + 1, shot.n, ixsrc, izsrc);

		/* ----- Forward modeling ----- */
		initCheckpoints(&chk, &mod, chk_skipdt, 0, chk_base_str);

		fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna,
			ixsrc, izsrc, src_nwav, ishot, shot.n, fileno,
			&chk, verbose);

		/* ----- Compute hydrophone if elastic ----- */
		if (mod.ischeme > 2) {
			char syn_tzz[512], syn_txx[512], syn_p[512];
			snprintf(syn_tzz, sizeof(syn_tzz), "%s_%03d_rtzz.su", rec.file_rcv, fileno);
			snprintf(syn_txx, sizeof(syn_txx), "%s_%03d_rtxx.su", rec.file_rcv, fileno);
			snprintf(syn_p,   sizeof(syn_p),   "%s_%03d_rp.su",   rec.file_rcv, fileno);
			computeHydrophone(syn_tzz, syn_txx, syn_p, verbose);
		}

		/* ----- Compute residual ----- */
		{
			char obs_fname[1024], syn_fname[1024], res_fname[1024];
			const char *obs_files[1], *syn_files[1];
			float shot_misfit;

			snprintf(obs_fname, sizeof(obs_fname), "%s_%03d%s.su",
				file_obs, ishot, comp_str);
			snprintf(syn_fname, sizeof(syn_fname), "%s_%03d%s.su",
				rec.file_rcv, ishot, comp_str);
			snprintf(res_fname, sizeof(res_fname), "res_%03d.su", ishot);

			obs_files[0] = obs_fname;
			syn_files[0] = syn_fname;

			shot_misfit = computeResidual(1, obs_files, syn_files,
				res_fname, MISFIT_L2, NULL, verbose);

			total_misfit += shot_misfit;
			vmess("  Shot %d misfit: %.6e (cumulative: %.6e)",
				ishot + 1, shot_misfit, total_misfit);

			/* ----- Read residual and run adjoint ----- */
			memset(&adj, 0, sizeof(adjSrcPar));
			readResidual(res_fname, &adj, &mod, &bnd);
			applyCosineTaper(&adj, res_taper);

			/* Zero per-shot gradient */
			memset(g1_shot, 0, sizem * sizeof(float));
			memset(g2_shot, 0, sizem * sizeof(float));
			memset(g3_shot, 0, sizem * sizeof(float));

			{
				snaPar sna_adj;
				memset(&sna_adj, 0, sizeof(snaPar));

				/* Always accumulate gradient in Lamé (param=1) to avoid
				 * double chain-rule conversion; hessian is always in Lamé */
				adj_shot(&mod, &src, &wav, &bnd, &rec, &adj,
					ixsrc, izsrc, src_nwav, &chk, &sna_adj,
					g1_shot, g2_shot, g3_shot,
					hess_lam, hess_muu, hess_rho,
					hess_lam_muu, hess_lam_rho, hess_muu_rho,
					1, verbose);
			}

			freeResidual(&adj);
		}

		/* Accumulate into global gradient */
		for (i = 0; i < (int)sizem; i++) {
			grad1[i] += g1_shot[i];
			grad2[i] += g2_shot[i];
			grad3[i] += g3_shot[i];
		}

		cleanCheckpoints(&chk);
	}

	vmess("=== Total misfit: %.6e ===", total_misfit);

	/* ============================================================ */
	/* Extract gradient to flat vector (with chain rule if param=2)  */
	/* Gradient was accumulated in Lamé (param=1); extractGradient   */
	/* Vector applies convertGradientToVelocity when param=2.        */
	/* ============================================================ */
	grad_vec = (float *)calloc((size_t)nparam * nmodel, sizeof(float));
	extractGradientVector(grad_vec, grad1, grad2, grad3, &mod, &bnd, param);

	/* Split into component views for statistics and SU output */
	g1_int = grad_vec;
	g2_int = grad_vec + nmodel;
	g3_int = grad_vec + 2 * nmodel;

	/* ============================================================ */
	/* Parameter scaling                                             */
	/* ============================================================ */
	vmess("--- Parameter scaling (scaling=%d) ---", scaling);

	/* Extract model vector for scaling computation */
	x_model = (float *)malloc((size_t)nparam * nmodel * sizeof(float));
	extractModelVector(x_model, &mod, &bnd, param);

	/* Compute m0 scaling factors */
	scaling_compute_m0(scaling, x_model, nmodel, nparam, m0, m_shift);
	vmess("  m0[0] = %.6e  m0[1] = %.6e  m0[2] = %.6e", m0[0], m0[1], m0[2]);
	vmess("  m_shift[0] = %.6e  m_shift[1] = %.6e  m_shift[2] = %.6e",
		m_shift[0], m_shift[1], m_shift[2]);

	/* Build scaled gradient: copy raw then apply scaling */
	g1_sc = (float *)malloc(nmodel * sizeof(float));
	g2_sc = (float *)malloc(nmodel * sizeof(float));
	g3_sc = (float *)malloc(nmodel * sizeof(float));

	memcpy(g1_sc, g1_int, nmodel * sizeof(float));
	memcpy(g2_sc, g2_int, nmodel * sizeof(float));
	memcpy(g3_sc, g3_int, nmodel * sizeof(float));

	/* Apply scaling: g_tilde = g * m0 */
	for (i = 0; i < nmodel; i++) g1_sc[i] *= m0[0];
	for (i = 0; i < nmodel; i++) g2_sc[i] *= m0[1];
	for (i = 0; i < nmodel; i++) g3_sc[i] *= m0[2];

	/* ============================================================ */
	/* Build and apply Yang pseudo-Hessian preconditioner            */
	/* ============================================================ */
	vmess("--- Preconditioner (Yang pseudo-Hessian, eps=%.2e) ---", precond_eps);

	P11 = (float *)calloc(nmodel, sizeof(float));
	P12 = (float *)calloc(nmodel, sizeof(float));
	P13 = (float *)calloc(nmodel, sizeof(float));
	P22 = (float *)calloc(nmodel, sizeof(float));
	P23 = (float *)calloc(nmodel, sizeof(float));
	P33 = (float *)calloc(nmodel, sizeof(float));

	{
		/* Use m0 as Yang s_i scaling when parameter scaling is active,
		 * otherwise s_i = 1 (Yang eq 48, consistent with fwi_inversion.c) */
		float s1 = (scaling > 0) ? m0[0] : 1.0f;
		float s2 = (scaling > 0) ? m0[1] : 1.0f;
		float s3 = (scaling > 0) ? m0[2] : 1.0f;
		int elastic = (mod.ischeme > 2);

		vmess("  s1=%.6e  s2=%.6e  s3=%.6e", s1, s2, s3);

		buildBlockPrecond(hess_lam, hess_muu, hess_rho,
			hess_lam_muu, hess_lam_rho, hess_muu_rho,
			&mod, &bnd, param, elastic,
			precond_eps, s1, s2, s3,
			P11, P12, P13, P22, P23, P33,
			nmodel, 0);
	}

	/* Apply preconditioner: solve P * z = g_scaled at each grid point.
	 * Input: concatenated scaled gradient [g1_sc|g2_sc|g3_sc].
	 * Output: preconditioned gradient grad_preco_vec = P^{-1} g_scaled.
	 * Note: applyBlockPrecond expects/produces [nparam*nmodel] vectors. */
	{
		float *grad_scaled_vec = (float *)malloc((size_t)nparam * nmodel * sizeof(float));
		grad_preco_vec = (float *)malloc((size_t)nparam * nmodel * sizeof(float));

		/* Build concatenated scaled gradient vector */
		memcpy(grad_scaled_vec,              g1_sc, nmodel * sizeof(float));
		memcpy(grad_scaled_vec + nmodel,     g2_sc, nmodel * sizeof(float));
		memcpy(grad_scaled_vec + 2 * nmodel, g3_sc, nmodel * sizeof(float));

		applyBlockPrecond(grad_preco_vec, grad_scaled_vec,
			P11, P12, P13, P22, P23, P33,
			nmodel, nparam);

		free(grad_scaled_vec);
	}

	/* ============================================================ */
	/* Apply source/receiver taper as the LAST step                  */
	/* Applied after scaling and preconditioning to all gradients.   */
	/* ============================================================ */
	if (grad_taper > 0) {
		float *taper_mask;
		float *gp1 = grad_preco_vec;
		float *gp2 = grad_preco_vec + nmodel;
		float *gp3 = grad_preco_vec + 2 * nmodel;

		vmess("Applying gradient taper (radius=%d) around %d sources and %d receivers",
			grad_taper, shot.n, rec.n);
		taper_mask = buildGradTaperMask(mod.nx, mod.nz, &shot, &rec, grad_taper);

		/* Taper raw gradient */
		applyMask(g1_int, taper_mask, nmodel);
		applyMask(g2_int, taper_mask, nmodel);
		applyMask(g3_int, taper_mask, nmodel);
		/* Taper scaled gradient */
		applyMask(g1_sc, taper_mask, nmodel);
		applyMask(g2_sc, taper_mask, nmodel);
		applyMask(g3_sc, taper_mask, nmodel);
		/* Taper preconditioned gradient */
		applyMask(gp1, taper_mask, nmodel);
		applyMask(gp2, taper_mask, nmodel);
		applyMask(gp3, taper_mask, nmodel);

		free(taper_mask);
	}

	/* ============================================================ */
	/* Statistics and write all gradient outputs                     */
	/* ============================================================ */

	/* --- Raw gradient --- */
	vmess("--- Raw gradient statistics ---");
	if (param == 1) {
		print_grad_stats("g_lambda", g1_int, nmodel);
		print_grad_stats("g_mu", g2_int, nmodel);
		print_grad_stats("g_rho", g3_int, nmodel);
	} else {
		print_grad_stats("g_Vp", g1_int, nmodel);
		print_grad_stats("g_Vs", g2_int, nmodel);
		print_grad_stats("g_rho", g3_int, nmodel);
	}
	{
		double n1_sq = 0.0, n2_sq = 0.0, n3_sq = 0.0;
		for (i = 0; i < nmodel; i++) {
			n1_sq += (double)g1_int[i] * (double)g1_int[i];
			n2_sq += (double)g2_int[i] * (double)g2_int[i];
			n3_sq += (double)g3_int[i] * (double)g3_int[i];
		}
		vmess("--- Component ratios (raw) ---");
		vmess("  ||g1||/||g2|| = %.6e", sqrt(n1_sq) / (sqrt(n2_sq) > 0 ? sqrt(n2_sq) : 1.0));
		vmess("  ||g1||/||g3|| = %.6e", sqrt(n1_sq) / (sqrt(n3_sq) > 0 ? sqrt(n3_sq) : 1.0));
		vmess("  ||g2||/||g3|| = %.6e", sqrt(n2_sq) / (sqrt(n3_sq) > 0 ? sqrt(n3_sq) : 1.0));
		vmess("  ||g||_total = %.6e", sqrt(n1_sq + n2_sq + n3_sq));
		vmess("  ||g||^2 / f = %.6e",
			(n1_sq + n2_sq + n3_sq) / (total_misfit > 0 ? total_misfit : 1.0));
	}
	{
		char fname[1024];
		if (param == 1) {
			snprintf(fname, sizeof(fname), "%s_lam_raw.su", out_base);
			writesufile(fname, g1_int, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "%s_mu_raw.su", out_base);
			writesufile(fname, g2_int, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
		} else {
			snprintf(fname, sizeof(fname), "%s_vp_raw.su", out_base);
			writesufile(fname, g1_int, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "%s_vs_raw.su", out_base);
			writesufile(fname, g2_int, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
		}
		snprintf(fname, sizeof(fname), "%s_rho_raw.su", out_base);
		writesufile(fname, g3_int, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		vmess("Wrote raw gradient SU files");
	}

	/* --- Scaled gradient --- */
	vmess("--- Scaled gradient statistics ---");
	if (param == 1) {
		print_grad_stats("g_lambda_sc", g1_sc, nmodel);
		print_grad_stats("g_mu_sc", g2_sc, nmodel);
		print_grad_stats("g_rho_sc", g3_sc, nmodel);
	} else {
		print_grad_stats("g_Vp_sc", g1_sc, nmodel);
		print_grad_stats("g_Vs_sc", g2_sc, nmodel);
		print_grad_stats("g_rho_sc", g3_sc, nmodel);
	}
	{
		double n1_sq = 0.0, n2_sq = 0.0, n3_sq = 0.0;
		for (i = 0; i < nmodel; i++) {
			n1_sq += (double)g1_sc[i] * (double)g1_sc[i];
			n2_sq += (double)g2_sc[i] * (double)g2_sc[i];
			n3_sq += (double)g3_sc[i] * (double)g3_sc[i];
		}
		vmess("--- Component ratios (scaled) ---");
		vmess("  ||g1||/||g2|| = %.6e", sqrt(n1_sq) / (sqrt(n2_sq) > 0 ? sqrt(n2_sq) : 1.0));
		vmess("  ||g1||/||g3|| = %.6e", sqrt(n1_sq) / (sqrt(n3_sq) > 0 ? sqrt(n3_sq) : 1.0));
		vmess("  ||g2||/||g3|| = %.6e", sqrt(n2_sq) / (sqrt(n3_sq) > 0 ? sqrt(n3_sq) : 1.0));
		vmess("  ||g||_total_sc = %.6e", sqrt(n1_sq + n2_sq + n3_sq));
		vmess("  ||g_sc||^2 / f = %.6e",
			(n1_sq + n2_sq + n3_sq) / (total_misfit > 0 ? total_misfit : 1.0));
	}
	{
		char fname[1024];
		if (param == 1) {
			snprintf(fname, sizeof(fname), "%s_lam_scaled.su", out_base);
			writesufile(fname, g1_sc, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "%s_mu_scaled.su", out_base);
			writesufile(fname, g2_sc, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
		} else {
			snprintf(fname, sizeof(fname), "%s_vp_scaled.su", out_base);
			writesufile(fname, g1_sc, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "%s_vs_scaled.su", out_base);
			writesufile(fname, g2_sc, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
		}
		snprintf(fname, sizeof(fname), "%s_rho_scaled.su", out_base);
		writesufile(fname, g3_sc, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		vmess("Wrote scaled gradient SU files");
	}

	/* --- Preconditioner --- */
	{
		char fname[1024];
		if (param == 1) {
			snprintf(fname, sizeof(fname), "precond_lam.su");
			writesufile(fname, P11, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_mu.su");
			writesufile(fname, P22, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_rho.su");
			writesufile(fname, P33, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_lam_mu.su");
			writesufile(fname, P12, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_lam_rho.su");
			writesufile(fname, P13, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_mu_rho.su");
			writesufile(fname, P23, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
		} else {
			snprintf(fname, sizeof(fname), "precond_vp.su");
			writesufile(fname, P11, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_vs.su");
			writesufile(fname, P22, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_rho.su");
			writesufile(fname, P33, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_vp_vs.su");
			writesufile(fname, P12, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_vp_rho.su");
			writesufile(fname, P13, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "precond_vs_rho.su");
			writesufile(fname, P23, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
		}
		vmess("Wrote preconditioner SU files (6 components)");
	}

	/* --- Preconditioned gradient --- */
	{
		float *gp1 = grad_preco_vec;
		float *gp2 = grad_preco_vec + nmodel;
		float *gp3 = grad_preco_vec + 2 * nmodel;
		char fname[1024];

		vmess("--- Preconditioned gradient statistics ---");
		if (param == 1) {
			print_grad_stats("g_lambda_preco", gp1, nmodel);
			print_grad_stats("g_mu_preco", gp2, nmodel);
			print_grad_stats("g_rho_preco", gp3, nmodel);
		} else {
			print_grad_stats("g_Vp_preco", gp1, nmodel);
			print_grad_stats("g_Vs_preco", gp2, nmodel);
			print_grad_stats("g_rho_preco", gp3, nmodel);
		}
		{
			double n1_sq = 0.0, n2_sq = 0.0, n3_sq = 0.0;
			for (i = 0; i < nmodel; i++) {
				n1_sq += (double)gp1[i] * (double)gp1[i];
				n2_sq += (double)gp2[i] * (double)gp2[i];
				n3_sq += (double)gp3[i] * (double)gp3[i];
			}
			vmess("--- Component ratios (preconditioned) ---");
			vmess("  ||g1||/||g2|| = %.6e", sqrt(n1_sq) / (sqrt(n2_sq) > 0 ? sqrt(n2_sq) : 1.0));
			vmess("  ||g1||/||g3|| = %.6e", sqrt(n1_sq) / (sqrt(n3_sq) > 0 ? sqrt(n3_sq) : 1.0));
			vmess("  ||g2||/||g3|| = %.6e", sqrt(n2_sq) / (sqrt(n3_sq) > 0 ? sqrt(n3_sq) : 1.0));
			vmess("  ||g||_total_preco = %.6e", sqrt(n1_sq + n2_sq + n3_sq));
		}

		if (param == 1) {
			snprintf(fname, sizeof(fname), "%s_lam_preco.su", out_base);
			writesufile(fname, gp1, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "%s_mu_preco.su", out_base);
			writesufile(fname, gp2, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
		} else {
			snprintf(fname, sizeof(fname), "%s_vp_preco.su", out_base);
			writesufile(fname, gp1, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "%s_vs_preco.su", out_base);
			writesufile(fname, gp2, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
		}
		snprintf(fname, sizeof(fname), "%s_rho_preco.su", out_base);
		writesufile(fname, gp3, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		vmess("Wrote preconditioned gradient SU files");
	}

	/* ============================================================ */
	/* Final summary                                                 */
	/* ============================================================ */
	t1 = wallclock_time();

	printf("\n");
	printf("=== GRADIENT VISUALIZATION SUMMARY ===\n");
	printf("  Parameterization: %s (param=%d)\n", param == 1 ? "Lamé" : "Velocity", param);
	printf("  Shots: %d\n", shot.n);
	printf("  Total misfit: %.6e\n", total_misfit);
	printf("  Scaling m0: [%.6e, %.6e, %.6e]\n", m0[0], m0[1], m0[2]);
	printf("  Precond eps: %.2e\n", precond_eps);
	printf("  Wall time: %.2f s\n", t1 - t0);
	printf("  Output files: %s_*_raw.su, %s_*_scaled.su, %s_*_preco.su\n",
		out_base, out_base, out_base);
	printf("  Preconditioner: precond_*.su (6 components)\n");
	printf("=======================================\n");

	/* ============================================================ */
	/* Cleanup                                                       */
	/* ============================================================ */
	free(grad1);
	free(grad2);
	free(grad3);
	free(g1_shot);
	free(g2_shot);
	free(g3_shot);
	free(grad_vec);       /* g1_int, g2_int, g3_int are views into this */
	free(g1_sc);
	free(g2_sc);
	free(g3_sc);
	free(x_model);
	free(grad_preco_vec);
	free(hess_lam);
	free(hess_muu);
	free(hess_rho);
	free(hess_lam_muu);
	free(hess_lam_rho);
	free(hess_muu_rho);
	free(P11);
	free(P12);
	free(P13);
	free(P22);
	free(P23);
	free(P33);
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

	return 0;
}
