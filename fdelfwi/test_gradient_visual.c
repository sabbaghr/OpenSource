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

int readResidual(const char *filename, adjSrcPar *adj, modPar *mod, bndPar *bnd, recPar *rec);
void freeResidual(adjSrcPar *adj);
void writeSnapTimesReset(void);

int adj_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
             recPar *rec, adjSrcPar *adj,
             int ixsrc, int izsrc, float **src_nwav,
             checkpointPar *chk, snaPar *sna,
             float *grad1, float *grad2, float *grad3,
             float *hess_lam, float *hess_muu, float *hess_rho,
             float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
             int param, int verbose,
             float *wfld_energy);

void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho,
                               size_t sizem);

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

float computeResidual(int ncomp, const char **obs_files, const char **syn_files,
                      const char *res_file, misfitType mtype,
                      const float *comp_weights, const energyWeight *ew,
                      const khPar *kh, int verbose);
float computeDataRMS(const char *filename);

void scaling_compute_m0_from_bounds(const float *m_min, const float *m_max,
                                    int nparam, float *m0, float *m_shift);
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
"   shot_grad=0   1: also write the per-shot gradient before it is summed",
"                 into the total, as <out_base>_shotNNN_*_raw.su (the",
"                 panels of a shot-index common-image gather)",
"   file_snap_adj= base name for adjoint wavefield snapshots (none)",
"                 requires file_snap/tsnap1/tsnap2/dtsnap to be set;",
"                 reuses that same snapshot schedule",
"   res_taper=    cosine taper on residual traces in samples (100)",
"   grad_taper=   Gaussian taper radius around src/rcv in grid pts (0=off)",
"   scaling=1     parameter scaling (0=none, 1=Brossier, 2=Yang)",
"   data_weight=0 data weighting: 0=off, 2=energy norm W_r (rho, compliance)",
"   misfit=0      objective: 0=L2, 2=Kirchhoff-Helmholtz energy flux",
"   kh_normal=1   z component of the recording-surface normal (misfit=2 only);",
"                 needs comp=_rvx,_rvz,_rtxz,_rtzz and data_weight=0",
"   precond_eps=  epsilon damping for pseudo-Hessian preconditioner (1e-3)",
"   energy_ic=0   energy-norm scattering-angle filter on the gradient",
"                 (Rocha et al. 2016, elastic form -- see energy_imaging.c)",
"                 0=off, 1=tomographic (keep transmission),",
"                 2=migration (keep specular reflections),",
"                 3=write the diagnostics but leave the gradient alone.",
"                 Elastic only. Non-zero modes change the gradient and so",
"                 break the Taylor test, as grad_taper/smooth_grad do.",
"                 Writes <out_base>_shotNNN_ei_{kin,pot,tomo,migr}.su, plus",
"                 {kin_s,pot_s,tomo_p,migr_p,tomo_s,migr_s} at energy_gamma=-1.",
"   energy_gamma=2 which potential-energy pairing energy_ic uses:",
"                 2 = eps:C:eps~ strain energy (historical behaviour), or",
"                 0 = (lam+2mu) div.div~ + mu curl.curl~ (Navier form), or",
"                -1 = per-mode acoustic energy norm (mode-separated).",
"                 gamma=0 gives mode-INDEPENDENT 2sin^2/2cos^2 angle weights",
"                 (no (lam -+ 2mu cos2theta) modulation, no 59/121 deg sign",
"                 change) and makes V vanish identically for a P-S pair.",
"                 gamma=-1 applies Rocha's ACOUSTIC norm separately to",
"                 D=div v (a scalar wave at Vp) and C=curl v (at Vs), each",
"                 with its own velocity, giving an EXACT (1 +- cos psi) with",
"                 no modulation at all.  g_lam takes the P-channel weight",
"                 (sigma_kk is blind to S), g_mu and g_rho the energy mix.",
"                 Converted P-S pairs contribute to NEITHER channel.",
"   energy_alpha=0 0 = linear weight a = |k+p|/(|k+p|+|k-p|), a partition of",
"                 unity.  >0 selects Rocha's exponential mask (eq 45),",
"                 a_tomo = exp(-alpha*c2), a_migr = exp(-alpha*(1-c2)), which",
"                 is NOT a partition of unity.  Needed to REJECT rather than",
"                 half-pass converted modes, which sit at c2 = 1/2 when",
"                 energy_gamma=0.  Rocha uses alpha ~ 100.",
" ",
"   lag_nfreq=0   time-lag extended imaging condition (lag_gradient.c).",
"                 Number of frequencies used to form the lag correlation;",
"                 0 = off.  Evaluates the gradient kernels at non-zero",
"                 time lag, g(x,tau) = dt sum_t A(x,t) B(x,t+tau) with A the",
"                 forward strain and B the adjoint stress.  Accumulated in",
"                 lambda/mu and converted per lag to g_Vp, g_Vs by the usual",
"                 pointwise chain rule; there is no g_rho cube.",
"                 The adjoint source is the data residual,",
"                 so tau=0 is the ordinary gradient (band limited to the",
"                 frequencies below) and lag ASYMMETRY measures the time",
"                 shift between modelled and observed arrivals.  Elastic only.",
"                 Writes <out_base>_shotNNN_lag{vp,vs}.su as nz x (nx*nlag)",
"                 with the lag SLOWEST (trace = ilag*nx + ix), plus",
"                 <out_base>_shotNNN_lagaxis.txt describing the axes.",
"   lag_fmin=     lowest frequency in Hz, > 0.  Required when lag_nfreq>0.",
"   lag_fmax=     highest frequency in Hz (default lag_fmin, i.e. one line).",
"                 Frequencies are uniform on [lag_fmin, lag_fmax]; the lag",
"                 reconstruction is periodic with period 1/df in the RELATIVE",
"                 lag, which the symmetric convention makes 1/(2 df) on the",
"                 tau axis.  Cover the source band densely enough that",
"                 1/(2 df) exceeds the lag range.",
"   lag_nt=21     number of lags, forced odd so tau=0 is sampled exactly.",
"                 The lag is applied SYMMETRICALLY: the forward wavefield is",
"                 shifted by -tau and the adjoint by +tau, so tau is the",
"                 one-way shift and 2*tau the relative one.",
"   lag_dt=       lag increment in seconds (default 10*dt).  Need not be a",
"                 multiple of dt: the shift is a phase rotation, so",
"                 sub-sample lags are exact.  The run aborts if the implied",
"                 |tau| range reaches 1/(4 df), half the aliasing period on",
"                 the tau axis, since wrap-around is indistinguishable from",
"                 genuine lag moveout.",
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
	char  *chk_base_str, *comp_str, *file_obs, *out_base, *file_snap_adj;
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
	float *wfld_energy;
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
	int    shot_grad;
	float *shot_vec = NULL;   /* scratch for the per-shot gradient write */
	float *shot_c1 = NULL, *shot_c2 = NULL, *shot_c3 = NULL;

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
	/* Adjoint-wavefield snapshots: reuse the forward snapshot schedule
	 * (tsnap1/tsnap2/dtsnap/sna_type_*) but write to a different base name
	 * so the forward snapshots are not overwritten. NULL = disabled. */
	if (!getparstring("file_snap_adj", &file_snap_adj)) file_snap_adj = NULL;
	/* Per-shot gradients, written before they are summed into the total.
	 * These are the panels of a shot-index common-image gather. */
	if (!getparint("shot_grad", &shot_grad)) shot_grad = 0;
	if (!getparint("res_taper", &res_taper)) res_taper = 100;
	if (!getparint("grad_taper", &grad_taper)) grad_taper = 0;
	if (!getparint("scaling", &scaling)) scaling = 1;
	if (!getparfloat("precond_eps", &precond_eps)) precond_eps = 1e-2f;
	int precond_mode = 1;  /* 1-5, see preconditioner modes */
	if (!getparint("precond", &precond_mode)) precond_mode = 1;
	int smooth_grad = 0, smooth_hess = 0;
	if (!getparint("smooth_grad", &smooth_grad)) smooth_grad = 0;
	if (!getparint("smooth_hess", &smooth_hess)) smooth_hess = 0;
	float precond_blend = 0.0f;
	float precond_alpha = 0.30f;
	float srt_radius = 0.0f;
	if (!getparfloat("precond_blend", &precond_blend)) precond_blend = 0.0f;
	if (!getparfloat("precond_alpha", &precond_alpha)) precond_alpha = 0.30f;
	if (!getparfloat("srt_radius", &srt_radius)) srt_radius = 0.0f;
	int srt_filtsize = 1;
	if (!getparint("srt_filtsize", &srt_filtsize)) srt_filtsize = 1;

	/* Frequency band filtering (optional, for multiscale gradient testing) */
	float band_flo = 0.0f, band_fhi = 0.0f;
	getparfloat("freq_lo", &band_flo);
	getparfloat("freq_hi", &band_fhi);

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
	wfld_energy  = (float *)calloc(sizem, sizeof(float));

	total_misfit = 0.0f;

	/* Pre-filter observed data if frequency band specified */
	if (band_flo > 0.0f || band_fhi > 0.0f) {
		vmess("Filtering observed data to [%.1f, %.1f] Hz...", band_flo, band_fhi);
		bandpass_filter_obsdata(file_obs, comp_str, shot.n,
		                       band_flo, band_fhi, 0, 0);
	}

	/* Energy-norm data weighting W_r (data_weight=2).  Built once from the
	 * starting model so that Q is a constant metric. */
	energyWeight *ew = NULL;
	int data_weight;
	{
		if (!getparint("data_weight", &data_weight)) data_weight = 0;
		if (data_weight == 2)
			ew = energyWeightsBuild(&mod, &bnd, &rec, 1);
		else if (data_weight != 0)
			verr("test_gradient_visual: data_weight=%d not supported here "
			     "(0=off, 2=energy norm).", data_weight);
	}

	/* Objective function.  misfit=2 selects the Kirchhoff-Helmholtz
	 * energy flux, whose adjoint source is the tensorial (traction +
	 * deformation-rate) injection of Ravasi & Curtis -- see kh_misfit.c. */
	misfitType mtype;
	khPar *kh = NULL;
	{
		int misfit_int;
		float kh_normal;

		if (!getparint("misfit", &misfit_int)) misfit_int = 0;
		if (misfit_int != 0 && misfit_int != 2)
			verr("test_gradient_visual: misfit=%d not supported here "
			     "(0=L2, 2=Kirchhoff-Helmholtz energy flux).", misfit_int);
		mtype = (misfitType)misfit_int;

		if (mtype == MISFIT_KH) {
			if (data_weight != 0)
				verr("test_gradient_visual: misfit=2 requires data_weight=0; "
				     "the energy flux pairs a traction with a velocity, so "
				     "rescaling the velocity channels relative to the stress "
				     "channels destroys the flux interpretation.");
			/* Sign of n_z decides which side of the recording line the
			 * traction and velocity terms cancel on, and that is measured
			 * rather than assumed -- hence a parameter, not a constant. */
			if (!getparfloat("kh_normal", &kh_normal)) kh_normal = 1.0f;
			kh = khWeightsBuild(&rec, kh_normal, 1);
		}
	}

	/* Energy-norm scattering-angle filter (energy_imaging.c). */
	int energy_ic = EI_OFF;
	if (!getparint("energy_ic", &energy_ic)) energy_ic = EI_OFF;
	if (energy_ic < EI_OFF || energy_ic > EI_QC)
		verr("test_gradient_visual: energy_ic=%d is out of range "
		     "(0=off, 1=tomographic, 2=migration, 3=diagnostics only).",
		     energy_ic);
	if (energy_ic != EI_OFF && mod.ischeme <= 2)
		verr("test_gradient_visual: energy_ic=%d needs the elastic scheme "
		     "(ischeme=3); got ischeme=%d.", energy_ic, mod.ischeme);

	int energy_gamma = EI_GAMMA2;
	if (!getparint("energy_gamma", &energy_gamma)) energy_gamma = EI_GAMMA2;
	if (energy_gamma != EI_GAMMA0 && energy_gamma != EI_GAMMA2 &&
	    energy_gamma != EI_MODESEP)
		verr("test_gradient_visual: energy_gamma=%d is not implemented. "
		     "Use 2 (eps:C:eps~), 0 (Navier div/curl form) or -1 "
		     "(per-mode acoustic norm).", energy_gamma);

	float energy_alpha = 0.0f;
	if (!getparfloat("energy_alpha", &energy_alpha)) energy_alpha = 0.0f;
	if (energy_alpha < 0.0f)
		verr("test_gradient_visual: energy_alpha=%g must be >= 0.",
		     energy_alpha);

	/* Time-lag extended imaging condition (lag_gradient.c).
	 * lag_nfreq=0 disables it and leaves adj_shot untouched. */
	int lag_nfreq = 0, lag_nt = 21;
	float lag_fmin = 0.0f, lag_fmax = 0.0f, lag_dt = 0.0f;
	if (!getparint("lag_nfreq", &lag_nfreq)) lag_nfreq = 0;
	if (lag_nfreq < 0)
		verr("test_gradient_visual: lag_nfreq=%d must be >= 0.", lag_nfreq);
	if (lag_nfreq > 0) {
		if (mod.ischeme <= 2)
			verr("test_gradient_visual: lag_nfreq=%d needs the elastic scheme "
			     "(ischeme=3); got ischeme=%d.", lag_nfreq, mod.ischeme);
		if (!getparfloat("lag_fmin", &lag_fmin))
			verr("test_gradient_visual: lag_nfreq>0 requires lag_fmin= "
			     "(lowest frequency in Hz, > 0).");
		if (!getparfloat("lag_fmax", &lag_fmax)) lag_fmax = lag_fmin;
		if (!getparint("lag_nt", &lag_nt)) lag_nt = 21;
		if (!getparfloat("lag_dt", &lag_dt)) lag_dt = 10.0f * mod.dt;
		if (lag_dt <= 0.0f)
			verr("test_gradient_visual: lag_dt=%g must be > 0 s.", lag_dt);
	}

	/* grad_taper is applied far below, to g*_int/g*_sc, long after the lag
	 * cubes have been written from lag_gradient's own accumulators -- so the
	 * lag cubes need the mask handed to them explicitly.  Building it here
	 * (same function, same radius, same shot/rec lists) keeps the two in
	 * step, which is what keeps the tau=0 check meaningful: both sides of
	 * that comparison are then tapered identically. */
	float *lag_mask = NULL;
	if (lag_nfreq > 0 && grad_taper > 0) {
		lag_mask = buildGradTaperMask(mod.nx, mod.nz, &shot, &rec, grad_taper);
		if (verbose) vmess("Lag cubes share the gradient taper (radius=%d)",
			grad_taper);
	}

	/* ============================================================ */
	/* Multi-shot loop: forward + residual + adjoint                 */
	/* ============================================================ */
	/* Snapshot files are named <base><ext> with no shot index, and the
	 * truncate-vs-append flag is a single global that only adj_shot resets.
	 * Multi-shot therefore needs a per-shot base name, otherwise every
	 * forward shot APPENDS into one file and every adjoint shot TRUNCATES
	 * the previous one, leaving only the last.  Single-shot naming is left
	 * exactly as it was so existing scripts keep working. */
	char *snap_base_fwd = sna.file_snap;
	char  snap_fwd_name[1024];

	for (ishot = 0; ishot < shot.n; ishot++) {

		ixsrc  = shot.x[ishot];
		izsrc  = shot.z[ishot];
		fileno = ishot;

		vmess("--- Shot %d/%d: ix=%d iz=%d ---", ishot + 1, shot.n, ixsrc, izsrc);

		/* ----- Forward modeling ----- */
		initCheckpoints(&chk, &mod, chk_skipdt, 0, chk_base_str);

		if (sna.nsnap > 0 && snap_base_fwd != NULL) {
			if (shot.n > 1) {
				snprintf(snap_fwd_name, sizeof(snap_fwd_name), "%s_s%03d",
					snap_base_fwd, ishot);
				sna.file_snap = snap_fwd_name;
			}
			writeSnapTimesReset();
		}

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

		/* Filter synthetic data for this band (same sufilter as observed) */
		if (band_flo > 0.0f || band_fhi > 0.0f)
			bandpass_filter_syndata(rec.file_rcv, comp_str, fileno,
			                       band_flo, band_fhi);

		/* ----- Compute residual (multi-component aware) ----- */
		{
			#define TGV_MAX_COMP 8
			char comp_buf[1024];
			char *comp_suffixes[TGV_MAX_COMP];
			int ncomp = 0;
			char *token_c;
			float shot_misfit;
			char res_fname[1024];

			/* Parse comma-separated comp_str */
			strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
			comp_buf[sizeof(comp_buf) - 1] = '\0';
			token_c = strtok(comp_buf, ",");
			while (token_c && ncomp < TGV_MAX_COMP) {
				comp_suffixes[ncomp++] = token_c;
				token_c = strtok(NULL, ",");
			}

			char obs_names[TGV_MAX_COMP][512], syn_names[TGV_MAX_COMP][512];
			const char *obs_arr[TGV_MAX_COMP], *syn_arr[TGV_MAX_COMP];

			for (int ic = 0; ic < ncomp; ic++) {
				snprintf(obs_names[ic], sizeof(obs_names[ic]), "%s_%03d%s.su",
					file_obs, ishot, comp_suffixes[ic]);
				snprintf(syn_names[ic], sizeof(syn_names[ic]), "%s_%03d%s.su",
					rec.file_rcv, ishot, comp_suffixes[ic]);
				obs_arr[ic] = obs_names[ic];
				syn_arr[ic] = syn_names[ic];
			}

			snprintf(res_fname, sizeof(res_fname), "res_%03d.su", ishot);

			shot_misfit = computeResidual(ncomp, obs_arr, syn_arr,
				res_fname, mtype, NULL, ew, kh, verbose);

			total_misfit += shot_misfit;
			vmess("  Shot %d misfit: %.6e (cumulative: %.6e)",
				ishot + 1, shot_misfit, total_misfit);

			/* ----- Read residual and run adjoint ----- */
			memset(&adj, 0, sizeof(adjSrcPar));
			readResidual(res_fname, &adj, &mod, &bnd, &rec);
			applyCosineTaper(&adj, res_taper);
			applyApertureTaper(&adj, mod.dx, rec.aper_taper, verbose);

			/* Zero per-shot gradient */
			memset(g1_shot, 0, sizem * sizeof(float));
			memset(g2_shot, 0, sizem * sizeof(float));
			memset(g3_shot, 0, sizem * sizeof(float));

			/* Per-shot Hessian and wavefield energy arrays */
			float *sh_lam = NULL, *sh_muu = NULL, *sh_rho = NULL;
			float *sh_lm = NULL, *sh_lr = NULL, *sh_mr = NULL;
			float *sh_wfld = (float *)calloc(sizem, sizeof(float));
			if (srt_radius > 0.0f || precond_mode >= 4) {
				sh_lam = (float *)calloc(sizem, sizeof(float));
				sh_muu = (float *)calloc(sizem, sizeof(float));
				sh_rho = (float *)calloc(sizem, sizeof(float));
				sh_lm  = (float *)calloc(sizem, sizeof(float));
				sh_lr  = (float *)calloc(sizem, sizeof(float));
				sh_mr  = (float *)calloc(sizem, sizeof(float));
			}

			{
				snaPar sna_adj;
				char   snap_adj_name[1024];
				memset(&sna_adj, 0, sizeof(snaPar));

				/* Enable adjoint snapshots only when the user supplied a
				 * separate base name AND forward snapshots are configured
				 * (that is where the timing schedule comes from). */
				if (file_snap_adj != NULL && sna.nsnap > 0) {
					sna_adj = sna;
					if (shot.n > 1) {
						snprintf(snap_adj_name, sizeof(snap_adj_name),
							"%s_s%03d", file_snap_adj, ishot);
						sna_adj.file_snap = snap_adj_name;
					} else {
						sna_adj.file_snap = file_snap_adj;
					}
					if (verbose) vmess("  Adjoint snapshots -> %s_s*.su (%d snaps)",
						sna_adj.file_snap, sna_adj.nsnap);
				}

				/* Zeroes the two energy accumulators for this shot; a no-op
				 * at energy_ic=0, which leaves adj_shot bit-identical. */
				energyImagingInit(&mod, energy_ic, energy_gamma,
				                  energy_alpha, verbose);

				/* Zeroes this shot's time-lag DFT accumulators; a no-op at
				 * lag_nfreq=0, which leaves adj_shot untouched. */
				lagGradientInit(&mod, lag_nfreq, lag_fmin, lag_fmax, verbose);

				adj_shot(&mod, &src, &wav, &bnd, &rec, &adj,
					ixsrc, izsrc, src_nwav, &chk, &sna_adj,
					g1_shot, g2_shot, g3_shot,
					sh_lam ? sh_lam : hess_lam,
					sh_muu ? sh_muu : hess_muu,
					sh_rho ? sh_rho : hess_rho,
					sh_lm  ? sh_lm  : hess_lam_muu,
					sh_lr  ? sh_lr  : hess_lam_rho,
					sh_mr  ? sh_mr  : hess_muu_rho,
					1, verbose,
					sh_wfld);
			}

			/* Time-lag gradient: reconstruct g_vp(x,tau) and g_vs(x,tau)
			 * from this shot's DFT accumulators and write the cubes.  Reads
			 * only its own accumulators, so it is independent of anything
			 * done to g*_shot below. */
			if (lag_nfreq > 0)
				lagGradientWrite(&mod, &bnd, out_base, ishot,
					lag_nt, lag_dt, lag_mask, verbose);

			/* Energy-norm angle filter: build the weight from the two
			 * accumulators and apply it to this shot's gradient, which is
			 * still in Lame space (adj_shot was called with param=1).  The
			 * weight is one scalar per point and the Lame -> velocity chain
			 * rule is pointwise linear, so the order does not matter. */
			if (energy_ic != EI_OFF) {
				float *ei_buf = (float *)malloc(sizem * sizeof(float));
				float *ei_int = (float *)malloc((size_t)nmodel * sizeof(float));
				static const char *ei_name[EI_GET_N] = {
					"kin",    "pot",    "tomo",   "migr",   "kin_s",
					"pot_s",  "tomo_p", "migr_p", "tomo_s", "migr_s"};
				int iw;
				char fname[1024];

				if (!ei_buf || !ei_int)
					verr("test_gradient_visual: energy_ic diagnostic "
					     "allocation failed.");

				/* energyImagingGet returns 0 for the six mode-separated codes
				 * at the gamma-family settings, so this loop writes 4 files at
				 * gamma=0/2 and all 10 at gamma=-1. */
				for (iw = 0; iw < EI_GET_N; iw++) {
					if (!energyImagingGet(&mod, &bnd, iw, ei_buf)) continue;
					stripBoundary(ei_int, ei_buf, &mod, &bnd);
					snprintf(fname, sizeof(fname), "%s_shot%03d_ei_%s.su",
						out_base, ishot, ei_name[iw]);
					writesufile(fname, ei_int, (size_t)mod.nz, (size_t)mod.nx,
						mod.z0, mod.x0, mod.dz, mod.dx);
				}
				free(ei_buf);
				free(ei_int);

				energyImagingApply(&mod, &bnd, g1_shot, g2_shot, g3_shot,
					verbose);
				energyImagingFree();

				if (verbose)
					vmess("  Wrote %s_shot%03d_ei_{kin,pot,tomo,migr%s}.su",
						out_base, ishot,
						(energy_gamma == EI_MODESEP)
						  ? ",kin_s,pot_s,tomo_p,migr_p,tomo_s,migr_s" : "");
			}

			/* For precond=4,5: overwrite per-shot Hessian with P4 cross-correlation */
			if (precond_mode >= 4 && sh_lam) {
				memset(sh_lam, 0, sizem * sizeof(float));
				memset(sh_muu, 0, sizem * sizeof(float));
				memset(sh_rho, 0, sizem * sizeof(float));
				memset(sh_lm, 0, sizem * sizeof(float));
				memset(sh_lr, 0, sizem * sizeof(float));
				memset(sh_mr, 0, sizem * sizeof(float));

				precond_shot(&mod, &src, &wav, &bnd, &rec,
				             ixsrc, izsrc, src_nwav, &chk,
				             sh_lam, sh_muu, sh_rho,
				             sh_lm, sh_lr, sh_mr,
				             verbose,
				             rec.file_rcv, ishot, comp_str);
			}

			/* Per-shot radial source taper */
			if (srt_radius > 0.0f) {
				int ibx = mod.ioPx, ibz = mod.ioPz;
				if (bnd.lef == 4 || bnd.lef == 2) ibx += bnd.ntap;
				if (bnd.top == 4 || bnd.top == 2) ibz += bnd.ntap;
				float *src_mask = buildSourceTaperMask(
					mod.nax, mod.naz, ixsrc + ibx, izsrc + ibz,
					srt_radius, mod.dx, srt_filtsize);

				applySourceTaper(g1_shot, src_mask, sizem);
				applySourceTaper(g2_shot, src_mask, sizem);
				applySourceTaper(g3_shot, src_mask, sizem);
				applySourceTaper(sh_lam, src_mask, sizem);
				applySourceTaper(sh_muu, src_mask, sizem);
				applySourceTaper(sh_rho, src_mask, sizem);
				applySourceTaper(sh_lm, src_mask, sizem);
				applySourceTaper(sh_lr, src_mask, sizem);
				applySourceTaper(sh_mr, src_mask, sizem);
				applySourceTaper(sh_wfld, src_mask, sizem);

				free(src_mask);
			}

			/* Accumulate per-shot Hessian into total
			 * (always, regardless of source taper) */
			if (sh_lam) {
				for (i = 0; i < (int)sizem; i++) {
					hess_lam[i] += sh_lam[i];
					hess_muu[i] += sh_muu[i];
					hess_rho[i] += sh_rho[i];
					hess_lam_muu[i] += sh_lm[i];
					hess_lam_rho[i] += sh_lr[i];
					hess_muu_rho[i] += sh_mr[i];
				}
				free(sh_lam); free(sh_muu); free(sh_rho);
				free(sh_lm); free(sh_lr); free(sh_mr);
			}

			/* Accumulate wavefield energy (always, even without source taper) */
			for (i = 0; i < (int)sizem; i++)
				wfld_energy[i] += sh_wfld[i];
			free(sh_wfld);

			freeResidual(&adj);
		}

		/* Per-shot gradient, written BEFORE the sum so it is the raw
		 * contribution of this shot alone -- no smoothing, no scaling, no
		 * preconditioner (those are all applied to the total further down).
		 * g*_shot are in Lame because adj_shot is called with param=1 above;
		 * extractGradientVector applies the chain rule to velocity exactly
		 * once, the same way it does for the total.
		 *
		 * The chain rule (convertGradientToVelocity, fwi_gradient.c:574) runs
		 * IN PLACE on its three input arrays, so g*_shot must be copied first.
		 * Converting them directly would leave velocity-space values in the
		 * per-shot buffers, the accumulation below would add those into the
		 * Lame totals, and the final extractGradientVector would apply the
		 * chain rule a SECOND time -- inflating the total by ~2*rho*Vp (1e7
		 * here). Measured on a 2-shot run before this copy was added:
		 * |total| = 1.0e-08 against |per-shot| = 8e-16, with cos = 0.94,
		 * i.e. the right shape at a hopelessly wrong scale. This is the same
		 * double-conversion that was fixed in fwi_inversion.c. */
		if (shot_grad) {
			char fname[1024];
			if (shot_vec == NULL) {
				shot_vec = (float *)calloc((size_t)nparam * nmodel, sizeof(float));
				shot_c1  = (float *)calloc(sizem, sizeof(float));
				shot_c2  = (float *)calloc(sizem, sizeof(float));
				shot_c3  = (float *)calloc(sizem, sizeof(float));
			}
			memcpy(shot_c1, g1_shot, sizem * sizeof(float));
			memcpy(shot_c2, g2_shot, sizem * sizeof(float));
			memcpy(shot_c3, g3_shot, sizem * sizeof(float));
			extractGradientVector(shot_vec, shot_c1, shot_c2, shot_c3,
			                      &mod, &bnd, param);
			snprintf(fname, sizeof(fname), "%s_shot%03d_%s_raw.su",
				out_base, ishot, (param == 1) ? "lam" : "vp");
			writesufile(fname, shot_vec, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "%s_shot%03d_%s_raw.su",
				out_base, ishot, (param == 1) ? "mu" : "vs");
			writesufile(fname, shot_vec + nmodel, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			snprintf(fname, sizeof(fname), "%s_shot%03d_rho_raw.su",
				out_base, ishot);
			writesufile(fname, shot_vec + 2 * nmodel, (size_t)mod.nz, (size_t)mod.nx,
				mod.z0, mod.x0, mod.dz, mod.dx);
			if (verbose)
				vmess("  Wrote per-shot gradient %s_shot%03d_*_raw.su",
					out_base, ishot);
		}

		/* Accumulate into global gradient */
		for (i = 0; i < (int)sizem; i++) {
			grad1[i] += g1_shot[i];
			grad2[i] += g2_shot[i];
			grad3[i] += g3_shot[i];
		}

		cleanCheckpoints(&chk);
	}

	/* The lag cube that corresponds to <out_base>_{vp,vs}_raw.su is the sum
	 * over shots, exactly as grad1..3 above are. */
	if (lag_nfreq > 0) {
		lagGradientWriteStack(&mod, out_base, verbose);
		lagGradientFree();
		if (lag_mask) free(lag_mask);
	}

	if (ew) energyWeightsFree(ew);
	if (kh) khWeightsFree(kh);

	vmess("=== Total misfit: %.6e ===", total_misfit);

	/* ============================================================ */
	/* Smoothing (optional)                                          */
	/* ============================================================ */
	{
		int ibx = mod.ioPx, ibz = mod.ioPz;
		if (bnd.lef == 4 || bnd.lef == 2) ibx += bnd.ntap;
		if (bnd.top == 4 || bnd.top == 2) ibz += bnd.ntap;

		if (smooth_grad > 0) {
			vmess("Smoothing gradient: half-width=%d grid points", smooth_grad);
			smooth2d_padded(grad1, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_grad, smooth_grad);
			smooth2d_padded(grad2, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_grad, smooth_grad);
			smooth2d_padded(grad3, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_grad, smooth_grad);
		}
		if (smooth_hess > 0) {
			vmess("Smoothing Hessian: half-width=%d grid points", smooth_hess);
			smooth2d_padded(hess_lam, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_hess, smooth_hess);
			smooth2d_padded(hess_muu, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_hess, smooth_hess);
			smooth2d_padded(hess_rho, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_hess, smooth_hess);
			smooth2d_padded(hess_lam_muu, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_hess, smooth_hess);
			smooth2d_padded(hess_lam_rho, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_hess, smooth_hess);
			smooth2d_padded(hess_muu_rho, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_hess, smooth_hess);
			smooth2d_padded(wfld_energy, mod.nax, mod.naz, mod.nx, mod.nz,
			                ibx, ibz, smooth_hess, smooth_hess);
		}
	}

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
	if (scaling == 1) {
		/* Range normalization: m0 = m_max - m_min from bounds, no shift */
		float vp_mn, vp_mx, vs_mn, vs_mx, rho_mn, rho_mx;
		if (!getparfloat("vp_min", &vp_mn) || !getparfloat("vp_max", &vp_mx))
			verr("scaling=1 requires vp_min= and vp_max= parameters");
		if (!getparfloat("vs_min", &vs_mn)) vs_mn = 0.0f;
		if (!getparfloat("vs_max", &vs_mx)) vs_mx = vp_mx * 0.7f;
		if (!getparfloat("rho_min", &rho_mn)) rho_mn = 500.0f;
		if (!getparfloat("rho_max", &rho_mx)) rho_mx = 5000.0f;
		if (param == 2) {
			float p_min[3] = {vp_mn, vs_mn, rho_mn};
			float p_max[3] = {vp_mx, vs_mx, rho_mx};
			scaling_compute_m0_from_bounds(p_min, p_max, nparam, m0, m_shift);
		} else {
			float lam_mn = rho_mn * (vp_mn*vp_mn - 2.0f*vs_mx*vs_mx);
			float lam_mx = rho_mx * (vp_mx*vp_mx - 2.0f*vs_mn*vs_mn);
			float mu_mn  = rho_mn * vs_mn * vs_mn;
			float mu_mx  = rho_mx * vs_mx * vs_mx;
			float p_min[3] = {lam_mn, mu_mn, rho_mn};
			float p_max[3] = {lam_mx, mu_mx, rho_mx};
			scaling_compute_m0_from_bounds(p_min, p_max, nparam, m0, m_shift);
		}
	} else {
		/* scaling=0: no scaling, m0=1, m_shift=0 */
		int p;
		for (p = 0; p < nparam; p++) { m0[p] = 1.0f; m_shift[p] = 0.0f; }
	}
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

		/* Additional tunable boost per parameter (Yang et al. 2018, eq 48) */
		float boost1 = 1.0f, boost2 = 1.0f, boost3 = 1.0f;
		getparfloat("precond_boost_1", &boost1);
		getparfloat("precond_boost_2", &boost2);
		getparfloat("precond_boost_3", &boost3);
		s1 *= boost1;
		s2 *= boost2;
		s3 *= boost3;

		int elastic = (mod.ischeme > 2);

		vmess("  s1=%.6e  s2=%.6e  s3=%.6e (boost: %.2f, %.2f, %.2f)",
		      s1, s2, s3, boost1, boost2, boost3);

		buildBlockPrecond(hess_lam, hess_muu, hess_rho,
			hess_lam_muu, hess_lam_rho, hess_muu_rho,
			&mod, &bnd, param, elastic,
			precond_eps, s1, s2, s3,
			P11, P12, P13, P22, P23, P33,
			nmodel, 0, precond_blend, precond_alpha,
			(precond_mode == 1) ? wfld_energy : NULL,
			mod.x0, mod.x0 + (mod.nx-1)*mod.dx,
			precond_mode);
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
			nmodel, nparam, 1);

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
	if (shot_vec) free(shot_vec);
	if (shot_c1) { free(shot_c1); free(shot_c2); free(shot_c3); }
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

	/* Clean up observed data backups from bandpass filtering */
	if (band_flo > 0.0f || band_fhi > 0.0f)
		bandpass_cleanup_obsbackups(file_obs, comp_str, shot.n);

	return 0;
}
