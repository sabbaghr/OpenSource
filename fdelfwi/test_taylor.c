/*
 * test_taylor.c - Taylor (finite-difference) gradient verification test.
 *
 * Verifies that the FWI gradient correctly predicts first-order misfit
 * changes by checking:
 *
 *   r(eps) = [phi(m0+eps*dm) - phi(m0)] / [eps * <g, dm>] -> 1.0
 *
 * as eps -> 0. Also checks second-order convergence:
 *   |phi(m0+eps*dm) - phi(m0) - eps*<g,dm>| = O(eps^2)
 *
 * Parameters (same interface as fdelmodc, plus):
 *   file_obs=     observed data base name (REQUIRED)
 *   chk_skipdt=   checkpoint interval in time steps (100)
 *   chk_base=     checkpoint file base path (chk_taylor)
 *   pert_pct=     model perturbation amplitude fraction (0.01)
 *   seed=         random number seed (12345)
 *   comp=_rvz     component suffix for recording
 *   test_param=0  parameter to test (0=all, 1=Vp, 2=Vs, 3=rho, 4=rho_Lame)
 *   neps=10       number of epsilon values
 *   eps_min=1e-8  smallest epsilon
 *   eps_max=1e-1  largest epsilon
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

void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho,
                               size_t sizem);

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

char *sdoc[] = {
" ",
" test_taylor - Taylor (finite-difference) gradient verification test",
" ",
" Same parameters as fdelmodc, plus:",
"   file_obs=     observed data base name (REQUIRED)",
"   chk_skipdt=   checkpoint interval in time steps (100)",
"   chk_base=     checkpoint file base path (chk_taylor)",
"   pert_pct=     model perturbation amplitude fraction (0.01)",
"   seed=         random number seed (12345)",
"   comp=_rvz     component suffix for recording",
"   test_param=0  parameter to test (0=all, 1=Vp, 2=Vs, 3=rho)",
"   neps=10       number of epsilon values",
"   eps_min=1e-8  smallest epsilon",
"   eps_max=1e-1  largest epsilon",
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
	checkpointPar chk;
	adjSrcPar adj;
	float **src_nwav;
	float  sinkvel;
	double t0, t1;
	size_t nsamp, sizew, sizem;
	int    n1, ix, iz, ir, i, k;
	int    ioPx, ioPz;
	int    ixsrc, izsrc, fileno;
	int    verbose;

	/* Test parameters */
	char  *chk_base, *comp_str, *file_obs;
	int    chk_skipdt, seed, test_param, neps;
	float  pert_pct, eps_min_f, eps_max_f;
	double eps_min, eps_max;
	float *grad_vp, *grad_vs, *grad_rho;

	/* Model perturbation arrays (interior grid) */
	float *delta_cp, *delta_cs, *delta_ro;
	/* Base model arrays (interior grid) */
	float *cp_base, *cs_base, *ro_base;

	/* Component parsing */
	char   comp_buf[1024];
	char  *comp_suffixes[MAX_COMP];
	int    ncomp;
	char  *token;

	/* Observed/synthetic file names */
	char   obs_names[MAX_COMP][1024], syn_names[MAX_COMP][1024];
	const char *obs_arr[MAX_COMP], *syn_arr[MAX_COMP];

	t0 = wallclock_time();

	/* ============================================================ */
	/* Parse parameters                                              */
	/* ============================================================ */
	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;
	if (!getparstring("file_obs", &file_obs))
		verr("file_obs= parameter is required (observed data base name).");
	if (!getparint("chk_skipdt", &chk_skipdt)) chk_skipdt = 100;
	if (!getparstring("chk_base", &chk_base)) chk_base = "chk_taylor";
	if (!getparint("seed", &seed)) seed = 12345;
	if (!getparfloat("pert_pct", &pert_pct)) pert_pct = 0.01f;
	if (!getparstring("comp", &comp_str)) comp_str = "_rvz";
	if (!getparint("test_param", &test_param)) test_param = 0;
	if (!getparint("neps", &neps)) neps = 10;
	if (!getparfloat("eps_min", &eps_min_f)) eps_min_f = 1.0e-8f;
	if (!getparfloat("eps_max", &eps_max_f)) eps_max_f = 1.0e-1f;
	eps_min = (double)eps_min_f;
	eps_max = (double)eps_max_f;

	/* Parse component suffixes */
	strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
	comp_buf[sizeof(comp_buf) - 1] = '\0';
	ncomp = 0;
	token = strtok(comp_buf, ",");
	while (token && ncomp < MAX_COMP) {
		comp_suffixes[ncomp++] = token;
		token = strtok(NULL, ",");
	}

	/* ============================================================ */
	/* Standard model setup (same as test_dotproduct.c)              */
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

	printf("=== Taylor Gradient Verification Test ===\n");
	vmess("Grid: nx=%d nz=%d (padded: nax=%d naz=%d)", mod.nx, mod.nz, mod.nax, mod.naz);
	vmess("Source at grid ix=%d iz=%d", ixsrc, izsrc);
	vmess("Perturbation: %.2f%%  Seed: %d", pert_pct * 100.0f, seed);
	vmess("Epsilon range: [%.1e, %.1e], %d values", eps_min, eps_max, neps);
	vmess("Observed data: %s", file_obs);

	/* ============================================================ */
	/* Step 1: Forward model m0 with checkpointing                   */
	/* ============================================================ */
	vmess("--- Step 1: Forward model (base m0) ---");
	{
		checkpointPar chk_base_fwd;
		initCheckpoints(&chk_base_fwd, &mod, chk_skipdt, 0, chk_base);

		fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna,
			ixsrc, izsrc, src_nwav, 0, 1, fileno, &chk_base_fwd, verbose);

		/* Save checkpoint metadata for adjoint step */
		memcpy(&chk, &chk_base_fwd, sizeof(checkpointPar));
	}
	vmess("Base forward complete.");

	/* ============================================================ */
	/* Step 2: Compute base misfit phi0                              */
	/* ============================================================ */
	vmess("--- Step 2: Compute base misfit phi(m0) ---");

	/* Build observed and synthetic file name arrays */
	for (k = 0; k < ncomp; k++) {
		snprintf(obs_names[k], sizeof(obs_names[k]), "%s%s.su",
			file_obs, comp_suffixes[k]);
		snprintf(syn_names[k], sizeof(syn_names[k]), "%s_000%s.su",
			rec.file_rcv, comp_suffixes[k]);
		obs_arr[k] = obs_names[k];
		syn_arr[k] = syn_names[k];
	}

	float phi0 = computeResidual(ncomp, obs_arr, syn_arr,
		"residual_taylor.su", MISFIT_L2, verbose);

	vmess("phi(m0) = %.10e", phi0);

	/* ============================================================ */
	/* Step 3: Adjoint pass to get gradient                          */
	/* ============================================================ */
	vmess("--- Step 3: Adjoint backpropagation -> gradient ---");

	memset(&adj, 0, sizeof(adjSrcPar));
	readResidual("residual_taylor.su", &adj, &mod, &bnd);
	vmess("Loaded %d adjoint source traces, nt=%d", adj.nsrc, adj.nt);

	grad_vp  = (float *)calloc(sizem, sizeof(float));
	grad_vs  = (float *)calloc(sizem, sizeof(float));
	grad_rho = (float *)calloc(sizem, sizeof(float));

	{
		snaPar sna_adj;
		memset(&sna_adj, 0, sizeof(snaPar));

		/* param=1: returns Lame gradients (g_lambda, g_mu, g_rho_direct) */
		adj_shot(&mod, &src, &wav, &bnd, &rec, &adj,
			ixsrc, izsrc, src_nwav, &chk, &sna_adj,
			grad_vp, grad_vs, grad_rho, 1, verbose);
	}

	/* Save Lame gradients before chain rule conversion (for diagnostics) */
	float *g_lam_save = (float *)malloc(sizem * sizeof(float));
	float *g_mu_save  = (float *)malloc(sizem * sizeof(float));
	float *g_rho_direct_save = (float *)malloc(sizem * sizeof(float));
	memcpy(g_lam_save, grad_vp, sizem * sizeof(float));
	memcpy(g_mu_save, grad_vs, sizem * sizeof(float));
	memcpy(g_rho_direct_save, grad_rho, sizem * sizeof(float));

	/* Convert Lame gradients to velocity parameterization */
	convertGradientToVelocity(grad_vp, grad_vs, grad_rho,
	                          mod.cp, mod.cs, mod.rho, sizem);

	/* Done with checkpoints and adjoint sources */
	cleanCheckpoints(&chk);
	freeResidual(&adj);

	vmess("Gradient computed in velocity space (g_Vp, g_Vs, g_rho).");

	/* Diagnostic: gradient norms (Lame and velocity) */
	{
		double norm_vp = 0.0, norm_vs = 0.0, norm_rho = 0.0;
		double norm_lam = 0.0, norm_mu = 0.0, norm_rho_d = 0.0;
		for (ix = 0; ix < mod.nx; ix++) {
			for (iz = 0; iz < mod.nz; iz++) {
				int ig = (ix + ioPx) * mod.naz + iz + ioPz;
				norm_vp  += (double)grad_vp[ig]  * (double)grad_vp[ig];
				norm_vs  += (double)grad_vs[ig]  * (double)grad_vs[ig];
				norm_rho += (double)grad_rho[ig] * (double)grad_rho[ig];
				norm_lam += (double)g_lam_save[ig] * (double)g_lam_save[ig];
				norm_mu  += (double)g_mu_save[ig]  * (double)g_mu_save[ig];
				norm_rho_d += (double)g_rho_direct_save[ig] * (double)g_rho_direct_save[ig];
			}
		}
		vmess("DIAG Lame: ||g_lam||=%.6e  ||g_mu||=%.6e  ||g_rho_direct||=%.6e",
			sqrt(norm_lam), sqrt(norm_mu), sqrt(norm_rho_d));
		vmess("DIAG Vel:  ||g_Vp||=%.6e  ||g_Vs||=%.6e  ||g_rho_full||=%.6e",
			sqrt(norm_vp), sqrt(norm_vs), sqrt(norm_rho));
	}

	/* ============================================================ */
	/* Step 4: Generate random perturbation dm                       */
	/* ============================================================ */
	vmess("--- Step 4: Generate random perturbation ---");

	delta_cp = (float *)calloc((size_t)mod.nx * mod.nz, sizeof(float));
	delta_cs = (float *)calloc((size_t)mod.nx * mod.nz, sizeof(float));
	delta_ro = (float *)calloc((size_t)mod.nx * mod.nz, sizeof(float));

	{
		float cp0 = 0.5f * (mod.cp_min + mod.cp_max);
		float cs0 = 0.5f * (mod.cs_min + mod.cs_max);
		float ro0 = 0.5f * (mod.ro_min + mod.ro_max);

		srand48(seed);
		for (ix = 0; ix < mod.nx; ix++) {
			for (iz = 0; iz < mod.nz; iz++) {
				int idx = ix * mod.nz + iz;
				delta_cp[idx] = pert_pct * cp0 * (float)(drand48() * 2.0 - 1.0);
				delta_cs[idx] = pert_pct * cs0 * (float)(drand48() * 2.0 - 1.0);
				delta_ro[idx] = pert_pct * ro0 * (float)(drand48() * 2.0 - 1.0);
			}
		}

		/* Zero out perturbations based on test_param */
		if (test_param == 1) {
			memset(delta_cs, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			memset(delta_ro, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			vmess("test_param=1: Vp perturbation only");
		} else if (test_param == 2) {
			memset(delta_cp, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			memset(delta_ro, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			vmess("test_param=2: Vs perturbation only");
		} else if (test_param == 3) {
			memset(delta_cp, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			memset(delta_cs, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			vmess("test_param=3: rho perturbation only");
		} else if (test_param == 4) {
			/* Lame density direct: perturb rho keeping lambda,mu fixed.
			 * delta_cp, delta_cs set to 0 here (Vp/Vs adjustments
			 * computed in the epsilon loop to keep lambda,mu constant). */
			memset(delta_cp, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			memset(delta_cs, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			vmess("test_param=4: Lame density direct (rho perturbation, lambda/mu fixed)");
		}

		/* Zero perturbation outside gradient accumulation region */
		{
			int margin_lef = 0, margin_rig = 0, margin_top = 0, margin_bot = 0;
			int half_fd = mod.iorder / 2;

			if (bnd.lef == 4 || bnd.lef == 2) margin_lef = bnd.ntap;
			if (bnd.rig == 4 || bnd.rig == 2) margin_rig = bnd.ntap;
			if (bnd.top == 4 || bnd.top == 2) margin_top = bnd.ntap;
			if (bnd.bot == 4 || bnd.bot == 2) margin_bot = bnd.ntap;

			if (bnd.top == 1 && margin_top < 1) margin_top = 1;

			if (margin_lef < half_fd) margin_lef = half_fd;
			if (margin_rig < half_fd) margin_rig = half_fd;
			if (margin_top < half_fd) margin_top = half_fd;
			if (margin_bot < half_fd) margin_bot = half_fd;

			for (ix = 0; ix < mod.nx; ix++) {
				for (iz = 0; iz < mod.nz; iz++) {
					if (ix < margin_lef || ix >= mod.nx - margin_rig ||
					    iz < margin_top  || iz >= mod.nz - margin_bot) {
						int idx = ix * mod.nz + iz;
						delta_cp[idx] = 0.0f;
						delta_cs[idx] = 0.0f;
						delta_ro[idx] = 0.0f;
					}
				}
			}
			vmess("Zeroed perturbation in boundary margin (l=%d r=%d t=%d b=%d)",
				margin_lef, margin_rig, margin_top, margin_bot);
		}

		/* Zero perturbation near source injection point */
		{
			int src_margin = 3;
			int iix, iiz;
			for (iix = ixsrc - src_margin; iix <= ixsrc + src_margin; iix++) {
				for (iiz = izsrc - src_margin; iiz <= izsrc + src_margin; iiz++) {
					if (iix >= 0 && iix < mod.nx && iiz >= 0 && iiz < mod.nz) {
						int idx = iix * mod.nz + iiz;
						delta_cp[idx] = 0.0f;
						delta_cs[idx] = 0.0f;
						delta_ro[idx] = 0.0f;
					}
				}
			}
			vmess("Zeroed perturbation near source (ix=%d, iz=%d) +/- %d",
				ixsrc, izsrc, src_margin);
		}
	}

	/* ============================================================ */
	/* Step 5: Compute predicted directional derivative <g, dm>      */
	/* ============================================================ */
	vmess("--- Step 5: Compute <grad, dm> ---");

	double predicted = 0.0;
	{
		double dot_vp = 0.0, dot_vs = 0.0, dot_rho = 0.0;
		double dot_rho_direct = 0.0, dot_rho_chain_lam = 0.0, dot_rho_chain_mu = 0.0;
		for (ix = 0; ix < mod.nx; ix++) {
			for (iz = 0; iz < mod.nz; iz++) {
				int idx = ix * mod.nz + iz;
				int ig  = (ix + ioPx) * mod.naz + iz + ioPz;
				dot_vp  += (double)grad_vp[ig]  * (double)delta_cp[idx];
				dot_vs  += (double)grad_vs[ig]  * (double)delta_cs[idx];
				/* For test_param=4, use g_rho_direct (Lame space) */
				if (test_param == 4)
					dot_rho += (double)g_rho_direct_save[ig] * (double)delta_ro[idx];
				else
					dot_rho += (double)grad_rho[ig] * (double)delta_ro[idx];
				/* Decompose g_rho_full = g_rho_direct + g_lam*(Vp²-2Vs²) + g_mu*Vs² */
				double vp2 = (double)mod.cp[ig] * (double)mod.cp[ig];
				double vs2 = (double)mod.cs[ig] * (double)mod.cs[ig];
				dot_rho_direct   += (double)g_rho_direct_save[ig] * (double)delta_ro[idx];
				dot_rho_chain_lam += (double)g_lam_save[ig] * (vp2 - 2.0*vs2) * (double)delta_ro[idx];
				dot_rho_chain_mu  += (double)g_mu_save[ig] * vs2 * (double)delta_ro[idx];
			}
		}
		predicted = dot_vp + dot_vs + dot_rho;
		vmess("DIAG: <g_Vp, dVp>=%.6e  <g_Vs, dVs>=%.6e  <g_rho, drho>=%.6e",
			dot_vp, dot_vs, dot_rho);
		vmess("DIAG rho decomp: direct=%.6e  chain_lam=%.6e  chain_mu=%.6e  sum=%.6e",
			dot_rho_direct, dot_rho_chain_lam, dot_rho_chain_mu,
			dot_rho_direct + dot_rho_chain_lam + dot_rho_chain_mu);
		if (test_param == 4)
			vmess("DIAG: test_param=4 uses g_rho_DIRECT for prediction (no chain rule)");
		vmess("<grad, dm> = %.10e", predicted);
	}

	if (fabs(predicted) < 1.0e-30) {
		verr("Predicted directional derivative is zero. "
		     "Check that gradient and perturbation are nonzero.");
	}

	/* ============================================================ */
	/* Step 6: Extract base model (for perturbation loop)            */
	/* ============================================================ */
	cp_base = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
	cs_base = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
	ro_base = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));

	for (ix = 0; ix < mod.nx; ix++) {
		for (iz = 0; iz < mod.nz; iz++) {
			int ig  = (ix + ioPx) * mod.naz + iz + ioPz;
			int idx = ix * mod.nz + iz;
			cp_base[idx] = mod.cp[ig];
			cs_base[idx] = mod.cs[ig];
			ro_base[idx] = mod.rho[ig];
		}
	}

	/* ============================================================ */
	/* Step 7: Epsilon sweep                                         */
	/* ============================================================ */
	vmess("--- Step 7: Epsilon sweep ---");

	/* Generate logarithmically-spaced epsilon values */
	double *eps_values  = (double *)malloc(neps * sizeof(double));
	double *phi_pert    = (double *)malloc(neps * sizeof(double));
	double *ratio_arr   = (double *)malloc(neps * sizeof(double));
	double *error1_arr  = (double *)malloc(neps * sizeof(double));
	double *error2_arr  = (double *)malloc(neps * sizeof(double));
	double *order1_arr  = (double *)malloc(neps * sizeof(double));
	double *order2_arr  = (double *)malloc(neps * sizeof(double));

	for (i = 0; i < neps; i++) {
		if (neps > 1)
			eps_values[i] = pow(10.0, log10(eps_max) +
				(double)i * (log10(eps_min) - log10(eps_max)) / (double)(neps - 1));
		else
			eps_values[i] = eps_max;
	}

	/* Perturbed model arrays (reused across iterations) */
	float *cp_pert = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
	float *cs_pert = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
	float *ro_pert = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));

	/* Save original file pointers */
	char *orig_cp  = mod.file_cp;
	char *orig_cs  = mod.file_cs;
	char *orig_ro  = mod.file_ro;
	char *orig_rcv = rec.file_rcv;

	for (int ieps = 0; ieps < neps; ieps++) {
		double eps = eps_values[ieps];

		vmess("  eps=%.2e (%d/%d)", eps, ieps + 1, neps);

		/* (a) Build perturbed model: m0 + eps*dm */
		for (ix = 0; ix < mod.nx; ix++) {
			for (iz = 0; iz < mod.nz; iz++) {
				int idx = ix * mod.nz + iz;
				ro_pert[idx] = ro_base[idx] + (float)(eps * (double)delta_ro[idx]);
				if (test_param == 4) {
					/* Keep lambda,mu fixed: adjust Vp,Vs for perturbed rho
					 * lambda0 = rho0*(Vp0^2 - 2*Vs0^2), mu0 = rho0*Vs0^2
					 * Vp_pert = sqrt((lambda0+2*mu0)/rho_pert) = Vp0*sqrt(rho0/rho_pert)
					 * Vs_pert = sqrt(mu0/rho_pert) = Vs0*sqrt(rho0/rho_pert) */
					double scale = sqrt((double)ro_base[idx] / (double)ro_pert[idx]);
					cp_pert[idx] = (float)((double)cp_base[idx] * scale);
					cs_pert[idx] = (float)((double)cs_base[idx] * scale);
				} else {
					cp_pert[idx] = cp_base[idx] + (float)(eps * (double)delta_cp[idx]);
					cs_pert[idx] = cs_base[idx] + (float)(eps * (double)delta_cs[idx]);
				}
			}
		}

		/* (b) Write perturbed model to SU files */
		writesufile("cp_taylor_pert.su", cp_pert, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		writesufile("cs_taylor_pert.su", cs_pert, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		writesufile("ro_taylor_pert.su", ro_pert, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);

		/* (c) Reload perturbed model */
		mod.file_cp  = "cp_taylor_pert.su";
		mod.file_cs  = "cs_taylor_pert.su";
		mod.file_ro  = "ro_taylor_pert.su";
		rec.file_rcv = "syn_taylor_pert";

		readModel(&mod, &bnd);
		freeStoreSourceOnSurface();
		allocStoreSourceOnSurface(src);

		/* (d) Forward model (no checkpoints needed - only need misfit) */
		{
			checkpointPar chk_pert;
			initCheckpoints(&chk_pert, &mod, mod.nt + 10000, 0, "chk_taylor_dummy");

			snaPar sna_off;
			memset(&sna_off, 0, sizeof(snaPar));

			fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna_off,
				ixsrc, izsrc, src_nwav, 0, 1, fileno, &chk_pert, verbose);

			cleanCheckpoints(&chk_pert);
		}

		/* (e) Compute perturbed misfit */
		{
			char syn_pert_names[MAX_COMP][1024];
			const char *syn_pert_arr[MAX_COMP];
			for (k = 0; k < ncomp; k++) {
				snprintf(syn_pert_names[k], sizeof(syn_pert_names[k]),
					"syn_taylor_pert_000%s.su", comp_suffixes[k]);
				syn_pert_arr[k] = syn_pert_names[k];
			}

			float phi_p = computeResidual(ncomp, obs_arr, syn_pert_arr,
				"residual_taylor_pert.su", MISFIT_L2, 0);

			phi_pert[ieps] = (double)phi_p;
		}

		/* (f) Compute Taylor test quantities */
		double dphi     = phi_pert[ieps] - (double)phi0;
		double eps_gTdm = eps * predicted;

		ratio_arr[ieps]  = (fabs(eps_gTdm) > 1.0e-30) ? dphi / eps_gTdm : 0.0;
		error1_arr[ieps] = fabs(dphi);                    /* zeroth order */
		error2_arr[ieps] = fabs(dphi - eps_gTdm);         /* first order  */

		/* Convergence orders */
		if (ieps > 0 && error1_arr[ieps - 1] > 0.0 && error1_arr[ieps] > 0.0) {
			order1_arr[ieps] = log(error1_arr[ieps] / error1_arr[ieps - 1]) /
				log(eps_values[ieps] / eps_values[ieps - 1]);
		} else {
			order1_arr[ieps] = 0.0;
		}
		if (ieps > 0 && error2_arr[ieps - 1] > 0.0 && error2_arr[ieps] > 0.0) {
			order2_arr[ieps] = log(error2_arr[ieps] / error2_arr[ieps - 1]) /
				log(eps_values[ieps] / eps_values[ieps - 1]);
		} else {
			order2_arr[ieps] = 0.0;
		}

		vmess("  phi_pert=%.6e  dphi=%+.6e  ratio=%.6f  err2=%.2e",
			phi_pert[ieps], dphi, ratio_arr[ieps], error2_arr[ieps]);

		/* (g) Restore base model */
		mod.file_cp  = orig_cp;
		mod.file_cs  = orig_cs;
		mod.file_ro  = orig_ro;
		rec.file_rcv = orig_rcv;

		readModel(&mod, &bnd);
		freeStoreSourceOnSurface();
		allocStoreSourceOnSurface(src);
	}

	/* ============================================================ */
	/* Step 8: Print results table                                   */
	/* ============================================================ */
	printf("\n=========================================================\n");
	printf("  TAYLOR GRADIENT TEST RESULTS\n");
	printf("=========================================================\n");
	printf("  phi(m0)     = %.10e\n", (double)phi0);
	printf("  <grad, dm>  = %.10e\n", predicted);
	printf("\n");
	printf("  %-12s  %-14s  %-14s  %-10s  %-14s  %-6s  %-6s\n",
		"epsilon", "phi_pert", "dphi", "ratio", "err2", "ord1", "ord2");
	printf("  %-12s  %-14s  %-14s  %-10s  %-14s  %-6s  %-6s\n",
		"--------", "---------", "---------", "-----", "---------", "----", "----");

	for (i = 0; i < neps; i++) {
		printf("  %.4e  %+.6e  %+.6e  %10.6f  %.6e  %5.2f  %5.2f\n",
			eps_values[i], phi_pert[i],
			phi_pert[i] - (double)phi0,
			ratio_arr[i], error2_arr[i],
			order1_arr[i], order2_arr[i]);
	}

	/* ============================================================ */
	/* Step 9: PASS/FAIL evaluation                                  */
	/* ============================================================ */
	int pass = 0;
	double best_ratio_err = 1.0e30;
	int best_idx = 0;

	/* Find best ratio among intermediate epsilon values */
	for (i = 1; i < neps - 1; i++) {
		double re = fabs(ratio_arr[i] - 1.0);
		if (re < best_ratio_err) {
			best_ratio_err = re;
			best_idx = i;
		}
	}

	/* Check for second-order convergence in intermediate range */
	int has_order2 = 0;
	for (i = 2; i < neps - 1; i++) {
		if (order2_arr[i] > 1.5 && order2_arr[i] < 2.5) has_order2 = 1;
	}

	pass = (best_ratio_err < 0.05);

	printf("\n  Best |ratio - 1| = %.6e (at eps=%.2e)\n",
		best_ratio_err, eps_values[best_idx]);
	printf("  Second-order convergence observed: %s\n",
		has_order2 ? "YES" : "NO");
	printf("  %s (best |ratio-1| %s 0.05)\n\n",
		pass ? "PASS" : "FAIL",
		pass ? "<" : ">=");

	t1 = wallclock_time();
	printf("Total wall time: %.2f s\n", t1 - t0);

	/* ============================================================ */
	/* Cleanup                                                       */
	/* ============================================================ */
	free(grad_vp);
	free(grad_vs);
	free(grad_rho);
	free(delta_cp);
	free(delta_cs);
	free(delta_ro);
	free(cp_base);
	free(cs_base);
	free(ro_base);
	free(cp_pert);
	free(cs_pert);
	free(ro_pert);
	free(eps_values);
	free(phi_pert);
	free(ratio_arr);
	free(error1_arr);
	free(error2_arr);
	free(order1_arr);
	free(order2_arr);

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

	printf("\n=== Taylor test completed ===\n");

	return pass ? 0 : 1;
}
