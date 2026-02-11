/*
 * test_dotproduct.c - Claerbout dot product test for elastic forward-adjoint.
 *
 * Implements two verification tests following Peter Mora (Appendix B):
 *
 * Test 1: Summation By Parts (SBP) for staggered-grid FD stencils
 *   Verifies: <D_minus(f), g> + <f, D_plus(g)> = 0  (to machine precision)
 *   for FD orders 4, 6, 8 in both x and z directions.
 *
 * Test 2: Claerbout dot product test (system-level)
 *   Verifies: x^T A^T y = y^T A x  +/- epsilon
 *   where A is the linearized forward operator (computed as f(m0+x)-f(m0))
 *   and A^T is the adjoint operator (computed via adj_shot).
 *   Expected epsilon ~ 10^{-3} for 2D elastic (Peter Mora B.3.3).
 *
 * Parameters (same interface as test_fwi_gradient / fdelmodc):
 *   chk_skipdt=   checkpoint interval in time steps (100)
 *   chk_base=     checkpoint file base path (chk)
 *   pert_pct=     model perturbation amplitude as fraction (0.01)
 *   seed=         random number seed (12345)
 *   comp=_rvz     component suffix for recording (default: _rvz)
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
             int param, int verbose);

void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho,
                               size_t sizem);

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

char *sdoc[] = {
" ",
" test_dotproduct - Claerbout dot product test for elastic forward-adjoint",
" ",
" Tests:",
"   1. SBP stencil test (no parameters needed)",
"   2. Full forward-adjoint dot product test (needs model + source)",
" ",
" Same parameters as fdelmodc, plus:",
"   chk_skipdt=   checkpoint interval in time steps (100)",
"   chk_base=     checkpoint file base path (chk)",
"   pert_pct=     model perturbation amplitude fraction (0.01)",
"   seed=         random number seed (12345)",
"   comp=_rvz     component suffix for recording",
" ",
NULL};


/*====================================================================
 * Test 1: Summation By Parts (SBP) for staggered-grid stencils
 *
 * For the staggered grid, the backward difference (velocity update)
 * and forward difference (stress update) must satisfy:
 *   sum_i D_minus(f)[i] * g[i] = -sum_i f[i] * D_plus(g)[i]
 * for interior points with zero boundary padding.
 *====================================================================*/

static int test_sbp_1d(int order, int N)
{
	double *f, *g;
	double lhs, rhs, rel_err;
	int i, half;
	int pass = 1;

	/* Stencil coefficients */
	double c1, c2, c3, c4;
	c1 = c2 = c3 = c4 = 0.0;

	switch (order) {
		case 4:
			c1 = 9.0/8.0;
			c2 = -1.0/24.0;
			break;
		case 6:
			c1 = 75.0/64.0;
			c2 = -25.0/384.0;
			c3 = 3.0/640.0;
			break;
		case 8:
			c1 = 1225.0/1024.0;
			c2 = -245.0/3072.0;
			c3 = 49.0/5120.0;
			c4 = -5.0/7168.0;
			break;
		default:
			fprintf(stderr, "test_sbp: unsupported order %d\n", order);
			return 0;
	}

	half = order / 2;
	f = (double *)calloc(N, sizeof(double));
	g = (double *)calloc(N, sizeof(double));

	/* Fill interior with random values, leave boundary zeros */
	srand48(42 + order);
	for (i = half; i < N - half; i++) {
		f[i] = drand48() * 2.0 - 1.0;
		g[i] = drand48() * 2.0 - 1.0;
	}

	/* Compute lhs = sum D_minus(f)[i] * g[i]
	 * D_minus[i] = c1*(f[i] - f[i-1]) + c2*(f[i+1] - f[i-2]) + ... */
	lhs = 0.0;
	for (i = half; i < N - half; i++) {
		double Df = c1*(f[i] - f[i-1]) + c2*(f[i+1] - f[i-2]);
		if (order >= 6) Df += c3*(f[i+2] - f[i-3]);
		if (order >= 8) Df += c4*(f[i+3] - f[i-4]);
		lhs += Df * g[i];
	}

	/* Compute rhs = sum f[i] * D_plus(g)[i]
	 * D_plus[i] = c1*(g[i+1] - g[i]) + c2*(g[i+2] - g[i-1]) + ... */
	rhs = 0.0;
	for (i = half; i < N - half; i++) {
		double Dg = c1*(g[i+1] - g[i]) + c2*(g[i+2] - g[i-1]);
		if (order >= 6) Dg += c3*(g[i+3] - g[i-2]);
		if (order >= 8) Dg += c4*(g[i+4] - g[i-3]);
		rhs += f[i] * Dg;
	}

	rel_err = fabs(lhs + rhs) / (fabs(lhs) > 0.0 ? fabs(lhs) : 1.0);
	pass = (rel_err < 1.0e-10);

	printf("  Order %d 1D (N=%d): |lhs+rhs|/|lhs| = %.2e  %s\n",
		order, N, rel_err, pass ? "PASS" : "FAIL");

	free(f);
	free(g);
	return pass;
}

static int test_sbp(void)
{
	int all_pass = 1;
	int N = 200;

	printf("\n--- Test 1: Summation By Parts (SBP) ---\n");

	all_pass &= test_sbp_1d(4, N);
	all_pass &= test_sbp_1d(6, N);
	all_pass &= test_sbp_1d(8, N);

	/* Larger array to check accumulation */
	all_pass &= test_sbp_1d(4, 2000);
	all_pass &= test_sbp_1d(8, 2000);

	return all_pass;
}


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
 * Helper: Read all traces from an SU file into a flat array.
 * Returns number of traces read. Sets *ns_out, *dt_us_out.
 * Allocates *data_out = float[ntr * ns].
 * Also optionally returns headers in *hdrs_out (ntr segy structs).
 *====================================================================*/
static int readSUtraces(const char *fname, float **data_out, segy **hdrs_out,
                        int *ns_out, int *dt_us_out)
{
	FILE *fp;
	segy hdr;
	float *data;
	segy *hdrs;
	int ntr, ns, itr;
	size_t tracesize;
	off_t filesize;

	fp = fopen(fname, "r");
	if (!fp) { fprintf(stderr, "readSUtraces: cannot open %s\n", fname); return 0; }

	if (fread(&hdr, 1, TRCBYTES, fp) != TRCBYTES) {
		fclose(fp); return 0;
	}
	ns = (int)hdr.ns;
	*ns_out = ns;
	*dt_us_out = (int)hdr.dt;

	tracesize = TRCBYTES + (size_t)ns * sizeof(float);
	fseeko(fp, 0, SEEK_END);
	filesize = ftello(fp);
	ntr = (int)(filesize / (off_t)tracesize);
	fseeko(fp, 0, SEEK_SET);

	data = (float *)malloc((size_t)ntr * ns * sizeof(float));
	hdrs = (segy *)malloc((size_t)ntr * sizeof(segy));

	for (itr = 0; itr < ntr; itr++) {
		fread(&hdrs[itr], 1, TRCBYTES, fp);
		fread(&data[itr * ns], sizeof(float), ns, fp);
	}
	fclose(fp);

	*data_out = data;
	if (hdrs_out) *hdrs_out = hdrs;
	else free(hdrs);

	return ntr;
}


/*====================================================================
 * Helper: Write SU traces with given headers and data.
 *====================================================================*/
static void writeSUtraces(const char *fname, float *data, segy *hdrs,
                          int ntr, int ns)
{
	FILE *fp;
	int itr;

	fp = fopen(fname, "w");
	if (!fp) { fprintf(stderr, "writeSUtraces: cannot create %s\n", fname); return; }

	for (itr = 0; itr < ntr; itr++) {
		fwrite(&hdrs[itr], 1, TRCBYTES, fp);
		fwrite(&data[itr * ns], sizeof(float), ns, fp);
	}
	fclose(fp);
}


/*====================================================================
 * Test 2: Claerbout dot product test (full system-level)
 *
 * Verifies x^T A^T y = y^T A x  (Sabbagh Eq. B.1)
 *
 * Forward side:  Ax = f(m0+x) - f(m0), then y^T(Ax)
 * Adjoint side:  A^T y via adj_shot with y as residual, then x^T(A^T y)
 *====================================================================*/

int main(int argc, char **argv)
{
	modPar  mod, mod_pert;
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
	char  *chk_base, *comp_str;
	int    chk_skipdt, seed, test_param;
	float  pert_pct;
	float *grad_vp, *grad_vs, *grad_rho;

	/* Dot product variables */
	float *d0_data, *d1_data, *y_data;
	segy  *d0_hdrs;
	int    ntr_d0, ns_d0, dt_us_d0;
	int    ntr_d1, ns_d1, dt_us_d1;
	float  dt_sec;
	double forward_dot, adjoint_dot, ratio, epsilon;

	/* Model perturbation arrays (interior grid) */
	float *delta_cp, *delta_cs, *delta_ro;

	int sbp_pass;

	t0 = wallclock_time();

	/* ============================================================ */
	/* Test 1: SBP (no parameters needed)                           */
	/* ============================================================ */
	printf("=== Elastic Dot Product Test (Sabbagh Appendix B) ===\n");
	sbp_pass = test_sbp();
	printf("SBP test: %s\n\n", sbp_pass ? "ALL PASS" : "FAIL");

	/* ============================================================ */
	/* Test 2: Full Claerbout dot product test                      */
	/* ============================================================ */
	printf("--- Test 2: Claerbout Dot Product Test ---\n");

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;
	if (!getparint("chk_skipdt", &chk_skipdt)) chk_skipdt = 100;
	if (!getparstring("chk_base", &chk_base)) chk_base = "chk_dp";
	if (!getparint("seed", &seed)) seed = 12345;
	if (!getparfloat("pert_pct", &pert_pct)) pert_pct = 0.01f;
	if (!getparstring("comp", &comp_str)) comp_str = "_rvz";
	if (!getparint("test_param", &test_param)) test_param = 0;

	/* Standard setup (same as test_fwi_gradient.c) */
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

	vmess("*******************************************");
	vmess("***** CLAERBOUT DOT PRODUCT TEST       *****");
	vmess("*******************************************");
	vmess("Grid: nx=%d nz=%d (padded: nax=%d naz=%d)", mod.nx, mod.nz, mod.nax, mod.naz);
	vmess("Source at grid ix=%d iz=%d", ixsrc, izsrc);
	vmess("Perturbation: %.2f%%  Seed: %d", pert_pct * 100.0f, seed);
	vmess("Component: %s", comp_str);

	/* ============================================================ */
	/* Step 1: Generate random model perturbation x                 */
	/* ============================================================ */
	vmess("--- Step 1: Generate random perturbation vectors ---");

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
		/* Zero out perturbations based on test_param:
		 * 0=all, 1=Vp only (tests g_Vp), 2=Vs only (tests g_Vs),
		 * 3=rho only (tests g_rho_full) */
		if (test_param == 1) {
			memset(delta_cs, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			memset(delta_ro, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			vmess("test_param=1: Vp perturbation only (tests g_Vp)");
		} else if (test_param == 2) {
			memset(delta_cp, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			memset(delta_ro, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			vmess("test_param=2: Vs perturbation only (tests g_Vs)");
		} else if (test_param == 3) {
			memset(delta_cp, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			memset(delta_cs, 0, (size_t)mod.nx * mod.nz * sizeof(float));
			vmess("test_param=3: rho perturbation only (tests g_rho_full)");
		}

		vmess("Perturbation amplitudes: dcp_max=%.1f  dcs_max=%.1f  dro_max=%.1f",
			pert_pct * cp0, pert_pct * cs0, pert_pct * ro0);

		/* ---- Zero perturbation outside the gradient accumulation region ----
		 * The gradient is NOT accumulated in the taper-adjacent physical
		 * boundary strip (ntap points on each absorbing side, 1 point for
		 * free surface).  readModel also EXTENDS boundary values into the
		 * taper zone, so any perturbation at the physical edge creates
		 * Born scattering in the taper that the gradient cannot capture.
		 * Zeroing the perturbation here ensures domain consistency. */
		{
			int margin_lef = 0, margin_rig = 0, margin_top = 0, margin_bot = 0;
			int half_fd = mod.iorder / 2;

			/* Absorbing taper margins */
			if (bnd.lef == 4 || bnd.lef == 2) margin_lef = bnd.ntap;
			if (bnd.rig == 4 || bnd.rig == 2) margin_rig = bnd.ntap;
			if (bnd.top == 4 || bnd.top == 2) margin_top = bnd.ntap;
			if (bnd.bot == 4 || bnd.bot == 2) margin_bot = bnd.ntap;

			/* Free surface margin (skip 1 point) */
			if (bnd.top == 1 && margin_top < 1) margin_top = 1;

			/* FD stencil safety (half points for spatial derivatives) */
			if (margin_lef < half_fd) margin_lef = half_fd;
			if (margin_rig < half_fd) margin_rig = half_fd;
			if (margin_top < half_fd) margin_top = half_fd;
			if (margin_bot < half_fd) margin_bot = half_fd;

			for (ix = 0; ix < mod.nx; ix++) {
				for (iz = 0; iz < mod.nz; iz++) {
					if (ix < margin_lef || ix >= mod.nx - margin_rig ||
					    iz < margin_top  || iz >= mod.nz - margin_bot) {
						int idx2 = ix * mod.nz + iz;
						delta_cp[idx2] = 0.0f;
						delta_cs[idx2] = 0.0f;
						delta_ro[idx2] = 0.0f;
					}
				}
			}
			vmess("Zeroed perturbation in boundary margin (l=%d r=%d t=%d b=%d)",
				margin_lef, margin_rig, margin_top, margin_bot);
			vmess("Active perturbation region: %d x %d of %d x %d physical grid",
				mod.nx - margin_lef - margin_rig,
				mod.nz - margin_top - margin_bot,
				mod.nx, mod.nz);
		}

		/* ---- Zero perturbation near source injection point ----
		 * applySource uses material parameters (rox/roz, l2m) at
		 * the source grid point, so model perturbation there changes
		 * the effective source amplitude.  The gradient does not
		 * include this source sensitivity, so zero δm near the
		 * source to prevent this term from contaminating the test. */
		{
			int src_margin = 3;  /* covers stagger averaging */
			int iix, iiz;
			for (iix = ixsrc - src_margin; iix <= ixsrc + src_margin; iix++) {
				for (iiz = izsrc - src_margin; iiz <= izsrc + src_margin; iiz++) {
					if (iix >= 0 && iix < mod.nx && iiz >= 0 && iiz < mod.nz) {
						int idx2 = iix * mod.nz + iiz;
						delta_cp[idx2] = 0.0f;
						delta_cs[idx2] = 0.0f;
						delta_ro[idx2] = 0.0f;
					}
				}
			}
			vmess("Zeroed perturbation near source (ix=%d, iz=%d) +/- %d",
				ixsrc, izsrc, src_margin);
		}
	}

	/* ============================================================ */
	/* Step 2a: Forward model with base model m0 -> d0              */
	/* ============================================================ */
	vmess("--- Step 2a: Forward model (base) ---");
	{
		checkpointPar chk_base_fwd;
		initCheckpoints(&chk_base_fwd, &mod, chk_skipdt, 0, chk_base);

		fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna,
			ixsrc, izsrc, src_nwav, 0, 1, fileno, &chk_base_fwd, verbose);

		/* Save checkpoint for adjoint step later - copy chk metadata */
		memcpy(&chk, &chk_base_fwd, sizeof(checkpointPar));
	}

	vmess("Base forward complete. Reading synthetic data...");

	/* Read base synthetic data */
	{
		char syn_fname[1024];
		snprintf(syn_fname, sizeof(syn_fname), "%s_000%s.su", rec.file_rcv, comp_str);
		ntr_d0 = readSUtraces(syn_fname, &d0_data, &d0_hdrs, &ns_d0, &dt_us_d0);
		if (ntr_d0 <= 0)
			verr("Cannot read base synthetic data from %s", syn_fname);
		vmess("Read %d traces (%d samples) from %s", ntr_d0, ns_d0, syn_fname);
	}

	/* ============================================================ */
	/* Step 2b: Write perturbed model and forward model -> d1       */
	/* ============================================================ */
	vmess("--- Step 2b: Forward model (perturbed) ---");

	{
		/* Write perturbed model files */
		float *cp_pert = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
		float *cs_pert = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
		float *ro_pert = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
		float *cp_base = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
		float *cs_base = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
		float *ro_base = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));

		/* Extract base model from padded arrays */
		for (ix = 0; ix < mod.nx; ix++) {
			for (iz = 0; iz < mod.nz; iz++) {
				int ig = (ix + ioPx) * mod.naz + iz + ioPz;
				int idx = ix * mod.nz + iz;

				if (mod.cp && mod.cs && mod.rho) {
					cp_base[idx] = mod.cp[ig];
					cs_base[idx] = mod.cs[ig];
					ro_base[idx] = mod.rho[ig];
				} else {
					float rho_val = 1.0f / mod.rox[ig];
					cp_base[idx] = sqrtf(mod.l2m[ig] / rho_val);
					cs_base[idx] = sqrtf(mod.muu[ig] / rho_val);
					ro_base[idx] = rho_val;
				}

				cp_pert[idx] = cp_base[idx] + delta_cp[idx];
				cs_pert[idx] = cs_base[idx] + delta_cs[idx];
				ro_pert[idx] = ro_base[idx] + delta_ro[idx];
			}
		}

		/* Diagnostics: verify perturbation amplitudes */
		{
			float max_dcp = 0.0f, max_dcs = 0.0f, max_dro = 0.0f;
			float max_cp = 0.0f, max_cs = 0.0f, max_ro = 0.0f;
			for (ix = 0; ix < mod.nx * mod.nz; ix++) {
				float acp = fabsf(cp_pert[ix] - cp_base[ix]);
				float acs = fabsf(cs_pert[ix] - cs_base[ix]);
				float aro = fabsf(ro_pert[ix] - ro_base[ix]);
				if (acp > max_dcp) max_dcp = acp;
				if (acs > max_dcs) max_dcs = acs;
				if (aro > max_dro) max_dro = aro;
				if (cp_pert[ix] > max_cp) max_cp = cp_pert[ix];
				if (cs_pert[ix] > max_cs) max_cs = cs_pert[ix];
				if (ro_pert[ix] > max_ro) max_ro = ro_pert[ix];
			}
			vmess("DIAG pert model: max|dcp|=%.2f max|dcs|=%.2f max|dro|=%.2f",
				max_dcp, max_dcs, max_dro);
			vmess("DIAG pert model: max(cp)=%.1f max(cs)=%.1f max(ro)=%.1f",
				max_cp, max_cs, max_ro);
		}

		/* Write perturbed model to SU files (temporary) */
		writesufile("cp_pert.su", cp_pert, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		writesufile("cs_pert.su", cs_pert, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		writesufile("ro_pert.su", ro_pert, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);

		free(cp_pert);
		free(cs_pert);
		free(ro_pert);
		free(cp_base);
		free(cs_base);
		free(ro_base);
	}

	/* Load perturbed model and forward model */
	{
		/* Save original model file pointers */
		char *orig_cp = mod.file_cp;
		char *orig_cs = mod.file_cs;
		char *orig_ro = mod.file_ro;
		char *orig_rcv = rec.file_rcv;

		/* Point to perturbed files */
		mod.file_cp = "cp_pert.su";
		mod.file_cs = "cs_pert.su";
		mod.file_ro = "ro_pert.su";
		rec.file_rcv = "shot_pert";

		/* Re-read model with perturbed values */
		readModel(&mod, &bnd);

		/* Diagnostics: verify readModel loaded perturbed values */
		{
			float max_l2m = 0.0f, min_l2m = 1e30f;
			int ig;
			for (ix = 0; ix < mod.nx; ix++) {
				for (iz = 0; iz < mod.nz; iz++) {
					ig = (ix + ioPx) * mod.naz + iz + ioPz;
					if (mod.l2m[ig] > max_l2m) max_l2m = mod.l2m[ig];
					if (mod.l2m[ig] < min_l2m) min_l2m = mod.l2m[ig];
				}
			}
			vmess("DIAG readModel pert: l2m range [%.6e, %.6e] (expect variation)",
				min_l2m, max_l2m);
			vmess("DIAG readModel pert: cp range [%.1f, %.1f]",
				mod.cp_min, mod.cp_max);
		}

		/* Need to redo source setup after model change */
		freeStoreSourceOnSurface();
		allocStoreSourceOnSurface(src);

		{
			checkpointPar chk_pert;
			initCheckpoints(&chk_pert, &mod, chk_skipdt, 0, "chk_pert");

			fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna,
				ixsrc, izsrc, src_nwav, 0, 1, fileno, &chk_pert, verbose);

			cleanCheckpoints(&chk_pert);
		}

		vmess("Perturbed forward complete.");

		/* Read perturbed synthetic data */
		{
			char syn_fname[1024];
			snprintf(syn_fname, sizeof(syn_fname), "shot_pert_000%s.su", comp_str);
			ntr_d1 = readSUtraces(syn_fname, &d1_data, NULL, &ns_d1, &dt_us_d1);
			if (ntr_d1 <= 0)
				verr("Cannot read perturbed synthetic data from %s", syn_fname);
			if (ntr_d1 != ntr_d0 || ns_d1 != ns_d0)
				verr("Trace count/ns mismatch: base(%d,%d) vs pert(%d,%d)",
					ntr_d0, ns_d0, ntr_d1, ns_d1);
			vmess("Read %d traces (%d samples) from %s", ntr_d1, ns_d1, syn_fname);
		}

		/* Restore original model */
		mod.file_cp = orig_cp;
		mod.file_cs = orig_cs;
		mod.file_ro = orig_ro;
		rec.file_rcv = orig_rcv;

		readModel(&mod, &bnd);
		freeStoreSourceOnSurface();
		allocStoreSourceOnSurface(src);
	}

	/* ============================================================ */
	/* Step 2c: Generate random data y and compute y^T(Ax)          */
	/* ============================================================ */
	vmess("--- Step 2c: Compute y^T(Ax) ---");

	dt_sec = (float)dt_us_d0 * 1.0e-6f;

	/* Generate random data vector y */
	y_data = (float *)malloc((size_t)ntr_d0 * ns_d0 * sizeof(float));
	srand48(seed + 99999);  /* Different seed from model perturbation */
	for (i = 0; i < ntr_d0 * ns_d0; i++)
		y_data[i] = (float)(drand48() * 2.0 - 1.0);

	/* Diagnostics: check data amplitudes */
	{
		double max_d0 = 0.0, max_d1 = 0.0, max_diff = 0.0;
		for (i = 0; i < ntr_d0 * ns_d0; i++) {
			double ad0 = fabs((double)d0_data[i]);
			double ad1 = fabs((double)d1_data[i]);
			double adf = fabs((double)(d1_data[i] - d0_data[i]));
			if (ad0 > max_d0) max_d0 = ad0;
			if (ad1 > max_d1) max_d1 = ad1;
			if (adf > max_diff) max_diff = adf;
		}
		vmess("DIAG: max|d0| = %.6e  max|d1| = %.6e  max|d1-d0| = %.6e",
			max_d0, max_d1, max_diff);
	}

	/* Compute forward dot product: y^T(Ax) = sum y * (d1 - d0)
	 * Pure algebraic dot product (no dt weight). */
	forward_dot = 0.0;
	for (i = 0; i < ntr_d0 * ns_d0; i++)
		forward_dot += (double)y_data[i] * (double)(d1_data[i] - d0_data[i]);

	vmess("y^T A x (pure) = %.10e", forward_dot);
	vmess("y^T A x (with dt_rcv=%.6f) = %.10e", dt_sec, forward_dot * dt_sec);

	/* ============================================================ */
	/* Step 3: Adjoint side -- compute x^T(A^T y)                   */
	/* ============================================================ */
	vmess("--- Step 3: Compute x^T(A^T y) via adj_shot ---");

	/* Write random y as a "residual" SU file.
	 * Copy headers from base synthetic (correct gx, gelev, ns, dt),
	 * replace data with random y, set appropriate TRID. */
	{
		int trid;
		/* Determine TRID from component suffix */
		if (strcmp(comp_str, "_rvz") == 0)       trid = 7;
		else if (strcmp(comp_str, "_rvx") == 0)  trid = 6;
		else if (strcmp(comp_str, "_rp") == 0)   trid = 1;
		else if (strcmp(comp_str, "_rtzz") == 0) trid = 3;
		else if (strcmp(comp_str, "_rtxx") == 0) trid = 4;
		else if (strcmp(comp_str, "_rtxz") == 0) trid = 2;
		else trid = 7;  /* default Vz */

		/* Set TRID in headers */
		for (i = 0; i < ntr_d0; i++)
			d0_hdrs[i].trid = (short)trid;

		writeSUtraces("residual_dp.su", y_data, d0_hdrs, ntr_d0, ns_d0);
		vmess("Wrote random adjoint source (%d traces, trid=%d) to residual_dp.su",
			ntr_d0, trid);
	}

	/* Read residual into adjSrcPar */
	memset(&adj, 0, sizeof(adjSrcPar));
	readResidual("residual_dp.su", &adj, &mod, &bnd);
	vmess("Loaded %d adjoint source traces, nt=%d", adj.nsrc, adj.nt);

	/* Allocate gradient arrays (velocity parameterization via param=2) */
	grad_vp  = (float *)calloc(sizem, sizeof(float));
	grad_vs  = (float *)calloc(sizem, sizeof(float));
	grad_rho = (float *)calloc(sizem, sizeof(float));

	/* Run adjoint backpropagation with param=1 (Lamé) first for diagnostics,
	 * then convert to velocity space manually. */
	{
		snaPar sna_adj;
		if (sna.nsnap > 0) {
			sna_adj = sna;
			sna_adj.file_snap = "adj_snap";
			sna_adj.type.vz  = 1;
			sna_adj.type.vx  = 1;
			sna_adj.type.txx = 1;
			sna_adj.type.tzz = 1;
			sna_adj.type.txz = 1;
			sna_adj.type.p   = 0;
			vmess("Writing adjoint snapshots: %d snaps, dt_snap=%d steps",
				sna_adj.nsnap, sna_adj.skipdt);
		} else {
			memset(&sna_adj, 0, sizeof(snaPar));
		}

		/* param=1: returns (g_lambda, g_mu, g_rho_direct) */
		adj_shot(&mod, &src, &wav, &bnd, &rec, &adj,
			ixsrc, izsrc, src_nwav, &chk, &sna_adj,
			grad_vp, grad_vs, grad_rho, 1, verbose);
	}

	vmess("Adjoint backpropagation complete (Lamé space).");

	/* --- Lamé-space dot product diagnostic ---
	 * grad_vp=g_λ, grad_vs=g_μ, grad_rho=g_ρ_direct (from param=1).
	 *
	 * In Lame mode: delta_cp=δλ, delta_cs=δμ, delta_ro=δρ (direct).
	 * In velocity mode: transform (δVp,δVs,δρ) → (δλ,δμ,δρ) via chain rule:
	 *   δλ = 2ρVp·δVp + (-4ρVs)·δVs + (Vp²-2Vs²)·δρ
	 *   δμ = 2ρVs·δVs + Vs²·δρ
	 *   δρ_direct = δρ */
	{
		double dot_lam = 0.0, dot_muu = 0.0, dot_rho_d = 0.0;
		for (ix = 0; ix < mod.nx; ix++) {
			for (iz = 0; iz < mod.nz; iz++) {
				int idx = ix * mod.nz + iz;
				int ig  = (ix + ioPx) * mod.naz + iz + ioPz;
				double dlam, dmuu, drho;

				/* Velocity → Lamé perturbation */
				float rho_val = mod.rho[ig];
				float vp = mod.cp[ig], vs = mod.cs[ig];
				float vp2 = vp*vp, vs2 = vs*vs;
				dlam = 2.0*rho_val*vp*delta_cp[idx]
				     + (-4.0*rho_val*vs)*delta_cs[idx]
				     + (vp2 - 2.0*vs2)*delta_ro[idx];
				dmuu = 2.0*rho_val*vs*delta_cs[idx]
				     + vs2*delta_ro[idx];
				drho = delta_ro[idx];
				dot_lam  += dlam * (double)grad_vp[ig];
				dot_muu  += dmuu * (double)grad_vs[ig];
				dot_rho_d += drho * (double)grad_rho[ig];
			}
		}
		vmess("DIAG Lame decomposition: dlam*g_lam=%.6e  dmuu*g_muu=%.6e  drho*g_rho=%.6e",
			dot_lam, dot_muu, dot_rho_d);
		vmess("DIAG Lame total: %.6e", dot_lam + dot_muu + dot_rho_d);
	}

	/* Convert to velocity space for the standard dot product */
	convertGradientToVelocity(grad_vp, grad_vs, grad_rho,
	                          mod.cp, mod.cs, mod.rho, sizem);

	/* ============================================================ */
	/* Step 3b: Model-space dot product in velocity space           */
	/*                                                               */
	/* adj_shot(param=2) returns gradients in velocity space:        */
	/*   grad_vp  = g_Vp   (sensitivity to Vp)                      */
	/*   grad_vs  = g_Vs   (sensitivity to Vs)                      */
	/*   grad_rho = g_rho_full (includes chain rule contributions)   */
	/*                                                               */
	/* Simple direct dot: x^T(A^T y) = <δVp, g_Vp> + <δVs, g_Vs>   */
	/*                                + <δρ, g_ρ_full>               */
	/* ============================================================ */
	vmess("--- Step 3b: Compute model-space dot product (velocity space) ---");

	adjoint_dot = 0.0;
	{
		double dot_p1 = 0.0, dot_p2 = 0.0, dot_rho = 0.0;

		for (ix = 0; ix < mod.nx; ix++) {
			for (iz = 0; iz < mod.nz; iz++) {
				int idx = ix * mod.nz + iz;
				int ig  = (ix + ioPx) * mod.naz + iz + ioPz;
				dot_p1  += (double)delta_cp[idx] * (double)grad_vp[ig];
				dot_p2  += (double)delta_cs[idx] * (double)grad_vs[ig];
				dot_rho += (double)delta_ro[idx] * (double)grad_rho[ig];
			}
		}
		adjoint_dot = dot_p1 + dot_p2 + dot_rho;
		vmess("DIAG velocity decomposition: vp=%.6e  vs=%.6e  rho=%.6e",
			dot_p1, dot_p2, dot_rho);
		vmess("DIAG per-component ratio: vp=%.6f  vs=%.6f  rho=%.6f",
			(fabs(forward_dot) > 0.0 && fabs(dot_p1) > 0.0) ? dot_p1/forward_dot : 0.0,
			(fabs(forward_dot) > 0.0 && fabs(dot_p2) > 0.0) ? dot_p2/forward_dot : 0.0,
			(fabs(forward_dot) > 0.0 && fabs(dot_rho) > 0.0) ? dot_rho/forward_dot : 0.0);
	}

	vmess("x^T A^T y = %.10e", adjoint_dot);

	/* ============================================================ */
	/* Step 4: Evaluate                                              */
	/* ============================================================ */
	printf("\n--- Dot Product Test Results ---\n");
	printf("  y^T A x    = %+.10e\n", forward_dot);
	printf("  x^T A^T y  = %+.10e\n", adjoint_dot);

	if (fabs(forward_dot) > 0.0) {
		ratio = adjoint_dot / forward_dot;
		epsilon = fabs(ratio - 1.0);
	} else {
		ratio = 0.0;
		epsilon = fabs(adjoint_dot);
	}

	printf("  ratio      = %.8f\n", ratio);
	printf("  epsilon    = %.2e\n", epsilon);
	printf("  %s (epsilon %s 1e-02)\n",
		epsilon < 0.01 ? "PASS" : "FAIL",
		epsilon < 0.01 ? "<" : ">=");

	t1 = wallclock_time();
	printf("\nTotal wall time: %.2f s\n", t1 - t0);

	/* ============================================================ */
	/* Cleanup                                                       */
	/* ============================================================ */
	cleanCheckpoints(&chk);
	freeResidual(&adj);

	free(grad_vp);
	free(grad_vs);
	free(grad_rho);
	free(delta_cp);
	free(delta_cs);
	free(delta_ro);
	free(d0_data);
	free(d0_hdrs);
	free(d1_data);
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

	printf("\n=== All tests completed ===\n");

	return (sbp_pass && epsilon < 0.01) ? 0 : 1;
}
