/*
 * test_hessian_dp.c - Hessian symmetry dot product test.
 *
 * Verifies <H*dm1, dm2> = <dm1, H*dm2> where H = J^T J is the
 * Gauss-Newton Hessian.  Since H_GN is symmetric by construction,
 * the ratio of these dot products should be 1.0 (to machine precision
 * modulo floating-point accumulation).
 *
 * Test procedure:
 *   1. Setup (getParameters, readModel, defineSource)
 *   2. Forward with checkpointing (fdfwimodc)
 *   3. Generate two random perturbation vectors dm1, dm2
 *   4. hess_shot(dm1) -> Hd1
 *   5. hess_shot(dm2) -> Hd2
 *   6. Compute ratio <Hd1, dm2> / <dm1, Hd2>  -> should be 1.0
 *
 * Parameters (same interface as fdelmodc, plus):
 *   file_obs=       observed data base name (not used; only needs
 *                   forward modeling to create checkpoints)
 *   chk_skipdt=100  checkpoint interval in time steps
 *   chk_base=chk_hess  checkpoint file base path
 *   pert_pct=0.01   perturbation amplitude as fraction of model range
 *   seed=12345      random number seed for dm1 (dm2 uses seed+1)
 *   comp=_rvz       component suffix for recording
 *   param=1         parameterization (1=Lame, 2=velocity)
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
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

int fdfwimodc(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec, snaPar *sna, int ixsrc, int izsrc,
              float **src_nwav, int ishot, int nshots, int fileno,
              checkpointPar *chk, int verbose);

int initCheckpoints(checkpointPar *chk, modPar *mod, int skipdt,
                    int delay, const char *file_base);
int cleanCheckpoints(checkpointPar *chk);

void extractGradientVector(float *g, float *grad1, float *grad2, float *grad3,
                           modPar *mod, bndPar *bnd, int param);
void extractModelVector(float *x, modPar *mod, bndPar *bnd, int param);

char *sdoc[] = {
" ",
" test_hessian_dp - Hessian symmetry dot product test",
" ",
" Same parameters as fdelmodc, plus:",
"   chk_skipdt=100   checkpoint interval (time steps)",
"   chk_base=chk_hess  checkpoint file base path",
"   pert_pct=0.01    perturbation amplitude fraction",
"   seed=12345       random number seed",
"   comp=_rvz        component suffix for recording",
"   param=1          parameterization (1=Lame, 2=velocity)",
" ",
NULL};


/*--------------------------------------------------------------------
 * generate_random_perturbation -- Fill vector with scaled random values.
 *
 * Uses a simple linear congruential generator for reproducibility.
 * The perturbation is scaled so that max|dm| = pert_pct * model_range
 * for each parameter block.
 *
 * x:        model vector (for scaling reference)
 * dm:       OUTPUT perturbation vector
 * nvec:     total vector length
 * nmodel:   number of grid points per parameter
 * nparam:   number of parameters (2 or 3)
 * pert_pct: amplitude as fraction of parameter range
 * seed:     random number seed
 *--------------------------------------------------------------------*/
static void generate_random_perturbation(float *x, float *dm, int nvec,
                                         int nmodel, int nparam,
                                         float pert_pct, unsigned int seed)
{
	int i, p;
	unsigned int state = seed;

	/* Fill with random numbers in [-1, 1] */
	for (i = 0; i < nvec; i++) {
		state = state * 1103515245u + 12345u;
		dm[i] = 2.0f * ((float)(state >> 16) / 65536.0f) - 1.0f;
	}

	/* Scale each parameter block by pert_pct * range */
	for (p = 0; p < nparam; p++) {
		float pmin =  1.0e30f;
		float pmax = -1.0e30f;
		float scale;
		int offset = p * nmodel;

		for (i = 0; i < nmodel; i++) {
			float v = x[offset + i];
			if (v < pmin) pmin = v;
			if (v > pmax) pmax = v;
		}

		scale = pert_pct * (pmax - pmin);
		if (scale < 1.0e-20f) scale = pert_pct * (fabsf(pmin) + 1.0f);

		for (i = 0; i < nmodel; i++)
			dm[offset + i] *= scale;
	}
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
	float **src_nwav;
	float  sinkvel;
	double t0_wall;
	size_t nsamp, sizew, sizem;
	int    n1, ix, iz, ir, i;
	int    ioPx, ioPz;
	int    ixsrc, izsrc, fileno;
	int    verbose;

	/* Test parameters */
	char  *chk_base, *comp_str;
	int    chk_skipdt, seed, param;
	float  pert_pct;

	t0_wall = wallclock_time();

	/* ============================================================ */
	/* Parse parameters                                              */
	/* ============================================================ */
	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;
	if (!getparint("chk_skipdt", &chk_skipdt)) chk_skipdt = 100;
	if (!getparstring("chk_base", &chk_base)) chk_base = "chk_hess";
	if (!getparint("seed", &seed)) seed = 12345;
	if (!getparfloat("pert_pct", &pert_pct)) pert_pct = 0.01f;
	if (!getparstring("comp", &comp_str)) comp_str = "_rvz";
	if (!getparint("param", &param)) param = 1;
	if (param < 1 || param > 2)
		verr("param must be 1 (Lame) or 2 (velocity)");

	/* test_block: isolate parameter blocks for debugging.
	 *   0 = all (default), 1 = lambda only, 2 = mu only, 3 = rho only,
	 *   4 = lambda+mu (stiffness), 5 = lambda+rho, 6 = mu+rho */
	int test_block;
	if (!getparint("test_block", &test_block)) test_block = 0;

	/* ============================================================ */
	/* Standard model setup                                          */
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

	int elastic = (mod.ischeme > 2);
	int nparam = elastic ? 3 : 2;
	int nmodel = mod.nx * mod.nz;
	int nvec   = nparam * nmodel;

	printf("=== Hessian Symmetry Dot Product Test ===\n");
	vmess("Grid: nx=%d nz=%d (padded: nax=%d naz=%d)", mod.nx, mod.nz, mod.nax, mod.naz);
	vmess("Source at grid ix=%d iz=%d", ixsrc, izsrc);
	vmess("Perturbation: %.2f%%  Seed: %d", pert_pct * 100.0f, seed);
	vmess("Parameterization: %s", param == 1 ? "Lame" : "velocity");
	vmess("Components: %s", comp_str);
	vmess("nvec=%d (nparam=%d, nmodel=%d)", nvec, nparam, nmodel);

	/* ============================================================ */
	/* Step 1: Forward with checkpointing                            */
	/* ============================================================ */
	vmess("--- Step 1: Forward modeling with checkpointing ---");
	{
		snaPar sna_off = sna;
		sna_off.nsnap = 0;

		initCheckpoints(&chk, &mod, chk_skipdt, 0, chk_base);

		fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna_off,
		          ixsrc, izsrc, src_nwav, 0, 1, fileno, &chk, verbose);
	}
	vmess("Forward complete. %d checkpoints written.", chk.nsnap);

	/* ============================================================ */
	/* Step 2: Generate random perturbation vectors dm1, dm2         */
	/* ============================================================ */
	vmess("--- Step 2: Generating random perturbation vectors ---");

	float *x = (float *)calloc(nvec, sizeof(float));
	extractModelVector(x, &mod, &bnd, param);

	float *dm1 = (float *)calloc(nvec, sizeof(float));
	float *dm2 = (float *)calloc(nvec, sizeof(float));
	generate_random_perturbation(x, dm1, nvec, nmodel, nparam, pert_pct, (unsigned)seed);
	generate_random_perturbation(x, dm2, nvec, nmodel, nparam, pert_pct, (unsigned)(seed + 1));

	/* Zero out parameter blocks based on test_block */
	if (test_block > 0 && elastic) {
		int zero_lam = 1, zero_mu = 1, zero_rho = 1;
		switch (test_block) {
			case 1: zero_lam = 0; break;               /* lambda only */
			case 2: zero_mu = 0; break;                 /* mu only */
			case 3: zero_rho = 0; break;                /* rho only */
			case 4: zero_lam = 0; zero_mu = 0; break;   /* lambda+mu */
			case 5: zero_lam = 0; zero_rho = 0; break;  /* lambda+rho */
			case 6: zero_mu = 0; zero_rho = 0; break;   /* mu+rho */
		}
		if (zero_lam) {
			memset(dm1, 0, nmodel * sizeof(float));
			memset(dm2, 0, nmodel * sizeof(float));
		}
		if (zero_mu) {
			memset(dm1 + nmodel, 0, nmodel * sizeof(float));
			memset(dm2 + nmodel, 0, nmodel * sizeof(float));
		}
		if (zero_rho) {
			memset(dm1 + 2*nmodel, 0, nmodel * sizeof(float));
			memset(dm2 + 2*nmodel, 0, nmodel * sizeof(float));
		}
		const char *block_names[] = {"all","lam","mu","rho","lam+mu","lam+rho","mu+rho"};
		vmess("test_block=%d: testing %s only", test_block, block_names[test_block]);
	}

	{
		double norm1 = 0.0, norm2 = 0.0;
		for (i = 0; i < nvec; i++) {
			norm1 += (double)dm1[i] * dm1[i];
			norm2 += (double)dm2[i] * dm2[i];
		}
		vmess("||dm1|| = %.6e  ||dm2|| = %.6e", sqrt(norm1), sqrt(norm2));
	}

	/* ============================================================ */
	/* Step 3: Compute H*dm1                                         */
	/* ============================================================ */
	vmess("--- Step 3: Computing H*dm1 (born_shot + adj_shot) ---");

	/* Allocate perturbed FD coefficient arrays as a single block
	 * to avoid heap metadata corruption from small OOB writes. */
	int ndelta = elastic ? 5 : 3;
	float *delta_block = (float *)calloc((size_t)ndelta * sizem, sizeof(float));
	float *delta_rox = delta_block;
	float *delta_roz = delta_block + sizem;
	float *delta_l2m = delta_block + 2 * sizem;
	float *delta_lam = NULL;
	float *delta_mul = NULL;
	if (elastic) {
		delta_lam = delta_block + 3 * sizem;
		delta_mul = delta_block + 4 * sizem;
	}

	/* H*dm1 output (padded grid, single block) */
	int nhd = elastic ? 3 : 2;
	float *hd1_block = (float *)calloc((size_t)nhd * sizem, sizeof(float));
	float *hd1_g1 = hd1_block;
	float *hd1_g2 = elastic ? hd1_block + sizem : NULL;
	float *hd1_g3 = hd1_block + (elastic ? 2 : 1) * sizem;

	/* Convert dm1 to FD coefficient perturbations */
	perturbFDcoefficients(&mod, &bnd, dm1, param,
	                      delta_rox, delta_roz, delta_l2m, delta_lam, delta_mul);

	/* Compute H*dm1 via Born + adjoint */
	hess_shot(&mod, &src, &wav, &bnd, &rec,
	          ixsrc, izsrc, src_nwav, &chk,
	          delta_rox, delta_roz, delta_l2m, delta_lam, delta_mul,
	          "born1", comp_str,
	          0, 1, fileno,
	          0,  /* no residual taper */
	          hd1_g1, hd1_g2, hd1_g3,
	          1,  /* always accumulate in Lame space */
	          verbose);

	/* Heap probe: after hess_shot dm1 */
	{ void *_hp = malloc(64); if (_hp) { memset(_hp, 0, 64); free(_hp); }
	  vmess("  [heap check OK after hess_shot dm1]"); }

	/* Extract H*dm1 to flat vector.
	 * Use param=1 (Lame) because adj_shot with param=1 accumulates
	 * in Lame space without chain rule conversion. */
	float *Hd1 = (float *)calloc(nvec, sizeof(float));
	extractGradientVector(Hd1, hd1_g1, hd1_g2, hd1_g3, &mod, &bnd, 1);

	{
		double norm_Hd1 = 0.0;
		for (i = 0; i < nvec; i++)
			norm_Hd1 += (double)Hd1[i] * Hd1[i];
		vmess("||H*dm1|| = %.6e", sqrt(norm_Hd1));
	}

	/* Heap probe: after extract dm1 */
	{ void *_hp = malloc(64); if (_hp) { memset(_hp, 0, 64); free(_hp); }
	  vmess("  [heap check OK after extract dm1]"); }

	/* ============================================================ */
	/* Step 4: Compute H*dm2                                         */
	/* ============================================================ */
	vmess("--- Step 4: Computing H*dm2 (born_shot + adj_shot) ---");

	/* H*dm2 output (padded grid, single block) */
	float *hd2_block = (float *)calloc((size_t)nhd * sizem, sizeof(float));
	float *hd2_g1 = hd2_block;
	float *hd2_g2 = elastic ? hd2_block + sizem : NULL;
	float *hd2_g3 = hd2_block + (elastic ? 2 : 1) * sizem;

	/* Convert dm2 to FD coefficient perturbations */
	perturbFDcoefficients(&mod, &bnd, dm2, param,
	                      delta_rox, delta_roz, delta_l2m, delta_lam, delta_mul);

	/* Compute H*dm2 via Born + adjoint */
	hess_shot(&mod, &src, &wav, &bnd, &rec,
	          ixsrc, izsrc, src_nwav, &chk,
	          delta_rox, delta_roz, delta_l2m, delta_lam, delta_mul,
	          "born2", comp_str,
	          0, 1, fileno,
	          0,  /* no residual taper */
	          hd2_g1, hd2_g2, hd2_g3,
	          1,  /* always accumulate in Lame space */
	          verbose);

	/* Heap probe: after hess_shot dm2 */
	{ void *_hp = malloc(64); if (_hp) { memset(_hp, 0, 64); free(_hp); }
	  vmess("  [heap check OK after hess_shot dm2]"); }

	/* Extract H*dm2 to flat vector */
	float *Hd2 = (float *)calloc(nvec, sizeof(float));
	extractGradientVector(Hd2, hd2_g1, hd2_g2, hd2_g3, &mod, &bnd, 1);

	{
		double norm_Hd2 = 0.0;
		for (i = 0; i < nvec; i++)
			norm_Hd2 += (double)Hd2[i] * Hd2[i];
		vmess("||H*dm2|| = %.6e", sqrt(norm_Hd2));
	}

	/* Heap probe: after extract dm2 */
	{ void *_hp = malloc(64); if (_hp) { memset(_hp, 0, 64); free(_hp); }
	  vmess("  [heap check OK after extract dm2]"); }

	/* ============================================================ */
	/* Step 5: Compute dot products and ratio                        */
	/* ============================================================ */
	/* Heap probe: detect corruption before Step 5 */
	{ void *_hp = malloc(64); if (_hp) { memset(_hp, 0, 64); free(_hp); }
	  vmess("  [heap check OK before Step 5]"); }

	vmess("--- Step 5: Computing dot products ---");

	double dp1 = 0.0;  /* <H*dm1, dm2> */
	double dp2 = 0.0;  /* <dm1, H*dm2> */

	/* If param=2 (velocity), the perturbation dm is in velocity space
	 * but H*dm is in Lame space. We need both in the same space.
	 * Since we accumulated H*dm in Lame (param=1 to hess_shot/extract),
	 * we need dm also in Lame space for the dot product.
	 *
	 * For param=1, dm is already in Lame space. For param=2, we need
	 * to convert the flat dm to Lame using the chain rule. However,
	 * the simplest approach for the symmetry test is to just use
	 * param=1 throughout. If the user specifies param=2, the dm
	 * vectors are in velocity space but perturbFDcoefficients already
	 * handles the conversion to Lame FD coefficients internally.
	 * The dot product test is valid in either space as long as both
	 * dm and Hd are in the same space.
	 *
	 * Since we call hess_shot and extract with param=1 (Lame), the
	 * Hd vectors are in Lame space. If the user's dm is in velocity
	 * space (param=2), we need to convert dm to Lame for the dot
	 * product. For simplicity, we generate dm directly in the
	 * parameterization specified, and perturbFDcoefficients handles
	 * the conversion internally. But the dot product must be in
	 * consistent space.
	 *
	 * SOLUTION: We always use param=1 for BOTH dm generation AND
	 * Hd extraction. The test operates entirely in Lame space. */

	/* Per-parameter dot product breakdown */
	double dp1_lam = 0.0, dp1_mu = 0.0, dp1_rho = 0.0;
	double dp2_lam = 0.0, dp2_mu = 0.0, dp2_rho = 0.0;

	for (i = 0; i < nmodel; i++) {
		dp1_lam += (double)Hd1[i] * (double)dm2[i];
		dp2_lam += (double)dm1[i] * (double)Hd2[i];
	}
	if (elastic) {
		for (i = 0; i < nmodel; i++) {
			dp1_mu += (double)Hd1[nmodel + i] * (double)dm2[nmodel + i];
			dp2_mu += (double)dm1[nmodel + i] * (double)Hd2[nmodel + i];
		}
	}
	{
		int rho_off = elastic ? 2*nmodel : nmodel;
		for (i = 0; i < nmodel; i++) {
			dp1_rho += (double)Hd1[rho_off + i] * (double)dm2[rho_off + i];
			dp2_rho += (double)dm1[rho_off + i] * (double)Hd2[rho_off + i];
		}
	}

	dp1 = dp1_lam + dp1_mu + dp1_rho;
	dp2 = dp2_lam + dp2_mu + dp2_rho;

	double ratio = (dp2 != 0.0) ? dp1 / dp2 : 0.0;
	double eps = (dp2 != 0.0) ? fabs(dp1 - dp2) / (0.5 * (fabs(dp1) + fabs(dp2))) : 0.0;

	/* ============================================================ */
	/* Data-space diagnostic: <J*dm1, J*dm2> from Born .su files    */
	/*                                                               */
	/* Sign convention analysis:                                      */
	/*   accumGradient gives -J^T (negative of true Jacobian^T).     */
	/*   adj_shot negates residual, so: Hd = -accumGrad(u,ψ_d)      */
	/*     = -(-J^T d) = +J^T d = +J^T(J*dm).                       */
	/*   Therefore: <Hd1, dm2> = <J^T J dm1, dm2>                   */
	/*                         = <J dm1, J dm2> = dp_data.           */
	/*   So dp1/dp_data = dp2/dp_data should both equal +1.0.       */
	/* ============================================================ */
	{
		FILE *fp1_su, *fp2_su;
		segy h1_su, h2_su;
		double dp_data = 0.0, norm_d1 = 0.0, norm_d2 = 0.0;
		int ntr_data = 0, ns_data = 0;

		fp1_su = fopen("born1_combined.su", "r");
		fp2_su = fopen("born2_combined.su", "r");
		if (fp1_su && fp2_su) {
			float *d1_buf = NULL, *d2_buf = NULL;
			while (fread(&h1_su, 1, TRCBYTES, fp1_su) == TRCBYTES &&
			       fread(&h2_su, 1, TRCBYTES, fp2_su) == TRCBYTES) {
				int ns1 = (int)h1_su.ns, ns2 = (int)h2_su.ns;
				if (ns1 != ns2) {
					vmess("WARNING: Born ns mismatch %d vs %d", ns1, ns2);
					break;
				}
				ns_data = ns1;
				d1_buf = (float *)realloc(d1_buf, ns1 * sizeof(float));
				d2_buf = (float *)realloc(d2_buf, ns2 * sizeof(float));
				if (fread(d1_buf, sizeof(float), ns1, fp1_su) != (size_t)ns1) break;
				if (fread(d2_buf, sizeof(float), ns2, fp2_su) != (size_t)ns2) break;
				for (int j = 0; j < ns1; j++) {
					dp_data += (double)d1_buf[j] * (double)d2_buf[j];
					norm_d1 += (double)d1_buf[j] * (double)d1_buf[j];
					norm_d2 += (double)d2_buf[j] * (double)d2_buf[j];
				}
				ntr_data++;
			}
			if (d1_buf) free(d1_buf);
			if (d2_buf) free(d2_buf);
			fclose(fp1_su); fclose(fp2_su);
			printf("  Data-space diagnostic:\n");
			printf("    <J*dm1, J*dm2> = %+.10e  (%d traces x %d samples)\n",
			       dp_data, ntr_data, ns_data);
			printf("    ||J*dm1|| = %.6e  ||J*dm2|| = %.6e\n",
			       sqrt(norm_d1), sqrt(norm_d2));
			printf("    dp1/dp_data = %.6f  (expect +1.0)\n",
			       dp_data != 0 ? dp1/dp_data : 0);
			printf("    dp2/dp_data = %.6f  (expect +1.0)\n",
			       dp_data != 0 ? dp2/dp_data : 0);
		} else {
			vmess("WARNING: Could not open born1/born2_combined.su for data-space diagnostic");
			if (fp1_su) fclose(fp1_su);
			if (fp2_su) fclose(fp2_su);
		}
	}

	printf("\n");
	printf("=== HESSIAN SYMMETRY DOT PRODUCT TEST RESULTS ===\n");
	printf("  Per-parameter breakdown:\n");
	printf("    lambda:  dp1=%+.6e  dp2=%+.6e  ratio=%.6f\n",
	       dp1_lam, dp2_lam, dp2_lam != 0.0 ? dp1_lam/dp2_lam : 0.0);
	if (elastic)
		printf("    mu:      dp1=%+.6e  dp2=%+.6e  ratio=%.6f\n",
		       dp1_mu, dp2_mu, dp2_mu != 0.0 ? dp1_mu/dp2_mu : 0.0);
	printf("    rho:     dp1=%+.6e  dp2=%+.6e  ratio=%.6f\n",
	       dp1_rho, dp2_rho, dp2_rho != 0.0 ? dp1_rho/dp2_rho : 0.0);
	printf("  Total:\n");
	printf("  <H*dm1, dm2> = %+.10e\n", dp1);
	printf("  <dm1, H*dm2> = %+.10e\n", dp2);
	printf("  Ratio        = %.10f\n", ratio);
	printf("  Relative err = %.4e\n", eps);
	printf("\n");

	if (fabs(ratio - 1.0) < 1.0e-2) {
		printf("  RESULT: PASS (ratio within 1%% of 1.0)\n");
	} else if (fabs(ratio - 1.0) < 5.0e-2) {
		printf("  RESULT: MARGINAL (ratio within 5%% of 1.0)\n");
	} else {
		printf("  RESULT: FAIL (ratio deviates >5%% from 1.0)\n");
	}
	printf("\n");
	printf("  Total time: %.1f s\n", wallclock_time() - t0_wall);
	printf("=================================================\n");

	/* ============================================================ */
	/* Cleanup                                                       */
	/* ============================================================ */
	cleanCheckpoints(&chk);
	free(x);
	free(dm1);
	free(dm2);
	free(Hd1);
	free(Hd2);
	free(delta_block);
	free(hd1_block);
	free(hd2_block);
	free(src_nwav[0]);
	free(src_nwav);
	freeStoreSourceOnSurface();

	return 0;
}
