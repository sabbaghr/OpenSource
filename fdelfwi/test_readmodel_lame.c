/*
 * test_readmodel_lame.c - Verify readModel produces identical FD parameters
 * from (Vp, Vs, rho) and equivalent (lambda, mu, rho) inputs.
 *
 * Steps:
 *   1. Read model in velocity mode -> save reference FD arrays
 *   2. Compute lambda, mu from stored Vp, Vs, rho -> write SU files
 *   3. Re-read model with lame_input=1 -> compare FD arrays
 *
 * Usage (same model params as other tests):
 *   test_readmodel_lame file_cp=vp.su file_cs=vs.su file_den=ro.su \
 *       ischeme=3 iorder=4 file_src=wave.su ...
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
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);
int readModel(modPar *mod, bndPar *bnd);
int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

char *sdoc[] = {
" test_readmodel_lame - Verify readModel Lame input mode",
" ",
" Reads model in velocity mode, creates equivalent lambda/mu files,",
" re-reads in Lame mode, and compares all FD arrays.",
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
	int     verbose;
	size_t  sizem, i;
	int     ix, iz, n1, ioPx, ioPz;

	float *ref_rox, *ref_roz, *ref_l2m, *ref_lam, *ref_muu;
	float *ref_cp, *ref_cs, *ref_rho;

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;

	/* ============================================================ */
	/* Step 1: Read model in velocity mode                          */
	/* ============================================================ */
	printf("=== Test: readModel Lame input verification ===\n\n");
	printf("--- Step 1: Read model in velocity mode ---\n");

	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);
	mod.lame_input = 0;  /* Force velocity mode */
	readModel(&mod, &bnd);

	sizem = (size_t)mod.nax * mod.naz;
	n1 = mod.naz;

	/* Save reference FD arrays */
	ref_rox = (float *)malloc(sizem * sizeof(float));
	ref_roz = (float *)malloc(sizem * sizeof(float));
	ref_l2m = (float *)malloc(sizem * sizeof(float));
	ref_lam = (float *)malloc(sizem * sizeof(float));
	ref_muu = (float *)malloc(sizem * sizeof(float));
	ref_cp  = (float *)malloc(sizem * sizeof(float));
	ref_cs  = (float *)malloc(sizem * sizeof(float));
	ref_rho = (float *)malloc(sizem * sizeof(float));

	memcpy(ref_rox, mod.rox, sizem * sizeof(float));
	memcpy(ref_roz, mod.roz, sizem * sizeof(float));
	memcpy(ref_l2m, mod.l2m, sizem * sizeof(float));
	memcpy(ref_lam, mod.lam, sizem * sizeof(float));
	memcpy(ref_muu, mod.muu, sizem * sizeof(float));
	memcpy(ref_cp,  mod.cp,  sizem * sizeof(float));
	memcpy(ref_cs,  mod.cs,  sizem * sizeof(float));
	memcpy(ref_rho, mod.rho, sizem * sizeof(float));

	printf("  Grid: nax=%d naz=%d (sizem=%zu)\n", mod.nax, mod.naz, sizem);

	/* Print some reference values at a central point */
	ioPx = mod.ioPx;
	ioPz = mod.ioPz;
	if (bnd.lef == 4 || bnd.lef == 2) ioPx += bnd.ntap;
	if (bnd.top == 4 || bnd.top == 2) ioPz += bnd.ntap;

	{
		int ig = (mod.nx/2 + ioPx) * n1 + mod.nz/2 + ioPz;
		printf("  Reference at center: rox=%.6e roz=%.6e\n", ref_rox[ig], ref_roz[ig]);
		printf("  Reference at center: l2m=%.6e lam=%.6e muu=%.6e\n",
			ref_l2m[ig], ref_lam[ig], ref_muu[ig]);
		printf("  Reference at center: cp=%.1f cs=%.1f rho=%.1f\n",
			ref_cp[ig], ref_cs[ig], ref_rho[ig]);
	}

	/* ============================================================ */
	/* Step 2: Create Lame parameter SU files from loaded model     */
	/* ============================================================ */
	printf("\n--- Step 2: Create lambda, mu SU files ---\n");

	{
		float *lam_data = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));
		float *mu_data  = (float *)malloc((size_t)mod.nx * mod.nz * sizeof(float));

		for (ix = 0; ix < mod.nx; ix++) {
			for (iz = 0; iz < mod.nz; iz++) {
				int ig  = (ix + ioPx) * n1 + iz + ioPz;
				int idx = ix * mod.nz + iz;
				float vp  = mod.cp[ig];
				float vs  = mod.cs[ig];
				float rho = mod.rho[ig];
				lam_data[idx] = rho * (vp*vp - 2.0f*vs*vs);
				mu_data[idx]  = rho * vs * vs;
			}
		}

		printf("  lambda[center] = %.1f, mu[center] = %.1f\n",
			lam_data[(mod.nx/2)*mod.nz + mod.nz/2],
			mu_data[(mod.nx/2)*mod.nz + mod.nz/2]);
		printf("  Back-check: lambda + 2*mu = %.1f (should = rho*Vp^2 = %.1f)\n",
			lam_data[(mod.nx/2)*mod.nz + mod.nz/2] + 2.0f*mu_data[(mod.nx/2)*mod.nz + mod.nz/2],
			ref_rho[(mod.nx/2+ioPx)*n1+mod.nz/2+ioPz] *
			ref_cp[(mod.nx/2+ioPx)*n1+mod.nz/2+ioPz] *
			ref_cp[(mod.nx/2+ioPx)*n1+mod.nz/2+ioPz]);

		writesufile("_test_lam.su", lam_data, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		writesufile("_test_mu.su", mu_data, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);

		free(lam_data);
		free(mu_data);
	}

	/* ============================================================ */
	/* Step 3: Re-read model with lame_input=1                      */
	/* ============================================================ */
	printf("\n--- Step 3: Read model in Lame mode ---\n");

	mod.file_cp = "_test_lam.su";
	mod.file_cs = "_test_mu.su";
	/* file_ro stays the same (density unchanged) */
	mod.lame_input = 1;

	readModel(&mod, &bnd);

	/* Print Lame-mode values at same central point */
	{
		int ig = (mod.nx/2 + ioPx) * n1 + mod.nz/2 + ioPz;
		printf("  Lame mode at center: rox=%.6e roz=%.6e\n", mod.rox[ig], mod.roz[ig]);
		printf("  Lame mode at center: l2m=%.6e lam=%.6e muu=%.6e\n",
			mod.l2m[ig], mod.lam[ig], mod.muu[ig]);
		printf("  Lame mode at center: cp=%.1f cs=%.1f rho=%.1f\n",
			mod.cp[ig], mod.cs[ig], mod.rho[ig]);
	}

	/* ============================================================ */
	/* Step 4: Compare all FD arrays                                */
	/* ============================================================ */
	printf("\n--- Step 4: Compare FD parameters ---\n");

	{
		double max_err_rox = 0, max_err_roz = 0;
		double max_err_l2m = 0, max_err_lam = 0, max_err_muu = 0;
		double max_err_cp = 0, max_err_cs = 0, max_err_rho = 0;
		double max_ref_l2m = 0, max_ref_muu = 0, max_ref_rox = 0;
		double e;
		int all_pass;

		for (i = 0; i < sizem; i++) {
			e = fabs((double)mod.rox[i] - (double)ref_rox[i]);
			if (e > max_err_rox) max_err_rox = e;
			if (fabs(ref_rox[i]) > max_ref_rox) max_ref_rox = fabs(ref_rox[i]);

			e = fabs((double)mod.roz[i] - (double)ref_roz[i]);
			if (e > max_err_roz) max_err_roz = e;

			e = fabs((double)mod.l2m[i] - (double)ref_l2m[i]);
			if (e > max_err_l2m) max_err_l2m = e;
			if (fabs(ref_l2m[i]) > max_ref_l2m) max_ref_l2m = fabs(ref_l2m[i]);

			e = fabs((double)mod.lam[i] - (double)ref_lam[i]);
			if (e > max_err_lam) max_err_lam = e;

			e = fabs((double)mod.muu[i] - (double)ref_muu[i]);
			if (e > max_err_muu) max_err_muu = e;
			if (fabs(ref_muu[i]) > max_ref_muu) max_ref_muu = fabs(ref_muu[i]);

			e = fabs((double)mod.cp[i] - (double)ref_cp[i]);
			if (e > max_err_cp) max_err_cp = e;

			e = fabs((double)mod.cs[i] - (double)ref_cs[i]);
			if (e > max_err_cs) max_err_cs = e;

			e = fabs((double)mod.rho[i] - (double)ref_rho[i]);
			if (e > max_err_rho) max_err_rho = e;
		}

		printf("\n  Max absolute differences (velocity vs Lame input):\n");
		printf("    rox:  %.6e  (rel %.6e)\n", max_err_rox,
			max_ref_rox > 0 ? max_err_rox / max_ref_rox : 0);
		printf("    roz:  %.6e\n", max_err_roz);
		printf("    l2m:  %.6e  (rel %.6e)\n", max_err_l2m,
			max_ref_l2m > 0 ? max_err_l2m / max_ref_l2m : 0);
		printf("    lam:  %.6e\n", max_err_lam);
		printf("    muu:  %.6e  (rel %.6e)\n", max_err_muu,
			max_ref_muu > 0 ? max_err_muu / max_ref_muu : 0);
		printf("    cp:   %.6e\n", max_err_cp);
		printf("    cs:   %.6e\n", max_err_cs);
		printf("    rho:  %.6e\n", max_err_rho);

		/* For a homogeneous model, the Lame->velocity->FD path should
		 * be exact to single-precision float round-off (~1e-7 relative). */
		all_pass = 1;
		if (max_ref_l2m > 0 && max_err_l2m / max_ref_l2m > 1e-5) all_pass = 0;
		if (max_ref_muu > 0 && max_err_muu / max_ref_muu > 1e-5) all_pass = 0;
		if (max_ref_rox > 0 && max_err_rox / max_ref_rox > 1e-5) all_pass = 0;

		printf("\n  %s (relative errors < 1e-5)\n\n", all_pass ? "PASS" : "FAIL");

		/* Cleanup */
		free(ref_rox); free(ref_roz); free(ref_l2m);
		free(ref_lam); free(ref_muu);
		free(ref_cp); free(ref_cs); free(ref_rho);

		return all_pass ? 0 : 1;
	}
}
