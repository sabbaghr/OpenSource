/*
 * test_fs_col.c -- brute-force extraction of ONE column of F^T, to name the
 *                  exact entry where the free-surface adjoint is wrong.
 *
 * WHY
 * ---
 * demo/test_cpml_dp.sh case 6 (top=1, PML sides) fails at ~2e-3 even at
 * nstep=1, so it is a spatial-transpose gap.  The fs_proj/fs_side diagnostic
 * in test_cpml_dp.c narrowed it to the COLUMNS of the discrepancy indexed by
 * txx on the free-surface row: the code's adjoint returns the wrong vector
 * when fed a nonzero adj_txx there.  Reading the code found nothing -- the
 * scatter coefficients, the operation ordering and c1/c2 are all correct.
 *
 * So stop reading and measure.  For a single output slot
 *
 *     o = txx[ix0, izs]            (izs = free surface row)
 *
 * the true transpose column is, by definition,
 *
 *     (F^T e_o)_j = <F e_j, e_o> = (F e_j)_o
 *
 * i.e. run the FORWARD on a unit vector e_j and read the output at o.  The
 * code's answer is whatever one adjoint step returns when fed R = e_o.
 * Comparing the two entry by entry names the wrong entry, which a dot
 * product (a single scalar) can never do.
 *
 * Both operators are linear here: src.n = 0 and adj.nsrc = 0, exactly as in
 * test_cpml_dp.c.  Each forward probe is run at itime=0, which is also where
 * boundaries.c memsets the CFS-CPML psi, so every probe starts from psi = 0
 * and the probes are independent; cpmlStateZero() is called too, belt and
 * braces.  cpmlAdjReset() does the same for the adjoint psi.
 *
 * Parameters: the usual fdelfwi set, plus
 *   ix0=          column of the probed output slot (default: grid centre)
 *   halo=4        probe box half-width in x
 *   hz=3          probe box half-width in z
 *   tol=1e-6      relative tolerance for flagging an entry
 */

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"segy.h"
#include"fdelfwi.h"
#include"fdelfwi_offload.h"

double wallclock_time(void);

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);
int readModel(modPar *mod, bndPar *bnd);
int allocStoreSourceOnSurface(srcPar src);

int elastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime,
    int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz,
    float *tzz, float *txx, float *txz, float *rox, float *roz,
    float *l2m, float *lam, float *mul, int verbose);

int boundariesV_adj(modPar mod, bndPar bnd, float *vx, float *vz, float *tzz,
    float *txx, float *txz, float *rox, float *roz, float *l2m,
    float *lam, float *mul, int itime, int verbose);
void cpmlAdjGhostT(modPar mod, bndPar bnd, float *vx, float *vz);

int elastic4_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *rox, float *roz, float *l2m, float *lam, float *mul,
    int rec_delay, int rec_skipdt, int verbose);

char *sdoc[] = {
" ",
" test_fs_col - extract one column of F^T by brute force and compare it",
"               against the adjoint the code actually computes.",
" ",
" Pinpoints the entry where the free-surface transpose is wrong.",
" Run with left=2 right=2 top=1 bottom=2 (the failing case 6).",
" ",
" Extra parameters:",
"   ix0=          column of the probed output slot (default grid centre)",
"   halo=4        probe box half-width in x",
"   hz=3          probe box half-width in z",
"   tol=1e-6      relative tolerance for flagging an entry",
" ",
NULL};

static const char *CNAME[5] = { "vx ", "vz ", "tzz", "txx", "txz" };

int main(int argc, char **argv)
{
	modPar  mod;  recPar  rec;  snaPar sna;  wavPar wav;
	srcPar  src;  bndPar  bnd;  shotPar shot; adjSrcPar adj;
	float **src_nwav;
	size_t  sizem;
	int     ix, iz, i, it, verbose, ix0, halo, hz, izs, n1, nbad, show, stage;
	int     nstep, ocomp, oiz;
	float   tolf;
	double  tol, maxref;

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;
	if (!getparint("halo", &halo))       halo    = 4;
	if (!getparint("hz", &hz))           hz      = 3;
	if (!getparint("show", &show))       show    = 0;
	if (!getparint("stage", &stage))     stage   = 0;
	if (!getparint("nstep", &nstep))     nstep   = 1;
	if (!getparint("ocomp", &ocomp))     ocomp   = 3;  /* 0=vx 1=vz 2=tzz 3=txx 4=txz */
	if (!getparfloat("tol", &tolf))      tolf    = 1.0e-6f;
	tol = (double)tolf;

	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);

	if (mod.ischeme != 3) verr("test_fs_col needs ischeme=3; got %d", mod.ischeme);
	if (mod.iorder  != 4) verr("test_fs_col needs iorder=4; got %d",  mod.iorder);
	if (bnd.top != 1 && verbose) vmess("note: top=%d (not a free surface)", bnd.top);

	src.n = 0; src.type = 1;
	allocStoreSourceOnSurface(src);
	memset(&adj, 0, sizeof(adjSrcPar));
	adj.nsrc = 0; adj.nt = 1;

	readModel(&mod, &bnd);

	n1    = mod.naz;
	sizem = (size_t)mod.nax * mod.naz;

	/* same surface construction as test_cpml_dp.c */
	{
		int ioPx = mod.ioPx, ioPz = mod.ioPz;
		if (bnd.lef == 4 || bnd.lef == 2) ioPx += bnd.ntap;
		if (bnd.top == 4 || bnd.top == 2) ioPz += bnd.ntap;
		for (ix = 0; ix < mod.nx; ix++) {
			iz = ioPz;
			while (mod.l2m[(ix + ioPx) * n1 + iz] == 0.0) iz++;
			bnd.surface[ix + ioPx] = iz;
		}
		for (ix = 0; ix < ioPx; ix++) bnd.surface[ix] = bnd.surface[ioPx];
		for (ix = ioPx + mod.nx; ix < mod.iePx; ix++)
			bnd.surface[ix] = bnd.surface[mod.iePx - 1];
	}

	if (!getparint("ix0", &ix0)) ix0 = (mod.ioPx + mod.iePx) / 2;
	izs = (bnd.top==1) ? bnd.surface[ix0] : (mod.ioPz + mod.iePz)/2;
	if (getparint("oiz", &oiz)) izs = oiz;

	src_nwav    = (float **)calloc(1, sizeof(float *));
	src_nwav[0] = (float *)calloc(wav.nt > 0 ? wav.nt : 1, sizeof(float));

	printf("=========================================================\n");
	printf(" test_fs_col -- one column of F^T, measured vs computed\n");
	printf("=========================================================\n");
	printf("  grid        : nax=%d naz=%d   ioPx=%d iePx=%d\n",
	       mod.nax, mod.naz, mod.ioPx, mod.iePx);
	printf("  boundaries  : left=%d right=%d top=%d bottom=%d\n",
	       bnd.lef, bnd.rig, bnd.top, bnd.bot);
	printf("  output slot : o = %s[ix=%d, iz=%d]   nstep=%d\n\n",
	       CNAME[ocomp], ix0, izs, nstep);

	/* ============================================================== */
	/* (1) the code's adjoint column:  a = adjF e_o                   */
	/* ============================================================== */
	float *avx  = (float *)calloc(sizem, sizeof(float));
	float *avz  = (float *)calloc(sizem, sizeof(float));
	float *atzz = (float *)calloc(sizem, sizeof(float));
	float *atxx = (float *)calloc(sizem, sizeof(float));
	float *atxz = (float *)calloc(sizem, sizeof(float));

	{
		float *afld[5];
		afld[0]=avx; afld[1]=avz; afld[2]=atzz; afld[3]=atxx; afld[4]=atxz;
		afld[ocomp][ix0 * n1 + izs] = 1.0f;
	}

	cpmlAdjReset();
	if (stage == 1) {
FD_HOST_PARALLEL
{
		boundariesV_adj(mod, bnd, avx, avz, atzz, atxx, atxz,
			mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu, 0, verbose);
}
	}
	else if (stage == 2) {
FD_HOST_PARALLEL
{
		boundariesV_adj(mod, bnd, avx, avz, atzz, atxx, atxz,
			mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu, 0, verbose);
		cpmlAdjGhostT(mod, bnd, avx, avz);
}
	}
	else {
		for (it = nstep-1; it >= 0; it--) {
FD_HOST_PARALLEL
{
			elastic4_adj(mod, adj, bnd, it,
				avx, avz, atzz, atxx, atxz,
				mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu, 0, 1, verbose);
}
		}
	}
	printf("  stage=%d  adj avx[ix,%d] for ix=0..4 : %+.6e %+.6e %+.6e %+.6e %+.6e\n\n",
	       stage, izs, avx[0*n1+izs], avx[1*n1+izs], avx[2*n1+izs],
	       avx[3*n1+izs], avx[4*n1+izs]);

	/* ============================================================== */
	/* (2) the measured column: (F e_j)_o for every j in a box        */
	/* ============================================================== */
	float *fvx  = (float *)calloc(sizem, sizeof(float));
	float *fvz  = (float *)calloc(sizem, sizeof(float));
	float *ftzz = (float *)calloc(sizem, sizeof(float));
	float *ftxx = (float *)calloc(sizem, sizeof(float));
	float *ftxz = (float *)calloc(sizem, sizeof(float));
	float *fld[5];

	printf("  probing box : ix in [%d,%d]  iz in [%d,%d]  x 5 components\n",
	       ix0-halo, ix0+halo, izs-hz, izs+hz+2);
	printf("  (%d forward steps)\n\n",
	       (2*halo+1) * (hz + hz + 3) * 5);

	/* first pass: find the largest |entry| so "tol" can be relative */
	maxref = 0.0;
	nbad   = 0;

	printf("  %-4s %6s %5s   %-18s %-18s %-10s\n",
	       "comp", "ix", "iz", "measured (F e_j)_o", "code adjoint a_j", "diff");
	printf("  ---------------------------------------------------------------------------------\n");

	/* collect everything first so we can normalise, then print */
	{
		int    nb = (2*halo+1) * (hz + hz + 3) * 5;
		double *meas = (double *)calloc(nb, sizeof(double));
		double *code = (double *)calloc(nb, sizeof(double));
		int    *pix  = (int *)calloc(nb, sizeof(int));
		int    *piz  = (int *)calloc(nb, sizeof(int));
		int    *pc   = (int *)calloc(nb, sizeof(int));
		int     k = 0, c;

		for (c = 0; c < 5; c++) {
			for (ix = ix0-halo; ix <= ix0+halo; ix++) {
				for (iz = izs-hz; iz <= izs+hz+2; iz++) {
					size_t j;
					if (ix < 0 || ix >= mod.nax) continue;
					if (iz < 0 || iz >= mod.naz) continue;
					j = (size_t)ix * n1 + iz;

					memset(fvx,  0, sizem*sizeof(float));
					memset(fvz,  0, sizem*sizeof(float));
					memset(ftzz, 0, sizem*sizeof(float));
					memset(ftxx, 0, sizem*sizeof(float));
					memset(ftxz, 0, sizem*sizeof(float));
					fld[0]=fvx; fld[1]=fvz; fld[2]=ftzz; fld[3]=ftxx; fld[4]=ftxz;
					fld[c][j] = 1.0f;

					cpmlStateZero();
					for (it = 0; it < nstep; it++) {
FD_HOST_PARALLEL
{
						elastic4(mod, src, wav, bnd, it, 0, 0, src_nwav,
							fvx, fvz, ftzz, ftxx, ftxz,
							mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu, verbose);
}
					}
					meas[k] = (double)fld[ocomp][ix0*n1 + izs];
					switch (c) {
						case 0: code[k] = (double)avx [j]; break;
						case 1: code[k] = (double)avz [j]; break;
						case 2: code[k] = (double)atzz[j]; break;
						case 3: code[k] = (double)atxx[j]; break;
						case 4: code[k] = (double)atxz[j]; break;
					}
					pix[k]=ix; piz[k]=iz; pc[k]=c;
					if (fabs(meas[k]) > maxref) maxref = fabs(meas[k]);
					if (fabs(code[k]) > maxref) maxref = fabs(code[k]);
					k++;
				}
			}
		}

		if (maxref == 0.0) maxref = 1.0;
		for (i = 0; i < k; i++) {
			double d = fabs(meas[i] - code[i]);
			if (d <= tol * maxref) {
				if (!show) continue;
				if (fabs(meas[i]) <= tol*maxref && fabs(code[i]) <= tol*maxref)
					continue;
				printf("  %-4s %6d %5d   %+18.10e %+18.10e %10.3e   ok\n",
				       CNAME[pc[i]], pix[i], piz[i], meas[i], code[i], d);
				continue;
			}
			nbad++;
			printf("  %-4s %6d %5d   %+18.10e %+18.10e %10.3e%s\n",
			       CNAME[pc[i]], pix[i], piz[i], meas[i], code[i], d,
			       (fabs(code[i]) < tol*maxref) ? "   <- adjoint MISSES it" :
			       (fabs(meas[i]) < tol*maxref) ? "   <- adjoint INVENTS it" : "");
		}
		free(meas); free(code); free(pix); free(piz); free(pc);
	}

	printf("\n  largest |entry| in column : %.6e\n", maxref);
	if (nbad == 0)
		printf("  column MATCHES to %.1e relative -- the error is outside this box\n", tol);
	else
		printf("  %d MISMATCHED entries\n", nbad);
	printf("\n");

	return 0;
}
