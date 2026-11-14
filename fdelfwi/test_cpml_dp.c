/*
 * test_cpml_dp.c -- dot-product test of the BARE PROPAGATOR, boundaries
 *                   included, with NO source and NO receiver machinery.
 *
 * WHAT IS BEING TESTED, AND WHY THIS FORM
 * ---------------------------------------
 * test_wave_op_dp.c tests the full operator  source -> propagate -> record.
 * That is the right end-to-end test, but when it fails it does not say which
 * of the three parts is wrong, and three quarters of its code is source
 * injection and receiver registration.  Those are already validated.  The new
 * code is the CFS-CPML transpose (boundaries_cpml_adj.c), which lives entirely
 * inside the propagator, so this test removes everything else:
 *
 *     src.n = 0        no source injection at all
 *     adj.nsrc = 0     no adjoint source injection at all
 *
 * and tests the N-step state-to-state map directly:
 *
 *     S_N = F^N S_0          forward,  nstep steps of elastic4
 *     R_0 = (F^T)^N R_N      adjoint,  nstep steps of elastic4_adj
 *     <S_N, R_N>  ==  <S_0, R_0>
 *
 * The state is the full padded wavefield (vx, vz, txx, tzz, txz) over every
 * point of the [nax x naz] arrays -- halo, PML pad and all.  Seeding the halo
 * with random numbers too is deliberate: if the forward stencil reads a point
 * that the transpose never writes back to, the test fails.  That is the whole
 * class of bug a loop-bound mismatch produces, and it is invisible to a test
 * that only scores the interior.
 *
 * WHY psi DOES NOT NEED TO BE EXPOSED
 * -----------------------------------
 * The true forward state is larger than (v,t): CFS-CPML carries 8 memory
 * variables psi, and one time step is a linear map on (v, t, psi).  Both sets
 * of psi are static inside boundaries.c / boundaries_cpml_adj.c and there is
 * no accessor.  There does not need to be one.  Writing E for "embed (v,t)
 * with psi = 0" and P for "project onto (v,t)", what this test actually
 * measures is
 *
 *     L = P F^N E       and       L^T = E^T (F^T)^N P^T
 *
 * E^T is "read off the (v,t) part", P^T is "start with psi_a = 0".  So running
 * the forward from psi = 0 and the adjoint from psi_a = 0, and scoring only
 * (v,t) at both ends, is EXACTLY the dot-product test for L.  Both are true in
 * a fresh process -- the forward psi arrays are calloc'd on first use and this
 * program runs the forward sweep exactly once; cpmlAdjReset() zeroes the
 * adjoint ones.  Nothing is hidden by this choice: L is adjoint-consistent if
 * and only if F is, because psi is reachable from and observable in (v,t).
 *
 * nstep MATTERS.  With nstep=1 the psi recursion contributes only its c*D(u)
 * term and the b*psi feedback is never exercised -- a wrong b would pass.  Use
 * nstep >= 3 for a real test; the default is 10.
 *
 * HOW TO KNOW THE TEST ITSELF IS RIGHT
 * ------------------------------------
 * Run it first with taper boundaries (left=4 right=4 top=4 bottom=4).  A taper
 * is a pointwise multiply, hence diagonal, hence exactly self-adjoint, and
 * that path through elastic4_adj.c is untouched by the CPML work.  If the
 * taper run does not pass, the harness is wrong, not the CPML.  Only then
 * switch to left=2 right=2 top=2 bottom=2.
 *
 * Parameters: the usual fdelmodc/fdelfwi set, plus
 *   nstep=10      number of time steps to propagate
 *   seed=12345    random number seed
 */

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<string.h>
#include"par.h"
#include"segy.h"
#include"fdelfwi.h"

#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

double wallclock_time(void);

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);
int readModel(modPar *mod, bndPar *bnd);
int allocStoreSourceOnSurface(srcPar src);
int freeStoreSourceOnSurface(void);

int elastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime,
    int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz,
    float *tzz, float *txx, float *txz, float *rox, float *roz,
    float *l2m, float *lam, float *mul, int verbose);

int elastic4_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *rox, float *roz, float *l2m, float *lam, float *mul,
    int rec_delay, int rec_skipdt, int verbose);

char *sdoc[] = {
" ",
" test_cpml_dp - dot-product test of the bare propagator (no source, no receivers)",
" ",
"   Verifies  <F^N S0, R> = <S0, (F^T)^N R>  over the full padded wavefield.",
"   Isolates the boundary transpose, CFS-CPML in particular.",
" ",
" Same parameters as fdelmodc, plus:",
"   nstep=10      number of time steps to propagate",
"   seed=12345    random number seed",
" ",
" Run with left=4 right=4 top=4 bottom=4 FIRST (taper is exactly self-adjoint,",
" so it validates the harness), then with left=2 right=2 top=2 bottom=2.",
" ",
NULL};


/* <a,b> over the five components of the full padded state */
static double state_dot(size_t n,
    float *avx, float *avz, float *atzz, float *atxx, float *atxz,
    float *bvx, float *bvz, float *btzz, float *btxx, float *btxz)
{
	double s = 0.0;
	size_t i;
	for (i = 0; i < n; i++) {
		s += (double)avx[i]  * (double)bvx[i];
		s += (double)avz[i]  * (double)bvz[i];
		s += (double)atzz[i] * (double)btzz[i];
		s += (double)atxx[i] * (double)btxx[i];
		s += (double)atxz[i] * (double)btxz[i];
	}
	return s;
}

static void fill_random(size_t n, float *a)
{
	size_t i;
	for (i = 0; i < n; i++) a[i] = (float)(drand48() * 2.0 - 1.0);
}

static double maxabs(size_t n, float *a)
{
	double m = 0.0, v;
	size_t i;
	for (i = 0; i < n; i++) { v = fabs((double)a[i]); if (v > m) m = v; }
	return m;
}


/*--------------------------------------------------------------------
 * fs_project -- remove free-surface CONSTRAINED rows from the scoring.
 *
 * Diagnostic for the top free surface only.  Zeroing a slot in BOTH S0
 * and R_N deletes it from both sides of <F S0, R> = <S0, F^T R>, which
 * restricts the test to the subspace where the variable is a genuine
 * degree of freedom.  That matters because the elastic forward does not
 * treat every slot as one:
 *
 *   vz[surface]            F1's vz loop starts at ioZz = surface+1, so the
 *                          forward never updates it, and boundariesV then
 *                          overwrites tzz[surface], discarding its only
 *                          downstream use.  The output is INDEPENDENT of
 *                          this slot, so an exact transpose must return 0
 *                          there -- the adjoint instead leaves it untouched.
 *   tzz[surface]           forced to 0 by boundariesV
 *   txx[surface]           overwritten by boundariesV from vx
 *   txz[surface], [s-1]    ghost rows, overwritten by the antisymmetric mirror
 *
 * Bitmask so each can be isolated:  1=vz  2=tzz  4=txx  8=txz ghosts.
 * Default 0 = off, every pre-existing case is bit-identical.
 *------------------------------------------------------------------*/
static void fs_project(int mask, modPar mod, bndPar bnd,
                       float *vz, float *tzz, float *txx, float *txz)
{
	int ix, iz, n1 = mod.naz;

	if (mask == 0 || bnd.top != 1) return;

	for (ix = 0; ix < mod.nax; ix++) {
		iz = bnd.surface[ix];
		if (iz < 1 || iz >= mod.naz - 2) continue;
		if (mask & 1) vz [ix*n1+iz] = 0.0f;
		if (mask & 2) tzz[ix*n1+iz] = 0.0f;
		if (mask & 4) txx[ix*n1+iz] = 0.0f;
		if (mask & 8) {
			txz[ix*n1+iz]   = 0.0f;
			txz[ix*n1+iz-1] = 0.0f;
		}
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
	adjSrcPar adj;
	float **src_nwav;
	double  t0, t1;
	size_t  sizem;
	int     ix, iz, i, it, nstep, seed, verbose, fs_proj, fs_side;
	double  lhs, rhs, ratio, epsilon;

	t0 = wallclock_time();

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;
	if (!getparint("nstep", &nstep))     nstep   = 10;
	if (!getparint("seed", &seed))       seed    = 12345;
	if (!getparint("fs_proj", &fs_proj)) fs_proj = 0;
	/* which side of E = S0^T (F^T - adjF) R the mask is applied to.
	 * Masking R kills COLUMNS of the discrepancy, masking S0 kills ROWS,
	 * so the two separate "the adjoint mishandles an input adj_txx[s]"
	 * from "the forward's dependence on input txx[s] is not reproduced". */
	if (!getparint("fs_side", &fs_side)) fs_side = 0;  /* 0=both 1=S0 2=R */

	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);

	if (mod.ischeme != 3)
		verr("test_cpml_dp needs ischeme=3 (elastic); got %d", mod.ischeme);
	if (mod.iorder != 4)
		verr("test_cpml_dp needs iorder=4; got %d", mod.iorder);

	/* ---- kill every source, forward and adjoint ------------------- */
	/* elastic4 always calls applySource (type<6 or type>5), but the
	 * injection loop is `for (isrc=0; isrc<src.n; isrc++)` so n=0 makes it
	 * a no-op, as does storeSourceOnSurface/reStoreSourceOnSurface.
	 * type is forced to 1 only to keep it away from the special txz path
	 * at applySource.c:58, which is guarded on type==2. */
	src.n    = 0;
	src.type = 1;
	allocStoreSourceOnSurface(src);

	memset(&adj, 0, sizeof(adjSrcPar));
	adj.nsrc = 0;
	adj.nt   = nstep;

	readModel(&mod, &bnd);

	/* bnd.surface is read by the free-surface paths; readModel does not
	 * fill it. Under PML/taper nothing uses it, but it must not be garbage. */
	{
		int ioPx = mod.ioPx, ioPz = mod.ioPz, n1 = mod.naz;
		if (bnd.lef == 4 || bnd.lef == 2) ioPx += bnd.ntap;
		if (bnd.top == 4 || bnd.top == 2) ioPz += bnd.ntap;
		for (ix = 0; ix < mod.nx; ix++) {
			iz = ioPz;
			while (mod.l2m[(ix + ioPx) * n1 + iz] == 0.0) iz++;
			bnd.surface[ix + ioPx] = iz;
		}
		for (ix = 0; ix < ioPx; ix++)
			bnd.surface[ix] = bnd.surface[ioPx];
		for (ix = ioPx + mod.nx; ix < mod.iePx; ix++)
			bnd.surface[ix] = bnd.surface[mod.iePx - 1];
	}

	/* src_nwav is dereferenced on the way in even when nothing is injected */
	src_nwav    = (float **)calloc(1, sizeof(float *));
	src_nwav[0] = (float *)calloc(wav.nt > 0 ? wav.nt : 1, sizeof(float));

	sizem = (size_t)mod.nax * mod.naz;

	printf("=========================================================\n");
	printf(" test_cpml_dp -- bare propagator dot product test\n");
	printf("=========================================================\n");
	printf("  grid           : nx=%d nz=%d  (padded nax=%d naz=%d)\n",
	       mod.nx, mod.nz, mod.nax, mod.naz);
	printf("  boundaries     : left=%d right=%d top=%d bottom=%d  (1=free 2=pml 3=rigid 4=taper)\n",
	       bnd.lef, bnd.rig, bnd.top, bnd.bot);
	printf("  npml / ntaper  : %d / %d\n", bnd.npml, bnd.ntap);
	printf("  time steps     : %d\n", nstep);
	printf("  state size     : %zu points x 5 components = %zu unknowns\n",
	       sizem, 5 * sizem);
	printf("  seed           : %d\n\n", seed);

	/* ============================================================== */
	/* S_0 : random initial state                                     */
	/* ============================================================== */
	float *fvx  = (float *)calloc(sizem, sizeof(float));
	float *fvz  = (float *)calloc(sizem, sizeof(float));
	float *ftzz = (float *)calloc(sizem, sizeof(float));
	float *ftxx = (float *)calloc(sizem, sizeof(float));
	float *ftxz = (float *)calloc(sizem, sizeof(float));

	float *s0vx  = (float *)calloc(sizem, sizeof(float));
	float *s0vz  = (float *)calloc(sizem, sizeof(float));
	float *s0tzz = (float *)calloc(sizem, sizeof(float));
	float *s0txx = (float *)calloc(sizem, sizeof(float));
	float *s0txz = (float *)calloc(sizem, sizeof(float));

	srand48(seed);
	fill_random(sizem, fvx);
	fill_random(sizem, fvz);
	fill_random(sizem, ftzz);
	fill_random(sizem, ftxx);
	fill_random(sizem, ftxz);

	if (fs_side != 2) fs_project(fs_proj, mod, bnd, fvz, ftzz, ftxx, ftxz);

	memcpy(s0vx,  fvx,  sizem * sizeof(float));
	memcpy(s0vz,  fvz,  sizem * sizeof(float));
	memcpy(s0tzz, ftzz, sizem * sizeof(float));
	memcpy(s0txx, ftxx, sizem * sizeof(float));
	memcpy(s0txz, ftxz, sizem * sizeof(float));

	/* ============================================================== */
	/* S_N = F^N S_0                                                  */
	/* ============================================================== */
	for (it = 0; it < nstep; it++) {
#pragma omp parallel default(shared)
{
		elastic4(mod, src, wav, bnd, it, 0, 0, src_nwav,
			fvx, fvz, ftzz, ftxx, ftxz,
			mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu, verbose);
}
	}

	printf("  forward  %d steps done : max|vx|=%.6e max|txx|=%.6e\n",
	       nstep, maxabs(sizem, fvx), maxabs(sizem, ftxx));

	/* ============================================================== */
	/* R_N : random adjoint state                                     */
	/* ============================================================== */
	float *avx  = (float *)calloc(sizem, sizeof(float));
	float *avz  = (float *)calloc(sizem, sizeof(float));
	float *atzz = (float *)calloc(sizem, sizeof(float));
	float *atxx = (float *)calloc(sizem, sizeof(float));
	float *atxz = (float *)calloc(sizem, sizeof(float));

	srand48(seed + 77777);
	fill_random(sizem, avx);
	fill_random(sizem, avz);
	fill_random(sizem, atzz);
	fill_random(sizem, atxx);
	fill_random(sizem, atxz);

	if (fs_side != 1) fs_project(fs_proj, mod, bnd, avz, atzz, atxx, atxz);

	/* LHS = <F^N S_0, R_N>, taken BEFORE the adjoint sweep overwrites R_N */
	lhs = state_dot(sizem, fvx, fvz, ftzz, ftxx, ftxz,
	                       avx, avz, atzz, atxx, atxz);

	/* ============================================================== */
	/* R_0 = (F^T)^N R_N                                              */
	/* ============================================================== */
	/* ONCE, outside the loop: the adjoint psi recursion runs continuously
	 * backwards in time and must not be restarted at each step. */
	cpmlAdjReset();

	for (it = nstep - 1; it >= 0; it--) {
#pragma omp parallel default(shared)
{
		elastic4_adj(mod, adj, bnd, it,
			avx, avz, atzz, atxx, atxz,
			mod.rox, mod.roz, mod.l2m, mod.lam, mod.muu,
			/*rec_delay=*/0, /*rec_skipdt=*/1, verbose);
}
	}

	printf("  adjoint  %d steps done : max|vx|=%.6e max|txx|=%.6e\n\n",
	       nstep, maxabs(sizem, avx), maxabs(sizem, atxx));

	rhs = state_dot(sizem, s0vx, s0vz, s0tzz, s0txx, s0txz,
	                       avx,  avz,  atzz,  atxx,  atxz);

	/* ============================================================== */
	/* verdict                                                        */
	/* ============================================================== */
	printf("  <F^N S0, R>    = %+.15e\n", lhs);
	printf("  <S0, (F^T)^N R> = %+.15e\n", rhs);

	if (fabs(lhs) > 0.0) {
		ratio   = rhs / lhs;
		epsilon = fabs(ratio - 1.0);
	} else {
		ratio   = 0.0;
		epsilon = fabs(rhs);
	}

	printf("  ratio          = %.14f\n", ratio);
	printf("  epsilon        = %.3e\n", epsilon);
	printf("\n  %s\n", epsilon < 1e-5 ? "PASS" : "FAIL");

	t1 = wallclock_time();
	printf("\n  wall time: %.2f s\n", t1 - t0);

	free(fvx); free(fvz); free(ftzz); free(ftxx); free(ftxz);
	free(s0vx); free(s0vz); free(s0tzz); free(s0txx); free(s0txz);
	free(avx); free(avz); free(atzz); free(atxx); free(atxz);
	free(src_nwav[0]); free(src_nwav);
	freeStoreSourceOnSurface();

	return (epsilon < 1e-5) ? 0 : 1;
}
