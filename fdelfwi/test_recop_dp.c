/*
 * test_recop_dp.c - Dot-product test for the RECEIVER operator pair.
 *
 *   <R u, y>  ==  <u, R^T y>
 *
 *   R   = getRecTimes()        (wavefield state -> receiver data)
 *   R^T = applyAdjointSource() (receiver data   -> wavefield state)
 *
 * This is a PURE ALGEBRAIC test: no time stepping, no boundaries, no
 * absorbing layers.  R and R^T are applied once, to a single random
 * wavefield state u = (vx, vz, txx, tzz, txz) and a single random data
 * vector y.  The identity must therefore hold to FLOATING-POINT
 * ROUND-OFF (epsilon ~ 1e-6 in single precision), not to the 1e-2
 * tolerance that test_wave_op_dp accepts because of the non-self-adjoint
 * sponge layers.
 *
 * Why this test exists
 * --------------------
 * The Kirchhoff-Helmholtz (energy-flux) objective multiplies a velocity
 * residual by a stress residual AT THE SAME RECEIVER.  With the default
 * registration (rec_int_vx=0, rec_int_vz=0) those two factors live dt/2
 * apart in time and dx/2 (or dz/2) apart in space, so the product is not
 * a consistent discretisation of t_i v_i.
 *
 * rec_int_vx=2 / rec_int_vz=2 fixes that: getRecTimes.c co-locates the
 * velocities onto the Txx/Tzz node in BOTH space (2-node mean) and time
 * (half a velocity update, -0.5*ro*Dsigma).  applyAdjointSource.c must
 * then apply the exact transpose of that compound operator, which
 * scatters a velocity trace into the stress arrays as well.  This test
 * is the arbiter for that transpose.
 *
 * Run it for BOTH modes:
 *   rec_int_vx=0 rec_int_vz=0   -> exercises the original mode-0 path
 *   rec_int_vx=2 rec_int_vz=2   -> exercises the new co-located transpose
 *
 * Receiver placement
 * ------------------
 * The mode-2 transpose scatters over a 4th-order stencil, reaching
 * ix-2 .. ix+1 and iz-2 .. iz+2 around the receiver node.  Receivers
 * must therefore sit at least 3 grid points inside the padded array.
 * With ntaper>=10 and left/right/top/bottom=4 this is automatic.
 *
 * Parameters: the usual fdelmodc set, plus
 *   seed=12345    random number seed
 *   nthreads=1    0 = run R^T inside an OpenMP parallel region too
 *                 (checks the atomics), 1 = serial (default, bitwise
 *                 reproducible)
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "par.h"
#include "segy.h"
#include "fdelfwi.h"
#include "fdelfwi_offload.h"

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);
int readModel(modPar *mod, bndPar *bnd);
int allocStoreSourceOnSurface(srcPar src);
int freeStoreSourceOnSurface(void);

int getRecTimes(modPar mod, recPar rec, bndPar bnd, int itime, int isam,
    float *vx, float *vz, float *tzz, float *txx, float *txz, float *q,
    float *l2m, float *lam, float *rox, float *roz,
    float *rec_vx, float *rec_vz, float *rec_txx, float *rec_tzz,
    float *rec_txz, float *rec_p, float *rec_pp, float *rec_ss,
    float *rec_q, float *rec_udp, float *rec_udvz,
    float *rec_dxvx, float *rec_dzvz, int verbose);

int applyAdjointSource(modPar mod, adjSrcPar adj, int itime,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *rox, float *roz, float *mul,
    int rec_delay, int rec_skipdt, int phase, int verbose);

double wallclock_time(void);

char *sdoc[] = {
" ",
" test_recop_dp - dot-product test of the receiver restriction/injection pair",
" ",
"   <R u, y> == <u, R^T y>",
" ",
" R   = getRecTimes()        (wavefield -> data)",
" R^T = applyAdjointSource() (data -> wavefield)",
" ",
" Pure algebra: no time stepping, no boundaries. Must pass to round-off.",
" Run with rec_int_vx=2 rec_int_vz=2 to test the co-located transpose.",
" ",
" Same parameters as fdelmodc, plus:",
"   seed=12345    random number seed",
"   nthreads=1    0 = also run R^T multi-threaded (checks the atomics)",
" ",
NULL};

/* One scalar entry of the data vector: which receiver, which component. */
typedef struct {
	int   irec;
	int   typ;      /* adjoint source type: 1=P 2=Txz 3=Tzz 4=Txx 6=Fx 7=Fz */
	const char *name;
} dataEntry;

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
	size_t  sizem;
	int     n1, ibndx, ibndz;
	int     verbose, seed, nthreads;
	int     i, ir, ic, ncomp, nd;
	double  t0, t1, lhs, rhs, epsilon, denom;
	dataEntry *de;

	t0 = wallclock_time();

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose",  &verbose))  verbose  = 0;
	if (!getparint("seed",     &seed))     seed     = 12345;
	if (!getparint("nthreads", &nthreads)) nthreads = 1;

	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);
	allocStoreSourceOnSurface(src);
	readModel(&mod, &bnd);

	n1    = mod.naz;
	sizem = (size_t)mod.nax * mod.naz;

	ibndx = mod.ioPx;
	ibndz = mod.ioPz;
	if (bnd.lef == 4 || bnd.lef == 2) ibndx += bnd.ntap;
	if (bnd.top == 4 || bnd.top == 2) ibndz += bnd.ntap;

	if (!rec.n) verr("test_recop_dp: no receivers defined.");

	printf("=== Receiver Operator Dot Product Test ===\n");
	printf("  <R u, y> == <u, R^T y>   (R = getRecTimes, R^T = applyAdjointSource)\n\n");
	printf("  grid        : nx=%d nz=%d  padded nax=%d naz=%d  n1=%d\n",
	       mod.nx, mod.nz, mod.nax, mod.naz, n1);
	printf("  ischeme     : %d   iorder: %d\n", mod.ischeme, mod.iorder);
	printf("  ibndx/ibndz : %d / %d   (ioXx=%d ioZz=%d)\n",
	       ibndx, ibndz, mod.ioXx, mod.ioZz);
	printf("  receivers   : %d\n", rec.n);
	printf("  int_vx=%d int_vz=%d int_p=%d  <-- recording mode under test\n",
	       rec.int_vx, rec.int_vz, rec.int_p);
	printf("  rec_type    : vx=%d vz=%d txx=%d tzz=%d txz=%d p=%d\n",
	       rec.type.vx, rec.type.vz, rec.type.txx, rec.type.tzz,
	       rec.type.txz, rec.type.p);
	printf("  seed        : %d\n\n", seed);

	if (rec.int_p == 3)
		verr("test_recop_dp: rec_int_p=3 (bilinear) has no transpose in "
		     "applyAdjointSource yet.");
	if (rec.int_vx == 1 || rec.int_vz == 1)
		verr("test_recop_dp: rec_int_vx/vz=1 has no transpose in "
		     "applyAdjointSource yet.");

	/* ---- receivers must be far enough from the padded array edge ---- */
	{
		int halo = (rec.int_vx == 2 || rec.int_vz == 2) ? 3 : 1;
		for (ir = 0; ir < rec.n; ir++) {
			int rx = rec.x[ir] + ibndx;
			int rz = rec.z[ir] + ibndz;
			if (rx - halo < 0 || rx + halo >= mod.nax ||
			    rz - halo < 0 || rz + halo >= mod.naz)
				verr("test_recop_dp: receiver %d at padded (%d,%d) is closer "
				     "than %d points to the array edge; increase ntaper.",
				     ir, rx, rz, halo);
		}
	}

	/*==============================================================*/
	/* 1. random wavefield state u                                  */
	/*==============================================================*/
	float *u_vx  = (float *)calloc(sizem, sizeof(float));
	float *u_vz  = (float *)calloc(sizem, sizeof(float));
	float *u_txx = (float *)calloc(sizem, sizeof(float));
	float *u_tzz = (float *)calloc(sizem, sizeof(float));
	float *u_txz = (float *)calloc(sizem, sizeof(float));
	assert(u_vx && u_vz && u_txx && u_tzz && u_txz);

	srand48(seed);
	for (i = 0; i < (int)sizem; i++) {
		u_vx[i]  = (float)(drand48() * 2.0 - 1.0);
		u_vz[i]  = (float)(drand48() * 2.0 - 1.0);
		u_txx[i] = (float)(drand48() * 2.0 - 1.0);
		u_tzz[i] = (float)(drand48() * 2.0 - 1.0);
		u_txz[i] = (float)(drand48() * 2.0 - 1.0);
	}

	/*==============================================================*/
	/* 2. build the data-vector layout                              */
	/*==============================================================*/
	/* Order must match nothing in particular -- d and y are indexed
	 * identically on both sides -- but one entry per (receiver,
	 * component) keeps the bookkeeping trivial. */
	ncomp = 0;
	if (rec.type.vx)  ncomp++;
	if (rec.type.vz)  ncomp++;
	if (rec.type.txx) ncomp++;
	if (rec.type.tzz) ncomp++;
	if (rec.type.txz) ncomp++;
	if (rec.type.p)   ncomp++;
	if (!ncomp)
		verr("test_recop_dp: enable at least one of rec_type_vx/vz/txx/tzz/txz/p.");

	nd = rec.n * ncomp;
	de = (dataEntry *)malloc(nd * sizeof(dataEntry));
	i  = 0;
	for (ir = 0; ir < rec.n; ir++) {
		if (rec.type.vx)  { de[i].irec=ir; de[i].typ=6; de[i].name="vx";  i++; }
		if (rec.type.vz)  { de[i].irec=ir; de[i].typ=7; de[i].name="vz";  i++; }
		if (rec.type.txx) { de[i].irec=ir; de[i].typ=4; de[i].name="txx"; i++; }
		if (rec.type.tzz) { de[i].irec=ir; de[i].typ=3; de[i].name="tzz"; i++; }
		if (rec.type.txz) { de[i].irec=ir; de[i].typ=2; de[i].name="txz"; i++; }
		if (rec.type.p)   { de[i].irec=ir; de[i].typ=1; de[i].name="p";   i++; }
	}

	/*==============================================================*/
	/* 3. d = R u   via getRecTimes (isam=0, one sample)            */
	/*==============================================================*/
	recPar rec1 = rec;
	rec1.nt = 1;   /* getRecTimes indexes rec_*[irec*rec.nt + isam] */

	float *r_vx  = (float *)calloc(rec.n, sizeof(float));
	float *r_vz  = (float *)calloc(rec.n, sizeof(float));
	float *r_txx = (float *)calloc(rec.n, sizeof(float));
	float *r_tzz = (float *)calloc(rec.n, sizeof(float));
	float *r_txz = (float *)calloc(rec.n, sizeof(float));
	float *r_p   = (float *)calloc(rec.n, sizeof(float));
	float *r_pp  = (float *)calloc(rec.n, sizeof(float));
	float *r_ss  = (float *)calloc(rec.n, sizeof(float));
	float *r_q   = (float *)calloc(rec.n, sizeof(float));
	float *r_udp = (float *)calloc(rec.n, sizeof(float));
	float *r_udv = (float *)calloc(rec.n, sizeof(float));
	float *r_dxv = (float *)calloc(rec.n, sizeof(float));
	float *r_dzv = (float *)calloc(rec.n, sizeof(float));

	getRecTimes(mod, rec1, bnd, 0, 0,
	            u_vx, u_vz, u_tzz, u_txx, u_txz, NULL,
	            mod.l2m, mod.lam, mod.rox, mod.roz,
	            r_vx, r_vz, r_txx, r_tzz, r_txz, r_p, r_pp, r_ss,
	            r_q, r_udp, r_udv, r_dxv, r_dzv, verbose);

	float *d = (float *)calloc(nd, sizeof(float));
	for (i = 0; i < nd; i++) {
		ir = de[i].irec;
		switch (de[i].typ) {
			case 6: d[i] = r_vx[ir];  break;
			case 7: d[i] = r_vz[ir];  break;
			case 4: d[i] = r_txx[ir]; break;
			case 3: d[i] = r_tzz[ir]; break;
			case 2: d[i] = r_txz[ir]; break;
			case 1: d[i] = r_p[ir];   break;
		}
	}

	/*==============================================================*/
	/* 4. random data y, and LHS = <R u, y>                         */
	/*==============================================================*/
	float *y = (float *)calloc(nd, sizeof(float));
	srand48(seed + 77777);
	for (i = 0; i < nd; i++) y[i] = (float)(drand48() * 2.0 - 1.0);

	lhs = 0.0;
	for (i = 0; i < nd; i++) lhs += (double)d[i] * (double)y[i];

	/*==============================================================*/
	/* 5. z = R^T y   via applyAdjointSource                        */
	/*==============================================================*/
	/* Grid registration replicates readResidual.c exactly:
	 *   types 2,6 -> xi = ioXx + grid_x   (= ibndx + grid_x + 1)
	 *   others    -> xi = ibndx + grid_x
	 *   types 2,7 -> zi = ioZz + grid_z   (= ibndz + grid_z + 1)
	 *   others    -> zi = ibndz + grid_z                              */
	memset(&adj, 0, sizeof(adjSrcPar));
	adj.nsrc   = nd;
	adj.nt     = 1;
	adj.xi     = (size_t *)malloc(nd * sizeof(size_t));
	adj.zi     = (size_t *)malloc(nd * sizeof(size_t));
	adj.typ    = (int *)malloc(nd * sizeof(int));
	adj.orient = (int *)malloc(nd * sizeof(int));
	adj.x      = (float *)malloc(nd * sizeof(float));
	adj.z      = (float *)malloc(nd * sizeof(float));
	adj.wav    = (float *)malloc(nd * sizeof(float));
	/* The transpose must know which recording mode produced the data. */
	adj.int_vx = rec.int_vx;
	adj.int_vz = rec.int_vz;

	for (i = 0; i < nd; i++) {
		int typ = de[i].typ;
		ir = de[i].irec;
		adj.typ[i]    = typ;
		adj.orient[i] = 1;                      /* monopole */
		adj.xi[i] = (typ == 2 || typ == 6) ? (size_t)(mod.ioXx + rec.x[ir])
		                                   : (size_t)(ibndx    + rec.x[ir]);
		adj.zi[i] = (typ == 2 || typ == 7) ? (size_t)(mod.ioZz + rec.z[ir])
		                                   : (size_t)(ibndz    + rec.z[ir]);
		adj.x[i]  = rec.xr[ir];
		adj.z[i]  = rec.zr[ir];
		adj.wav[i] = y[i];
	}

	float *z_vx  = (float *)calloc(sizem, sizeof(float));
	float *z_vz  = (float *)calloc(sizem, sizeof(float));
	float *z_txx = (float *)calloc(sizem, sizeof(float));
	float *z_tzz = (float *)calloc(sizem, sizeof(float));
	float *z_txz = (float *)calloc(sizem, sizeof(float));

	/* Phase order mirrors the adjoint kernel: phase 2 (stress injection
	 * point) then phase 1 (force injection point).  Because the fields
	 * start at zero and nothing propagates in between, the order is
	 * irrelevant here -- what matters is that EVERY contribution of
	 * R^T is emitted across the two calls, exactly once. */
	if (nthreads == 0) {
FD_HOST_PARALLEL
		{
			applyAdjointSource(mod, adj, 0, z_vx, z_vz, z_tzz, z_txx, z_txz,
			                   mod.rox, mod.roz, mod.muu, 0, 1, 2, verbose);
		}
FD_HOST_PARALLEL
		{
			applyAdjointSource(mod, adj, 0, z_vx, z_vz, z_tzz, z_txx, z_txz,
			                   mod.rox, mod.roz, mod.muu, 0, 1, 1, verbose);
		}
	}
	else {
		applyAdjointSource(mod, adj, 0, z_vx, z_vz, z_tzz, z_txx, z_txz,
		                   mod.rox, mod.roz, mod.muu, 0, 1, 2, verbose);
		applyAdjointSource(mod, adj, 0, z_vx, z_vz, z_tzz, z_txx, z_txz,
		                   mod.rox, mod.roz, mod.muu, 0, 1, 1, verbose);
	}

	/*==============================================================*/
	/* 6. RHS = <u, R^T y>                                          */
	/*==============================================================*/
	rhs = 0.0;
	for (i = 0; i < (int)sizem; i++) {
		rhs += (double)u_vx[i]  * (double)z_vx[i];
		rhs += (double)u_vz[i]  * (double)z_vz[i];
		rhs += (double)u_txx[i] * (double)z_txx[i];
		rhs += (double)u_tzz[i] * (double)z_tzz[i];
		rhs += (double)u_txz[i] * (double)z_txz[i];
	}

	/*==============================================================*/
	/* 7. report                                                    */
	/*==============================================================*/
	denom = fabs(lhs) > fabs(rhs) ? fabs(lhs) : fabs(rhs);
	epsilon = (denom > 0.0) ? fabs(lhs - rhs) / denom : fabs(lhs - rhs);

	printf("  data vector : %d entries (%d receivers x %d components)\n",
	       nd, rec.n, ncomp);
	printf("\n  <R u, y>     = %+.15e\n", lhs);
	printf("  <u, R^T y>   = %+.15e\n", rhs);
	printf("  ratio        = %.12f\n", (fabs(lhs) > 0.0) ? rhs / lhs : 0.0);
	printf("  rel. error   = %.3e\n", epsilon);
	printf("  %s  (threshold 1e-5, single precision round-off)\n",
	       epsilon < 1e-5 ? "PASS" : "FAIL");

	/* Per-component breakdown: rerun with only one component live, so a
	 * failure points at the guilty transpose instead of the total. */
	if (ncomp > 1) {
		printf("\n  per-component breakdown:\n");
		printf("    %-5s %18s %18s %12s\n", "comp", "<R u, y>", "<u, R^T y>", "rel.err");
		for (ic = 0; ic < ncomp; ic++) {
			double l = 0.0, r = 0.0, dn, ep;
			int typ = de[ic].typ;

			memset(z_vx,  0, sizem * sizeof(float));
			memset(z_vz,  0, sizem * sizeof(float));
			memset(z_txx, 0, sizem * sizeof(float));
			memset(z_tzz, 0, sizem * sizeof(float));
			memset(z_txz, 0, sizem * sizeof(float));

			for (i = 0; i < nd; i++)
				adj.wav[i] = (de[i].typ == typ) ? y[i] : 0.0f;

			applyAdjointSource(mod, adj, 0, z_vx, z_vz, z_tzz, z_txx, z_txz,
			                   mod.rox, mod.roz, mod.muu, 0, 1, 2, verbose);
			applyAdjointSource(mod, adj, 0, z_vx, z_vz, z_tzz, z_txx, z_txz,
			                   mod.rox, mod.roz, mod.muu, 0, 1, 1, verbose);

			for (i = 0; i < nd; i++)
				if (de[i].typ == typ) l += (double)d[i] * (double)y[i];
			for (i = 0; i < (int)sizem; i++) {
				r += (double)u_vx[i]  * (double)z_vx[i];
				r += (double)u_vz[i]  * (double)z_vz[i];
				r += (double)u_txx[i] * (double)z_txx[i];
				r += (double)u_tzz[i] * (double)z_tzz[i];
				r += (double)u_txz[i] * (double)z_txz[i];
			}
			dn = fabs(l) > fabs(r) ? fabs(l) : fabs(r);
			ep = (dn > 0.0) ? fabs(l - r) / dn : fabs(l - r);
			printf("    %-5s %+18.9e %+18.9e %12.3e  %s\n",
			       de[ic].name, l, r, ep, ep < 1e-5 ? "PASS" : "FAIL");
		}
	}

	t1 = wallclock_time();
	printf("\n  wall time    : %.2f s\n", t1 - t0);
	printf("=== Receiver operator dot product test completed ===\n");

	free(u_vx); free(u_vz); free(u_txx); free(u_tzz); free(u_txz);
	free(z_vx); free(z_vz); free(z_txx); free(z_tzz); free(z_txz);
	free(r_vx); free(r_vz); free(r_txx); free(r_tzz); free(r_txz); free(r_p);
	free(r_pp); free(r_ss); free(r_q); free(r_udp); free(r_udv);
	free(r_dxv); free(r_dzv);
	free(d); free(y); free(de);
	free(adj.xi); free(adj.zi); free(adj.typ); free(adj.orient);
	free(adj.x); free(adj.z); free(adj.wav);
	freeStoreSourceOnSurface();

	return (epsilon < 1e-5) ? 0 : 1;
}
