/*
 * test_ei_planewave.c -- analytic plane-wave calibration of the gamma=0
 * energy-imaging accumulators.
 *
 * WHY.  On a pure-P source cell the two gamma=0 halves sit at equipartition
 * (rms(P)/rms(K) = 0.9917), but on a pure-S cell they do not (0.2468).  Only
 * one of V_0's two paths is exercised in each case -- P waves are curl-free,
 * S waves are divergence-free -- so the pure-P agreement calibrates the div
 * path ONLY, and the curl path has never been checked independently.  Two
 * explanations fit the S number equally well and the field data cannot tell
 * them apart:
 *
 *   (a) a scaling bug in the curl term of ei_accum_potential_gamma0, or
 *   (b) correct behaviour -- V_0 vanishes identically for a P-S pair while K
 *       does not, so the ratio must fall wherever converted energy dominates,
 *       and in the pure-S cell the adjoint field is strongly converted.
 *
 * Running gamma=2 on the same wavefields gives 0.9986, which is consistent
 * with BOTH: gamma=2 never touches the curl path, and its V_2 does not null
 * conversions.  Hence this test, which removes the ambiguity by removing the
 * conversions: a single analytic plane wave paired with ITSELF.
 *
 * THE PREDICTION, which is what makes this decisive.  For v = A p cos(phi),
 * phi = k.x - wt, with the field paired against itself:
 *
 *   K   = b |grad.tau|^2 = (1/rho)(rho A w sin phi)^2 = rho A^2 w^2 sin^2 phi
 *
 *   P wave   curl v = 0,  div v = -(A w/Vp) sin phi
 *            V_0 = (l+2m) div^2 = rho Vp^2 A^2 w^2/Vp^2 sin^2 = rho A^2 w^2 sin^2
 *   S wave   div v = 0,   curl v =  (A w/Vs) sin phi
 *            V_0 = m curl^2  = rho Vs^2 A^2 w^2/Vs^2 sin^2 = rho A^2 w^2 sin^2
 *
 * Both modes give K = V_0 POINTWISE, and the velocity cancels in each -- that
 * cancellation IS the mode independence gamma=0 is built on.  So the test is
 * a single number with no tolerance to argue about:
 *
 *                      rms(P)/rms(K) = 1  for P and for S.
 *
 * If the S case returns ~1 the curl path is sound and explanation (b) stands.
 * If it returns ~1/4 or ~4 the curl term is mis-scaled and every gamma=0
 * number measured on S-dominated data is wrong.
 *
 * RESULT.  S returns 0.9927 and P 1.0024 at 30 cells per S wavelength, at
 * every angle.  The residual is discretisation, not scaling: halving freq
 * shrinks it 4x (0.24% -> 0.06% -> 0.016% for P), so it is O((k dx)^2) and
 * converges to 1.  A mis-scaling would be a fixed factor and would not move.
 * Explanation (b) therefore stands -- the curl path is correctly scaled, and
 * the 0.2468 measured on the pure-S shot is V_0 nulling the P-S conversions
 * that dominate that adjoint field, which is the intended behaviour.
 *
 * WHAT IS FED IN.  The accumulators do not take physical fields directly:
 *   - the kinetic half forms sigma_a = C psi_sigma, so psi_sigma must be the
 *     STRAIN (eps_xx, eps_zz, 2 eps_xz) for C psi_sigma to be sigma;
 *   - the potential half forms b*psi_v, so psi_v must be rho*v.
 * Both are built analytically below.  fdelmodc's T_ij = -sigma_ij sign is
 * irrelevant here because every product is quadratic in the same field.
 *
 * Each array is sampled at ITS OWN staggered node:
 *   Txx,Tzz at (ix,iz)   Vx at (ix-1/2,iz)   Vz at (ix,iz-1/2)
 *   Txz at (ix-1/2,iz-1/2)
 * which is the registration the stencils in energy_imaging.c assume.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "par.h"
#include "fdelfwi.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NAX  200
#define NAZ  200
#define HALO 6

static float VP = 2600.0f, VS = 1495.0f, RHO = 2000.0f;
static float DX = 5.0f, DT = 5.0e-4f, FREQ = 10.0f, AMP = 1.0f;

/* docpkge.o (pulled in via $(OBJC)) references sdoc unconditionally. */
char *sdoc[] = {
" ",
" test_ei_planewave - analytic plane-wave calibration of the gamma=0 halves",
" ",
"   A single plane wave paired with itself gives K = V_0 pointwise for BOTH",
"   modes, so rms(P)/rms(K) = 1 exactly.  P exercises only the div path of",
"   V_0, S only the curl path.",
" ",
"   freq=10       plane-wave frequency in Hz.  The only knob: it sets the",
"                 grid sampling (Vs/freq/dx cells per S wavelength).  What",
"                 is left at freq=10 is discretisation, not a mis-scaling:",
"                 halving freq shrinks the residual 4x (measured 0.24% ->",
"                 0.06% -> 0.016%), i.e. it is O((k dx)^2) and converges to",
"                 1.  A mis-scaled curl term would be a constant factor",
"                 (1/4, 4, or a power of Vp/Vs) and would not move at all.",
" ",
"   file_out=     if set, also write <pfx>_<mode><angle>_{kin,pot}.su, the",
"                 two accumulator maps, for plot_ei_planewave.py.  Default",
"                 is no I/O at all: the test is eight numbers.",
" ",
NULL};

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

/* Set by file_out=; NULL means numbers only, no I/O. */
static char *ei_out = NULL;

/* rms over the region ei_bounds hands to ei_build_weight. */
static double rms_in(float *a, int io, int ie)
{
	double s = 0.0;
	int n = 0, ix, iz;
	for (ix = io; ix < ie; ix++)
		for (iz = io; iz < ie; iz++, n++)
			s += (double)a[ix*NAZ+iz] * a[ix*NAZ+iz];
	return n ? sqrt(s/n) : 0.0;
}

/*
 * One mode, one propagation direction.  Returns rms(P)/rms(K).
 *
 * mode 'P': polarisation p = n, wavenumber w/Vp.
 * mode 'S': polarisation p = (-nz,nx), wavenumber w/Vs.
 * In both cases eps_ij = (A/V) (n_i p_j + n_j p_i)/2 cos(phi), which reduces
 * to (A/Vp) n_i n_j cos(phi) for P and is trace-free for S.
 */
static double run_case(modPar *mod, bndPar *bnd, char mode, double ang_deg)
{
	size_t sz = (size_t)NAX*NAZ;
	float *vx, *vz, *txx, *tzz, *txz;
	float *pvx, *pvz, *pxx, *pzz, *pxz;
	float *oK, *oP;
	wflPar adj;
	double w = 2.0*M_PI*FREQ;
	double nx = cos(ang_deg*M_PI/180.0), nz = sin(ang_deg*M_PI/180.0);
	double px, pz, V, k, lam, mu, l2m, ratio;
	int ix, iz;

	mu  = RHO*VS*VS;
	l2m = RHO*VP*VP;
	lam = l2m - 2.0*mu;

	if (mode == 'P') { V = VP; px =  nx; pz = nz; }
	else             { V = VS; px = -nz; pz = nx; }
	k = w/V;

	vx  = calloc(sz,sizeof(float)); vz  = calloc(sz,sizeof(float));
	txx = calloc(sz,sizeof(float)); tzz = calloc(sz,sizeof(float));
	txz = calloc(sz,sizeof(float));
	pvx = calloc(sz,sizeof(float)); pvz = calloc(sz,sizeof(float));
	pxx = calloc(sz,sizeof(float)); pzz = calloc(sz,sizeof(float));
	pxz = calloc(sz,sizeof(float));
	oK  = calloc(sz,sizeof(float)); oP  = calloc(sz,sizeof(float));

	for (ix = 0; ix < NAX; ix++) {
		for (iz = 0; iz < NAZ; iz++) {
			size_t ig = (size_t)ix*NAZ + iz;
			double xP = ix*DX,         zP = iz*DX;
			double xV = (ix-0.5)*DX,   zV = (iz-0.5)*DX;
			double cP = cos(k*(nx*xP + nz*zP));          /* P node   */
			double cX = cos(k*(nx*xV + nz*zP));          /* Vx node  */
			double cZ = cos(k*(nx*xP + nz*zV));          /* Vz node  */
			double cT = cos(k*(nx*xV + nz*zV));          /* Txz node */
			double e = AMP/V;
			/* strain components, each at the node that needs them */
			double exx = e*nx*px*cP, ezz = e*nz*pz*cP;
			double gxz = e*(nx*pz + nz*px)*cT;           /* = 2 eps_xz */

			vx[ig]  = (float)(AMP*px*cX);
			vz[ig]  = (float)(AMP*pz*cZ);
			txx[ig] = (float)(l2m*exx + lam*ezz);
			tzz[ig] = (float)(lam*exx + l2m*ezz);
			txz[ig] = (float)(mu*gxz);

			/* psi_v = rho v  ->  b*psi_v = v */
			pvx[ig] = (float)(RHO*AMP*px*cX);
			pvz[ig] = (float)(RHO*AMP*pz*cZ);
			/* psi_sigma = strain  ->  C psi_sigma = sigma */
			pxx[ig] = (float)exx;
			pzz[ig] = (float)ezz;
			pxz[ig] = (float)gxz;
		}
	}

	memset(&adj, 0, sizeof(adj));
	adj.vx = pvx; adj.vz = pvz;
	adj.txx = pxx; adj.tzz = pzz; adj.txz = pxz;

	energyImagingInit(mod, 1, EI_GAMMA0, 0.0f, 0);
	energyImagingAccumAtSigma(mod, bnd, txx, tzz, txz, &adj, DT);
	energyImagingAccumAtVel(mod, bnd, vx, vz, &adj, DT);
	energyImagingGet(mod, bnd, EI_GET_KIN, oK);
	energyImagingGet(mod, bnd, EI_GET_POT, oP);
	ratio = rms_in(oP, HALO+2, NAX-HALO-2) / rms_in(oK, HALO+2, NAX-HALO-2);
	energyImagingFree();

	/* The whole NAX x NAZ array, halo included, so the figure shows that the
	 * halo is the ONLY place the two halves disagree. */
	if (ei_out) {
		char fn[1024];
		snprintf(fn, sizeof(fn), "%s_%c%02.0f_kin.su", ei_out, mode, ang_deg);
		writesufile(fn, oK, (size_t)NAZ, (size_t)NAX, 0.0f, 0.0f, DX, DX);
		snprintf(fn, sizeof(fn), "%s_%c%02.0f_pot.su", ei_out, mode, ang_deg);
		writesufile(fn, oP, (size_t)NAZ, (size_t)NAX, 0.0f, 0.0f, DX, DX);
	}

	free(vx); free(vz); free(txx); free(tzz); free(txz);
	free(pvx); free(pvz); free(pxx); free(pzz); free(pxz);
	free(oK); free(oP);
	return ratio;
}

int main(int argc, char **argv)
{
	modPar mod;
	bndPar bnd;
	size_t sz = (size_t)NAX*NAZ, i;
	double angs[] = {0.0, 30.0, 45.0, 90.0};
	int ia, bad = 0;

	initargs(argc, argv);
	requestdoc(0);
	getparfloat("freq", &FREQ);
	if (!getparstring("file_out", &ei_out)) ei_out = NULL;

	memset(&mod, 0, sizeof(mod));
	memset(&bnd, 0, sizeof(bnd));

	mod.iorder = 4; mod.ischeme = 3;
	mod.nax = NAX; mod.naz = NAZ;
	mod.nx = NAX;  mod.nz = NAZ;
	mod.dx = DX;   mod.dz = DX;  mod.dt = DT;
	mod.ioXx = mod.ioXz = mod.ioZx = mod.ioZz = HALO;
	mod.ioPx = mod.ioPz = mod.ioTx = mod.ioTz = HALO;
	mod.ieXx = mod.ieXz = mod.ieZx = mod.ieZz = NAX-HALO;
	mod.iePx = mod.iePz = mod.ieTx = mod.ieTz = NAX-HALO;

	/* ntap = 0, so ei_bounds leaves the loop bounds exactly as set above. */
	bnd.lef = bnd.rig = bnd.top = bnd.bot = 4;
	bnd.ntap = 0;

	mod.rox = malloc(sz*sizeof(float)); mod.roz = malloc(sz*sizeof(float));
	mod.l2m = malloc(sz*sizeof(float)); mod.lam = malloc(sz*sizeof(float));
	mod.muu = malloc(sz*sizeof(float));
	for (i = 0; i < sz; i++) {
		mod.rox[i] = mod.roz[i] = DT/(RHO*DX);
		mod.l2m[i] = RHO*VP*VP;
		mod.muu[i] = RHO*VS*VS;
		mod.lam[i] = mod.l2m[i] - 2.0f*mod.muu[i];
	}

	printf("plane-wave calibration of the gamma=0 halves\n");
	printf("  Vp=%g Vs=%g rho=%g  dx=%g  f=%g Hz  (%.1f cells per S wavelength)\n",
	       VP, VS, RHO, DX, FREQ, VS/FREQ/DX);
	printf("  a single analytic plane wave paired with itself:\n");
	printf("  P exercises the div path of V_0 only, S the curl path only.\n");
	printf("  EXACT prediction, both modes, every angle:  rms(P)/rms(K) = 1\n\n");
	printf("    %-6s %-8s %10s %10s\n", "mode", "angle", "ratio", "error");
	for (ia = 0; ia < (int)(sizeof(angs)/sizeof(angs[0])); ia++) {
		double rp = run_case(&mod, &bnd, 'P', angs[ia]);
		double rs = run_case(&mod, &bnd, 'S', angs[ia]);
		printf("    %-6s %6.0f   %10.5f %9.2f%%\n", "P", angs[ia],
		       rp, 100.0*(rp-1.0));
		printf("    %-6s %6.0f   %10.5f %9.2f%%\n", "S", angs[ia],
		       rs, 100.0*(rs-1.0));
		if (fabs(rp-1.0) > 0.02) bad++;
		if (fabs(rs-1.0) > 0.02) bad++;
	}
	printf("\n  %s\n", bad ? "FAIL -- at least one mode is mis-scaled"
	                       : "PASS -- both paths of V_0 agree with K to 2%");
	free(mod.rox); free(mod.roz); free(mod.l2m); free(mod.lam); free(mod.muu);
	return bad ? 1 : 0;
}
