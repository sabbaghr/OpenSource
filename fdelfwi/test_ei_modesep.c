/*
 * test_ei_modesep.c -- analytic two-plane-wave calibration of the
 * mode-separated accumulators (energy_gamma = EI_MODESEP = -1).
 *
 * WHY A SEPARATE TEST.  test_ei_planewave.c pairs ONE plane wave with ITSELF,
 * which pins the relative scaling of the two halves but cannot see the angle
 * dependence at all: self-pairing is the single point psi = 0.  The entire
 * claim of EI_MODESEP is about the angle, so it needs two waves.
 *
 * THE PREDICTION.  Build a snapshot (at t = 0, phase phi = k.x) of
 *
 *     v = A p cos(phi),   k = (w/V) n,   p = n (mode P) or (-nz,nx) (mode S)
 *
 * and the stresses from the matching strain.  Then a = b grad.sig evaluates
 * analytically to
 *
 *     a = -A w n sin(phi)   (P)        a = -A w p sin(phi)   (S)
 *
 * and the four Helmholtz scalars the accumulators form are
 *
 *     P mode:  D    = -A k sin(phi)     Ddot = -A w k cos(phi)   C = Cdot = 0
 *              grad D = -A k^2 n cos(phi)
 *     S mode:  C    =  A k sin(phi)     Cdot =  A w k cos(phi)   D = Ddot = 0
 *              grad C =  A k^2 n cos(phi)
 *
 * Pairing a forward wave along n_f with an adjoint wave along n_a of the SAME
 * mode, and writing psi for the angle between them:
 *
 *     K = dt Ddot_f Ddot_a = dt A^2 w^2 k^2          cos(phi_f) cos(phi_a)
 *     V = dt V^2 grad D_f . grad D_a
 *       = dt V^2 A^2 k^4 (n_f.n_a) cos(phi_f) cos(phi_a)
 *       = dt A^2 w^2 k^2 cos(psi)  cos(phi_f) cos(phi_a)        [V = w/k]
 *
 * so, with the velocity cancelling identically in both lines,
 *
 *                   V = cos(psi) * K     POINTWISE,
 *
 * for mode P at Vp and for mode S at Vs, with the SAME coefficient.  That is
 * the whole point of EI_MODESEP and the thing this test measures: at gamma = 2
 * the same ratio carries an isotropic modulation (lam -+ 2mu cos 2theta) that
 * changes sign near 59 deg, and at gamma = 0 it is clean only after the
 * (lam+2mu)/mu weights conspire.  Here there is no material parameter left to
 * get wrong.
 *
 * Two numbers are therefore reported per case, both exact, neither with a
 * tolerance to argue about:
 *
 *     r     = sum(V K) / sum(K K)                     =  cos(psi)
 *     abar  = sum|K+V| / sum(|K+V| + |K-V|)           =  cos^2(psi/2)
 *
 * r is signed, so it distinguishes psi from 180-psi -- which matters, because
 * the spatial phase cos(k.x) is EVEN in k and cannot.  The direction survives
 * through the polarisation and the acceleration instead: flipping n flips both
 * v and a, leaving K even and V odd, hence r -> -r.  A snapshot is therefore
 * enough to reach the backscattering limit psi = 180 deg, where r = -1 and
 * abar = 0.
 *
 * abar is the aggregated form of the weight ei_build_weight_modesep actually
 * builds, with the eps floor factored out: since V = rK pointwise,
 * |K+V| = |K||1+r| and |K-V| = |K||1-r|, so the ratio of the sums is
 * |1+r|/(|1+r|+|1-r|) = cos^2(psi/2) with no eps bias at the nodes where
 * cos(phi_f)cos(phi_a) passes through zero.  Comparing it against the mean of
 * the weight FIELD instead would be dominated by those nodes.
 *
 * MODE LEAKAGE.  A P wave is curl-free and an S wave divergence-free, so a
 * pure-P input must leave K_S = V_S = 0 and vice versa.  Reported as
 * rms(other channel)/rms(this channel).
 *
 * RESULT.  PASS at every angle and both reference directions.  Worst error at
 * 30 cells per S wavelength is 0.83% on r and 0.36% on a_tomo (mode S, the
 * more poorly sampled one).  The residual is discretisation, not scaling:
 * halving freq shrinks it 4x,
 *
 *     f (Hz)       20      10       5     2.5
 *     PP r_err  -1.10%  -0.28%  -0.07%  -0.02%
 *     SS r_err   3.29%   0.83%   0.21%   0.05%
 *
 * so it is O((k dx)^2) and converges on cos(psi).  A mis-scaled channel would
 * be a fixed factor and would not move at all.
 *
 * The mode leakage comes out at ROUNDOFF, 1e-11, not at O((k dx)^2) -- and at
 * the on-axis reference direction it is identically zero.  That is better than
 * the continuum argument alone promises, and it is a property of the staggered
 * operators: for a plane wave the discrete curl of the P field reduces to
 *
 *     curl v = -(A/dx) sin(phi) [ n_x Sig(k n_z dx) - n_z Sig(k n_x dx) ],
 *     Sig(q) = 2 sum_m c_m sin((2m-1)q/2),
 *
 * and Sig(q) = q + O(q^5) exactly for the 4th-order staggered coefficients
 * (c1 = 9/8, c2 = -1/24 kill the q^3 term identically).  The bracket is
 * therefore O((k dx)^5) * (n_x n_z^5 - n_z n_x^5), and each channel pairs TWO
 * such factors, so the leaked product is O((k dx)^10).  The two modes are
 * separated to machine precision, not merely to truncation order.
 *
 * THE P-S BLIND SPOT, which is the known limitation and is measured here
 * rather than asserted.  For a CONVERTED pair (forward P, adjoint S) all four
 * accumulators vanish: K_P needs Ddot_a, which an S wave does not have, and
 * K_S needs Cdot_f, which a P wave does not have.  So a converted pair
 * contributes to NEITHER channel -- it is dropped, not mis-angled.  That is a
 * different failure mode from gamma = 0, where a converted pair lands at
 * c2 = 1/2 exactly and the linear weight HALF-PASSES it into both outputs.
 * Reported as the residual relative to the unconverted PP pairing at the same
 * geometry.
 *
 * WHAT IS FED IN, and the one convention this test does NOT settle.  The
 * accumulators take the DUAL variables, not the fields:
 *   - the kinetic halves form sigma_a = C psi_sigma, so psi_sigma must be the
 *     STRAIN (eps_xx, eps_zz, 2 eps_xz);
 *   - the potential halves form b psi_v, so psi_v must be rho v.
 * Both are built analytically below, with NO relative sign flip between them
 * -- i.e. the adjoint is fed as a physical state read in the common loop time,
 * which is the frame in which psi = 0 is transmission.  Whether a real
 * adj_shot run delivers that frame is NOT testable here; it is the point of
 * the demo/gradient/test_rocha_fig9.sh calibration, where the direct-path
 * banana must land in a_tomo.  Getting it wrong EXCHANGES the two filters
 * without changing any number below.
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
/* rms/sum region, kept well inside the narrow EI_MODESEP box (HALO+half). */
#define EDGE (HALO+4)

static float VP = 2600.0f, VS = 1495.0f, RHO = 2000.0f;
static float DX = 5.0f, DT = 5.0e-4f, FREQ = 10.0f, AMP = 1.0f;

/* docpkge.o (pulled in via $(OBJC)) references sdoc unconditionally. */
char *sdoc[] = {
" ",
" test_ei_modesep - analytic angle calibration of the EI_MODESEP accumulators",
" ",
"   Two analytic plane waves at an opening angle psi give, for BOTH modes and",
"   with the velocity cancelling identically,",
" ",
"     V = cos(psi) * K   pointwise   ->   a_tomo = cos^2(psi/2)",
" ",
"   with no material parameter and no modulation left in it.  That exactness",
"   is what distinguishes energy_gamma=-1 from gamma=2 (modulated by",
"   (lam -+ 2mu cos2theta), sign change near 59 deg) and from gamma=0.",
" ",
"   Also measured: mode leakage (P into the S channel and back, which must be",
"   pure discretisation) and the P-S blind spot (a converted pair contributes",
"   to NEITHER channel -- the known limitation, quantified rather than",
"   asserted).",
" ",
"   freq=10       plane-wave frequency in Hz.  The only knob: it sets the grid",
"                 sampling (Vs/freq/dx cells per S wavelength).  Everything",
"                 reported is exact in the continuum, so every residual must",
"                 shrink as O((k dx)^2) when freq is halved.  A mis-scaling is",
"                 a fixed factor and does not move.",
" ",
"   tol=0.02      pass/fail tolerance on cos(psi) and cos^2(psi/2).",
" ",
"   dpsi=         if set, sweep psi = 0..180 in steps of dpsi instead of the",
"                 built-in 9 angles.  Use with file_out= for a smooth curve.",
" ",
"   file_out=     write the measured angle response as a CSV",
"                 (ref,mode,psi,r,cos_psi,a_tomo,cos2_half,leak) for",
"                 plot_ei_modesep.py.  Default is no I/O.",
" ",
NULL};

/*
 * Fill the physical state (vx,vz,txx,tzz,txz) and the matching DUAL variables
 * (pvx,pvz = rho v, and pxx,pzz,pxz = strain) for one plane wave.
 *
 * mode 'P': polarisation p = n, wavenumber w/Vp.
 * mode 'S': polarisation p = (-nz,nx), wavenumber w/Vs.
 * In both cases eps_ij = (A/V)(n_i p_j + n_j p_i)/2 cos(phi), which reduces to
 * (A/V) n_i n_j cos(phi) for P and is trace-free for S.
 */
static void build_wave(char mode, double ang_deg,
                       float *vx, float *vz,
                       float *txx, float *tzz, float *txz,
                       float *pvx, float *pvz,
                       float *pxx, float *pzz, float *pxz)
{
	double w = 2.0*M_PI*FREQ;
	double nx = cos(ang_deg*M_PI/180.0), nz = sin(ang_deg*M_PI/180.0);
	double px, pz, V, k, lam, mu, l2m;
	int ix, iz;

	mu  = RHO*VS*VS;
	l2m = RHO*VP*VP;
	lam = l2m - 2.0*mu;

	if (mode == 'P') { V = VP; px =  nx; pz = nz; }
	else             { V = VS; px = -nz; pz = nx; }
	k = w/V;

	for (ix = 0; ix < NAX; ix++) {
		for (iz = 0; iz < NAZ; iz++) {
			size_t ig = (size_t)ix*NAZ + iz;
			double xP = ix*DX,       zP = iz*DX;
			double xV = (ix-0.5)*DX, zV = (iz-0.5)*DX;
			double cP = cos(k*(nx*xP + nz*zP));          /* P node   */
			double cX = cos(k*(nx*xV + nz*zP));          /* Vx node  */
			double cZ = cos(k*(nx*xP + nz*zV));          /* Vz node  */
			double cT = cos(k*(nx*xV + nz*zV));          /* Txz node */
			double e = AMP/V;
			double exx = e*nx*px*cP, ezz = e*nz*pz*cP;
			double gxz = e*(nx*pz + nz*px)*cT;           /* = 2 eps_xz */

			if (vx)  vx[ig]  = (float)(AMP*px*cX);
			if (vz)  vz[ig]  = (float)(AMP*pz*cZ);
			if (txx) txx[ig] = (float)(l2m*exx + lam*ezz);
			if (tzz) tzz[ig] = (float)(lam*exx + l2m*ezz);
			if (txz) txz[ig] = (float)(mu*gxz);

			/* psi_v = rho v   ->   b*psi_v = v */
			if (pvx) pvx[ig] = (float)(RHO*AMP*px*cX);
			if (pvz) pvz[ig] = (float)(RHO*AMP*pz*cZ);
			/* psi_sigma = strain   ->   C psi_sigma = sigma */
			if (pxx) pxx[ig] = (float)exx;
			if (pzz) pzz[ig] = (float)ezz;
			if (pxz) pxz[ig] = (float)gxz;
		}
	}
}

static double rms_in(float *a)
{
	double s = 0.0;
	int n = 0, ix, iz;
	for (ix = EDGE; ix < NAX-EDGE; ix++)
		for (iz = EDGE; iz < NAZ-EDGE; iz++, n++)
			s += (double)a[ix*NAZ+iz] * a[ix*NAZ+iz];
	return n ? sqrt(s/n) : 0.0;
}

/* Signed pointwise ratio V/K, least-squares over the interior: = cos(psi). */
static double signed_ratio(float *K, float *V)
{
	double num = 0.0, den = 0.0;
	int ix, iz;
	for (ix = EDGE; ix < NAX-EDGE; ix++) {
		for (iz = EDGE; iz < NAZ-EDGE; iz++) {
			int ig = ix*NAZ+iz;
			num += (double)V[ig] * K[ig];
			den += (double)K[ig] * K[ig];
		}
	}
	return (den > 0.0) ? num/den : 0.0;
}

/* Aggregated tomographic weight, eps-free: = cos^2(psi/2). */
static double abar(float *K, float *V)
{
	double t = 0.0, m = 0.0;
	int ix, iz;
	for (ix = EDGE; ix < NAX-EDGE; ix++) {
		for (iz = EDGE; iz < NAZ-EDGE; iz++) {
			int ig = ix*NAZ+iz;
			t += fabs((double)K[ig] + V[ig]);
			m += fabs((double)K[ig] - V[ig]);
		}
	}
	return (t+m > 0.0) ? t/(t+m) : 0.0;
}

/*
 * Run one forward/adjoint plane-wave pairing through the EI_MODESEP
 * accumulators and hand back the four channel maps.
 */
static void run_pair(modPar *mod, bndPar *bnd,
                     char mf, double af, char ma, double aa,
                     float *KP, float *VP_, float *KS, float *VS_)
{
	size_t sz = (size_t)NAX*NAZ;
	float *vx, *vz, *txx, *tzz, *txz;
	float *pvx, *pvz, *pxx, *pzz, *pxz;
	wflPar adj;

	vx  = calloc(sz,sizeof(float)); vz  = calloc(sz,sizeof(float));
	txx = calloc(sz,sizeof(float)); tzz = calloc(sz,sizeof(float));
	txz = calloc(sz,sizeof(float));
	pvx = calloc(sz,sizeof(float)); pvz = calloc(sz,sizeof(float));
	pxx = calloc(sz,sizeof(float)); pzz = calloc(sz,sizeof(float));
	pxz = calloc(sz,sizeof(float));

	/* forward: physical half only; adjoint: dual half only. */
	build_wave(mf, af, vx, vz, txx, tzz, txz, NULL,NULL,NULL,NULL,NULL);
	build_wave(ma, aa, NULL,NULL,NULL,NULL,NULL, pvx, pvz, pxx, pzz, pxz);

	memset(&adj, 0, sizeof(adj));
	adj.vx  = pvx; adj.vz  = pvz;
	adj.txx = pxx; adj.tzz = pzz; adj.txz = pxz;

	energyImagingInit(mod, EI_TOMO, EI_MODESEP, 0.0f, 0);
	energyImagingAccumAtSigma(mod, bnd, txx, tzz, txz, &adj, DT);
	energyImagingAccumAtVel(mod, bnd, vx, vz, &adj, DT);
	energyImagingGet(mod, bnd, EI_GET_KIN,   KP);
	energyImagingGet(mod, bnd, EI_GET_POT,   VP_);
	energyImagingGet(mod, bnd, EI_GET_KIN_S, KS);
	energyImagingGet(mod, bnd, EI_GET_POT_S, VS_);
	energyImagingFree();

	free(vx); free(vz); free(txx); free(tzz); free(txz);
	free(pvx); free(pvz); free(pxx); free(pzz); free(pxz);
}

int main(int argc, char **argv)
{
	modPar mod;
	bndPar bnd;
	size_t sz = (size_t)NAX*NAZ, i;
	double fixed[] = {0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0};
	double refs[] = {0.0, 37.0};
	double *psis = fixed;
	int np = (int)(sizeof(fixed)/sizeof(fixed[0]));
	int nr = (int)(sizeof(refs)/sizeof(refs[0]));
	float *KP, *VPp, *KS, *VSp;
	double tol = 0.02, dpsi = 0.0;
	char *csv_name = NULL;
	FILE *csv = NULL;
	int ip, ir, bad = 0;

	initargs(argc, argv);
	requestdoc(0);
	getparfloat("freq", &FREQ);
	{ float t; if (getparfloat("tol", &t)) tol = t; }
	{ float t; if (getparfloat("dpsi", &t)) dpsi = t; }
	if (!getparstring("file_out", &csv_name)) csv_name = NULL;

	if (dpsi > 0.0) {
		np = (int)(180.0/dpsi) + 1;
		psis = malloc((size_t)np * sizeof(double));
		for (ip = 0; ip < np; ip++) psis[ip] = ip*dpsi;
	}
	if (csv_name) {
		csv = fopen(csv_name, "w");
		if (!csv) { fprintf(stderr, "cannot write %s\n", csv_name); return 2; }
		fprintf(csv, "ref,mode,psi,r,cos_psi,a_tomo,cos2_half,leak\n");
	}

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

	KP  = calloc(sz,sizeof(float)); VPp = calloc(sz,sizeof(float));
	KS  = calloc(sz,sizeof(float)); VSp = calloc(sz,sizeof(float));

	printf("EI_MODESEP angle calibration (energy_gamma=-1)\n");
	printf("  Vp=%g Vs=%g rho=%g  dx=%g  f=%g Hz"
	       "  (%.1f cells per P wavelength, %.1f per S)\n",
	       VP, VS, RHO, DX, FREQ, VP/FREQ/DX, VS/FREQ/DX);
	printf("  two analytic plane waves of the same mode at an opening angle psi.\n");
	printf("  EXACT prediction, BOTH modes, no Vp/Vs dependence:\n");
	printf("      r = V/K = cos(psi)          a_tomo = cos^2(psi/2)\n\n");

	for (ir = 0; ir < nr; ir++) {
		printf("  forward direction n_f = %g deg\n", refs[ir]);
		printf("    %-4s %6s | %9s %9s %8s | %9s %9s %8s | %8s\n",
		       "mode", "psi", "r", "cos psi", "err", "a_tomo",
		       "cos^2", "err", "leak");
		for (ip = 0; ip < np; ip++) {
			double psi = psis[ip];
			double cps = cos(psi*M_PI/180.0);
			double ct2 = 0.5*(1.0+cps);              /* cos^2(psi/2) */
			double rP, rS, aP, aS, lkP, lkS;

			run_pair(&mod, &bnd, 'P', refs[ir], 'P', refs[ir]+psi,
			         KP, VPp, KS, VSp);
			rP  = signed_ratio(KP, VPp);
			aP  = abar(KP, VPp);
			lkP = rms_in(KP) > 0.0 ? rms_in(KS)/rms_in(KP) : 0.0;

			run_pair(&mod, &bnd, 'S', refs[ir], 'S', refs[ir]+psi,
			         KP, VPp, KS, VSp);
			rS  = signed_ratio(KS, VSp);
			aS  = abar(KS, VSp);
			lkS = rms_in(KS) > 0.0 ? rms_in(KP)/rms_in(KS) : 0.0;

			printf("    %-4s %6.0f | %9.5f %9.5f %7.2f%% |"
			       " %9.5f %9.5f %7.2f%% | %8.1e\n",
			       "PP", psi, rP, cps, 100.0*(rP-cps),
			       aP, ct2, 100.0*(aP-ct2), lkP);
			printf("    %-4s %6.0f | %9.5f %9.5f %7.2f%% |"
			       " %9.5f %9.5f %7.2f%% | %8.1e\n",
			       "SS", psi, rS, cps, 100.0*(rS-cps),
			       aS, ct2, 100.0*(aS-ct2), lkS);

			if (csv) {
				fprintf(csv, "%g,PP,%g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
				        refs[ir], psi, rP, cps, aP, ct2, lkP);
				fprintf(csv, "%g,SS,%g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
				        refs[ir], psi, rS, cps, aS, ct2, lkS);
			}

			if (fabs(rP-cps) > tol || fabs(aP-ct2) > tol) bad++;
			if (fabs(rS-cps) > tol || fabs(aS-ct2) > tol) bad++;
			if (lkP > tol || lkS > tol) bad++;
		}
		printf("\n");
	}

	/*
	 * The P-S blind spot.  A converted pair has Ddot_a = 0 and Cdot_f = 0, so
	 * every channel vanishes and the pair is DROPPED.  Measured against the
	 * unconverted PP pairing at the same geometry, which is the amplitude it
	 * would have had if it were seen at all.
	 */
	{
		double psi = 45.0, ref = 0.0;
		double nPP, nSS, cKP, cVP, cKS, cVS;

		run_pair(&mod, &bnd, 'P', ref, 'P', ref+psi, KP, VPp, KS, VSp);
		nPP = rms_in(KP);
		run_pair(&mod, &bnd, 'S', ref, 'S', ref+psi, KP, VPp, KS, VSp);
		nSS = rms_in(KS);

		run_pair(&mod, &bnd, 'P', ref, 'S', ref+psi, KP, VPp, KS, VSp);
		cKP = rms_in(KP); cVP = rms_in(VPp);
		cKS = rms_in(KS); cVS = rms_in(VSp);

		printf("  P-S blind spot (forward P, adjoint S, psi=%g deg)\n", psi);
		printf("    a converted pair must contribute to NEITHER channel;\n");
		printf("    residuals relative to the unconverted pairing:\n");
		printf("      K_P %.2e   V_P %.2e   (vs PP %.2e)\n",
		       cKP/nPP, cVP/nPP, 1.0);
		printf("      K_S %.2e   V_S %.2e   (vs SS %.2e)\n",
		       cKS/nSS, cVS/nSS, 1.0);
		printf("    NOTE this is the documented limitation, not a bug: the\n");
		printf("    conversion is dropped, not mis-angled.  gamma=0 instead\n");
		printf("    puts it at c2=1/2 and half-passes it into both outputs.\n\n");
		if (cKP/nPP > tol || cVP/nPP > tol ||
		    cKS/nSS > tol || cVS/nSS > tol) bad++;
	}

	printf("  %s\n", bad ? "FAIL -- see the rows above"
	                     : "PASS -- cos(psi) and cos^2(psi/2) recovered at "
	                       "every angle, both modes");

	if (csv) { fclose(csv); printf("  wrote %s\n", csv_name); }
	if (psis != fixed) free(psis);
	free(KP); free(VPp); free(KS); free(VSp);
	free(mod.rox); free(mod.roz); free(mod.l2m); free(mod.lam); free(mod.muu);
	return bad ? 1 : 0;
}
