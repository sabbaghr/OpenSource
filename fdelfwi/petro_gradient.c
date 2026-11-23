#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "par.h"
#include "fdelfwi.h"

/*--------------------------------------------------------------------
 * petro_gradient -- add the petrophysical-constraint gradient a*g_P to an
 *                   EXISTING data gradient, with no wave propagation.
 *
 * J_P = a * sum_x D(m) (Aragao & Sava 2020, eq 16-20; petro_constraint.c)
 * depends on the model only, never on the data.  So the gradient of the
 * constrained objective J_D + J_P at a model m is simply
 *
 *     g_total = g_data(m) + a * g_P(m)
 *
 * and an already-computed data gradient (e.g. the stacked conventional
 * gradient of a lag experiment) can be combined with g_P directly.  This is
 * exactly the first gradient fwi_inversion would hand the optimizer with
 * petro_file= on -- same petroBuild / petroCostGrad / petroAccumGradient,
 * same water mask -- minus the inversion's scaling, which is not part of the
 * gradient itself.
 *
 * The gradients are in the VELOCITY parameterization (param=2: per m/s for
 * Vp and Vs, per kg/m^3 for rho), which is what test_lag_gradient writes as
 * *_{vp,vs,rho}_raw.su.
 *
 * TRADE-OFF WEIGHT 'a'.  Either fixed (petro_a=) or calibrated
 * (petro_balance=) so that ||a g_P|| = balance * ||g_data|| measured on ONE
 * parameter block, calib_block= (1=Vp, 2=Vs [default], 3=rho), or on all
 * three blocks weighted by dm_vp/dm_vs/dm_rho (calib_block=0) -- the rule
 * fwi_inversion applies with scaling=1, where dm is the bound range.  A
 * single block is unit-consistent by construction (both norms are per m/s,
 * say), so it needs no bounds; the raw all-block norm is NOT offered,
 * because it mixes per-(m/s) and per-(kg/m^3) entries and its value
 * depends on the units.
 *
 * Output (out_base=petro):
 *   <out_base>_P_{vp,vs,rho}.su       a * g_P      (the constraint's part)
 *   <out_base>_total_{vp,vs,rho}.su   g_data + a * g_P
 *   <out_base>_D.su                   D(m), crossplot units (GPa | km/s)
 * Headers are copied from the input gradient files.
 *--------------------------------------------------------------------*/

char *sdoc[] = {
" ",
" petro_gradient - data gradient + petrophysical-constraint gradient a*g_P",
" ",
" petro_gradient file_cp= file_cs= file_den= file_gvp= file_gvs= file_grho=",
"                petro_file= [optional parameters]",
" ",
" Required:",
"   file_cp=, file_cs=, file_den=   model at which the gradient was taken",
"   file_gvp=, file_gvs=, file_grho= data gradient, param=2 (Vp, Vs, rho)",
"   petro_file=        well-log crossplot, one pair per line, in the units of",
"                      petro_space",
" Optional:",
"   petro_space=1      1=(lambda,mu) GPa, 2=(Vp,Vs) km/s",
"   petro_balance=0.1  calibrate a: ||a g_P|| = balance*||g_data|| on calib_block",
"   petro_a=0          fixed a instead (overrides petro_balance)",
"   calib_block=2      1=Vp, 2=Vs, 3=rho, 0=all blocks weighted by dm_*",
"   dm_vp=,dm_vs=,dm_rho=  block weights for calib_block=0 (fwi_inversion's",
"                      scaling=1 bound ranges, vp_max-vp_min etc.)",
"   petro_nlam=100 petro_nmu=100 petro_smooth=1 petro_eps=0   PDF, as fwi_inversion",
"   out_base=petro     output prefix",
"   verbose=1",
" ",
NULL};

/* Read an SU file of ntr traces of ns samples: returns the samples
 * [ntr*ns] and, if hdr != NULL, the raw 240-byte headers [ntr*240]. */
static float *readSUraw(const char *fn, int *ntr, int *ns, unsigned char **hdr)
{
	FILE  *fp = fopen(fn, "rb");
	long   size;
	unsigned short ns16;
	unsigned char  h[240];
	float *d;
	unsigned char *H = NULL;
	int    n, i;

	if (!fp) verr("cannot open %s", fn);
	if (fread(h, 1, 240, fp) != 240) verr("%s: short header", fn);
	memcpy(&ns16, h + 114, 2);
	*ns = (int)ns16;
	fseek(fp, 0L, SEEK_END);
	size = ftell(fp);
	if (*ns <= 0 || size % (240 + 4L * *ns) != 0)
		verr("%s: size %ld is not a multiple of 240+4*ns (ns=%d)", fn, size, *ns);
	n = (int)(size / (240 + 4L * *ns));
	*ntr = n;
	rewind(fp);

	d = (float *)malloc((size_t)n * *ns * sizeof(float));
	if (hdr) H = (unsigned char *)malloc((size_t)n * 240);
	for (i = 0; i < n; i++) {
		if (fread(h, 1, 240, fp) != 240 ||
		    fread(d + (size_t)i * *ns, sizeof(float), *ns, fp) != (size_t)*ns)
			verr("%s: read error at trace %d", fn, i);
		if (H) memcpy(H + (size_t)i * 240, h, 240);
	}
	fclose(fp);
	if (hdr) *hdr = H;
	return d;
}

static void writeSUraw(const char *fn, const float *d, int ntr, int ns,
                       const unsigned char *hdr)
{
	FILE *fp = fopen(fn, "wb");
	int   i;
	if (!fp) verr("cannot write %s", fn);
	for (i = 0; i < ntr; i++) {
		fwrite(hdr + (size_t)i * 240, 1, 240, fp);
		fwrite(d + (size_t)i * ns, sizeof(float), ns, fp);
	}
	fclose(fp);
}

static double blocknorm(const float *g, int n)
{
	double s = 0.0;
	int i;
	for (i = 0; i < n; i++) s += (double)g[i] * g[i];
	return sqrt(s);
}

int main(int argc, char **argv)
{
	char  *fcp, *fcs, *fro, *fg[3], *petro_file, *out_base;
	int    space, nlam, nmu, calib, verbose, ntr, ns, ntr2, ns2, nmodel, i, k;
	float  smooth, eps, balance, a = 0.0f, dm[3];
	float *cp, *cs, *ro, *g[3], *xlame, *g1, *g2, *gp, *Dmap;
	unsigned char *hdr;
	int   *wmask = NULL, nwater = 0;
	petroPar pp;
	double dsum, nd, np;
	const char *pn[3] = {"vp", "vs", "rho"};
	char   fname[1024];

	initargs(argc, argv);
	requestdoc(1);

	if (!getparstring("file_cp", &fcp) || !getparstring("file_cs", &fcs) ||
	    !getparstring("file_den", &fro))
		verr("file_cp=, file_cs= and file_den= are required");
	if (!getparstring("file_gvp", &fg[0]) || !getparstring("file_gvs", &fg[1]) ||
	    !getparstring("file_grho", &fg[2]))
		verr("file_gvp=, file_gvs= and file_grho= are required");
	if (!getparstring("petro_file", &petro_file)) verr("petro_file= is required");
	if (!getparint("petro_space", &space)) space = PETRO_SPACE_LAMMU;
	if (!getparint("petro_nlam", &nlam)) nlam = 100;
	if (!getparint("petro_nmu", &nmu)) nmu = 100;
	if (!getparfloat("petro_smooth", &smooth)) smooth = 1.0f;
	if (!getparfloat("petro_eps", &eps)) eps = 0.0f;
	if (!getparfloat("petro_balance", &balance)) balance = 0.1f;
	getparfloat("petro_a", &a);
	if (!getparint("calib_block", &calib)) calib = 2;
	if (!getparfloat("dm_vp", &dm[0])) dm[0] = 0.0f;
	if (!getparfloat("dm_vs", &dm[1])) dm[1] = 0.0f;
	if (!getparfloat("dm_rho", &dm[2])) dm[2] = 0.0f;
	if (!getparstring("out_base", &out_base)) out_base = "petro";
	if (!getparint("verbose", &verbose)) verbose = 1;

	if (space != PETRO_SPACE_LAMMU && space != PETRO_SPACE_VPVS)
		verr("petro_space must be 1 or 2");
	if (calib < 0 || calib > 3) verr("calib_block must be 0..3");
	if (a <= 0.0f && calib == 0 && (dm[0] <= 0.0f || dm[1] <= 0.0f || dm[2] <= 0.0f))
		verr("calib_block=0 needs dm_vp=, dm_vs=, dm_rho= > 0");

	/* --- model and data gradient, all on the same grid --- */
	cp = readSUraw(fcp, &ntr, &ns, NULL);
	cs = readSUraw(fcs, &ntr2, &ns2, NULL);
	if (ntr2 != ntr || ns2 != ns) verr("%s and %s differ in size", fcp, fcs);
	ro = readSUraw(fro, &ntr2, &ns2, NULL);
	if (ntr2 != ntr || ns2 != ns) verr("%s and %s differ in size", fcp, fro);
	for (k = 0; k < 3; k++) {
		g[k] = readSUraw(fg[k], &ntr2, &ns2, (k == 0) ? &hdr : NULL);
		if (ntr2 != ntr || ns2 != ns)
			verr("%s is %d x %d, model is %d x %d", fg[k], ntr2, ns2, ntr, ns);
	}
	nmodel = ntr * ns;

	/* model -> [lambda | mu | rho] in Pa, the form petroCostGrad takes */
	xlame = (float *)malloc((size_t)3 * nmodel * sizeof(float));
	for (i = 0; i < nmodel; i++) {
		xlame[i]            = ro[i] * (cp[i]*cp[i] - 2.0f*cs[i]*cs[i]);
		xlame[nmodel + i]   = ro[i] * cs[i]*cs[i];
		xlame[2*nmodel + i] = ro[i];
		if (cs[i] == 0.0f) nwater++;
	}
	/* same water mask fwi_inversion builds: Vs == 0 is not a rock */
	if (nwater > 0) {
		wmask = (int *)calloc(nmodel, sizeof(int));
		for (i = 0; i < nmodel; i++) wmask[i] = (cs[i] == 0.0f);
	}

	if (petroBuild(petro_file, space, nlam, nmu, smooth, eps, &pp, verbose))
		verr("petroBuild failed");

	/* --- g_P at a = 1, in the velocity parameterization --- */
	g1   = (float *)malloc((size_t)nmodel * sizeof(float));
	g2   = (float *)malloc((size_t)nmodel * sizeof(float));
	gp   = (float *)calloc((size_t)3 * nmodel, sizeof(float));
	dsum = petroCostGrad(xlame, nmodel, &pp, wmask, g1, g2);
	petroAccumGradient(gp, nmodel, 2, xlame, g1, g2, 1.0f, &pp);
	if (wmask)
		for (k = 0; k < 3; k++)
			for (i = 0; i < nmodel; i++) if (wmask[i]) gp[k*nmodel + i] = 0.0f;

	/* D(m) map for QC: recompute per point (cheap, one model) */
	Dmap = (float *)calloc((size_t)nmodel, sizeof(float));
	{
		float *one = (float *)malloc(3 * sizeof(float));
		float  t1, t2;
		for (i = 0; i < nmodel; i++) {
			if (wmask && wmask[i]) continue;
			one[0] = xlame[i]; one[1] = xlame[nmodel+i]; one[2] = xlame[2*nmodel+i];
			Dmap[i] = (float)petroCostGrad(one, 1, &pp, NULL, &t1, &t2);
		}
		free(one);
	}

	/* --- trade-off weight --- */
	if (a > 0.0f) {
		vmess("fixed petro_a=%.4e", a);
	} else {
		nd = np = 0.0;
		for (k = 0; k < 3; k++) {
			double w;
			if (calib != 0 && k != calib - 1) continue;
			w  = (calib == 0) ? (double)dm[k] : 1.0;
			nd += w*w * blocknorm(g[k], nmodel) * blocknorm(g[k], nmodel);
			np += w*w * blocknorm(gp + (size_t)k*nmodel, nmodel)
			          * blocknorm(gp + (size_t)k*nmodel, nmodel);
		}
		nd = sqrt(nd); np = sqrt(np);
		if (np <= 0.0) verr("||g_P|| = 0 on the calibration block: nothing to add");
		a = (float)(balance * nd / np);
		vmess("calibrated a=%.4e on %s: ||g_data||=%.4e ||g_P||=%.4e balance=%.3f",
		      a, (calib == 0) ? "all blocks (dm-weighted)" : pn[calib-1],
		      nd, np, balance);
	}

	/* --- report per block --- */
	vmess("sum_x D = %.6e %s over %d points (mean D = %.4e), J_P = a*sum D = %.6e",
	      dsum, (space == PETRO_SPACE_VPVS) ? "km/s" : "GPa", nmodel - nwater,
	      dsum / (double)(nmodel - nwater), a * dsum);
	vmess("  block   ||g_data||      ||a g_P||       ratio     cos(g_data, g_P)");
	for (k = 0; k < 3; k++) {
		double gd = blocknorm(g[k], nmodel);
		double gq = a * blocknorm(gp + (size_t)k*nmodel, nmodel);
		double dot = 0.0;
		for (i = 0; i < nmodel; i++) dot += (double)g[k][i] * gp[(size_t)k*nmodel + i];
		vmess("  %-4s   %12.4e   %12.4e   %9.4f   %9.4f", pn[k], gd, gq,
		      gd > 0.0 ? gq / gd : 0.0,
		      (gd > 0.0 && gq > 0.0) ? a * dot / (gd * gq) : 0.0);
	}

	/* --- write --- */
	for (k = 0; k < 3; k++) {
		float *tmp = (float *)malloc((size_t)nmodel * sizeof(float));
		for (i = 0; i < nmodel; i++) tmp[i] = a * gp[(size_t)k*nmodel + i];
		snprintf(fname, sizeof(fname), "%s_P_%s.su", out_base, pn[k]);
		writeSUraw(fname, tmp, ntr, ns, hdr);
		for (i = 0; i < nmodel; i++) tmp[i] += g[k][i];
		snprintf(fname, sizeof(fname), "%s_total_%s.su", out_base, pn[k]);
		writeSUraw(fname, tmp, ntr, ns, hdr);
		free(tmp);
	}
	snprintf(fname, sizeof(fname), "%s_D.su", out_base);
	writeSUraw(fname, Dmap, ntr, ns, hdr);
	vmess("wrote %s_{P,total}_{vp,vs,rho}.su and %s_D.su", out_base, out_base);

	petroFree(&pp);
	free(cp); free(cs); free(ro); free(xlame); free(g1); free(g2); free(gp);
	free(Dmap); free(hdr); free(wmask);
	for (k = 0; k < 3; k++) free(g[k]);
	return 0;
}
