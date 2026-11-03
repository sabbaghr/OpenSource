#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"

/**
 * test_lag_cone.c -- Standalone driver for the lag-cube cone filter.
 *
 * Two jobs, and neither of them runs a wave equation:
 *
 *  1. `selftest=1` checks the FFT round-trip normalisation, the only part of
 *     lag_cone_filter.c that depends on how genfft was built.
 *
 *  2. Given a lag cube written by lag_gradient.c (<base>_lagvp.su plus
 *     <base>_lagaxis.txt), apply lagConeFilter and write the three panels
 *     <out>_passed.su, <out>_rejected.su and <out>_tapered.su.
 *
 * Purpose (2) exists so the C filter can be compared sample by sample
 * against demo/gradient/lag_slope_filter.py, which is the validated
 * reference.  The padded transform lengths actually used are printed, so the
 * Python run can be made to mirror them exactly -- otherwise the two would
 * differ merely because they padded to different lengths.
 *
 * This driver reads SU with a deliberately minimal reader: the cube is
 * nlag*nx fixed-length traces of nz samples, all geometry comes from the
 * companion lagaxis.txt, and nothing in the trace headers is trusted.
 */

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

char *sdoc[] = {
"  ",
" test_lag_cone - cone-filter a time-lag gradient cube",
"  ",
" test_lag_cone base= [optional parameters]",
"  ",
" Required parameters:",
"  ",
"   base=........... stem of the cube: reads base_lagvp.su (or _lagvs.su",
"                    with mode=vs) and base_lagaxis.txt",
"  ",
" Optional parameters:",
"  ",
"   selftest=0 ..... 1: only run the FFT normalisation self-test and exit",
"   mode=vp ........ which cube to filter: vp or vs",
"   out=<base>_cone  stem for the output panels",
"  ",
"   Cone response (slope in ms/m; a SMALL slope is a wide opening angle,",
"   i.e. the tomographic kernel):",
"   s_pass=0.20 .... H = 1 at or below this slope [ms/m]",
"   s_stop=0.30 .... H = 0 at or above this slope [ms/m]",
"  ",
"   ... or specify the cone as an opening angle instead, which converts",
"   through |p| = cos(theta)/v and overrides s_pass/s_stop:",
"   theta_pass=.... full opening angle 2theta kept in full [deg]",
"   theta_stop=.... full opening angle fully rejected [deg]",
"   v=.............. velocity used for the angle->slope conversion [m/s]",
"  ",
"   tap_tau=0.10 ... raised-cosine edge fraction on the lag axis",
"   tap_x=0.10 ..... ... on x  (NOTE: attenuates the cube near the edges)",
"   tap_z=0.10 ..... ... on z",
"   pad=2 .......... minimum zero-pad factor per axis",
"   nu0_keep=0 ..... 1: pass the whole nu=0 plane instead of its k=0 limit",
"   verbose=1 ...... verbosity",
"  ",
" The outputs satisfy passed + rejected = tapered exactly.  Compare a",
" filtered panel against TAPERED, never against the raw input, or the edge",
" taper gets credited to the filter.",
"  ",
" Author: lag-domain migration/tomography separation, 2026",
"  ",
NULL};


/***********************************************************************
 * readAxis -- Pull one named scalar out of a lagaxis.txt.
 ***********************************************************************/
static int readAxis(const char *fn, const char *key, double *val)
{
	FILE *fp = fopen(fn, "r");
	char line[512], name[128];
	double v;

	if (!fp) verr("test_lag_cone: cannot open %s", fn);
	while (fgets(line, sizeof(line), fp)) {
		if (line[0] == '#') continue;
		if (sscanf(line, "%127s %lf", name, &v) == 2) {
			if (strcmp(name, key) == 0) { *val = v; fclose(fp); return 1; }
		}
	}
	fclose(fp);
	return 0;
}

static double axisGet(const char *fn, const char *key)
{
	double v;
	if (!readAxis(fn, key, &v))
		verr("test_lag_cone: %s does not contain '%s'.", fn, key);
	return v;
}


/***********************************************************************
 * readCube -- Read ntrace fixed-length SU traces of ns samples.
 *
 * Minimal on purpose: the geometry is taken from lagaxis.txt, so the only
 * thing required of the file is that it hold exactly the expected number of
 * samples behind 240-byte headers.
 ***********************************************************************/
static void readCube(const char *fn, float *out, size_t ntrace, size_t ns)
{
	FILE *fp = fopen(fn, "rb");
	char hdr[240];
	size_t i;

	if (!fp) verr("test_lag_cone: cannot open %s", fn);
	for (i = 0; i < ntrace; i++) {
		if (fread(hdr, 1, 240, fp) != 240)
			verr("test_lag_cone: %s ended after %zu of %zu trace headers.",
			     fn, i, ntrace);
		if (fread(out + i*ns, sizeof(float), ns, fp) != ns)
			verr("test_lag_cone: %s ended inside trace %zu (expected %zu "
			     "samples).", fn, i, ns);
	}
	if (fgetc(fp) != EOF)
		vwarn("test_lag_cone: %s has data beyond the expected %zu traces; "
		      "the extra is ignored.", fn, ntrace);
	fclose(fp);
}


int main(int argc, char **argv)
{
	char *base, *out, *mode, fname[1024], axisfn[1024], oname[1024];
	int   nlag, nx, nz, verbose, selftest, i;
	float dx, dz, dlag, x0, z0;
	float theta_pass, theta_stop, v;
	float *cube, *passed, *tapered, *rejected;
	double dv;
	size_t ncube;
	lagConePar p;
	lagConeStats st;

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 1;
	if (!getparint("selftest", &selftest)) selftest = 0;

	if (selftest) {
		float err = lagConeSelfTest(1);
		vmess("round-trip relative error %.3e -- %s", err,
		      err < 1.0e-5f ? "PASS" : "FAIL");
		return (err < 1.0e-5f) ? 0 : 1;
	}

	if (!getparstring("base", &base))
		verr("test_lag_cone: base= is required (or run with selftest=1).");
	if (!getparstring("mode", &mode)) mode = "vp";
	if (strcmp(mode, "vp") != 0 && strcmp(mode, "vs") != 0)
		verr("test_lag_cone: mode must be vp or vs, got '%s'.", mode);

	snprintf(axisfn, sizeof(axisfn), "%s_lagaxis.txt", base);
	snprintf(fname,  sizeof(fname),  "%s_lag%s.su", base, mode);

	nlag = (int)axisGet(axisfn, "nlag");
	nx   = (int)axisGet(axisfn, "nx");
	nz   = (int)axisGet(axisfn, "nz");
	dlag = (float)axisGet(axisfn, "dlag");
	dx   = (float)axisGet(axisfn, "dx");
	dz   = (float)axisGet(axisfn, "dz");
	x0   = readAxis(axisfn, "x0", &dv) ? (float)dv : 0.0f;
	z0   = readAxis(axisfn, "z0", &dv) ? (float)dv : 0.0f;

	if (!getparstring("out", &out)) {
		snprintf(oname, sizeof(oname), "%s_cone", base);
		out = oname;
	}

	lagConeDefaults(&p);
	if (!getparfloat("s_pass",  &p.s_pass))  ;
	if (!getparfloat("s_stop",  &p.s_stop))  ;
	if (!getparfloat("tap_tau", &p.tap_tau)) ;
	if (!getparfloat("tap_x",   &p.tap_x))   ;
	if (!getparfloat("tap_z",   &p.tap_z))   ;
	if (!getparint  ("pad",     &p.pad))     ;
	if (!getparint  ("nu0_keep",&p.nu0_keep));

	/* An opening angle, if given, overrides the slopes. */
	if (getparfloat("theta_pass", &theta_pass) &&
	    getparfloat("theta_stop", &theta_stop)) {
		if (!getparfloat("v", &v))
			verr("test_lag_cone: theta_pass/theta_stop need v= (the velocity "
			     "that turns an angle into a slope).");
		if (!(theta_stop < theta_pass))
			verr("test_lag_cone: need theta_stop < theta_pass (a WIDER angle "
			     "is a SMALLER slope), got %g and %g deg.",
			     theta_pass, theta_stop);
		p.s_pass = lagConeAngleToSlope(theta_pass, v);
		p.s_stop = lagConeAngleToSlope(theta_stop, v);
		if (verbose)
			vmess("cone from angles: 2theta=%g deg -> s_pass=%.4f ms/m, "
			      "2theta=%g deg -> s_stop=%.4f ms/m  (v=%g m/s)",
			      theta_pass, p.s_pass, theta_stop, p.s_stop, v);
	}

	ncube    = (size_t)nlag * nx * nz;
	cube     = (float *)malloc(ncube * sizeof(float));
	passed   = (float *)malloc(ncube * sizeof(float));
	tapered  = (float *)malloc(ncube * sizeof(float));
	rejected = (float *)malloc(ncube * sizeof(float));
	if (!cube || !passed || !tapered || !rejected)
		verr("test_lag_cone: cannot allocate 4 cubes of %d x %d x %d.",
		     nlag, nx, nz);

	if (verbose)
		vmess("reading %s: %d lags, %d x %d, dx=%g dz=%g, dlag=%g s",
		      fname, nlag, nx, nz, dx, dz, dlag);

	readCube(fname, cube, (size_t)nlag * nx, (size_t)nz);

	lagConeFilter(cube, nlag, nx, nz, dx, dz, dlag,
	              &p, passed, tapered, &st, verbose);

	for (i = 0; i < (int)ncube; i++) rejected[i] = tapered[i] - passed[i];

	/* The partition is the whole point, so measure it rather than assert it. */
	{
		double mx = 0.0, rms = 0.0;
		for (i = 0; i < (int)ncube; i++) {
			double r = (double)passed[i] + rejected[i] - tapered[i];
			if (fabs(r) > mx) mx = fabs(r);
			rms += (double)tapered[i] * tapered[i];
		}
		rms = sqrt(rms / (double)ncube);
		vmess("partition check: max|passed+rejected-tapered| = %.3e, "
		      "rms(tapered) = %.3e, ratio = %.3e",
		      mx, rms, (rms > 0.0) ? mx/rms : mx);
	}

	if (verbose) {
		double ep = 0.0, er = 0.0, et = 0.0;
		for (i = 0; i < (int)ncube; i++) {
			ep += (double)passed[i]*passed[i];
			er += (double)rejected[i]*rejected[i];
			et += (double)tapered[i]*tapered[i];
		}
		vmess("rms  passed=%.6e  rejected=%.6e  tapered=%.6e",
		      sqrt(ep/ncube), sqrt(er/ncube), sqrt(et/ncube));
		vmess("padded transform lengths Nt=%d Nx=%d Nz=%d  (mirror these in "
		      "the python reference run)", st.Nt, st.Nx, st.Nz);
	}

	snprintf(fname, sizeof(fname), "%s_passed.su", out);
	writesufile(fname, passed,   (size_t)nz, (size_t)nx*nlag, z0, x0, dz, dx);
	snprintf(fname, sizeof(fname), "%s_rejected.su", out);
	writesufile(fname, rejected, (size_t)nz, (size_t)nx*nlag, z0, x0, dz, dx);
	snprintf(fname, sizeof(fname), "%s_tapered.su", out);
	writesufile(fname, tapered,  (size_t)nz, (size_t)nx*nlag, z0, x0, dz, dx);

	if (verbose)
		vmess("wrote %s_{passed,rejected,tapered}.su", out);

	free(cube); free(passed); free(tapered); free(rejected);
	return 0;
}
