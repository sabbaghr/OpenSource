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
 *  3. tau0=1 (implied by nv>1): take the SAME cone through
 *     lagConePspiTau0 and write only <out>_tau0.su, the tau = 0 slice --
 *     which IS the tomographic gradient the inversion is driven by.  With
 *     file_vel= and nv>1 the cutoff FOLLOWS v(x,z) instead of being pinned
 *     to one velocity.
 *
 * Purpose (2) exists so the C filter can be compared sample by sample
 * against demo/gradient/lag_slope_filter.py, which is the validated
 * reference.  The padded transform lengths actually used are printed, so the
 * Python run can be made to mirror them exactly -- otherwise the two would
 * differ merely because they padded to different lengths.
 *
 * Purpose (3) exists so the cone can be re-run at several nv -- or several
 * angles -- WITHOUT repeating the DFT accumulation that built the cube,
 * which is hours per shot on a production-sized model and is identical for
 * every nv.  It is the same call lagGradientTomo makes, with the same
 * arguments, so a tau0 panel from here is the per-shot contribution
 * fwi_inversion would have added to g_Vp.
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
"   selftest=0 ..... 1: FFT normalisation.  2: the tau=0 projection-slice",
"                    shortcut vs the full 3D inverse.  3: the cutoff really",
"                    follows v.  4: the cutoff error in DEGREES.  Each one",
"                    runs and exits; no cube is read.",
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
"   v=.............. velocity used for the angle->slope conversion [m/s].",
"                    Not needed when nv>1: the model supplies it.",
"  ",
"   Varying-velocity cutoff (lag_cone_pspi.c).  One slope enforces one",
"   opening angle only where the model is v; elsewhere it drifts by",
"   d(2theta) = -2 (dv/v) cot(theta).  nv>1 removes that:",
"   nv=1 ........... reference-velocity bins.  1 = one stationary cutoff,",
"                    identical to the lagConeFilter path above.  >1 makes",
"                    the cutoff follow v(x,z) and REQUIRES theta_pass/",
"                    theta_stop and file_vel (a slope has one velocity",
"                    already baked in).",
"   file_vel=....... SU model of the MODE velocity on the interior grid --",
"                    Vp for mode=vp, Vs for mode=vs.  nx traces of nz",
"                    samples, matching lagaxis.txt.  Mixing the two mis-cuts",
"                    both cubes by the Vp/Vs ratio.",
"   tau0=0 ......... 1: go through lagConePspiTau0 and write ONLY",
"                    <out>_tau0.su, the tau=0 slice (= the tomographic",
"                    gradient).  Implied by nv>1.  Much cheaper than the",
"                    full cube: the nv inverse transforms are 2D, not 3D.",
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
	char *base, *out, *mode, *file_vel, fname[1024], axisfn[1024], oname[1024];
	int   nlag, nx, nz, verbose, selftest, i, tau0_only, have_angles;
	float dx, dz, dlag, x0, z0;
	float theta_pass, theta_stop, v;
	float *cube, *passed, *tapered, *rejected, *vel, *tau0;
	double dv;
	size_t ncube;
	lagConePar p;
	lagConeStats st;

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 1;
	if (!getparint("selftest", &selftest)) selftest = 0;

	if (selftest == 1) {
		float err = lagConeSelfTest(1);
		vmess("round-trip relative error %.3e -- %s", err,
		      err < 1.0e-5f ? "PASS" : "FAIL");
		return (err < 1.0e-5f) ? 0 : 1;
	}
	if (selftest == 2) {
		/* The tau=0 projection-slice shortcut, checked against the full 3D
		 * inverse of lagConeFilter itself.  This is what validates the
		 * phase weight e^{2 pi i n mid / Nt} and the Hermitian doubling. */
		float err = lagConePspiSelfTest(1);
		vmess("tau=0 slice relative error %.3e -- %s", err,
		      err < 1.0e-5f ? "PASS" : "FAIL");
		return (err < 1.0e-5f) ? 0 : 1;
	}
	if (selftest == 3) {
		/* Does the cutoff actually follow v(x,z)? */
		float err = lagConePspiVaryTest(1);
		vmess("v-following relative error %.3e -- %s", err,
		      err < 1.0e-5f ? "PASS" : "FAIL");
		return (err < 1.0e-5f) ? 0 : 1;
	}
	if (selftest == 4) {
		/* WHICH angle does the filter cut at?  Reports DEGREES, and covers the
		 * narrow/migration end where the stationary cutoff is unreachable. */
		float deg = lagConePspiCutoffTest(1);
		vmess("worst nv=4 cutoff error %.3f deg -- %s", deg,
		      deg < 1.0f ? "PASS" : "FAIL");
		return (deg < 1.0f) ? 0 : 1;
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
	if (!getparint  ("nv",      &p.nv))      ;
	if (p.nv < 1)
		verr("test_lag_cone: nv=%d must be >= 1 (1 = one stationary cutoff).",
		     p.nv);
	if (!getparint("tau0", &tau0_only)) tau0_only = 0;
	if (p.nv > 1) tau0_only = 1;
	if (!getparstring("file_vel", &file_vel)) file_vel = NULL;

	/* An opening angle, if given, overrides the slopes.  The ANGLES are kept
	 * as well as the slopes: with nv>1 the slope is rebuilt per velocity bin,
	 * so a single pre-baked s_pass cannot be used. */
	have_angles = (getparfloat("theta_pass", &theta_pass) &&
	               getparfloat("theta_stop", &theta_stop));
	if (have_angles) {
		if (!(theta_stop < theta_pass))
			verr("test_lag_cone: need theta_stop < theta_pass (a WIDER angle "
			     "is a SMALLER slope), got %g and %g deg.",
			     theta_pass, theta_stop);
		p.theta_pass = theta_pass;
		p.theta_stop = theta_stop;
		if (getparfloat("v", &v)) {
			p.s_pass = lagConeAngleToSlope(theta_pass, v);
			p.s_stop = lagConeAngleToSlope(theta_stop, v);
			if (verbose)
				vmess("cone from angles: 2theta=%g deg -> s_pass=%.4f ms/m, "
				      "2theta=%g deg -> s_stop=%.4f ms/m  (v=%g m/s)",
				      theta_pass, p.s_pass, theta_stop, p.s_stop, v);
		} else if (p.nv <= 1) {
			verr("test_lag_cone: theta_pass/theta_stop need either v= (one "
			     "velocity for the angle->slope map) or nv>1 (the cutoff then "
			     "follows file_vel).");
		}
	} else if (p.nv > 1) {
		verr("test_lag_cone: nv=%d makes the cutoff follow v(x,z), which is "
		     "only defined for an ANGLE: give theta_pass= and theta_stop= "
		     "(deg).  A slope already has one velocity baked into it.", p.nv);
	}

	if (p.nv > 1 && !file_vel)
		verr("test_lag_cone: nv=%d needs file_vel= (the %s model on the "
		     "interior grid, %d traces of %d samples).", p.nv, mode, nx, nz);
	if (file_vel && p.nv <= 1)
		vwarn("test_lag_cone: file_vel= is ignored at nv=1 -- the cutoff is "
		      "the single slope s_pass=%.4f / s_stop=%.4f ms/m.",
		      p.s_pass, p.s_stop);

	ncube = (size_t)nlag * nx * nz;
	cube  = (float *)malloc(ncube * sizeof(float));
	if (!cube)
		verr("test_lag_cone: cannot allocate a cube of %d x %d x %d.",
		     nlag, nx, nz);

	if (verbose)
		vmess("reading %s: %d lags, %d x %d, dx=%g dz=%g, dlag=%g s",
		      fname, nlag, nx, nz, dx, dz, dlag);

	readCube(fname, cube, (size_t)nlag * nx, (size_t)nz);

	/* ---------------------------------------------------------------- *
	 * tau = 0 only, through the PSPI path.  This is the exact call
	 * lagGradientTomo makes, so the panel written here is the per-shot
	 * tomographic gradient contribution.
	 * ---------------------------------------------------------------- */
	if (tau0_only) {
		vel  = NULL;
		tau0 = (float *)malloc((size_t)nx * nz * sizeof(float));
		if (!tau0)
			verr("test_lag_cone: cannot allocate the tau=0 slice.");
		if (p.nv > 1) {
			vel = (float *)malloc((size_t)nx * nz * sizeof(float));
			if (!vel)
				verr("test_lag_cone: cannot allocate the velocity model.");
			if (verbose)
				vmess("reading %s as the %s cutoff velocity: %d traces of %d "
				      "samples", file_vel, mode, nx, nz);
			readCube(file_vel, vel, (size_t)nx, (size_t)nz);
		}

		lagConePspiTau0(cube, nlag, nx, nz, dx, dz, dlag,
		                &p, vel, tau0, &st, verbose);

		if (verbose) {
			double e = 0.0;
			for (i = 0; i < nx*nz; i++) e += (double)tau0[i]*tau0[i];
			vmess("tau=0 slice: ||g||=%.6e  rms=%.6e", sqrt(e),
			      sqrt(e/((double)nx*nz)));
			vmess("spectral energy kept: %.4f%%  (nu=0 plane holds %.4f%%)",
			      100.0*st.energy_frac_passed, 100.0*st.energy_frac_nu0);
			vmess("padded transform lengths Nt=%d Nx=%d Nz=%d",
			      st.Nt, st.Nx, st.Nz);
		}

		snprintf(fname, sizeof(fname), "%s_tau0.su", out);
		writesufile(fname, tau0, (size_t)nz, (size_t)nx, z0, x0, dz, dx);
		if (verbose) vmess("wrote %s", fname);

		free(cube); free(tau0); free(vel);
		return 0;
	}

	passed   = (float *)malloc(ncube * sizeof(float));
	tapered  = (float *)malloc(ncube * sizeof(float));
	rejected = (float *)malloc(ncube * sizeof(float));
	if (!passed || !tapered || !rejected)
		verr("test_lag_cone: cannot allocate 3 more cubes of %d x %d x %d.",
		     nlag, nx, nz);

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
