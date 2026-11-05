#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<string.h>
#include"par.h"
#include"segy.h"
#include"fdelfwi.h"

/**
 * test_space_lag_gradient.c -- Standalone driver for the space-lag extended
 *                              gradient (space_lag_gradient.c).
 *
 * The sibling of test_lag_gradient.c, and deliberately the same program with
 * the lag axis moved from time to space, so that the two can be run on the
 * same setup and their cubes compared without any other difference between
 * the runs.  Per shot it does exactly four things:
 *
 *   1. forward model, saving checkpoints          (fdfwimodc)
 *   2. residual against the observed data         (computeResidual, L2)
 *   3. adjoint, adding each step into every lag   (adj_shot + slagGradient*)
 *   4. write that shot's lag cubes                (slagGradientWrite)
 *
 * then writes the shot-stacked lag cubes and the RAW (Vp, Vs, rho) gradient.
 *
 * WHAT IS DELIBERATELY ABSENT
 * ---------------------------
 * No preconditioner, no pseudo-Hessian output, no parameter scaling, no
 * gradient smoothing, no energy_ic angle filter, no cross-correlation or K-H
 * misfit, no multiscale bands, no Born/TRN path.  The gradient is written RAW
 * -- nothing is applied to it but grad_taper -- because the h=0 slice of the
 * lag cube has nothing else to be compared against.
 *
 * Accumulation is in Lame (adj_shot param=1) and the chain rule to
 * (Vp, Vs, rho) is applied ONCE, on the shot sum.  Converting per shot and
 * again on the total is the double-chain-rule bug.
 *
 * THE CHECK THIS PROGRAM IS BUILT AROUND
 * --------------------------------------
 * The h=0 slice of the stacked cube must reproduce the raw gradient TO FLOAT
 * ROUNDING, not merely correlate with it.  This is the sharp version of the
 * check test_lag_gradient can only make approximately: there, the tau=0 slice
 * is band limited by the frequency set, so the comparison is a ratio that
 * measures how much of the kernel the frequencies captured.  Here the h=0
 * accumulation is accumGradient's arithmetic over accumGradient's bounds in
 * accumGradient's order, so anything above float rounding is a BUG -- in the
 * loop bounds, the 0.25 Txz scatter, or the lag indexing.  See zeroLagCheck
 * for the two criteria and why they are what they are.  The comparison is
 * therefore made numerically, in this program, and printed as a PASS/FAIL
 * rather than left to a plotting script.
 *
 * It does NOT reach the h != 0 axis.  h=0 is invariant under a factor-2 error
 * in the lag scale and under a sign flip, exactly as tau=0 is.  That is what
 * slag_shiftx / slag_shiftz are for; see below.
 *
 * Output
 * ------
 *   <out_base>_vp_raw.su  <out_base>_vs_raw.su  <out_base>_rho_raw.su
 *   <out_base>_shotNNN_slagvp.su  <out_base>_shotNNN_slagvs.su
 *   <out_base>_shotNNN_slagaxis.txt
 *   <out_base>_slagvp.su  <out_base>_slagvs.su  <out_base>_slagaxis.txt
 *                                                    (the shot stack)
 */

#define MAX_COMP 8

double wallclock_time(void);

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);
int readModel(modPar *mod, bndPar *bnd);
int defineSource(wavPar wav, srcPar src, modPar mod, recPar rec,
                 shotPar shot, float **src_nwav, int reverse, int verbose);
int allocStoreSourceOnSurface(srcPar src);

int fdfwimodc(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec, snaPar *sna, int ixsrc, int izsrc,
              float **src_nwav, int ishot, int nshots, int fileno,
              checkpointPar *chk, int verbose);

int initCheckpoints(checkpointPar *chk, modPar *mod, int skipdt,
                    int delay, const char *file_base);
int cleanCheckpoints(checkpointPar *chk);

int readResidual(const char *filename, adjSrcPar *adj, modPar *mod,
                 bndPar *bnd, recPar *rec);
void freeResidual(adjSrcPar *adj);

int adj_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
             recPar *rec, adjSrcPar *adj,
             int ixsrc, int izsrc, float **src_nwav,
             checkpointPar *chk, snaPar *sna,
             float *grad1, float *grad2, float *grad3,
             float *hess_lam, float *hess_muu, float *hess_rho,
             float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
             int param, int verbose,
             float *wfld_energy);

void extractGradientVector(float *g, float *grad1, float *grad2, float *grad3,
                           modPar *mod, bndPar *bnd, int param);

int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

/* Shared helpers in their own C files.  Prototyped here rather than in
 * fdelfwi.h because four older drivers still carry file-local `static`
 * copies; see the note at that spot in fdelfwi.h. */
int computeHydrophone(const char *file_tzz, const char *file_txx,
                      const char *file_p, int verbose);
void applyCosineTaper(adjSrcPar *adj, int taper_len);
float *buildGradTaperMask(int nx, int nz,
                          shotPar *shot, recPar *rec, int radius, int core);
void applyGradTaperMask(float *data, const float *mask, int nmodel);


char *sdoc[] = {
" ",
" test_space_lag_gradient - space-lag extended FWI gradient g(x,h), raw only",
" ",
" Forward model, residual, adjoint; cross-correlate the forward and adjoint",
" wavefields separated by a space lag VECTOR h = (hx,hz) and write the",
" g_Vp(x,h), g_Vs(x,h) cubes.  Elastic only (ischeme>2).",
" ",
" The lag is applied SYMMETRICALLY -- forward at x-h, adjoint at x+h -- so h",
" is the one-way shift each field carries and the RELATIVE separation is 2h.",
" The conjugate variable of h is k_A+k_B = (2 w cos(theta)/v) N, the migration",
" wavenumber: its DIRECTION is the reflector normal and needs no velocity.",
" That is why the lag is a vector and not a scalar offset.",
" ",
" This driver computes NOTHING ELSE: no preconditioner, no pseudo-Hessian, no",
" parameter scaling, no smoothing, no angle filter.  Use test_gradient_visual",
" for those, and test_lag_gradient for the TIME-lag cube.",
" ",
" Required file_* and model parameters are those of fdelmodc/fdelfwi.",
" ",
" Lag parameters:",
"   slag_nhx=41   number of lags along x, forced odd so hx=0 is sampled.",
"   slag_nhz=1    number of lags along z, forced odd.",
"                 Memory is 2*nhx*nhz*nax*naz floats -- space lags cost per",
"                 LAG, where time lags cost per FREQUENCY.  (nhx,1) and",
"                 (1,nhz) give both components of k_A+k_B at cost nhx+nhz",
"                 instead of nhx*nhz, which is enough wherever one reflector",
"                 normal dominates per image point; use the full plane when",
"                 two events with different normals overlap.",
"   slag_dhx=1    lag stride along x in GRID POINTS (>=1).  A space lag is an",
"   slag_dhz=1    index offset applied during accumulation, not a phase",
"                 rotation applied at write time, so unlike lag_dt there is",
"                 no sub-sample freedom.  Physical lag is ihx*slag_dhx*dx.",
"   slag_shiftx=0 CALIBRATION TEST of the h axis.  Shift the ADJOINT field by",
"   slag_shiftz=0 this many grid points during accumulation.  Then",
"                   g_new(x,h) = g_old(x + D/2, h + D/2),",
"                 a DIAGONAL translation in (x,h).  Run twice (0 and D, D",
"                 EVEN) and cross-correlate: this validates the SCALE and the",
"                 SIGN of the h axis, neither of which the h=0 check can see.",
" ",
" Gradient parameters:",
"   comp=_rp      residual component(s), comma separated: _rvx,_rvz,_rp,",
"                 _rtxx,_rtzz,_rtxz.  _rp needs rec_type_txx=1 rec_type_tzz=1.",
"   file_obs=obs  observed data stem; files are <stem>_NNN<comp>.su",
"   out_base=slag output stem",
"   res_taper=0   cosine ramp (samples) on each end of the residual trace",
"   grad_taper=0  source/receiver taper radius in grid points.  Applied to",
"                 the raw gradient AND to every lag, with the same mask.",
"   grad_taper_core=0  grid points HARD ZEROED around every source and",
"                 receiver inside that radius.  The default pure r^2/R^2",
"                 ramp is zero only AT the point, so it rescales the",
"                 imaging-condition singularity instead of removing it.",
"   chk_skipdt=50 forward checkpoint interval in time steps",
"   chk_base=chk_slag  checkpoint file stem",
"   verbose=1     1 = per-shot progress, 2 = per-lag gradient norms",
" ",
" Output:",
"   <out_base>_{vp,vs,rho}_raw.su          raw gradient (shot sum)",
"   <out_base>_shotNNN_slag{vp,vs}.su      per-shot lag cubes",
"   <out_base>_slag{vp,vs}.su              shot-stacked lag cubes",
"   <out_base>_slagaxis.txt                lag/grid axis description",
" Cubes are nz samples by (nx*nhx*nhz) traces, lag SLOWEST:",
" trace = (ihz*nhx + ihx)*nx + ix.",
" ",
" The check, made numerically by this program: the h=0 slice of the stacked",
" cube must reproduce <out_base>_vp_raw.su to float rounding.  Unlike the",
" tau=0 check of test_lag_gradient there is no band limiting here, so a",
" discrepancy above float rounding is a bug, not a resolution effect.",
" ",
NULL};


/*====================================================================
 * gradStats -- norm, extrema and a NaN/Inf count
 *====================================================================*/
static void gradStats(const char *name, float *data, int nmodel)
{
	int i, nbad = 0;
	double sum2 = 0.0;
	float mn = data[0], mx = data[0];

	for (i = 0; i < nmodel; i++) {
		if (isnan(data[i]) || isinf(data[i])) { nbad++; continue; }
		sum2 += (double)data[i] * (double)data[i];
		if (data[i] < mn) mn = data[i];
		if (data[i] > mx) mx = data[i];
	}
	vmess("  %s: ||g||=%.6e  min=%.6e  max=%.6e  NaN/Inf=%d",
		name, (float)sqrt(sum2), mn, mx, nbad);
}


/*====================================================================
 * zeroLagCheck -- compare the h=0 plane against the conventional gradient
 *
 * Returns 0 on pass.  The two are meant to be the SAME arithmetic over the
 * same bounds in the same order, so the only admissible difference is float
 * rounding -- in practice whether the compiler contracted a*b+c into an FMA,
 * which it can decide differently in two translation units.
 *
 * Two criteria, because they fail in different ways:
 *
 *   L2REL = ||h0 - raw|| / ||raw||      <  1e-6
 *       The aggregate.  ULP noise is incoherent across the grid, so this sits
 *       at or below one float eps (measured 5e-8 / 9e-8 here) however many
 *       points carry a rounding difference -- about 40% of them do.
 *
 *   DMAX  = max|h0 - raw| / max|raw|    <  1e-5
 *       The worst point, normalised by the PEAK and not by the RMS.  Rounding
 *       error is relative to the local magnitude of the accumulated sum, so
 *       the peak is the scale it should be measured against; dividing by the
 *       RMS instead adds a factor rms/max (here ~3.5) that has nothing to do
 *       with the arithmetic and that was enough on its own to fail g_vs on a
 *       single-ULP difference.  Measured 2.2e-7 / 2.4e-7, i.e. 2 eps.
 *
 * Neither is a soft tolerance: the bugs this check exists to catch -- a
 * missing 0.25 in the Txz scatter, an off-by-one in the per-lag loop bounds,
 * a lag index that does not resolve to h=0 -- all move the h=0 plane by a
 * finite FRACTION of itself, four or more orders of magnitude above these.
 *====================================================================*/
static int zeroLagCheck(const char *name, const float *h0, const float *raw,
                        int nmodel)
{
	int i;
	double s2 = 0.0, d2 = 0.0, dot = 0.0, r2 = 0.0;
	double amax = 0.0, dmax = 0.0;
	double l2rel, drel;
	int ok;

	if (h0 == NULL) {
		vwarn("zeroLagCheck(%s): no stacked cube to check.", name);
		return 1;
	}

	for (i = 0; i < nmodel; i++) {
		double a = raw[i], b = h0[i], d = b - a;
		s2  += a*a;
		r2  += b*b;
		d2  += d*d;
		dot += a*b;
		if (fabs(a) > amax) amax = fabs(a);
		if (fabs(d) > dmax) dmax = fabs(d);
	}

	l2rel = (s2    > 0.0) ? sqrt(d2 / s2) : 0.0;
	drel  = (amax  > 0.0) ? dmax / amax   : 0.0;
	ok    = (l2rel < 1.0e-6) && (drel < 1.0e-5);

	vmess("  %s: ||raw||=%.6e  ||h=0||=%.6e  ratio=%.8f  corr=%.8f",
		name, sqrt(s2), sqrt(r2),
		(s2 > 0.0) ? sqrt(r2/s2) : 0.0,
		(s2 > 0.0 && r2 > 0.0) ? dot/sqrt(s2*r2) : 0.0);
	vmess("  %s: ||diff||/||raw||=%.3e (<1e-6)  max|diff|/max|raw|=%.3e "
		"(<1e-5)   ->  %s", name, l2rel, drel, ok ? "PASS" : "FAIL");

	return ok ? 0 : 1;
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
	adjSrcPar adj;
	float **src_nwav;
	float   sinkvel;
	double  t0, t1;
	size_t  nsamp, sizew, sizem;
	int     n1, ix, iz, ir, i, ishot;
	int     ioPx, ioPz, ixsrc, izsrc, fileno;
	int     nmodel, verbose, nfail;

	char   *chk_base, *comp_str, *file_obs, *out_base;
	int     chk_skipdt, res_taper, grad_taper, grad_taper_core;
	int     slag_nhx, slag_nhz, slag_dhx, slag_dhz, slag_shx, slag_shz;

	float  *grad1, *grad2, *grad3;
	float  *g1_shot, *g2_shot, *g3_shot;
	float  *hess1, *hess2, *hess3, *hess12, *hess13, *hess23, *wfld;
	float  *grad_vec, *taper_mask = NULL;
	float   total_misfit = 0.0f;

	t0 = wallclock_time();

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 1;
	if (!getparint("chk_skipdt", &chk_skipdt)) chk_skipdt = 50;
	if (!getparstring("chk_base", &chk_base)) chk_base = "chk_slag";
	if (!getparstring("comp", &comp_str)) comp_str = "_rp";
	if (!getparstring("file_obs", &file_obs)) file_obs = "obs";
	if (!getparstring("out_base", &out_base)) out_base = "slag";
	if (!getparint("res_taper", &res_taper)) res_taper = 0;
	if (!getparint("grad_taper", &grad_taper)) grad_taper = 0;
	if (!getparint("grad_taper_core", &grad_taper_core)) grad_taper_core = 0;

	/* ---- Standard model / source / receiver setup ---- */
	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);
	n1 = mod.naz;

	allocStoreSourceOnSurface(src);
	readModel(&mod, &bnd);

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

	/* ---- Sink sources/receivers to the first non-zero medium row ---- */
	ioPx = mod.ioPx;
	ioPz = mod.ioPz;
	if (bnd.lef == 4 || bnd.lef == 2) ioPx += bnd.ntap;
	if (bnd.top == 4 || bnd.top == 2) ioPz += bnd.ntap;

	ir = mod.ioZz + rec.z[0] + (rec.x[0] + mod.ioZx) * n1;
	rec.rho = mod.dt / (mod.dx * mod.roz[ir]);
	rec.cp  = sqrt(mod.l2m[ir] * (mod.roz[ir])) * mod.dx / mod.dt;

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

	sizem  = (size_t)mod.nax * mod.naz;
	nmodel = mod.nx * mod.nz;

	/* ---- Lag parameters.  This program has no purpose without them. ---- */
	if (mod.ischeme <= 2)
		verr("test_space_lag_gradient: the space-lag gradient needs the "
		     "elastic scheme (ischeme=3); got ischeme=%d.", mod.ischeme);

	if (!getparint("slag_nhx", &slag_nhx)) slag_nhx = 41;
	if (!getparint("slag_nhz", &slag_nhz)) slag_nhz = 1;
	if (!getparint("slag_dhx", &slag_dhx)) slag_dhx = 1;
	if (!getparint("slag_dhz", &slag_dhz)) slag_dhz = 1;
	if (!getparint("slag_shiftx", &slag_shx)) slag_shx = 0;
	if (!getparint("slag_shiftz", &slag_shz)) slag_shz = 0;

	if (slag_nhx <= 0 || slag_nhz <= 0)
		verr("test_space_lag_gradient: slag_nhx=%d slag_nhz=%d must both be "
		     "> 0 -- this driver exists only to compute the lag cubes.  Use "
		     "test_gradient_visual for a plain gradient.", slag_nhx, slag_nhz);
	if ((slag_shx % 2) || (slag_shz % 2))
		vwarn("test_space_lag_gradient: slag_shift (%d, %d) is ODD, so the "
		      "predicted translation D/2 falls between grid points and the "
		      "calibration cannot be read off exactly.  Use an even shift.",
		      slag_shx, slag_shz);

	vmess("********************************************");
	vmess("*****   SPACE-LAG GRADIENT (RAW ONLY)  *****");
	vmess("********************************************");
	vmess("Grid: nx=%d nz=%d (padded nax=%d naz=%d)",
		mod.nx, mod.nz, mod.nax, mod.naz);
	vmess("Shots: %d   component(s): %s", shot.n, comp_str);
	vmess("Lags: %d x %d  (strides %d, %d pts = %g, %g m)",
		slag_nhx, slag_nhz, slag_dhx, slag_dhz,
		slag_dhx * mod.dx, slag_dhz * mod.dz);
	vmess("Residual taper: %d samples   gradient taper: %d grid pts",
		res_taper, grad_taper);

	/* The taper mask is built ONCE and used both for the lag cubes (inside
	 * slagGradientWrite, per lag) and for the raw gradient below, so the h=0
	 * comparison is between two identically tapered fields. */
	if (grad_taper > 0) {
		taper_mask = buildGradTaperMask(mod.nx, mod.nz, &shot, &rec,
			grad_taper, grad_taper_core);
		vmess("Gradient taper mask shared with the lag cubes (radius=%d)",
			grad_taper);
	}

	/* ---- Allocate ---- */
	grad1   = (float *)calloc(sizem, sizeof(float));
	grad2   = (float *)calloc(sizem, sizeof(float));
	grad3   = (float *)calloc(sizem, sizeof(float));
	g1_shot = (float *)calloc(sizem, sizeof(float));
	g2_shot = (float *)calloc(sizem, sizeof(float));
	g3_shot = (float *)calloc(sizem, sizeof(float));

	/* adj_shot writes the pseudo-Hessian unconditionally, so it needs
	 * somewhere to put it.  Nothing here reads these back -- they exist to
	 * satisfy the signature, and are the reason this driver does not need a
	 * preconditioner to run. */
	hess1  = (float *)calloc(sizem, sizeof(float));
	hess2  = (float *)calloc(sizem, sizeof(float));
	hess3  = (float *)calloc(sizem, sizeof(float));
	hess12 = (float *)calloc(sizem, sizeof(float));
	hess13 = (float *)calloc(sizem, sizeof(float));
	hess23 = (float *)calloc(sizem, sizeof(float));
	wfld   = (float *)calloc(sizem, sizeof(float));
	if (!grad1 || !grad2 || !grad3 || !g1_shot || !g2_shot || !g3_shot ||
	    !hess1 || !hess2 || !hess3 || !hess12 || !hess13 || !hess23 || !wfld)
		verr("test_space_lag_gradient: cannot allocate the gradient arrays "
		     "(%zu points each).", sizem);

	/* ================================================================ *
	 * Shot loop: forward -> residual -> adjoint (+ lag planes) -> cubes
	 * ================================================================ */
	for (ishot = 0; ishot < shot.n; ishot++) {

		char  obs_names[MAX_COMP][512], syn_names[MAX_COMP][512];
		const char *obs_arr[MAX_COMP], *syn_arr[MAX_COMP];
		char  comp_buf[1024], res_fname[1024];
		char *token;
		int   ncomp = 0;
		float shot_misfit;

		ixsrc  = shot.x[ishot];
		izsrc  = shot.z[ishot];
		fileno = ishot;

		vmess("--- Shot %d/%d: ix=%d iz=%d ---",
			ishot + 1, shot.n, ixsrc, izsrc);

		/* ----- 1. Forward, saving checkpoints ----- */
		initCheckpoints(&chk, &mod, chk_skipdt, 0, chk_base);

		fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna,
			ixsrc, izsrc, src_nwav, ishot, shot.n, fileno,
			&chk, verbose);

		/* Elastic hydrophone: fdelmodc records Txx and Tzz, P is derived.
		 * Unconditional because comp= may name _rp in any position. */
		{
			char syn_tzz[512], syn_txx[512], syn_p[512];
			snprintf(syn_tzz, sizeof(syn_tzz), "%s_%03d_rtzz.su",
				rec.file_rcv, fileno);
			snprintf(syn_txx, sizeof(syn_txx), "%s_%03d_rtxx.su",
				rec.file_rcv, fileno);
			snprintf(syn_p,   sizeof(syn_p),   "%s_%03d_rp.su",
				rec.file_rcv, fileno);
			computeHydrophone(syn_tzz, syn_txx, syn_p, verbose);
		}

		/* ----- 2. Residual, L2 ----- */
		strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
		comp_buf[sizeof(comp_buf) - 1] = '\0';
		token = strtok(comp_buf, ",");
		while (token && ncomp < MAX_COMP) {
			snprintf(obs_names[ncomp], sizeof(obs_names[ncomp]),
				"%s_%03d%s.su", file_obs, ishot, token);
			snprintf(syn_names[ncomp], sizeof(syn_names[ncomp]),
				"%s_%03d%s.su", rec.file_rcv, ishot, token);
			obs_arr[ncomp] = obs_names[ncomp];
			syn_arr[ncomp] = syn_names[ncomp];
			ncomp++;
			token = strtok(NULL, ",");
		}
		if (ncomp == 0)
			verr("test_space_lag_gradient: comp= parsed to zero components.");

		snprintf(res_fname, sizeof(res_fname), "res_slag_%03d.su", ishot);

		shot_misfit = computeResidual(ncomp, obs_arr, syn_arr, res_fname,
			MISFIT_L2, NULL, NULL, NULL, verbose);
		total_misfit += shot_misfit;
		vmess("  misfit %.6e (cumulative %.6e)", shot_misfit, total_misfit);

		/* ----- 3. Adjoint, adding every step into every lag plane ----- */
		memset(&adj, 0, sizeof(adjSrcPar));
		readResidual(res_fname, &adj, &mod, &bnd, &rec);
		applyCosineTaper(&adj, res_taper);
		applyApertureTaper(&adj, mod.dx, rec.aper_taper, verbose);

		memset(g1_shot, 0, sizem * sizeof(float));
		memset(g2_shot, 0, sizem * sizeof(float));
		memset(g3_shot, 0, sizem * sizeof(float));

		/* Clears this shot's lag planes; the shot STACK is kept.  The
		 * calibration shift is a property of the accumulation, so it goes in
		 * here and not at write time -- the opposite of lag_dt, which is a
		 * phase rotation applied when the cube is reconstructed. */
		slagGradientInit(&mod, slag_nhx, slag_nhz, slag_dhx, slag_dhz,
			slag_shx, slag_shz, verbose);

		/* param=1: accumulate in Lame.  Converting per shot and again on the
		 * total is the double-chain-rule bug; here nothing converts at all,
		 * which is the point. */
		adj_shot(&mod, &src, &wav, &bnd, &rec, &adj,
			ixsrc, izsrc, src_nwav, &chk, &sna,
			g1_shot, g2_shot, g3_shot,
			hess1, hess2, hess3, hess12, hess13, hess23,
			1, verbose, wfld);

		/* ----- 4. This shot's lag cubes ----- */
		slagGradientWrite(&mod, &bnd, out_base, ishot, taper_mask, verbose);

		for (i = 0; i < (int)sizem; i++) {
			grad1[i] += g1_shot[i];
			grad2[i] += g2_shot[i];
			grad3[i] += g3_shot[i];
		}

		freeResidual(&adj);
		cleanCheckpoints(&chk);
	}

	vmess("=== Total misfit: %.6e ===", total_misfit);

	/* ---- Shot-stacked lag cubes: the ones that match the raw gradient ---- */
	slagGradientWriteStack(&mod, out_base, verbose);

	/* ---- Raw (Vp, Vs, rho) gradient, same taper, for the h=0 check ----
	 * adj_shot ran with param=1 so every shot accumulated in Lame; the chain
	 * rule to velocity is applied ONCE here, on the shot sum.  This is the
	 * same pointwise map slagGradientWrite applies to each lag, so the h=0
	 * slice and these files are directly comparable. */
	grad_vec = (float *)calloc((size_t)3 * nmodel, sizeof(float));
	if (!grad_vec)
		verr("test_space_lag_gradient: cannot allocate the gradient vector.");
	extractGradientVector(grad_vec, grad1, grad2, grad3, &mod, &bnd, 2);

	if (taper_mask) {
		applyGradTaperMask(grad_vec,                  taper_mask, nmodel);
		applyGradTaperMask(grad_vec + nmodel,         taper_mask, nmodel);
		applyGradTaperMask(grad_vec + 2 * (size_t)nmodel, taper_mask, nmodel);
	}

	vmess("--- Raw gradient (Vp, Vs, rho) ---");
	gradStats("g_vp",  grad_vec,                      nmodel);
	gradStats("g_vs",  grad_vec + nmodel,             nmodel);
	gradStats("g_rho", grad_vec + 2 * (size_t)nmodel, nmodel);

	/* ---- THE check: h=0 against the conventional gradient ----
	 * Done here, numerically, while the stack is still allocated.  With a
	 * calibration shift in force the h=0 plane is no longer the gradient --
	 * it is the gradient translated by D/2 -- so the check is skipped rather
	 * than reported as a failure. */
	nfail = 0;
	if (slag_shx == 0 && slag_shz == 0) {
		vmess("--- h=0 slice vs conventional gradient ---");
		nfail += zeroLagCheck("g_vp", slagGradientStackZeroLag(0),
			grad_vec, nmodel);
		nfail += zeroLagCheck("g_vs", slagGradientStackZeroLag(1),
			grad_vec + nmodel, nmodel);
	}
	else {
		vmess("--- h=0 check SKIPPED: slag_shift=(%d,%d) translates the whole "
			"cube by (%d,%d) grid points in x and h ---",
			slag_shx, slag_shz, slag_shx/2, slag_shz/2);
	}

	slagGradientFree();

	{
		char fname[1024];
		snprintf(fname, sizeof(fname), "%s_vp_raw.su", out_base);
		writesufile(fname, grad_vec, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		snprintf(fname, sizeof(fname), "%s_vs_raw.su", out_base);
		writesufile(fname, grad_vec + nmodel, (size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		snprintf(fname, sizeof(fname), "%s_rho_raw.su", out_base);
		writesufile(fname, grad_vec + 2 * (size_t)nmodel,
			(size_t)mod.nz, (size_t)mod.nx,
			mod.z0, mod.x0, mod.dz, mod.dx);
		vmess("Wrote %s_{vp,vs,rho}_raw.su", out_base);
	}

	t1 = wallclock_time();
	printf("\n");
	printf("=== SPACE-LAG GRADIENT SUMMARY ===\n");
	printf("  Shots:        %d\n", shot.n);
	printf("  Total misfit: %.6e\n", total_misfit);
	printf("  Lag axes:     hx %d x %g m   hz %d x %g m\n",
		slag_nhx, slag_dhx * mod.dx, slag_nhz, slag_dhz * mod.dz);
	printf("                |hx| <= %g m   |hz| <= %g m  (one-way)\n",
		((slag_nhx-1)/2) * slag_dhx * mod.dx,
		((slag_nhz-1)/2) * slag_dhz * mod.dz);
	if (slag_shx || slag_shz)
		printf("  Calibration:  adjoint shifted (%d, %d) pts -> cube must "
			"move (%d, %d) pts in x AND h\n",
			slag_shx, slag_shz, slag_shx/2, slag_shz/2);
	else
		printf("  h=0 check:    %s\n", nfail ? "FAIL" : "PASS");
	printf("  Wall time:    %.2f s\n", t1 - t0);
	printf("==================================\n");

	free(grad1);  free(grad2);  free(grad3);
	free(g1_shot); free(g2_shot); free(g3_shot);
	free(hess1); free(hess2); free(hess3);
	free(hess12); free(hess13); free(hess23);
	free(wfld);
	free(grad_vec);
	if (taper_mask) free(taper_mask);

	return nfail ? 1 : 0;
}
