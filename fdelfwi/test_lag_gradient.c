#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<string.h>
#include"par.h"
#include"segy.h"
#include"fdelfwi.h"

/**
 * test_lag_gradient.c -- Standalone driver for the time-lag extended
 *                        gradient (lag_gradient.c).
 *
 * WHY THIS EXISTS AS ITS OWN PROGRAM
 * ----------------------------------
 * The lag gradient used to be driven from test_gradient_visual, which also
 * builds the pseudo-Hessian, the Yang block preconditioner, parameter
 * scaling, the energy-norm angle filter, the Kirchhoff-Helmholtz misfit and
 * multiscale band filtering.  None of that is needed to look at g(x,tau):
 * the lag cubes are read straight out of lag_gradient's own DFT accumulators
 * and are untouched by scaling, smoothing or preconditioning.  Driving them
 * from there meant every lag experiment paid for -- and could be perturbed by
 * -- machinery it does not use, and every change to the lag path risked a
 * test that other features depend on.
 *
 * So this program does exactly four things per shot:
 *
 *   1. forward model, saving checkpoints          (fdfwimodc)
 *   2. residual against the observed data         (computeResidual, L2)
 *   3. adjoint, folding each step into the lag    (adj_shot + lagGradient*)
 *      DFTs
 *   4. write that shot's lag cubes                (lagGradientWrite)
 *
 * and then writes the shot-stacked lag cubes plus the RAW (Vp, Vs, rho)
 * gradient.
 *
 * WHAT IS DELIBERATELY ABSENT
 * ---------------------------
 * No preconditioner, no pseudo-Hessian output, no parameter scaling, no
 * gradient smoothing, no energy_ic angle filter, no cross-correlation or
 * K-H misfit, no multiscale bands, no Born/TRN path.  The gradient is
 * written RAW -- nothing is applied to it but grad_taper -- because the
 * tau=0 slice of the lag cube has nothing else to be compared against.
 *
 * Accumulation is in Lame (adj_shot param=1) and the chain rule to
 * (Vp, Vs, rho) is applied ONCE, on the shot sum.  Converting per shot and
 * again on the total is the double-chain-rule bug.
 *
 * The one thing that IS applied is grad_taper, and it has to be: without it
 * the source/receiver singularity holds essentially all of |g|^2 and every
 * panel is a picture of two bright dots.  The same mask goes to the raw
 * gradient and to every lag (buildGradTaperMask, shared with the lag module
 * precisely so the two cannot drift), which is what keeps the tau=0 check
 * like for like.
 *
 * THE CHECK THIS PROGRAM IS BUILT AROUND
 * --------------------------------------
 * The tau=0 slice of <out_base>_lagvp.su must reproduce
 * <out_base>_vp_raw.su (and likewise Vs), both being sums over shots.  That
 * single comparison exercises the DFT accumulation, the 2*df reconstruction
 * weight, the conj(A)B complex algebra, the 0.25 Txz scatter and the loop
 * bounds at once.  It is not bit-exact -- the lag cube is band limited to
 * nfreq discrete lines over [lag_fmin, lag_fmax] -- so the ratio is itself
 * the measurement: it says how much of the kernel the frequency set captures.
 *
 * Output
 * ------
 *   <out_base>_vp_raw.su  <out_base>_vs_raw.su  <out_base>_rho_raw.su
 *   <out_base>_shotNNN_lagvp.su   <out_base>_shotNNN_lagvs.su
 *   <out_base>_shotNNN_lagaxis.txt
 *   <out_base>_lagvp.su  <out_base>_lagvs.su  <out_base>_lagaxis.txt
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
                          shotPar *shot, recPar *rec, int radius);
void applyGradTaperMask(float *data, const float *mask, int nmodel);


char *sdoc[] = {
" ",
" test_lag_gradient - time-lag extended FWI gradient g(x,tau), raw only",
" ",
" Forward model, residual, adjoint; cross-correlate the forward and adjoint",
" wavefields at a range of TIME LAGS and write the g_Vp(x,tau), g_Vs(x,tau)",
" cubes.  Elastic only (ischeme>2).",
" ",
" This driver computes NOTHING ELSE: no preconditioner, no pseudo-Hessian,",
" no parameter scaling, no smoothing, no angle filter.  Use",
" test_gradient_visual for those.",
" ",
" Required file_* and model parameters are those of fdelmodc/fdelfwi.",
" ",
" Lag parameters:",
"   lag_nfreq=16  number of frequencies used to form the lag correlation.",
"                 Must be > 0; this program does nothing otherwise.",
"   lag_fmin=     lowest frequency in Hz, > 0 (DC carries no signal).",
"   lag_fmax=     highest frequency in Hz (default lag_fmin, one line).",
"                 Frequencies are uniform on [lag_fmin, lag_fmax].  The lag",
"                 reconstruction is periodic with period 1/df in the",
"                 RELATIVE lag, which the symmetric convention puts at",
"                 1/(2 df) on the tau axis -- cover the source band densely",
"                 enough that 1/(2 df) exceeds the lag range, or the run",
"                 aborts rather than show wrap-around as lag moveout.",
"   lag_nt=21     number of lags, forced odd so tau=0 is sampled exactly.",
"   lag_dt=       lag increment in s (default 10*dt).  Need not be a",
"                 multiple of dt: the shift is a phase rotation.",
" ",
" Gradient parameters:",
"   comp=_rp      residual component(s), comma separated: _rvx,_rvz,_rp,",
"                 _rtxx,_rtzz,_rtxz.  _rp needs rec_type_txx=1 rec_type_tzz=1.",
"   file_obs=obs  observed data stem; files are <stem>_NNN<comp>.su",
"   out_base=lag  output stem",
"   res_taper=0   cosine ramp (samples) on each end of the residual trace",
"   grad_taper=0  source/receiver taper radius in grid points.  Applied to",
"                 the raw gradient AND to every lag, with the same mask.",
"   lag_shift=0   CALIBRATION TEST.  Delay the adjoint source by this many",
"                 samples.  The entire lag cube must then translate by",
"                 +lag_shift*dtrcv/2 in tau -- HALF, because the lag is",
"                 applied symmetrically so the relative shift is 2*tau.",
"                 Run twice (0 and N) and cross-correlate the two cubes",
"                 along tau.  This is the only check that validates the",
"                 tau != 0 axis; tau=0 is invariant under both a factor-2",
"                 error in the lag scale and a sign flip.",
"   chk_skipdt=50 forward checkpoint interval in time steps",
"   chk_base=chk_lag  checkpoint file stem",
"   verbose=1     1 = per-shot progress, 2 = per-lag gradient norms",
" ",
" Output:",
"   <out_base>_{vp,vs,rho}_raw.su           raw gradient (shot sum)",
"   <out_base>_shotNNN_lag{vp,vs}.su       per-shot lag cubes",
"   <out_base>_lag{vp,vs}.su               shot-stacked lag cubes",
"   <out_base>_lagaxis.txt                  lag/grid axis description",
" Cubes are nz samples by (nx*nlag) traces, lag SLOWEST: trace = ilag*nx+ix.",
" ",
" The check: the tau=0 slice of <out_base>_lagvp.su must reproduce",
" <out_base>_vp_raw.su, up to the band limiting of the frequency set.",
" ",
NULL};


/*====================================================================
 * stripBoundary -- copy the interior of a padded array into a dense one
 *====================================================================*/
static void stripBoundary(float *dst, float *src, modPar *mod, bndPar *bnd)
{
	int ix, iz;
	int ibndx = mod->ioPx;
	int ibndz = mod->ioPz;
	if (bnd->lef == 4 || bnd->lef == 2) ibndx += bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibndz += bnd->ntap;
	for (ix = 0; ix < mod->nx; ix++)
		for (iz = 0; iz < mod->nz; iz++)
			dst[ix * mod->nz + iz] =
				src[(ix + ibndx) * mod->naz + iz + ibndz];
}


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
	int     nmodel, verbose;

	char   *chk_base, *comp_str, *file_obs, *out_base;
	int     chk_skipdt, res_taper, grad_taper, lag_shift;
	int     lag_nfreq, lag_nt, nfreq_dense;
	float   lag_fmin, lag_fmax, lag_dt, trec;

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
	if (!getparstring("chk_base", &chk_base)) chk_base = "chk_lag";
	if (!getparstring("comp", &comp_str)) comp_str = "_rp";
	if (!getparstring("file_obs", &file_obs)) file_obs = "obs";
	if (!getparstring("out_base", &out_base)) out_base = "lag";
	if (!getparint("res_taper", &res_taper)) res_taper = 0;
	if (!getparint("grad_taper", &grad_taper)) grad_taper = 0;

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
		verr("test_lag_gradient: the time-lag gradient needs the elastic "
		     "scheme (ischeme=3); got ischeme=%d.", mod.ischeme);
	if (!getparfloat("lag_fmin", &lag_fmin))
		verr("test_lag_gradient: lag_fmin= is required (lowest frequency "
		     "in Hz, > 0).");
	if (!getparfloat("lag_fmax", &lag_fmax)) lag_fmax = lag_fmin;

	/* lag_nfreq defaults to the DENSE value, df = 1/T.  A sparse set is not a
	 * cheaper approximation of the lag gradient, it is a different and wrong
	 * object (the replication at period 1/df folds correlation energy back
	 * into the lag window), so there is no reason to make the user compute
	 * this and no reason to default to anything coarser.  lagGradientInit
	 * rejects anything sparser; this just means it never has to. */
	trec        = (float)mod.nt * mod.dt;
	nfreq_dense = (int)ceilf((lag_fmax - lag_fmin) * trec) + 1;
	if (!getparint("lag_nfreq", &lag_nfreq)) {
		lag_nfreq = nfreq_dense;
		vmess("lag_nfreq defaulted to %d, the dense value for a %.3f s record "
			"over [%g, %g] Hz (df = 1/T = %.4f Hz)",
			lag_nfreq, trec, lag_fmin, lag_fmax, 1.0f/trec);
	}
	if (lag_nfreq <= 0)
		verr("test_lag_gradient: lag_nfreq=%d must be > 0 -- this driver "
		     "exists only to compute the lag cubes.  Use "
		     "test_gradient_visual for a plain gradient.", lag_nfreq);
	if (lag_nfreq < nfreq_dense)
		verr("test_lag_gradient: lag_nfreq=%d is SPARSE for a %.3f s record "
		     "over [%g, %g] Hz; %d is the dense value (df <= 1/T = %.4f Hz).  "
		     "A sparse set does not approximate the lag gradient, it folds "
		     "correlation energy back into the lag window.  Omit lag_nfreq to "
		     "get %d automatically.",
		     lag_nfreq, trec, lag_fmin, lag_fmax, nfreq_dense,
		     1.0f/trec, nfreq_dense);
	if (!getparint("lag_nt", &lag_nt)) lag_nt = 21;
	if (!getparint("lag_shift", &lag_shift)) lag_shift = 0;
	if (!getparfloat("lag_dt", &lag_dt)) lag_dt = 10.0f * mod.dt;
	if (lag_dt <= 0.0f)
		verr("test_lag_gradient: lag_dt=%g must be > 0 s.", lag_dt);

	vmess("*******************************************");
	vmess("*****   TIME-LAG GRADIENT (RAW ONLY)  *****");
	vmess("*******************************************");
	vmess("Grid: nx=%d nz=%d (padded nax=%d naz=%d)",
		mod.nx, mod.nz, mod.nax, mod.naz);
	vmess("Shots: %d   component(s): %s", shot.n, comp_str);
	vmess("Lags: %d x %g s   frequencies: %d on [%g, %g] Hz",
		lag_nt, lag_dt, lag_nfreq, lag_fmin, lag_fmax);
	vmess("Residual taper: %d samples   gradient taper: %d grid pts",
		res_taper, grad_taper);

	/* The taper mask is built ONCE and used both for the lag cubes (inside
	 * lagGradientWrite, per lag) and for the raw gradient below, so the
	 * tau=0 comparison is between two identically tapered fields. */
	if (grad_taper > 0) {
		taper_mask = buildGradTaperMask(mod.nx, mod.nz, &shot, &rec, grad_taper);
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
		verr("test_lag_gradient: cannot allocate the gradient arrays "
		     "(%zu points each).", sizem);

	/* ================================================================ *
	 * Shot loop: forward -> residual -> adjoint (+ lag DFTs) -> cubes
	 * ================================================================ */
	for (ishot = 0; ishot < shot.n; ishot++) {

		char  obs_names[MAX_COMP][512], syn_names[MAX_COMP][512];
		const char *obs_arr[MAX_COMP], *syn_arr[MAX_COMP];
		char  comp_buf[1024], res_fname[1024];
		char *token;
		int   ncomp = 0, ic;
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
			if (ncomp >= MAX_COMP)
				verr("test_lag_gradient: more than %d components in comp=.",
				     MAX_COMP);
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
			verr("test_lag_gradient: comp= parsed to zero components.");

		snprintf(res_fname, sizeof(res_fname), "res_lag_%03d.su", ishot);

		shot_misfit = computeResidual(ncomp, obs_arr, syn_arr, res_fname,
			MISFIT_L2, NULL, NULL, NULL, verbose);
		total_misfit += shot_misfit;
		vmess("  misfit %.6e (cumulative %.6e)", shot_misfit, total_misfit);

		/* ----- 3. Adjoint, folding every step into the lag DFTs ----- */
		memset(&adj, 0, sizeof(adjSrcPar));
		readResidual(res_fname, &adj, &mod, &bnd, &rec);
		applyCosineTaper(&adj, res_taper);
		applyApertureTaper(&adj, mod.dx, rec.aper_taper, verbose);

		/* ----- lag_shift: the calibration test of the lag AXIS -----
		 * Delaying the adjoint source by Delta must translate the whole lag
		 * cube by exactly Delta/2, and in the +tau direction:
		 *   g(tau) = sum_t A(t-tau) B(t+tau);  B(t) -> B(t-D) gives
		 *   g_new(tau) = sum_t A(t-tau)B(t+tau-D) = g_old(tau - D/2)
		 * (substitute s = t - D/2).  The factor 1/2 is the whole point: it is
		 * there because the lag is applied symmetrically, forward by -tau and
		 * adjoint by +tau, so the RELATIVE shift is 2*tau.  A cube that moves
		 * by Delta instead of Delta/2 has the lag axis mis-scaled by two, and
		 * a cube that moves the wrong way has the sign convention flipped --
		 * neither of which the tau=0 check can see, since tau=0 is invariant
		 * under both errors. */
		if (lag_shift != 0) {
			int isrc, it, ns = adj.nt;
			/* readResidual stores adj->wav[k] = the datum at forward time
			 * k*rec.skipdt, so one sample here is rec.skipdt model steps. */
			float dt_adj = (float)rec.skipdt * mod.dt;
			float *tmp = (float *)malloc((size_t)ns * sizeof(float));
			if (!tmp) verr("test_lag_gradient: lag_shift scratch.");
			for (isrc = 0; isrc < adj.nsrc; isrc++) {
				float *w = &adj.wav[(size_t)isrc * ns];
				memcpy(tmp, w, (size_t)ns * sizeof(float));
				for (it = 0; it < ns; it++) {
					int is = it - lag_shift;      /* delay by +lag_shift */
					w[it] = (is >= 0 && is < ns) ? tmp[is] : 0.0f;
				}
			}
			free(tmp);
			vmess("  lag_shift: adjoint source delayed %d samples (%.5f s); "
				"the lag cube must translate by %+.5f s in tau",
				lag_shift, lag_shift * dt_adj, 0.5f * lag_shift * dt_adj);
		}

		memset(g1_shot, 0, sizem * sizeof(float));
		memset(g2_shot, 0, sizem * sizeof(float));
		memset(g3_shot, 0, sizem * sizeof(float));

		/* Clears this shot's DFT accumulators; the shot STACK is kept. */
		lagGradientInit(&mod, lag_nfreq, lag_fmin, lag_fmax, verbose);

		/* param=1: accumulate in Lame.  Converting per shot and again on
		 * the total is the double-chain-rule bug; here nothing converts at
		 * all, which is the point. */
		adj_shot(&mod, &src, &wav, &bnd, &rec, &adj,
			ixsrc, izsrc, src_nwav, &chk, &sna,
			g1_shot, g2_shot, g3_shot,
			hess1, hess2, hess3, hess12, hess13, hess23,
			1, verbose, wfld);

		/* ----- 4. This shot's lag cubes ----- */
		lagGradientWrite(&mod, &bnd, out_base, ishot,
			lag_nt, lag_dt, taper_mask, verbose);

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
	lagGradientWriteStack(&mod, out_base, verbose);
	lagGradientFree();

	/* ---- Raw (Vp, Vs, rho) gradient, same taper, for the tau=0 check ----
	 * adj_shot ran with param=1 so every shot accumulated in Lame; the
	 * chain rule to velocity is applied ONCE here, on the shot sum.
	 * Converting per shot AND here is the double-chain-rule bug.  This is
	 * the same pointwise map lagGradientWrite applies to each lag, so the
	 * tau=0 slice and these files are directly comparable. */
	grad_vec = (float *)calloc((size_t)3 * nmodel, sizeof(float));
	if (!grad_vec)
		verr("test_lag_gradient: cannot allocate the gradient vector.");
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
	printf("=== TIME-LAG GRADIENT SUMMARY ===\n");
	printf("  Shots:        %d\n", shot.n);
	printf("  Total misfit: %.6e\n", total_misfit);
	printf("  Lag axis:     %d lags x %g s  (|tau| <= %g s)\n",
		lag_nt, lag_dt, lag_dt * ((lag_nt - 1) / 2));
	printf("  Frequencies:  %d on [%g, %g] Hz\n",
		lag_nfreq, lag_fmin, lag_fmax);
	printf("  Wall time:    %.2f s\n", t1 - t0);
	printf("  Check:        tau=0 slice of %s_lagvp.su  vs  %s_vp_raw.su\n",
		out_base, out_base);
	printf("=================================\n");

	free(grad1);  free(grad2);  free(grad3);
	free(g1_shot); free(g2_shot); free(g3_shot);
	free(hess1); free(hess2); free(hess3);
	free(hess12); free(hess13); free(hess23);
	free(wfld);
	free(grad_vec);
	if (taper_mask) free(taper_mask);

	return 0;
}
