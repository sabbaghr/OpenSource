/*
 * fdelfwi_fwd.c - Forward-only modeling driver for fdelfwi.
 *
 * Generates receiver data with fdelfwi's OWN forward engine (fdfwimodc), i.e.
 * the exact same discretization and the exact same boundary implementation
 * that fwi_inversion / test_gradient_visual use to compute synthetics.
 *
 * WHY THIS EXISTS
 * ---------------
 * Observed data used to be generated with fdelmodc. The two codes agree to
 * float roundoff in the interior (verified: diff/obs ~ 3e-9 for a source 200
 * cells away from the absorbing pad), but every boundary implementation
 * differs, because fdelfwi was forked from fdelmodc before the upstream
 * commits
 *     b6b5292  new CFS-CPML implementation for elastic and acoustic schemes
 *     60eaadf  boundaries for P/Tzz fields the same as other boundaries
 *     6f89b30  redesign boundaries for P/T fields
 * reworked the boundary / P-T index design, and that rework was never
 * propagated into fdelfwi. On top of that, fdelmodc's elastic taper boundary
 * (left/right/top/bottom=4) produces NaN, so fdelmodc data had to be made with
 * PML while fdelfwi must run with the taper (boundaries_adj.c implements the
 * elastic adjoint only for types 4, 3 and 1).
 *
 * The result was a ~1% residual on the direct wave even when the "true" and
 * "background" models were bit-identical, caused purely by the two codes
 * absorbing the top boundary differently (PML reflects ~1e-5, the cosine taper
 * ~1e-2). With a source a few cells below the top of the model that mismatch
 * is large, and it contaminates both the misfit and the gradient.
 *
 * Generating the observed data here instead makes the residual at the true
 * model exactly zero, which is what a gradient / inversion test needs.
 *
 * USAGE
 * -----
 * Takes the same parameters as fdelmodc. Use the SAME boundary settings as the
 * subsequent gradient / inversion run:
 *
 *   fdelfwi_fwd file_cp=... file_cs=... file_den=... file_src=wave.su \
 *       file_rcv=obs ischeme=3 iorder=4 src_type=1 \
 *       rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 \
 *       dtrcv=0.004 tmod=4.0 \
 *       xrcv1=0 xrcv2=5000 dxrcv=10 zrcv1=50 zrcv2=50 \
 *       xsrc=500 zsrc=20 nshot=1 dxshot=0 \
 *       ntaper=100 left=4 right=4 top=4 bottom=4
 *
 * Output is written by writeRec as <file_rcv>_<NNN>_r<comp>.su with a
 * zero-padded 3-digit shot number (writeRec.c:77), which is already the naming
 * fwi_inversion and test_gradient_visual expect for file_obs= -- unlike
 * fdelmodc, no renaming step is needed.
 *
 * No checkpoints are written (chk=NULL): this is forward modeling only.
 */

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<assert.h>
#include<string.h>
#include"par.h"
#include"segy.h"
#include"fdelfwi.h"

double wallclock_time(void);
void threadAffinity(void);

int getParameters(modPar *mod, recPar *rec, snaPar *sna, wavPar *wav,
                  srcPar *src, shotPar *shot, bndPar *bnd, int verbose);
int readModel(modPar *mod, bndPar *bnd);
int defineSource(wavPar wav, srcPar src, modPar mod, recPar rec,
                 shotPar shot, float **src_nwav, int reverse, int verbose);
int allocStoreSourceOnSurface(srcPar src);
int freeStoreSourceOnSurface(void);

int fdfwimodc(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec, snaPar *sna, int ixsrc, int izsrc,
              float **src_nwav, int ishot, int nshots, int fileno,
              checkpointPar *chk, int verbose);

char *sdoc[] = {
" ",
" fdelfwi_fwd - forward-only modeling with the fdelfwi engine",
" ",
" Records receiver data using the same kernel and the same boundary",
" implementation as fwi_inversion / test_gradient_visual, so that observed",
" data generated here gives an exactly zero residual at the true model.",
" ",
" Use fdelmodc instead only if you deliberately want a different engine;",
" fdelmodc's boundary treatment differs from fdelfwi's and leaves a ~1%",
" residual on the direct wave even for identical models.",
" ",
" Same parameters as fdelmodc. Output:",
"   <file_rcv>_<NNN>_r<comp>.su   zero-padded shot number, one file per shot",
" ",
" Notes:",
"   * use the SAME left/right/top/bottom and ntaper as the gradient run",
"   * fdelfwi has no elastic PML adjoint, so use the taper boundary (=4)",
"   * for an elastic hydrophone, record rec_type_txx=1 rec_type_tzz=1 and",
"     form P = 0.5*(Txx+Tzz) afterwards (suop2/sugain), as the demo scripts do",
" ",
NULL};

int main(int argc, char **argv)
{
	modPar  mod;
	recPar  rec;
	snaPar  sna;
	wavPar  wav;
	srcPar  src;
	bndPar  bnd;
	shotPar shot;
	float **src_nwav;
	double  t0, t1;
	size_t  nsamp, sizew;
	int     i, ishot, ixsrc, izsrc;
	int     verbose;

	t0 = wallclock_time();

	initargs(argc, argv);
	requestdoc(0);

	if (!getparint("verbose", &verbose)) verbose = 0;

	getParameters(&mod, &rec, &sna, &wav, &src, &shot, &bnd, verbose);

	allocStoreSourceOnSurface(src);
	readModel(&mod, &bnd);

	/* Source wavelet setup (identical to the other drivers) */
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
		sizew       = (size_t)wav.nt * wav.nx;
		src_nwav    = (float **)calloc(wav.nx, sizeof(float *));
		src_nwav[0] = (float *)calloc(sizew, sizeof(float));
		assert(src_nwav[0] != NULL);
		for (i = 0; i < wav.nx; i++)
			src_nwav[i] = (float *)(src_nwav[0] + (size_t)(wav.nt * i));
	}

	defineSource(wav, src, mod, rec, shot, src_nwav, mod.grid_dir, verbose);

	if (verbose) {
		vmess("fdelfwi_fwd: forward-only modeling, %d shot(s)", shot.n);
		vmess("fdelfwi_fwd: boundaries left=%d right=%d top=%d bottom=%d ntaper=%d",
		      bnd.lef, bnd.rig, bnd.top, bnd.bot, bnd.ntap);
	}

	for (ishot = 0; ishot < shot.n; ishot++) {

		ixsrc = shot.x[ishot];
		izsrc = shot.z[ishot];

		if (verbose)
			vmess("--- Shot %d/%d: ix=%d iz=%d ---",
			      ishot + 1, shot.n, ixsrc, izsrc);

		/* chk=NULL: no checkpointing, this is forward modeling only */
		fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna,
		          ixsrc, izsrc, src_nwav, ishot, shot.n, ishot,
		          NULL, verbose);
	}

	freeStoreSourceOnSurface();
	free(src_nwav[0]);
	free(src_nwav);

	t1 = wallclock_time();
	if (verbose)
		vmess("fdelfwi_fwd: total wall time = %.3f s", t1 - t0);

	return 0;
}
