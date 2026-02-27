#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"par.h"
#include"fdelfwi.h"

/**
 * born_shot.c -- Forward propagation of Born (mu_1) wavefield.
 *
 * Computes J*dm (the Jacobian-vector product) for one shot by
 * propagating the Born/scattered wavefield forward in time with
 * distributed virtual sources derived from the model perturbation dm.
 *
 * Uses the same checkpoint re-propagation approach as adj_shot:
 * for each segment between checkpoints, the forward wavefield is
 * re-propagated from the checkpoint while the Born wavefield is
 * advanced with virtual source injection at every time step.
 *
 * Time-step ordering within each step:
 *   1. inject_born_vsrc_vel(sigma_fwd^n -> born_v)
 *   2. callKernel(fwd, with source)          [re-propagate forward]
 *   3. callKernelNoSrc(born)                  [advance born]
 *   4. inject_born_vsrc_stress(v_fwd^{n+1} -> born_sigma)
 *   5. getRecTimes(born)                      [record at receivers]
 *
 * The Born data recorded at receivers is J*dm.  This is then used as
 * adjoint source in adj_shot to compute J^T(J*dm) = H_GN * dm.
 *
 * See HESSIAN_MATH.md for the full derivation.
 */

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

double wallclock_time(void);

/* Forward FD kernels (for re-propagation) */
int elastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime,
	int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz,
	float *tzz, float *txx, float *txz, float *rox, float *roz,
	float *l2m, float *lam, float *mul, int verbose);
int elastic6(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime,
	int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz,
	float *tzz, float *txx, float *txz, float *rox, float *roz,
	float *l2m, float *lam, float *mul, int verbose);
int elastic8(modPar mod, srcPar src, wavPar wav, bndPar bnd, int itime,
	int ixsrc, int izsrc, float **src_nwav, float *vx, float *vz,
	float *tzz, float *txx, float *txz, float *rox, float *roz,
	float *l2m, float *lam, float *mul, int verbose);

/* Checkpoint I/O */
int readCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl);

/* Receiver recording and output (from fdelmodc) */
int getRecTimes(modPar mod, recPar rec, bndPar bnd, int itime, int isam,
	float *vx, float *vz, float *tzz, float *txx, float *txz,
	float *q, float *l2m, float *lam, float *rox, float *roz,
	float *rec_vx, float *rec_vz, float *rec_txx, float *rec_tzz,
	float *rec_txz, float *rec_p, float *rec_pp, float *rec_ss,
	float *rec_q, float *rec_udp, float *rec_udvz,
	float *rec_dxvx, float *rec_dzvz, int verbose);

int writeRec(recPar rec, modPar mod, bndPar bnd, wavPar wav,
	int ixsrc, int izsrc, int nsam, int ishot, int nshots, int fileno,
	float *rec_vx, float *rec_vz, float *rec_txx, float *rec_tzz,
	float *rec_txz, float *rec_p, float *rec_pp, float *rec_ss,
	float *rec_q, float *rec_udp, float *rec_udvz,
	float *rec_dxvx, float *rec_dzvz, int verbose);

/* Virtual source injection (born_vsrc.c) */
void inject_born_vsrc_vel(modPar *mod,
	float *txx_fwd, float *tzz_fwd, float *txz_fwd,
	float *delta_rox, float *delta_roz,
	float *born_vx, float *born_vz);
void inject_born_vsrc_stress(modPar *mod, bndPar *bnd,
	float *vx_fwd, float *vz_fwd,
	float *delta_l2m, float *delta_lam, float *delta_mul,
	float *born_txx, float *born_tzz, float *born_txz);


/***********************************************************************
 * callKernel -- Dispatch to the forward FD kernel (with source).
 ***********************************************************************/
static void callKernel(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                       int it, int ixsrc, int izsrc, float **src_nwav,
                       wflPar *wfl, int verbose)
{
	switch (mod->ischeme) {
		case 3:
		case 5:
			if (mod->iorder == 4)
				elastic4(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			else if (mod->iorder == 6)
				elastic6(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			else if (mod->iorder == 8)
				elastic8(*mod, *src, *wav, *bnd, it, ixsrc, izsrc, src_nwav,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			break;
	}
}


/***********************************************************************
 * callKernelNoSrc -- Dispatch to FD kernel without source injection.
 *
 * Uses a dummy srcPar with n=0 so applySource loops are empty.
 * storeSourceOnSurface/reStoreSourceOnSurface also loop over n=0.
 ***********************************************************************/
static void callKernelNoSrc(modPar *mod, bndPar *bnd, int it,
                            wflPar *wfl, int verbose)
{
	srcPar src_null;
	wavPar wav_null;
	float *dummy = NULL;
	float **src_nwav_null = &dummy;

	memset(&src_null, 0, sizeof(srcPar));
	memset(&wav_null, 0, sizeof(wavPar));
	src_null.type = 20;  /* skip source injection in all elastic kernels */

	switch (mod->ischeme) {
		case 3:
		case 5:
			if (mod->iorder == 4)
				elastic4(*mod, src_null, wav_null, *bnd, it, 0, 0, src_nwav_null,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			else if (mod->iorder == 6)
				elastic6(*mod, src_null, wav_null, *bnd, it, 0, 0, src_nwav_null,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			else if (mod->iorder == 8)
				elastic8(*mod, src_null, wav_null, *bnd, it, 0, 0, src_nwav_null,
					wfl->vx, wfl->vz, wfl->tzz, wfl->txx, wfl->txz,
					mod->rox, mod->roz, mod->l2m, mod->lam, mod->muu, verbose);
			break;
	}
}


/***********************************************************************
 * born_shot -- Forward Born propagation for one shot (J * dm).
 *
 * Re-propagates the forward wavefield from checkpoints while
 * simultaneously advancing the Born wavefield with virtual source
 * injection.  Records the Born wavefield at receiver positions
 * and writes .su output files.
 *
 * Parameters:
 *   mod, bnd                - model and boundary parameters
 *   src, wav, src_nwav,     - forward source (for re-propagation)
 *     ixsrc, izsrc
 *   rec                     - receiver parameters (positions, timing)
 *   chk                     - checkpoint metadata
 *   delta_rox, ..., delta_mul - perturbed FD coefficients from
 *                               perturbFDcoefficients()
 *   file_born               - output .su file base name
 *   ishot, nshots, fileno   - shot indices for output naming
 *   verbose                 - verbosity level
 *
 * Output:
 *   Writes Born receiver data to .su files at file_born location,
 *   using the same naming convention as fdfwimodc (via writeRec).
 *   Returns 0 on success.
 ***********************************************************************/
int born_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec,
              int ixsrc, int izsrc, float **src_nwav,
              checkpointPar *chk,
              float *delta_rox, float *delta_roz,
              float *delta_l2m, float *delta_lam, float *delta_mul,
              const char *file_born,
              int ishot, int nshots, int fileno,
              int verbose)
{
	wflPar wfl_fwd, wfl_born;
	float *saved_txx, *saved_tzz, *saved_txz;
	int k, j, it, seg_start, seg_end, nsteps, isam;
	size_t sizem;
	int it1;
	double t0_wall;

	it1   = mod->nt + NINT(-mod->t0 / mod->dt);
	sizem = (size_t)mod->nax * mod->naz;

	/* ------------------------------------------------------------ */
	/* 1. Allocate forward re-propagation wavefield                 */
	/* ------------------------------------------------------------ */
	memset(&wfl_fwd, 0, sizeof(wflPar));
	wfl_fwd.vx  = (float *)calloc(sizem, sizeof(float));
	wfl_fwd.vz  = (float *)calloc(sizem, sizeof(float));
	wfl_fwd.tzz = (float *)calloc(sizem, sizeof(float));
	if (mod->ischeme > 2) {
		wfl_fwd.txz = (float *)calloc(sizem, sizeof(float));
		wfl_fwd.txx = (float *)calloc(sizem, sizeof(float));
	}

	/* ------------------------------------------------------------ */
	/* 2. Allocate Born wavefield (zero initial conditions)          */
	/* ------------------------------------------------------------ */
	memset(&wfl_born, 0, sizeof(wflPar));
	wfl_born.vx  = (float *)calloc(sizem, sizeof(float));
	wfl_born.vz  = (float *)calloc(sizem, sizeof(float));
	wfl_born.tzz = (float *)calloc(sizem, sizeof(float));
	if (mod->ischeme > 2) {
		wfl_born.txz = (float *)calloc(sizem, sizeof(float));
		wfl_born.txx = (float *)calloc(sizem, sizeof(float));
	}

	/* ------------------------------------------------------------ */
	/* 3. Saved stress for cross-segment Phase F1 virtual source    */
	/*                                                               */
	/* At the boundary between segments, Phase F1 vsrc needs        */
	/* sigma_fwd at the END of the previous segment (before the     */
	/* checkpoint overwrites wfl_fwd).  For k=0, this is zero (IC). */
	/* ------------------------------------------------------------ */
	saved_txx = (float *)calloc(sizem, sizeof(float));
	saved_tzz = (float *)calloc(sizem, sizeof(float));
	saved_txz = (float *)calloc(sizem, sizeof(float));

	/* ------------------------------------------------------------ */
	/* 4. Allocate receiver recording buffers                        */
	/* ------------------------------------------------------------ */
	float *rec_vx=NULL, *rec_vz=NULL, *rec_p=NULL;
	float *rec_txx=NULL, *rec_tzz=NULL, *rec_txz=NULL;
	float *rec_pp=NULL, *rec_ss=NULL, *rec_q=NULL;
	float *rec_udp=NULL, *rec_udvz=NULL;
	float *rec_dxvx=NULL, *rec_dzvz=NULL;

	size_t size = (size_t)rec->n * rec->nt;
	if (size > 0) {
		if (rec->type.vx)   rec_vx   = (float *)calloc(size, sizeof(float));
		if (rec->type.vz)   rec_vz   = (float *)calloc(size, sizeof(float));
		if (rec->type.p)    rec_p    = (float *)calloc(size, sizeof(float));
		if (rec->type.txx)  rec_txx  = (float *)calloc(size, sizeof(float));
		if (rec->type.tzz)  rec_tzz  = (float *)calloc(size, sizeof(float));
		if (rec->type.txz)  rec_txz  = (float *)calloc(size, sizeof(float));
		if (rec->type.dxvx) rec_dxvx = (float *)calloc(size, sizeof(float));
		if (rec->type.dzvz) rec_dzvz = (float *)calloc(size, sizeof(float));
		if (rec->type.pp)   rec_pp   = (float *)calloc(size, sizeof(float));
		if (rec->type.ss)   rec_ss   = (float *)calloc(size, sizeof(float));
	}

	if (verbose) {
		vmess("born_shot: %d segments, sizem=%zu, file=%s",
			chk->nsnap, sizem, file_born);
		t0_wall = wallclock_time();
	}

	/* ------------------------------------------------------------ */
	/* 5. Temporarily redirect receiver output to file_born          */
	/*                                                               */
	/* Save original file_rcv pointer and replace with file_born so  */
	/* that writeRec writes Born data to a distinct output file.     */
	/* NOTE: file_rcv is a char* pointer (not fixed array), so we    */
	/* swap the pointer — never write into the getparstring buffer.  */
	/* ------------------------------------------------------------ */
	char *saved_file_rcv = rec->file_rcv;
	rec->file_rcv = (char *)file_born;

	/* ------------------------------------------------------------ */
	/* 6. Segment loop: forward in time (k=0 to nsnap-1)            */
	/*                                                               */
	/* Unlike adj_shot which processes segments in reverse, the Born */
	/* wavefield propagates forward.  The Born wavefield is          */
	/* continuous across segments (NOT reset at each checkpoint),    */
	/* while wfl_fwd is reset from each checkpoint.                  */
	/* ------------------------------------------------------------ */
	isam = 0;

	for (k = 0; k < chk->nsnap; k++) {

		seg_start = chk->delay + k * chk->skipdt;
		seg_end   = (k == chk->nsnap - 1) ? it1 : chk->delay + (k + 1) * chk->skipdt;
		nsteps    = seg_end - seg_start;

		if (verbose > 1)
			vmess("born_shot: Segment %d/%d  it=[%d,%d)  nsteps=%d",
				k, chk->nsnap, seg_start, seg_end, nsteps);

		/* Save forward stress from end of previous segment BEFORE
		 * checkpoint load overwrites wfl_fwd.
		 * For k=0, saved arrays are zero (initial conditions via calloc). */
		if (k > 0) {
			memcpy(saved_txx, wfl_fwd.txx, sizem * sizeof(float));
			memcpy(saved_tzz, wfl_fwd.tzz, sizem * sizeof(float));
			memcpy(saved_txz, wfl_fwd.txz, sizem * sizeof(float));
		}

		/* Load forward checkpoint */
		readCheckpoint(chk, k, &wfl_fwd);

		/* Process each time step in this segment */
		for (j = 0; j < nsteps; j++) {
			it = seg_start + j;

			/* ---- Phase F1 virtual source (velocity, from density) ----
			 * Inject BEFORE callKernel so that the Born Phase F2
			 * (inside callKernelNoSrc) sees the correct born_v.
			 * Uses forward stress BEFORE the elastic step at time 'it':
			 *   j=0: saved_stress (end of previous segment, or zero IC for k=0)
			 *   j>0: wfl_fwd stress (current time, before this step's kernel) */
#pragma omp parallel default(shared)
{
			inject_born_vsrc_vel(mod,
				(j == 0) ? saved_txx : wfl_fwd.txx,
				(j == 0) ? saved_tzz : wfl_fwd.tzz,
				(j == 0) ? saved_txz : wfl_fwd.txz,
				delta_rox, delta_roz,
				wfl_born.vx, wfl_born.vz);
}

			/* ---- Advance forward wavefield (re-propagation with source) ----
			 * SKIP at j=0: the checkpoint already contains the state
			 * AFTER elastic4(seg_start) was applied in fdfwimodc.
			 * Re-applying it would double-step the forward wavefield.
			 * (Same convention as adj_shot.c which starts at j=1.) */
			if (j > 0) {
#pragma omp parallel default(shared)
{
			callKernel(mod, src, wav, bnd, it, ixsrc, izsrc,
			           src_nwav, &wfl_fwd, verbose);
}
			}

			/* ---- Advance Born wavefield (no physical source) ---- */
#pragma omp parallel default(shared)
{
			callKernelNoSrc(mod, bnd, it, &wfl_born, verbose);
}

			/* ---- Phase F2 virtual source (stress, from stiffness) ----
			 * Inject AFTER both kernels.
			 * Uses forward velocity at time n+1 (after elastic step). */
#pragma omp parallel default(shared)
{
			inject_born_vsrc_stress(mod, bnd,
				wfl_fwd.vx, wfl_fwd.vz,
				delta_l2m, delta_lam, delta_mul,
				wfl_born.txx, wfl_born.tzz, wfl_born.txz);
}

			/* ---- Record Born wavefield at receivers ----
			 * CRITICAL: Use same isam formula as applyAdjointSource
			 * (0-based) so that the recording and injection operators
			 * are exact adjoints of each other.  The fdfwimodc formula
			 * has a +1 offset that breaks the R/R^T duality required
			 * for Hessian symmetry (J^T J must be symmetric).
			 *
			 * BOUNDS CHECK: isam must be < rec->nt to avoid writing
			 * past the allocated receiver buffer.  The simulation may
			 * run longer than tmod (e.g. nt=1024 > tmod/dt=1000),
			 * so time steps past the recording window are skipped. */
			if ((((it - rec->delay) % rec->skipdt) == 0) &&
			    (it >= rec->delay) && size > 0) {
				isam = (it - rec->delay) / rec->skipdt;
				if (isam >= rec->nt) continue;

				getRecTimes(*mod, *rec, *bnd, it, isam,
					wfl_born.vx, wfl_born.vz,
					wfl_born.tzz, wfl_born.txx, wfl_born.txz,
					NULL, mod->l2m, mod->lam, mod->rox, mod->roz,
					rec_vx, rec_vz, rec_txx, rec_tzz, rec_txz,
					rec_p, rec_pp, rec_ss, rec_q,
					rec_udp, rec_udvz, rec_dxvx, rec_dzvz,
					verbose);
			}

		} /* end time step loop */

	} /* end segment loop */

	/* ------------------------------------------------------------ */
	/* 7. Write Born receiver data to .su files                      */
	/* ------------------------------------------------------------ */
	if (size > 0) {
		writeRec(*rec, *mod, *bnd, *wav, ixsrc, izsrc,
		         isam + 1, ishot, nshots, fileno,
		         rec_vx, rec_vz, rec_txx, rec_tzz, rec_txz,
		         rec_p, rec_pp, rec_ss, rec_q,
		         rec_udp, rec_udvz, rec_dxvx, rec_dzvz, verbose);
	}

	/* ------------------------------------------------------------ */
	/* 8. Restore original receiver file name pointer                */
	/* ------------------------------------------------------------ */
	rec->file_rcv = saved_file_rcv;

	if (verbose)
		vmess("born_shot: Completed Born propagation in %.2f s.",
		      wallclock_time() - t0_wall);

	/* ------------------------------------------------------------ */
	/* 9. Free all allocated memory                                  */
	/* ------------------------------------------------------------ */
	free(wfl_fwd.vx); free(wfl_fwd.vz); free(wfl_fwd.tzz);
	if (wfl_fwd.txx) free(wfl_fwd.txx);
	if (wfl_fwd.txz) free(wfl_fwd.txz);

	free(wfl_born.vx); free(wfl_born.vz); free(wfl_born.tzz);
	if (wfl_born.txx) free(wfl_born.txx);
	if (wfl_born.txz) free(wfl_born.txz);

	free(saved_txx); free(saved_tzz); free(saved_txz);

	if (rec_vx)   free(rec_vx);
	if (rec_vz)   free(rec_vz);
	if (rec_p)    free(rec_p);
	if (rec_txx)  free(rec_txx);
	if (rec_tzz)  free(rec_tzz);
	if (rec_txz)  free(rec_txz);
	if (rec_pp)   free(rec_pp);
	if (rec_ss)   free(rec_ss);
	if (rec_dxvx) free(rec_dxvx);
	if (rec_dzvz) free(rec_dzvz);

	return 0;
}
