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
 * checkpoint.c  --  Disk-based checkpoint I/O for FWI re-propagation.
 *
 * Saves and restores the complete wavefield state (vx, vz, txx, tzz, txz)
 * so the forward simulation can be restarted from any checkpoint during
 * the adjoint pass.
 *
 * Checkpoint format:  raw float arrays, nax*naz per snapshot, concatenated.
 *   Acoustic  (ischeme <= 2): vx, vz, tzz          (3 files)
 *   Elastic   (ischeme >  2): vx, vz, tzz, txx, txz (5 files)
 *
 * Each snapshot = nax * naz * sizeof(float) bytes per component.
 */

#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))


/***********************************************************************
 * initCheckpoints -- Set up checkpoint parameters and generate paths.
 *
 * Parameters:
 *   chk       - checkpoint struct to initialize
 *   mod       - model parameters (grid dimensions, time info, ischeme)
 *   skipdt    - time steps between checkpoints
 *   delay     - first checkpoint time step (0 = start)
 *   file_base - base path for checkpoint files (e.g. "chk_shot001")
 ***********************************************************************/
int initCheckpoints(checkpointPar *chk, modPar *mod, int skipdt, int delay, const char *file_base)
{
	int it1, ncomp;
	double snap_mb;

	if (!chk || !mod || !file_base) {
		verr("initCheckpoints: NULL pointer argument.");
		return -1;
	}

	it1 = mod->nt + NINT(-mod->t0 / mod->dt);

	chk->skipdt  = skipdt;
	chk->delay   = delay;
	chk->naz     = mod->naz;
	chk->nax     = mod->nax;
	chk->ischeme = mod->ischeme;
	chk->it      = 0;

	/* Calculate number of snapshots */
	if (it1 <= delay || skipdt < 1) {
		chk->nsnap = 0;
		return 0;
	}
	chk->nsnap = (it1 - 1 - delay) / skipdt + 1;

	/* Generate file paths for all components */
	snprintf(chk->file_vx,  sizeof(chk->file_vx),  "%s_chk_vx.bin",  file_base);
	snprintf(chk->file_vz,  sizeof(chk->file_vz),  "%s_chk_vz.bin",  file_base);
	snprintf(chk->file_tzz, sizeof(chk->file_tzz), "%s_chk_tzz.bin", file_base);
	snprintf(chk->file_txx, sizeof(chk->file_txx), "%s_chk_txx.bin", file_base);
	snprintf(chk->file_txz, sizeof(chk->file_txz), "%s_chk_txz.bin", file_base);

	if (chk->nsnap > 0) {
		ncomp = (mod->ischeme > 2) ? 5 : 3;
		snap_mb = (double)chk->nax * chk->naz * sizeof(float) / (1024.0 * 1024.0);

		vmess("initCheckpoints: %d checkpoints, skipdt=%d, delay=%d, ischeme=%d",
			chk->nsnap, skipdt, delay, mod->ischeme);
		vmess("  grid: %d x %d = %.1f MB per component per checkpoint",
			chk->nax, chk->naz, snap_mb);
		vmess("  components: %d (%s)", ncomp,
			ncomp == 5 ? "vx, vz, tzz, txx, txz" : "vx, vz, tzz");
		vmess("  total disk: %.1f MB",
			(double)ncomp * chk->nsnap * snap_mb);
	}

	return 0;
}


/***********************************************************************
 * writeOneComponent -- Write one raw float array to a checkpoint file.
 ***********************************************************************/
static int writeOneComponent(const char *path, int isnap, float *data, size_t n)
{
	FILE *fp;
	size_t count;
	off_t offset;

	offset = (off_t)isnap * n * sizeof(float);

	fp = fopen(path, isnap == 0 ? "w" : "r+");
	if (!fp) { verr("writeCheckpoint: Cannot open %s", path); return -1; }
	if (fseeko(fp, offset, SEEK_SET)) { verr("writeCheckpoint: fseek failed on %s", path); fclose(fp); return -1; }
	count = fwrite(data, sizeof(float), n, fp);
	fclose(fp);
	if (count != n) { verr("writeCheckpoint: Short write on %s (%zu/%zu)", path, count, n); return -1; }

	return 0;
}

/***********************************************************************
 * writeCheckpoint -- Write the complete wavefield state to disk.
 *
 * Acoustic  (ischeme <= 2): writes vx, vz, tzz
 * Elastic   (ischeme >  2): writes vx, vz, tzz, txx, txz
 ***********************************************************************/
int writeCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl)
{
	size_t n;

	n = (size_t)chk->nax * chk->naz;

	if (writeOneComponent(chk->file_vx,  isnap, wfl->vx,  n)) return -1;
	if (writeOneComponent(chk->file_vz,  isnap, wfl->vz,  n)) return -1;
	if (writeOneComponent(chk->file_tzz, isnap, wfl->tzz, n)) return -1;

	if (chk->ischeme > 2) {
		if (writeOneComponent(chk->file_txx, isnap, wfl->txx, n)) return -1;
		if (writeOneComponent(chk->file_txz, isnap, wfl->txz, n)) return -1;
	}

	return 0;
}


/***********************************************************************
 * readOneComponent -- Read one raw float array from a checkpoint file.
 ***********************************************************************/
static int readOneComponent(const char *path, int isnap, float *data, size_t n)
{
	FILE *fp;
	size_t count;
	off_t offset;

	offset = (off_t)isnap * n * sizeof(float);

	fp = fopen(path, "r");
	if (!fp) { verr("readCheckpoint: Cannot open %s", path); return -1; }
	if (fseeko(fp, offset, SEEK_SET)) { verr("readCheckpoint: fseek failed on %s", path); fclose(fp); return -1; }
	count = fread(data, sizeof(float), n, fp);
	fclose(fp);
	if (count != n) { verr("readCheckpoint: Short read on %s (%zu/%zu)", path, count, n); return -1; }

	return 0;
}

/***********************************************************************
 * readCheckpoint -- Read the complete wavefield state from disk.
 *
 * Reads snapshot isnap into the wflPar struct.  All arrays must be
 * pre-allocated (nax * naz floats each).  For acoustic models, txx
 * and txz pointers in wfl may be NULL.
 ***********************************************************************/
int readCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl)
{
	size_t n;

	if (!chk || !wfl) {
		verr("readCheckpoint: NULL pointer argument.");
		return -1;
	}
	if (isnap < 0 || isnap >= chk->nsnap) {
		verr("readCheckpoint: isnap=%d out of range [0, %d).", isnap, chk->nsnap);
		return -1;
	}

	n = (size_t)chk->nax * chk->naz;

	if (readOneComponent(chk->file_vx,  isnap, wfl->vx,  n)) return -1;
	if (readOneComponent(chk->file_vz,  isnap, wfl->vz,  n)) return -1;
	if (readOneComponent(chk->file_tzz, isnap, wfl->tzz, n)) return -1;

	if (chk->ischeme > 2) {
		if (!wfl->txx || !wfl->txz) {
			verr("readCheckpoint: elastic checkpoint requires txx/txz arrays.");
			return -1;
		}
		if (readOneComponent(chk->file_txx, isnap, wfl->txx, n)) return -1;
		if (readOneComponent(chk->file_txz, isnap, wfl->txz, n)) return -1;
	}

	return 0;
}


/***********************************************************************
 * cleanCheckpoints -- Delete checkpoint files from disk.
 ***********************************************************************/
int cleanCheckpoints(checkpointPar *chk)
{
	if (!chk) return -1;

	if (chk->nsnap > 0) {
		remove(chk->file_vx);
		remove(chk->file_vz);
		remove(chk->file_tzz);
		if (chk->ischeme > 2) {
			remove(chk->file_txx);
			remove(chk->file_txz);
		}
	}

	chk->nsnap = 0;
	return 0;
}
