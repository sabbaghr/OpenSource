#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "par.h"
#include "segy.h"
#include "fdelfwi.h"

/**
 * hess_shot.c - Gauss-Newton Hessian-vector product for one shot.
 *
 * Computes H_GN * dm = J^T(J * dm) by orchestrating:
 *   1. born_shot()        -> Forward Born propagation (J * dm)
 *   2. combineBornFiles() -> Merge component .su files with TRID headers
 *   3. readResidual()     -> Load Born data into adjSrcPar
 *   4. adj_shot()         -> Second adjoint backpropagation (J^T * d_born)
 *
 * Sign convention (verified against adjoint state formulation):
 *   born_shot writes d_born = R * mu_1 (Born scattered data at receivers).
 *   adj_shot negates input internally (line 271-275): injects -(d_born).
 *   The second adjoint equation is A^T mu_2 = -R^T(d_born), which is
 *   exactly what adj_shot produces with the Born data as input.
 *   accumGradient then gives J^T(d_born) = H_GN * dm.
 *
 * See HESSIAN_MATH.md for the full derivation.
 */

#define MAX_COMP 8

void vmess(char *fmt, ...);
void verr(char *fmt, ...);

/* From born_shot.c */
int born_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec,
              int ixsrc, int izsrc, float **src_nwav,
              checkpointPar *chk,
              float *delta_rox, float *delta_roz,
              float *delta_l2m, float *delta_lam, float *delta_mul,
              const char *file_born,
              int ishot, int nshots, int fileno,
              int verbose);

/* From readResidual.c */
int readResidual(const char *filename, adjSrcPar *adj, modPar *mod, bndPar *bnd);
void freeResidual(adjSrcPar *adj);

/* From adj_shot.c */
int adj_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
             recPar *rec, adjSrcPar *adj,
             int ixsrc, int izsrc, float **src_nwav,
             checkpointPar *chk, snaPar *sna,
             float *grad1, float *grad2, float *grad3,
             int param, int verbose);

double wallclock_time(void);


/***********************************************************************
 * detectTRID -- Parse filename suffix to determine component TRID.
 *
 * Same logic as computeResidual.c:detectTRID.  Searches for a known
 * suffix before ".su" and returns the corresponding TRID value.
 *
 *   _rp   -> 1 (P/hydrophone)
 *   _rtxz -> 2 (Txz)
 *   _rtzz -> 3 (Tzz)
 *   _rtxx -> 4 (Txx)
 *   _rvx  -> 6 (Vx/geophone)
 *   _rvz  -> 7 (Vz/geophone)
 ***********************************************************************/
static int detectTRID(const char *filename)
{
	const char *dot;
	size_t baselen;

	dot = strstr(filename, ".su");
	if (!dot)
		baselen = strlen(filename);
	else
		baselen = (size_t)(dot - filename);

	if (baselen >= 5 && strncmp(filename + baselen - 5, "_rtxz", 5) == 0) return 2;
	if (baselen >= 5 && strncmp(filename + baselen - 5, "_rtzz", 5) == 0) return 3;
	if (baselen >= 5 && strncmp(filename + baselen - 5, "_rtxx", 5) == 0) return 4;
	if (baselen >= 4 && strncmp(filename + baselen - 4, "_rvx",  4) == 0) return 6;
	if (baselen >= 4 && strncmp(filename + baselen - 4, "_rvz",  4) == 0) return 7;
	if (baselen >= 3 && strncmp(filename + baselen - 3, "_rp",   3) == 0) return 1;

	return -1;
}


/***********************************************************************
 * combineBornFiles -- Merge per-component Born .su files into one.
 *
 * Reads each component file, sets the TRID header (auto-detected from
 * filename suffix), and writes all traces sequentially to a single
 * output file.  This creates the combined format readResidual expects.
 *
 * Parameters:
 *   ncomp      - number of component files
 *   born_files - array of per-component Born .su file paths
 *   out_file   - output combined file path
 *   verbose    - verbosity level
 *
 * Returns 0 on success.
 ***********************************************************************/
static int combineBornFiles(int ncomp, const char **born_files,
                            const char *out_file, int verbose)
{
	FILE *fp_in, *fp_out;
	segy hdr;
	float *data = NULL;
	int k, ns = 0, trid, global_tracl = 0;

	fp_out = fopen(out_file, "w");
	if (!fp_out)
		verr("combineBornFiles: Cannot open output file %s", out_file);

	for (k = 0; k < ncomp; k++) {
		trid = detectTRID(born_files[k]);
		if (trid < 0)
			verr("combineBornFiles: Cannot detect component from '%s'",
			     born_files[k]);

		fp_in = fopen(born_files[k], "r");
		if (!fp_in)
			verr("combineBornFiles: Cannot open Born file %s",
			     born_files[k]);

		while (fread(&hdr, 1, TRCBYTES, fp_in) == TRCBYTES) {
			/* Allocate data buffer on first trace */
			if (ns == 0) {
				ns = (int)hdr.ns;
				data = (float *)malloc(ns * sizeof(float));
				if (!data)
					verr("combineBornFiles: malloc failed for ns=%d", ns);
			}

			if ((int)hdr.ns != ns)
				verr("combineBornFiles: ns mismatch in %s (%d vs %d)",
				     born_files[k], hdr.ns, ns);

			if (fread(data, sizeof(float), ns, fp_in) != (size_t)ns)
				verr("combineBornFiles: Short read in %s", born_files[k]);

			/* Set TRID and sequential trace number */
			hdr.trid  = (short)trid;
			global_tracl++;
			hdr.tracl = global_tracl;
			hdr.tracf = global_tracl;

			fwrite(&hdr, 1, TRCBYTES, fp_out);
			fwrite(data, sizeof(float), ns, fp_out);
		}

		fclose(fp_in);

		if (verbose > 1)
			vmess("combineBornFiles: Component %d/%d: %s  trid=%d",
			      k + 1, ncomp, born_files[k], trid);
	}

	if (data) free(data);
	fclose(fp_out);

	if (verbose)
		vmess("combineBornFiles: %d traces, %d components -> %s",
		      global_tracl, ncomp, out_file);

	return 0;
}


/***********************************************************************
 * applyCosineTaper -- Apply cosine taper to adjoint source trace ends.
 *
 * Same logic as fwi_inversion.c:applyCosineTaper.
 ***********************************************************************/
static void applyCosineTaper(adjSrcPar *adj, int taper_len)
{
	int isrc, it;
	float *wav, w;

	if (taper_len <= 0 || adj->nt <= 2 * taper_len) return;

	for (isrc = 0; isrc < adj->nsrc; isrc++) {
		wav = &adj->wav[isrc * adj->nt];
		for (it = 0; it < taper_len; it++) {
			w = 0.5f * (1.0f - cosf(M_PI * it / taper_len));
			wav[it] *= w;
			wav[adj->nt - 1 - it] *= w;
		}
	}
}


/***********************************************************************
 * hess_shot -- Gauss-Newton Hessian-vector product for one shot.
 *
 * Orchestrates born_shot + adj_shot to compute H_GN * dm for a single
 * shot.  The forward checkpoints must already exist on disk (from the
 * preceding gradient computation; they are only read, not written).
 *
 * Parameters:
 *   mod, src, wav, bnd          - model, source, wavelet, boundary
 *   rec                         - receiver parameters
 *   ixsrc, izsrc, src_nwav     - source position and wavelet data
 *   chk                         - checkpoint parameters (pointing to
 *                                 existing checkpoint files)
 *   delta_rox..delta_mul        - perturbed FD coefficients from
 *                                 perturbFDcoefficients(opt->d)
 *   file_born                   - base name for Born .su output files
 *                                 (e.g., "fwi_rank000/born")
 *   comp_str                    - component suffix string, comma-sep
 *                                 (e.g., "_rvz" or "_rvz,_rvx")
 *   ishot, nshots, fileno       - shot indices for file naming
 *   res_taper                   - cosine taper length for Born data
 *                                 (0 = no taper)
 *   hd1, hd2, hd3              - OUTPUT: per-shot H*dm (padded grid,
 *                                 may contain accumulated values from
 *                                 previous shots; pass NULL to skip)
 *   param                       - 1=Lame, 2=velocity
 *   verbose                     - verbosity level
 *
 * Returns 0 on success.
 ***********************************************************************/
int hess_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec,
              int ixsrc, int izsrc, float **src_nwav,
              checkpointPar *chk,
              float *delta_rox, float *delta_roz,
              float *delta_l2m, float *delta_lam, float *delta_mul,
              const char *file_born, const char *comp_str,
              int ishot, int nshots, int fileno,
              int res_taper,
              float *hd1, float *hd2, float *hd3,
              int param, int verbose)
{
	adjSrcPar adj;
	double t0_wall;
	int ret;

	if (verbose)
		t0_wall = wallclock_time();

	/* ============================================================ */
	/* 1. Born forward propagation: J * dm -> Born receiver data    */
	/* ============================================================ */
	ret = born_shot(mod, src, wav, bnd, rec,
	                ixsrc, izsrc, src_nwav, chk,
	                delta_rox, delta_roz,
	                delta_l2m, delta_lam, delta_mul,
	                file_born, ishot, nshots, fileno, verbose);
	if (ret != 0)
		verr("hess_shot: born_shot failed for shot %d", ishot);

	/* ============================================================ */
	/* 2. Combine Born component .su files into residual format      */
	/*                                                               */
	/* born_shot writes per-component files (e.g., born_000_rvz.su). */
	/* readResidual expects a single file with TRID headers.         */
	/* combineBornFiles reads each component, sets TRID from the     */
	/* filename suffix, and concatenates into one file.              */
	/* ============================================================ */
	{
		char comp_buf[1024];
		char *comp_suffixes[MAX_COMP];
		int ncomp = 0;
		char *token;

		strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
		comp_buf[sizeof(comp_buf) - 1] = '\0';
		token = strtok(comp_buf, ",");
		while (token && ncomp < MAX_COMP) {
			comp_suffixes[ncomp++] = token;
			token = strtok(NULL, ",");
		}

		char born_names[MAX_COMP][512];
		const char *born_arr[MAX_COMP];
		char combined_file[512];
		int i;

		for (i = 0; i < ncomp; i++) {
			snprintf(born_names[i], sizeof(born_names[i]),
			         "%s_%03d%s.su", file_born, fileno, comp_suffixes[i]);
			born_arr[i] = born_names[i];
		}

		snprintf(combined_file, sizeof(combined_file),
		         "%s_combined.su", file_born);

		ret = combineBornFiles(ncomp, born_arr, combined_file, verbose);
		if (ret != 0)
			verr("hess_shot: combineBornFiles failed for shot %d", ishot);

		/* ======================================================== */
		/* 3. Read combined Born data into adjoint source structure  */
		/* ======================================================== */
		memset(&adj, 0, sizeof(adjSrcPar));
		readResidual(combined_file, &adj, mod, bnd);

		if (verbose)
			vmess("hess_shot: Born data loaded: %d traces, %d samples",
			      adj.nsrc, adj.nt);

		/* Apply cosine taper to Born data (same as gradient residual
		 * to suppress edge effects at trace ends) */
		if (res_taper > 0)
			applyCosineTaper(&adj, res_taper);
	}

	/* ============================================================ */
	/* 4. Second adjoint: J^T * d_born -> H_GN * dm                 */
	/*                                                               */
	/* adj_shot internally negates the Born data (line 271-275) to   */
	/* produce the correct second adjoint source: -R^T(d_born).      */
	/* The adjoint wavefield mu_2 satisfies A^T mu_2 = -R^T(d_born) */
	/* and accumGradient computes the cross-correlation with the     */
	/* re-propagated forward wavefield, giving H_GN * dm.            */
	/*                                                               */
	/* param controls parameterization:                               */
	/*   1 = Lame:     hd1=Hd_lam, hd2=Hd_mu,  hd3=Hd_rho          */
	/*   2 = velocity: hd1=Hd_Vp,  hd2=Hd_Vs,  hd3=Hd_rho          */
	/* ============================================================ */
	adj_shot(mod, src, wav, bnd, rec, &adj,
	         ixsrc, izsrc, src_nwav, chk, NULL,
	         hd1, hd2, hd3, param, verbose);

	/* ============================================================ */
	/* 5. Cleanup                                                    */
	/* ============================================================ */
	freeResidual(&adj);

	if (verbose)
		vmess("hess_shot: Shot %d H*dm completed in %.2f s",
		      ishot, wallclock_time() - t0_wall);

	return 0;
}
