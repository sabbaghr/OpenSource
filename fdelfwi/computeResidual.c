#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include"fdelfwi.h"
#include"segy.h"
#include"par.h"

/**
 * computeResidual -- Compute adjoint sources and misfit for one shot.
 *
 * Reads ncomp pairs of (observed, synthetic) SU files, computes the
 * adjoint source traces according to the chosen misfit function,
 * writes all adjoint traces to a single output SU file with correct
 * TRID headers (auto-detected from observed filename suffix), and
 * returns the total misfit value.
 *
 * TRID auto-detection from filename suffix:
 *   _rp   -> trid=1 (P/hydrophone)
 *   _rtxz -> trid=2 (Txz)
 *   _rtzz -> trid=3 (Tzz)
 *   _rtxx -> trid=4 (Txx)
 *   _rvx  -> trid=6 (Vx/geophone)
 *   _rvz  -> trid=7 (Vz/geophone)
 *
 * AUTHOR:
 *   Based on fdelfwi SU I/O conventions (writeRec.c, readResidual.c).
 */


/***********************************************************************
 * detectTRID -- Parse filename suffix to determine component TRID.
 *
 * Searches for a known suffix before ".su" (e.g., "_rvz.su").
 * Returns the TRID value, or -1 if no known suffix is found.
 ***********************************************************************/
static int detectTRID(const char *filename)
{
	const char *dot;
	size_t baselen;

	/* Find last ".su" in filename */
	dot = strstr(filename, ".su");
	if (!dot) {
		/* No .su extension -- try matching at end of string */
		baselen = strlen(filename);
	}
	else {
		baselen = (size_t)(dot - filename);
	}

	/* Check suffixes from longest to shortest to avoid partial matches */
	if (baselen >= 5 && strncmp(filename + baselen - 5, "_rtxz", 5) == 0) return 2;
	if (baselen >= 5 && strncmp(filename + baselen - 5, "_rtzz", 5) == 0) return 3;
	if (baselen >= 5 && strncmp(filename + baselen - 5, "_rtxx", 5) == 0) return 4;
	if (baselen >= 4 && strncmp(filename + baselen - 4, "_rvx",  4) == 0) return 6;
	if (baselen >= 4 && strncmp(filename + baselen - 4, "_rvz",  4) == 0) return 7;
	if (baselen >= 3 && strncmp(filename + baselen - 3, "_rp",   3) == 0) return 1;

	return -1;
}


/***********************************************************************
 * countTraces -- Count traces in an SU file.
 *
 * Reads the first header to get ns, then counts by seeking.
 * Returns the number of traces, sets *ns_out and *dt_us_out.
 * Rewinds the file before returning.
 ***********************************************************************/
static int countTraces(FILE *fp, const char *fname, int *ns_out, int *dt_us_out)
{
	segy hdr;
	off_t filesize;
	size_t tracesize;
	int ntr;

	if (fread(&hdr, 1, TRCBYTES, fp) != TRCBYTES)
		verr("computeResidual: Cannot read header from %s", fname);

	*ns_out    = (int)hdr.ns;
	*dt_us_out = (int)hdr.dt;

	tracesize = TRCBYTES + (size_t)hdr.ns * sizeof(float);

	/* Get file size */
	if (fseeko(fp, 0, SEEK_END)) verr("computeResidual: fseek failed on %s", fname);
	filesize = ftello(fp);

	if (filesize % (off_t)tracesize != 0)
		vwarn("computeResidual: File %s size not a multiple of trace size.", fname);

	ntr = (int)(filesize / (off_t)tracesize);

	/* Rewind */
	if (fseeko(fp, 0, SEEK_SET)) verr("computeResidual: fseek failed on %s", fname);

	return ntr;
}


/***********************************************************************
 * computeResidual -- Main function.
 ***********************************************************************/
float computeResidual(int ncomp, const char **obs_files, const char **syn_files,
                      const char *res_file, misfitType mtype, int verbose)
{
	FILE *fp_obs, *fp_syn, *fp_res;
	segy hdr_obs, hdr_syn;
	float *buf_obs, *buf_syn;
	float misfit;
	int k, itr, ns, ns_syn, dt_obs, dt_syn;
	int ntr_obs, ntr_syn;
	int trid;
	int global_tracl;
	float dt_sec;

	if (!obs_files || !syn_files || !res_file || ncomp < 1)
		verr("computeResidual: Invalid arguments.");

	if (mtype != MISFIT_L2)
		verr("computeResidual: Only MISFIT_L2 is currently implemented (requested %d).", (int)mtype);

	/* Open output residual file */
	fp_res = fopen(res_file, "w");
	if (!fp_res) verr("computeResidual: Cannot open output file %s", res_file);

	misfit = 0.0f;
	global_tracl = 0;
	ns = 0;

	/* -------------------------------------------------------- */
	/* Process each component pair                               */
	/* -------------------------------------------------------- */
	for (k = 0; k < ncomp; k++) {

		/* Auto-detect TRID from observed filename suffix */
		trid = detectTRID(obs_files[k]);
		if (trid < 0)
			verr("computeResidual: Cannot detect component type from filename '%s'.\n"
			     "  Expected suffix: _rp, _rvx, _rvz, _rtxx, _rtzz, or _rtxz before .su",
			     obs_files[k]);

		if (verbose)
			vmess("computeResidual: Component %d/%d: obs=%s  syn=%s  trid=%d",
				k + 1, ncomp, obs_files[k], syn_files[k], trid);

		/* Open observed and synthetic files */
		fp_obs = fopen(obs_files[k], "r");
		if (!fp_obs)
			verr("computeResidual: Cannot open observed file %s", obs_files[k]);

		fp_syn = fopen(syn_files[k], "r");
		if (!fp_syn)
			verr("computeResidual: Cannot open synthetic file %s", syn_files[k]);

		/* Count traces and validate */
		int ns_obs_file, dt_obs_file, ns_syn_file, dt_syn_file;
		ntr_obs = countTraces(fp_obs, obs_files[k], &ns_obs_file, &dt_obs_file);
		ntr_syn = countTraces(fp_syn, syn_files[k], &ns_syn_file, &dt_syn_file);

		if (ntr_obs != ntr_syn)
			verr("computeResidual: Trace count mismatch for component %d: obs=%d, syn=%d",
				k, ntr_obs, ntr_syn);
		if (ns_obs_file != ns_syn_file)
			verr("computeResidual: Sample count mismatch for component %d: obs ns=%d, syn ns=%d",
				k, ns_obs_file, ns_syn_file);
		if (dt_obs_file != dt_syn_file)
			verr("computeResidual: Sample interval mismatch for component %d: obs dt=%d, syn dt=%d",
				k, dt_obs_file, dt_syn_file);

		ns     = ns_obs_file;
		dt_obs = dt_obs_file;
		dt_sec = (float)dt_obs * 1.0e-6f;

		/* Cross-component consistency (ns and dt must match across all components) */
		if (k == 0) {
			ns_syn = ns;
			dt_syn = dt_obs;
		} else {
			if (ns_obs_file != ns_syn)
				verr("computeResidual: Component %d has ns=%d, but component 0 has ns=%d",
					k, ns_obs_file, ns_syn);
			if (dt_obs_file != dt_syn)
				verr("computeResidual: Component %d has dt=%d, but component 0 has dt=%d",
					k, dt_obs_file, dt_syn);
		}

		/* Allocate trace buffers */
		buf_obs = (float *)malloc(ns * sizeof(float));
		buf_syn = (float *)malloc(ns * sizeof(float));
		if (!buf_obs || !buf_syn)
			verr("computeResidual: Memory allocation failed for ns=%d", ns);

		/* ---- Process traces ---- */
		for (itr = 0; itr < ntr_obs; itr++) {
			int isamp;

			/* Read observed trace */
			if (fread(&hdr_obs, 1, TRCBYTES, fp_obs) != TRCBYTES)
				verr("computeResidual: Error reading obs header, component %d, trace %d", k, itr);
			if (fread(buf_obs, sizeof(float), ns, fp_obs) != (size_t)ns)
				verr("computeResidual: Error reading obs data, component %d, trace %d", k, itr);

			/* Read synthetic trace */
			if (fread(&hdr_syn, 1, TRCBYTES, fp_syn) != TRCBYTES)
				verr("computeResidual: Error reading syn header, component %d, trace %d", k, itr);
			if (fread(buf_syn, sizeof(float), ns, fp_syn) != (size_t)ns)
				verr("computeResidual: Error reading syn data, component %d, trace %d", k, itr);

			/* Verify receiver positions match */
			if (hdr_obs.gx != hdr_syn.gx || hdr_obs.gelev != hdr_syn.gelev)
				vwarn("computeResidual: Receiver position mismatch at component %d, trace %d: "
				      "obs(gx=%d,gelev=%d) vs syn(gx=%d,gelev=%d)",
				      k, itr, hdr_obs.gx, hdr_obs.gelev, hdr_syn.gx, hdr_syn.gelev);

			/* Compute adjoint source and misfit (L2)
			 * Misfit uses discrete norm J = 0.5*sum(r^2) so that the
			 * adjoint source dJ/dd = r is consistent without dt_rec. */
			for (isamp = 0; isamp < ns; isamp++) {
				float r = buf_syn[isamp] - buf_obs[isamp];
				misfit += 0.5f * r * r;
				buf_obs[isamp] = r;  /* reuse buffer for residual */
			}

			/* Prepare output header: copy observed header, set TRID */
			hdr_obs.trid  = (short)trid;
			global_tracl++;
			hdr_obs.tracl = global_tracl;
			hdr_obs.tracf = global_tracl;

			/* Write residual trace */
			if (fwrite(&hdr_obs, 1, TRCBYTES, fp_res) != TRCBYTES)
				verr("computeResidual: Error writing residual header, trace %d", global_tracl);
			if (fwrite(buf_obs, sizeof(float), ns, fp_res) != (size_t)ns)
				verr("computeResidual: Error writing residual data, trace %d", global_tracl);
		}

		free(buf_obs);
		free(buf_syn);
		fclose(fp_obs);
		fclose(fp_syn);

		if (verbose)
			vmess("computeResidual: Component %d: %d traces, trid=%d, partial misfit=%.6e",
				k + 1, ntr_obs, trid, misfit);
	}

	fclose(fp_res);

	if (verbose)
		vmess("computeResidual: Total misfit = %.6e  (%d traces, %d components)",
			misfit, global_tracl, ncomp);

	return misfit;
}
