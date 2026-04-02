/*
 * bandpass_filter.c - Bandpass filtering for multiscale FWI.
 *
 * Two filtering mechanisms:
 *
 * 1. Observed data: filtered via Seismic Unix `sufilter` (zero-phase,
 *    sine-squared tapered bandpass). Done once per frequency band.
 *
 * 2. Source wavelet: filtered via FFT (genfft rc1fft/cr1fft) with a
 *    cosine-tapered spectral window. Done once per frequency band.
 *    All synthetic data is then naturally band-limited.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "genfft.h"
#include "fdelfwi.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void verr(char *fmt, ...);
void vmess(char *fmt, ...);


/*--------------------------------------------------------------------
 * optimalFFTsize -- Find smallest FFT-friendly size >= n.
 *
 * Returns the smallest integer >= n whose prime factors are only
 * 2, 3, 5 (efficient for genfft / MKL FFTs).
 *--------------------------------------------------------------------*/
static int optimalFFTsize(int n)
{
	int m;
	if (n <= 1) return 1;
	for (m = n; ; m++) {
		int t = m;
		while (t % 2 == 0) t /= 2;
		while (t % 3 == 0) t /= 3;
		while (t % 5 == 0) t /= 5;
		if (t == 1) return m;
	}
}


/*--------------------------------------------------------------------
 * computeTaperParams -- Compute 4-corner taper from flo/fhi.
 *
 * Taper width = 10% of bandwidth, clamped to [0.1, 0.5*bw] Hz.
 * Returns f1 < f2 <= f3 < f4 for trapezoidal spectral window.
 *--------------------------------------------------------------------*/
static void computeTaperParams(float flo, float fhi,
                               float *f1, float *f2, float *f3, float *f4)
{
	float bw = (fhi > 0.0f ? fhi : 500.0f) - (flo > 0.0f ? flo : 0.0f);
	float tw = 0.10f * bw;
	if (tw < 0.5f) tw = 0.5f;
	if (tw > 0.5f * bw) tw = 0.5f * bw;
	if (tw < 0.1f) tw = 0.1f;

	*f1 = (flo > 0.0f) ? flo - tw : 0.0f;
	*f2 = (flo > 0.0f) ? flo       : 0.0f;
	*f3 = (fhi > 0.0f) ? fhi       : 500.0f;
	*f4 = (fhi > 0.0f) ? fhi + tw  : 500.0f;
	if (*f1 < 0.0f) *f1 = 0.0f;
}


/*--------------------------------------------------------------------
 * bandpass_filter_sufile -- Apply bandpass filter to an SU file.
 *
 * Calls `sufilter` via system().  The input file is filtered in-place.
 *
 * Parameters:
 *   filename  - Path to the .su file (filtered in-place)
 *   flo       - Low corner frequency (Hz). 0 = no highpass
 *   fhi       - High corner frequency (Hz). 0 = no lowpass
 *--------------------------------------------------------------------*/
void bandpass_filter_sufile(const char *filename, float flo, float fhi)
{
	char cmd[2048];
	char tmpfile[1100];
	float f1, f2, f3, f4;
	int ret;

	if (flo <= 0.0f && fhi <= 0.0f) return;

	computeTaperParams(flo, fhi, &f1, &f2, &f3, &f4);

	snprintf(tmpfile, sizeof(tmpfile), "%s.filt_tmp", filename);

	if (flo > 0.0f && fhi > 0.0f) {
		snprintf(cmd, sizeof(cmd),
		         "sufilter < \"%s\" f=%.2f,%.2f,%.2f,%.2f > \"%s\"",
		         filename, f1, f2, f3, f4, tmpfile);
	} else if (flo > 0.0f) {
		snprintf(cmd, sizeof(cmd),
		         "sufilter < \"%s\" f=%.2f,%.2f amps=0.,1. > \"%s\"",
		         filename, f1, f2, tmpfile);
	} else {
		snprintf(cmd, sizeof(cmd),
		         "sufilter < \"%s\" f=%.2f,%.2f amps=1.,0. > \"%s\"",
		         filename, f3, f4, tmpfile);
	}

	ret = system(cmd);
	if (ret != 0) {
		verr("bandpass_filter_sufile: sufilter failed (ret=%d) on %s\n"
		     "  Command: %s", ret, filename, cmd);
	}

	snprintf(cmd, sizeof(cmd), "mv \"%s\" \"%s\"", tmpfile, filename);
	ret = system(cmd);
	if (ret != 0) {
		verr("bandpass_filter_sufile: mv failed (ret=%d): %s -> %s",
		     ret, tmpfile, filename);
	}
}




/*--------------------------------------------------------------------
 * bandpass_filter_obsdata -- Pre-filter all observed data files.
 *
 * For each shot and component, filters the observed .su file in-place.
 * Called once at the start of each frequency band.
 *
 * On the first band (iband==0), saves unfiltered backups (.unfilt).
 * On subsequent bands, restores from backup before filtering.
 *--------------------------------------------------------------------*/
void bandpass_filter_obsdata(const char *file_obs, const char *comp_str,
                             int nshots, float flo, float fhi,
                             int iband, int mpi_rank)
{
	char comp_buf[1024];
	char *comp_suffixes[8];
	char *token;
	int ncomp = 0, ishot, ic;
	char fname[512], bkname[512], cmd[1100];

	if (flo <= 0.0f && fhi <= 0.0f) return;

	/* Parse component suffixes */
	strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
	comp_buf[sizeof(comp_buf) - 1] = '\0';
	token = strtok(comp_buf, ",");
	while (token && ncomp < 8) {
		comp_suffixes[ncomp++] = token;
		token = strtok(NULL, ",");
	}

	for (ishot = 0; ishot < nshots; ishot++) {
		for (ic = 0; ic < ncomp; ic++) {
			snprintf(fname, sizeof(fname), "%s_%03d%s.su",
			         file_obs, ishot, comp_suffixes[ic]);

			if (iband == 0) {
				snprintf(bkname, sizeof(bkname), "%s.unfilt", fname);
				snprintf(cmd, sizeof(cmd), "cp \"%s\" \"%s\"", fname, bkname);
				system(cmd);
			} else {
				snprintf(bkname, sizeof(bkname), "%s.unfilt", fname);
				snprintf(cmd, sizeof(cmd), "cp \"%s\" \"%s\"", bkname, fname);
				system(cmd);
			}

			bandpass_filter_sufile(fname, flo, fhi);
		}
	}

	/* Also filter hydrophone (rp) observed data if it exists */
	for (ishot = 0; ishot < nshots; ishot++) {
		snprintf(fname, sizeof(fname), "%s_%03d_rp.su", file_obs, ishot);
		FILE *fp = fopen(fname, "r");
		if (fp) {
			fclose(fp);
			if (iband == 0) {
				snprintf(bkname, sizeof(bkname), "%s.unfilt", fname);
				snprintf(cmd, sizeof(cmd), "cp \"%s\" \"%s\"", fname, bkname);
				system(cmd);
			} else {
				snprintf(bkname, sizeof(bkname), "%s.unfilt", fname);
				snprintf(cmd, sizeof(cmd), "cp \"%s\" \"%s\"", bkname, fname);
				system(cmd);
			}
			bandpass_filter_sufile(fname, flo, fhi);
		}
	}
}


/*--------------------------------------------------------------------
 * bandpass_filter_syndata -- Filter synthetic data for one shot.
 *
 * Called after forward modeling + hydrophone computation. Filters
 * each component .su file in-place using sufilter. This guarantees
 * identical filtering to the observed data.
 *--------------------------------------------------------------------*/
void bandpass_filter_syndata(const char *file_rcv, const char *comp_str,
                             int fileno, float flo, float fhi)
{
	char comp_buf[1024];
	char *comp_suffixes[8];
	char *token;
	int ncomp = 0, ic;
	char fname[512];

	if (flo <= 0.0f && fhi <= 0.0f) return;

	strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
	comp_buf[sizeof(comp_buf) - 1] = '\0';
	token = strtok(comp_buf, ",");
	while (token && ncomp < 8) {
		comp_suffixes[ncomp++] = token;
		token = strtok(NULL, ",");
	}

	for (ic = 0; ic < ncomp; ic++) {
		snprintf(fname, sizeof(fname), "%s_%03d%s.su",
		         file_rcv, fileno, comp_suffixes[ic]);
		bandpass_filter_sufile(fname, flo, fhi);
	}

	/* Also filter hydrophone if it exists */
	snprintf(fname, sizeof(fname), "%s_%03d_rp.su", file_rcv, fileno);
	{
		FILE *fp = fopen(fname, "r");
		if (fp) {
			fclose(fp);
			bandpass_filter_sufile(fname, flo, fhi);
		}
	}
}


/*--------------------------------------------------------------------
 * bandpass_cleanup_obsbackups -- Remove .unfilt backup files.
 *--------------------------------------------------------------------*/
void bandpass_cleanup_obsbackups(const char *file_obs, const char *comp_str,
                                 int nshots)
{
	char comp_buf[1024];
	char *comp_suffixes[8];
	char *token;
	int ncomp = 0, ishot, ic;
	char bkname[512];

	strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
	comp_buf[sizeof(comp_buf) - 1] = '\0';
	token = strtok(comp_buf, ",");
	while (token && ncomp < 8) {
		comp_suffixes[ncomp++] = token;
		token = strtok(NULL, ",");
	}

	for (ishot = 0; ishot < nshots; ishot++) {
		for (ic = 0; ic < ncomp; ic++) {
			snprintf(bkname, sizeof(bkname), "%s_%03d%s.su.unfilt",
			         file_obs, ishot, comp_suffixes[ic]);
			remove(bkname);
		}
		snprintf(bkname, sizeof(bkname), "%s_%03d_rp.su.unfilt",
		         file_obs, ishot);
		remove(bkname);
	}
}
