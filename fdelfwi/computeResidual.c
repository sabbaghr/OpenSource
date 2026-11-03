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
float computeDataRMS(const char *filename)
{
	FILE *fp;
	segy hdr;
	float *buf;
	double sum2 = 0.0;
	long nsamp_total = 0;
	int ns;

	fp = fopen(filename, "r");
	if (!fp)
		verr("computeDataRMS: Cannot open %s", filename);

	/* Read first header to get ns */
	if (fread(&hdr, 1, TRCBYTES, fp) != TRCBYTES)
		verr("computeDataRMS: Cannot read header from %s", filename);
	ns = (int)hdr.ns;
	rewind(fp);

	buf = (float *)malloc(ns * sizeof(float));

	while (fread(&hdr, 1, TRCBYTES, fp) == TRCBYTES) {
		int isamp;
		if (fread(buf, sizeof(float), ns, fp) != (size_t)ns) break;
		for (isamp = 0; isamp < ns; isamp++)
			sum2 += (double)buf[isamp] * (double)buf[isamp];
		nsamp_total += ns;
	}

	free(buf);
	fclose(fp);

	if (nsamp_total == 0) return 1.0f;
	return (float)sqrt(sum2 / (double)nsamp_total);
}


float computeResidual(int ncomp, const char **obs_files, const char **syn_files,
                      const char *res_file, misfitType mtype,
                      const float *comp_weights, const energyWeight *ew,
                      const khPar *kh, int verbose)
{
	FILE *fp_obs[MAX_COMP], *fp_syn[MAX_COMP], *fp_res;
	segy hdr_obs[MAX_COMP], hdr_syn;
	float *buf_obs, *buf_syn, *res_buf, *psi_buf;
	float *r_rec, *psi_rec;
	double *mis_comp;
	float misfit;
	int trid[MAX_COMP];
	int k, itr, isamp, ns, dt_us, ntr;
	int idx_txx = -1, idx_tzz = -1;
	int kh_vx = -1, kh_vz = -1, kh_txz = -1, kh_tzz = -1, kh_txx = -1;
	double kh_total = 0.0, kh_shear = 0.0, kh_normal = 0.0;
	int global_tracl;

	if (!obs_files || !syn_files || !res_file || ncomp < 1)
		verr("computeResidual: Invalid arguments.");
	if (ncomp > MAX_COMP)
		verr("computeResidual: ncomp=%d exceeds MAX_COMP=%d.", ncomp, MAX_COMP);

	if (mtype != MISFIT_L2 && mtype != MISFIT_CORRELATION && mtype != MISFIT_KH)
		verr("computeResidual: Unknown misfit type %d (0=L2, 1=correlation, 2=kh).",
		     (int)mtype);

	if (ew && comp_weights)
		verr("computeResidual: comp_weights (RMS weighting) and ew (energy norm) "
		     "are mutually exclusive.");
	if (ew && mtype == MISFIT_CORRELATION)
		verr("computeResidual: the energy norm applies to the L2 misfit only; the "
		     "cross-correlation misfit is already self-normalizing.");

	/* The energy flux r = dt_i dv_i is a product of one traction and one
	 * velocity channel, so ANY weighting that rescales velocities relative
	 * to stresses destroys the balance that gives r its meaning.  See the
	 * "WHY data_weight MUST BE 0" block in kh_misfit.c. */
	if (mtype == MISFIT_KH) {
		if (!kh)
			verr("computeResidual: misfit=kh needs the receiver quadrature "
			     "weights; khWeightsBuild was not called.");
		if (comp_weights || ew)
			verr("computeResidual: misfit=kh requires data_weight=0.  The "
			     "energy flux pairs a traction with a velocity, so rescaling "
			     "the velocity channels relative to the stress channels "
			     "destroys the flux interpretation.");
	}
	else if (kh) {
		verr("computeResidual: receiver quadrature weights were supplied but "
		     "misfit=%d is not the energy-flux objective.", (int)mtype);
	}

	/* -------------------------------------------------------- */
	/* Open all component pairs and validate consistency         */
	/* -------------------------------------------------------- */
	ns = 0; dt_us = 0; ntr = 0;

	for (k = 0; k < ncomp; k++) {
		int ns_o, dt_o, ns_s, dt_s, ntr_o, ntr_s;

		/* Auto-detect TRID from observed filename suffix */
		trid[k] = detectTRID(obs_files[k]);
		if (trid[k] < 0)
			verr("computeResidual: Cannot detect component type from filename '%s'.\n"
			     "  Expected suffix: _rp, _rvx, _rvz, _rtxx, _rtzz, or _rtxz before .su",
			     obs_files[k]);

		if (trid[k] == 4) idx_txx = k;
		if (trid[k] == 3) idx_tzz = k;

		if (verbose)
			vmess("computeResidual: Component %d/%d: obs=%s  syn=%s  trid=%d",
				k + 1, ncomp, obs_files[k], syn_files[k], trid[k]);

		fp_obs[k] = fopen(obs_files[k], "r");
		if (!fp_obs[k])
			verr("computeResidual: Cannot open observed file %s", obs_files[k]);

		fp_syn[k] = fopen(syn_files[k], "r");
		if (!fp_syn[k])
			verr("computeResidual: Cannot open synthetic file %s", syn_files[k]);

		ntr_o = countTraces(fp_obs[k], obs_files[k], &ns_o, &dt_o);
		ntr_s = countTraces(fp_syn[k], syn_files[k], &ns_s, &dt_s);

		if (ntr_o != ntr_s)
			verr("computeResidual: Trace count mismatch for component %d: obs=%d, syn=%d",
				k, ntr_o, ntr_s);
		if (ns_o != ns_s)
			verr("computeResidual: Sample count mismatch for component %d: obs ns=%d, syn ns=%d",
				k, ns_o, ns_s);
		if (dt_o != dt_s)
			verr("computeResidual: Sample interval mismatch for component %d: obs dt=%d, syn dt=%d",
				k, dt_o, dt_s);

		/* Cross-component consistency: the trace-major loop below reads
		 * trace itr from every component as the SAME receiver, so the
		 * gathers must be congruent. */
		if (k == 0) { ns = ns_o; dt_us = dt_o; ntr = ntr_o; }
		else {
			if (ns_o != ns)
				verr("computeResidual: Component %d has ns=%d, but component 0 has ns=%d",
					k, ns_o, ns);
			if (dt_o != dt_us)
				verr("computeResidual: Component %d has dt=%d, but component 0 has dt=%d",
					k, dt_o, dt_us);
			if (ntr_o != ntr)
				verr("computeResidual: Component %d has %d traces, but component 0 has %d",
					k, ntr_o, ntr);
		}
	}

	if (ew && ew->nrec != ntr)
		verr("computeResidual: energy weights were built for %d receivers but the "
		     "gathers hold %d traces per component.", ew->nrec, ntr);

	if (kh) {
		if (kh->nrec != ntr)
			verr("computeResidual: receiver quadrature weights were built for %d "
			     "receivers but the gathers hold %d traces per component.",
			     kh->nrec, ntr);
		/* Once per gather, so the per-sample loop stays branch-free. */
		khResolveChannels(ncomp, trid, &kh_vx, &kh_vz, &kh_txz, &kh_tzz, &kh_txx);
		if (verbose)
			vmess("computeResidual: K-H channels vx=%d vz=%d txz=%d tzz=%d txx=%d, "
			      "n_z=%+.1f", kh_vx, kh_vz, kh_txz, kh_tzz, kh_txx, kh->nz);
	}

	/* Open output residual file */
	fp_res = fopen(res_file, "w");
	if (!fp_res) verr("computeResidual: Cannot open output file %s", res_file);

	/* -------------------------------------------------------- */
	/* Allocate: one trace per component, plus residual/adjoint  */
	/* -------------------------------------------------------- */
	buf_obs = (float *)malloc((size_t)ncomp * ns * sizeof(float));
	buf_syn = (float *)malloc((size_t)ncomp * ns * sizeof(float));
	res_buf = (float *)malloc((size_t)ncomp * ns * sizeof(float));
	psi_buf = (float *)malloc((size_t)ncomp * ns * sizeof(float));
	r_rec   = (float *)malloc((size_t)ncomp * sizeof(float));
	psi_rec = (float *)malloc((size_t)ncomp * sizeof(float));
	mis_comp = (double *)calloc(ncomp, sizeof(double));
	if (!buf_obs || !buf_syn || !res_buf || !psi_buf || !r_rec || !psi_rec || !mis_comp)
		verr("computeResidual: Memory allocation failed for ncomp=%d, ns=%d", ncomp, ns);

	misfit = 0.0f;
	global_tracl = 0;

	/* -------------------------------------------------------- */
	/* Trace-major loop: all components of one receiver together */
	/* -------------------------------------------------------- */
	for (itr = 0; itr < ntr; itr++) {

		/* ---- Read one trace from every component ---- */
		for (k = 0; k < ncomp; k++) {
			if (fread(&hdr_obs[k], 1, TRCBYTES, fp_obs[k]) != TRCBYTES)
				verr("computeResidual: Error reading obs header, component %d, trace %d", k, itr);
			if (fread(buf_obs + (size_t)k * ns, sizeof(float), ns, fp_obs[k]) != (size_t)ns)
				verr("computeResidual: Error reading obs data, component %d, trace %d", k, itr);

			if (fread(&hdr_syn, 1, TRCBYTES, fp_syn[k]) != TRCBYTES)
				verr("computeResidual: Error reading syn header, component %d, trace %d", k, itr);
			if (fread(buf_syn + (size_t)k * ns, sizeof(float), ns, fp_syn[k]) != (size_t)ns)
				verr("computeResidual: Error reading syn data, component %d, trace %d", k, itr);

			/* Verify receiver positions match */
			if (hdr_obs[k].gx != hdr_syn.gx || hdr_obs[k].gelev != hdr_syn.gelev)
				vwarn("computeResidual: Receiver position mismatch at component %d, trace %d: "
				      "obs(gx=%d,gelev=%d) vs syn(gx=%d,gelev=%d)",
				      k, itr, hdr_obs[k].gx, hdr_obs[k].gelev, hdr_syn.gx, hdr_syn.gelev);

			/* All components must describe the same receiver, otherwise the
			 * cross-component metric Q would mix unrelated locations. */
			if (k > 0 && (hdr_obs[k].gx != hdr_obs[0].gx ||
			              hdr_obs[k].gelev != hdr_obs[0].gelev))
				verr("computeResidual: trace %d of component %d is at "
				     "(gx=%d,gelev=%d) but component 0 is at (gx=%d,gelev=%d). "
				     "Components must be recorded at identical receivers.",
				     itr, k, hdr_obs[k].gx, hdr_obs[k].gelev,
				     hdr_obs[0].gx, hdr_obs[0].gelev);
		}

		if (mtype == MISFIT_KH) {
			/* Kirchhoff-Helmholtz energy flux, linear penalty P(r) = r:
			 *   J   = sum_r w_r sum_n n_z (dsig_xz dv_x + dsig_zz dv_z)
			 *   psi = dJ/dd_syn
			 *
			 * With P linear, r is a symmetric bilinear form in the
			 * residual, so this is structurally the same as the L2 branch
			 * in a metric Q -- just one that is indefinite and couples
			 * (vx,txz) and (vz,tzz) instead of (txx,tzz).  Hence
			 * psi = Q dd and J = 0.5 dd^T Q dd exactly as above, and
			 * mis_comp splits the shear and normal flux in half between
			 * each channel pair. */
			for (k = 0; k < ncomp; k++) {
				const float *o = buf_obs + (size_t)k * ns;
				const float *s = buf_syn + (size_t)k * ns;
				float *rr = res_buf + (size_t)k * ns;
				for (isamp = 0; isamp < ns; isamp++)
					rr[isamp] = s[isamp] - o[isamp];
			}

			for (isamp = 0; isamp < ns; isamp++) {
				for (k = 0; k < ncomp; k++)
					r_rec[k] = res_buf[(size_t)k * ns + isamp];

				kh_total += khApply(ncomp, psi_rec, r_rec, kh, itr,
				                    kh_vx, kh_vz, kh_txz, kh_tzz,
				                    &kh_shear, &kh_normal);

				for (k = 0; k < ncomp; k++) {
					psi_buf[(size_t)k * ns + isamp] = psi_rec[k];
					mis_comp[k] += 0.5 * (double)r_rec[k] * (double)psi_rec[k];
				}
			}
		}
		else if (mtype == MISFIT_CORRELATION) {
			/* Per-trace normalized cross-correlation -- independent across
			 * components (no metric coupling; see the guard above). */
			for (k = 0; k < ncomp; k++) {
				const float *o = buf_obs + (size_t)k * ns;
				const float *s = buf_syn + (size_t)k * ns;
				float *out = psi_buf + (size_t)k * ns;
				double norm_obs2 = 0.0, norm_syn2 = 0.0, dot_os = 0.0;
				float w2 = comp_weights ? comp_weights[k] * comp_weights[k] : 1.0f;

				for (isamp = 0; isamp < ns; isamp++) {
					norm_obs2 += (double)o[isamp] * (double)o[isamp];
					norm_syn2 += (double)s[isamp] * (double)s[isamp];
					dot_os    += (double)o[isamp] * (double)s[isamp];
				}
				double norm_obs = sqrt(norm_obs2);
				double norm_syn = sqrt(norm_syn2);

				if (norm_obs < 1.0e-30 || norm_syn < 1.0e-30) {
					/* Zero-amplitude trace -- skip (write zeros) */
					memset(out, 0, ns * sizeof(float));
				}
				else {
					double cos_theta = dot_os / (norm_obs * norm_syn);
					mis_comp[k] += w2 * (1.0 - cos_theta);
					for (isamp = 0; isamp < ns; isamp++) {
						double adj = -(double)o[isamp] / (norm_obs * norm_syn)
						             + cos_theta * (double)s[isamp] / norm_syn2;
						out[isamp] = (float)(w2 * adj);
					}
				}
			}
		}
		else {
			/* L2 misfit in the metric Q:
			 *   J   = 0.5 * dd^T Q dd
			 *   psi = dJ/dd_syn = Q dd
			 * Q = I            when neither weighting is active
			 * Q = diag(w_k^2)  for RMS (Brossier) weighting
			 * Q = W_r^T W_r    for the energy norm (couples Txx/Tzz) */
			for (k = 0; k < ncomp; k++) {
				const float *o = buf_obs + (size_t)k * ns;
				const float *s = buf_syn + (size_t)k * ns;
				float *rr = res_buf + (size_t)k * ns;
				for (isamp = 0; isamp < ns; isamp++)
					rr[isamp] = s[isamp] - o[isamp];
			}

			if (ew) {
				/* Metric couples components, so step through time samples
				 * and apply Q to the cross-component vector. */
				for (isamp = 0; isamp < ns; isamp++) {
					for (k = 0; k < ncomp; k++)
						r_rec[k] = res_buf[(size_t)k * ns + isamp];

					energyWeightApply(ncomp, trid, r_rec, psi_rec, ew, itr,
					                  idx_txx, idx_tzz);

					for (k = 0; k < ncomp; k++) {
						psi_buf[(size_t)k * ns + isamp] = psi_rec[k];
						mis_comp[k] += 0.5 * (double)r_rec[k] * (double)psi_rec[k];
					}
				}
			}
			else {
				for (k = 0; k < ncomp; k++) {
					const float *rr = res_buf + (size_t)k * ns;
					float *out = psi_buf + (size_t)k * ns;
					float w2 = comp_weights ? comp_weights[k] * comp_weights[k] : 1.0f;
					double acc = 0.0;
					for (isamp = 0; isamp < ns; isamp++) {
						out[isamp] = w2 * rr[isamp];
						acc += 0.5 * (double)rr[isamp] * (double)out[isamp];
					}
					mis_comp[k] += acc;
				}
			}
		}

		/* ---- Write the adjoint source trace for every component ---- */
		for (k = 0; k < ncomp; k++) {
			hdr_obs[k].trid  = (short)trid[k];
			global_tracl++;
			hdr_obs[k].tracl = global_tracl;
			hdr_obs[k].tracf = global_tracl;

			if (fwrite(&hdr_obs[k], 1, TRCBYTES, fp_res) != TRCBYTES)
				verr("computeResidual: Error writing residual header, trace %d", global_tracl);
			if (fwrite(psi_buf + (size_t)k * ns, sizeof(float), ns, fp_res) != (size_t)ns)
				verr("computeResidual: Error writing residual data, trace %d", global_tracl);
		}
	}

	for (k = 0; k < ncomp; k++) {
		misfit += (float)mis_comp[k];
		if (verbose)
			vmess("computeResidual: Component %d: %d traces, trid=%d, misfit=%.6e",
				k + 1, ntr, trid[k], mis_comp[k]);
		fclose(fp_obs[k]);
		fclose(fp_syn[k]);
	}

	free(buf_obs);
	free(buf_syn);
	free(res_buf);
	free(psi_buf);
	free(r_rec);
	free(psi_rec);
	free(mis_comp);
	fclose(fp_res);

	if (verbose)
		vmess("computeResidual: Total misfit = %.6e  (%d traces, %d components)",
			misfit, global_tracl, ncomp);

	/* The two totals are computed independently (khApply's return vs the
	 * 0.5 dd^T Q dd bookkeeping above), so a mismatch means the bilinear
	 * form and the adjoint source have drifted apart -- exactly the error
	 * that would silently corrupt the gradient of a signed objective. */
	if (mtype == MISFIT_KH && verbose)
		vmess("computeResidual: K-H flux J = %.6e  (shear %.6e + normal %.6e), "
		      "direct sum %.6e", misfit, kh_shear, kh_normal, kh_total);

	return misfit;
}
