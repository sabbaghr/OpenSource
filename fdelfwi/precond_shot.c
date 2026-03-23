/*--------------------------------------------------------------------
 * precond_shot.c -- Receiver wavefield backpropagation for P4 preconditioner.
 *
 * Backpropagates synthetic data at receiver positions to obtain the
 * receiver wavefield ṽ, then cross-correlates with the forward wavefield
 * v (from checkpoints) to build the P4 pseudo-Hessian:
 *
 *   H̃_ij = |∫ K_i(v) · K̃_j(ṽ) dt|    (Liu et al. 2022, eq A22)
 *
 * This function mirrors adj_shot's checkpoint-based backpropagation
 * but does NOT compute the gradient. It only accumulates the
 * cross-correlation Hessian products.
 *
 * Called only when precond >= 4. Requires checkpoints from the
 * forward modeling to still be available.
 *--------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "fdelfwi.h"

/* External functions (same as adj_shot uses) */
void vmess(char *fmt, ...);
void verr(char *fmt, ...);

void readCheckpoint(checkpointPar *chk, int snap, wflPar *wfl);

/* Kernel dispatch (from adj_shot.c, non-static) */
void callKernel(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                int it, int ixsrc, int izsrc,
                float **src_nwav, wflPar *wfl, int verbose);
void callAdjKernel(modPar *mod, adjSrcPar *adj, bndPar *bnd,
                   int it, wflPar *wfl,
                   int rec_delay, int rec_skipdt, int verbose);

/* From fwi_gradient.c */
void accumHessianP4(modPar *mod, bndPar *bnd,
                    float *fwd_vx, float *fwd_vz,
                    float *fwd_vx_prev, float *fwd_vz_prev,
                    float *rcv_vx, float *rcv_vz,
                    float *rcv_vx_prev, float *rcv_vz_prev,
                    float dt,
                    float *hess_lam, float *hess_muu, float *hess_rho,
                    float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho);

/* From readResidual.c — reuse to read synthetic data in adjSrcPar format */
void readResidual(const char *filename, adjSrcPar *adj, modPar *mod, bndPar *bnd);
void freeResidual(adjSrcPar *adj);


/*--------------------------------------------------------------------
 * precond_shot -- Backpropagate synthetic data for P4 Hessian.
 *
 * Uses the same checkpoint/segment loop as adj_shot but:
 * - Reads synthetic data (not residuals) as adjoint source
 * - Calls accumHessianP4 instead of accumGradient
 * - Does NOT compute the gradient
 *
 * The synthetic data file is constructed from syn_base, fileno, comp_str:
 *   e.g., "syn_000_rvz.su" from syn_base="syn", fileno=0, comp_str="_rvz"
 *--------------------------------------------------------------------*/
int precond_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                 recPar *rec,
                 int ixsrc, int izsrc, float **src_nwav,
                 checkpointPar *chk,
                 float *hess_lam, float *hess_muu, float *hess_rho,
                 float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
                 int verbose,
                 const char *syn_base, int fileno, const char *comp_str)
{
	wflPar wfl_fwd, wfl_rcv;
	adjSrcPar syn_adj;
	int k, j, it, seg_start, seg_end, nsteps;
	size_t sizem = (size_t)mod->nax * mod->naz;
	float dt = mod->dt;

	/* Allocate forward wavefield (for re-propagation from checkpoints) */
	wfl_fwd.vx  = (float *)calloc(sizem, sizeof(float));
	wfl_fwd.vz  = (float *)calloc(sizem, sizeof(float));
	wfl_fwd.txx = (float *)calloc(sizem, sizeof(float));
	wfl_fwd.tzz = (float *)calloc(sizem, sizeof(float));
	wfl_fwd.txz = (float *)calloc(sizem, sizeof(float));

	/* Allocate receiver wavefield (for synthetic backpropagation) */
	wfl_rcv.vx  = (float *)calloc(sizem, sizeof(float));
	wfl_rcv.vz  = (float *)calloc(sizem, sizeof(float));
	wfl_rcv.txx = (float *)calloc(sizem, sizeof(float));
	wfl_rcv.tzz = (float *)calloc(sizem, sizeof(float));
	wfl_rcv.txz = (float *)calloc(sizem, sizeof(float));

	/* Read synthetic data as adjoint source.
	 * Parse comp_str (e.g. "_rvx", "_rvz", "_rvx,_rvz", "_rvx,_rvz,_rp")
	 * to determine which components to read, matching adj_shot's
	 * multicomponent support. Each component file is read via
	 * readResidual() (which sets TRID-based type/orient) and appended
	 * into a single adjSrcPar for backpropagation. */
	{
		#define PRECOND_MAX_COMP 8
		char comp_buf[1024];
		char *comp_suffixes[PRECOND_MAX_COMP];
		int ncomps = 0;
		int total_traces = 0;
		char *token;

		/* Parse comma-separated comp_str into individual suffixes */
		strncpy(comp_buf, comp_str, sizeof(comp_buf) - 1);
		comp_buf[sizeof(comp_buf) - 1] = '\0';
		token = strtok(comp_buf, ",");
		while (token && ncomps < PRECOND_MAX_COMP) {
			comp_suffixes[ncomps++] = token;
			token = strtok(NULL, ",");
		}

		if (ncomps == 0) {
			vmess("precond_shot: WARNING — empty comp_str, skipping");
			goto cleanup;
		}

		memset(&syn_adj, 0, sizeof(adjSrcPar));

		for (int ic = 0; ic < ncomps; ic++) {
			char syn_file[512];
			snprintf(syn_file, sizeof(syn_file), "%s_%03d%s.su",
			         syn_base, fileno, comp_suffixes[ic]);

			FILE *fp_check = fopen(syn_file, "r");
			if (!fp_check) {
				vmess("precond_shot: WARNING — file %s not found, skipping component",
				      syn_file);
				continue;
			}
			fclose(fp_check);

			if (total_traces == 0) {
				/* First component: read normally */
				readResidual(syn_file, &syn_adj, mod, bnd);
				total_traces = syn_adj.nsrc;
			} else {
				/* Additional components: read into temp and append */
				adjSrcPar tmp;
				memset(&tmp, 0, sizeof(adjSrcPar));
				readResidual(syn_file, &tmp, mod, bnd);

				/* Reallocate and append */
				int new_total = total_traces + tmp.nsrc;
				syn_adj.xi = (size_t *)realloc(syn_adj.xi, new_total * sizeof(size_t));
				syn_adj.zi = (size_t *)realloc(syn_adj.zi, new_total * sizeof(size_t));
				syn_adj.typ = (int *)realloc(syn_adj.typ, new_total * sizeof(int));
				syn_adj.orient = (int *)realloc(syn_adj.orient, new_total * sizeof(int));
				syn_adj.x = (float *)realloc(syn_adj.x, new_total * sizeof(float));
				syn_adj.z = (float *)realloc(syn_adj.z, new_total * sizeof(float));
				syn_adj.wav = (float *)realloc(syn_adj.wav,
				              (size_t)new_total * syn_adj.nt * sizeof(float));

				for (int k = 0; k < tmp.nsrc; k++) {
					syn_adj.xi[total_traces + k] = tmp.xi[k];
					syn_adj.zi[total_traces + k] = tmp.zi[k];
					syn_adj.typ[total_traces + k] = tmp.typ[k];
					syn_adj.orient[total_traces + k] = tmp.orient[k];
					syn_adj.x[total_traces + k] = tmp.x[k];
					syn_adj.z[total_traces + k] = tmp.z[k];
					memcpy(&syn_adj.wav[(total_traces + k) * syn_adj.nt],
					       &tmp.wav[k * tmp.nt],
					       syn_adj.nt * sizeof(float));
				}
				syn_adj.nsrc = new_total;
				total_traces = new_total;
				freeResidual(&tmp);
			}
		}

		if (total_traces == 0) {
			vmess("precond_shot: WARNING — no synthetic files found for comp=%s, skipping",
			      comp_str);
			goto cleanup;
		}

		vmess("precond_shot: read %d traces (%d components: %s) (nt=%d)",
		      syn_adj.nsrc, ncomps, comp_str, syn_adj.nt);
	}

	/* NOTE: We do NOT negate the synthetic data (unlike adj_shot which
	 * negates residuals for the Lagrangian sign convention). The P4
	 * Hessian takes the absolute value at the end anyway. */

	/* Segment timing (same as adj_shot) */
	int it0 = 0;
	int it1 = wav->nt - 1;
	if (rec->delay > 0) it0 = rec->delay;

	/* Allocate forward re-propagation buffers */
	int max_nsteps = chk->skipdt;
	{
		int last_start = chk->delay + (chk->nsnap - 1) * chk->skipdt;
		int last_n = it1 - last_start;
		if (last_n > max_nsteps) max_nsteps = last_n;
	}

	float *buf_vx  = (float *)malloc((size_t)max_nsteps * sizem * sizeof(float));
	float *buf_vz  = (float *)malloc((size_t)max_nsteps * sizem * sizeof(float));

	/* Also need receiver wavefield from previous timestep for ∂ₜṽ */
	float *rcv_vx_prev = (float *)calloc(sizem, sizeof(float));
	float *rcv_vz_prev = (float *)calloc(sizem, sizeof(float));

	if (verbose > 0)
		vmess("precond_shot: %d segments, max_nsteps=%d",
		      chk->nsnap, max_nsteps);

	/* ================================================================
	 * Segment loop (backward through checkpoints)
	 * Same structure as adj_shot.
	 * ================================================================ */
	for (k = chk->nsnap - 1; k >= 0; k--) {
		seg_start = chk->delay + k * chk->skipdt;
		seg_end = (k == chk->nsnap - 1) ? it1 : chk->delay + (k + 1) * chk->skipdt;
		nsteps = seg_end - seg_start;

		/* Step 1: Load checkpoint and re-propagate forward */
		readCheckpoint(chk, k, &wfl_fwd);
		memcpy(buf_vx, wfl_fwd.vx, sizem * sizeof(float));
		memcpy(buf_vz, wfl_fwd.vz, sizem * sizeof(float));

		for (j = 1; j < nsteps; j++) {
			it = seg_start + j;
#pragma omp parallel default(shared)
{
			callKernel(mod, src, wav, bnd, it, ixsrc, izsrc,
			           src_nwav, &wfl_fwd, 0);
}
			memcpy(buf_vx + (size_t)j * sizem, wfl_fwd.vx, sizem * sizeof(float));
			memcpy(buf_vz + (size_t)j * sizem, wfl_fwd.vz, sizem * sizeof(float));
		}

		/* Step 2: Backward through segment — inject synthetics,
		 *         cross-correlate K(fwd) × K̃(rcv) */
		for (j = nsteps - 1; j >= 0; j--) {
			it = seg_start + j;

			/* Save receiver wavefield from previous step for ∂ₜṽ */
			memcpy(rcv_vx_prev, wfl_rcv.vx, sizem * sizeof(float));
			memcpy(rcv_vz_prev, wfl_rcv.vz, sizem * sizeof(float));

			/* Inject synthetic data into receiver wavefield */
#pragma omp parallel default(shared)
{
			callAdjKernel(mod, &syn_adj, bnd, it, &wfl_rcv,
			              rec->delay, rec->skipdt, 0);
}

			/* Cross-correlate forward × receiver kernels */
			accumHessianP4(mod, bnd,
				buf_vx + (size_t)j * sizem,
				buf_vz + (size_t)j * sizem,
				(j > 0) ? buf_vx + (size_t)(j-1) * sizem : NULL,
				(j > 0) ? buf_vz + (size_t)(j-1) * sizem : NULL,
				wfl_rcv.vx, wfl_rcv.vz,
				rcv_vx_prev, rcv_vz_prev,
				dt,
				hess_lam, hess_muu, hess_rho,
				hess_lam_muu, hess_lam_rho, hess_muu_rho);
		}
	}

	/* Diagnostic: check if anything was accumulated */
	{
		float max_hl = 0, max_hm = 0, max_hr = 0;
		float max_rvx = 0, max_rvz = 0;
		size_t i;
		for (i = 0; i < sizem; i++) {
			if (fabsf(hess_lam[i]) > max_hl) max_hl = fabsf(hess_lam[i]);
			if (hess_muu && fabsf(hess_muu[i]) > max_hm) max_hm = fabsf(hess_muu[i]);
			if (hess_rho && fabsf(hess_rho[i]) > max_hr) max_hr = fabsf(hess_rho[i]);
			if (fabsf(wfl_rcv.vx[i]) > max_rvx) max_rvx = fabsf(wfl_rcv.vx[i]);
			if (fabsf(wfl_rcv.vz[i]) > max_rvz) max_rvz = fabsf(wfl_rcv.vz[i]);
		}
		vmess("precond_shot diag: max|H_ll|=%.4e max|H_mm|=%.4e max|H_rr|=%.4e",
		      max_hl, max_hm, max_hr);
		vmess("precond_shot diag: max|rcv_vx|=%.4e max|rcv_vz|=%.4e",
		      max_rvx, max_rvz);
	}

	/* Take absolute value of accumulated cross-correlations
	 * (Liu et al. 2022, eq A22: H̃ = |∫ K·K̃ dt|) */
	{
		size_t i;
		if (hess_lam)
			for (i = 0; i < sizem; i++) hess_lam[i] = fabsf(hess_lam[i]);
		if (hess_muu)
			for (i = 0; i < sizem; i++) hess_muu[i] = fabsf(hess_muu[i]);
		if (hess_rho)
			for (i = 0; i < sizem; i++) hess_rho[i] = fabsf(hess_rho[i]);
		if (hess_lam_muu)
			for (i = 0; i < sizem; i++) hess_lam_muu[i] = fabsf(hess_lam_muu[i]);
		if (hess_lam_rho)
			for (i = 0; i < sizem; i++) hess_lam_rho[i] = fabsf(hess_lam_rho[i]);
		if (hess_muu_rho)
			for (i = 0; i < sizem; i++) hess_muu_rho[i] = fabsf(hess_muu_rho[i]);
	}

cleanup:
	/* Cleanup */
	if (syn_adj.wav) freeResidual(&syn_adj);
	free(wfl_fwd.vx); free(wfl_fwd.vz);
	free(wfl_fwd.txx); free(wfl_fwd.tzz); free(wfl_fwd.txz);
	free(wfl_rcv.vx); free(wfl_rcv.vz);
	free(wfl_rcv.txx); free(wfl_rcv.tzz); free(wfl_rcv.txz);
	if (buf_vx) free(buf_vx);
	if (buf_vz) free(buf_vz);
	if (rcv_vx_prev) free(rcv_vx_prev);
	if (rcv_vz_prev) free(rcv_vz_prev);

	return 0;
}
