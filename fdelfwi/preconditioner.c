/*
 * yang_precond.c -- Diagonal pseudo-Hessian preconditioner for FWI.
 *
 * Implements the Shin et al. (2001) preconditioner with the norm-preserving
 * formulation of Métivier et al. (2013):
 *
 *   C_k = max_j(H̃_jj)                              (eq 38)
 *   P_θ = diag( 1 / (H̃_ii + θ·C_k) )               (eq 39)
 *   P_{ν,θ} = ν · P_θ,  ν = ||∇f|| / ||P_θ·∇f||    (eq 40-41)
 *
 * Near-surface blending (Métivier, pers. comm.):
 *   Near sources/receivers, the pseudo-Hessian has singularities.
 *   A smooth depth-averaged scaling is blended with the full H̃:
 *     H̃_blend(x,z) = (1-w(z)) · H̃_avg(z) + w(z) · H̃(x,z)
 *   where w(z) ramps from 0 at surface to 1 at blend_depth.
 *
 * References:
 *   Shin et al. (2001), Geophysics 66(6), 1895–1903.
 *   Métivier et al. (2014), Geophys. J. Int. 194(3), 1490–1518, eq 37-41.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "fdelfwi.h"

/* External logging */
void vmess(char *fmt, ...);

/* File-scope state shared between build and apply */
static int stored_blend_nz = 0;
static int stored_nx = 0;
static int stored_nz = 0;


/*--------------------------------------------------------------------
 * applyDepthTaper -- Multiply array by exponential depth weight w(z).
 *
 *   w(iz) = 1 - exp( -(alpha * iz / ntaper)² )
 *
 *   ntaper = blend_nz (blend depth in grid points)
 *   alpha  = steepness parameter (user-supplied):
 *     alpha=0.30 → gentle taper, w(ntaper)=0.91
 *     alpha=0.50 → moderate,     w(ntaper)=0.97
 *     alpha=1.00 → steep,        w(ntaper)=1.00
 *
 * At iz=0: w=0 (full suppression).
 * For iz >> ntaper: w→1 asymptotically.
 *
 * Applied to H̃ before β (removes surface spikes from max),
 * and to the preconditioned gradient (removes surface artifacts).
 *--------------------------------------------------------------------*/
static float stored_taper_alpha = 0.30f;

static void applyDepthTaper(float *h, int nx, int nz, int blend_nz)
{
	int ix, iz;
	float alpha = stored_taper_alpha;
	if (blend_nz <= 0) return;
	float inv_ntaper = 1.0f / (float)blend_nz;

	for (iz = 0; iz < nz; iz++) {
		float arg = alpha * (float)iz * inv_ntaper;
		float w = 1.0f - expf(-arg * arg);
		if (w > 0.9999f) break;  /* w ≈ 1 for remaining depths */
		for (ix = 0; ix < nx; ix++)
			h[ix * nz + iz] *= w;
	}
}


/*--------------------------------------------------------------------
 * buildBlockPrecond -- Build Shin diagonal pseudo-Hessian preconditioner.
 *
 * Pipeline:
 *   1. Extract padded diagonal hessian → flat arrays (3 arrays)
 *   2. Chain rule if param=2: H_v_pp = Σ_k J_kp² · H_L_kk
 *   3. Scaling: H̃_ii = m0_i² · H_ii
 *   4. Depth blending: smooth near surface, full H̃ at depth
 *   5. Global damping: C = max(H̃_ii), store P_ii = H̃_ii + θ·C
 *
 * Output: P11,P22,P33 = H̃_ii + θ·C
 *         P12,P13,P23 = 0 (unused)
 *
 * blend_depth: depth in meters for the near-surface blending zone.
 *   0 = no blending (original Shin formula).
 *--------------------------------------------------------------------*/
void buildBlockPrecond(
	float *hess_lam, float *hess_muu, float *hess_rho,
	float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
	modPar *mod, bndPar *bnd,
	int param, int elastic,
	float precond_eps, float s1, float s2, float s3,
	float *P11, float *P12, float *P13,
	float *P22, float *P23, float *P33,
	int nmodel, int mpi_rank,
	float blend_depth, float taper_alpha,
	float *wfld_energy,
	float xmin, float xmax)
{
	int i, ix, iz;
	int ibndx, ibndz, n1;
	int nx = mod->nx;
	int nz = mod->nz;

	(void)hess_lam_muu; (void)hess_lam_rho; (void)hess_muu_rho;

	/* Store alpha for applyBlockPrecond's depth taper */
	stored_taper_alpha = (taper_alpha > 0.0f) ? taper_alpha : 5.0f;

	n1 = mod->naz;
	ibndx = mod->ioPx;
	ibndz = mod->ioPz;
	if (bnd->lef == 4 || bnd->lef == 2) ibndx += bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibndz += bnd->ntap;

	/* Extract 3 diagonal hessian components from padded → flat (nmodel) */
	float *h11 = (float *)calloc(nmodel, sizeof(float));
	float *h22 = (float *)calloc(nmodel, sizeof(float));
	float *h33 = (float *)calloc(nmodel, sizeof(float));

	for (ix = 0; ix < nx; ix++) {
		for (iz = 0; iz < nz; iz++) {
			int ig = (ix + ibndx) * n1 + (iz + ibndz);
			int im = ix * nz + iz;
			h11[im] = hess_lam[ig];
			h22[im] = hess_muu ? hess_muu[ig] : 0.0f;
			h33[im] = hess_rho[ig];
		}
	}

	/* Diagnostic: raw Lamé hessian diagonal magnitudes */
	if (mpi_rank == 0) {
		float max_h11 = 0.0f, max_h22 = 0.0f, max_h33 = 0.0f;
		for (i = 0; i < nmodel; i++) {
			float v;
			v = fabsf(h11[i]); if (v > max_h11) max_h11 = v;
			v = fabsf(h22[i]); if (v > max_h22) max_h22 = v;
			v = fabsf(h33[i]); if (v > max_h33) max_h33 = v;
		}
		vmess("Precond: raw Lame H: |H_ll|=%.4e |H_mm|=%.4e |H_rr|=%.4e",
		      max_h11, max_h22, max_h33);
	}

	/* If param=2: diagonal chain rule Lamé → velocity
	 * H_v_pp = Σ_k J_kp² · H_L_kk  (diagonal of J^T diag(H_L) J) */
	if (param == 2 && elastic) {
		for (i = 0; i < nmodel; i++) {
			ix = i / nz;
			iz = i % nz;
			int ig = (ix + ibndx) * n1 + (iz + ibndz);
			float rho_val = mod->rho[ig];
			float vp  = mod->cp[ig];
			float vs  = mod->cs[ig];
			float vp2 = vp * vp;
			float vs2 = vs * vs;

			float j11 = 2.0f * rho_val * vp;
			float j12 = -4.0f * rho_val * vs;
			float j13 = vp2 - 2.0f * vs2;
			float j22 = 2.0f * rho_val * vs;
			float j23 = vs2;

			float Hll = h11[i], Hmm = h22[i], Hrr = h33[i];

			h11[i] = j11*j11 * Hll;
			h22[i] = j12*j12 * Hll + j22*j22 * Hmm;
			h33[i] = j13*j13 * Hll + j23*j23 * Hmm + Hrr;
		}
	}

	/* Scaling: H̃_ii = m0_i² · H_ii  (chain rule for normalized space) */
	for (i = 0; i < nmodel; i++) {
		h11[i] *= s1 * s1;
		h22[i] *= s2 * s2;
		h33[i] *= s3 * s3;
	}

	/* Depth taper on H̃ (Métivier, pers. comm.):
	 * Zero out H̃ near the surface to remove source/receiver singularities.
	 * This ensures β = θ·max(H̃) is driven by depth values, not surface spikes.
	 * The same taper is applied to the output in applyBlockPrecond. */
	int blend_nz = (blend_depth > 0.0f && mod->dz > 0.0f)
	             ? (int)(blend_depth / mod->dz + 0.5f) : 0;
	stored_blend_nz = blend_nz;
	stored_nx = nx;
	stored_nz = nz;
	if (blend_nz > 0) {
		applyDepthTaper(h11, nx, nz, blend_nz);
		applyDepthTaper(h22, nx, nz, blend_nz);
		applyDepthTaper(h33, nx, nz, blend_nz);
		if (mpi_rank == 0)
			vmess("Precond: depth taper = %.0f m (%d grid points)",
			      blend_depth, blend_nz);
	}

	/* ================================================================
	 * Build preconditioner diagonal P_ii.
	 *
	 * If wfld_energy is provided: Plessix & Mulder (2004) EPRECOND=3
	 *   We(x,z) = sqrt(Ws(x,z)) * [asinh((xmax-x)/z) - asinh((xmin-x)/z)]
	 *   P_ii = We * s_i² + β
	 *
	 * Otherwise: Yang/Shin diagonal pseudo-Hessian
	 *   P_ii = H̃_ii + β
	 *
	 * In both cases: β = θ · max(P_ii)  (Métivier eq 38-39)
	 * ================================================================ */
	if (wfld_energy) {
		/* Plessix & Mulder (2004): source energy × analytical receiver term */
		float *We = (float *)calloc(nmodel, sizeof(float));
		float dz = mod->dz;
		float dx_grid = mod->dx;

		for (ix = 0; ix < nx; ix++) {
			for (iz = 0; iz < nz; iz++) {
				int ig = (ix + ibndx) * n1 + (iz + ibndz);
				int im = ix * nz + iz;
				float x_phys = (float)(ix + ibndx) * dx_grid;
				float z_phys = (float)(iz + ibndz) * dz;

				/* Avoid division by zero at z=0 */
				if (z_phys < dz) z_phys = dz;

				float Ws = wfld_energy[ig];
				float rcv_term = asinhf((xmax - x_phys) / z_phys)
				               - asinhf((xmin - x_phys) / z_phys);
				We[im] = sqrtf(Ws > 0.0f ? Ws : 0.0f) * rcv_term;
			}
		}

		/* Apply parameter scaling and store in P arrays */
		for (i = 0; i < nmodel; i++) {
			h11[i] = We[i] * s1 * s1;
			h22[i] = We[i] * s2 * s2;
			h33[i] = We[i] * s3 * s3;
		}
		free(We);

		if (mpi_rank == 0)
			vmess("Precond: Plessix & Mulder (EPRECOND=3), xmin=%.0f xmax=%.0f",
			      xmin, xmax);
	}

	/* Global damping (Métivier eq 38-39):
	 *   C = max_j(P_jj)  over all params and grid points
	 *   P_ii = P_ii + θ·C                                   */
	double C_k = 0.0;
	for (i = 0; i < nmodel; i++) {
		if ((double)h11[i] > C_k) C_k = (double)h11[i];
		if ((double)h22[i] > C_k) C_k = (double)h22[i];
		if ((double)h33[i] > C_k) C_k = (double)h33[i];
	}
	double beta = (double)precond_eps * C_k;

	for (i = 0; i < nmodel; i++) {
		P11[i] = (float)((double)h11[i] + beta);
		P22[i] = (float)((double)h22[i] + beta);
		P33[i] = (float)((double)h33[i] + beta);
	}

	/* Off-diagonal outputs zeroed (unused in diagonal preconditioner) */
	memset(P12, 0, nmodel * sizeof(float));
	memset(P13, 0, nmodel * sizeof(float));
	memset(P23, 0, nmodel * sizeof(float));

	/* Diagnostics */
	if (mpi_rank == 0) {
		const char *lame_n[] = {"lam", "mu", "rho"};
		const char *vel_n[]  = {"Vp", "Vs", "rho"};
		const char **n = (param == 2) ? vel_n : lame_n;
		vmess("Precond (Shin/Metivier): C_k=%.4e  theta=%.4e  beta=%.4e",
		      C_k, precond_eps, beta);

		float min_all = P11[0], max_all = P11[0];
		for (i = 0; i < nmodel; i++) {
			if (P11[i] < min_all) min_all = P11[i];
			if (P22[i] < min_all) min_all = P22[i];
			if (P33[i] < min_all) min_all = P33[i];
			if (P11[i] > max_all) max_all = P11[i];
			if (P22[i] > max_all) max_all = P22[i];
			if (P33[i] > max_all) max_all = P33[i];
		}
		vmess("Precond: P_%s range [%.4e, %.4e]", n[0],
		      P11[0], P11[(nx/2)*nz]);
		vmess("Precond: P_%s range [%.4e, %.4e]", n[1],
		      P22[0], P22[(nx/2)*nz]);
		vmess("Precond: P_%s range [%.4e, %.4e]", n[2],
		      P33[0], P33[(nx/2)*nz]);
		vmess("Precond: P global [%.4e, %.4e]  cond=%.2f",
		      min_all, max_all, (double)max_all / (double)min_all);
	}

	free(h11); free(h22); free(h33);
}


/*--------------------------------------------------------------------
 * applyBlockPrecond -- Apply Shin preconditioner with norm preservation.
 *
 * Métivier et al. (2014), eq 39-41:
 *   z_i = g_i / P_ii                  (division in double precision)
 *   ν = ||g|| / ||z||                 (norm preservation factor)
 *   out_i = ν · z_i                   (preserves gradient norm)
 *
 * The preconditioner changes direction only, not magnitude.
 * Double precision avoids underflow when g ~ O(10^-10) and P ~ O(10^30).
 *
 * out and in may be the same pointer (in-place operation).
 *--------------------------------------------------------------------*/
void applyBlockPrecond(
	float *out, const float *in,
	const float *P11, const float *P12, const float *P13,
	const float *P22, const float *P23, const float *P33,
	int nmodel, int nparam)
{
	int i, ntot;

	(void)P12; (void)P13; (void)P23;

	ntot = nparam * nmodel;

	/* Step 1: Compute ||g||² before the solve (needed for in-place case) */
	double norm_in_sq = 0.0;
	for (i = 0; i < ntot; i++)
		norm_in_sq += (double)in[i] * (double)in[i];

	/* Step 2: Apply P_θ · g = g / (H̃ + β) in double precision */
	if (nparam == 3) {
		for (i = 0; i < nmodel; i++) {
			out[i]          = (float)((double)in[i]          / (double)P11[i]);
			out[nmodel+i]   = (float)((double)in[nmodel+i]   / (double)P22[i]);
			out[2*nmodel+i] = (float)((double)in[2*nmodel+i] / (double)P33[i]);
		}
	} else {
		for (i = 0; i < nmodel; i++) {
			out[i]        = (float)((double)in[i]        / (double)P11[i]);
			out[nmodel+i] = (float)((double)in[nmodel+i] / (double)P33[i]);
		}
	}

	/* Step 3: Norm preservation (Métivier eq 40-41):
	 *   ν = ||g|| / ||P_θ·g||
	 *   out = ν · P_θ · g
	 * Ensures ||out|| = ||g||: preconditioner changes direction only. */
	double norm_out_sq = 0.0;
	for (i = 0; i < ntot; i++)
		norm_out_sq += (double)out[i] * (double)out[i];

	if (norm_out_sq > 0.0 && norm_in_sq > 0.0) {
		float nu = (float)sqrt(norm_in_sq / norm_out_sq);
		for (i = 0; i < ntot; i++)
			out[i] *= nu;
	}

	/* Step 4: Depth taper — zero out near-surface artifacts.
	 * Same w(z) taper that was applied to H̃ in buildBlockPrecond. */
	if (stored_blend_nz > 0 && stored_nx > 0 && stored_nz > 0) {
		int p;
		for (p = 0; p < nparam; p++)
			applyDepthTaper(out + (size_t)p * nmodel,
			                stored_nx, stored_nz, stored_blend_nz);
	}
}
