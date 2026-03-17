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
 * The pseudo-Hessian diagonal H_ii = Σ_t K_i² is accumulated in Lamé
 * space (λ,μ,ρ) by accumGradient() in fwi_gradient.c. This file handles:
 *   1. Chain rule transformation to velocity diagonal (if param=2)
 *   2. Scaling: H̃_ii = m0_i² · H_ii  (chain rule for normalized space)
 *   3. Global damping: C = max(H̃_ii), store H̃_ii + θ·C
 *   4. Division in double precision + norm preservation
 *
 * References:
 *   Shin et al. (2001), Geophysics 66(6), 1895–1903.
 *   Métivier et al. (2013), Geophys. J. Int. 194(3), 1490–1518, eq 37-41.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "fdelfwi.h"

/* External logging */
void vmess(char *fmt, ...);


/*--------------------------------------------------------------------
 * buildBlockPrecond -- Build Shin diagonal pseudo-Hessian preconditioner.
 *
 * Pipeline:
 *   1. Extract padded diagonal hessian → flat arrays (3 arrays)
 *   2. Chain rule if param=2: H_v_pp = Σ_k J_kp² · H_L_kk
 *   3. Scaling: H̃_ii = m0_i² · H_ii
 *   4. Global damping: C = max(H̃_ii), store P_ii = H̃_ii + θ·C
 *
 * Output: P11,P22,P33 = H̃_ii + θ·C  (unnormalized)
 *         P12,P13,P23 = 0 (unused)
 *
 * Off-diagonal hess arrays are accepted for signature compatibility
 * but ignored.
 *--------------------------------------------------------------------*/
void buildBlockPrecond(
	float *hess_lam, float *hess_muu, float *hess_rho,
	float *hess_lam_muu, float *hess_lam_rho, float *hess_muu_rho,
	modPar *mod, bndPar *bnd,
	int param, int elastic,
	float precond_eps, float s1, float s2, float s3,
	float *P11, float *P12, float *P13,
	float *P22, float *P23, float *P33,
	int nmodel, int mpi_rank)
{
	int i, ix, iz;
	int ibndx, ibndz, n1;

	(void)hess_lam_muu; (void)hess_lam_rho; (void)hess_muu_rho;

	n1 = mod->naz;
	ibndx = mod->ioPx;
	ibndz = mod->ioPz;
	if (bnd->lef == 4 || bnd->lef == 2) ibndx += bnd->ntap;
	if (bnd->top == 4 || bnd->top == 2) ibndz += bnd->ntap;

	/* Extract 3 diagonal hessian components from padded → flat (nmodel) */
	float *h11 = (float *)calloc(nmodel, sizeof(float));
	float *h22 = (float *)calloc(nmodel, sizeof(float));
	float *h33 = (float *)calloc(nmodel, sizeof(float));

	for (ix = 0; ix < mod->nx; ix++) {
		for (iz = 0; iz < mod->nz; iz++) {
			int ig = (ix + ibndx) * n1 + (iz + ibndz);
			int im = ix * mod->nz + iz;
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
		vmess("Precond diag: raw Lame H: |H_ll|=%.4e |H_mm|=%.4e |H_rr|=%.4e",
		      max_h11, max_h22, max_h33);
	}

	/* If param=2: diagonal chain rule Lamé → velocity
	 * H_v_pp = Σ_k J_kp² · H_L_kk  (only diagonal of J^T diag(H_L) J)
	 *
	 * Jacobian J = ∂(λ,μ,ρ)/∂(Vp,Vs,ρ):
	 *   j11 = 2ρVp,  j12 = -4ρVs,  j13 = Vp²-2Vs²
	 *   j21 = 0,      j22 = 2ρVs,   j23 = Vs²
	 *   j31 = 0,      j32 = 0,       j33 = 1               */
	if (param == 2 && elastic) {
		for (i = 0; i < nmodel; i++) {
			ix = i / mod->nz;
			iz = i % mod->nz;
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

		if (mpi_rank == 0) {
			float max_h11 = 0.0f, max_h22 = 0.0f, max_h33 = 0.0f;
			for (i = 0; i < nmodel; i++) {
				float v;
				v = fabsf(h11[i]); if (v > max_h11) max_h11 = v;
				v = fabsf(h22[i]); if (v > max_h22) max_h22 = v;
				v = fabsf(h33[i]); if (v > max_h33) max_h33 = v;
			}
			vmess("Precond diag: post chain-rule H_vel: |H_VpVp|=%.4e |H_VsVs|=%.4e |H_rhorho|=%.4e",
			      max_h11, max_h22, max_h33);
		}
	}

	/* Scaling: H̃_ii = m0_i² · H_ii  (chain rule for normalized space) */
	for (i = 0; i < nmodel; i++) {
		h11[i] *= s1 * s1;
		h22[i] *= s2 * s2;
		h33[i] *= s3 * s3;
	}

	/* Global damping (Métivier eq 38-39):
	 *   C = max_j(H̃_jj)  over all params and grid points
	 *   P_ii = H̃_ii + θ·C                                   */
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
		vmess("Shin precond (Metivier eq 39): C_k=%.4e  theta=%.4e  beta=theta*C_k=%.4e",
		      C_k, precond_eps, beta);
		vmess("Shin precond: P_%s = H̃+β: [%.4e, %.4e]", n[0],
		      (double)h11[0] + beta, C_k + beta);
		vmess("Shin precond: P_%s = H̃+β: [%.4e, %.4e]", n[1],
		      beta, (double)h22[0] + beta);
		vmess("Shin precond: P_%s = H̃+β: [%.4e, %.4e]", n[2],
		      beta, (double)h33[0] + beta);

		/* Compute actual min/max for accurate diagnostics */
		float min_all = P11[0], max_all = P11[0];
		for (i = 0; i < nmodel; i++) {
			if (P11[i] < min_all) min_all = P11[i];
			if (P22[i] < min_all) min_all = P22[i];
			if (P33[i] < min_all) min_all = P33[i];
			if (P11[i] > max_all) max_all = P11[i];
			if (P22[i] > max_all) max_all = P22[i];
			if (P33[i] > max_all) max_all = P33[i];
		}
		vmess("Shin precond: P range [%.4e, %.4e]  cond=%.2f",
		      min_all, max_all, (double)max_all / (double)min_all);
	}

	free(h11); free(h22); free(h33);
}


/*--------------------------------------------------------------------
 * applyBlockPrecond -- Apply Shin preconditioner with norm preservation.
 *
 * Métivier et al. (2013), eq 39-41:
 *   z_i = g_i / P_ii                  (division in double precision)
 *   ν = ||g|| / ||z||                 (norm preservation factor)
 *   out_i = ν · z_i                   (preserves gradient norm)
 *
 * The preconditioner changes direction only, not magnitude.
 * Double precision is used throughout to avoid underflow when
 * g ~ O(10^-10) and P ~ O(10^30).
 *
 * P11,P22,P33 store H̃_ii + θ·C from buildBlockPrecond.
 * P12,P13,P23 are unused (zero).
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
	 * This ensures ||out|| = ||g||: the preconditioner changes
	 * direction only, preserving the gradient magnitude. */
	double norm_out_sq = 0.0;
	for (i = 0; i < ntot; i++)
		norm_out_sq += (double)out[i] * (double)out[i];

	if (norm_out_sq > 0.0 && norm_in_sq > 0.0) {
		float nu = (float)sqrt(norm_in_sq / norm_out_sq);
		for (i = 0; i < ntot; i++)
			out[i] *= nu;
	}
}
