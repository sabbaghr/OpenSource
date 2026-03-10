/*
 * yang_precond.c -- Diagonal pseudo-Hessian preconditioner for FWI.
 *
 * Implements the Shin et al. (2001) diagonal preconditioner:
 *   Δm̃_i = g̃_i / (H̃_ii + β),  β = θ · max(H̃_ii)
 *
 * The pseudo-Hessian diagonal H_ii = Σ_t K_i² is accumulated in Lamé
 * space (λ,μ,ρ) by accumGradient() in fwi_gradient.c. This file handles:
 *   1. Chain rule transformation to velocity diagonal (if param=2)
 *   2. Brossier m0 scaling: H̃_ii = m0_i² · H_ii
 *   3. Global damping: β = θ · max(H̃_ii) over all params and grid points
 *   4. Pointwise division: Δm̃_i = g̃_i / (H̃_ii + β)
 *
 * References:
 *   Shin et al. (2001), Geophysics 66(6), 1895–1903.
 *   Brossier et al. (2011), Computers & Geosciences 37, 444–455.
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
 *   3. Brossier scaling: H̃_ii = m0_i² · H_ii
 *   4. Global damping: β = θ · max(H̃_ii), store H̃_ii + β
 *
 * Output: P11,P22,P33 = H̃_11+β, H̃_22+β, H̃_33+β
 *         P12,P13,P23 = 0 (unused, kept for signature compatibility)
 *
 * Off-diagonal hess arrays (hess_lam_muu etc.) are accepted for
 * signature compatibility but ignored.
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
	 *   j31 = 0,      j32 = 0,       j33 = 1
	 *
	 * H_VpVp  = j11² · H_ll
	 * H_VsVs  = j12² · H_ll + j22² · H_mm
	 * H_rhorho = j13² · H_ll + j23² · H_mm + H_rr                  */
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

	/* Brossier m0 scaling: H̃_ii = m0_i² · H_ii */
	for (i = 0; i < nmodel; i++) {
		h11[i] *= s1 * s1;
		h22[i] *= s2 * s2;
		h33[i] *= s3 * s3;
	}

	/* Normalize H̃ by global max and add damping θ:
	 *   h_i = H̃_ii / max(H̃_ii) ∈ [0, 1]
	 *   P_ii = h_i + θ            ∈ [θ, 1+θ]
	 *
	 * The 1/max(H̃) factor is a global scalar absorbed by the
	 * L-BFGS γ-scaling or line search step length. */
	float global_max = 0.0f;
	for (i = 0; i < nmodel; i++) {
		if (h11[i] > global_max) global_max = h11[i];
		if (h22[i] > global_max) global_max = h22[i];
		if (h33[i] > global_max) global_max = h33[i];
	}
	float inv_max = (global_max > 0.0f) ? 1.0f / global_max : 1.0f;

	for (i = 0; i < nmodel; i++) {
		P11[i] = h11[i] * inv_max + precond_eps;
		P22[i] = h22[i] * inv_max + precond_eps;
		P33[i] = h33[i] * inv_max + precond_eps;
	}

	/* Off-diagonal outputs zeroed (unused in diagonal preconditioner) */
	memset(P12, 0, nmodel * sizeof(float));
	memset(P13, 0, nmodel * sizeof(float));
	memset(P23, 0, nmodel * sizeof(float));

	/* Diagnostics */
	if (mpi_rank == 0) {
		float max_P11 = 0.0f, max_P22 = 0.0f, max_P33 = 0.0f;
		float min_P11 = P11[0], min_P22 = P22[0], min_P33 = P33[0];
		for (i = 0; i < nmodel; i++) {
			if (P11[i] > max_P11) max_P11 = P11[i];
			if (P22[i] > max_P22) max_P22 = P22[i];
			if (P33[i] > max_P33) max_P33 = P33[i];
			if (P11[i] < min_P11) min_P11 = P11[i];
			if (P22[i] < min_P22) min_P22 = P22[i];
			if (P33[i] < min_P33) min_P33 = P33[i];
		}
		const char *lame_n[] = {"lam", "mu", "rho"};
		const char *vel_n[]  = {"Vp", "Vs", "rho"};
		const char **n = (param == 2) ? vel_n : lame_n;
		float min_all = min_P11;
		if (min_P22 < min_all) min_all = min_P22;
		if (min_P33 < min_all) min_all = min_P33;
		float max_all = max_P11;
		if (max_P22 > max_all) max_all = max_P22;
		if (max_P33 > max_all) max_all = max_P33;
		vmess("Shin precond: global_max(H̃_ii)=%.4e  theta=%.4e",
		      global_max, precond_eps);
		vmess("Shin precond: P_%s = h_norm+θ: [%.6f, %.6f]", n[0], min_P11, max_P11);
		vmess("Shin precond: P_%s = h_norm+θ: [%.6f, %.6f]", n[1], min_P22, max_P22);
		vmess("Shin precond: P_%s = h_norm+θ: [%.6f, %.6f]", n[2], min_P33, max_P33);
		vmess("Shin precond: cond(P) = %.2f  (max=%.6f, min=%.6f)",
		      max_all / min_all, max_all, min_all);
	}

	free(h11); free(h22); free(h33);
}


/*--------------------------------------------------------------------
 * applyBlockPrecond -- Apply diagonal preconditioner: Δm̃_i = g̃_i / P_ii.
 *
 * P11,P22,P33 store the damped diagonal H̃_ii + β from buildBlockPrecond.
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
	int i;

	(void)P12; (void)P13; (void)P23;

	if (nparam == 3) {
		for (i = 0; i < nmodel; i++) {
			out[i]           = in[i]           / P11[i];
			out[nmodel+i]    = in[nmodel+i]    / P22[i];
			out[2*nmodel+i]  = in[2*nmodel+i]  / P33[i];
		}
	} else {
		/* 2-parameter (acoustic): params 1 and 3 */
		for (i = 0; i < nmodel; i++) {
			out[i]        = in[i]        / P11[i];
			out[nmodel+i] = in[nmodel+i] / P33[i];
		}
	}
}
