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

/* SU file writer (writesufile.c) */
int writesufile(char *filename, float *data, size_t n1, size_t n2,
                float f1, float f2, float d1, float d2);

/* File-scope state shared between build and apply */
static int stored_blend_nz = 0;
static int stored_nx = 0;
static int stored_nz = 0;
static int stored_precond_mode = 1;  /* 1=diagonal, 3=block inverse */

/* Stored norm-preservation factor ν (Métivier et al. 2014, eq 42-43).
 * Computed once from the gradient at each outer iteration via
 * applyBlockPrecond(..., norm_preserve=1).  Retrieved by getPrecondNu()
 * for use as a fixed scalar in PTRN/PENRICHED inner CG loops, where
 * the preconditioner must be a fixed linear operator. */
static float stored_precond_nu = 1.0f;


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
	float xmin, float xmax,
	int precond_mode)
{
	int i, ix, iz;
	int ibndx, ibndz, n1;
	int nx = mod->nx;
	int nz = mod->nz;

	(void)hess_lam_muu; (void)hess_lam_rho; (void)hess_muu_rho;

	/* Store mode and alpha for applyBlockPrecond */
	stored_precond_mode = precond_mode;
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

		/* Write raw Lamé Hessian diagonals (smoothed but pre-chain-rule) */
		writesufile("hess_raw_lam.su", h11, nz, nx,
		            mod->z0, mod->x0, mod->dz, mod->dx);
		writesufile("hess_raw_mu.su", h22, nz, nx,
		            mod->z0, mod->x0, mod->dz, mod->dx);
		writesufile("hess_raw_rho.su", h33, nz, nx,
		            mod->z0, mod->x0, mod->dz, mod->dx);
	}

	/* If param=2: chain rule Lamé → velocity.
	 *
	 * H_vel = J^T · H_Lamé · J  where J = ∂(λ,μ,ρ)/∂(Vp,Vs,ρ).
	 *
	 * In Lamé space (proven analytically from radiation patterns):
	 *   H_λμ = H_λλ  (always identical)
	 *   H_λρ = H_μρ = 0  (stress vs velocity subspaces are orthogonal)
	 *
	 * Jacobian:
	 *   j11=2ρVp   j12=-4ρVs  j13=Vp²-2Vs²
	 *   j21=0      j22=2ρVs   j23=Vs²
	 *   j31=0      j32=0      j33=1
	 *
	 * Full 6-element chain rule (for precond=2 diagonal AND precond=3 block):
	 *
	 * Diagonal (Liu et al. 2022, eq A21):
	 *   H_VpVp = j11²·Hll
	 *   H_VsVs = j12²·Hll + 2·j12·j22·Hll + j22²·Hmm  (using Hlm=Hll)
	 *   H_ρρ   = j13²·Hll + 2·j13·j23·Hll + j23²·Hmm + Hrr
	 *
	 * Off-diagonal (for precond=3,5 block):
	 *   H_VpVs = j11·j12·Hll + j11·j22·Hll  (using Hlm=Hll, Hlr=0)
	 *          = j11·Hll·(j12 + j22)
	 *   H_Vpρ  = j11·j13·Hll + j11·j23·Hll  (using Hlm=Hll, Hlr=0)
	 *          = j11·Hll·(j13 + j23)
	 *   H_Vsρ  = j12·j13·Hll + (j12·j23+j22·j13)·Hll + j22·j23·Hmm
	 *          = Hll·(j12·j13 + j12·j23 + j22·j13) + j22·j23·Hmm
	 *
	 * Store off-diagonals in h12_vel, h13_vel, h23_vel (allocated only
	 * for block modes precond=3,5). */
	float *h12_vel = NULL, *h13_vel = NULL, *h23_vel = NULL;
	if (param == 2 && elastic) {
		if (precond_mode == 3 || precond_mode == 5) {
			h12_vel = (float *)calloc(nmodel, sizeof(float));
			h13_vel = (float *)calloc(nmodel, sizeof(float));
			h23_vel = (float *)calloc(nmodel, sizeof(float));
		}

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

			/* Diagonal elements */
			h11[i] = j11*j11 * Hll;
			h22[i] = j12*j12 * Hll + 2.0f*j12*j22 * Hll + j22*j22 * Hmm;
			h33[i] = j13*j13 * Hll + 2.0f*j13*j23 * Hll + j23*j23 * Hmm + Hrr;

			/* Off-diagonal elements (block modes only) */
			if (h12_vel) {
				h12_vel[i] = j11 * Hll * (j12 + j22);
				h13_vel[i] = j11 * Hll * (j13 + j23);
				h23_vel[i] = Hll * (j12*j13 + j12*j23 + j22*j13)
				           + j22*j23 * Hmm;
			}
		}
	} else if (precond_mode == 3 || precond_mode == 5) {
		/* Lamé parameterization: extract off-diags directly
		 * H_λμ = H_λλ (from hess_lam_muu), H_λρ = H_μρ = 0 */
		h12_vel = (float *)calloc(nmodel, sizeof(float));
		h13_vel = (float *)calloc(nmodel, sizeof(float));
		h23_vel = (float *)calloc(nmodel, sizeof(float));
		for (i = 0; i < nmodel; i++) {
			h12_vel[i] = h11[i];  /* H_λμ = H_λλ */
			/* h13_vel[i] = 0, h23_vel[i] = 0 (already from calloc) */
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

	/* Per-parameter diagnostics (after chain rule + m0² + taper) */
	if (mpi_rank == 0) {
		const char *vel_n[] = {"Vp", "Vs", "rho"};
		const char *lam_n[] = {"lam", "mu", "rho"};
		const char **pn = (param == 2) ? vel_n : lam_n;
		double min1=1e30, max1=0, min2=1e30, max2=0, min3=1e30, max3=0;
		double sum1=0, sum2=0, sum3=0;
		int nz1=0, nz2=0, nz3=0;
		for (i = 0; i < nmodel; i++) {
			if (h11[i] > 0) { if (h11[i]<min1) min1=h11[i]; if (h11[i]>max1) max1=h11[i]; sum1+=h11[i]; nz1++; }
			if (h22[i] > 0) { if (h22[i]<min2) min2=h22[i]; if (h22[i]>max2) max2=h22[i]; sum2+=h22[i]; nz2++; }
			if (h33[i] > 0) { if (h33[i]<min3) min3=h33[i]; if (h33[i]>max3) max3=h33[i]; sum3+=h33[i]; nz3++; }
		}
		vmess("Precond: H̃ after chain rule + s²:");
		vmess("  H̃_%s: min=%.4e max=%.4e mean=%.4e ratio=%.1f",
		      pn[0], min1, max1, nz1>0?sum1/nz1:0, max1/(min1>0?min1:1));
		vmess("  H̃_%s: min=%.4e max=%.4e mean=%.4e ratio=%.1f",
		      pn[1], min2, max2, nz2>0?sum2/nz2:0, max2/(min2>0?min2:1));
		vmess("  H̃_%s: min=%.4e max=%.4e mean=%.4e ratio=%.1f",
		      pn[2], min3, max3, nz3>0?sum3/nz3:0, max3/(min3>0?min3:1));
		vmess("  Inter-param ratio: max(H̃_%s)/max(H̃_%s)=%.1f  max(H̃_%s)/max(H̃_%s)=%.1f",
		      pn[0], pn[1], max1/(max2>0?max2:1),
		      pn[0], pn[2], max1/(max3>0?max3:1));
	}

	/* Write pre-β Hessian diagonals (after chain rule + m₀² + taper).
	 * These are the values that β is computed from — lets you see
	 * the scale differences between parameters before regularization. */
	if (mpi_rank == 0) {
		const char *pn1 = (param == 2) ? "vp" : "lam";
		const char *pn2 = (param == 2) ? "vs" : "mu";
		char fn[512];
		snprintf(fn, sizeof(fn), "hess_prebeta_%s.su", pn1);
		writesufile(fn, h11, nz, nx, mod->z0, mod->x0, mod->dz, mod->dx);
		snprintf(fn, sizeof(fn), "hess_prebeta_%s.su", pn2);
		writesufile(fn, h22, nz, nx, mod->z0, mod->x0, mod->dz, mod->dx);
		writesufile("hess_prebeta_rho.su", h33, nz, nx,
		            mod->z0, mod->x0, mod->dz, mod->dx);
		vmess("Precond: wrote hess_prebeta_{%s,%s,rho}.su", pn1, pn2);
	}

	/* Per-parameter damping (corrected Métivier eq 38-39 for multi-parameter):
	 *   C_k = max_x(H̃_kk(x))  per parameter class k=1,2,3
	 *   β_k = θ · C_k
	 *   P_kk = H̃_kk + β_k
	 *
	 * Métivier (2014) defined C = max_j(H̃_jj) for single-parameter
	 * acoustic FWI.  For multi-parameter elastic, a global max causes
	 * the dominant parameter (Vp) to swamp the damping of weaker
	 * parameters (Vs, ρ), destroying their spatial illumination
	 * structure.  Per-parameter β_k preserves O(1/θ) dynamic range
	 * within each parameter's preconditioner.
	 *
	 * For the block preconditioner, A = H̃_block + diag(β_1,β_2,β_3)
	 * remains SPD since H̃_block is PSD and diag(β) is PD. */
	double C1 = 0.0, C2 = 0.0, C3 = 0.0;
	for (i = 0; i < nmodel; i++) {
		if ((double)h11[i] > C1) C1 = (double)h11[i];
		if ((double)h22[i] > C2) C2 = (double)h22[i];
		if ((double)h33[i] > C3) C3 = (double)h33[i];
	}
	double beta1 = (double)precond_eps * C1;
	double beta2 = (double)precond_eps * C2;
	double beta3 = (double)precond_eps * C3;

	if (precond_mode == 3 || precond_mode == 5) {
		/* ============================================================
		 * Block-diagonal preconditioner (Wang et al. 2016, eq 12-14).
		 *
		 * Off-diagonal elements h12_vel/h13_vel/h23_vel were computed
		 * during the chain rule step above (or from Lamé identities).
		 * Apply m0² scaling, then per grid point: compute
		 * (H̃_b + βI)^{-1} via Cramer's rule in double precision.
		 * Store 6 elements of the inverse.
		 * ============================================================ */

		/* Apply m0² scaling to off-diagonals */
		if (h12_vel) {
			for (i = 0; i < nmodel; i++) {
				h12_vel[i] *= s1 * s2;
				h13_vel[i] *= s1 * s3;
				h23_vel[i] *= s2 * s3;
			}
		}

		/* Per grid point: Cramer's rule on (H̃_b + βI) in double */
		for (i = 0; i < nmodel; i++) {
			double a11 = (double)h11[i] + beta1;
			double a22 = (double)h22[i] + beta2;
			double a33 = (double)h33[i] + beta3;
			double a12 = h12_vel ? (double)h12_vel[i] : 0.0;
			double a13 = h13_vel ? (double)h13_vel[i] : 0.0;
			double a23 = h23_vel ? (double)h23_vel[i] : 0.0;

			/* Determinant */
			double det = a11*(a22*a33 - a23*a23)
			           - a12*(a12*a33 - a23*a13)
			           + a13*(a12*a23 - a22*a13);

			if (det == 0.0) det = 1.0;  /* safety */
			double idet = 1.0 / det;

			/* Cofactors (symmetric adjugate) = inverse * det */
			double c11 = a22*a33 - a23*a23;
			double c12 = a13*a23 - a12*a33;
			double c13 = a12*a23 - a13*a22;
			double c22 = a11*a33 - a13*a13;
			double c23 = a12*a13 - a11*a23;
			double c33 = a11*a22 - a12*a12;

			/* Store inverse elements */
			P11[i] = (float)(idet * c11);
			P22[i] = (float)(idet * c22);
			P33[i] = (float)(idet * c33);
			P12[i] = (float)(idet * c12);
			P13[i] = (float)(idet * c13);
			P23[i] = (float)(idet * c23);
		}

		if (mpi_rank == 0)
			vmess("Precond: Block-diagonal (Wang et al. 2016), beta1=%.4e beta2=%.4e beta3=%.4e",
			      beta1, beta2, beta3);

		if (h12_vel) free(h12_vel);
		if (h13_vel) free(h13_vel);
		if (h23_vel) free(h23_vel);
		h12_vel = h13_vel = h23_vel = NULL;
	} else {
		/* Diagonal preconditioner: store H̃_ii + β_k */
		for (i = 0; i < nmodel; i++) {
			P11[i] = (float)((double)h11[i] + beta1);
			P22[i] = (float)((double)h22[i] + beta2);
			P33[i] = (float)((double)h33[i] + beta3);
		}
		memset(P12, 0, nmodel * sizeof(float));
		memset(P13, 0, nmodel * sizeof(float));
		memset(P23, 0, nmodel * sizeof(float));
	}

	/* Diagnostics */
	if (mpi_rank == 0) {
		const char *lame_n[] = {"lam", "mu", "rho"};
		const char *vel_n[]  = {"Vp", "Vs", "rho"};
		const char **n = (param == 2) ? vel_n : lame_n;
		vmess("Precond (Shin/Metivier): theta=%.4e  beta1=%.4e beta2=%.4e beta3=%.4e",
		      precond_eps, beta1, beta2, beta3);

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
	int nmodel, int nparam, int norm_preserve)
{
	int i, ntot;

	ntot = nparam * nmodel;

	/* Step 1: Compute ||g||² before the solve (needed for in-place case) */
	double norm_in_sq = 0.0;
	for (i = 0; i < ntot; i++)
		norm_in_sq += (double)in[i] * (double)in[i];

	/* Step 2: Apply preconditioner */
	if ((stored_precond_mode == 3 || stored_precond_mode == 5) && nparam == 3) {
		/* Block-diagonal: P stores (H̃+γI)^{-1}, do matrix-vector multiply.
		 * out = P^{-1} · in where P^{-1} is stored in P11-P33, P12-P23. */
		for (i = 0; i < nmodel; i++) {
			double g1 = in[i], g2 = in[nmodel+i], g3 = in[2*nmodel+i];
			double p11 = P11[i], p12 = P12[i], p13 = P13[i];
			double p22 = P22[i], p23 = P23[i], p33 = P33[i];

			out[i]          = (float)(p11*g1 + p12*g2 + p13*g3);
			out[nmodel+i]   = (float)(p12*g1 + p22*g2 + p23*g3);
			out[2*nmodel+i] = (float)(p13*g1 + p23*g2 + p33*g3);
		}
	} else if (nparam == 3) {
		/* Diagonal: P stores H̃_ii + β, divide */
		for (i = 0; i < nmodel; i++) {
			out[i]          = (float)((double)in[i]          / (double)P11[i]);
			out[nmodel+i]   = (float)((double)in[nmodel+i]   / (double)P22[i]);
			out[2*nmodel+i] = (float)((double)in[2*nmodel+i] / (double)P33[i]);
		}
	} else {
		/* 2-parameter (acoustic) */
		for (i = 0; i < nmodel; i++) {
			out[i]        = (float)((double)in[i]        / (double)P11[i]);
			out[nmodel+i] = (float)((double)in[nmodel+i] / (double)P33[i]);
		}
	}

	/* Step 3: Norm preservation (Métivier et al. 2014, eq 42-43).
	 *
	 * norm_preserve=1: Compute ν = ||in|| / ||P^θ·in||, apply it,
	 *   and store ν for later use.  This is called when the input is
	 *   the gradient ∇f — once per outer iteration.
	 *
	 * norm_preserve=0: Skip norm preservation entirely.  The raw P^θ
	 *   output is returned.  Caller is responsible for multiplying by
	 *   the stored ν (via getPrecondNu()) if the Métivier formulation
	 *   P^{ν,θ} = ν · P^θ is desired.  This mode is used inside the
	 *   PTRN/PENRICHED inner CG loop, where the preconditioner must
	 *   be a fixed linear operator (same ν for all CG iterations). */
	if (norm_preserve) {
		double norm_out_sq = 0.0;
		for (i = 0; i < ntot; i++)
			norm_out_sq += (double)out[i] * (double)out[i];

		if (norm_out_sq > 0.0 && norm_in_sq > 0.0) {
			stored_precond_nu = (float)sqrt(norm_in_sq / norm_out_sq);
			for (i = 0; i < ntot; i++)
				out[i] *= stored_precond_nu;
		}
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


/*--------------------------------------------------------------------
 * getPrecondNu -- Return the stored norm-preservation factor ν.
 *
 * ν = ||∇f|| / ||P^θ · ∇f||  (Métivier et al. 2014, eq 43)
 *
 * Computed and stored by the most recent applyBlockPrecond() call
 * with norm_preserve=1.  Used by PTRN/PENRICHED inner CG to apply
 * a fixed P^{ν,θ} = ν · P^θ at every CG iteration.
 *--------------------------------------------------------------------*/
float getPrecondNu(void)
{
	return stored_precond_nu;
}
