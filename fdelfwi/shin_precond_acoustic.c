#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*--------------------------------------------------------------------
 * shin_precond_acoustic.c -- Shin diagonal pseudo-Hessian preconditioner.
 *
 * Matches TOY2DAC V2.6 (sub_Shin_preco.f90) exactly.
 *
 * The diagonal pseudo-Hessian H_diag is accumulated per-shot by
 * accumGradientAcoustic() during the adjoint backpropagation pass.
 *
 * Application:
 *   preco[i] = 1 / (H_diag[i] + eps * max(H_diag))
 *   g_preco[i] = preco[i] * g[i]
 *   g_preco *= ||g|| / ||g_preco||     (norm-preserving rescaling)
 *
 * TOY2DAC uses a two-pass approach:
 *   1. First call with scal_preco=1 to compute ||g_preco||
 *   2. Compute scal_preco = ||g|| / ||g_preco||
 *   3. Second call applies with correct norm preservation
 *
 * This implementation combines both passes in a single call.
 *
 * Reference: Shin et al. (2001), Geophysics, pseudo-Hessian.
 *--------------------------------------------------------------------*/


/*--------------------------------------------------------------------
 * shin_apply_precond -- Apply Shin preconditioner with norm preservation.
 *
 * Parameters:
 *   grad_preco    - OUTPUT: preconditioned gradient [n]
 *   grad          - INPUT: raw gradient [n]
 *   hess_diag     - INPUT: pseudo-Hessian diagonal [n]
 *   n             - vector length
 *   eps_thresh    - threshold parameter (e.g., 1e-4)
 *   norm_ratio_out - OUTPUT: ||g|| / ||g_preco|| (may be NULL)
 *--------------------------------------------------------------------*/
void shin_apply_precond(float *grad_preco, const float *grad,
                        const float *hess_diag, int n, float eps_thresh,
                        float *norm_ratio_out)
{
    int i;
    float max_H = 0.0f;
    double norm_g2 = 0.0, norm_gp2 = 0.0;
    float scal;

    /* Find max of pseudo-Hessian diagonal */
    for (i = 0; i < n; i++) {
        if (hess_diag[i] > max_H) max_H = hess_diag[i];
    }

    if (max_H < 1.0e-30f) {
        /* Zero pseudo-Hessian: copy gradient unchanged */
        for (i = 0; i < n; i++)
            grad_preco[i] = grad[i];
        if (norm_ratio_out) *norm_ratio_out = 1.0f;
        return;
    }

    float threshold = eps_thresh * max_H;

    /* Apply preconditioner: g_preco = (1/H_reg) * g */
    for (i = 0; i < n; i++) {
        float preco_i = 1.0f / (hess_diag[i] + threshold);
        grad_preco[i] = preco_i * grad[i];
    }

    /* Compute norm ratio for norm-preserving rescaling */
    for (i = 0; i < n; i++) {
        norm_g2  += (double)grad[i] * (double)grad[i];
        norm_gp2 += (double)grad_preco[i] * (double)grad_preco[i];
    }

    if (norm_gp2 < 1.0e-30) {
        scal = 1.0f;
    } else {
        scal = (float)(sqrt(norm_g2 / norm_gp2));
    }

    /* Apply norm-preserving rescaling: ||g_preco|| = ||g|| */
    for (i = 0; i < n; i++)
        grad_preco[i] *= scal;

    if (norm_ratio_out) *norm_ratio_out = scal;
}
