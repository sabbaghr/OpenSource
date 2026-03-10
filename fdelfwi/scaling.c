/*--------------------------------------------------------------------
 * scaling.c -- Parameter scaling for multi-parameter FWI
 *
 * scaling=0 : none (raw physical units)
 * scaling=1 : Brossier (2011) — m0[p] = mean(|x_init_p|), x̃ = x/m0
 * scaling=2 : Yang (2018) eq 47 — m̄ = (m - m_min)/(m_max - m_min)
 *             maps parameters to [0,1] using user-provided bounds.
 *
 * For scaling=1:
 *   x_tilde = x / m0,  g_tilde = g * m0
 *   m_shift = 0 (no shift)
 *
 * For scaling=2:
 *   m0[p] = m_max[p] - m_min[p]  (range)
 *   m_shift[p] = m_min[p]
 *   x_bar = (x - m_shift) / m0,  g_bar = g * m0
 *
 * When combined with the Yang pseudo-Hessian preconditioner,
 * s_p = m0_p makes the pseudo-Hessian consistent with the
 * normalized gradient (eq 48).
 *--------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*--------------------------------------------------------------------
 * scaling_compute_m0 -- Compute scaling constants from initial model.
 *
 *   scaling : 1 = Brossier (mean |x|), 2 = Yang (uses bounds)
 *   x       : model vector [nparam * nmodel]
 *   nmodel  : grid points per parameter
 *   nparam  : number of parameter classes (2 or 3)
 *   m0      : output array [nparam] — scale factor
 *   m_shift : output array [nparam] — shift (0 for Brossier)
 *--------------------------------------------------------------------*/
void scaling_compute_m0(int scaling, const float *x, int nmodel, int nparam,
                        float *m0, float *m_shift)
{
    int p, i;
    for (p = 0; p < nparam; p++) {
        m_shift[p] = 0.0f;
        const float *xp = x + (size_t)p * nmodel;
        if (scaling == 1) {
            /* Brossier: mean absolute value */
            double sum = 0.0;
            for (i = 0; i < nmodel; i++)
                sum += fabsf(xp[i]);
            m0[p] = (nmodel > 0) ? (float)(sum / nmodel) : 1.0f;
        } else if (scaling == 2) {
            /* Yang eq 47: m0 = m_max - m_min from bounds (set by caller).
             * If called before bounds are set, compute from initial model. */
            float mn = xp[0], mx = xp[0];
            for (i = 1; i < nmodel; i++) {
                if (xp[i] < mn) mn = xp[i];
                if (xp[i] > mx) mx = xp[i];
            }
            m0[p] = mx - mn;
            m_shift[p] = mn;
        } else {
            m0[p] = 1.0f;
        }
        if (m0[p] < 1.0e-30f) m0[p] = 1.0f;  /* safety for zero/constant fields */
    }
}

/*--------------------------------------------------------------------
 * scaling_set_yang_bounds -- Set m0 and m_shift from user-provided bounds.
 *
 * For Yang eq 47: m̄ = (m - m_min) / (m_max - m_min)
 *--------------------------------------------------------------------*/
void scaling_set_yang_bounds(float *m0, float *m_shift,
                             const float *m_min, const float *m_max,
                             int nparam)
{
    int p;
    for (p = 0; p < nparam; p++) {
        m_shift[p] = m_min[p];
        m0[p] = m_max[p] - m_min[p];
        if (m0[p] < 1.0e-30f) m0[p] = 1.0f;
    }
}

/*--------------------------------------------------------------------
 * scaling_normalize -- x̃ = (x - m_shift) / m0
 *--------------------------------------------------------------------*/
void scaling_normalize(float *v, int nmodel, int nparam,
                       const float *m0, const float *m_shift)
{
    int p, i;
    for (p = 0; p < nparam; p++) {
        float shift = m_shift[p];
        float inv = 1.0f / m0[p];
        float *vp = v + (size_t)p * nmodel;
        for (i = 0; i < nmodel; i++)
            vp[i] = (vp[i] - shift) * inv;
    }
}

/*--------------------------------------------------------------------
 * scaling_denormalize -- x = x̃ * m0 + m_shift
 *--------------------------------------------------------------------*/
void scaling_denormalize(float *v, int nmodel, int nparam,
                         const float *m0, const float *m_shift)
{
    int p, i;
    for (p = 0; p < nparam; p++) {
        float *vp = v + (size_t)p * nmodel;
        for (i = 0; i < nmodel; i++)
            vp[i] = vp[i] * m0[p] + m_shift[p];
    }
}

/*--------------------------------------------------------------------
 * scaling_scale_gradient -- ḡ = g * m0  (chain rule, no shift)
 *--------------------------------------------------------------------*/
void scaling_scale_gradient(float *g, int nmodel, int nparam, const float *m0)
{
    int p, i;
    for (p = 0; p < nparam; p++) {
        float *gp = g + (size_t)p * nmodel;
        for (i = 0; i < nmodel; i++)
            gp[i] *= m0[p];
    }
}

/*--------------------------------------------------------------------
 * scaling_scale_hessian_vec -- H̄d = m0 * H * (m0 * d̃)
 *
 * Caller must: (1) denorm_direction d before H*d, (2) call this on Hd.
 * Note: for Hessian-vector products, only m0 scaling applies (no shift).
 *--------------------------------------------------------------------*/
void scaling_scale_hessian_vec(float *Hd, int nmodel, int nparam,
                               const float *m0)
{
    scaling_scale_gradient(Hd, nmodel, nparam, m0);
}

/*--------------------------------------------------------------------
 * scaling_denorm_direction -- d_phys = d̃ * m0  (direction only, no shift)
 *
 * For Hessian-vector products: the perturbation direction d must be
 * converted from normalized to physical space. Since d is a PERTURBATION
 * (not a model), only the scale applies: d_phys = m0 * d̃.
 *--------------------------------------------------------------------*/
void scaling_denorm_direction(float *d, int nmodel, int nparam, const float *m0)
{
    scaling_scale_gradient(d, nmodel, nparam, m0);
}
