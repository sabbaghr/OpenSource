/*--------------------------------------------------------------------
 * scaling.c -- Parameter scaling for multi-parameter FWI
 *
 * scaling=0 : none (raw physical units)
 * scaling=1 : Range normalization (Métivier, pers. comm.)
 *             m* = m / Δm,  Δm = m_max - m_min.
 *             Divides by expected parameter range, no shift.
 *             Requires user-supplied bounds (vp_min/max, etc.).
 *
 * Chain rule:
 *   g* = g · Δm         (gradient)
 *   H* = Δm² · H        (Hessian diagonal)
 *   d_phys = d* · Δm    (perturbation direction)
 *
 * Reference: Métivier (2025), personal communication.
 *--------------------------------------------------------------------*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*--------------------------------------------------------------------
 * scaling_compute_m0 -- Compute m0 = Δm = m_max - m_min from bounds.
 *
 *   m_min, m_max : parameter bounds [nparam]
 *   m0           : output scale factors [nparam] (= Δm)
 *   m_shift      : output shift [nparam] (always 0)
 *--------------------------------------------------------------------*/
void scaling_compute_m0_from_bounds(const float *m_min, const float *m_max,
                                    int nparam, float *m0, float *m_shift)
{
    int p;
    for (p = 0; p < nparam; p++) {
        m0[p] = m_max[p] - m_min[p];
        m_shift[p] = 0.0f;
        if (m0[p] < 1.0e-30f) m0[p] = 1.0f;
    }
}

/*--------------------------------------------------------------------
 * scaling_normalize -- m* = (m - shift) / m0
 * With shift=0: m* = m / Δm
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
 * scaling_denormalize -- m = m* · m0 + shift
 * With shift=0: m = m* · Δm
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
 * scaling_scale_gradient -- g* = g · m0  (chain rule)
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
 * scaling_scale_hessian_vec -- Hd* = m0 · (H · (m0 · d*))
 *
 * For Hessian-vector products in normalized space.
 * Caller must: (1) denorm_direction d before H*d, (2) call this on Hd.
 *--------------------------------------------------------------------*/
void scaling_scale_hessian_vec(float *Hd, int nmodel, int nparam,
                               const float *m0)
{
    scaling_scale_gradient(Hd, nmodel, nparam, m0);
}

/*--------------------------------------------------------------------
 * scaling_denorm_direction -- d_phys = d* · m0
 *
 * Perturbation direction: only scale applies (no shift).
 *--------------------------------------------------------------------*/
void scaling_denorm_direction(float *d, int nmodel, int nparam, const float *m0)
{
    scaling_scale_gradient(d, nmodel, nparam, m0);
}
