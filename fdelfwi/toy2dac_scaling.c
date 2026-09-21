#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*--------------------------------------------------------------------
 * toy2dac_scaling.c -- TOY2DAC first-iteration scaling factor.
 *
 * Matches TOY2DAC V2.6 (sub_modeling.f90, lines 247-261) exactly.
 *
 * The scaling factor normalizes cost and gradient so that the first
 * optimizer step has a controlled magnitude, independent of physical
 * units.
 *
 * Formula:
 *   S_temp = 10e-5 * sqrt(max(||m||^2, 1e3*N) / ||g||^2)
 *
 * CRITICAL: In Fortran, 10e-5 = 10 * 10^-5 = 1e-4, NOT 1e-5.
 *
 *   scalingfactor = 1 / S_temp
 *   fcost /= scalingfactor
 *   grad  /= scalingfactor
 *
 * Computed once at the first gradient evaluation, then fixed.
 *
 * Reference: Brossier, Operto & Virieux (2009), Geophysics, eq. A-9.
 *--------------------------------------------------------------------*/

/*--------------------------------------------------------------------
 * compute_toy2dac_scalingfactor -- Compute the scaling factor.
 *
 * Uses only the FIRST parameter's model and gradient norms
 * (TOY2DAC uses model(:,:,1) and gradient(:,:,1)).
 *
 * For multi-parameter: pass the full model_vec and grad_vec, but
 * only the first nmodel elements (first parameter block) are used.
 *
 * Parameters:
 *   model_vec  - model vector (first parameter block, nmodel values)
 *   grad_vec   - gradient vector (first parameter block, nmodel values)
 *   nmodel     - number of grid points for the first parameter
 *   verbose    - print diagnostic info
 *
 * Returns: scalingfactor (divide fcost and gradient by this value)
 *--------------------------------------------------------------------*/
float compute_toy2dac_scalingfactor(float *model_vec, float *grad_vec,
                                     int nmodel, int verbose)
{
    double model_norm2 = 0.0;
    double grad_norm2  = 0.0;
    double S_temp, scalingfactor;
    int i;

    for (i = 0; i < nmodel; i++) {
        model_norm2 += (double)model_vec[i] * (double)model_vec[i];
        grad_norm2  += (double)grad_vec[i]  * (double)grad_vec[i];
    }

    /* Guard: prevent division by zero gradient */
    if (grad_norm2 < 1.0e-30) {
        if (verbose)
            fprintf(stderr, "toy2dac_scaling: WARNING gradient norm near zero, "
                    "scalingfactor set to 1\n");
        return 1.0f;
    }

    /* max(||m||^2, 1e3*N) -- prevents scaling blowup for small models */
    double model_guard = (model_norm2 > 1.0e3 * nmodel)
                       ? model_norm2 : 1.0e3 * nmodel;

    /* S_temp = 10e-5 * sqrt(model_guard / grad_norm2)
     * Fortran 10e-5 = 1e-4 */
    S_temp = 1.0e-4 * sqrt(model_guard / grad_norm2);

    /* scalingfactor = 1 / S_temp */
    scalingfactor = 1.0 / S_temp;

    if (verbose) {
        fprintf(stderr, "toy2dac_scaling: ||m||^2=%.6e  ||g||^2=%.6e  "
                "S_temp=%.6e  scalingfactor=%.6e\n",
                model_norm2, grad_norm2, S_temp, scalingfactor);
    }

    return (float)scalingfactor;
}


/*--------------------------------------------------------------------
 * apply_toy2dac_scaling -- Divide cost function and gradient by
 * the scaling factor.
 *
 * Called at EVERY iteration (not just the first).
 *
 * Parameters:
 *   fcost - pointer to cost function value (modified in-place)
 *   grad  - gradient vector (modified in-place)
 *   nvec  - total length of gradient vector
 *   sf    - scaling factor (computed by compute_toy2dac_scalingfactor)
 *--------------------------------------------------------------------*/
void apply_toy2dac_scaling(float *fcost, float *grad, int nvec, float sf)
{
    int i;
    float inv_sf = 1.0f / sf;

    *fcost *= inv_sf;
    for (i = 0; i < nvec; i++)
        grad[i] *= inv_sf;
}
