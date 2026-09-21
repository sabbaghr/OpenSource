#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*--------------------------------------------------------------------
 * regularization.c -- Tikhonov and prior model regularization.
 *
 * Matches TOY2DAC V2.6 (sub_Tikhonov.f90, sub_prior.f90) exactly.
 *
 * Tikhonov regularization penalizes model roughness:
 *   f_tik = 0.5 * (lambda/h^2) * [lz * sum(dm/dz)^2 + lx * sum(dm/dx)^2]
 *   g_tik = (lambda/h^2) * [lz * d2m/dz2 + lx * d2m/dx2]
 *
 * Prior model regularization penalizes deviation from a reference:
 *   f_prior = 0.5 * alpha * sum((m - m0)^2)
 *   g_prior = alpha * (m - m0)
 *
 * The model vector layout is column-major: m[ix*nz + iz]
 * with iz=0 at surface (depth-fast).
 *
 * Bathymetry masking via ibathy[ix]: regularization starts at
 * iz = ibathy[ix] (0-indexed). Set ibathy=NULL for no masking.
 *--------------------------------------------------------------------*/


/*--------------------------------------------------------------------
 * tikhonov_cost -- Compute Tikhonov regularization cost.
 *
 * f_tik = 0.5 * (lambda/dx^2) * [lz * Σ(m[i+1,j]-m[i,j])^2
 *                                + lx * Σ(m[i,j+1]-m[i,j])^2]
 *
 * Parameters:
 *   model    - flat model vector [npar * nx * nz], column-major
 *   nx, nz   - model dimensions (physical, no padding)
 *   dx       - grid spacing (assumed dx=dz)
 *   lambda   - regularization strength
 *   lambda_x - horizontal weight
 *   lambda_z - vertical weight
 *   ibathy   - bathymetry indices [nx] (start from ibathy[ix]), or NULL
 *   npar     - number of parameters (loop over each)
 *
 * Returns: total Tikhonov cost (NOT yet divided by scalingfactor)
 *--------------------------------------------------------------------*/
float tikhonov_cost(float *model, int nx, int nz, float dx,
                    float lambda, float lambda_x, float lambda_z,
                    const int *ibathy, int npar)
{
    double fcost_z = 0.0, fcost_x = 0.0;
    float h2_inv = 1.0f / (dx * dx);
    int ip, ix, iz;
    int nmodel = nx * nz;

    for (ip = 0; ip < npar; ip++) {
        float *m = model + ip * nmodel;

        /* Vertical differences (z direction) */
        for (ix = 0; ix < nx; ix++) {
            int iz_start = ibathy ? ibathy[ix] : 0;
            for (iz = iz_start; iz < nz - 1; iz++) {
                float diff = m[ix*nz + iz + 1] - m[ix*nz + iz];
                fcost_z += (double)(diff * diff);
            }
        }

        /* Horizontal differences (x direction) */
        for (ix = 0; ix < nx - 1; ix++) {
            int iz_start = ibathy ? ibathy[ix] : 0;
            for (iz = iz_start; iz < nz; iz++) {
                float diff = m[(ix+1)*nz + iz] - m[ix*nz + iz];
                fcost_x += (double)(diff * diff);
            }
        }
    }

    return (float)(0.5 * lambda * h2_inv *
                   (lambda_z * fcost_z + lambda_x * fcost_x));
}


/*--------------------------------------------------------------------
 * tikhonov_gradient -- Add Tikhonov regularization to gradient.
 *
 * g_tik = (lambda/dx^2) * [lz * d2m/dz2 + lx * d2m/dx2]
 *
 * Uses standard 3-point stencil with one-sided BCs at boundaries.
 * Matches TOY2DAC sub_Tikhonov_fgrad exactly.
 *
 * Parameters:
 *   grad     - gradient vector (accumulated in-place) [npar*nx*nz]
 *   model    - model vector [npar * nx * nz]
 *   nx, nz   - model dimensions
 *   dx       - grid spacing
 *   lambda   - regularization strength
 *   lambda_x - horizontal weight
 *   lambda_z - vertical weight
 *   ibathy   - bathymetry indices [nx], or NULL
 *   npar     - number of parameters
 *--------------------------------------------------------------------*/
void tikhonov_gradient(float *grad, float *model, int nx, int nz, float dx,
                       float lambda, float lambda_x, float lambda_z,
                       const int *ibathy, int npar)
{
    float h2_inv = 1.0f / (dx * dx);
    float scale_z = lambda * h2_inv * lambda_z;
    float scale_x = lambda * h2_inv * lambda_x;
    int ip, ix, iz;
    int nmodel = nx * nz;

    for (ip = 0; ip < npar; ip++) {
        float *g = grad  + ip * nmodel;
        float *m = model + ip * nmodel;

        /* Vertical Laplacian (z direction) */
        for (ix = 0; ix < nx; ix++) {
            int ib = ibathy ? ibathy[ix] : 0;

            /* Bottom boundary (one-sided) */
            if (ib < nz)
                g[ix*nz + ib] += scale_z *
                    (-m[ix*nz + ib + 1] + m[ix*nz + ib]);

            /* Interior */
            for (iz = ib + 1; iz < nz - 1; iz++)
                g[ix*nz + iz] += scale_z *
                    (-m[ix*nz + iz + 1] + 2.0f*m[ix*nz + iz] - m[ix*nz + iz - 1]);

            /* Top boundary (one-sided) */
            if (nz - 1 > ib)
                g[ix*nz + nz - 1] += scale_z *
                    (m[ix*nz + nz - 1] - m[ix*nz + nz - 2]);
        }

        /* Horizontal Laplacian (x direction) */
        /* Left boundary (ix=0) */
        for (iz = (ibathy ? ibathy[0] : 0); iz < nz; iz++)
            g[0*nz + iz] += scale_x * (-m[1*nz + iz] + m[0*nz + iz]);

        /* Interior */
        for (ix = 1; ix < nx - 1; ix++) {
            int ib = ibathy ? ibathy[ix] : 0;
            for (iz = ib; iz < nz; iz++)
                g[ix*nz + iz] += scale_x *
                    (-m[(ix+1)*nz + iz] + 2.0f*m[ix*nz + iz] - m[(ix-1)*nz + iz]);
        }

        /* Right boundary (ix=nx-1) */
        for (iz = (ibathy ? ibathy[nx-1] : 0); iz < nz; iz++)
            g[(nx-1)*nz + iz] += scale_x *
                (m[(nx-1)*nz + iz] - m[(nx-2)*nz + iz]);
    }
}


/*--------------------------------------------------------------------
 * prior_cost -- Compute prior model regularization cost.
 *
 * f_prior = 0.5 * alpha * ||m - m0||^2
 *
 * Parameters:
 *   model - current model vector
 *   prior - prior (reference) model vector
 *   n     - vector length
 *   alpha - regularization strength
 *
 * Returns: prior regularization cost
 *--------------------------------------------------------------------*/
float prior_cost(float *model, float *prior, int n, float alpha)
{
    double fcost = 0.0;
    int i;

    for (i = 0; i < n; i++) {
        float diff = model[i] - prior[i];
        fcost += (double)(diff * diff);
    }

    return (float)(0.5 * alpha * fcost);
}


/*--------------------------------------------------------------------
 * prior_gradient -- Add prior model regularization to gradient.
 *
 * g_prior = alpha * (m - m0)
 *
 * Parameters:
 *   grad  - gradient vector (accumulated in-place)
 *   model - current model vector
 *   prior - prior model vector
 *   n     - vector length
 *   alpha - regularization strength
 *--------------------------------------------------------------------*/
void prior_gradient(float *grad, float *model, float *prior, int n,
                    float alpha)
{
    int i;

    for (i = 0; i < n; i++)
        grad[i] += alpha * (model[i] - prior[i]);
}
