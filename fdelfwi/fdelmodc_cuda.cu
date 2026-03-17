/*********************************************************************
 *
 * fdelmodc_cuda.cu — CUDA kernels for forward elastic FD modeling
 *
 * Implements GPU versions of:
 *   - elastic4/6/8.c velocity + stress update (4th/6th/8th order stencils)
 *   - boundaries.c taper boundaries (bnd.type==4, elastic)
 *   - boundaries.c free surface (bnd.top==1)
 *   - applySource.c source injection
 *   - getRecTimes.c receiver extraction
 *   - Device memory management for all wavefield + material arrays
 *
 * Stencil order is handled via C++ templates: template<int ORDER>
 * generates 3 kernel variants (ORDER=4,6,8) at compile time.
 * The compiler eliminates dead branches for unused c3/c4 terms.
 *
 * All kernels operate on LOCAL subdomain arrays (domain-decomposed).
 * Halo exchange is handled by domain_decomp.c between kernel launches.
 *
 * Author: FWI CUDA porting project
 *
 *********************************************************************/

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_utils.h"

/*====================================================================
 * 1. CONSTANT MEMORY
 *====================================================================*/

/* FD stencil coefficients: c1, c2, c3, c4 (up to 8th order) */
__constant__ float d_fd_coeff[4];

/*====================================================================
 * 2. FORWARD VELOCITY UPDATE KERNEL — template<int ORDER>
 *
 * vx -= rox * [D-x(txx) + D+z(txz)]
 * vz -= roz * [D+x(txz) + D-z(tzz)]
 *
 * ORDER = 4, 6, or 8.  Stencil radius = ORDER/2.
 * Compiler eliminates dead branches for c3 (ORDER<6) and c4 (ORDER<8).
 *
 * Each thread computes one (ix, iz) grid point.
 * Uses register-based stencil — L1 cache handles the halo reads.
 *====================================================================*/

template<int ORDER>
__global__ void update_velocity_kernel(
    float* __restrict__ vx,
    float* __restrict__ vz,
    const float* __restrict__ txx,
    const float* __restrict__ tzz,
    const float* __restrict__ txz,
    const float* __restrict__ rox,
    const float* __restrict__ roz,
    int n1,        /* naz (fast dimension = depth) */
    int ioXx, int ioXz, int ieXx, int ieXz,  /* Vx loop bounds */
    int ioZx, int ioZz, int ieZx, int ieZz)   /* Vz loop bounds */
{
    const float c1 = d_fd_coeff[0];
    const float c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2];  /* used only if ORDER >= 6 */
    const float c4 = d_fd_coeff[3];  /* used only if ORDER >= 8 */

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    /* --- Vx update: vx -= rox * [D-x(txx) + D+z(txz)] --- */
    {
        int gx = ix + ioXx;
        int gz = iz + ioXz;
        if (gx < ieXx && gz < ieXz) {
            int idx = gx * n1 + gz;
            float val =
                c1 * (txx[idx]          - txx[(gx-1)*n1+gz] +
                      txz[gx*n1+gz+1]   - txz[idx]) +
                c2 * (txx[(gx+1)*n1+gz] - txx[(gx-2)*n1+gz] +
                      txz[gx*n1+gz+2]   - txz[gx*n1+gz-1]);
            if (ORDER >= 6) val +=
                c3 * (txx[(gx+2)*n1+gz] - txx[(gx-3)*n1+gz] +
                      txz[gx*n1+gz+3]   - txz[gx*n1+gz-2]);
            if (ORDER >= 8) val +=
                c4 * (txx[(gx+3)*n1+gz] - txx[(gx-4)*n1+gz] +
                      txz[gx*n1+gz+4]   - txz[gx*n1+gz-3]);
            vx[idx] -= rox[idx] * val;
        }
    }

    /* --- Vz update: vz -= roz * [D+x(txz) + D-z(tzz)] --- */
    {
        int gx = ix + ioZx;
        int gz = iz + ioZz;
        if (gx < ieZx && gz < ieZz) {
            int idx = gx * n1 + gz;
            float val =
                c1 * (tzz[idx]          - tzz[gx*n1+gz-1] +
                      txz[(gx+1)*n1+gz] - txz[idx]) +
                c2 * (tzz[gx*n1+gz+1]   - tzz[gx*n1+gz-2] +
                      txz[(gx+2)*n1+gz] - txz[(gx-1)*n1+gz]);
            if (ORDER >= 6) val +=
                c3 * (tzz[gx*n1+gz+2]   - tzz[gx*n1+gz-3] +
                      txz[(gx+3)*n1+gz] - txz[(gx-2)*n1+gz]);
            if (ORDER >= 8) val +=
                c4 * (tzz[gx*n1+gz+3]   - tzz[gx*n1+gz-4] +
                      txz[(gx+4)*n1+gz] - txz[(gx-3)*n1+gz]);
            vz[idx] -= roz[idx] * val;
        }
    }
}

/*====================================================================
 * 3. FORWARD STRESS UPDATE KERNEL — template<int ORDER>
 *
 * txx -= l2m * D+x(vx) + lam * D+z(vz)
 * tzz -= lam * D+x(vx) + l2m * D+z(vz)
 * txz -= mul * [D-z(vx) + D-x(vz)]
 *====================================================================*/

template<int ORDER>
__global__ void update_stress_kernel(
    const float* __restrict__ vx,
    const float* __restrict__ vz,
    float* __restrict__ txx,
    float* __restrict__ tzz,
    float* __restrict__ txz,
    const float* __restrict__ l2m,
    const float* __restrict__ lam,
    const float* __restrict__ mul,
    int n1,
    int ioPx, int ioPz, int iePx, int iePz,  /* P/Txx/Tzz bounds */
    int ioTx, int ioTz, int ieTx, int ieTz)   /* Txz bounds */
{
    const float c1 = d_fd_coeff[0];
    const float c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2];
    const float c4 = d_fd_coeff[3];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    /* --- Txx, Tzz update (P grid) --- */
    {
        int gx = ix + ioPx;
        int gz = iz + ioPz;
        if (gx < iePx && gz < iePz) {
            int idx = gx * n1 + gz;
            float dvx = c1 * (vx[(gx+1)*n1+gz] - vx[idx]) +
                        c2 * (vx[(gx+2)*n1+gz] - vx[(gx-1)*n1+gz]);
            float dvz = c1 * (vz[gx*n1+gz+1]   - vz[idx]) +
                        c2 * (vz[gx*n1+gz+2]   - vz[gx*n1+gz-1]);
            if (ORDER >= 6) {
                dvx += c3 * (vx[(gx+3)*n1+gz] - vx[(gx-2)*n1+gz]);
                dvz += c3 * (vz[gx*n1+gz+3]   - vz[gx*n1+gz-2]);
            }
            if (ORDER >= 8) {
                dvx += c4 * (vx[(gx+4)*n1+gz] - vx[(gx-3)*n1+gz]);
                dvz += c4 * (vz[gx*n1+gz+4]   - vz[gx*n1+gz-3]);
            }
            txx[idx] -= l2m[idx] * dvx + lam[idx] * dvz;
            tzz[idx] -= l2m[idx] * dvz + lam[idx] * dvx;
        }
    }

    /* --- Txz update (Txz grid) --- */
    {
        int gx = ix + ioTx;
        int gz = iz + ioTz;
        if (gx < ieTx && gz < ieTz) {
            int idx = gx * n1 + gz;
            float val =
                c1 * (vx[gx*n1+gz]     - vx[gx*n1+gz-1] +
                      vz[gx*n1+gz]     - vz[(gx-1)*n1+gz]) +
                c2 * (vx[gx*n1+gz+1]   - vx[gx*n1+gz-2] +
                      vz[(gx+1)*n1+gz] - vz[(gx-2)*n1+gz]);
            if (ORDER >= 6) val +=
                c3 * (vx[gx*n1+gz+2]   - vx[gx*n1+gz-3] +
                      vz[(gx+2)*n1+gz] - vz[(gx-3)*n1+gz]);
            if (ORDER >= 8) val +=
                c4 * (vx[gx*n1+gz+3]   - vx[gx*n1+gz-4] +
                      vz[(gx+3)*n1+gz] - vz[(gx-4)*n1+gz]);
            txz[idx] -= mul[idx] * val;
        }
    }
}

/* Explicit template instantiations — generates 3 compiled kernel variants */
template __global__ void update_velocity_kernel<4>(float*, float*, const float*, const float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void update_velocity_kernel<6>(float*, float*, const float*, const float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void update_velocity_kernel<8>(float*, float*, const float*, const float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void update_stress_kernel<4>(const float*, const float*, float*, float*, float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void update_stress_kernel<6>(const float*, const float*, float*, float*, float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void update_stress_kernel<8>(const float*, const float*, float*, float*, float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int);

/*====================================================================
 * 4. TAPER BOUNDARY KERNELS — template<int ORDER>
 *
 * The taper boundary applies the full FD stencil in the taper zone
 * and then multiplies the result by a cosine taper weight.
 *
 * Four sides (top, bottom, left, right) + four corners.
 * Each kernel handles one side's velocity or stress taper.
 * Uses same template<int ORDER> pattern as interior kernels.
 *
 * Strategy: one combined kernel per side that does:
 *   1. FD stencil update (same as interior, order-aware)
 *   2. Multiply by taper weight
 *====================================================================*/

/*--------------------------------------------------------------------
 * Helper device functions for Vx/Vz stencil computation at a point.
 * Used by both interior and taper kernels to avoid code duplication.
 *--------------------------------------------------------------------*/
template<int ORDER>
__device__ __forceinline__ float stencil_Dmx_txx_Dpz_txz(
    const float *txx, const float *txz, int gx, int gz, int n1)
{
    const float c1 = d_fd_coeff[0], c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2], c4 = d_fd_coeff[3];
    float val = c1 * (txx[gx*n1+gz]     - txx[(gx-1)*n1+gz] +
                      txz[gx*n1+gz+1]   - txz[gx*n1+gz]) +
                c2 * (txx[(gx+1)*n1+gz] - txx[(gx-2)*n1+gz] +
                      txz[gx*n1+gz+2]   - txz[gx*n1+gz-1]);
    if (ORDER >= 6) val +=
                c3 * (txx[(gx+2)*n1+gz] - txx[(gx-3)*n1+gz] +
                      txz[gx*n1+gz+3]   - txz[gx*n1+gz-2]);
    if (ORDER >= 8) val +=
                c4 * (txx[(gx+3)*n1+gz] - txx[(gx-4)*n1+gz] +
                      txz[gx*n1+gz+4]   - txz[gx*n1+gz-3]);
    return val;
}

template<int ORDER>
__device__ __forceinline__ float stencil_Dpx_txz_Dmz_tzz(
    const float *tzz, const float *txz, int gx, int gz, int n1)
{
    const float c1 = d_fd_coeff[0], c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2], c4 = d_fd_coeff[3];
    float val = c1 * (tzz[gx*n1+gz]     - tzz[gx*n1+gz-1] +
                      txz[(gx+1)*n1+gz] - txz[gx*n1+gz]) +
                c2 * (tzz[gx*n1+gz+1]   - tzz[gx*n1+gz-2] +
                      txz[(gx+2)*n1+gz] - txz[(gx-1)*n1+gz]);
    if (ORDER >= 6) val +=
                c3 * (tzz[gx*n1+gz+2]   - tzz[gx*n1+gz-3] +
                      txz[(gx+3)*n1+gz] - txz[(gx-2)*n1+gz]);
    if (ORDER >= 8) val +=
                c4 * (tzz[gx*n1+gz+3]   - tzz[gx*n1+gz-4] +
                      txz[(gx+4)*n1+gz] - txz[(gx-3)*n1+gz]);
    return val;
}

template<int ORDER>
__device__ __forceinline__ void stencil_stress_dvx_dvz(
    const float *vx, const float *vz, int gx, int gz, int n1,
    float *dvx_out, float *dvz_out)
{
    const float c1 = d_fd_coeff[0], c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2], c4 = d_fd_coeff[3];
    float dvx = c1 * (vx[(gx+1)*n1+gz] - vx[gx*n1+gz]) +
                c2 * (vx[(gx+2)*n1+gz] - vx[(gx-1)*n1+gz]);
    float dvz = c1 * (vz[gx*n1+gz+1]   - vz[gx*n1+gz]) +
                c2 * (vz[gx*n1+gz+2]   - vz[gx*n1+gz-1]);
    if (ORDER >= 6) {
        dvx += c3 * (vx[(gx+3)*n1+gz] - vx[(gx-2)*n1+gz]);
        dvz += c3 * (vz[gx*n1+gz+3]   - vz[gx*n1+gz-2]);
    }
    if (ORDER >= 8) {
        dvx += c4 * (vx[(gx+4)*n1+gz] - vx[(gx-3)*n1+gz]);
        dvz += c4 * (vz[gx*n1+gz+4]   - vz[gx*n1+gz-3]);
    }
    *dvx_out = dvx;
    *dvz_out = dvz;
}

template<int ORDER>
__device__ __forceinline__ float stencil_txz_update(
    const float *vx, const float *vz, int gx, int gz, int n1)
{
    const float c1 = d_fd_coeff[0], c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2], c4 = d_fd_coeff[3];
    float val = c1 * (vx[gx*n1+gz]     - vx[gx*n1+gz-1] +
                      vz[gx*n1+gz]     - vz[(gx-1)*n1+gz]) +
                c2 * (vx[gx*n1+gz+1]   - vx[gx*n1+gz-2] +
                      vz[(gx+1)*n1+gz] - vz[(gx-2)*n1+gz]);
    if (ORDER >= 6) val +=
                c3 * (vx[gx*n1+gz+2]   - vx[gx*n1+gz-3] +
                      vz[(gx+2)*n1+gz] - vz[(gx-3)*n1+gz]);
    if (ORDER >= 8) val +=
                c4 * (vx[gx*n1+gz+3]   - vx[gx*n1+gz-4] +
                      vz[(gx+3)*n1+gz] - vz[(gx-4)*n1+gz]);
    return val;
}

/*--------------------------------------------------------------------
 * Taper velocity kernels — one per side, template<int ORDER>
 *--------------------------------------------------------------------*/

/* --- Top taper: velocity update in taper zone above interior --- */
template<int ORDER>
__global__ void taper_top_velocity_kernel(
    float* __restrict__ vx,
    float* __restrict__ vz,
    const float* __restrict__ txx,
    const float* __restrict__ tzz,
    const float* __restrict__ txz,
    const float* __restrict__ rox,
    const float* __restrict__ roz,
    const float* __restrict__ tapx,   /* taper weights [ntap] */
    int n1, int ntap,
    int ioXx, int ieXx, int ioXz,    /* Vx bounds */
    int ioZx, int ieZx, int ioZz)    /* Vz bounds */
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz_tap = blockIdx.y * blockDim.y + threadIdx.y;
    int ib = ntap + (ioXz - ntap) - 1;

    int gx = ix + ioXx;
    int gz = iz_tap + (ioXz - ntap);
    if (gx < ieXx && iz_tap < ntap) {
        int idx = gx * n1 + gz;
        vx[idx] -= rox[idx] * stencil_Dmx_txx_Dpz_txz<ORDER>(txx, txz, gx, gz, n1);
        vx[idx] *= tapx[ib - gz];
    }

    int ib_vz = ntap + (ioZz - ntap) - 1;
    int gx_vz = ix + ioZx;
    int gz_vz = iz_tap + (ioZz - ntap);
    if (gx_vz < ieZx && iz_tap < ntap) {
        int idx = gx_vz * n1 + gz_vz;
        vz[idx] -= roz[idx] * stencil_Dpx_txz_Dmz_tzz<ORDER>(tzz, txz, gx_vz, gz_vz, n1);
        vz[idx] *= tapx[ib_vz - gz_vz];
    }
}

/* --- Bottom taper: velocity update --- */
template<int ORDER>
__global__ void taper_bot_velocity_kernel(
    float* __restrict__ vx,
    float* __restrict__ vz,
    const float* __restrict__ txx,
    const float* __restrict__ tzz,
    const float* __restrict__ txz,
    const float* __restrict__ rox,
    const float* __restrict__ roz,
    const float* __restrict__ tapx,
    int n1, int ntap,
    int ioXx, int ieXx, int ieXz,
    int ioZx, int ieZx, int ieZz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz_tap = blockIdx.y * blockDim.y + threadIdx.y;

    int gx = ix + ioXx;  int gz = iz_tap + ieXz;
    if (gx < ieXx && iz_tap < ntap) {
        int idx = gx * n1 + gz;
        vx[idx] -= rox[idx] * stencil_Dmx_txx_Dpz_txz<ORDER>(txx, txz, gx, gz, n1);
        vx[idx] *= tapx[gz - ieXz];
    }

    int gx_vz = ix + ioZx;  int gz_vz = iz_tap + ieZz;
    if (gx_vz < ieZx && iz_tap < ntap) {
        int idx = gx_vz * n1 + gz_vz;
        vz[idx] -= roz[idx] * stencil_Dpx_txz_Dmz_tzz<ORDER>(tzz, txz, gx_vz, gz_vz, n1);
        vz[idx] *= tapx[gz_vz - ieZz];
    }
}

/* --- Left taper: velocity update --- */
template<int ORDER>
__global__ void taper_left_velocity_kernel(
    float* __restrict__ vx,
    float* __restrict__ vz,
    const float* __restrict__ txx,
    const float* __restrict__ tzz,
    const float* __restrict__ txz,
    const float* __restrict__ rox,
    const float* __restrict__ roz,
    const float* __restrict__ tapz,
    int n1, int ntap,
    int ioXx, int ioXz, int ieXz,
    int ioZx, int ioZz, int ieZz)
{
    int ix_tap = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int ib = ntap + (ioXx - ntap) - 1;

    int gx = ix_tap + (ioXx - ntap);  int gz = iz + ioXz;
    if (ix_tap < ntap && gz < ieXz) {
        int idx = gx * n1 + gz;
        vx[idx] -= rox[idx] * stencil_Dmx_txx_Dpz_txz<ORDER>(txx, txz, gx, gz, n1);
        vx[idx] *= tapz[ib - gx];
    }

    int ib_vz = ntap + (ioZx - ntap) - 1;
    int gx_vz = ix_tap + (ioZx - ntap);  int gz_vz = iz + ioZz;
    if (ix_tap < ntap && gz_vz < ieZz) {
        int idx = gx_vz * n1 + gz_vz;
        vz[idx] -= roz[idx] * stencil_Dpx_txz_Dmz_tzz<ORDER>(tzz, txz, gx_vz, gz_vz, n1);
        vz[idx] *= tapz[ib_vz - gx_vz];
    }
}

/* --- Right taper: velocity update --- */
template<int ORDER>
__global__ void taper_right_velocity_kernel(
    float* __restrict__ vx,
    float* __restrict__ vz,
    const float* __restrict__ txx,
    const float* __restrict__ tzz,
    const float* __restrict__ txz,
    const float* __restrict__ rox,
    const float* __restrict__ roz,
    const float* __restrict__ tapz,
    int n1, int ntap,
    int ieXx, int ioXz, int ieXz,
    int ieZx, int ioZz, int ieZz)
{
    int ix_tap = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    int gx = ix_tap + ieXx;  int gz = iz + ioXz;
    if (ix_tap < ntap && gz < ieXz) {
        int idx = gx * n1 + gz;
        vx[idx] -= rox[idx] * stencil_Dmx_txx_Dpz_txz<ORDER>(txx, txz, gx, gz, n1);
        vx[idx] *= tapz[gx - ieXx];
    }

    int gx_vz = ix_tap + ieZx;  int gz_vz = iz + ioZz;
    if (ix_tap < ntap && gz_vz < ieZz) {
        int idx = gx_vz * n1 + gz_vz;
        vz[idx] -= roz[idx] * stencil_Dpx_txz_Dmz_tzz<ORDER>(tzz, txz, gx_vz, gz_vz, n1);
        vz[idx] *= tapz[gx_vz - ieZx];
    }
}

/*====================================================================
 * 5. TAPER STRESS KERNELS — template<int ORDER>
 *====================================================================*/

#define TAPER_STRESS_BODY_P(tap_weight_expr) \
    { \
        float dvx, dvz; \
        stencil_stress_dvx_dvz<ORDER>(vx, vz, gx, gz, n1, &dvx, &dvz); \
        txx[idx] -= l2m[idx] * dvx + lam[idx] * dvz; \
        tzz[idx] -= l2m[idx] * dvz + lam[idx] * dvx; \
        float tw = (tap_weight_expr); \
        txx[idx] *= tw;  tzz[idx] *= tw; \
    }

#define TAPER_STRESS_BODY_TXZ(tap_weight_expr) \
    { \
        txz[idx] -= mul[idx] * stencil_txz_update<ORDER>(vx, vz, gx_t, gz_t, n1); \
        txz[idx] *= (tap_weight_expr); \
    }

template<int ORDER>
__global__ void taper_top_stress_kernel(
    const float* __restrict__ vx, const float* __restrict__ vz,
    float* __restrict__ txx, float* __restrict__ tzz, float* __restrict__ txz,
    const float* __restrict__ l2m, const float* __restrict__ lam,
    const float* __restrict__ mul, const float* __restrict__ tapx,
    int n1, int ntap, int ioPx, int iePx, int ioPz, int ioTx, int ieTx, int ioTz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz_tap = blockIdx.y * blockDim.y + threadIdx.y;
    int ib = ntap + (ioPz - ntap) - 1;
    int gx = ix + ioPx;  int gz = iz_tap + (ioPz - ntap);
    if (gx < iePx && iz_tap < ntap) { int idx = gx*n1+gz; TAPER_STRESS_BODY_P(tapx[ib - gz]) }
    int ib_t = ntap + (ioTz - ntap) - 1;
    int gx_t = ix + ioTx;  int gz_t = iz_tap + (ioTz - ntap);
    if (gx_t < ieTx && iz_tap < ntap) { int idx = gx_t*n1+gz_t; TAPER_STRESS_BODY_TXZ(tapx[ib_t - gz_t]) }
}

template<int ORDER>
__global__ void taper_bot_stress_kernel(
    const float* __restrict__ vx, const float* __restrict__ vz,
    float* __restrict__ txx, float* __restrict__ tzz, float* __restrict__ txz,
    const float* __restrict__ l2m, const float* __restrict__ lam,
    const float* __restrict__ mul, const float* __restrict__ tapx,
    int n1, int ntap, int ioPx, int iePx, int iePz, int ioTx, int ieTx, int ieTz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz_tap = blockIdx.y * blockDim.y + threadIdx.y;
    int gx = ix + ioPx;  int gz = iz_tap + iePz;
    if (gx < iePx && iz_tap < ntap) { int idx = gx*n1+gz; TAPER_STRESS_BODY_P(tapx[gz - iePz]) }
    int gx_t = ix + ioTx;  int gz_t = iz_tap + ieTz;
    if (gx_t < ieTx && iz_tap < ntap) { int idx = gx_t*n1+gz_t; TAPER_STRESS_BODY_TXZ(tapx[gz_t - ieTz]) }
}

template<int ORDER>
__global__ void taper_left_stress_kernel(
    const float* __restrict__ vx, const float* __restrict__ vz,
    float* __restrict__ txx, float* __restrict__ tzz, float* __restrict__ txz,
    const float* __restrict__ l2m, const float* __restrict__ lam,
    const float* __restrict__ mul, const float* __restrict__ tapz,
    int n1, int ntap, int ioPx, int ioPz, int iePz, int ioTx, int ioTz, int ieTz)
{
    int ix_tap = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int ib = ntap + (ioPx - ntap) - 1;
    int gx = ix_tap + (ioPx - ntap);  int gz = iz + ioPz;
    if (ix_tap < ntap && gz < iePz) { int idx = gx*n1+gz; TAPER_STRESS_BODY_P(tapz[ib - gx]) }
    int ib_t = ntap + (ioTx - ntap) - 1;
    int gx_t = ix_tap + (ioTx - ntap);  int gz_t = iz + ioTz;
    if (ix_tap < ntap && gz_t < ieTz) { int idx = gx_t*n1+gz_t; TAPER_STRESS_BODY_TXZ(tapz[ib_t - gx_t]) }
}

template<int ORDER>
__global__ void taper_right_stress_kernel(
    const float* __restrict__ vx, const float* __restrict__ vz,
    float* __restrict__ txx, float* __restrict__ tzz, float* __restrict__ txz,
    const float* __restrict__ l2m, const float* __restrict__ lam,
    const float* __restrict__ mul, const float* __restrict__ tapz,
    int n1, int ntap, int iePx, int ioPz, int iePz, int ieTx, int ioTz, int ieTz)
{
    int ix_tap = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int gx = ix_tap + iePx;  int gz = iz + ioPz;
    if (ix_tap < ntap && gz < iePz) { int idx = gx*n1+gz; TAPER_STRESS_BODY_P(tapz[gx - iePx]) }
    int gx_t = ix_tap + ieTx;  int gz_t = iz + ioTz;
    if (ix_tap < ntap && gz_t < ieTz) { int idx = gx_t*n1+gz_t; TAPER_STRESS_BODY_TXZ(tapz[gx_t - ieTx]) }
}

#undef TAPER_STRESS_BODY_P
#undef TAPER_STRESS_BODY_TXZ

/*====================================================================
 * 6. FREE SURFACE KERNEL (bnd.top == 1)
 *
 * Mirrors vz at the surface: vz[iz] = vz[iz+1], vz[iz-1] = vz[iz+2]
 * For stress: txx = 0 and tzz = -(l2m-2*lam)/l2m * txx at surface
 * (handled by boundariesV in the CPU code)
 *====================================================================*/

__global__ void free_surface_mirror_kernel(
    float* __restrict__ vz,
    const int* __restrict__ surface,  /* surface[ix] = iz of surface */
    int n1, int ioPx, int iePx)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x + ioPx;
    if (ix < iePx) {
        int iz = surface[ix];
        vz[ix * n1 + iz]     = vz[ix * n1 + iz + 1];
        vz[ix * n1 + iz - 1] = vz[ix * n1 + iz + 2];
    }
}

/*====================================================================
 * 7. SOURCE INJECTION KERNEL
 *
 * Elastic monopole source (src.type == 1, src.orient == 1):
 *   txx[ix,iz] += src_ampl
 *   tzz[ix,iz] += src_ampl
 *
 * where src_ampl = wavelet_sample * (1/dx) * l2m[ix,iz]
 *
 * For simplicity, this kernel handles the most common case.
 * Other source types (force, dipole, txz) can be added similarly.
 *====================================================================*/

__global__ void inject_source_kernel(
    float* __restrict__ txx,
    float* __restrict__ tzz,
    float* __restrict__ txz,
    float* __restrict__ vx,
    float* __restrict__ vz,
    const float* __restrict__ l2m,
    const float* __restrict__ rox,
    const float* __restrict__ roz,
    int ix_src, int iz_src,   /* local grid position (padded coords) */
    float src_ampl,           /* pre-computed wavelet amplitude for this time step */
    int src_type,             /* 1=stress monopole, 6=force-x, 7=force-z */
    int src_orient,           /* 1=monopole, 2-5=dipoles */
    int n1)
{
    /* Single-thread kernel — source injection is not a bottleneck */
    if (threadIdx.x != 0) return;

    int idx = ix_src * n1 + iz_src;

    if (src_type == 1) {  /* Stress source */
        if (src_orient == 1) {  /* monopole */
            txx[idx] += src_ampl;
            tzz[idx] += src_ampl;
        }
        else if (src_orient == 2) { /* dipole +/- in z */
            txx[idx] += src_ampl;
            tzz[idx] += src_ampl;
            txx[ix_src*n1+iz_src+1] -= src_ampl;
            tzz[ix_src*n1+iz_src+1] -= src_ampl;
        }
        else if (src_orient == 3) { /* dipole -/+ in x */
            txx[idx] += src_ampl;
            tzz[idx] += src_ampl;
            txx[(ix_src-1)*n1+iz_src] -= src_ampl;
            tzz[(ix_src-1)*n1+iz_src] -= src_ampl;
        }
    }
    else if (src_type == 2) { /* Txz source */
        txz[idx] += src_ampl;
    }
    else if (src_type == 6) { /* Force-x */
        vx[idx] += 0.5f * src_ampl * rox[idx] / l2m[idx];
    }
    else if (src_type == 7) { /* Force-z */
        vz[idx] += src_ampl * roz[idx] / l2m[idx];
    }
}

/*====================================================================
 * 8. RECEIVER EXTRACTION KERNEL
 *
 * Records wavefield values at receiver positions for the current
 * time step. Each thread handles one receiver.
 *
 * rec_data layout: [nrec × nt], column-major (irec fast, it slow)
 *====================================================================*/

__global__ void extract_receivers_kernel(
    const float* __restrict__ vx,
    const float* __restrict__ vz,
    const float* __restrict__ txx,
    const float* __restrict__ tzz,
    const float* __restrict__ txz,
    float* __restrict__ rec_vx,     /* [nrec × nt] or NULL */
    float* __restrict__ rec_vz,
    float* __restrict__ rec_txx,
    float* __restrict__ rec_tzz,
    float* __restrict__ rec_p,      /* hydrophone: 0.5*(txx+tzz) */
    const int* __restrict__ rec_ix,  /* local x-indices [nrec] */
    const int* __restrict__ rec_iz,  /* z-indices [nrec] */
    int nrec, int it, int nt, int n1)
{
    int irec = blockIdx.x * blockDim.x + threadIdx.x;
    if (irec >= nrec) return;

    int ix = rec_ix[irec];
    int iz = rec_iz[irec];
    int idx = ix * n1 + iz;
    int out_idx = it * nrec + irec;  /* column-major: [nrec × nt] */

    if (rec_vx)  rec_vx[out_idx]  = vx[idx];
    if (rec_vz)  rec_vz[out_idx]  = vz[idx];
    if (rec_txx) rec_txx[out_idx] = txx[idx];
    if (rec_tzz) rec_tzz[out_idx] = tzz[idx];
    if (rec_p)   rec_p[out_idx]   = 0.5f * (txx[idx] + tzz[idx]);
}

/*====================================================================
 * 9. DEVICE MEMORY MANAGEMENT
 *====================================================================*/

/* Wavefield arrays on device (one set per shot at a time) */
typedef struct _deviceWfl {
    float *vx, *vz;
    float *txx, *tzz, *txz;
    size_t field_size;   /* nax_local × naz × sizeof(float) */
    int nax, naz;
} deviceWfl;

/* Material parameter arrays on device (persistent across shots) */
typedef struct _deviceMod {
    float *l2m, *lam, *muu;
    float *rox, *roz;
    size_t field_size;
    int nax, naz;
} deviceMod;

/* Taper arrays on device */
typedef struct _deviceBnd {
    float *tapx;       /* z-direction taper weights [ntap] */
    float *tapz;       /* x-direction taper weights [ntap] */
    float *tapxz;      /* corner taper weights [ntap × ntap] */
    int   *surface;    /* surface index array [nax] */
    int    ntap;
} deviceBnd;

/* Receiver recording buffers on device */
typedef struct _deviceRec {
    float *rec_vx, *rec_vz, *rec_txx, *rec_tzz, *rec_p;
    int   *rec_ix, *rec_iz;   /* local receiver positions */
    int    nrec;
    int    nt;
} deviceRec;

#ifdef __cplusplus
extern "C" {
#endif

void cuda_alloc_wfl(deviceWfl *wfl, int nax, int naz)
{
    wfl->nax = nax;
    wfl->naz = naz;
    wfl->field_size = (size_t)nax * naz * sizeof(float);

    CUDA_CHECK(cudaMalloc(&wfl->vx,  wfl->field_size));
    CUDA_CHECK(cudaMalloc(&wfl->vz,  wfl->field_size));
    CUDA_CHECK(cudaMalloc(&wfl->txx, wfl->field_size));
    CUDA_CHECK(cudaMalloc(&wfl->tzz, wfl->field_size));
    CUDA_CHECK(cudaMalloc(&wfl->txz, wfl->field_size));

    CUDA_CHECK(cudaMemset(wfl->vx,  0, wfl->field_size));
    CUDA_CHECK(cudaMemset(wfl->vz,  0, wfl->field_size));
    CUDA_CHECK(cudaMemset(wfl->txx, 0, wfl->field_size));
    CUDA_CHECK(cudaMemset(wfl->tzz, 0, wfl->field_size));
    CUDA_CHECK(cudaMemset(wfl->txz, 0, wfl->field_size));
}

void cuda_free_wfl(deviceWfl *wfl)
{
    if (wfl->vx)  CUDA_CHECK(cudaFree(wfl->vx));
    if (wfl->vz)  CUDA_CHECK(cudaFree(wfl->vz));
    if (wfl->txx) CUDA_CHECK(cudaFree(wfl->txx));
    if (wfl->tzz) CUDA_CHECK(cudaFree(wfl->tzz));
    if (wfl->txz) CUDA_CHECK(cudaFree(wfl->txz));
}

void cuda_zero_wfl(deviceWfl *wfl)
{
    CUDA_CHECK(cudaMemset(wfl->vx,  0, wfl->field_size));
    CUDA_CHECK(cudaMemset(wfl->vz,  0, wfl->field_size));
    CUDA_CHECK(cudaMemset(wfl->txx, 0, wfl->field_size));
    CUDA_CHECK(cudaMemset(wfl->tzz, 0, wfl->field_size));
    CUDA_CHECK(cudaMemset(wfl->txz, 0, wfl->field_size));
}

void cuda_alloc_mod(deviceMod *mod, int nax, int naz)
{
    mod->nax = nax;
    mod->naz = naz;
    mod->field_size = (size_t)nax * naz * sizeof(float);

    CUDA_CHECK(cudaMalloc(&mod->l2m, mod->field_size));
    CUDA_CHECK(cudaMalloc(&mod->lam, mod->field_size));
    CUDA_CHECK(cudaMalloc(&mod->muu, mod->field_size));
    CUDA_CHECK(cudaMalloc(&mod->rox, mod->field_size));
    CUDA_CHECK(cudaMalloc(&mod->roz, mod->field_size));
}

void cuda_free_mod(deviceMod *mod)
{
    if (mod->l2m) CUDA_CHECK(cudaFree(mod->l2m));
    if (mod->lam) CUDA_CHECK(cudaFree(mod->lam));
    if (mod->muu) CUDA_CHECK(cudaFree(mod->muu));
    if (mod->rox) CUDA_CHECK(cudaFree(mod->rox));
    if (mod->roz) CUDA_CHECK(cudaFree(mod->roz));
}

void cuda_upload_mod(deviceMod *dmod,
                      const float *h_l2m, const float *h_lam, const float *h_muu,
                      const float *h_rox, const float *h_roz)
{
    CUDA_CHECK(cudaMemcpy(dmod->l2m, h_l2m, dmod->field_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmod->lam, h_lam, dmod->field_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmod->muu, h_muu, dmod->field_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmod->rox, h_rox, dmod->field_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmod->roz, h_roz, dmod->field_size, cudaMemcpyHostToDevice));
}

void cuda_alloc_bnd(deviceBnd *bnd, int ntap, int nax,
                     const float *h_tapx, const float *h_tapz,
                     const float *h_tapxz, const int *h_surface)
{
    bnd->ntap = ntap;
    if (ntap > 0 && h_tapx) {
        CUDA_CHECK(cudaMalloc(&bnd->tapx, ntap * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(bnd->tapx, h_tapx, ntap * sizeof(float), cudaMemcpyHostToDevice));
    } else { bnd->tapx = NULL; }

    if (ntap > 0 && h_tapz) {
        CUDA_CHECK(cudaMalloc(&bnd->tapz, ntap * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(bnd->tapz, h_tapz, ntap * sizeof(float), cudaMemcpyHostToDevice));
    } else { bnd->tapz = NULL; }

    if (ntap > 0 && h_tapxz) {
        CUDA_CHECK(cudaMalloc(&bnd->tapxz, ntap * ntap * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(bnd->tapxz, h_tapxz, ntap * ntap * sizeof(float), cudaMemcpyHostToDevice));
    } else { bnd->tapxz = NULL; }

    if (h_surface) {
        CUDA_CHECK(cudaMalloc(&bnd->surface, nax * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(bnd->surface, h_surface, nax * sizeof(int), cudaMemcpyHostToDevice));
    } else { bnd->surface = NULL; }
}

void cuda_free_bnd(deviceBnd *bnd)
{
    if (bnd->tapx)    CUDA_CHECK(cudaFree(bnd->tapx));
    if (bnd->tapz)    CUDA_CHECK(cudaFree(bnd->tapz));
    if (bnd->tapxz)   CUDA_CHECK(cudaFree(bnd->tapxz));
    if (bnd->surface) CUDA_CHECK(cudaFree(bnd->surface));
}

void cuda_alloc_rec(deviceRec *rec, int nrec, int nt,
                     const int *h_rec_ix, const int *h_rec_iz,
                     int rec_vx_flag, int rec_vz_flag,
                     int rec_txx_flag, int rec_tzz_flag, int rec_p_flag)
{
    rec->nrec = nrec;
    rec->nt = nt;
    size_t trace_bytes = (size_t)nrec * nt * sizeof(float);

    rec->rec_vx  = NULL; rec->rec_vz = NULL;
    rec->rec_txx = NULL; rec->rec_tzz = NULL;
    rec->rec_p   = NULL;

    if (nrec > 0) {
        CUDA_CHECK(cudaMalloc(&rec->rec_ix, nrec * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&rec->rec_iz, nrec * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(rec->rec_ix, h_rec_ix, nrec * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rec->rec_iz, h_rec_iz, nrec * sizeof(int), cudaMemcpyHostToDevice));

        if (rec_vx_flag)  { CUDA_CHECK(cudaMalloc(&rec->rec_vx,  trace_bytes)); CUDA_CHECK(cudaMemset(rec->rec_vx,  0, trace_bytes)); }
        if (rec_vz_flag)  { CUDA_CHECK(cudaMalloc(&rec->rec_vz,  trace_bytes)); CUDA_CHECK(cudaMemset(rec->rec_vz,  0, trace_bytes)); }
        if (rec_txx_flag) { CUDA_CHECK(cudaMalloc(&rec->rec_txx, trace_bytes)); CUDA_CHECK(cudaMemset(rec->rec_txx, 0, trace_bytes)); }
        if (rec_tzz_flag) { CUDA_CHECK(cudaMalloc(&rec->rec_tzz, trace_bytes)); CUDA_CHECK(cudaMemset(rec->rec_tzz, 0, trace_bytes)); }
        if (rec_p_flag)   { CUDA_CHECK(cudaMalloc(&rec->rec_p,   trace_bytes)); CUDA_CHECK(cudaMemset(rec->rec_p,   0, trace_bytes)); }
    }
}

void cuda_free_rec(deviceRec *rec)
{
    if (rec->rec_vx)  CUDA_CHECK(cudaFree(rec->rec_vx));
    if (rec->rec_vz)  CUDA_CHECK(cudaFree(rec->rec_vz));
    if (rec->rec_txx) CUDA_CHECK(cudaFree(rec->rec_txx));
    if (rec->rec_tzz) CUDA_CHECK(cudaFree(rec->rec_tzz));
    if (rec->rec_p)   CUDA_CHECK(cudaFree(rec->rec_p));
    if (rec->rec_ix)  CUDA_CHECK(cudaFree(rec->rec_ix));
    if (rec->rec_iz)  CUDA_CHECK(cudaFree(rec->rec_iz));
}

/*====================================================================
 * 10. FD COEFFICIENT UPLOAD
 *====================================================================*/

void cuda_set_fd_coefficients(int iorder)
{
    float h_coeff[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (iorder <= 4) {
        h_coeff[0] = 9.0f / 8.0f;
        h_coeff[1] = -1.0f / 24.0f;
    } else if (iorder == 6) {
        h_coeff[0] = 75.0f / 64.0f;
        h_coeff[1] = -25.0f / 384.0f;
        h_coeff[2] = 3.0f / 640.0f;
    } else { /* iorder >= 8 */
        h_coeff[0] = 1225.0f / 1024.0f;
        h_coeff[1] = -245.0f / 3072.0f;
        h_coeff[2] = 49.0f / 5120.0f;
        h_coeff[3] = -5.0f / 7168.0f;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(d_fd_coeff, h_coeff, 4 * sizeof(float)));
}

/*====================================================================
 * 11. KERNEL LAUNCH WRAPPERS
 *
 * These are called from the C host code (fdfwimodc.c or fwi_inversion.c)
 * via extern "C" linkage.
 *====================================================================*/

/*--------------------------------------------------------------------
 * Order-dispatch macro: selects template<4>, <6>, or <8> at runtime
 *--------------------------------------------------------------------*/
#define DISPATCH_ORDER(iorder, kernel_call_4, kernel_call_6, kernel_call_8) \
    do { \
        if ((iorder) <= 4)      { kernel_call_4; } \
        else if ((iorder) == 6) { kernel_call_6; } \
        else                    { kernel_call_8; } \
    } while(0)

void cuda_launch_velocity_update(
    deviceWfl *wfl, deviceMod *dmod,
    int n1, int iorder,
    int ioXx, int ioXz, int ieXx, int ieXz,
    int ioZx, int ioZz, int ieZx, int ieZz,
    cudaStream_t stream)
{
    int nx_vx = ieXx - ioXx;  int nz_vx = ieXz - ioXz;
    int nx_vz = ieZx - ioZx;  int nz_vz = ieZz - ioZz;
    int nx_max = (nx_vx > nx_vz) ? nx_vx : nx_vz;
    int nz_max = (nz_vx > nz_vz) ? nz_vx : nz_vz;

    dim3 block(BLOCK_X, BLOCK_Z);
    dim3 grid(cuda_div_ceil(nx_max, BLOCK_X),
              cuda_div_ceil(nz_max, BLOCK_Z));

    DISPATCH_ORDER(iorder,
        update_velocity_kernel<4><<<grid, block, 0, stream>>>(
            wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz,
            dmod->rox, dmod->roz, n1,
            ioXx, ioXz, ieXx, ieXz, ioZx, ioZz, ieZx, ieZz),
        update_velocity_kernel<6><<<grid, block, 0, stream>>>(
            wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz,
            dmod->rox, dmod->roz, n1,
            ioXx, ioXz, ieXx, ieXz, ioZx, ioZz, ieZx, ieZz),
        update_velocity_kernel<8><<<grid, block, 0, stream>>>(
            wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz,
            dmod->rox, dmod->roz, n1,
            ioXx, ioXz, ieXx, ieXz, ioZx, ioZz, ieZx, ieZz)
    );
}

void cuda_launch_stress_update(
    deviceWfl *wfl, deviceMod *dmod,
    int n1, int iorder,
    int ioPx, int ioPz, int iePx, int iePz,
    int ioTx, int ioTz, int ieTx, int ieTz,
    cudaStream_t stream)
{
    int nx_p = iePx - ioPx;  int nz_p = iePz - ioPz;
    int nx_t = ieTx - ioTx;  int nz_t = ieTz - ioTz;
    int nx_max = (nx_p > nx_t) ? nx_p : nx_t;
    int nz_max = (nz_p > nz_t) ? nz_p : nz_t;

    dim3 block(BLOCK_X, BLOCK_Z);
    dim3 grid(cuda_div_ceil(nx_max, BLOCK_X),
              cuda_div_ceil(nz_max, BLOCK_Z));

    DISPATCH_ORDER(iorder,
        update_stress_kernel<4><<<grid, block, 0, stream>>>(
            wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz,
            dmod->l2m, dmod->lam, dmod->muu, n1,
            ioPx, ioPz, iePx, iePz, ioTx, ioTz, ieTx, ieTz),
        update_stress_kernel<6><<<grid, block, 0, stream>>>(
            wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz,
            dmod->l2m, dmod->lam, dmod->muu, n1,
            ioPx, ioPz, iePx, iePz, ioTx, ioTz, ieTx, ieTz),
        update_stress_kernel<8><<<grid, block, 0, stream>>>(
            wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz,
            dmod->l2m, dmod->lam, dmod->muu, n1,
            ioPx, ioPz, iePx, iePz, ioTx, ioTz, ieTx, ieTz)
    );
}

/* Taper launch helper macro — avoids repeating the dispatch for each side */
#define LAUNCH_TAPER_VEL(side, order, ...) \
    taper_##side##_velocity_kernel<order><<<grid, block, 0, stream>>>(__VA_ARGS__)

#define LAUNCH_TAPER_STR(side, order, ...) \
    taper_##side##_stress_kernel<order><<<grid, block, 0, stream>>>(__VA_ARGS__)

void cuda_launch_taper_velocity(
    deviceWfl *wfl, deviceMod *dmod, deviceBnd *dbnd,
    int n1, int ntap, int iorder,
    int ioXx, int ioXz, int ieXx, int ieXz,
    int ioZx, int ioZz, int ieZx, int ieZz,
    int has_top, int has_bot, int has_left, int has_right,
    cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Z);

    if (has_top && dbnd->tapx) {
        int nx_range = ieXx - ioXx;
        if (nx_range < (ieZx - ioZx)) nx_range = ieZx - ioZx;
        dim3 grid(cuda_div_ceil(nx_range, BLOCK_X), cuda_div_ceil(ntap, BLOCK_Z));
        DISPATCH_ORDER(iorder,
            LAUNCH_TAPER_VEL(top, 4, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapx, n1, ntap, ioXx, ieXx, ioXz, ioZx, ieZx, ioZz),
            LAUNCH_TAPER_VEL(top, 6, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapx, n1, ntap, ioXx, ieXx, ioXz, ioZx, ieZx, ioZz),
            LAUNCH_TAPER_VEL(top, 8, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapx, n1, ntap, ioXx, ieXx, ioXz, ioZx, ieZx, ioZz));
    }

    if (has_bot && dbnd->tapx) {
        int nx_range = ieXx - ioXx;
        if (nx_range < (ieZx - ioZx)) nx_range = ieZx - ioZx;
        dim3 grid(cuda_div_ceil(nx_range, BLOCK_X), cuda_div_ceil(ntap, BLOCK_Z));
        DISPATCH_ORDER(iorder,
            LAUNCH_TAPER_VEL(bot, 4, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapx, n1, ntap, ioXx, ieXx, ieXz, ioZx, ieZx, ieZz),
            LAUNCH_TAPER_VEL(bot, 6, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapx, n1, ntap, ioXx, ieXx, ieXz, ioZx, ieZx, ieZz),
            LAUNCH_TAPER_VEL(bot, 8, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapx, n1, ntap, ioXx, ieXx, ieXz, ioZx, ieZx, ieZz));
    }

    if (has_left && dbnd->tapz) {
        int nz_range = ieXz - ioXz;
        if (nz_range < (ieZz - ioZz)) nz_range = ieZz - ioZz;
        dim3 grid(cuda_div_ceil(ntap, BLOCK_X), cuda_div_ceil(nz_range, BLOCK_Z));
        DISPATCH_ORDER(iorder,
            LAUNCH_TAPER_VEL(left, 4, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapz, n1, ntap, ioXx, ioXz, ieXz, ioZx, ioZz, ieZz),
            LAUNCH_TAPER_VEL(left, 6, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapz, n1, ntap, ioXx, ioXz, ieXz, ioZx, ioZz, ieZz),
            LAUNCH_TAPER_VEL(left, 8, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapz, n1, ntap, ioXx, ioXz, ieXz, ioZx, ioZz, ieZz));
    }

    if (has_right && dbnd->tapz) {
        int nz_range = ieXz - ioXz;
        if (nz_range < (ieZz - ioZz)) nz_range = ieZz - ioZz;
        dim3 grid(cuda_div_ceil(ntap, BLOCK_X), cuda_div_ceil(nz_range, BLOCK_Z));
        DISPATCH_ORDER(iorder,
            LAUNCH_TAPER_VEL(right, 4, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapz, n1, ntap, ieXx, ioXz, ieXz, ieZx, ioZz, ieZz),
            LAUNCH_TAPER_VEL(right, 6, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapz, n1, ntap, ieXx, ioXz, ieXz, ieZx, ioZz, ieZz),
            LAUNCH_TAPER_VEL(right, 8, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->rox, dmod->roz, dbnd->tapz, n1, ntap, ieXx, ioXz, ieXz, ieZx, ioZz, ieZz));
    }
}

void cuda_launch_taper_stress(
    deviceWfl *wfl, deviceMod *dmod, deviceBnd *dbnd,
    int n1, int ntap, int iorder,
    int ioPx, int ioPz, int iePx, int iePz,
    int ioTx, int ioTz, int ieTx, int ieTz,
    int has_top, int has_bot, int has_left, int has_right,
    cudaStream_t stream)
{
    dim3 block(BLOCK_X, BLOCK_Z);

    if (has_top && dbnd->tapx) {
        int nx_range = iePx - ioPx;
        if (nx_range < (ieTx - ioTx)) nx_range = ieTx - ioTx;
        dim3 grid(cuda_div_ceil(nx_range, BLOCK_X), cuda_div_ceil(ntap, BLOCK_Z));
        DISPATCH_ORDER(iorder,
            LAUNCH_TAPER_STR(top, 4, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapx, n1, ntap, ioPx, iePx, ioPz, ioTx, ieTx, ioTz),
            LAUNCH_TAPER_STR(top, 6, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapx, n1, ntap, ioPx, iePx, ioPz, ioTx, ieTx, ioTz),
            LAUNCH_TAPER_STR(top, 8, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapx, n1, ntap, ioPx, iePx, ioPz, ioTx, ieTx, ioTz));
    }

    if (has_bot && dbnd->tapx) {
        int nx_range = iePx - ioPx;
        if (nx_range < (ieTx - ioTx)) nx_range = ieTx - ioTx;
        dim3 grid(cuda_div_ceil(nx_range, BLOCK_X), cuda_div_ceil(ntap, BLOCK_Z));
        DISPATCH_ORDER(iorder,
            LAUNCH_TAPER_STR(bot, 4, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapx, n1, ntap, ioPx, iePx, iePz, ioTx, ieTx, ieTz),
            LAUNCH_TAPER_STR(bot, 6, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapx, n1, ntap, ioPx, iePx, iePz, ioTx, ieTx, ieTz),
            LAUNCH_TAPER_STR(bot, 8, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapx, n1, ntap, ioPx, iePx, iePz, ioTx, ieTx, ieTz));
    }

    if (has_left && dbnd->tapz) {
        int nz_range = iePz - ioPz;
        if (nz_range < (ieTz - ioTz)) nz_range = ieTz - ioTz;
        dim3 grid(cuda_div_ceil(ntap, BLOCK_X), cuda_div_ceil(nz_range, BLOCK_Z));
        DISPATCH_ORDER(iorder,
            LAUNCH_TAPER_STR(left, 4, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapz, n1, ntap, ioPx, ioPz, iePz, ioTx, ioTz, ieTz),
            LAUNCH_TAPER_STR(left, 6, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapz, n1, ntap, ioPx, ioPz, iePz, ioTx, ioTz, ieTz),
            LAUNCH_TAPER_STR(left, 8, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapz, n1, ntap, ioPx, ioPz, iePz, ioTx, ioTz, ieTz));
    }

    if (has_right && dbnd->tapz) {
        int nz_range = iePz - ioPz;
        if (nz_range < (ieTz - ioTz)) nz_range = ieTz - ioTz;
        dim3 grid(cuda_div_ceil(ntap, BLOCK_X), cuda_div_ceil(nz_range, BLOCK_Z));
        DISPATCH_ORDER(iorder,
            LAUNCH_TAPER_STR(right, 4, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapz, n1, ntap, iePx, ioPz, iePz, ieTx, ioTz, ieTz),
            LAUNCH_TAPER_STR(right, 6, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapz, n1, ntap, iePx, ioPz, iePz, ieTx, ioTz, ieTz),
            LAUNCH_TAPER_STR(right, 8, wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz, dmod->l2m, dmod->lam, dmod->muu, dbnd->tapz, n1, ntap, iePx, ioPz, iePz, ieTx, ioTz, ieTz));
    }
}

void cuda_launch_free_surface(
    deviceWfl *wfl, deviceBnd *dbnd,
    int n1, int ioPx, int iePx,
    cudaStream_t stream)
{
    if (!dbnd->surface) return;
    int nx = iePx - ioPx;
    dim3 block(256);
    dim3 grid(cuda_div_ceil(nx, 256));
    free_surface_mirror_kernel<<<grid, block, 0, stream>>>(
        wfl->vz, dbnd->surface, n1, ioPx, iePx);
}

void cuda_launch_inject_source(
    deviceWfl *wfl, deviceMod *dmod,
    int ix_src, int iz_src, float src_ampl,
    int src_type, int src_orient, int n1,
    cudaStream_t stream)
{
    inject_source_kernel<<<1, 1, 0, stream>>>(
        wfl->txx, wfl->tzz, wfl->txz, wfl->vx, wfl->vz,
        dmod->l2m, dmod->rox, dmod->roz,
        ix_src, iz_src, src_ampl, src_type, src_orient, n1);
}

void cuda_launch_extract_receivers(
    deviceWfl *wfl, deviceRec *drec,
    int it, int n1, cudaStream_t stream)
{
    if (drec->nrec <= 0) return;
    dim3 block(256);
    dim3 grid(cuda_div_ceil(drec->nrec, 256));
    extract_receivers_kernel<<<grid, block, 0, stream>>>(
        wfl->vx, wfl->vz, wfl->txx, wfl->tzz, wfl->txz,
        drec->rec_vx, drec->rec_vz, drec->rec_txx, drec->rec_tzz, drec->rec_p,
        drec->rec_ix, drec->rec_iz,
        drec->nrec, it, drec->nt, n1);
}

#ifdef __cplusplus
}
#endif

#endif /* USE_CUDA */
