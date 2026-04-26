/*********************************************************************
 *
 * fdelfwi_cuda.cu — CUDA kernels for FWI (adjoint, gradient, checkpoint)
 *
 * Implements GPU versions of:
 *   - elastic4/6/8_adj.c  adjoint velocity + stress update (templated)
 *   - fwi_gradient.c      gradient cross-correlation (lambda, mu, rho)
 *   - applyAdjointSource.c adjoint source injection
 *   - checkpoint save/load  async D2H / H2D
 *   - Free-surface txz correction for adjoint (order-dependent)
 *
 * All kernels use template<int ORDER> for 4th/6th/8th order support.
 * Stencil helper functions from fdelmodc_cuda.cu are NOT reused here
 * because the adjoint stencil has material params INSIDE the derivative.
 *
 * Author: FWI CUDA porting project
 *
 *********************************************************************/

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_utils.h"

/* FD coefficients in constant memory (defined in fdelmodc_cuda.cu) */
__constant__ float d_fd_coeff[4];

/* Forward declarations of device structs from fdelmodc_cuda.cu */
struct _deviceWfl;
struct _deviceMod;
typedef struct _deviceWfl deviceWfl;
typedef struct _deviceMod deviceMod;

/* Dispatch macro (same as in fdelmodc_cuda.cu) */
#define DISPATCH_ORDER(iorder, call4, call6, call8) \
    do { \
        if ((iorder) <= 4)      { call4; } \
        else if ((iorder) == 6) { call6; } \
        else                    { call8; } \
    } while(0)

/*====================================================================
 * 1. ADJOINT VELOCITY UPDATE KERNEL — template<int ORDER>
 *
 * TRUE ADJOINT of the forward stress update (F2)^T:
 *   vx += D-x(l2m*txx) + D-x(lam*tzz) + D+z(mul*txz)
 *   vz += D-z(lam*txx) + D-z(l2m*tzz) + D+x(mul*txz)
 *
 * KEY DIFFERENCE from forward: material params (l2m, lam, mul) are
 * INSIDE the spatial derivative (pre-multiply field before differencing).
 *====================================================================*/

template<int ORDER>
__global__ void adj_update_velocity_kernel(
    float* __restrict__ vx,
    float* __restrict__ vz,
    const float* __restrict__ txx,
    const float* __restrict__ tzz,
    const float* __restrict__ txz,
    const float* __restrict__ l2m,
    const float* __restrict__ lam,
    const float* __restrict__ mul,
    int n1,
    int ioXx, int ioXz, int ieXx, int ieXz,
    int ioZx, int ioZz, int ieZx, int ieZz)
{
    const float c1 = d_fd_coeff[0], c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2], c4 = d_fd_coeff[3];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    /* --- Adjoint Vx update --- */
    {
        int gx = ix + ioXx;
        int gz = iz + ioXz;
        if (gx < ieXx && gz < ieXz) {
            int idx = gx * n1 + gz;
            float val =
                c1*(l2m[idx]*txx[idx]             - l2m[(gx-1)*n1+gz]*txx[(gx-1)*n1+gz] +
                    lam[idx]*tzz[idx]             - lam[(gx-1)*n1+gz]*tzz[(gx-1)*n1+gz] +
                    mul[gx*n1+gz+1]*txz[gx*n1+gz+1] - mul[idx]*txz[idx]) +
                c2*(l2m[(gx+1)*n1+gz]*txx[(gx+1)*n1+gz] - l2m[(gx-2)*n1+gz]*txx[(gx-2)*n1+gz] +
                    lam[(gx+1)*n1+gz]*tzz[(gx+1)*n1+gz] - lam[(gx-2)*n1+gz]*tzz[(gx-2)*n1+gz] +
                    mul[gx*n1+gz+2]*txz[gx*n1+gz+2]   - mul[gx*n1+gz-1]*txz[gx*n1+gz-1]);
            if (ORDER >= 6) val +=
                c3*(l2m[(gx+2)*n1+gz]*txx[(gx+2)*n1+gz] - l2m[(gx-3)*n1+gz]*txx[(gx-3)*n1+gz] +
                    lam[(gx+2)*n1+gz]*tzz[(gx+2)*n1+gz] - lam[(gx-3)*n1+gz]*tzz[(gx-3)*n1+gz] +
                    mul[gx*n1+gz+3]*txz[gx*n1+gz+3]   - mul[gx*n1+gz-2]*txz[gx*n1+gz-2]);
            if (ORDER >= 8) val +=
                c4*(l2m[(gx+3)*n1+gz]*txx[(gx+3)*n1+gz] - l2m[(gx-4)*n1+gz]*txx[(gx-4)*n1+gz] +
                    lam[(gx+3)*n1+gz]*tzz[(gx+3)*n1+gz] - lam[(gx-4)*n1+gz]*tzz[(gx-4)*n1+gz] +
                    mul[gx*n1+gz+4]*txz[gx*n1+gz+4]   - mul[gx*n1+gz-3]*txz[gx*n1+gz-3]);
            vx[idx] += val;
        }
    }

    /* --- Adjoint Vz update --- */
    {
        int gx = ix + ioZx;
        int gz = iz + ioZz;
        if (gx < ieZx && gz < ieZz) {
            int idx = gx * n1 + gz;
            float val =
                c1*(lam[idx]*txx[idx]             - lam[gx*n1+gz-1]*txx[gx*n1+gz-1] +
                    l2m[idx]*tzz[idx]             - l2m[gx*n1+gz-1]*tzz[gx*n1+gz-1] +
                    mul[(gx+1)*n1+gz]*txz[(gx+1)*n1+gz] - mul[idx]*txz[idx]) +
                c2*(lam[gx*n1+gz+1]*txx[gx*n1+gz+1] - lam[gx*n1+gz-2]*txx[gx*n1+gz-2] +
                    l2m[gx*n1+gz+1]*tzz[gx*n1+gz+1] - l2m[gx*n1+gz-2]*tzz[gx*n1+gz-2] +
                    mul[(gx+2)*n1+gz]*txz[(gx+2)*n1+gz] - mul[(gx-1)*n1+gz]*txz[(gx-1)*n1+gz]);
            if (ORDER >= 6) val +=
                c3*(lam[gx*n1+gz+2]*txx[gx*n1+gz+2] - lam[gx*n1+gz-3]*txx[gx*n1+gz-3] +
                    l2m[gx*n1+gz+2]*tzz[gx*n1+gz+2] - l2m[gx*n1+gz-3]*tzz[gx*n1+gz-3] +
                    mul[(gx+3)*n1+gz]*txz[(gx+3)*n1+gz] - mul[(gx-2)*n1+gz]*txz[(gx-2)*n1+gz]);
            if (ORDER >= 8) val +=
                c4*(lam[gx*n1+gz+3]*txx[gx*n1+gz+3] - lam[gx*n1+gz-4]*txx[gx*n1+gz-4] +
                    l2m[gx*n1+gz+3]*tzz[gx*n1+gz+3] - l2m[gx*n1+gz-4]*tzz[gx*n1+gz-4] +
                    mul[(gx+4)*n1+gz]*txz[(gx+4)*n1+gz] - mul[(gx-3)*n1+gz]*txz[(gx-3)*n1+gz]);
            vz[idx] += val;
        }
    }
}

/*====================================================================
 * 2. ADJOINT STRESS UPDATE KERNEL — template<int ORDER>
 *
 * TRUE ADJOINT of the forward velocity update (F1)^T:
 *   txx += D+x(rox*vx)
 *   tzz += D+z(roz*vz)
 *   txz += D-z(rox*vx) + D-x(roz*vz)
 *
 * Buoyancy (rox, roz) INSIDE the derivative.
 *====================================================================*/

template<int ORDER>
__global__ void adj_update_stress_kernel(
    const float* __restrict__ vx,
    const float* __restrict__ vz,
    float* __restrict__ txx,
    float* __restrict__ tzz,
    float* __restrict__ txz,
    const float* __restrict__ rox,
    const float* __restrict__ roz,
    int n1,
    int ioPx, int ioPz, int iePx, int iePz,
    int ioTx, int ioTz, int ieTx, int ieTz)
{
    const float c1 = d_fd_coeff[0], c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2], c4 = d_fd_coeff[3];

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    /* --- txx += D+x(rox*vx),  tzz += D+z(roz*vz) --- */
    {
        int gx = ix + ioPx;
        int gz = iz + ioPz;
        if (gx < iePx && gz < iePz) {
            int idx = gx * n1 + gz;
            float txx_val =
                c1*(rox[(gx+1)*n1+gz]*vx[(gx+1)*n1+gz] - rox[idx]*vx[idx]) +
                c2*(rox[(gx+2)*n1+gz]*vx[(gx+2)*n1+gz] - rox[(gx-1)*n1+gz]*vx[(gx-1)*n1+gz]);
            float tzz_val =
                c1*(roz[gx*n1+gz+1]*vz[gx*n1+gz+1] - roz[idx]*vz[idx]) +
                c2*(roz[gx*n1+gz+2]*vz[gx*n1+gz+2] - roz[gx*n1+gz-1]*vz[gx*n1+gz-1]);
            if (ORDER >= 6) {
                txx_val += c3*(rox[(gx+3)*n1+gz]*vx[(gx+3)*n1+gz] - rox[(gx-2)*n1+gz]*vx[(gx-2)*n1+gz]);
                tzz_val += c3*(roz[gx*n1+gz+3]*vz[gx*n1+gz+3]   - roz[gx*n1+gz-2]*vz[gx*n1+gz-2]);
            }
            if (ORDER >= 8) {
                txx_val += c4*(rox[(gx+4)*n1+gz]*vx[(gx+4)*n1+gz] - rox[(gx-3)*n1+gz]*vx[(gx-3)*n1+gz]);
                tzz_val += c4*(roz[gx*n1+gz+4]*vz[gx*n1+gz+4]   - roz[gx*n1+gz-3]*vz[gx*n1+gz-3]);
            }
            txx[idx] += txx_val;
            tzz[idx] += tzz_val;
        }
    }

    /* --- txz += D-z(rox*vx) + D-x(roz*vz) --- */
    {
        int gx = ix + ioTx;
        int gz = iz + ioTz;
        if (gx < ieTx && gz < ieTz) {
            int idx = gx * n1 + gz;
            float val =
                c1*(rox[idx]*vx[idx]             - rox[gx*n1+gz-1]*vx[gx*n1+gz-1] +
                    roz[idx]*vz[idx]             - roz[(gx-1)*n1+gz]*vz[(gx-1)*n1+gz]) +
                c2*(rox[gx*n1+gz+1]*vx[gx*n1+gz+1] - rox[gx*n1+gz-2]*vx[gx*n1+gz-2] +
                    roz[(gx+1)*n1+gz]*vz[(gx+1)*n1+gz] - roz[(gx-2)*n1+gz]*vz[(gx-2)*n1+gz]);
            if (ORDER >= 6) val +=
                c3*(rox[gx*n1+gz+2]*vx[gx*n1+gz+2] - rox[gx*n1+gz-3]*vx[gx*n1+gz-3] +
                    roz[(gx+2)*n1+gz]*vz[(gx+2)*n1+gz] - roz[(gx-3)*n1+gz]*vz[(gx-3)*n1+gz]);
            if (ORDER >= 8) val +=
                c4*(rox[gx*n1+gz+3]*vx[gx*n1+gz+3] - rox[gx*n1+gz-4]*vx[gx*n1+gz-4] +
                    roz[(gx+3)*n1+gz]*vz[(gx+3)*n1+gz] - roz[(gx-4)*n1+gz]*vz[(gx-4)*n1+gz]);
            txz[idx] += val;
        }
    }
}

/*====================================================================
 * 3. FREE SURFACE TXZ CORRECTION — template<int ORDER>
 *
 * Supplement to adjoint stress update Step 1^T.
 * Phase F1 vx at iz=surface reads txz[surface] through D+z, but
 * Phase A2 txz starts at ioTz=surface+1 and misses these transpose
 * contributions. This kernel fills in the missing terms.
 *====================================================================*/

template<int ORDER>
__global__ void adj_free_surface_txz_kernel(
    float* __restrict__ txz,
    const float* __restrict__ rox,
    const float* __restrict__ vx,
    int n1, int izs,   /* surface z-index */
    int ioTx, int ieTx)
{
    const float c1 = d_fd_coeff[0], c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2], c4 = d_fd_coeff[3];

    int gx = blockIdx.x * blockDim.x + threadIdx.x + ioTx;
    if (gx >= ieTx) return;

    /* izs row */
    txz[gx*n1+izs] +=
        c1*rox[gx*n1+izs]*vx[gx*n1+izs] +
        c2*rox[gx*n1+izs+1]*vx[gx*n1+izs+1];
    /* izs-1 row */
    txz[gx*n1+izs-1] +=
        c2*rox[gx*n1+izs]*vx[gx*n1+izs];

    if (ORDER >= 6) {
        txz[gx*n1+izs]   += c3*rox[gx*n1+izs+2]*vx[gx*n1+izs+2];
        txz[gx*n1+izs-1] += c3*rox[gx*n1+izs+1]*vx[gx*n1+izs+1];
        txz[gx*n1+izs-2] += c3*rox[gx*n1+izs]*vx[gx*n1+izs];
    }
    if (ORDER >= 8) {
        txz[gx*n1+izs]   += c4*rox[gx*n1+izs+3]*vx[gx*n1+izs+3];
        txz[gx*n1+izs-1] += c4*rox[gx*n1+izs+2]*vx[gx*n1+izs+2];
        txz[gx*n1+izs-2] += c4*rox[gx*n1+izs+1]*vx[gx*n1+izs+1];
        txz[gx*n1+izs-3] += c4*rox[gx*n1+izs]*vx[gx*n1+izs];
    }
}

/*====================================================================
 * 4. GRADIENT CROSS-CORRELATION KERNEL — template<int ORDER>
 *
 * Accumulates per-timestep gradient contributions:
 *   g_λ  += dt*(ψ_txx + ψ_tzz)*(D+x(vx_fwd) + D+z(vz_fwd))
 *   g_μ  += dt*2*(ψ_txx*dvxdx + ψ_tzz*dvzdz)       [P grid, normal]
 *   g_μ  += dt*ψ_txz*(dvxdz + dvzdx)                 [Txz grid, shear → scattered to P]
 *   g_ρ  += (1/ρ)*ψ_v*(v_fwd - v_fwd_prev)           [scatter from V grids to P]
 *
 * For GPU: lambda + mu normal-stress part in one kernel (P grid).
 * Shear and density are separate kernels due to different grid locations.
 *====================================================================*/

template<int ORDER>
__global__ void gradient_lambda_mu_P_kernel(
    const float* __restrict__ fwd_vx,
    const float* __restrict__ fwd_vz,
    const float* __restrict__ adj_txx,
    const float* __restrict__ adj_tzz,
    float* __restrict__ grad_lam,
    float* __restrict__ grad_muu,
    float dt, int n1,
    int ibPx, int ibPz, int iePx, int iePz)
{
    const float c1 = d_fd_coeff[0], c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2], c4 = d_fd_coeff[3];

    int ix = blockIdx.x * blockDim.x + threadIdx.x + ibPx;
    int iz = blockIdx.y * blockDim.y + threadIdx.y + ibPz;
    if (ix >= iePx || iz >= iePz) return;

    int idx = ix * n1 + iz;

    /* D+x(vx) and D+z(vz) at P grid — same stencil as forward stress */
    float dvxdx = c1*(fwd_vx[(ix+1)*n1+iz] - fwd_vx[idx]) +
                  c2*(fwd_vx[(ix+2)*n1+iz] - fwd_vx[(ix-1)*n1+iz]);
    float dvzdz = c1*(fwd_vz[ix*n1+iz+1]   - fwd_vz[idx]) +
                  c2*(fwd_vz[ix*n1+iz+2]   - fwd_vz[ix*n1+iz-1]);
    if (ORDER >= 6) {
        dvxdx += c3*(fwd_vx[(ix+3)*n1+iz] - fwd_vx[(ix-2)*n1+iz]);
        dvzdz += c3*(fwd_vz[ix*n1+iz+3]   - fwd_vz[ix*n1+iz-2]);
    }
    if (ORDER >= 8) {
        dvxdx += c4*(fwd_vx[(ix+4)*n1+iz] - fwd_vx[(ix-3)*n1+iz]);
        dvzdz += c4*(fwd_vz[ix*n1+iz+4]   - fwd_vz[ix*n1+iz-3]);
    }

    float psi_sum = adj_txx[idx] + adj_tzz[idx];
    float div_f = dvxdx + dvzdz;

    /* Lambda gradient */
    if (grad_lam) grad_lam[idx] += dt * psi_sum * div_f;

    /* Mu gradient (normal-stress part, factor 2 from ∂l2m/∂μ = 2) */
    if (grad_muu) grad_muu[idx] += dt * 2.0f * (adj_txx[idx]*dvxdx + adj_tzz[idx]*dvzdz);
}

/* Mu gradient shear part at Txz grid, scattered to 4 surrounding P points */
template<int ORDER>
__global__ void gradient_mu_shear_kernel(
    const float* __restrict__ fwd_vx,
    const float* __restrict__ fwd_vz,
    const float* __restrict__ adj_txz,
    float* __restrict__ grad_muu,
    float dt, int n1,
    int ibTx, int ibTz, int ieTx, int ieTz)
{
    const float c1 = d_fd_coeff[0], c2 = d_fd_coeff[1];
    const float c3 = d_fd_coeff[2], c4 = d_fd_coeff[3];

    int ix = blockIdx.x * blockDim.x + threadIdx.x + ibTx;
    int iz = blockIdx.y * blockDim.y + threadIdx.y + ibTz;
    if (ix >= ieTx || iz >= ieTz) return;

    int idx = ix * n1 + iz;

    /* D-z(vx) and D-x(vz) at Txz grid */
    float dvxdz = c1*(fwd_vx[idx]   - fwd_vx[ix*n1+iz-1]) +
                  c2*(fwd_vx[ix*n1+iz+1] - fwd_vx[ix*n1+iz-2]);
    float dvzdx = c1*(fwd_vz[idx]   - fwd_vz[(ix-1)*n1+iz]) +
                  c2*(fwd_vz[(ix+1)*n1+iz] - fwd_vz[(ix-2)*n1+iz]);
    if (ORDER >= 6) {
        dvxdz += c3*(fwd_vx[ix*n1+iz+2] - fwd_vx[ix*n1+iz-3]);
        dvzdx += c3*(fwd_vz[(ix+2)*n1+iz] - fwd_vz[(ix-3)*n1+iz]);
    }
    if (ORDER >= 8) {
        dvxdz += c4*(fwd_vx[ix*n1+iz+3] - fwd_vx[ix*n1+iz-4]);
        dvzdx += c4*(fwd_vz[(ix+3)*n1+iz] - fwd_vz[(ix-4)*n1+iz]);
    }

    float shear = dt * adj_txz[idx] * (dvxdz + dvzdx);

    /* Scatter to 4 surrounding P-grid points (atomicAdd for thread safety) */
    atomicAdd(&grad_muu[(ix-1)*n1+iz-1], 0.25f * shear);
    atomicAdd(&grad_muu[ix*n1+iz-1],     0.25f * shear);
    atomicAdd(&grad_muu[(ix-1)*n1+iz],   0.25f * shear);
    atomicAdd(&grad_muu[idx],            0.25f * shear);
}

/* Density gradient: ψ_v · ∂v/∂t, scattered from V grids to P grid */
__global__ void gradient_rho_kernel(
    const float* __restrict__ fwd_vx,
    const float* __restrict__ fwd_vx_prev,
    const float* __restrict__ fwd_vz,
    const float* __restrict__ fwd_vz_prev,
    const float* __restrict__ adj_vx,
    const float* __restrict__ adj_vz,
    const float* __restrict__ rho,
    float* __restrict__ grad_rho,
    float dt, int n1,
    int ibVx_x, int ibVx_z, int ieVx_x, int ieVx_z,
    int ibVz_x, int ibVz_z, int ieVz_x, int ieVz_z)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    float sdt = 1.0f / dt;

    /* Vx contribution */
    {
        int gx = ix + ibVx_x;
        int gz = iz + ibVx_z;
        if (gx < ieVx_x && gz < ieVx_z) {
            int idx = gx * n1 + gz;
            float dvx_dt = (fwd_vx[idx] - fwd_vx_prev[idx]) * sdt;
            float vx_contrib = dt * adj_vx[idx] * dvx_dt;
            /* Scatter to 2 P-grid neighbors in x */
            atomicAdd(&grad_rho[(gx-1)*n1+gz], 0.5f * vx_contrib / rho[(gx-1)*n1+gz]);
            atomicAdd(&grad_rho[idx],          0.5f * vx_contrib / rho[idx]);
        }
    }

    /* Vz contribution */
    {
        int gx = ix + ibVz_x;
        int gz = iz + ibVz_z;
        if (gx < ieVz_x && gz < ieVz_z) {
            int idx = gx * n1 + gz;
            float dvz_dt = (fwd_vz[idx] - fwd_vz_prev[idx]) * sdt;
            float vz_contrib = dt * adj_vz[idx] * dvz_dt;
            /* Scatter to 2 P-grid neighbors in z */
            atomicAdd(&grad_rho[idx-1], 0.5f * vz_contrib / rho[idx-1]);
            atomicAdd(&grad_rho[idx],   0.5f * vz_contrib / rho[idx]);
        }
    }
}

/*====================================================================
 * 5. ADJOINT SOURCE INJECTION KERNEL
 *
 * Injects residual traces into the adjoint wavefield.
 * Each thread handles one adjoint source (receiver position).
 * Phase filtering done on host — kernel receives pre-filtered list.
 *
 * adj_wav: [nsrc × nt], column-major (isrc fast)
 * adj_xi/zi: grid positions (padded coords)
 * adj_typ: source type (1=P, 2=Txz, 3=Tzz, 4=Txx, 6=Fx, 7=Fz)
 *====================================================================*/

__global__ void inject_adjoint_source_kernel(
    float* __restrict__ vx,
    float* __restrict__ vz,
    float* __restrict__ txx,
    float* __restrict__ tzz,
    float* __restrict__ txz,
    const float* __restrict__ adj_wav,
    const int* __restrict__ adj_xi,
    const int* __restrict__ adj_zi,
    const int* __restrict__ adj_typ,
    int nsrc, int isam, int nt, int n1,
    int phase)   /* 1=force injection, 2=stress injection */
{
    int isrc = blockIdx.x * blockDim.x + threadIdx.x;
    if (isrc >= nsrc) return;

    int stype = adj_typ[isrc];

    /* Phase filtering */
    if (phase == 1 && stype <= 5) return;
    if (phase == 2 && stype > 5)  return;

    float ampl = adj_wav[isrc * nt + isam];
    if (ampl == 0.0f) return;

    int ix = adj_xi[isrc];
    int iz = adj_zi[isrc];
    int idx = ix * n1 + iz;

    if (stype == 6)       vx[idx] += ampl;
    else if (stype == 7)  vz[idx] += ampl;
    else if (stype == 1) { txx[idx] += 0.5f * ampl; tzz[idx] += 0.5f * ampl; }
    else if (stype == 3)  tzz[idx] += ampl;
    else if (stype == 4)  txx[idx] += ampl;
    else if (stype == 2)  txz[idx] += ampl;
}

/*====================================================================
 * 6. CHECKPOINT SAVE/LOAD (ASYNC)
 *
 * Saves/loads 5 wavefield arrays (vx, vz, txx, tzz, txz) to/from
 * pinned host memory using async cudaMemcpy on stream_io.
 *====================================================================*/

typedef struct _checkpointBuf {
    float *h_buf;       /* pinned host buffer: [nsnap × 5 × nax × naz] */
    int    nsnap;
    int    nax, naz;
    size_t snap_bytes;  /* 5 × nax × naz × sizeof(float) */
} checkpointBuf;

#ifdef __cplusplus
extern "C" {
#endif

void cuda_checkpoint_create(checkpointBuf *chk, int nsnap, int nax, int naz)
{
    chk->nsnap = nsnap;
    chk->nax = nax;
    chk->naz = naz;
    chk->snap_bytes = (size_t)5 * nax * naz * sizeof(float);
    size_t total = (size_t)nsnap * chk->snap_bytes;

    CUDA_CHECK(cudaMallocHost(&chk->h_buf, total));
    memset(chk->h_buf, 0, total);
}

void cuda_checkpoint_destroy(checkpointBuf *chk)
{
    if (chk->h_buf) CUDA_CHECK(cudaFreeHost(chk->h_buf));
    chk->h_buf = NULL;
}

void cuda_checkpoint_save(checkpointBuf *chk, int isnap,
                           const float *d_vx, const float *d_vz,
                           const float *d_txx, const float *d_tzz,
                           const float *d_txz,
                           cudaStream_t stream)
{
    size_t fld = (size_t)chk->nax * chk->naz;
    size_t fld_bytes = fld * sizeof(float);
    float *dst = chk->h_buf + (size_t)isnap * 5 * fld;

    CUDA_CHECK(cudaMemcpyAsync(dst + 0*fld, d_vx,  fld_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst + 1*fld, d_vz,  fld_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst + 2*fld, d_txx, fld_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst + 3*fld, d_tzz, fld_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst + 4*fld, d_txz, fld_bytes, cudaMemcpyDeviceToHost, stream));
}

void cuda_checkpoint_load(checkpointBuf *chk, int isnap,
                           float *d_vx, float *d_vz,
                           float *d_txx, float *d_tzz,
                           float *d_txz,
                           cudaStream_t stream)
{
    size_t fld = (size_t)chk->nax * chk->naz;
    size_t fld_bytes = fld * sizeof(float);
    float *src = chk->h_buf + (size_t)isnap * 5 * fld;

    CUDA_CHECK(cudaMemcpyAsync(d_vx,  src + 0*fld, fld_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vz,  src + 1*fld, fld_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_txx, src + 2*fld, fld_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tzz, src + 3*fld, fld_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_txz, src + 4*fld, fld_bytes, cudaMemcpyHostToDevice, stream));
}

/*====================================================================
 * 7. LAUNCH WRAPPERS (extern "C")
 *====================================================================*/

void cuda_launch_adj_velocity_update(
    float *d_vx, float *d_vz,
    const float *d_txx, const float *d_tzz, const float *d_txz,
    const float *d_l2m, const float *d_lam, const float *d_mul,
    int n1, int iorder,
    int ioXx, int ioXz, int ieXx, int ieXz,
    int ioZx, int ioZz, int ieZx, int ieZz,
    cudaStream_t stream)
{
    int nx_max = ieXx - ioXx;
    int nz_max = ieXz - ioXz;
    if ((ieZx - ioZx) > nx_max) nx_max = ieZx - ioZx;
    if ((ieZz - ioZz) > nz_max) nz_max = ieZz - ioZz;

    dim3 block(BLOCK_X, BLOCK_Z);
    dim3 grid(cuda_div_ceil(nx_max, BLOCK_X), cuda_div_ceil(nz_max, BLOCK_Z));

    DISPATCH_ORDER(iorder,
        (adj_update_velocity_kernel<4><<<grid, block, 0, stream>>>(d_vx, d_vz, d_txx, d_tzz, d_txz, d_l2m, d_lam, d_mul, n1, ioXx, ioXz, ieXx, ieXz, ioZx, ioZz, ieZx, ieZz)),
        (adj_update_velocity_kernel<6><<<grid, block, 0, stream>>>(d_vx, d_vz, d_txx, d_tzz, d_txz, d_l2m, d_lam, d_mul, n1, ioXx, ioXz, ieXx, ieXz, ioZx, ioZz, ieZx, ieZz)),
        (adj_update_velocity_kernel<8><<<grid, block, 0, stream>>>(d_vx, d_vz, d_txx, d_tzz, d_txz, d_l2m, d_lam, d_mul, n1, ioXx, ioXz, ieXx, ieXz, ioZx, ioZz, ieZx, ieZz))
    );
}

void cuda_launch_adj_stress_update(
    const float *d_vx, const float *d_vz,
    float *d_txx, float *d_tzz, float *d_txz,
    const float *d_rox, const float *d_roz,
    int n1, int iorder,
    int ioPx, int ioPz, int iePx, int iePz,
    int ioTx, int ioTz, int ieTx, int ieTz,
    cudaStream_t stream)
{
    int nx_max = iePx - ioPx;
    int nz_max = iePz - ioPz;
    if ((ieTx - ioTx) > nx_max) nx_max = ieTx - ioTx;
    if ((ieTz - ioTz) > nz_max) nz_max = ieTz - ioTz;

    dim3 block(BLOCK_X, BLOCK_Z);
    dim3 grid(cuda_div_ceil(nx_max, BLOCK_X), cuda_div_ceil(nz_max, BLOCK_Z));

    DISPATCH_ORDER(iorder,
        (adj_update_stress_kernel<4><<<grid, block, 0, stream>>>(d_vx, d_vz, d_txx, d_tzz, d_txz, d_rox, d_roz, n1, ioPx, ioPz, iePx, iePz, ioTx, ioTz, ieTx, ieTz)),
        (adj_update_stress_kernel<6><<<grid, block, 0, stream>>>(d_vx, d_vz, d_txx, d_tzz, d_txz, d_rox, d_roz, n1, ioPx, ioPz, iePx, iePz, ioTx, ioTz, ieTx, ieTz)),
        (adj_update_stress_kernel<8><<<grid, block, 0, stream>>>(d_vx, d_vz, d_txx, d_tzz, d_txz, d_rox, d_roz, n1, ioPx, ioPz, iePx, iePz, ioTx, ioTz, ieTx, ieTz))
    );
}

void cuda_launch_adj_free_surface_txz(
    float *d_txz, const float *d_rox, const float *d_vx,
    int n1, int iorder, int izs, int ioTx, int ieTx,
    cudaStream_t stream)
{
    int nx = ieTx - ioTx;
    dim3 block(256);
    dim3 grid(cuda_div_ceil(nx, 256));

    DISPATCH_ORDER(iorder,
        (adj_free_surface_txz_kernel<4><<<grid, block, 0, stream>>>(d_txz, d_rox, d_vx, n1, izs, ioTx, ieTx)),
        (adj_free_surface_txz_kernel<6><<<grid, block, 0, stream>>>(d_txz, d_rox, d_vx, n1, izs, ioTx, ieTx)),
        (adj_free_surface_txz_kernel<8><<<grid, block, 0, stream>>>(d_txz, d_rox, d_vx, n1, izs, ioTx, ieTx))
    );
}

void cuda_launch_gradient_lambda_mu_P(
    const float *d_fwd_vx, const float *d_fwd_vz,
    const float *d_adj_txx, const float *d_adj_tzz,
    float *d_grad_lam, float *d_grad_muu,
    float dt, int n1, int iorder,
    int ibPx, int ibPz, int iePx, int iePz,
    cudaStream_t stream)
{
    int nx = iePx - ibPx;
    int nz = iePz - ibPz;
    dim3 block(BLOCK_X, BLOCK_Z);
    dim3 grid(cuda_div_ceil(nx, BLOCK_X), cuda_div_ceil(nz, BLOCK_Z));

    DISPATCH_ORDER(iorder,
        (gradient_lambda_mu_P_kernel<4><<<grid, block, 0, stream>>>(d_fwd_vx, d_fwd_vz, d_adj_txx, d_adj_tzz, d_grad_lam, d_grad_muu, dt, n1, ibPx, ibPz, iePx, iePz)),
        (gradient_lambda_mu_P_kernel<6><<<grid, block, 0, stream>>>(d_fwd_vx, d_fwd_vz, d_adj_txx, d_adj_tzz, d_grad_lam, d_grad_muu, dt, n1, ibPx, ibPz, iePx, iePz)),
        (gradient_lambda_mu_P_kernel<8><<<grid, block, 0, stream>>>(d_fwd_vx, d_fwd_vz, d_adj_txx, d_adj_tzz, d_grad_lam, d_grad_muu, dt, n1, ibPx, ibPz, iePx, iePz))
    );
}

void cuda_launch_gradient_mu_shear(
    const float *d_fwd_vx, const float *d_fwd_vz,
    const float *d_adj_txz, float *d_grad_muu,
    float dt, int n1, int iorder,
    int ibTx, int ibTz, int ieTx, int ieTz,
    cudaStream_t stream)
{
    int nx = ieTx - ibTx;
    int nz = ieTz - ibTz;
    dim3 block(BLOCK_X, BLOCK_Z);
    dim3 grid(cuda_div_ceil(nx, BLOCK_X), cuda_div_ceil(nz, BLOCK_Z));

    DISPATCH_ORDER(iorder,
        (gradient_mu_shear_kernel<4><<<grid, block, 0, stream>>>(d_fwd_vx, d_fwd_vz, d_adj_txz, d_grad_muu, dt, n1, ibTx, ibTz, ieTx, ieTz)),
        (gradient_mu_shear_kernel<6><<<grid, block, 0, stream>>>(d_fwd_vx, d_fwd_vz, d_adj_txz, d_grad_muu, dt, n1, ibTx, ibTz, ieTx, ieTz)),
        (gradient_mu_shear_kernel<8><<<grid, block, 0, stream>>>(d_fwd_vx, d_fwd_vz, d_adj_txz, d_grad_muu, dt, n1, ibTx, ibTz, ieTx, ieTz))
    );
}

void cuda_launch_gradient_rho(
    const float *d_fwd_vx, const float *d_fwd_vx_prev,
    const float *d_fwd_vz, const float *d_fwd_vz_prev,
    const float *d_adj_vx, const float *d_adj_vz,
    const float *d_rho, float *d_grad_rho,
    float dt, int n1,
    int ibVx_x, int ibVx_z, int ieVx_x, int ieVx_z,
    int ibVz_x, int ibVz_z, int ieVz_x, int ieVz_z,
    cudaStream_t stream)
{
    int nx_max = ieVx_x - ibVx_x;
    int nz_max = ieVx_z - ibVx_z;
    if ((ieVz_x - ibVz_x) > nx_max) nx_max = ieVz_x - ibVz_x;
    if ((ieVz_z - ibVz_z) > nz_max) nz_max = ieVz_z - ibVz_z;

    dim3 block(BLOCK_X, BLOCK_Z);
    dim3 grid(cuda_div_ceil(nx_max, BLOCK_X), cuda_div_ceil(nz_max, BLOCK_Z));

    gradient_rho_kernel<<<grid, block, 0, stream>>>(
        d_fwd_vx, d_fwd_vx_prev, d_fwd_vz, d_fwd_vz_prev,
        d_adj_vx, d_adj_vz, d_rho, d_grad_rho, dt, n1,
        ibVx_x, ibVx_z, ieVx_x, ieVx_z,
        ibVz_x, ibVz_z, ieVz_x, ieVz_z);
}

void cuda_launch_inject_adjoint_source(
    float *d_vx, float *d_vz,
    float *d_txx, float *d_tzz, float *d_txz,
    const float *d_adj_wav,
    const int *d_adj_xi, const int *d_adj_zi, const int *d_adj_typ,
    int nsrc, int isam, int nt, int n1, int phase,
    cudaStream_t stream)
{
    if (nsrc <= 0) return;
    dim3 block(256);
    dim3 grid(cuda_div_ceil(nsrc, 256));
    inject_adjoint_source_kernel<<<grid, block, 0, stream>>>(
        d_vx, d_vz, d_txx, d_tzz, d_txz,
        d_adj_wav, d_adj_xi, d_adj_zi, d_adj_typ,
        nsrc, isam, nt, n1, phase);
}

#ifdef __cplusplus
}
#endif

/* Explicit template instantiations */
template __global__ void adj_update_velocity_kernel<4>(float*, float*, const float*, const float*, const float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void adj_update_velocity_kernel<6>(float*, float*, const float*, const float*, const float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void adj_update_velocity_kernel<8>(float*, float*, const float*, const float*, const float*, const float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void adj_update_stress_kernel<4>(const float*, const float*, float*, float*, float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void adj_update_stress_kernel<6>(const float*, const float*, float*, float*, float*, const float*, const float*, int, int, int, int, int, int, int, int, int);
template __global__ void adj_update_stress_kernel<8>(const float*, const float*, float*, float*, float*, const float*, const float*, int, int, int, int, int, int, int, int, int);

#endif /* USE_CUDA */
