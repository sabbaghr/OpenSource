/*********************************************************************
 *
 * cuda_utils.h — CUDA foundation utilities for fdelfwi GPU port
 *
 * Provides:
 *   - CUDA_CHECK error-checking macro
 *   - Stream management (create/destroy for compute, halo, border, I/O)
 *   - Device query and GPU binding for MPI ranks
 *   - Constant memory declarations for FD coefficients
 *   - Timer utilities via cudaEvents
 *
 * Usage:
 *   #include "cuda_utils.h" in any .cu or .c file (guarded by USE_CUDA)
 *
 * Author: FWI CUDA porting project
 *
 *********************************************************************/

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/*====================================================================
 * 1. ERROR-CHECKING MACRO
 *====================================================================*/

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/*====================================================================
 * 2. CONSTANT MEMORY — FD stencil coefficients
 *
 * Broadcast to all threads at zero cost (cached in constant cache).
 * Supports up to 8th order (4 coefficients).
 *====================================================================*/

/* Each .cu file that uses FD coefficients must define its own:
 *   __constant__ float d_fd_coeff[4];
 * CUDA __constant__ variables are per-translation-unit.
 * Use cuda_set_fd_coefficients() from fdelmodc_cuda.cu to upload values. */

/*====================================================================
 * 3. STREAM MANAGEMENT
 *
 * Four streams per GPU for overlapping:
 *   stream_compute  — main stencil kernels (interior)
 *   stream_border   — border stencil kernels (after halo exchange)
 *   stream_halo     — halo pack/unpack kernels
 *   stream_io       — checkpoint D2H / H2D transfers
 *====================================================================*/

typedef struct _cudaStreamSet {
    cudaStream_t compute;
    cudaStream_t border;
    cudaStream_t halo;
    cudaStream_t io;
} cudaStreamSet;

static inline void cuda_streams_create(cudaStreamSet *s)
{
    CUDA_CHECK(cudaStreamCreate(&s->compute));
    CUDA_CHECK(cudaStreamCreate(&s->border));
    CUDA_CHECK(cudaStreamCreate(&s->halo));
    CUDA_CHECK(cudaStreamCreate(&s->io));
}

static inline void cuda_streams_destroy(cudaStreamSet *s)
{
    CUDA_CHECK(cudaStreamDestroy(s->compute));
    CUDA_CHECK(cudaStreamDestroy(s->border));
    CUDA_CHECK(cudaStreamDestroy(s->halo));
    CUDA_CHECK(cudaStreamDestroy(s->io));
}

/*====================================================================
 * 4. TIMER UTILITIES
 *====================================================================*/

typedef struct _cudaTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
} cudaTimer;

static inline void cuda_timer_create(cudaTimer *t)
{
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
}

static inline void cuda_timer_destroy(cudaTimer *t)
{
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
}

static inline void cuda_timer_start(cudaTimer *t, cudaStream_t stream)
{
    CUDA_CHECK(cudaEventRecord(t->start, stream));
}

static inline void cuda_timer_stop(cudaTimer *t, cudaStream_t stream)
{
    CUDA_CHECK(cudaEventRecord(t->stop, stream));
}

/* Returns elapsed time in milliseconds. Blocks until stop event completes. */
static inline float cuda_timer_elapsed_ms(cudaTimer *t)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms;
}

/*====================================================================
 * 5. GPU BINDING FOR MPI
 *
 * Bind each MPI rank to a unique GPU on the same node.
 * Assumes ranks on the same node have consecutive local IDs.
 *====================================================================*/

static inline int cuda_bind_gpu(int local_rank)
{
    int ndevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    if (ndevices == 0) {
        fprintf(stderr, "cuda_bind_gpu: no CUDA devices found\n");
        exit(EXIT_FAILURE);
    }
    int dev = local_rank % ndevices;
    CUDA_CHECK(cudaSetDevice(dev));
    return dev;
}

/*====================================================================
 * 6. DEVICE QUERY (verbose, for diagnostics)
 *====================================================================*/

static inline void cuda_print_device_info(int dev, int verbose)
{
    if (verbose < 1) return;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "GPU %d: %s, SM %d.%d, %.1f GB VRAM, %d SMs, "
            "smem %zu KB, const %zu KB\n",
            dev, prop.name, prop.major, prop.minor,
            prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
            prop.multiProcessorCount,
            prop.sharedMemPerBlock / 1024,
            prop.totalConstMem / 1024);
}

/*====================================================================
 * 7. KERNEL LAUNCH HELPERS
 *====================================================================*/

/* Standard block size for 2D stencil kernels */
#define BLOCK_X 16
#define BLOCK_Z 16

/* Halo width per FD order */
static inline int cuda_halo_width(int iorder)
{
    return iorder / 2;  /* 4th→2, 6th→3, 8th→4 */
}

/* Ceiling division */
static inline int cuda_div_ceil(int a, int b)
{
    return (a + b - 1) / b;
}

/*====================================================================
 * 8. PINNED HOST MEMORY HELPERS
 *====================================================================*/

static inline void cuda_alloc_pinned(void **ptr, size_t bytes)
{
    CUDA_CHECK(cudaMallocHost(ptr, bytes));
}

static inline void cuda_free_pinned(void *ptr)
{
    if (ptr) CUDA_CHECK(cudaFreeHost(ptr));
}

/*====================================================================
 * 9. Z-PADDING FOR COALESCED ACCESS
 *
 * Pad naz to next multiple of 32 floats (128 bytes) so that each
 * x-column starts at a 128-byte aligned address.
 *====================================================================*/

static inline int cuda_pad_naz(int naz)
{
    return ((naz + 31) / 32) * 32;
}

#endif /* USE_CUDA */
#endif /* CUDA_UTILS_H */
