/*********************************************************************
 *
 * fdelmodc_gpu.h — GPU forward modeling driver for fdelmodc
 *
 * Declares the GPU time-loop function that replaces the CPU elastic4/6/8
 * calls when compiled with USE_CUDA.
 *
 *********************************************************************/

#ifndef FDELMODC_GPU_H
#define FDELMODC_GPU_H

#ifdef USE_CUDA

#include "fdelmodc.h"

/*
 * fdelmodc_gpu_elastic — Run elastic forward modeling on GPU for one shot.
 *
 * Replaces the CPU time loop (elastic4/6/8 + boundaries + source + receivers).
 * Handles: GPU alloc, H2D upload, time loop, D2H download of receiver data.
 *
 * Parameters match what fdelmodc.c passes to the CPU time loop.
 * Model arrays (rox, roz, l2m, lam, mul) are uploaded to GPU once per shot
 * (or kept resident across shots if first_shot flag is set).
 *
 * Returns 0 on success.
 */
int fdelmodc_gpu_elastic(
    modPar mod, srcPar src, wavPar wav, bndPar bnd, recPar rec, snaPar sna,
    int ixsrc, int izsrc, int ishot, int nshots,
    float **src_nwav,
    float *rox, float *roz, float *l2m, float *lam, float *mul,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *rec_vx, float *rec_vz, float *rec_txx, float *rec_tzz,
    float *rec_txz, float *rec_p, float *rec_pp, float *rec_ss,
    int it0, int it1,
    int verbose);

/* Initialize GPU resources (call once before shot loop) */
void fdelmodc_gpu_init(modPar mod, bndPar bnd, recPar rec,
                        float *rox, float *roz, float *l2m,
                        float *lam, float *mul, int verbose);

/* Free GPU resources (call once after shot loop) */
void fdelmodc_gpu_cleanup(void);

#endif /* USE_CUDA */
#endif /* FDELMODC_GPU_H */
