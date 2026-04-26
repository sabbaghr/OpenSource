/*********************************************************************
 *
 * fdelfwi_gpu.h — GPU FWI driver interface
 *
 * Declares functions for GPU-accelerated Full Waveform Inversion:
 *   - One-time GPU initialization (model upload, stream setup)
 *   - GPU forward modeling with checkpointing (replaces fdfwimodc)
 *   - GPU adjoint backpropagation with gradient (replaces adj_shot)
 *   - GPU model re-upload after optimizer step
 *   - GPU cleanup
 *
 * Supports two modes:
 *   - Single-GPU (ndom=1): full model on one GPU
 *   - Domain-decomposed (ndom>1): 1D X-axis decomposition with halo
 *     exchange via domain_decomp.h/c. Combined with shot-parallel MPI.
 *
 *********************************************************************/

#ifndef FDELFWI_GPU_H
#define FDELFWI_GPU_H

#ifdef USE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

#include "fdelfwi.h"
#include "domain_decomp.h"

/*--------------------------------------------------------------------
 * fdelfwi_gpu_init -- One-time GPU setup.
 *
 * Allocates device arrays for model, wavefields, boundaries, gradients.
 * Uploads the initial model and boundary taper arrays.
 * Creates CUDA streams for overlapping compute and I/O.
 *
 * When dom is non-NULL and dom->ndom > 1, uses LOCAL subdomain
 * dimensions (nax_local × naz) and allocates halo exchange buffers.
 * Otherwise uses the full model (single GPU mode).
 *
 * Must be called once before any GPU forward/adjoint calls.
 *--------------------------------------------------------------------*/
void fdelfwi_gpu_init(modPar *mod, bndPar *bnd, recPar *rec,
                      int nsnap_max, int verbose);

/*--------------------------------------------------------------------
 * fdelfwi_gpu_init_domain -- One-time GPU setup with domain decomp.
 *
 * Like fdelfwi_gpu_init but uses local subdomain dimensions from dom.
 * Allocates halo exchange buffers. Each rank uploads its local slice.
 *--------------------------------------------------------------------*/
void fdelfwi_gpu_init_domain(modPar *mod, bndPar *bnd, recPar *rec,
                             domainPar *dom, int nsnap_max, int verbose);

/*--------------------------------------------------------------------
 * fdelfwi_gpu_upload_model -- Re-upload model after optimizer update.
 *
 * Called after injectModelVector() updates mod->rox/roz/l2m/lam/muu.
 * In domain-decomposed mode, uploads LOCAL subdomain slice.
 *--------------------------------------------------------------------*/
void fdelfwi_gpu_upload_model(modPar *mod);

/*--------------------------------------------------------------------
 * fdelfwi_gpu_upload_model_domain -- Upload local model slice.
 *
 * Extracts the local subdomain from the global model arrays and
 * uploads to the GPU. Called after model scatter.
 *--------------------------------------------------------------------*/
void fdelfwi_gpu_upload_model_domain(modPar *mod, domainPar *dom);

/*--------------------------------------------------------------------
 * fdfwimodc_gpu -- GPU forward modeling with checkpointing.
 *
 * GPU replacement for fdfwimodc(). Single-GPU mode.
 *--------------------------------------------------------------------*/
int fdfwimodc_gpu(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                  recPar *rec, snaPar *sna, int ixsrc, int izsrc,
                  float **src_nwav, int ishot, int nshots, int fileno,
                  checkpointPar *chk, int verbose);

/*--------------------------------------------------------------------
 * fdfwimodc_gpu_domain -- GPU forward modeling with domain decomp.
 *
 * Domain-decomposed version. All ranks in domain_comm collaborate
 * on one shot with halo exchange every time step. Receiver traces
 * are gathered to rank 0 of domain_comm for .su file output.
 *--------------------------------------------------------------------*/
int fdfwimodc_gpu_domain(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                         recPar *rec, snaPar *sna, int ixsrc, int izsrc,
                         float **src_nwav, int ishot, int nshots, int fileno,
                         checkpointPar *chk, domainPar *dom, int verbose);

/*--------------------------------------------------------------------
 * adj_shot_gpu -- GPU adjoint backpropagation with gradient.
 *
 * GPU replacement for adj_shot(). Single-GPU mode.
 *--------------------------------------------------------------------*/
int adj_shot_gpu(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                 recPar *rec, adjSrcPar *adj,
                 int ixsrc, int izsrc, float **src_nwav,
                 checkpointPar *chk, snaPar *sna,
                 float *grad1, float *grad2, float *grad3,
                 int param, int verbose,
                 float *wfld_energy);

/*--------------------------------------------------------------------
 * adj_shot_gpu_domain -- GPU adjoint with domain decomposition.
 *
 * Domain-decomposed version. All ranks in domain_comm collaborate.
 * Gradient arrays are LOCAL (nax_local × naz), accumulated on device,
 * then downloaded and added to host arrays.
 *--------------------------------------------------------------------*/
int adj_shot_gpu_domain(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                        recPar *rec, adjSrcPar *adj,
                        int ixsrc, int izsrc, float **src_nwav,
                        checkpointPar *chk, snaPar *sna,
                        float *grad1, float *grad2, float *grad3,
                        domainPar *dom,
                        int param, int verbose,
                        float *wfld_energy);

/*--------------------------------------------------------------------
 * fdelfwi_gpu_cleanup -- Free all GPU resources.
 *--------------------------------------------------------------------*/
void fdelfwi_gpu_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* USE_CUDA */
#endif /* FDELFWI_GPU_H */
