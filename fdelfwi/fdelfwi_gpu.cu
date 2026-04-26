/*********************************************************************
 *
 * fdelfwi_gpu.cu — GPU FWI driver for elastic Full Waveform Inversion
 *
 * Implements GPU-accelerated forward modeling with checkpointing and
 * adjoint backpropagation with gradient accumulation.
 *
 * Architecture:
 *   - Persistent GPU state: model arrays, boundary tapers, CUDA streams
 *   - Forward pass: GPU time loop → checkpoints to pinned host → D2H receivers
 *   - Adjoint pass: load checkpoint from pinned host → GPU re-propagation +
 *     adjoint + gradient cross-correlation → D2H gradient download
 *
 * Supports single-GPU (ndom=1) and domain-decomposed multi-GPU (ndom>1).
 * In domain-decomposed mode, each GPU holds a local X-axis slice with
 * halo exchange every time step via GPU-aware MPI.
 * I/O (writeRec, computeResidual, readResidual) remains on CPU.
 *
 *********************************************************************/

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda_utils.h"

/* C headers for fdelfwi types */
extern "C" {
#include "fdelfwi.h"
#include "domain_decomp.h"
}

#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))

/*====================================================================
 * DEVICE DATA STRUCTURES (shared with fdelmodc_cuda.cu)
 *====================================================================*/

typedef struct _deviceWfl {
    float *vx, *vz;
    float *txx, *tzz, *txz;
    size_t field_size;
    int nax, naz;
} deviceWfl;

typedef struct _deviceMod {
    float *l2m, *lam, *muu;
    float *rox, *roz;
    size_t field_size;
    int nax, naz;
} deviceMod;

typedef struct _deviceBnd {
    float *tapx, *tapz, *tapxz;
    int   *surface;
    int    ntap;
} deviceBnd;

typedef struct _deviceRec {
    float *rec_vx, *rec_vz, *rec_txx, *rec_tzz, *rec_p;
    int   *rec_ix, *rec_iz;
    int    nrec;
    int    nt;
} deviceRec;

typedef struct _checkpointBuf {
    float *h_buf;
    int    nsnap;
    int    nax, naz;
    size_t snap_bytes;
} checkpointBuf;

/* Adjoint source device buffers */
typedef struct _deviceAdjSrc {
    float *wav;     /* [nsrc * nt] adjoint waveforms on device */
    int   *xi;      /* horizontal grid indices */
    int   *zi;      /* vertical grid indices */
    int   *typ;     /* source type per trace */
    int    nsrc;
    int    nt;
} deviceAdjSrc;

/*====================================================================
 * FORWARD DECLARATIONS — kernel launch wrappers from fdelmodc_cuda.cu
 *====================================================================*/

extern "C" {
/* Memory management */
void cuda_alloc_wfl(deviceWfl *wfl, int nax, int naz);
void cuda_free_wfl(deviceWfl *wfl);
void cuda_zero_wfl(deviceWfl *wfl);
void cuda_alloc_mod(deviceMod *mod, int nax, int naz);
void cuda_free_mod(deviceMod *mod);
void cuda_upload_mod(deviceMod *dmod, const float *h_l2m, const float *h_lam,
                      const float *h_muu, const float *h_rox, const float *h_roz);
void cuda_alloc_bnd(deviceBnd *bnd, int ntap, int nax,
                     const float *h_tapx, const float *h_tapz,
                     const float *h_tapxz, const int *h_surface);
void cuda_free_bnd(deviceBnd *bnd);
void cuda_alloc_rec(deviceRec *rec, int nrec, int nt,
                     const int *h_rec_ix, const int *h_rec_iz,
                     int rec_vx_flag, int rec_vz_flag,
                     int rec_txx_flag, int rec_tzz_flag, int rec_p_flag);
void cuda_free_rec(deviceRec *rec);
void cuda_set_fd_coefficients(int iorder);

/* Forward kernels */
void cuda_launch_velocity_update(deviceWfl *wfl, deviceMod *dmod,
    int n1, int iorder,
    int ioXx, int ioXz, int ieXx, int ieXz,
    int ioZx, int ioZz, int ieZx, int ieZz,
    cudaStream_t stream);
void cuda_launch_stress_update(deviceWfl *wfl, deviceMod *dmod,
    int n1, int iorder,
    int ioPx, int ioPz, int iePx, int iePz,
    int ioTx, int ioTz, int ieTx, int ieTz,
    cudaStream_t stream);
void cuda_launch_taper_velocity(deviceWfl *wfl, deviceMod *dmod, deviceBnd *dbnd,
    int n1, int ntap, int iorder,
    int ioXx, int ioXz, int ieXx, int ieXz,
    int ioZx, int ioZz, int ieZx, int ieZz,
    int has_top, int has_bot, int has_left, int has_right,
    cudaStream_t stream);
void cuda_launch_taper_stress(deviceWfl *wfl, deviceMod *dmod, deviceBnd *dbnd,
    int n1, int ntap, int iorder,
    int ioPx, int ioPz, int iePx, int iePz,
    int ioTx, int ioTz, int ieTx, int ieTz,
    int has_top, int has_bot, int has_left, int has_right,
    cudaStream_t stream);
void cuda_launch_free_surface(deviceWfl *wfl, deviceBnd *dbnd,
    int n1, int ioPx, int iePx, cudaStream_t stream);
void cuda_launch_inject_source(deviceWfl *wfl, deviceMod *dmod,
    int ix_src, int iz_src, float src_ampl,
    int src_type, int src_orient, int n1,
    cudaStream_t stream);
void cuda_launch_extract_receivers(deviceWfl *wfl, deviceRec *drec,
    int it, int n1, cudaStream_t stream);

/* Adjoint kernels (from fdelfwi_cuda.cu) */
void cuda_launch_adj_velocity_update(
    float *d_vx, float *d_vz,
    const float *d_txx, const float *d_tzz, const float *d_txz,
    const float *d_l2m, const float *d_lam, const float *d_mul,
    int n1, int iorder,
    int ioXx, int ioXz, int ieXx, int ieXz,
    int ioZx, int ioZz, int ieZx, int ieZz,
    cudaStream_t stream);
void cuda_launch_adj_stress_update(
    const float *d_vx, const float *d_vz,
    float *d_txx, float *d_tzz, float *d_txz,
    const float *d_rox, const float *d_roz,
    int n1, int iorder,
    int ioPx, int ioPz, int iePx, int iePz,
    int ioTx, int ioTz, int ieTx, int ieTz,
    cudaStream_t stream);
void cuda_launch_adj_free_surface_txz(
    float *d_txz, const float *d_rox, const float *d_vx,
    int n1, int iorder, int izs, int ioTx, int ieTx,
    cudaStream_t stream);
void cuda_launch_gradient_lambda_mu_P(
    const float *d_fwd_vx, const float *d_fwd_vz,
    const float *d_adj_txx, const float *d_adj_tzz,
    float *d_grad_lam, float *d_grad_muu,
    float dt, int n1, int iorder,
    int ibPx, int ibPz, int iePx, int iePz,
    cudaStream_t stream);
void cuda_launch_gradient_mu_shear(
    const float *d_fwd_vx, const float *d_fwd_vz,
    const float *d_adj_txz, float *d_grad_muu,
    float dt, int n1, int iorder,
    int ibTx, int ibTz, int ieTx, int ieTz,
    cudaStream_t stream);
void cuda_launch_gradient_rho(
    const float *d_fwd_vx, const float *d_fwd_vx_prev,
    const float *d_fwd_vz, const float *d_fwd_vz_prev,
    const float *d_adj_vx, const float *d_adj_vz,
    const float *d_rho, float *d_grad_rho,
    float dt, int n1,
    int ibVx_x, int ibVx_z, int ieVx_x, int ieVx_z,
    int ibVz_x, int ibVz_z, int ieVz_x, int ieVz_z,
    cudaStream_t stream);
void cuda_launch_inject_adjoint_source(
    float *d_vx, float *d_vz,
    float *d_txx, float *d_tzz, float *d_txz,
    const float *d_adj_wav,
    const int *d_adj_xi, const int *d_adj_zi, const int *d_adj_typ,
    int nsrc, int isam, int nt, int n1, int phase,
    cudaStream_t stream);

/* Checkpoint pinned memory */
void cuda_checkpoint_create(checkpointBuf *chk, int nsnap, int nax, int naz);
void cuda_checkpoint_destroy(checkpointBuf *chk);
void cuda_checkpoint_save(checkpointBuf *chk, int isnap,
                           const float *d_vx, const float *d_vz,
                           const float *d_txx, const float *d_tzz,
                           const float *d_txz,
                           cudaStream_t stream);
void cuda_checkpoint_load(checkpointBuf *chk, int isnap,
                           float *d_vx, float *d_vz,
                           float *d_txx, float *d_tzz,
                           float *d_txz,
                           cudaStream_t stream);

/* CPU I/O functions */
int writeRec(recPar rec, modPar mod, bndPar bnd, wavPar wav,
             int ixsrc, int izsrc, int nsam, int ishot, int nshots, int fileno,
             float *rec_vx, float *rec_vz, float *rec_txx, float *rec_tzz,
             float *rec_txz, float *rec_p, float *rec_pp, float *rec_ss,
             float *rec_q, float *rec_udp, float *rec_udvz,
             float *rec_dxvx, float *rec_dzvz, int verbose);

void vmess(char *fmt, ...);
void verr(char *fmt, ...);

void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho, size_t sizem);

double wallclock_time(void);
}


/*====================================================================
 * STATIC GPU STATE (persistent across shots and iterations)
 *====================================================================*/

static deviceWfl  s_wfl_fwd;     /* Forward wavefield */
static deviceWfl  s_wfl_adj;     /* Adjoint wavefield */
static deviceMod  s_dmod;        /* Material parameters */
static deviceBnd  s_dbnd;        /* Boundary tapers */
static cudaStreamSet s_streams;  /* CUDA streams */

/* Forward buffer for re-propagation (one segment's vx/vz on device) */
static float *s_d_buf_vx = NULL;
static float *s_d_buf_vz = NULL;
static int    s_buf_max_nsteps = 0;

/* Gradient arrays on device */
static float *s_d_grad_lam = NULL;
static float *s_d_grad_muu = NULL;
static float *s_d_grad_rho = NULL;

/* GPU checkpoint buffer (pinned host memory) */
static checkpointBuf s_gpu_chk;

/* Model dimensions (saved at init) */
static int s_nax = 0, s_naz = 0;
static size_t s_field_size = 0;  /* nax * naz */

static int s_gpu_initialized = 0;


/*====================================================================
 * GPU FORWARD TIME-STEP (one step, shared by forward and re-prop)
 *====================================================================*/

static void gpu_forward_one_step(
    modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
    int it, int ixsrc, int izsrc, float **src_nwav,
    deviceWfl *wfl, cudaStream_t stream)
{
    int n1 = mod->naz;
    int iorder = mod->iorder;
    float dt = mod->dt;

    int has_top   = (bnd->top == 4);
    int has_bot   = (bnd->bot == 4);
    int has_left  = (bnd->lef == 4);
    int has_right = (bnd->rig == 4);

    /* --- Phase 1: Velocity update --- */
    cuda_launch_velocity_update(wfl, &s_dmod, n1, iorder,
        mod->ioXx, mod->ioXz, mod->ieXx, mod->ieXz,
        mod->ioZx, mod->ioZz, mod->ieZx, mod->ieZz,
        stream);

    /* --- Phase 2: Force source injection (type > 5) --- */
    if (src->type > 5 && src->type != 20) {
        float time = it * dt + mod->t0;
        int id1 = (int)floor(time / dt);
        int id2 = id1 + 1;
        if (time >= 0.0f && id2 <= wav->nt) {
            float ampl = src_nwav[0][id1]*(id2 - time/dt)
                       + src_nwav[0][id2]*(time/dt - id1);
            if (ampl != 0.0f) {
                int ix_s, iz_s;
                if (src->type == 6) {
                    ix_s = ixsrc + mod->ioXx;
                    iz_s = izsrc + mod->ioXz;
                } else {
                    ix_s = ixsrc + mod->ioZx;
                    iz_s = izsrc + mod->ioZz;
                }
                cuda_launch_inject_source(wfl, &s_dmod,
                    ix_s, iz_s, ampl, src->type, src->orient, n1, stream);
            }
        }
    }

    /* --- Phase 3: Velocity boundary (taper + free surface) --- */
    cuda_launch_taper_velocity(wfl, &s_dmod, &s_dbnd,
        n1, bnd->ntap, iorder,
        mod->ioXx, mod->ioXz, mod->ieXx, mod->ieXz,
        mod->ioZx, mod->ioZz, mod->ieZx, mod->ieZz,
        has_top, has_bot, has_left, has_right, stream);

    if (bnd->top == 1)
        cuda_launch_free_surface(wfl, &s_dbnd, n1, mod->ioPx, mod->iePx, stream);

    /* --- Phase 4: Stress update --- */
    cuda_launch_stress_update(wfl, &s_dmod, n1, iorder,
        mod->ioPx, mod->ioPz, mod->iePx, mod->iePz,
        mod->ioTx, mod->ioTz, mod->ieTx, mod->ieTz,
        stream);

    /* --- Phase 5: Stress source injection (type < 6) --- */
    if (src->type < 6) {
        float time = it * dt + mod->t0;
        int id1 = (int)floor(time / dt);
        int id2 = id1 + 1;
        if (time >= 0.0f && id2 <= wav->nt) {
            float ampl = src_nwav[0][id1]*(id2 - time/dt)
                       + src_nwav[0][id2]*(time/dt - id1);
            if (ampl != 0.0f) {
                int src_ibndx = mod->ioPx;
                int src_ibndz = mod->ioPz;
                if (bnd->lef == 4 || bnd->lef == 2) src_ibndx += bnd->ntap;
                if (bnd->top == 4 || bnd->top == 2) src_ibndz += bnd->ntap;

                int ix_s = ixsrc + src_ibndx;
                int iz_s = izsrc + src_ibndz;

                /* Stress source scaling: (1/dx) * l2m at source */
                float l2m_src;
                CUDA_CHECK(cudaMemcpy(&l2m_src, s_dmod.l2m + ix_s * n1 + iz_s,
                           sizeof(float), cudaMemcpyDeviceToHost));
                ampl *= (1.0f / mod->dx) * l2m_src;

                cuda_launch_inject_source(wfl, &s_dmod,
                    ix_s, iz_s, ampl, src->type, src->orient, n1, stream);
            }
        }
    }

    /* --- Phase 6: Stress boundary (taper) --- */
    cuda_launch_taper_stress(wfl, &s_dmod, &s_dbnd,
        n1, bnd->ntap, iorder,
        mod->ioPx, mod->ioPz, mod->iePx, mod->iePz,
        mod->ioTx, mod->ioTz, mod->ieTx, mod->ieTz,
        has_top, has_bot, has_left, has_right, stream);
}


/*====================================================================
 * GPU ADJOINT TIME-STEP (one step, no source injection — handled separately)
 *====================================================================*/

static void gpu_adjoint_one_step(
    modPar *mod, bndPar *bnd,
    deviceWfl *wfl_adj, cudaStream_t stream)
{
    int n1 = mod->naz;
    int iorder = mod->iorder;

    int has_top   = (bnd->top == 4);
    int has_bot   = (bnd->bot == 4);
    int has_left  = (bnd->lef == 4);
    int has_right = (bnd->rig == 4);

    /* Phase A1: Adjoint velocity update (material inside derivative) */
    cuda_launch_adj_velocity_update(
        wfl_adj->vx, wfl_adj->vz,
        wfl_adj->txx, wfl_adj->tzz, wfl_adj->txz,
        s_dmod.l2m, s_dmod.lam, s_dmod.muu,
        n1, iorder,
        mod->ioXx, mod->ioXz, mod->ieXx, mod->ieXz,
        mod->ioZx, mod->ioZz, mod->ieZx, mod->ieZz,
        stream);

    /* Adjoint velocity taper boundaries */
    cuda_launch_taper_velocity(wfl_adj, &s_dmod, &s_dbnd,
        n1, bnd->ntap, iorder,
        mod->ioXx, mod->ioXz, mod->ieXx, mod->ieXz,
        mod->ioZx, mod->ioZz, mod->ieZx, mod->ieZz,
        has_top, has_bot, has_left, has_right, stream);

    if (bnd->top == 1)
        cuda_launch_free_surface(wfl_adj, &s_dbnd, n1, mod->ioPx, mod->iePx, stream);

    /* Phase A2: Adjoint stress update (buoyancy inside derivative) */
    cuda_launch_adj_stress_update(
        wfl_adj->vx, wfl_adj->vz,
        wfl_adj->txx, wfl_adj->tzz, wfl_adj->txz,
        s_dmod.rox, s_dmod.roz,
        n1, iorder,
        mod->ioPx, mod->ioPz, mod->iePx, mod->iePz,
        mod->ioTx, mod->ioTz, mod->ieTx, mod->ieTz,
        stream);

    /* Adjoint free-surface txz correction */
    if (bnd->top == 1) {
        int ibndz = mod->ioPz;
        if (bnd->top == 4 || bnd->top == 2) ibndz += bnd->ntap;
        cuda_launch_adj_free_surface_txz(
            wfl_adj->txz, s_dmod.rox, wfl_adj->vx,
            n1, iorder, ibndz, mod->ioTx, mod->ieTx, stream);
    }

    /* Adjoint stress taper boundaries */
    cuda_launch_taper_stress(wfl_adj, &s_dmod, &s_dbnd,
        n1, bnd->ntap, iorder,
        mod->ioPx, mod->ioPz, mod->iePx, mod->iePz,
        mod->ioTx, mod->ioTz, mod->ieTx, mod->ieTz,
        has_top, has_bot, has_left, has_right, stream);
}


/*====================================================================
 * PUBLIC API
 *====================================================================*/

extern "C"
void fdelfwi_gpu_init(modPar *mod, bndPar *bnd, recPar *rec,
                      int nsnap_max, int verbose)
{
    if (s_gpu_initialized) return;

    s_nax = mod->nax;
    s_naz = mod->naz;
    s_field_size = (size_t)s_nax * s_naz;

    /* Bind to GPU 0 */
    int dev = cuda_bind_gpu(0);
    if (verbose) cuda_print_device_info(dev, verbose);

    /* Create CUDA streams */
    cuda_streams_create(&s_streams);

    /* Upload FD coefficients */
    cuda_set_fd_coefficients(mod->iorder);

    /* Allocate and upload model */
    cuda_alloc_mod(&s_dmod, s_nax, s_naz);
    cuda_upload_mod(&s_dmod, mod->l2m, mod->lam, mod->muu, mod->rox, mod->roz);

    /* Allocate forward and adjoint wavefields */
    cuda_alloc_wfl(&s_wfl_fwd, s_nax, s_naz);
    cuda_alloc_wfl(&s_wfl_adj, s_nax, s_naz);

    /* Upload boundary taper arrays */
    cuda_alloc_bnd(&s_dbnd, bnd->ntap, s_nax,
                    bnd->tapz, bnd->tapx, bnd->tapxz, bnd->surface);

    /* Allocate gradient arrays on device */
    size_t bytes = s_field_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&s_d_grad_lam, bytes));
    CUDA_CHECK(cudaMalloc(&s_d_grad_muu, bytes));
    CUDA_CHECK(cudaMalloc(&s_d_grad_rho, bytes));

    /* Allocate GPU checkpoint buffer (pinned host memory) */
    if (nsnap_max > 0) {
        cuda_checkpoint_create(&s_gpu_chk, nsnap_max, s_nax, s_naz);
        if (verbose) {
            float mb = (float)nsnap_max * 5 * s_field_size * sizeof(float) / (1024.0*1024.0);
            vmess("GPU checkpoints: %d snapshots, %.1f MB pinned host", nsnap_max, mb);
        }
    }

    s_gpu_initialized = 1;
    if (verbose)
        vmess("fdelfwi GPU initialized: nax=%d naz=%d iorder=%d",
              s_nax, s_naz, mod->iorder);
}


extern "C"
void fdelfwi_gpu_upload_model(modPar *mod)
{
    if (!s_gpu_initialized) return;
    cuda_upload_mod(&s_dmod, mod->l2m, mod->lam, mod->muu, mod->rox, mod->roz);
}


extern "C"
void fdelfwi_gpu_cleanup(void)
{
    if (!s_gpu_initialized) return;

    cuda_free_wfl(&s_wfl_fwd);
    cuda_free_wfl(&s_wfl_adj);
    cuda_free_mod(&s_dmod);
    cuda_free_bnd(&s_dbnd);

    if (s_d_grad_lam) { cudaFree(s_d_grad_lam); s_d_grad_lam = NULL; }
    if (s_d_grad_muu) { cudaFree(s_d_grad_muu); s_d_grad_muu = NULL; }
    if (s_d_grad_rho) { cudaFree(s_d_grad_rho); s_d_grad_rho = NULL; }

    if (s_d_buf_vx) { cudaFree(s_d_buf_vx); s_d_buf_vx = NULL; }
    if (s_d_buf_vz) { cudaFree(s_d_buf_vz); s_d_buf_vz = NULL; }
    s_buf_max_nsteps = 0;

    if (s_gpu_chk.h_buf) cuda_checkpoint_destroy(&s_gpu_chk);

    cuda_streams_destroy(&s_streams);

    s_gpu_initialized = 0;
}


/*====================================================================
 * fdfwimodc_gpu — GPU forward modeling with checkpointing
 *
 * Drop-in replacement for fdfwimodc(). Runs the elastic time loop on
 * GPU, saves checkpoints to pinned host memory, and downloads receiver
 * traces at the end for .su file output via CPU writeRec.
 *====================================================================*/

extern "C"
int fdfwimodc_gpu(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                  recPar *rec, snaPar *sna, int ixsrc, int izsrc,
                  float **src_nwav, int ishot, int nshots, int fileno,
                  checkpointPar *chk, int verbose)
{
    int it;
    int n1 = mod->naz;
    int it0 = 0;
    int it1 = mod->nt + NINT(-mod->t0 / mod->dt);
    size_t rec_size;

    /* --- Allocate device receiver buffers --- */
    int ioPx = mod->ioPx, ioPz = mod->ioPz;
    if (bnd->lef == 4 || bnd->lef == 2) ioPx += bnd->ntap;
    if (bnd->top == 4 || bnd->top == 2) ioPz += bnd->ntap;

    int *h_rec_ix = NULL, *h_rec_iz = NULL;
    if (rec->n > 0) {
        h_rec_ix = (int *)malloc(rec->n * sizeof(int));
        h_rec_iz = (int *)malloc(rec->n * sizeof(int));
        for (int ir = 0; ir < rec->n; ir++) {
            h_rec_ix[ir] = rec->x[ir] + ioPx;
            h_rec_iz[ir] = rec->z[ir] + ioPz;
        }
    }

    deviceRec d_rec;
    memset(&d_rec, 0, sizeof(deviceRec));
    cuda_alloc_rec(&d_rec, rec->n, rec->nt, h_rec_ix, h_rec_iz,
                    rec->type.vx, rec->type.vz,
                    rec->type.txx, rec->type.tzz,
                    1); /* always record stress for hydrophone */

    /* --- Allocate host receiver buffers --- */
    rec_size = (size_t)rec->n * rec->nt;
    float *rec_vx=NULL, *rec_vz=NULL, *rec_txx=NULL, *rec_tzz=NULL, *rec_p=NULL;
    float *rec_txz=NULL;
    if (rec->type.vx)   rec_vx  = (float *)calloc(rec_size, sizeof(float));
    if (rec->type.vz)   rec_vz  = (float *)calloc(rec_size, sizeof(float));
    if (rec->type.txx)  rec_txx = (float *)calloc(rec_size, sizeof(float));
    if (rec->type.tzz)  rec_tzz = (float *)calloc(rec_size, sizeof(float));
    rec_p = (float *)calloc(rec_size, sizeof(float)); /* always for hydrophone */

    /* --- Zero wavefields on GPU for this shot --- */
    cuda_zero_wfl(&s_wfl_fwd);

    /* Reset checkpoint counter */
    int gpu_chk_it = 0;

    /* --- MAIN TIME LOOP ON GPU --- */
    for (it = it0; it < it1; it++) {

        /* One forward time step (velocity + source + boundary + stress) */
        gpu_forward_one_step(mod, src, wav, bnd, it, ixsrc, izsrc,
                             src_nwav, &s_wfl_fwd, s_streams.compute);

        /* --- Receiver extraction --- */
        if (rec->n > 0 && (((it - rec->delay) % rec->skipdt) == 0) &&
            (it >= rec->delay)) {
            int itwritten = 0;
            int isam = (it - rec->delay - itwritten + NINT(mod->t0/mod->dt))
                       / rec->skipdt + 1;
            if (isam < 0) isam = rec->nt + isam;

            cuda_launch_extract_receivers(&s_wfl_fwd, &d_rec,
                isam, n1, s_streams.compute);
        }

        /* --- GPU checkpoint (async D2H to pinned host) --- */
        if (chk && chk->nsnap > 0) {
            if (((it - chk->delay) % chk->skipdt == 0) &&
                (it >= chk->delay) &&
                (gpu_chk_it < s_gpu_chk.nsnap)) {
                cuda_checkpoint_save(&s_gpu_chk, gpu_chk_it,
                    s_wfl_fwd.vx, s_wfl_fwd.vz,
                    s_wfl_fwd.txx, s_wfl_fwd.tzz, s_wfl_fwd.txz,
                    s_streams.io);
                gpu_chk_it++;
            }
        }
    } /* end time loop */

    /* Synchronize and download receiver data */
    CUDA_CHECK(cudaStreamSynchronize(s_streams.compute));
    CUDA_CHECK(cudaStreamSynchronize(s_streams.io));

    size_t rec_bytes = rec_size * sizeof(float);
    if (rec_vx  && d_rec.rec_vx)
        CUDA_CHECK(cudaMemcpy(rec_vx,  d_rec.rec_vx,  rec_bytes, cudaMemcpyDeviceToHost));
    if (rec_vz  && d_rec.rec_vz)
        CUDA_CHECK(cudaMemcpy(rec_vz,  d_rec.rec_vz,  rec_bytes, cudaMemcpyDeviceToHost));
    if (rec_txx && d_rec.rec_txx)
        CUDA_CHECK(cudaMemcpy(rec_txx, d_rec.rec_txx, rec_bytes, cudaMemcpyDeviceToHost));
    if (rec_tzz && d_rec.rec_tzz)
        CUDA_CHECK(cudaMemcpy(rec_tzz, d_rec.rec_tzz, rec_bytes, cudaMemcpyDeviceToHost));
    if (d_rec.rec_p)
        CUDA_CHECK(cudaMemcpy(rec_p,   d_rec.rec_p,   rec_bytes, cudaMemcpyDeviceToHost));

    /* Free device receiver buffers */
    cuda_free_rec(&d_rec);
    if (h_rec_ix) free(h_rec_ix);
    if (h_rec_iz) free(h_rec_iz);

    /* Write receiver data via CPU (creates .su files for residual computation) */
    int isam_final = (it1 - 1 - rec->delay + NINT(mod->t0/mod->dt)) / rec->skipdt + 1;
    writeRec(*rec, *mod, *bnd, *wav, ixsrc, izsrc, isam_final + 1,
             ishot, nshots, fileno,
             rec_vx, rec_vz, rec_txx, rec_tzz, rec_txz,
             rec_p, NULL, NULL, NULL, NULL, NULL, NULL, NULL, verbose);

    /* Free host receiver arrays */
    if (rec_vx)  free(rec_vx);
    if (rec_vz)  free(rec_vz);
    if (rec_txx) free(rec_txx);
    if (rec_tzz) free(rec_tzz);
    if (rec_txz) free(rec_txz);
    if (rec_p)   free(rec_p);

    /* Update CPU checkpoint metadata to match GPU state */
    if (chk) chk->it = gpu_chk_it;

    if (verbose > 1)
        vmess("fdfwimodc_gpu: shot %d done, %d time steps, %d checkpoints",
              ishot, it1 - it0, gpu_chk_it);

    return 0;
}


/*====================================================================
 * adj_shot_gpu — GPU adjoint backpropagation with gradient
 *
 * GPU replacement for adj_shot(). Processes checkpoint segments in
 * reverse order:
 *   1. Load checkpoint from pinned host to GPU
 *   2. Re-propagate forward on GPU, buffer vx/vz in device memory
 *   3. Run adjoint on GPU backward through the segment
 *   4. Accumulate gradient via GPU cross-correlation kernels
 *   5. Download gradient from device and add to host arrays
 *====================================================================*/

/* Helper: upload adjoint source traces to device */
static void upload_adjoint_source(deviceAdjSrc *d_adj, adjSrcPar *adj)
{
    int nsrc = adj->nsrc;
    int nt   = adj->nt;

    d_adj->nsrc = nsrc;
    d_adj->nt   = nt;

    /* Waveforms */
    size_t wav_bytes = (size_t)nsrc * nt * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_adj->wav, wav_bytes));
    CUDA_CHECK(cudaMemcpy(d_adj->wav, adj->wav, wav_bytes, cudaMemcpyHostToDevice));

    /* Grid indices */
    size_t idx_bytes = nsrc * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_adj->xi,  idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_adj->zi,  idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_adj->typ, idx_bytes));

    /* Copy xi, zi (converting size_t to int on host before upload) */
    int *h_xi  = (int *)malloc(idx_bytes);
    int *h_zi  = (int *)malloc(idx_bytes);
    for (int i = 0; i < nsrc; i++) {
        h_xi[i] = (int)adj->xi[i];
        h_zi[i] = (int)adj->zi[i];
    }
    CUDA_CHECK(cudaMemcpy(d_adj->xi,  h_xi,      idx_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj->zi,  h_zi,      idx_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj->typ, adj->typ,   idx_bytes, cudaMemcpyHostToDevice));
    free(h_xi);
    free(h_zi);
}

static void free_adjoint_source(deviceAdjSrc *d_adj)
{
    if (d_adj->wav) cudaFree(d_adj->wav);
    if (d_adj->xi)  cudaFree(d_adj->xi);
    if (d_adj->zi)  cudaFree(d_adj->zi);
    if (d_adj->typ) cudaFree(d_adj->typ);
    memset(d_adj, 0, sizeof(deviceAdjSrc));
}


extern "C"
int adj_shot_gpu(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                 recPar *rec, adjSrcPar *adj,
                 int ixsrc, int izsrc, float **src_nwav,
                 checkpointPar *chk, snaPar *sna,
                 float *grad1, float *grad2, float *grad3,
                 int param, int verbose,
                 float *wfld_energy)
{
    int k, j, it, seg_start, seg_end, nsteps, max_nsteps;
    int it1 = mod->nt + NINT(-mod->t0 / mod->dt);
    float dt = mod->dt;
    int n1 = mod->naz;
    size_t field_bytes = s_field_size * sizeof(float);
    double t0_wall;

    float *grad_lam = grad1;
    float *grad_muu = grad2;
    float *grad_rho = grad3;

    if (verbose) t0_wall = wallclock_time();

    /* ------------------------------------------------------------ */
    /* 1. Compute max segment length and allocate device buffers     */
    /* ------------------------------------------------------------ */
    max_nsteps = chk->skipdt;
    {
        int last_start = chk->delay + (chk->nsnap - 1) * chk->skipdt;
        int last_n = it1 - last_start;
        if (last_n > max_nsteps) max_nsteps = last_n;
    }

    /* Allocate/resize forward vx/vz buffer on device.
     * This stores the forward vx/vz for all time steps in one segment
     * so we can cross-correlate with the adjoint at each step. */
    if (max_nsteps > s_buf_max_nsteps) {
        if (s_d_buf_vx) cudaFree(s_d_buf_vx);
        if (s_d_buf_vz) cudaFree(s_d_buf_vz);
        size_t buf_bytes = (size_t)max_nsteps * s_field_size * sizeof(float);
        CUDA_CHECK(cudaMalloc(&s_d_buf_vx, buf_bytes));
        CUDA_CHECK(cudaMalloc(&s_d_buf_vz, buf_bytes));
        s_buf_max_nsteps = max_nsteps;
        if (verbose) {
            float mb = 2.0f * buf_bytes / (1024.0*1024.0);
            vmess("adj_shot_gpu: allocated %.1f MB device buffer for %d steps", mb, max_nsteps);
        }
    }

    /* ------------------------------------------------------------ */
    /* 2. Zero gradient arrays on device                            */
    /* ------------------------------------------------------------ */
    CUDA_CHECK(cudaMemset(s_d_grad_lam, 0, field_bytes));
    CUDA_CHECK(cudaMemset(s_d_grad_muu, 0, field_bytes));
    CUDA_CHECK(cudaMemset(s_d_grad_rho, 0, field_bytes));

    /* ------------------------------------------------------------ */
    /* 3. Zero adjoint wavefield on device                          */
    /* ------------------------------------------------------------ */
    cuda_zero_wfl(&s_wfl_adj);

    /* ------------------------------------------------------------ */
    /* 4. Upload adjoint source (residual traces) to device         */
    /* ------------------------------------------------------------ */
    /* Negate residual: adjoint is driven by -residual */
    {
        size_t i, ntotal = (size_t)adj->nsrc * adj->nt;
        for (i = 0; i < ntotal; i++)
            adj->wav[i] = -adj->wav[i];
    }

    deviceAdjSrc d_adj;
    memset(&d_adj, 0, sizeof(deviceAdjSrc));
    upload_adjoint_source(&d_adj, adj);

    /* ------------------------------------------------------------ */
    /* 5. Process segments in reverse order                         */
    /* ------------------------------------------------------------ */
    for (k = chk->nsnap - 1; k >= 0; k--) {

        seg_start = chk->delay + k * chk->skipdt;
        seg_end   = (k == chk->nsnap - 1) ? it1 : chk->delay + (k+1) * chk->skipdt;
        nsteps    = seg_end - seg_start;

        if (verbose > 1)
            vmess("adj_shot_gpu: Segment %d/%d it=[%d,%d) nsteps=%d",
                  k, chk->nsnap, seg_start, seg_end, nsteps);

        /* ---- 5a. Load checkpoint from pinned host to GPU ---- */
        cuda_checkpoint_load(&s_gpu_chk, k,
            s_wfl_fwd.vx, s_wfl_fwd.vz,
            s_wfl_fwd.txx, s_wfl_fwd.tzz, s_wfl_fwd.txz,
            s_streams.io);
        CUDA_CHECK(cudaStreamSynchronize(s_streams.io));

        /* Store checkpoint state as buf[0] */
        CUDA_CHECK(cudaMemcpy(s_d_buf_vx, s_wfl_fwd.vx,
                   field_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(s_d_buf_vz, s_wfl_fwd.vz,
                   field_bytes, cudaMemcpyDeviceToDevice));

        /* ---- 5b. Re-propagate forward through segment ---- */
        for (j = 1; j < nsteps; j++) {
            it = seg_start + j;
            gpu_forward_one_step(mod, src, wav, bnd, it, ixsrc, izsrc,
                                 src_nwav, &s_wfl_fwd, s_streams.compute);

            /* Buffer fwd vx/vz at step j */
            CUDA_CHECK(cudaMemcpyAsync(
                s_d_buf_vx + (size_t)j * s_field_size, s_wfl_fwd.vx,
                field_bytes, cudaMemcpyDeviceToDevice, s_streams.compute));
            CUDA_CHECK(cudaMemcpyAsync(
                s_d_buf_vz + (size_t)j * s_field_size, s_wfl_fwd.vz,
                field_bytes, cudaMemcpyDeviceToDevice, s_streams.compute));
        }
        CUDA_CHECK(cudaStreamSynchronize(s_streams.compute));

        /* ---- 5c. Adjoint backward through segment ---- */
        for (j = nsteps - 1; j >= 0; j--) {
            it = seg_start + j;

            /* Lambda/mu gradient: needs adjoint stress BEFORE adjoint step.
             * Cross-correlate forward vx/vz with adjoint txx/tzz. */
            float *fwd_vx_j = s_d_buf_vx + (size_t)j * s_field_size;
            float *fwd_vz_j = s_d_buf_vz + (size_t)j * s_field_size;

            cuda_launch_gradient_lambda_mu_P(
                fwd_vx_j, fwd_vz_j,
                s_wfl_adj.txx, s_wfl_adj.tzz,
                s_d_grad_lam, s_d_grad_muu,
                dt, n1, mod->iorder,
                mod->ioPx, mod->ioPz, mod->iePx, mod->iePz,
                s_streams.compute);

            cuda_launch_gradient_mu_shear(
                fwd_vx_j, fwd_vz_j,
                s_wfl_adj.txz, s_d_grad_muu,
                dt, n1, mod->iorder,
                mod->ioTx, mod->ioTz, mod->ieTx, mod->ieTz,
                s_streams.compute);

            /* Density gradient: dv/dt approximation */
            if (j > 0) {
                float *fwd_vx_prev = s_d_buf_vx + (size_t)(j-1) * s_field_size;
                float *fwd_vz_prev = s_d_buf_vz + (size_t)(j-1) * s_field_size;

                cuda_launch_gradient_rho(
                    fwd_vx_j, fwd_vx_prev,
                    fwd_vz_j, fwd_vz_prev,
                    s_wfl_adj.vx, s_wfl_adj.vz,
                    s_dmod.rox, s_d_grad_rho,
                    dt, n1,
                    mod->ioXx, mod->ioXz, mod->ieXx, mod->ieXz,
                    mod->ioZx, mod->ioZz, mod->ieZx, mod->ieZz,
                    s_streams.compute);
            }

            /* Advance adjoint wavefield (stencil + taper + boundaries) */
            gpu_adjoint_one_step(mod, bnd, &s_wfl_adj, s_streams.compute);

            /* Inject adjoint source (force type - after velocity update) */
            {
                int rec_delay = rec->delay;
                int rec_skipdt = rec->skipdt;

                if (it >= rec_delay && ((it - rec_delay) % rec_skipdt == 0)) {
                    int isam = (it - rec_delay) / rec_skipdt;
                    if (isam >= 0 && isam < adj->nt) {
                        /* Phase 1: force sources (after velocity update) */
                        cuda_launch_inject_adjoint_source(
                            s_wfl_adj.vx, s_wfl_adj.vz,
                            s_wfl_adj.txx, s_wfl_adj.tzz, s_wfl_adj.txz,
                            d_adj.wav, d_adj.xi, d_adj.zi, d_adj.typ,
                            d_adj.nsrc, isam, d_adj.nt, n1, 1,
                            s_streams.compute);

                        /* Phase 2: stress sources (after stress update) */
                        cuda_launch_inject_adjoint_source(
                            s_wfl_adj.vx, s_wfl_adj.vz,
                            s_wfl_adj.txx, s_wfl_adj.tzz, s_wfl_adj.txz,
                            d_adj.wav, d_adj.xi, d_adj.zi, d_adj.typ,
                            d_adj.nsrc, isam, d_adj.nt, n1, 2,
                            s_streams.compute);
                    }
                }
            }
        } /* end backward loop through segment */

    } /* end segment loop */

    CUDA_CHECK(cudaStreamSynchronize(s_streams.compute));

    /* ------------------------------------------------------------ */
    /* 6. Download gradient from device and add to host arrays       */
    /* ------------------------------------------------------------ */
    {
        float *h_tmp = (float *)malloc(field_bytes);

        /* Lambda gradient */
        CUDA_CHECK(cudaMemcpy(h_tmp, s_d_grad_lam, field_bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < s_field_size; i++) grad_lam[i] += h_tmp[i];

        /* Mu gradient */
        if (grad_muu) {
            CUDA_CHECK(cudaMemcpy(h_tmp, s_d_grad_muu, field_bytes, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < s_field_size; i++) grad_muu[i] += h_tmp[i];
        }

        /* Rho gradient */
        if (grad_rho) {
            CUDA_CHECK(cudaMemcpy(h_tmp, s_d_grad_rho, field_bytes, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < s_field_size; i++) grad_rho[i] += h_tmp[i];
        }

        free(h_tmp);
    }

    /* ------------------------------------------------------------ */
    /* 7. Restore residual (undo negation)                          */
    /* ------------------------------------------------------------ */
    {
        size_t i, ntotal = (size_t)adj->nsrc * adj->nt;
        for (i = 0; i < ntotal; i++)
            adj->wav[i] = -adj->wav[i];
    }

    /* Free device adjoint source */
    free_adjoint_source(&d_adj);

    if (verbose)
        vmess("adj_shot_gpu: Gradient accumulation completed in %.2f s",
              wallclock_time() - t0_wall);

    /* ------------------------------------------------------------ */
    /* 8. Convert to velocity parameterization if param=2           */
    /* ------------------------------------------------------------ */
    if (param == 2 && mod->ischeme > 2) {
        if (verbose)
            vmess("adj_shot_gpu: Converting gradient to velocity parameterization");
        convertGradientToVelocity(grad1, grad2, grad3,
                                  mod->cp, mod->cs, mod->rho, s_field_size);
    }

    return 0;
}


/*====================================================================
 * DOMAIN-DECOMPOSED GPU FWI DRIVER
 *
 * When ndom > 1, each GPU holds a LOCAL X-axis slice of the model.
 * Halo exchange (via domain_decomp.c) happens every time step
 * between velocity and stress kernel phases.
 *
 * The key differences from single-GPU mode:
 *   - Arrays are nax_local × naz (not nax × naz)
 *   - Stagger offsets come from domainPar (not modPar)
 *   - Halo exchange between phases
 *   - Source injection only on the rank that owns the source
 *   - Receiver extraction on owning ranks, gathered to domain rank 0
 *   - Checkpoint size is per-subdomain
 *====================================================================*/

/* Domain decomposition state */
static domainPar *s_dom = NULL;
static haloBuf   s_halo_vel;    /* halo buffers for velocity exchange (2 fields) */
static haloBuf   s_halo_stress; /* halo buffers for stress exchange (3 fields) */
static int       s_domain_initialized = 0;

/* Local model arrays on host (nax_local × naz slices) */
static float *s_local_l2m = NULL;
static float *s_local_lam = NULL;
static float *s_local_muu = NULL;
static float *s_local_rox = NULL;
static float *s_local_roz = NULL;


/*--------------------------------------------------------------------
 * extract_local_slice -- Extract local subdomain from global array.
 *
 * Copies columns from the global array to a local array based on
 * the domain decomposition layout.
 *--------------------------------------------------------------------*/
static void extract_local_slice(float *local, const float *global,
                                const domainPar *dom, const modPar *mod)
{
    domain_scatter_model(local, global, dom, mod);
}


/*--------------------------------------------------------------------
 * gpu_forward_one_step_domain -- One forward time step with halo exchange.
 *
 * Same as gpu_forward_one_step but uses domain stagger offsets and
 * calls halo exchange between velocity and stress update phases.
 *--------------------------------------------------------------------*/
static void gpu_forward_one_step_domain(
    modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
    int it, int ixsrc, int izsrc, float **src_nwav,
    deviceWfl *wfl, domainPar *dom, cudaStream_t stream)
{
    int n1 = dom->naz_local;
    int iorder = mod->iorder;
    float dt = mod->dt;

    int has_top   = (bnd->top == 4);
    int has_bot   = (bnd->bot == 4);
    int has_left  = (bnd->lef == 4) && dom->has_physical_left;
    int has_right = (bnd->rig == 4) && dom->has_physical_right;

    /* --- Phase 1: Velocity update (local subdomain) --- */
    cuda_launch_velocity_update(wfl, &s_dmod, n1, iorder,
        dom->ioXx, dom->ioXz, dom->ieXx, dom->ieXz,
        dom->ioZx, dom->ioZz, dom->ieZx, dom->ieZz,
        stream);

    /* --- Phase 2: Halo exchange velocity (vx, vz) --- */
    if (dom->ndom > 1) {
        halo_exchange_velocity(wfl->vx, wfl->vz,
                               &s_halo_vel, dom, stream);
    }

    /* --- Phase 3: Force source injection (only on owning rank) --- */
    if (src->type > 5 && src->type != 20) {
        int local_ix = domain_owns_source(dom, ixsrc);
        if (local_ix >= 0) {
            float time = it * dt + mod->t0;
            int id1 = (int)floor(time / dt);
            int id2 = id1 + 1;
            if (time >= 0.0f && id2 <= wav->nt) {
                float ampl = src_nwav[0][id1]*(id2 - time/dt)
                           + src_nwav[0][id2]*(time/dt - id1);
                if (ampl != 0.0f) {
                    int ix_s, iz_s;
                    if (src->type == 6) {
                        iz_s = izsrc + mod->ioXz;
                    } else {
                        iz_s = izsrc + mod->ioZz;
                    }
                    ix_s = local_ix;
                    cuda_launch_inject_source(wfl, &s_dmod,
                        ix_s, iz_s, ampl, src->type, src->orient, n1, stream);
                }
            }
        }
    }

    /* --- Phase 4: Velocity boundary (taper on physical sides + free surface) --- */
    cuda_launch_taper_velocity(wfl, &s_dmod, &s_dbnd,
        n1, bnd->ntap, iorder,
        dom->ioXx, dom->ioXz, dom->ieXx, dom->ieXz,
        dom->ioZx, dom->ioZz, dom->ieZx, dom->ieZz,
        has_top, has_bot, has_left, has_right, stream);

    if (bnd->top == 1)
        cuda_launch_free_surface(wfl, &s_dbnd, n1, dom->ioPx, dom->iePx, stream);

    /* --- Phase 5: Stress update (local subdomain) --- */
    cuda_launch_stress_update(wfl, &s_dmod, n1, iorder,
        dom->ioPx, dom->ioPz, dom->iePx, dom->iePz,
        dom->ioTx, dom->ioTz, dom->ieTx, dom->ieTz,
        stream);

    /* --- Phase 6: Halo exchange stress (txx, tzz, txz) --- */
    if (dom->ndom > 1) {
        halo_exchange_stress(wfl->txx, wfl->tzz, wfl->txz,
                             &s_halo_stress, dom, stream);
    }

    /* --- Phase 7: Stress source injection (only on owning rank) --- */
    if (src->type < 6) {
        int local_ix = domain_owns_source(dom, ixsrc);
        if (local_ix >= 0) {
            float time = it * dt + mod->t0;
            int id1 = (int)floor(time / dt);
            int id2 = id1 + 1;
            if (time >= 0.0f && id2 <= wav->nt) {
                float ampl = src_nwav[0][id1]*(id2 - time/dt)
                           + src_nwav[0][id2]*(time/dt - id1);
                if (ampl != 0.0f) {
                    int src_ibndz = mod->ioPz;
                    if (bnd->top == 4 || bnd->top == 2) src_ibndz += bnd->ntap;

                    int ix_s = local_ix;
                    int iz_s = izsrc + src_ibndz;

                    /* Stress source scaling */
                    float l2m_src;
                    CUDA_CHECK(cudaMemcpy(&l2m_src, s_dmod.l2m + ix_s * n1 + iz_s,
                               sizeof(float), cudaMemcpyDeviceToHost));
                    ampl *= (1.0f / mod->dx) * l2m_src;

                    cuda_launch_inject_source(wfl, &s_dmod,
                        ix_s, iz_s, ampl, src->type, src->orient, n1, stream);
                }
            }
        }
    }

    /* --- Phase 8: Stress boundary (taper on physical sides) --- */
    cuda_launch_taper_stress(wfl, &s_dmod, &s_dbnd,
        n1, bnd->ntap, iorder,
        dom->ioPx, dom->ioPz, dom->iePx, dom->iePz,
        dom->ioTx, dom->ioTz, dom->ieTx, dom->ieTz,
        has_top, has_bot, has_left, has_right, stream);
}


/*--------------------------------------------------------------------
 * gpu_adjoint_one_step_domain -- Adjoint step with halo exchange.
 *--------------------------------------------------------------------*/
static void gpu_adjoint_one_step_domain(
    modPar *mod, bndPar *bnd,
    deviceWfl *wfl_adj, domainPar *dom, cudaStream_t stream)
{
    int n1 = dom->naz_local;
    int iorder = mod->iorder;

    int has_top   = (bnd->top == 4);
    int has_bot   = (bnd->bot == 4);
    int has_left  = (bnd->lef == 4) && dom->has_physical_left;
    int has_right = (bnd->rig == 4) && dom->has_physical_right;

    /* Phase A1: Adjoint velocity update */
    cuda_launch_adj_velocity_update(
        wfl_adj->vx, wfl_adj->vz,
        wfl_adj->txx, wfl_adj->tzz, wfl_adj->txz,
        s_dmod.l2m, s_dmod.lam, s_dmod.muu,
        n1, iorder,
        dom->ioXx, dom->ioXz, dom->ieXx, dom->ieXz,
        dom->ioZx, dom->ioZz, dom->ieZx, dom->ieZz,
        stream);

    /* Halo exchange adjoint velocity */
    if (dom->ndom > 1) {
        halo_exchange_velocity(wfl_adj->vx, wfl_adj->vz,
                               &s_halo_vel, dom, stream);
    }

    /* Adjoint velocity taper */
    cuda_launch_taper_velocity(wfl_adj, &s_dmod, &s_dbnd,
        n1, bnd->ntap, iorder,
        dom->ioXx, dom->ioXz, dom->ieXx, dom->ieXz,
        dom->ioZx, dom->ioZz, dom->ieZx, dom->ieZz,
        has_top, has_bot, has_left, has_right, stream);

    if (bnd->top == 1)
        cuda_launch_free_surface(wfl_adj, &s_dbnd, n1, dom->ioPx, dom->iePx, stream);

    /* Phase A2: Adjoint stress update */
    cuda_launch_adj_stress_update(
        wfl_adj->vx, wfl_adj->vz,
        wfl_adj->txx, wfl_adj->tzz, wfl_adj->txz,
        s_dmod.rox, s_dmod.roz,
        n1, iorder,
        dom->ioPx, dom->ioPz, dom->iePx, dom->iePz,
        dom->ioTx, dom->ioTz, dom->ieTx, dom->ieTz,
        stream);

    /* Halo exchange adjoint stress */
    if (dom->ndom > 1) {
        halo_exchange_stress(wfl_adj->txx, wfl_adj->tzz, wfl_adj->txz,
                             &s_halo_stress, dom, stream);
    }

    /* Adjoint free-surface txz correction */
    if (bnd->top == 1) {
        int ibndz = mod->ioPz;
        if (bnd->top == 4 || bnd->top == 2) ibndz += bnd->ntap;
        cuda_launch_adj_free_surface_txz(
            wfl_adj->txz, s_dmod.rox, wfl_adj->vx,
            n1, iorder, ibndz, dom->ioTx, dom->ieTx, stream);
    }

    /* Adjoint stress taper */
    cuda_launch_taper_stress(wfl_adj, &s_dmod, &s_dbnd,
        n1, bnd->ntap, iorder,
        dom->ioPx, dom->ioPz, dom->iePx, dom->iePz,
        dom->ioTx, dom->ioTz, dom->ieTx, dom->ieTz,
        has_top, has_bot, has_left, has_right, stream);
}


/*====================================================================
 * fdelfwi_gpu_init_domain -- GPU init with domain decomposition.
 *====================================================================*/

extern "C"
void fdelfwi_gpu_init_domain(modPar *mod, bndPar *bnd, recPar *rec,
                             domainPar *dom, int nsnap_max, int verbose)
{
    if (s_gpu_initialized) return;

    s_dom = dom;
    s_nax = dom->nax_local;
    s_naz = dom->naz_local;
    s_field_size = (size_t)s_nax * s_naz;

    /* GPU binding already done by domain_decomp_init */

    /* Create CUDA streams */
    cuda_streams_create(&s_streams);

    /* Upload FD coefficients */
    cuda_set_fd_coefficients(mod->iorder);

    /* Allocate local model arrays */
    size_t local_bytes = s_field_size * sizeof(float);
    s_local_l2m = (float *)malloc(local_bytes);
    s_local_lam = (float *)malloc(local_bytes);
    s_local_muu = (float *)malloc(local_bytes);
    s_local_rox = (float *)malloc(local_bytes);
    s_local_roz = (float *)malloc(local_bytes);

    /* Extract local slices from global model */
    extract_local_slice(s_local_l2m, mod->l2m, dom, mod);
    extract_local_slice(s_local_lam, mod->lam, dom, mod);
    extract_local_slice(s_local_muu, mod->muu, dom, mod);
    extract_local_slice(s_local_rox, mod->rox, dom, mod);
    extract_local_slice(s_local_roz, mod->roz, dom, mod);

    /* Allocate and upload local model to GPU */
    cuda_alloc_mod(&s_dmod, s_nax, s_naz);
    cuda_upload_mod(&s_dmod, s_local_l2m, s_local_lam, s_local_muu,
                    s_local_rox, s_local_roz);

    /* Allocate forward and adjoint wavefields (local dimensions) */
    cuda_alloc_wfl(&s_wfl_fwd, s_nax, s_naz);
    cuda_alloc_wfl(&s_wfl_adj, s_nax, s_naz);

    /* Upload boundary taper arrays.
     * tapz has global nax entries — extract local. tapx is naz (unchanged). */
    /* For simplicity, reuse global taper arrays — the taper kernel
     * only touches columns within the PML zones which exist only on
     * physical-boundary ranks. Interior ranks' taper calls are no-ops. */
    cuda_alloc_bnd(&s_dbnd, bnd->ntap, s_nax,
                    bnd->tapz, bnd->tapx, bnd->tapxz, bnd->surface);

    /* Allocate gradient arrays on device */
    CUDA_CHECK(cudaMalloc(&s_d_grad_lam, local_bytes));
    CUDA_CHECK(cudaMalloc(&s_d_grad_muu, local_bytes));
    CUDA_CHECK(cudaMalloc(&s_d_grad_rho, local_bytes));

    /* Allocate halo exchange buffers */
    halo_buf_create(&s_halo_vel,    2, dom->halo, s_naz);  /* vx, vz */
    halo_buf_create(&s_halo_stress, 3, dom->halo, s_naz);  /* txx, tzz, txz */

    /* GPU checkpoint buffer (local subdomain) */
    if (nsnap_max > 0) {
        cuda_checkpoint_create(&s_gpu_chk, nsnap_max, s_nax, s_naz);
        if (verbose) {
            float mb = (float)nsnap_max * 5 * s_field_size * sizeof(float) / (1024.0*1024.0);
            vmess("GPU checkpoints (local): %d snaps, %.1f MB pinned host (rank %d)",
                  nsnap_max, mb, dom->world_rank);
        }
    }

    s_domain_initialized = 1;
    s_gpu_initialized = 1;

    if (verbose)
        vmess("fdelfwi GPU domain init: rank %d, nax_local=%d naz=%d halo=%d",
              dom->world_rank, s_nax, s_naz, dom->halo);
}


/*====================================================================
 * fdelfwi_gpu_upload_model_domain -- Upload local model slice.
 *====================================================================*/

extern "C"
void fdelfwi_gpu_upload_model_domain(modPar *mod, domainPar *dom)
{
    if (!s_gpu_initialized || !s_domain_initialized) return;

    /* Re-extract local slices after model update */
    extract_local_slice(s_local_l2m, mod->l2m, dom, mod);
    extract_local_slice(s_local_lam, mod->lam, dom, mod);
    extract_local_slice(s_local_muu, mod->muu, dom, mod);
    extract_local_slice(s_local_rox, mod->rox, dom, mod);
    extract_local_slice(s_local_roz, mod->roz, dom, mod);

    cuda_upload_mod(&s_dmod, s_local_l2m, s_local_lam, s_local_muu,
                    s_local_rox, s_local_roz);
}


/*====================================================================
 * fdfwimodc_gpu_domain -- Domain-decomposed GPU forward modeling.
 *====================================================================*/

extern "C"
int fdfwimodc_gpu_domain(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                         recPar *rec, snaPar *sna, int ixsrc, int izsrc,
                         float **src_nwav, int ishot, int nshots, int fileno,
                         checkpointPar *chk, domainPar *dom, int verbose)
{
    int it;
    int n1 = dom->naz_local;
    int it0 = 0;
    int it1 = mod->nt + NINT(-mod->t0 / mod->dt);

    /* --- Map receivers to local subdomain --- */
    int *local_rec_ix = (int *)malloc(rec->n * sizeof(int));
    int *local_rec_iz = (int *)malloc(rec->n * sizeof(int));
    int *local_rec_map = (int *)malloc(rec->n * sizeof(int));
    int nrec_local = 0;

    int ioPz = mod->ioPz;
    if (bnd->top == 4 || bnd->top == 2) ioPz += bnd->ntap;

    domain_map_receivers(rec, dom, &nrec_local, local_rec_ix, local_rec_iz,
                         local_rec_map);

    /* Add z-offset to local receiver positions */
    for (int ir = 0; ir < nrec_local; ir++)
        local_rec_iz[ir] += ioPz;

    /* Allocate device receiver buffers (local receivers only) */
    deviceRec d_rec;
    memset(&d_rec, 0, sizeof(deviceRec));
    cuda_alloc_rec(&d_rec, nrec_local, rec->nt, local_rec_ix, local_rec_iz,
                    rec->type.vx, rec->type.vz,
                    rec->type.txx, rec->type.tzz, 1);

    /* Zero wavefields */
    cuda_zero_wfl(&s_wfl_fwd);

    int gpu_chk_it = 0;

    /* --- MAIN TIME LOOP --- */
    for (it = it0; it < it1; it++) {

        gpu_forward_one_step_domain(mod, src, wav, bnd, it, ixsrc, izsrc,
                                    src_nwav, &s_wfl_fwd, dom, s_streams.compute);

        /* Receiver extraction (local receivers) */
        if (nrec_local > 0 && (((it - rec->delay) % rec->skipdt) == 0) &&
            (it >= rec->delay)) {
            int isam = (it - rec->delay + NINT(mod->t0/mod->dt))
                       / rec->skipdt + 1;
            if (isam < 0) isam = rec->nt + isam;

            cuda_launch_extract_receivers(&s_wfl_fwd, &d_rec,
                isam, n1, s_streams.compute);
        }

        /* GPU checkpoint (async D2H) */
        if (chk && chk->nsnap > 0) {
            if (((it - chk->delay) % chk->skipdt == 0) &&
                (it >= chk->delay) &&
                (gpu_chk_it < s_gpu_chk.nsnap)) {
                cuda_checkpoint_save(&s_gpu_chk, gpu_chk_it,
                    s_wfl_fwd.vx, s_wfl_fwd.vz,
                    s_wfl_fwd.txx, s_wfl_fwd.tzz, s_wfl_fwd.txz,
                    s_streams.io);
                gpu_chk_it++;
            }
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(s_streams.compute));
    CUDA_CHECK(cudaStreamSynchronize(s_streams.io));

    /* --- Download local receiver data and gather to domain rank 0 --- */
    size_t local_rec_bytes = (size_t)nrec_local * rec->nt * sizeof(float);
    float *h_local_rec_vx = NULL, *h_local_rec_vz = NULL;
    float *h_local_rec_txx = NULL, *h_local_rec_tzz = NULL, *h_local_rec_p = NULL;

    if (nrec_local > 0) {
        if (rec->type.vx && d_rec.rec_vx) {
            h_local_rec_vx = (float *)calloc(nrec_local * rec->nt, sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_local_rec_vx, d_rec.rec_vx, local_rec_bytes, cudaMemcpyDeviceToHost));
        }
        if (rec->type.vz && d_rec.rec_vz) {
            h_local_rec_vz = (float *)calloc(nrec_local * rec->nt, sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_local_rec_vz, d_rec.rec_vz, local_rec_bytes, cudaMemcpyDeviceToHost));
        }
        if (rec->type.txx && d_rec.rec_txx) {
            h_local_rec_txx = (float *)calloc(nrec_local * rec->nt, sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_local_rec_txx, d_rec.rec_txx, local_rec_bytes, cudaMemcpyDeviceToHost));
        }
        if (rec->type.tzz && d_rec.rec_tzz) {
            h_local_rec_tzz = (float *)calloc(nrec_local * rec->nt, sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_local_rec_tzz, d_rec.rec_tzz, local_rec_bytes, cudaMemcpyDeviceToHost));
        }
        if (d_rec.rec_p) {
            h_local_rec_p = (float *)calloc(nrec_local * rec->nt, sizeof(float));
            CUDA_CHECK(cudaMemcpy(h_local_rec_p, d_rec.rec_p, local_rec_bytes, cudaMemcpyDeviceToHost));
        }
    }

    cuda_free_rec(&d_rec);

    /* Gather receiver traces to domain rank 0 via MPI */
    /* Domain rank 0 assembles full receiver arrays and calls writeRec */
#ifdef USE_MPI
    if (dom->ndom > 1) {
        size_t full_rec = (size_t)rec->n * rec->nt;
        float *full_rec_vx = NULL, *full_rec_vz = NULL;
        float *full_rec_txx = NULL, *full_rec_tzz = NULL, *full_rec_p = NULL;

        if (dom->domain_rank == 0) {
            if (rec->type.vx)  full_rec_vx  = (float *)calloc(full_rec, sizeof(float));
            if (rec->type.vz)  full_rec_vz  = (float *)calloc(full_rec, sizeof(float));
            if (rec->type.txx) full_rec_txx = (float *)calloc(full_rec, sizeof(float));
            if (rec->type.tzz) full_rec_tzz = (float *)calloc(full_rec, sizeof(float));
            full_rec_p = (float *)calloc(full_rec, sizeof(float));
        }

        /* Each rank scatters its local receiver traces into the global array
         * using the local_rec_map indices. Simple approach: each rank sends
         * its local traces to rank 0 which assembles them. */
        /* For simplicity, use MPI_Gatherv with counts per rank */
        {
            int *recv_counts = NULL, *recv_displs = NULL;
            if (dom->domain_rank == 0) {
                recv_counts = (int *)calloc(dom->ndom, sizeof(int));
                recv_displs = (int *)calloc(dom->ndom, sizeof(int));
            }

            /* Gather nrec_local counts */
            MPI_Gather(&nrec_local, 1, MPI_INT,
                       recv_counts, 1, MPI_INT, 0, dom->domain_comm);

            if (dom->domain_rank == 0) {
                /* Compute displacements */
                recv_displs[0] = 0;
                for (int r = 1; r < dom->ndom; r++)
                    recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
            }

            /* Gather receiver map indices to rank 0 */
            int *all_rec_map = NULL;
            if (dom->domain_rank == 0)
                all_rec_map = (int *)malloc(rec->n * sizeof(int));

            MPI_Gatherv(local_rec_map, nrec_local, MPI_INT,
                        all_rec_map, recv_counts, recv_displs, MPI_INT,
                        0, dom->domain_comm);

            /* Gather traces per component (multiply counts/displs by nt) */
            int *recv_counts_t = NULL, *recv_displs_t = NULL;
            if (dom->domain_rank == 0) {
                recv_counts_t = (int *)malloc(dom->ndom * sizeof(int));
                recv_displs_t = (int *)malloc(dom->ndom * sizeof(int));
                for (int r = 0; r < dom->ndom; r++) {
                    recv_counts_t[r] = recv_counts[r] * rec->nt;
                    recv_displs_t[r] = recv_displs[r] * rec->nt;
                }
            }

            /* Temporary gather buffer (contiguous, unsorted) */
            float *tmp_buf = NULL;
            if (dom->domain_rank == 0)
                tmp_buf = (float *)malloc(full_rec * sizeof(float));

            /* Gather and reassemble each component */
            #define GATHER_COMP(local_buf, full_buf) \
            do { \
                if (full_buf) { \
                    MPI_Gatherv(local_buf, nrec_local * rec->nt, MPI_FLOAT, \
                                tmp_buf, recv_counts_t, recv_displs_t, MPI_FLOAT, \
                                0, dom->domain_comm); \
                    if (dom->domain_rank == 0) { \
                        int total_gathered = recv_displs[dom->ndom-1] + recv_counts[dom->ndom-1]; \
                        for (int ir = 0; ir < total_gathered; ir++) { \
                            int global_ir = all_rec_map[ir]; \
                            memcpy(full_buf + (size_t)global_ir * rec->nt, \
                                   tmp_buf + (size_t)ir * rec->nt, \
                                   rec->nt * sizeof(float)); \
                        } \
                    } \
                } \
            } while(0)

            GATHER_COMP(h_local_rec_vx,  full_rec_vx);
            GATHER_COMP(h_local_rec_vz,  full_rec_vz);
            GATHER_COMP(h_local_rec_txx, full_rec_txx);
            GATHER_COMP(h_local_rec_tzz, full_rec_tzz);
            GATHER_COMP(h_local_rec_p,   full_rec_p);

            #undef GATHER_COMP

            if (tmp_buf) free(tmp_buf);
            if (recv_counts) free(recv_counts);
            if (recv_displs) free(recv_displs);
            if (recv_counts_t) free(recv_counts_t);
            if (recv_displs_t) free(recv_displs_t);
            if (all_rec_map) free(all_rec_map);
        }

        /* Domain rank 0: write receiver data */
        if (dom->domain_rank == 0) {
            int isam_final = (it1 - 1 - rec->delay + NINT(mod->t0/mod->dt)) / rec->skipdt + 1;
            writeRec(*rec, *mod, *bnd, *wav, ixsrc, izsrc, isam_final + 1,
                     ishot, nshots, fileno,
                     full_rec_vx, full_rec_vz, full_rec_txx, full_rec_tzz, NULL,
                     full_rec_p, NULL, NULL, NULL, NULL, NULL, NULL, NULL, verbose);
        }

        if (full_rec_vx)  free(full_rec_vx);
        if (full_rec_vz)  free(full_rec_vz);
        if (full_rec_txx) free(full_rec_txx);
        if (full_rec_tzz) free(full_rec_tzz);
        if (full_rec_p)   free(full_rec_p);
    }
#endif

    /* Free local receiver arrays */
    if (h_local_rec_vx)  free(h_local_rec_vx);
    if (h_local_rec_vz)  free(h_local_rec_vz);
    if (h_local_rec_txx) free(h_local_rec_txx);
    if (h_local_rec_tzz) free(h_local_rec_tzz);
    if (h_local_rec_p)   free(h_local_rec_p);
    free(local_rec_ix);
    free(local_rec_iz);
    free(local_rec_map);

    if (chk) chk->it = gpu_chk_it;

    if (verbose > 1)
        vmess("fdfwimodc_gpu_domain: rank %d, shot %d, %d steps, %d chk, %d local rcv",
              dom->world_rank, ishot, it1 - it0, gpu_chk_it, nrec_local);

    return 0;
}


/*====================================================================
 * adj_shot_gpu_domain -- Domain-decomposed GPU adjoint + gradient.
 *====================================================================*/

extern "C"
int adj_shot_gpu_domain(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                        recPar *rec, adjSrcPar *adj,
                        int ixsrc, int izsrc, float **src_nwav,
                        checkpointPar *chk, snaPar *sna,
                        float *grad1, float *grad2, float *grad3,
                        domainPar *dom,
                        int param, int verbose,
                        float *wfld_energy)
{
    int k, j, it, seg_start, seg_end, nsteps, max_nsteps;
    int it1 = mod->nt + NINT(-mod->t0 / mod->dt);
    float dt = mod->dt;
    int n1 = dom->naz_local;
    size_t field_bytes = s_field_size * sizeof(float);
    double t0_wall;

    float *grad_lam = grad1;
    float *grad_muu = grad2;
    float *grad_rho = grad3;

    if (verbose) t0_wall = wallclock_time();

    /* Max segment length */
    max_nsteps = chk->skipdt;
    {
        int last_start = chk->delay + (chk->nsnap - 1) * chk->skipdt;
        int last_n = it1 - last_start;
        if (last_n > max_nsteps) max_nsteps = last_n;
    }

    /* Allocate/resize forward vx/vz buffer on device */
    if (max_nsteps > s_buf_max_nsteps) {
        if (s_d_buf_vx) cudaFree(s_d_buf_vx);
        if (s_d_buf_vz) cudaFree(s_d_buf_vz);
        size_t buf_bytes = (size_t)max_nsteps * s_field_size * sizeof(float);
        CUDA_CHECK(cudaMalloc(&s_d_buf_vx, buf_bytes));
        CUDA_CHECK(cudaMalloc(&s_d_buf_vz, buf_bytes));
        s_buf_max_nsteps = max_nsteps;
    }

    /* Zero gradients and adjoint wavefield */
    CUDA_CHECK(cudaMemset(s_d_grad_lam, 0, field_bytes));
    CUDA_CHECK(cudaMemset(s_d_grad_muu, 0, field_bytes));
    CUDA_CHECK(cudaMemset(s_d_grad_rho, 0, field_bytes));
    cuda_zero_wfl(&s_wfl_adj);

    /* Negate residual */
    {
        size_t i, ntotal = (size_t)adj->nsrc * adj->nt;
        for (i = 0; i < ntotal; i++)
            adj->wav[i] = -adj->wav[i];
    }

    /* Upload adjoint source.
     * In domain-decomposed mode, each rank only uploads the adjoint sources
     * that correspond to receivers on its subdomain. However, the adjoint
     * source injection kernel handles positions outside the local grid by
     * bounds checking. For simplicity, upload ALL adjoint sources on all
     * ranks — the kernel skips out-of-range positions. */
    deviceAdjSrc d_adj;
    memset(&d_adj, 0, sizeof(deviceAdjSrc));
    upload_adjoint_source(&d_adj, adj);

    /* Process segments in reverse */
    for (k = chk->nsnap - 1; k >= 0; k--) {

        seg_start = chk->delay + k * chk->skipdt;
        seg_end   = (k == chk->nsnap - 1) ? it1 : chk->delay + (k+1) * chk->skipdt;
        nsteps    = seg_end - seg_start;

        /* Load checkpoint from pinned host */
        cuda_checkpoint_load(&s_gpu_chk, k,
            s_wfl_fwd.vx, s_wfl_fwd.vz,
            s_wfl_fwd.txx, s_wfl_fwd.tzz, s_wfl_fwd.txz,
            s_streams.io);
        CUDA_CHECK(cudaStreamSynchronize(s_streams.io));

        /* Buffer checkpoint as step 0 */
        CUDA_CHECK(cudaMemcpy(s_d_buf_vx, s_wfl_fwd.vx,
                   field_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(s_d_buf_vz, s_wfl_fwd.vz,
                   field_bytes, cudaMemcpyDeviceToDevice));

        /* Re-propagate forward with halo exchange */
        for (j = 1; j < nsteps; j++) {
            it = seg_start + j;
            gpu_forward_one_step_domain(mod, src, wav, bnd, it, ixsrc, izsrc,
                                        src_nwav, &s_wfl_fwd, dom, s_streams.compute);

            CUDA_CHECK(cudaMemcpyAsync(
                s_d_buf_vx + (size_t)j * s_field_size, s_wfl_fwd.vx,
                field_bytes, cudaMemcpyDeviceToDevice, s_streams.compute));
            CUDA_CHECK(cudaMemcpyAsync(
                s_d_buf_vz + (size_t)j * s_field_size, s_wfl_fwd.vz,
                field_bytes, cudaMemcpyDeviceToDevice, s_streams.compute));
        }
        CUDA_CHECK(cudaStreamSynchronize(s_streams.compute));

        /* Adjoint backward through segment */
        for (j = nsteps - 1; j >= 0; j--) {
            it = seg_start + j;

            float *fwd_vx_j = s_d_buf_vx + (size_t)j * s_field_size;
            float *fwd_vz_j = s_d_buf_vz + (size_t)j * s_field_size;

            /* Lambda/mu gradient */
            cuda_launch_gradient_lambda_mu_P(
                fwd_vx_j, fwd_vz_j,
                s_wfl_adj.txx, s_wfl_adj.tzz,
                s_d_grad_lam, s_d_grad_muu,
                dt, n1, mod->iorder,
                dom->ioPx, dom->ioPz, dom->iePx, dom->iePz,
                s_streams.compute);

            cuda_launch_gradient_mu_shear(
                fwd_vx_j, fwd_vz_j,
                s_wfl_adj.txz, s_d_grad_muu,
                dt, n1, mod->iorder,
                dom->ioTx, dom->ioTz, dom->ieTx, dom->ieTz,
                s_streams.compute);

            /* Rho gradient (dv/dt) */
            if (j > 0) {
                float *fwd_vx_prev = s_d_buf_vx + (size_t)(j-1) * s_field_size;
                float *fwd_vz_prev = s_d_buf_vz + (size_t)(j-1) * s_field_size;

                cuda_launch_gradient_rho(
                    fwd_vx_j, fwd_vx_prev,
                    fwd_vz_j, fwd_vz_prev,
                    s_wfl_adj.vx, s_wfl_adj.vz,
                    s_dmod.rox, s_d_grad_rho,
                    dt, n1,
                    dom->ioXx, dom->ioXz, dom->ieXx, dom->ieXz,
                    dom->ioZx, dom->ioZz, dom->ieZx, dom->ieZz,
                    s_streams.compute);
            }

            /* Adjoint time step with halo exchange */
            gpu_adjoint_one_step_domain(mod, bnd, &s_wfl_adj, dom, s_streams.compute);

            /* Inject adjoint source */
            {
                int rec_delay = rec->delay;
                int rec_skipdt = rec->skipdt;

                if (it >= rec_delay && ((it - rec_delay) % rec_skipdt == 0)) {
                    int isam = (it - rec_delay) / rec_skipdt;
                    if (isam >= 0 && isam < adj->nt) {
                        cuda_launch_inject_adjoint_source(
                            s_wfl_adj.vx, s_wfl_adj.vz,
                            s_wfl_adj.txx, s_wfl_adj.tzz, s_wfl_adj.txz,
                            d_adj.wav, d_adj.xi, d_adj.zi, d_adj.typ,
                            d_adj.nsrc, isam, d_adj.nt, n1, 1,
                            s_streams.compute);
                        cuda_launch_inject_adjoint_source(
                            s_wfl_adj.vx, s_wfl_adj.vz,
                            s_wfl_adj.txx, s_wfl_adj.tzz, s_wfl_adj.txz,
                            d_adj.wav, d_adj.xi, d_adj.zi, d_adj.typ,
                            d_adj.nsrc, isam, d_adj.nt, n1, 2,
                            s_streams.compute);
                    }
                }
            }
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(s_streams.compute));

    /* Download gradient and add to host arrays.
     * The host arrays are GLOBAL-sized (nax × naz). The GPU gradient is
     * LOCAL-sized (nax_local × naz). We map local positions to global
     * positions when adding. Only interior columns are mapped; padding
     * columns are not part of the physical gradient. */
    {
        float *h_tmp = (float *)malloc(field_bytes);
        int naz = dom->naz_local;
        int nax_global = mod->nax;
        int ioPx_global = mod->ioPx;

        /* Map: local column ix_local → global column ix_global
         *   ix_global = ix_local - pad_left + ioPx_global + ix_global_start
         * But we include ALL local columns (interior + padding) to capture
         * the full gradient including PML zones on boundary ranks. */

        #define MAP_GRAD(d_grad, h_grad) \
        do { \
            CUDA_CHECK(cudaMemcpy(h_tmp, d_grad, field_bytes, cudaMemcpyDeviceToHost)); \
            for (int ix_l = 0; ix_l < s_nax; ix_l++) { \
                int ix_g; \
                if (dom->has_physical_left) { \
                    ix_g = ix_l; \
                } else { \
                    ix_g = ioPx_global + dom->ix_global_start + ix_l - dom->pad_left; \
                } \
                if (ix_g >= 0 && ix_g < nax_global) { \
                    for (int iz = 0; iz < naz; iz++) \
                        h_grad[ix_g * naz + iz] += h_tmp[ix_l * naz + iz]; \
                } \
            } \
        } while(0)

        MAP_GRAD(s_d_grad_lam, grad_lam);
        if (grad_muu) MAP_GRAD(s_d_grad_muu, grad_muu);
        if (grad_rho) MAP_GRAD(s_d_grad_rho, grad_rho);

        #undef MAP_GRAD

        free(h_tmp);
    }

    /* Restore residual */
    {
        size_t i, ntotal = (size_t)adj->nsrc * adj->nt;
        for (i = 0; i < ntotal; i++)
            adj->wav[i] = -adj->wav[i];
    }

    free_adjoint_source(&d_adj);

    if (verbose)
        vmess("adj_shot_gpu_domain: rank %d, gradient done in %.2f s",
              dom->world_rank, wallclock_time() - t0_wall);

    /* Note: velocity conversion (param=2) is NOT applied here because
     * gradients are accumulated in Lamé space across shots. The conversion
     * is done once by extractGradientVector in the caller. */

    return 0;
}

#endif /* USE_CUDA */

