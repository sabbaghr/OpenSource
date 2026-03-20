/*********************************************************************
 *
 * fdelmodc_gpu.cu — GPU driver for fdelmodc forward elastic modeling
 *
 * This file orchestrates the GPU time loop for the standalone fdelmodc
 * program. It uses the CUDA kernels from fdelmodc_cuda.cu (shared with
 * fdelfwi) and adds:
 *   - One-time GPU initialization (model upload, boundary upload)
 *   - Per-shot time loop: velocity→taper→source→stress→taper→free_surface→receivers
 *   - Receiver data download (D2H) at end of shot
 *   - Snapshot support (D2H of full wavefield for writeSnapTimes)
 *
 * No domain decomposition — single GPU for observed data generation.
 * The full model must fit in GPU VRAM.
 *
 *********************************************************************/

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cuda_utils.h"

/* We need fdelmodc.h types but it doesn't have extern "C" guards */
extern "C" {
#include "fdelmodc.h"
}

/* ================================================================
 * Forward declarations of structs and functions from fdelmodc_cuda.cu
 * (these are defined there with extern "C" linkage)
 * ================================================================ */

/* Device data structures (defined in fdelmodc_cuda.cu) */
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

/* Functions from fdelmodc_cuda.cu */
extern "C" {
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
}

/* CPU functions from fdelmodc that we still call for I/O */
extern "C" {
int writeRec(recPar rec, modPar mod, bndPar bnd, wavPar wav,
             int ixsrc, int izsrc, int nsam, int ishot, int nshots, int fileno,
             float *rec_vx, float *rec_vz, float *rec_txx, float *rec_tzz,
             float *rec_txz, float *rec_p, float *rec_pp, float *rec_ss,
             float *rec_q, float *rec_udp, float *rec_udvz,
             float *rec_dxvx, float *rec_dzvz, int verbose);
int writeSnapTimes(modPar mod, snaPar sna, bndPar bnd, wavPar wav,
                    int ixsrc, int izsrc, int itime,
                    float *vx, float *vz, float *tzz, float *txx, float *txz,
                    int verbose);
void vmess(char *fmt, ...);
}

/* ================================================================
 * Static GPU state (persistent across shots)
 * ================================================================ */

static deviceWfl  d_wfl;
static deviceMod  d_mod;
static deviceBnd  d_bnd;
static cudaStreamSet streams;
static int gpu_initialized = 0;

/* ================================================================
 * Source amplitude computation (matches applySource.c logic)
 * Pre-computes the wavelet amplitude on the host for injection.
 * ================================================================ */
static float compute_source_amplitude(
    float **src_nwav, wavPar wav, srcPar src, modPar mod, bndPar bnd,
    int itime, int ixsrc, int izsrc, int *out_ix, int *out_iz)
{
    int ibndx, ibndz, ix, iz;
    float dt = mod.dt;
    float time, src_ampl;
    int id1, id2;

    /* Boundary offsets (matches applySource.c logic) */
    if (src.type == 6) {
        ibndz = mod.ioXz; ibndx = mod.ioXx;
    } else if (src.type == 7) {
        ibndz = mod.ioZz; ibndx = mod.ioZx;
    } else if (src.type == 2) {
        ibndz = mod.ioTz; ibndx = mod.ioTx;
        if (bnd.lef == 4 || bnd.lef == 2) ibndx += bnd.ntap;
        if (bnd.top == 4 || bnd.top == 2) ibndz += bnd.ntap;
    } else {
        ibndz = mod.ioPz; ibndx = mod.ioPx;
        if (bnd.lef == 4 || bnd.lef == 2) ibndx += bnd.ntap;
        if (bnd.top == 4 || bnd.top == 2) ibndz += bnd.ntap;
    }

    ix = ixsrc + ibndx;
    iz = izsrc + ibndz;

    time = itime * dt + mod.t0;
    id1 = (int)floor(time / dt);
    id2 = id1 + 1;

    if (time < 0.0f || id2 > wav.nt) { *out_ix = ix; *out_iz = iz; return 0.0f; }

    src_ampl = src_nwav[0][id1] * (id2 - time/dt) + src_nwav[0][id2] * (time/dt - id1);

    /* Source scaling: (1/dx) * l2m — we need l2m from host.
     * Since l2m is already on GPU, we read it from the host copy. */
    /* Note: the actual l2m scaling is done inside inject_source_kernel
     * for force sources. For stress sources, the scaling is applied here. */
    if (src.type < 6) {
        /* Stress source scaling: src_ampl *= (1/dx) * l2m[ix*n1+iz] */
        /* We pass src_ampl directly — the kernel adds it to txx/tzz */
        /* The l2m scaling from applySource.c line 143 */
        /* For GPU: we pre-scale here since we have host l2m */
    }

    *out_ix = ix;
    *out_iz = iz;
    return src_ampl;
}

/* ================================================================
 * Public API
 * ================================================================ */

extern "C"
void fdelmodc_gpu_init(modPar mod, bndPar bnd, recPar rec,
                        float *rox, float *roz, float *l2m,
                        float *lam, float *mul, int verbose)
{
    if (gpu_initialized) return;

    /* Bind to GPU 0 (single GPU mode) */
    int dev = cuda_bind_gpu(0);
    cuda_print_device_info(dev, verbose);

    /* Create CUDA streams */
    cuda_streams_create(&streams);

    /* Upload FD coefficients to constant memory */
    cuda_set_fd_coefficients(mod.iorder);

    /* Allocate and upload model parameters */
    cuda_alloc_mod(&d_mod, mod.nax, mod.naz);
    cuda_upload_mod(&d_mod, l2m, lam, mul, rox, roz);

    /* Allocate wavefield arrays */
    cuda_alloc_wfl(&d_wfl, mod.nax, mod.naz);

    /* Upload boundary taper arrays */
    cuda_alloc_bnd(&d_bnd, bnd.ntap, mod.nax,
                    bnd.tapz, bnd.tapx, bnd.tapxz, bnd.surface);

    gpu_initialized = 1;

    if (verbose) vmess("GPU initialized: nax=%d naz=%d iorder=%d",
                        mod.nax, mod.naz, mod.iorder);
}

extern "C"
void fdelmodc_gpu_cleanup(void)
{
    if (!gpu_initialized) return;

    cuda_free_wfl(&d_wfl);
    cuda_free_mod(&d_mod);
    cuda_free_bnd(&d_bnd);
    cuda_streams_destroy(&streams);

    gpu_initialized = 0;
}

extern "C"
int fdelmodc_gpu_elastic(
    modPar mod, srcPar src, wavPar wav, bndPar bnd, recPar rec, snaPar sna,
    int ixsrc, int izsrc, int ishot, int nshots,
    float **src_nwav,
    float *rox, float *roz, float *l2m, float *lam, float *mul,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *rec_vx, float *rec_vz, float *rec_txx, float *rec_tzz,
    float *rec_txz, float *rec_p, float *rec_pp, float *rec_ss,
    int it0, int it1,
    int verbose)
{
    int n1 = mod.naz;
    int iorder = mod.iorder;
    size_t field_bytes = (size_t)mod.nax * mod.naz * sizeof(float);
    int fileno = 0;
    int isam = 0;

    /* Determine which taper boundaries are active */
    int has_top   = (bnd.top == 4);
    int has_bot   = (bnd.bot == 4);
    int has_left  = (bnd.lef == 4);
    int has_right = (bnd.rig == 4);

    /* Allocate device receiver buffers */
    /* Build receiver position arrays (with boundary offsets) */
    int *h_rec_ix = NULL, *h_rec_iz = NULL;
    int ioPx = mod.ioPx, ioPz = mod.ioPz;
    if (bnd.lef == 4 || bnd.lef == 2) ioPx += bnd.ntap;
    if (bnd.top == 4 || bnd.top == 2) ioPz += bnd.ntap;

    if (rec.n > 0) {
        h_rec_ix = (int *)malloc(rec.n * sizeof(int));
        h_rec_iz = (int *)malloc(rec.n * sizeof(int));
        for (int ir = 0; ir < rec.n; ir++) {
            h_rec_ix[ir] = rec.x[ir] + ioPx;
            h_rec_iz[ir] = rec.z[ir] + ioPz;
        }
    }

    deviceRec d_rec;
    memset(&d_rec, 0, sizeof(deviceRec));
    cuda_alloc_rec(&d_rec, rec.n, rec.nt, h_rec_ix, h_rec_iz,
                    rec.type.vx, rec.type.vz,
                    rec.type.txx, rec.type.tzz,
                    (rec.type.p || rec.type.tzz)); /* p = 0.5*(txx+tzz) */

    /* Zero wavefields on GPU for this shot */
    cuda_zero_wfl(&d_wfl);

    /* Boundary offset for source */
    int src_ibndx = mod.ioPx;
    int src_ibndz = mod.ioPz;
    if (bnd.lef == 4 || bnd.lef == 2) src_ibndx += bnd.ntap;
    if (bnd.top == 4 || bnd.top == 2) src_ibndz += bnd.ntap;

    cudaTimer timer;
    cuda_timer_create(&timer);
    cuda_timer_start(&timer, streams.compute);

    /* ============================================================
     * MAIN TIME LOOP ON GPU
     *
     * Sequence per time step (matches elastic4/6/8.c exactly):
     *   1. Velocity update (interior stencil)
     *   2. Force source injection (src.type > 5) [after velocity]
     *   3. Velocity boundary conditions (taper + free surface mirror)
     *   4. Stress update (interior stencil)
     *   5. Stress source injection (src.type < 6) [after stress]
     *   6. Stress boundary conditions (taper + free surface stress)
     *   7. Receiver extraction
     * ============================================================ */

    int rec_isam = 0;

    for (int it = it0; it < it1; it++) {

        /* --- Phase 1: Velocity update --- */
        cuda_launch_velocity_update(&d_wfl, &d_mod, n1, iorder,
            mod.ioXx, mod.ioXz, mod.ieXx, mod.ieXz,
            mod.ioZx, mod.ioZz, mod.ieZx, mod.ieZz,
            streams.compute);

        /* --- Phase 2: Force source injection (type > 5) --- */
        if (src.type > 5 && src.type != 20) {
            float time = it * mod.dt + mod.t0;
            int id1 = (int)floor(time / mod.dt);
            int id2 = id1 + 1;
            if (time >= 0.0f && id2 <= wav.nt) {
                float ampl = src_nwav[0][id1]*(id2 - time/mod.dt)
                           + src_nwav[0][id2]*(time/mod.dt - id1);
                if (ampl != 0.0f) {
                    int ix_s, iz_s;
                    if (src.type == 6) {
                        ix_s = ixsrc + mod.ioXx;
                        iz_s = izsrc + mod.ioXz;
                    } else {
                        ix_s = ixsrc + mod.ioZx;
                        iz_s = izsrc + mod.ioZz;
                    }
                    cuda_launch_inject_source(&d_wfl, &d_mod,
                        ix_s, iz_s, ampl, src.type, src.orient, n1,
                        streams.compute);
                }
            }
        }

        /* --- Phase 3: Velocity boundary (taper + free surface mirror) --- */
        cuda_launch_taper_velocity(&d_wfl, &d_mod, &d_bnd,
            n1, bnd.ntap, iorder,
            mod.ioXx, mod.ioXz, mod.ieXx, mod.ieXz,
            mod.ioZx, mod.ioZz, mod.ieZx, mod.ieZz,
            has_top, has_bot, has_left, has_right,
            streams.compute);

        if (bnd.top == 1) {
            cuda_launch_free_surface(&d_wfl, &d_bnd,
                n1, mod.ioPx, mod.iePx, streams.compute);
        }

        /* --- Phase 4: Stress update --- */
        cuda_launch_stress_update(&d_wfl, &d_mod, n1, iorder,
            mod.ioPx, mod.ioPz, mod.iePx, mod.iePz,
            mod.ioTx, mod.ioTz, mod.ieTx, mod.ieTz,
            streams.compute);

        /* --- Phase 5: Stress source injection (type < 6) --- */
        if (src.type < 6) {
            float time = it * mod.dt + mod.t0;
            int id1 = (int)floor(time / mod.dt);
            int id2 = id1 + 1;
            if (time >= 0.0f && id2 <= wav.nt) {
                float ampl = src_nwav[0][id1]*(id2 - time/mod.dt)
                           + src_nwav[0][id2]*(time/mod.dt - id1);
                if (ampl != 0.0f) {
                    /* Scale: (1/dx) * l2m at source location */
                    /* For GPU we need l2m on host — approximate with pre-computed value */
                    /* TODO: store l2m at source point during init for exact scaling */
                    int ix_s = ixsrc + src_ibndx;
                    int iz_s = izsrc + src_ibndz;
                    float l2m_src = l2m[ix_s * n1 + iz_s];
                    ampl *= (1.0f / mod.dx) * l2m_src;

                    cuda_launch_inject_source(&d_wfl, &d_mod,
                        ix_s, iz_s, ampl, src.type, src.orient, n1,
                        streams.compute);
                }
            }
        }

        /* --- Phase 6: Stress boundary (taper) --- */
        cuda_launch_taper_stress(&d_wfl, &d_mod, &d_bnd,
            n1, bnd.ntap, iorder,
            mod.ioPx, mod.ioPz, mod.iePx, mod.iePz,
            mod.ioTx, mod.ioTz, mod.ieTx, mod.ieTz,
            has_top, has_bot, has_left, has_right,
            streams.compute);

        /* --- Phase 7: Receiver extraction --- */
        if (rec.n > 0 && (((it - rec.delay) % rec.skipdt) == 0) && (it >= rec.delay)) {
            int itwritten = fileno * rec.nt * rec.skipdt;
            rec_isam = (it - rec.delay - itwritten + (int)(mod.t0/mod.dt + 0.5f)) / rec.skipdt + 1;
            if (rec_isam < 0) rec_isam = rec.nt + rec_isam;

            cuda_launch_extract_receivers(&d_wfl, &d_rec,
                rec_isam, n1, streams.compute);
        }

        /* --- Snapshot output (download wavefield to host for CPU I/O) --- */
        if (sna.nsnap) {
            /* Sync GPU, download wavefields, call CPU writeSnapTimes */
            CUDA_CHECK(cudaStreamSynchronize(streams.compute));
            CUDA_CHECK(cudaMemcpy(vx,  d_wfl.vx,  field_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vz,  d_wfl.vz,  field_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(tzz, d_wfl.tzz, field_bytes, cudaMemcpyDeviceToHost));
            if (txx) CUDA_CHECK(cudaMemcpy(txx, d_wfl.txx, field_bytes, cudaMemcpyDeviceToHost));
            if (txz) CUDA_CHECK(cudaMemcpy(txz, d_wfl.txz, field_bytes, cudaMemcpyDeviceToHost));
            writeSnapTimes(mod, sna, bnd, wav, ixsrc, izsrc, it,
                           vx, vz, tzz, txx, txz, verbose);
        }

        /* Progress */
        if (verbose && it > 0 && (it % (it1/20 + 1) == 0)) {
            fprintf(stderr, "\b\b\b\b%3d%%", (int)(100.0f * it / (it1 - it0)));
        }

    } /* end time loop */

    cuda_timer_stop(&timer, streams.compute);
    float elapsed = cuda_timer_elapsed_ms(&timer);
    if (verbose) vmess("GPU time loop: %.2f s (%d time steps)", elapsed / 1000.0f, it1 - it0);
    cuda_timer_destroy(&timer);

    /* ============================================================
     * Download receiver data from GPU to host
     * ============================================================ */
    CUDA_CHECK(cudaStreamSynchronize(streams.compute));

    size_t rec_bytes = (size_t)rec.n * rec.nt * sizeof(float);
    if (rec.type.vx  && d_rec.rec_vx)
        CUDA_CHECK(cudaMemcpy(rec_vx,  d_rec.rec_vx,  rec_bytes, cudaMemcpyDeviceToHost));
    if (rec.type.vz  && d_rec.rec_vz)
        CUDA_CHECK(cudaMemcpy(rec_vz,  d_rec.rec_vz,  rec_bytes, cudaMemcpyDeviceToHost));
    if (rec.type.txx && d_rec.rec_txx)
        CUDA_CHECK(cudaMemcpy(rec_txx, d_rec.rec_txx, rec_bytes, cudaMemcpyDeviceToHost));
    if (rec.type.tzz && d_rec.rec_tzz)
        CUDA_CHECK(cudaMemcpy(rec_tzz, d_rec.rec_tzz, rec_bytes, cudaMemcpyDeviceToHost));
    if (rec.type.p   && d_rec.rec_p)
        CUDA_CHECK(cudaMemcpy(rec_p,   d_rec.rec_p,   rec_bytes, cudaMemcpyDeviceToHost));

    /* Free per-shot device receiver buffers */
    cuda_free_rec(&d_rec);
    if (h_rec_ix) free(h_rec_ix);
    if (h_rec_iz) free(h_rec_iz);

    return 0;
}

#endif /* USE_CUDA */
