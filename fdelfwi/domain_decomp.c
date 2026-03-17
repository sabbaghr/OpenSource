/*********************************************************************
 *
 * domain_decomp.c — MPI domain decomposition for multi-GPU FWI
 *
 * Implements two-level MPI parallelism:
 *   Level 1: 1D domain decomposition along X (halo exchange per time step)
 *   Level 2: Shot-parallel groups (gradient reduction per FWI iteration)
 *
 * See domain_decomp.h for full API documentation.
 *
 *********************************************************************/

#ifdef USE_CUDA

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "domain_decomp.h"
#include "cuda_utils.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

/*====================================================================
 * domain_decomp_init
 *====================================================================*/

void domain_decomp_init(domainPar *dom, const modPar *mod,
                         const bndPar *bnd, int ndom)
{
    memset(dom, 0, sizeof(domainPar));
    dom->ndom = ndom;
    dom->nx_global = mod->nx;
    dom->nz_global = mod->nz;
    dom->naz_local = mod->naz;  /* full depth on every rank */
    dom->halo = mod->iorder / 2;

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &dom->world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &dom->world_size);
#else
    dom->world_rank = 0;
    dom->world_size = 1;
#endif

    if (dom->world_size % ndom != 0) {
        if (dom->world_rank == 0)
            fprintf(stderr, "ERROR: world_size (%d) must be divisible by ndom (%d)\n",
                    dom->world_size, ndom);
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        exit(EXIT_FAILURE);
#endif
    }

    dom->nshot_groups  = dom->world_size / ndom;
    dom->shot_group_id = dom->world_rank / ndom;
    dom->domain_rank   = dom->world_rank % ndom;

    /* --- Create communicators --- */
#ifdef USE_MPI
    /* domain_comm: ranks that share one shot (for halo exchange) */
    MPI_Comm_split(MPI_COMM_WORLD, dom->shot_group_id,
                   dom->domain_rank, &dom->domain_comm);

    /* shot_comm: one rank per shot group at same domain position */
    MPI_Comm_split(MPI_COMM_WORLD, dom->domain_rank,
                   dom->shot_group_id, &dom->shot_comm);
#endif

    /* --- Partition global nx across ndom ranks --- */
    int nx_base = dom->nx_global / ndom;
    int nx_rem  = dom->nx_global % ndom;

    dom->nx_local = nx_base + (dom->domain_rank < nx_rem ? 1 : 0);
    dom->ix_global_start = dom->domain_rank * nx_base
                         + (dom->domain_rank < nx_rem ? dom->domain_rank : nx_rem);

    /* --- Boundary flags --- */
    dom->has_physical_left  = (dom->domain_rank == 0);
    dom->has_physical_right = (dom->domain_rank == ndom - 1);
    dom->left_neighbor  = dom->has_physical_left  ? -1 : dom->domain_rank - 1;
    dom->right_neighbor = dom->has_physical_right ? -1 : dom->domain_rank + 1;

    /* --- Padded local array dimensions --- */
    /* Left padding: PML + stagger padding if leftmost rank, else just halo ghost */
    dom->pad_left  = dom->has_physical_left  ? mod->ioPx : dom->halo;
    /* Right padding: PML + stagger padding if rightmost rank, else just halo ghost */
    dom->pad_right = dom->has_physical_right ? (mod->nax - mod->iePx) : dom->halo;
    dom->nax_local = dom->pad_left + dom->nx_local + dom->pad_right;

    /* --- Compute local stagger offsets --- */
    domain_compute_stagger_offsets(dom, mod, bnd);

    /* --- Bind GPU --- */
    int dev = cuda_bind_gpu(dom->domain_rank);
    cuda_print_device_info(dev, (dom->world_rank == 0) ? 1 : 0);

    /* --- Diagnostic output --- */
    if (dom->world_rank == 0) {
        fprintf(stderr, "Domain decomposition: ndom=%d, nshot_groups=%d, "
                "nx_global=%d, halo=%d\n",
                ndom, dom->nshot_groups, dom->nx_global, dom->halo);
    }
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    fprintf(stderr, "  rank %d: shot_group=%d, domain_rank=%d, "
            "nx_local=%d, ix_global_start=%d, nax_local=%d, "
            "pad_left=%d, pad_right=%d, phys_L=%d, phys_R=%d\n",
            dom->world_rank, dom->shot_group_id, dom->domain_rank,
            dom->nx_local, dom->ix_global_start, dom->nax_local,
            dom->pad_left, dom->pad_right,
            dom->has_physical_left, dom->has_physical_right);
}

/*====================================================================
 * domain_compute_stagger_offsets
 *
 * Map global stagger offsets to local subdomain coordinates.
 *
 * The key insight: for interior domain ranks (no physical left/right boundary),
 * the stagger offsets all start at pad_left (=halo) and end at
 * pad_left + nx_local (possibly +1 for extra Vx/Txz columns).
 *
 * For boundary ranks, the original ioPx/iePx offsets apply on the
 * physical side, and halo on the interior side.
 *====================================================================*/

void domain_compute_stagger_offsets(domainPar *dom, const modPar *mod,
                                     const bndPar *bnd)
{
    int halo = dom->halo;
    int pl = dom->pad_left;
    int nx = dom->nx_local;

    /* P-grid (Txx, Tzz, l2m, lam): same as forward mod offsets */
    dom->ioPx = pl;
    dom->iePx = pl + nx;
    dom->ioPz = mod->ioPz;   /* z-offsets unchanged (no z-decomposition) */
    dom->iePz = mod->iePz;

    /* Vx grid: starts 1 column left of P-grid, extends 1 right */
    dom->ioXx = pl;
    dom->ieXx = pl + nx + (dom->has_physical_right ? 1 : 0);
    dom->ioXz = mod->ioXz;
    dom->ieXz = mod->ieXz;

    /* Vz grid */
    dom->ioZx = pl;
    dom->ieZx = pl + nx;
    dom->ioZz = mod->ioZz;
    dom->ieZz = mod->ieZz;

    /* Txz grid: extra column left, extra row up from P grid */
    dom->ioTx = pl;
    dom->ieTx = pl + nx + (dom->has_physical_right ? 1 : 0);
    dom->ioTz = mod->ioTz;
    dom->ieTz = mod->ieTz;

    /*
     * Boundary rank adjustments:
     * - Leftmost rank: stagger offsets match the global ioPx, ioXx, etc.
     *   These are already set correctly because pad_left = mod->ioPx.
     * - Rightmost rank: ie* offsets match the global ie*
     *   Already correct because pad_right = (mod->nax - mod->iePx).
     *   We need ieXx and ieTx to include the extra column.
     */

    /* For leftmost rank with PML on left: stagger offsets may need to
     * account for ntap (already embedded in mod->ioPx, etc.) */
    if (dom->has_physical_left) {
        dom->ioXx = mod->ioXx;
        dom->ioZx = mod->ioXx;  /* Vz starts at same x as Vx */
        dom->ioPx = mod->ioPx;
        dom->ioTx = mod->ioTx;
    }

    if (dom->has_physical_right) {
        /* ie* for rightmost rank mirrors global ie* relative to local nax */
        dom->ieXx = dom->nax_local - (mod->nax - mod->ieXx);
        dom->ieZx = dom->nax_local - (mod->nax - mod->ieZx);
        dom->iePx = dom->nax_local - (mod->nax - mod->iePx);
        dom->ieTx = dom->nax_local - (mod->nax - mod->ieTx);
    }
}

/*====================================================================
 * domain_decomp_free
 *====================================================================*/

void domain_decomp_free(domainPar *dom)
{
#ifdef USE_MPI
    if (dom->ndom > 1) {
        MPI_Comm_free(&dom->domain_comm);
        MPI_Comm_free(&dom->shot_comm);
    }
#endif
    memset(dom, 0, sizeof(domainPar));
}

/*====================================================================
 * halo_buf_create / destroy
 *====================================================================*/

void halo_buf_create(haloBuf *hbuf, int nfields, int halo, int naz)
{
    hbuf->nfields  = nfields;
    hbuf->halo     = halo;
    hbuf->naz      = naz;
    hbuf->buf_size = (size_t)nfields * halo * naz * sizeof(float);

    CUDA_CHECK(cudaMalloc(&hbuf->send_left,  hbuf->buf_size));
    CUDA_CHECK(cudaMalloc(&hbuf->send_right, hbuf->buf_size));
    CUDA_CHECK(cudaMalloc(&hbuf->recv_left,  hbuf->buf_size));
    CUDA_CHECK(cudaMalloc(&hbuf->recv_right, hbuf->buf_size));
}

void halo_buf_destroy(haloBuf *hbuf)
{
    if (hbuf->send_left)  CUDA_CHECK(cudaFree(hbuf->send_left));
    if (hbuf->send_right) CUDA_CHECK(cudaFree(hbuf->send_right));
    if (hbuf->recv_left)  CUDA_CHECK(cudaFree(hbuf->recv_left));
    if (hbuf->recv_right) CUDA_CHECK(cudaFree(hbuf->recv_right));
    memset(hbuf, 0, sizeof(haloBuf));
}

/*====================================================================
 * Halo pack/unpack CUDA kernels (called from .c via extern linkage)
 *
 * These are simple 1D kernels that copy halo columns to/from
 * contiguous send/recv buffers.
 *
 * For now, these are CPU implementations that call cudaMemcpy2D.
 * In Step 3, they will become proper CUDA kernels for async operation.
 *====================================================================*/

/*
 * Pack halo columns from a device field into a contiguous buffer.
 *
 * field: device pointer [nax_local × naz], row-major field[ix*naz+iz]
 * buf:   device pointer [halo × naz], contiguous
 * ix_start: first column to pack
 * halo: number of columns
 * naz: depth dimension
 */
static void pack_halo_columns(float *buf, const float *field,
                                int ix_start, int halo, int naz)
{
    /* Copy halo contiguous columns: each column is naz floats at stride naz */
    for (int ih = 0; ih < halo; ih++) {
        CUDA_CHECK(cudaMemcpy(buf + ih * naz,
                               field + (ix_start + ih) * naz,
                               naz * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    }
}

static void unpack_halo_columns(float *field, const float *buf,
                                  int ix_start, int halo, int naz)
{
    for (int ih = 0; ih < halo; ih++) {
        CUDA_CHECK(cudaMemcpy(field + (ix_start + ih) * naz,
                               buf + ih * naz,
                               naz * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    }
}

/*====================================================================
 * halo_exchange_velocity
 *
 * Exchange vx, vz halo columns between domain neighbors.
 *====================================================================*/

void halo_exchange_velocity(float *d_vx, float *d_vz,
                             haloBuf *hbuf, const domainPar *dom,
                             cudaStream_t stream)
{
#ifdef USE_MPI
    if (dom->ndom <= 1) return;

    int halo = dom->halo;
    int naz  = dom->naz_local;
    int pl   = dom->pad_left;
    int nx   = dom->nx_local;
    size_t field_halo_bytes = halo * naz * sizeof(float);

    /* Synchronize stream before MPI calls (ensure kernels finished) */
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Pack: send leftward = columns [pl .. pl+halo-1] (first interior columns)
     *       send rightward = columns [pl+nx-halo .. pl+nx-1] (last interior columns)
     * Recv: recv from left → ghost columns [pl-halo .. pl-1]
     *       recv from right → ghost columns [pl+nx .. pl+nx+halo-1] */

    float *fields[2] = { d_vx, d_vz };
    MPI_Request reqs[8];  /* 2 fields × 2 directions × (send + recv) */
    int nreq = 0;

    for (int f = 0; f < 2; f++) {
        float *fld = fields[f];
        float *sL = hbuf->send_left  + f * halo * naz;
        float *sR = hbuf->send_right + f * halo * naz;
        float *rL = hbuf->recv_left  + f * halo * naz;
        float *rR = hbuf->recv_right + f * halo * naz;

        /* Pack left halo (interior columns closest to left boundary) */
        if (dom->left_neighbor >= 0)
            pack_halo_columns(sL, fld, pl, halo, naz);

        /* Pack right halo (interior columns closest to right boundary) */
        if (dom->right_neighbor >= 0)
            pack_halo_columns(sR, fld, pl + nx - halo, halo, naz);

        /* Non-blocking send/recv with GPU-aware MPI pointers */
        if (dom->left_neighbor >= 0) {
            MPI_Isend(sL, halo * naz, MPI_FLOAT, dom->left_neighbor,
                      100 + f, dom->domain_comm, &reqs[nreq++]);
            MPI_Irecv(rL, halo * naz, MPI_FLOAT, dom->left_neighbor,
                      200 + f, dom->domain_comm, &reqs[nreq++]);
        }
        if (dom->right_neighbor >= 0) {
            MPI_Isend(sR, halo * naz, MPI_FLOAT, dom->right_neighbor,
                      200 + f, dom->domain_comm, &reqs[nreq++]);
            MPI_Irecv(rR, halo * naz, MPI_FLOAT, dom->right_neighbor,
                      100 + f, dom->domain_comm, &reqs[nreq++]);
        }
    }

    /* Wait for all exchanges to complete */
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    /* Unpack received halos into ghost columns */
    for (int f = 0; f < 2; f++) {
        float *fld = fields[f];
        float *rL = hbuf->recv_left  + f * halo * naz;
        float *rR = hbuf->recv_right + f * halo * naz;

        if (dom->left_neighbor >= 0)
            unpack_halo_columns(fld, rL, pl - halo, halo, naz);
        if (dom->right_neighbor >= 0)
            unpack_halo_columns(fld, rR, pl + nx, halo, naz);
    }
#endif /* USE_MPI */
}

/*====================================================================
 * halo_exchange_stress
 *
 * Exchange txx, tzz, txz halo columns between domain neighbors.
 *====================================================================*/

void halo_exchange_stress(float *d_txx, float *d_tzz, float *d_txz,
                           haloBuf *hbuf, const domainPar *dom,
                           cudaStream_t stream)
{
#ifdef USE_MPI
    if (dom->ndom <= 1) return;

    int halo = dom->halo;
    int naz  = dom->naz_local;
    int pl   = dom->pad_left;
    int nx   = dom->nx_local;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float *fields[3] = { d_txx, d_tzz, d_txz };
    MPI_Request reqs[12];  /* 3 fields × 2 directions × 2 (send+recv) */
    int nreq = 0;

    for (int f = 0; f < 3; f++) {
        float *fld = fields[f];
        float *sL = hbuf->send_left  + f * halo * naz;
        float *sR = hbuf->send_right + f * halo * naz;
        float *rL = hbuf->recv_left  + f * halo * naz;
        float *rR = hbuf->recv_right + f * halo * naz;

        if (dom->left_neighbor >= 0)
            pack_halo_columns(sL, fld, pl, halo, naz);
        if (dom->right_neighbor >= 0)
            pack_halo_columns(sR, fld, pl + nx - halo, halo, naz);

        if (dom->left_neighbor >= 0) {
            MPI_Isend(sL, halo * naz, MPI_FLOAT, dom->left_neighbor,
                      300 + f, dom->domain_comm, &reqs[nreq++]);
            MPI_Irecv(rL, halo * naz, MPI_FLOAT, dom->left_neighbor,
                      400 + f, dom->domain_comm, &reqs[nreq++]);
        }
        if (dom->right_neighbor >= 0) {
            MPI_Isend(sR, halo * naz, MPI_FLOAT, dom->right_neighbor,
                      400 + f, dom->domain_comm, &reqs[nreq++]);
            MPI_Irecv(rR, halo * naz, MPI_FLOAT, dom->right_neighbor,
                      300 + f, dom->domain_comm, &reqs[nreq++]);
        }
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    for (int f = 0; f < 3; f++) {
        float *fld = fields[f];
        float *rL = hbuf->recv_left  + f * halo * naz;
        float *rR = hbuf->recv_right + f * halo * naz;

        if (dom->left_neighbor >= 0)
            unpack_halo_columns(fld, rL, pl - halo, halo, naz);
        if (dom->right_neighbor >= 0)
            unpack_halo_columns(fld, rR, pl + nx, halo, naz);
    }
#endif /* USE_MPI */
}

/*====================================================================
 * Source / Receiver Mapping
 *====================================================================*/

int domain_owns_source(const domainPar *dom, int src_ix_global)
{
    if (src_ix_global >= dom->ix_global_start &&
        src_ix_global <  dom->ix_global_start + dom->nx_local) {
        return src_ix_global - dom->ix_global_start + dom->pad_left;
    }
    return -1;  /* source not on this rank */
}

int domain_map_receivers(const recPar *rec, const domainPar *dom,
                          int *nrec_local, int *local_rec_ix,
                          int *local_rec_iz, int *local_rec_map)
{
    int count = 0;
    for (int irec = 0; irec < rec->n; irec++) {
        int gx = rec->x[irec];  /* global interior x-index */
        if (gx >= dom->ix_global_start &&
            gx <  dom->ix_global_start + dom->nx_local) {
            local_rec_ix[count]  = gx - dom->ix_global_start + dom->pad_left;
            local_rec_iz[count]  = rec->z[irec];  /* z unchanged */
            local_rec_map[count] = irec;
            count++;
        }
    }
    *nrec_local = count;
    return count;
}

/*====================================================================
 * Model / Gradient Scatter-Gather
 *====================================================================*/

void domain_scatter_model(float *local_field, const float *global_field,
                           const domainPar *dom, const modPar *mod)
{
#ifdef USE_MPI
    /* For now: each rank reads the global model file independently.
     * A proper MPI_Scatterv implementation goes here for production.
     *
     * Each rank extracts its local slice from the global array:
     *   global columns [ioPx + ix_global_start .. ioPx + ix_global_start + nx_local)
     * Plus halo/PML padding on each side. */

    /* Placeholder: assume each rank already has the full model loaded
     * (as in the current CPU code). Copy the relevant slice. */
    int naz = dom->naz_local;
    int ioPx = mod->ioPx;

    /* Copy interior columns */
    for (int ix = 0; ix < dom->nx_local; ix++) {
        int gx = ioPx + dom->ix_global_start + ix;
        int lx = dom->pad_left + ix;
        memcpy(local_field + lx * naz,
               global_field + gx * naz,
               naz * sizeof(float));
    }

    /* Copy left padding (PML or halo from neighbor's data) */
    for (int ix = 0; ix < dom->pad_left; ix++) {
        int gx;
        if (dom->has_physical_left) {
            gx = ix;  /* PML columns from global array start */
        } else {
            gx = ioPx + dom->ix_global_start - dom->pad_left + ix;
        }
        if (gx >= 0 && gx < mod->nax) {
            memcpy(local_field + ix * naz,
                   global_field + gx * naz,
                   naz * sizeof(float));
        }
    }

    /* Copy right padding */
    for (int ix = 0; ix < dom->pad_right; ix++) {
        int lx = dom->pad_left + dom->nx_local + ix;
        int gx;
        if (dom->has_physical_right) {
            gx = ioPx + dom->ix_global_start + dom->nx_local + ix;
        } else {
            gx = ioPx + dom->ix_global_start + dom->nx_local + ix;
        }
        if (gx >= 0 && gx < mod->nax) {
            memcpy(local_field + lx * naz,
                   global_field + gx * naz,
                   naz * sizeof(float));
        }
    }
#else
    /* No MPI: local == global */
    memcpy(local_field, global_field,
           (size_t)dom->nax_local * dom->naz_local * sizeof(float));
#endif
}

void domain_gather_gradient(float *global_grad, const float *local_grad,
                             const domainPar *dom, const modPar *mod)
{
#ifdef USE_MPI
    /* Gather interior-only columns from each rank to rank 0 of domain_comm */
    int naz = dom->naz_local;
    int nz  = dom->nz_global;

    /* Each rank contributes nx_local × nz interior values */
    int send_count = dom->nx_local * nz;

    /* Build displacements on rank 0 */
    int *recvcounts = NULL;
    int *displs = NULL;

    if (dom->domain_rank == 0 && dom->shot_group_id == 0) {
        recvcounts = (int *)malloc(dom->ndom * sizeof(int));
        displs = (int *)malloc(dom->ndom * sizeof(int));

        int offset = 0;
        int nx_base = dom->nx_global / dom->ndom;
        int nx_rem  = dom->nx_global % dom->ndom;
        for (int r = 0; r < dom->ndom; r++) {
            int nx_r = nx_base + (r < nx_rem ? 1 : 0);
            recvcounts[r] = nx_r * nz;
            displs[r] = offset;
            offset += nx_r * nz;
        }
    }

    /* Pack local interior into a contiguous send buffer */
    float *sendbuf = (float *)malloc(send_count * sizeof(float));
    int ioPz = mod->ioPz;
    for (int ix = 0; ix < dom->nx_local; ix++) {
        int lx = dom->pad_left + ix;
        memcpy(sendbuf + ix * nz,
               local_grad + lx * naz + ioPz,
               nz * sizeof(float));
    }

    MPI_Gatherv(sendbuf, send_count, MPI_FLOAT,
                global_grad, recvcounts, displs, MPI_FLOAT,
                0, dom->domain_comm);

    free(sendbuf);
    if (recvcounts) free(recvcounts);
    if (displs) free(displs);
#else
    /* No MPI: copy through */
    memcpy(global_grad, local_grad,
           (size_t)dom->nx_global * dom->nz_global * sizeof(float));
#endif
}

void domain_reduce_gradient_shots(float *grad, int size, const domainPar *dom)
{
#ifdef USE_MPI
    if (dom->nshot_groups > 1) {
        MPI_Allreduce(MPI_IN_PLACE, grad, size, MPI_FLOAT, MPI_SUM,
                      dom->shot_comm);
    }
#endif
}

void domain_broadcast_model(float *model, int size, const domainPar *dom)
{
#ifdef USE_MPI
    /* Broadcast from rank 0 within each shot group */
    MPI_Bcast(model, size, MPI_FLOAT, 0, dom->shot_comm);
#endif
}

/*====================================================================
 * Checkpoint size
 *====================================================================*/

size_t domain_checkpoint_size(const domainPar *dom, int ischeme)
{
    int nfields = (ischeme >= 3) ? 5 : 1;  /* elastic: vx,vz,txx,tzz,txz; acoustic: p */
    return (size_t)nfields * dom->nax_local * dom->naz_local * sizeof(float);
}

#endif /* USE_CUDA */
