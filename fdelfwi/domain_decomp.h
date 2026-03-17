/*********************************************************************
 *
 * domain_decomp.h — MPI domain decomposition for multi-GPU FWI
 *
 * Two-level parallelism:
 *   Level 1 (domain_comm): 1D domain decomposition along X-axis
 *           — ndom GPUs share one shot via halo exchange every time step
 *   Level 2 (shot_comm):   Shot parallelism
 *           — nshot_groups of GPUs process different shots concurrently
 *
 * Total MPI ranks = ndom × nshot_groups
 *
 * Author: FWI CUDA porting project
 *
 *********************************************************************/

#ifndef DOMAIN_DECOMP_H
#define DOMAIN_DECOMP_H

#ifdef USE_CUDA
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "fdelfwi.h"

/*====================================================================
 * 1. DOMAIN DECOMPOSITION STRUCTURES
 *====================================================================*/

/* Per-rank subdomain layout (1D decomposition along X) */
typedef struct _domainPar {
    /* --- MPI topology --- */
    int world_rank;         /* rank in MPI_COMM_WORLD */
    int world_size;         /* total MPI ranks */
    int ndom;               /* GPUs per shot (domain decomp factor) */
    int nshot_groups;       /* = world_size / ndom */
    int domain_rank;        /* position within domain (0..ndom-1) */
    int shot_group_id;      /* which shot group (0..nshot_groups-1) */

#ifdef USE_MPI
    MPI_Comm domain_comm;   /* ranks sharing one shot (halo exchange) */
    MPI_Comm shot_comm;     /* one rank per shot group (gradient reduce) */
#endif

    /* --- Local subdomain (X-axis partitioning) --- */
    int nx_global;          /* global interior nx (from modPar.nx) */
    int nz_global;          /* global interior nz (from modPar.nz) = local nz */
    int nx_local;           /* local interior nx (this rank's slice) */
    int ix_global_start;    /* global x-offset of this rank's first interior column */

    /* --- Padded local array dimensions --- */
    int nax_local;          /* padded local x-dimension (pad_left + nx_local + pad_right) */
    int naz_local;          /* = global naz (full depth, unchanged) */
    int pad_left;           /* left padding: ioPx if rank==0, else halo */
    int pad_right;          /* right padding: (nax-iePx) if rank==ndom-1, else halo */
    int halo;               /* stencil radius: iorder/2 */

    /* --- Boundary flags per domain rank --- */
    int has_physical_left;  /* 1 if domain_rank == 0 */
    int has_physical_right; /* 1 if domain_rank == ndom-1 */
    int left_neighbor;      /* MPI rank of left neighbor (-1 if none) */
    int right_neighbor;     /* MPI rank of right neighbor (-1 if none) */

    /* --- Staggered grid offsets (local, mirrors modPar io*/ie* fields) --- */
    int ioXx, ioXz, ieXx, ieXz;   /* Vx grid bounds */
    int ioZx, ioZz, ieZx, ieZz;   /* Vz grid bounds */
    int ioPx, ioPz, iePx, iePz;   /* P/Txx/Tzz grid bounds */
    int ioTx, ioTz, ieTx, ieTz;   /* Txz grid bounds */
} domainPar;

/* Halo exchange buffer set (device pointers) */
typedef struct _haloBuf {
    float *send_left;       /* [nfields * halo * naz] */
    float *send_right;
    float *recv_left;
    float *recv_right;
    int    nfields;         /* 2 for velocity phase, 3 for stress phase */
    int    halo;
    int    naz;
    size_t buf_size;        /* bytes per direction per field */
} haloBuf;

/*====================================================================
 * 2. INITIALIZATION AND TEARDOWN
 *====================================================================*/

/*
 * domain_decomp_init — Set up two-level MPI communicators and local subdomain.
 *
 * Call AFTER MPI_Init and getParameters.
 * Reads ndom from command line (default 1 = no domain decomp).
 *
 * Parameters:
 *   dom   [out]  — filled with all decomposition info
 *   mod   [in]   — global model parameters (nx, nz, naz, nax, ioPx, etc.)
 *   bnd   [in]   — boundary parameters (for padding calculation)
 *   ndom  [in]   — number of GPUs per shot (1 = single GPU per shot)
 */
void domain_decomp_init(domainPar *dom, const modPar *mod,
                         const bndPar *bnd, int ndom);

/*
 * domain_decomp_free — Release MPI communicators and halo buffers.
 */
void domain_decomp_free(domainPar *dom);

/*====================================================================
 * 3. LOCAL STAGGER OFFSET COMPUTATION
 *====================================================================*/

/*
 * domain_compute_stagger_offsets — Compute local io*/ie* bounds
 *
 * Maps global stagger offsets to local subdomain coordinates.
 * PML offsets only apply on ranks with physical boundaries.
 */
void domain_compute_stagger_offsets(domainPar *dom, const modPar *mod,
                                     const bndPar *bnd);

/*====================================================================
 * 4. HALO EXCHANGE
 *====================================================================*/

/*
 * halo_buf_create — Allocate device-side halo send/recv buffers.
 *
 * Parameters:
 *   hbuf      [out]  — halo buffer struct
 *   nfields   [in]   — max fields per exchange (2 for vel, 3 for stress)
 *   halo      [in]   — stencil radius (= iorder/2)
 *   naz       [in]   — depth dimension
 */
void halo_buf_create(haloBuf *hbuf, int nfields, int halo, int naz);

/*
 * halo_buf_destroy — Free device halo buffers.
 */
void halo_buf_destroy(haloBuf *hbuf);

/*
 * halo_exchange_velocity — Exchange vx, vz halos between domain neighbors.
 *
 * Called AFTER velocity update kernel on interior, BEFORE border kernel.
 * Uses GPU-aware MPI if available, otherwise stages through pinned host.
 *
 * Parameters:
 *   d_vx, d_vz  [in/out] — device wavefield pointers (local subdomain)
 *   hbuf        [in]     — pre-allocated halo buffers
 *   dom         [in]     — domain decomposition info
 *   stream      [in]     — CUDA stream for pack/unpack kernels
 */
void halo_exchange_velocity(float *d_vx, float *d_vz,
                             haloBuf *hbuf, const domainPar *dom,
                             cudaStream_t stream);

/*
 * halo_exchange_stress — Exchange txx, tzz, txz halos.
 *
 * Called AFTER stress update kernel on interior, BEFORE border kernel.
 */
void halo_exchange_stress(float *d_txx, float *d_tzz, float *d_txz,
                           haloBuf *hbuf, const domainPar *dom,
                           cudaStream_t stream);

/*====================================================================
 * 5. SOURCE / RECEIVER MAPPING
 *====================================================================*/

/*
 * domain_owns_source — Check if source at global ix falls on this rank.
 *
 * Returns local ix (including left padding) or -1 if not owned.
 */
int domain_owns_source(const domainPar *dom, int src_ix_global);

/*
 * domain_map_receivers — Build local receiver list for this rank.
 *
 * Parameters:
 *   rec             [in]     — global receiver parameters
 *   dom             [in]     — domain decomposition info
 *   nrec_local      [out]    — number of receivers on this rank
 *   local_rec_ix    [out]    — local x-indices (allocated by caller, size rec.n)
 *   local_rec_iz    [out]    — local z-indices (same)
 *   local_rec_map   [out]    — mapping from local index to global index
 *
 * Returns the number of local receivers.
 */
int domain_map_receivers(const recPar *rec, const domainPar *dom,
                          int *nrec_local, int *local_rec_ix,
                          int *local_rec_iz, int *local_rec_map);

/*====================================================================
 * 6. MODEL AND GRADIENT SCATTER / GATHER
 *====================================================================*/

/*
 * domain_scatter_model — Distribute global model arrays to local subdomains.
 *
 * Global rank 0 has full model; scatters nx_local columns to each rank.
 * Each rank stores its local slice plus halo/PML padding.
 *
 * Parameters:
 *   local_field  [out]  — local padded array [nax_local × naz]
 *   global_field [in]   — full model array [nax × naz] (only valid on rank 0)
 *   dom          [in]   — domain decomposition info
 *   mod          [in]   — global model parameters (nax, naz, ioPx, etc.)
 */
void domain_scatter_model(float *local_field, const float *global_field,
                           const domainPar *dom, const modPar *mod);

/*
 * domain_gather_gradient — Assemble local gradient slices into global gradient.
 *
 * Rank 0 receives the full gradient for the optimizer.
 */
void domain_gather_gradient(float *global_grad, const float *local_grad,
                             const domainPar *dom, const modPar *mod);

/*
 * domain_reduce_gradient_shots — MPI_Allreduce gradients across shot groups.
 *
 * Each domain rank reduces with its counterpart in other shot groups.
 */
void domain_reduce_gradient_shots(float *grad, int size, const domainPar *dom);

/*
 * domain_broadcast_model — Broadcast updated model from rank 0 to all ranks.
 */
void domain_broadcast_model(float *model, int size, const domainPar *dom);

/*====================================================================
 * 7. CHECKPOINT MANAGEMENT (LOCAL SUBDOMAIN)
 *====================================================================*/

/*
 * domain_checkpoint_size — Bytes per checkpoint snapshot for local subdomain.
 *
 * Returns: 5 × nax_local × naz × sizeof(float) for elastic.
 */
size_t domain_checkpoint_size(const domainPar *dom, int ischeme);

#endif /* USE_CUDA */
#endif /* DOMAIN_DECOMP_H */
