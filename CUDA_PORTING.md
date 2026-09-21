# CUDA Porting Project — OpenSource_SL10

## Project Overview
Port 2D elastic seismic modeling (fdelmodc) and Full Waveform Inversion (fdelfwi) from C/OpenMP to CUDA.

---

## STEP 1: CODE REVIEW (COMPLETED)

### Repository Structure
| Component | Files | Role |
|-----------|-------|------|
| fdelmodc/ | ~30 .c | Forward elastic/acoustic FD modeling |
| fdelfwi/ | ~35 .c | FWI: forward, adjoint, gradient, optimization |
| FFTlib/ | FFT lib | libgenfft.a — wavefield decomposition only (not hot path) |
| optimization/ | Fortran/C | liboptim.a — L-BFGS, TRN, NLCG (CPU only) |

### Core Computational Kernels

#### Forward Elastic Stencils (elastic4.c, elastic6.c, elastic8.c)
- Velocity-stress formulation, staggered grid
- 4th order: c1=9/8, c2=-1/24, stencil radius ±2
- 6th order: +c3=3/640, radius ±3
- 8th order: +c4=-5/7168, radius ±4
- Phase 1: velocity update (vx, vz from txx, tzz, txz)
- Phase 2: stress update (txx, tzz, txz from vx, vz)
- **Must synchronize between Phase 1 and Phase 2**

#### Adjoint Elastic Stencils (elastic4_adj.c, etc.)
- True discrete adjoint: material params INSIDE derivatives
- Same stencil structure, different coefficient placement

#### Gradient Cross-Correlation (fwi_gradient.c)
- g_λ, g_μ, g_ρ via zero-lag cross-correlation
- Same stencil derivatives as forward kernel
- Accumulated at every time step

#### PML Boundaries (boundaries.c, boundaries_adj.c)
- Split-field PML with auxiliary arrays
- 4 sides + 4 corners = 8 loop nests
- Polynomial damping σ(d) = σ_max * (d/L)^m

#### Source/Receiver (applySource.c, getRecTimes.c)
- Point injection, bilinear interpolation at receivers
- Multi-component recording (vx, vz, txx, tzz, txz, P)

### Production Grid Sizes (120 km × 12 km domain)
Dispersion criterion for 4th-order: dx ≤ Vs_min / (5 × f_max)

| Vs_min | f_max | dx   | nx     | nz    | Points/array | MB/array |
|:------:|:-----:|:----:|:------:|:-----:|:------------:|:--------:|
| 500    | 10 Hz | 10 m | 12,000 | 1,200 | 14.4M        | 55       |
| 500    | 15 Hz | 6.7m | 18,000 | 1,800 | 32.4M        | 124      |
| 300    | 10 Hz | 6 m  | 20,000 | 2,000 | 40.0M        | 153      |
| 300    | 15 Hz | 4 m  | 30,000 | 3,000 | 90.0M        | 344      |

Memory estimate (dx=5m, 24000×2400, typical production):
- Wavefields (5): 1.1 GB
- Material params (5): 1.1 GB
- Adjoint wavefields (5): 1.1 GB
- Gradients (3): 0.66 GB
- PML aux: ~0.05 GB
- **Resident FWI total: ~4 GB** (fits 8+ GB GPU)
- **Checkpoints (400 snaps): ~220 GB** → MUST go to host/disk

### Memory Layout
- Row-major: field[ix * naz + iz], fast dim = z (depth)
- Inner z-loop: stride-1 (cache-friendly)
- Outer x-loop: stride-naz
- 5 wavefields + 5 material params + PML aux + gradients per shot
- Read-only: rox, roz, l2m, lam, mul
- Read-write: vx, vz, txx, tzz, txz

### Existing Parallelism
- OpenMP: `#pragma omp for schedule(guided,1)` + `#pragma simd` on spatial loops
- MPI: shot-level round-robin, MPI_Allreduce on gradients
- Optimizer: rank 0 only (Fortran liboptim.a)

### Data Flow
```
Forward: readModel → [time loop: elastic4 → boundaries → getRecTimes → writeCheckpoint] → writeRec
Adjoint: For each checkpoint segment (reverse): readCheckpoint → [re-propagate + adjoint + gradient]
FWI iter: forward(+chk) → computeResidual → adjoint(+grad) → MPI_Allreduce → optimizer → update
```

### Checkpoint I/O (primary bottleneck)
- 5 wavefield arrays per snapshot
- Written every skipdt time steps during forward
- Read sequentially during adjoint re-propagation

### External Dependencies (none in hot path)
- MKL/FFTW: post-processing only
- liboptim.a: CPU optimizer
- MPI: shot-level parallelism (keep for multi-GPU)

### Memory Footprint
- Small grid (400×160): ~5 MB/shot — trivial for GPU
- Large grid (800×400): ~40 MB/shot — fits easily
- Checkpoints in VRAM: ~2.5 GB for 400 snapshots — feasible

### Loop Dependencies
- Time loop: sequential (each step depends on t-1)
- Spatial loops: no dependencies within a phase
- Velocity→Stress: barrier required between phases
- PML: time-sequential per boundary point
- Gradient: accumulate += at each time step

### GPU Porting Priorities
| Priority | Kernel | Est. Runtime | Parallelism |
|----------|--------|-------------|-------------|
| 1 | elastic4/6/8 forward | 40-50% | 2D stencil |
| 2 | elastic4_adj adjoint | 25-35% | Same |
| 3 | fwi_gradient | 10-15% | 2D + accumulate |
| 4 | boundaries/PML | 5-10% | 1D strips |
| 5 | source/receiver | <1% | Trivial |

---

## STEP 2: CUDA PORTING PLAN

### 2.1 Kernel Identification (ranked by impact)

| # | CPU Function | CUDA Kernel | Source File | Impact |
|---|-------------|-------------|-------------|--------|
| 1 | elastic4() velocity loop | `update_velocity_kernel` | elastic4.c:80-110 | Very High |
| 2 | elastic4() stress loop | `update_stress_kernel` | elastic4.c:119-158 | Very High |
| 3 | elastic4_adj() vel loop | `adj_update_velocity_kernel` | elastic4_adj.c | Very High |
| 4 | elastic4_adj() stress loop | `adj_update_stress_kernel` | elastic4_adj.c | Very High |
| 5 | accumGradient() λ,μ,ρ | `crosscorr_gradient_kernel` | fwi_gradient.c:210-354 | High |
| 6 | boundaries() PML sides | `pml_velocity_kernel` / `pml_stress_kernel` | boundaries.c:150-400 | Medium |
| 7 | boundaries() PML corners | `pml_corner_kernel` | boundaries.c:400-500 | Medium |
| 8 | boundaries_adj() | `adj_pml_velocity_kernel` / `adj_pml_stress_kernel` | boundaries_adj.c | Medium |
| 9 | applySource() | `inject_source_kernel` | applySource.c | Low |
| 10 | applyAdjointSource() | `inject_adjoint_source_kernel` | applyAdjointSource.c | Low |
| 11 | getRecTimes() | `extract_receivers_kernel` | getRecTimes.c | Low |
| 12 | computeResidual() misfit | `misfit_reduction_kernel` | computeResidual.c | Low |
| 13 | born_vsrc() | `born_virtual_source_kernel` | born_vsrc.c | Low (TRN only) |
| 14 | taperGradient | `gradient_taper_kernel` | fwi_inversion.c | Low |

Also for 6th/8th order: duplicate kernels with wider stencil (template or runtime switch).

### 2.2 Thread/Block Layout Strategy

#### Stencil Kernels (velocity, stress, adjoint, gradient)
```
Block: (BLOCK_X, BLOCK_Z) = (16, 16) → 256 threads/block
Grid:  ((nax - io - ie + BLOCK_X - 1) / BLOCK_X,
        (naz - io - ie + BLOCK_Z - 1) / BLOCK_Z)
```
- Each thread computes ONE grid point (ix, iz)
- ix = blockIdx.x * BLOCK_X + threadIdx.x + ioXx
- iz = blockIdx.y * BLOCK_Z + threadIdx.y + ioXz

#### Shared Memory Tiling (4th order, radius=2)
```
Tile: (BLOCK_X + 2*HALO, BLOCK_Z + 2*HALO) where HALO=2
     = (20, 20) floats = 1600 bytes per field
```
- Load tile from global memory (interior + halo)
- __syncthreads()
- Compute stencil from shared memory
- For 4th order: load 1 field into smem per stencil direction
- Total smem per kernel call: ~6-8 KB (well within 48 KB limit)

#### Warp-Level Considerations
- z-dimension (fast) maps to threadIdx.y within a warp
- With BLOCK_Z=16 and contiguous z-memory, each half-warp accesses 16 consecutive floats → **coalesced** (64 bytes per transaction)
- No warp divergence: all interior threads execute identical code
- Boundary threads handled by separate PML kernels (no divergence in stencil kernel)

#### PML Kernels
```
Block: (1, 256) for side strips (1D along boundary)
Grid:  varies per side — e.g., left PML: (npml, (naz+255)/256)
```
- Thin strips: 1D parallelism along the boundary
- Corner kernels: small 2D grids (npml × npml)

#### Source/Receiver Kernels
```
inject_source_kernel<<<1, nsrc>>> (nsrc typically 1-10)
extract_receivers_kernel<<<(nrec+255)/256, 256>>>
```

### 2.3 Memory Strategy

#### Device Memory Allocation
- **Explicit cudaMalloc** for all persistent arrays (not Unified Memory)
- Rationale: full control over transfers; avoid page-fault overhead; deterministic performance
- Allocate once at program init, reuse across shots and iterations

#### Pinned Host Memory
```c
cudaMallocHost(&h_checkpoint, 5 * nax * naz * sizeof(float));  // checkpoint staging
cudaMallocHost(&h_rec_data, nrec * nt * sizeof(float));         // receiver output
cudaMallocHost(&h_residual, nrec * nt * sizeof(float));         // adjoint source input
```
- Enables cudaMemcpyAsync for overlapped I/O

#### Constant Memory
```c
__constant__ float d_c[4];           // FD stencil coefficients (c1,c2,c3,c4)
__constant__ float d_src_wav[MAX_NT]; // source wavelet (if fits; else global)
__constant__ float d_pml_sigma[MAX_NPML]; // PML damping profile
__constant__ float d_pml_RA[MAX_NPML];    // PML coefficient
```
- FD coefficients: 4 floats — perfect for constant memory (broadcast to all threads)
- Source wavelet: up to 16 KB — fits if nt < 4096

#### Read-Only Global Memory
- Material parameter arrays (l2m, lam, muu, rox, roz): use `const float* __restrict__`
- Compiler will route through read-only cache (L1 texture path on sm_60+)
- Alternatively use `__ldg()` intrinsic explicitly

### 2.4 Data Layout Changes

#### Current Layout (Keep As-Is)
- Already SoA: separate arrays for vx, vz, txx, tzz, txz, l2m, lam, muu, rox, roz
- No AoS→SoA conversion needed
- Column-major `field[ix*naz + iz]` with fast z-stride

#### Padding for Alignment
```c
// Pad naz to multiple of 32 floats (128 bytes) for coalesced access
int naz_padded = ((naz + 31) / 32) * 32;
cudaMalloc(&d_vx, nax * naz_padded * sizeof(float));
```
- Ensures each x-column starts on 128-byte boundary
- Minimal wasted memory (~few KB)

#### Pitch Allocation (Alternative)
```c
cudaMallocPitch(&d_vx, &pitch, naz * sizeof(float), nax);
```
- Let CUDA runtime choose optimal row pitch
- Requires pitch-aware indexing: `d_vx[ix * (pitch/sizeof(float)) + iz]`
- Decision: use manual padding (simpler, same effect)

### 2.5 Two-Level MPI+CUDA Parallelism (Domain Decomposition + Shot Parallelism)

This is the central architectural change. The system uses **two orthogonal MPI communicator levels**:

```
Total MPI ranks = ndom (GPUs per shot) × nshot_groups (concurrent shots)

Example: 4 GPUs/shot × 10 shot groups = 40 MPI ranks, 40 GPUs
         Shot group 0: ranks 0-3   → shot 0, 10, 20, ...
         Shot group 1: ranks 4-7   → shot 1, 11, 21, ...
         ...
         Shot group 9: ranks 36-39 → shot 9, 19, 29, ...
```

#### 2.5.1 MPI Communicator Setup

```c
// Input parameters (command line or config)
int ndom;           // GPUs per shot (domain decomposition factor)
int world_size;     // total MPI ranks = ndom × nshot_groups
int nshot_groups;   // = world_size / ndom

// Derived communicators
int shot_group_id = world_rank / ndom;       // which shot group (0..nshot_groups-1)
int domain_rank   = world_rank % ndom;       // position within domain (0..ndom-1)

MPI_Comm domain_comm;   // ranks that share ONE shot (halo exchange)
MPI_Comm shot_comm;     // one rank per shot group (gradient reduction)

MPI_Comm_split(MPI_COMM_WORLD, shot_group_id, domain_rank, &domain_comm);
MPI_Comm_split(MPI_COMM_WORLD, domain_rank,   shot_group_id, &shot_comm);
```

**Communication patterns:**
| Communicator | When | What | Frequency |
|:---|:---|:---|:---|
| `domain_comm` | Every time step | Halo exchange (wavefield boundaries) | nt times per shot |
| `shot_comm` | After all shots | Gradient `MPI_Allreduce` + misfit sum | 1× per FWI iteration |
| `MPI_COMM_WORLD` | After reduction | `MPI_Bcast` model update from global rank 0 | 1× per FWI iteration |

#### 2.5.2 Domain Decomposition Along X-Axis (1D Slicing)

**Why X only:**
- X is the long axis (120 km → 12,000-30,000 points)
- Z is the short axis (12 km → 1,200-3,000 points) — fits in one GPU
- 1D decomposition is simpler (one neighbor left, one right)
- 2D decomposition not needed unless Z exceeds ~8,000 points

**Subdomain sizing:**
```c
// Global grid: nx_global (interior points, no padding)
int nx_local_base = nx_global / ndom;
int nx_remainder  = nx_global % ndom;

// Rank r gets nx_local = nx_local_base + (r < nx_remainder ? 1 : 0)
int nx_local = nx_local_base + (domain_rank < nx_remainder ? 1 : 0);

// Global x-offset for this rank's subdomain
int ix_global_start = domain_rank * nx_local_base
                    + (domain_rank < nx_remainder ? domain_rank : nx_remainder);
```

**Padded local arrays:**
```c
int halo = iorder / 2;  // stencil radius: 2 (4th), 3 (6th), 4 (8th order)

// Left boundary: PML if domain_rank == 0, else halo ghost zone
// Right boundary: PML if domain_rank == ndom-1, else halo ghost zone
int pad_left  = (domain_rank == 0)        ? mod.ioPx : halo;
int pad_right = (domain_rank == ndom - 1) ? (mod.nax - mod.iePx) : halo;

int nax_local = pad_left + nx_local + pad_right;
// naz unchanged — full depth on every rank
```

**Memory per GPU (production: nx_global=24000, nz=2400, ndom=4):**
```
nx_local = 6000, nax_local ≈ 6004 (with halo), naz = 2400+padding ≈ 2432
Per array: 6004 × 2432 × 4 bytes ≈ 56 MB
Wavefields (5):      280 MB
Material params (5): 280 MB
Adjoint wflds (5):   280 MB
Gradients (3):       168 MB
PML aux:             ~10 MB
Total resident:      ~1.0 GB  ← fits easily in any modern GPU
```

#### 2.5.3 Boundary Condition Assignment

Each domain rank handles different boundaries:

| Domain rank | Left BC | Right BC | Top BC | Bottom BC |
|:-----------:|:-------:|:--------:|:------:|:---------:|
| 0 (leftmost) | PML/absorbing (physical) | **Halo exchange** | free surface / PML | PML |
| 1..ndom-2 (interior) | **Halo exchange** | **Halo exchange** | free surface / PML | PML |
| ndom-1 (rightmost) | **Halo exchange** | PML/absorbing (physical) | free surface / PML | PML |

**Implementation in boundaries kernel:**
```c
typedef struct {
    int has_physical_left;   // domain_rank == 0
    int has_physical_right;  // domain_rank == ndom-1
    int left_bc_type;        // PML=4, absorbing=2, or HALO=0 (no BC, just exchange)
    int right_bc_type;       // same
    int top_bc_type;         // always from user (free surface=1, PML=4)
    int bot_bc_type;         // always from user (PML=4)
} domainBndPar;
```

- PML kernels only run on physical boundaries (rank 0 left, rank ndom-1 right)
- Top/bottom PML runs on all ranks (full depth per rank)

#### 2.5.4 Halo Exchange Protocol

For each time step, after updating velocity (Phase 1) and before stress update (Phase 2), and vice versa:

```
4th order stencil: halo = 2 grid columns in x-direction
Each column: naz floats
Transfer size per field: 2 × naz × sizeof(float)
For 5 fields (vx, vz, txx, tzz, txz): 10 × naz × 4 bytes
At naz=2432: 10 × 2432 × 4 = 95 KB per exchange — TINY
```

**Two exchanges per time step:**
1. After velocity update: exchange vx, vz halos → needed by stress update
2. After stress update: exchange txx, tzz, txz halos → needed by next velocity update

**Overlap halo exchange with interior computation:**
```c
// Phase 1: Velocity update
// Step A: Launch interior kernel (no halo dependency)
update_velocity_kernel<<<interior_grid, block, smem, stream_compute>>>(
    d_vx, d_vz, ..., ix_start=halo, ix_end=nx_local-halo, ...);

// Step B: Pack halo → send/recv via GPU-aware MPI (async)
// Left halo: columns [halo .. 2*halo-1] → send to left neighbor
// Right halo: columns [nx_local-2*halo .. nx_local-halo-1] → send to right neighbor
pack_halo_kernel<<<>>>(d_vx, d_halo_send_L, d_halo_send_R, ...);
cudaStreamSynchronize(stream_halo);
MPI_Isend(d_halo_send_L, ..., domain_rank-1, domain_comm, &req_send_L);
MPI_Isend(d_halo_send_R, ..., domain_rank+1, domain_comm, &req_send_R);
MPI_Irecv(d_halo_recv_L, ..., domain_rank-1, domain_comm, &req_recv_L);
MPI_Irecv(d_halo_recv_R, ..., domain_rank+1, domain_comm, &req_recv_R);

// Step C: Wait for halos, unpack
MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
unpack_halo_kernel<<<>>>(d_vx, d_halo_recv_L, d_halo_recv_R, ...);

// Step D: Launch border kernel (uses received halos)
update_velocity_kernel<<<border_grid, block, smem, stream_compute>>>(
    d_vx, d_vz, ..., ix_start=0, ix_end=halo, ...);  // left border
update_velocity_kernel<<<border_grid, block, smem, stream_compute>>>(
    d_vx, d_vz, ..., ix_start=nx_local-halo, ix_end=nx_local, ...);  // right border
```

**GPU-Aware MPI:**
- Use CUDA-aware MPI (OpenMPI + UCX, or MVAPICH2-GDR) to send/recv directly from device pointers
- Avoids D2H→MPI→H2D roundtrip — uses GPUDirect RDMA on NVLink/InfiniBand
- Fallback: explicit pack to pinned host buffer if GPU-aware MPI unavailable

#### 2.5.5 Source and Receiver Mapping to Subdomains

Sources and receivers have global x-positions. Each domain rank must determine which ones fall within its local subdomain.

```c
// Source at global grid position src_ix_global:
if (src_ix_global >= ix_global_start &&
    src_ix_global <  ix_global_start + nx_local) {
    // This source is on my subdomain
    int src_ix_local = src_ix_global - ix_global_start + pad_left;
    inject_source_kernel<<<1,1>>>(d_txx, d_tzz, src_ix_local, src_iz, ...);
}
// else: skip — another rank handles this source

// Same logic for receivers:
int nrec_local = 0;
for (int irec = 0; irec < nrec_global; irec++) {
    if (rec_ix_global[irec] >= ix_global_start &&
        rec_ix_global[irec] <  ix_global_start + nx_local) {
        local_rec_map[nrec_local] = irec;
        local_rec_ix[nrec_local] = rec_ix_global[irec] - ix_global_start + pad_left;
        nrec_local++;
    }
}
// After forward: MPI_Gatherv receiver traces to rank that writes .su files
```

#### 2.5.6 Shot Distribution Across Shot Groups

```c
// Each shot group processes shots round-robin:
for (int ishot = shot_group_id; ishot < nshots; ishot += nshot_groups) {
    // All ndom ranks in this shot group collaborate on this shot:
    // 1. Forward pass (domain-decomposed, halo exchange via domain_comm)
    // 2. Compute residual (rank with receivers, or gather first)
    // 3. Adjoint pass (domain-decomposed, halo exchange via domain_comm)
    // 4. Accumulate gradient locally
}

// After all shots: reduce gradients across shot groups
// Each domain_rank gathers its local gradient slice
MPI_Allreduce(local_grad, global_grad, nx_local*naz, MPI_FLOAT, MPI_SUM, shot_comm);
```

### 2.6 Wavefield Time-Stepping and Checkpointing

#### Time-Stepping on GPU (per domain rank)
- **No swap needed**: velocity-stress formulation updates arrays in-place
- Each time step = 2 kernel launches + 2 halo exchanges:
  1. `update_velocity_kernel` → halo exchange (vx, vz)
  2. `update_stress_kernel` → halo exchange (txx, tzz, txz)
- Halo exchanges overlap with interior computation (see 2.5.4)

#### FWI Checkpointing Strategy (Production Scale)

At production scale, checkpoints CANNOT fit in VRAM. Strategy:

**Primary: Stream checkpoints to pinned host memory**
```c
// Each domain rank checkpoints its LOCAL subdomain only
size_t chk_size = 5 * nax_local * naz * sizeof(float);  // per snapshot

// Forward: async D2H every skipdt steps
if (it % skipdt == 0) {
    int isnap = it / skipdt;
    float *h_dst = h_checkpoint_buf + isnap * 5 * nax_local * naz;
    cudaMemcpyAsync(h_dst,              d_vx,  field_bytes, cudaMemcpyDeviceToHost, stream_io);
    cudaMemcpyAsync(h_dst + 1*fld_len,  d_vz,  field_bytes, cudaMemcpyDeviceToHost, stream_io);
    cudaMemcpyAsync(h_dst + 2*fld_len,  d_txx, field_bytes, cudaMemcpyDeviceToHost, stream_io);
    cudaMemcpyAsync(h_dst + 3*fld_len,  d_tzz, field_bytes, cudaMemcpyDeviceToHost, stream_io);
    cudaMemcpyAsync(h_dst + 4*fld_len,  d_txz, field_bytes, cudaMemcpyDeviceToHost, stream_io);
}
```

**Memory budget per GPU node (example: dx=5m, ndom=4):**
```
nax_local ≈ 6004, naz ≈ 2432
Per snapshot: 5 × 6004 × 2432 × 4 = 280 MB
400 snapshots: 112 GB in host RAM per GPU
Typical node: 256-512 GB RAM, 4 GPUs → 64-128 GB per GPU → fits ~230-460 snapshots
```

**Fallback for very large grids or many snapshots:**
- Spill oldest checkpoints to NVMe SSD (async I/O thread)
- Or use optimal checkpointing (Griewank/Walther revolve algorithm): store O(√nt) snapshots, recompute segments

**Adjoint: load checkpoints from host → GPU**
```c
for (iseg = nsnap-1; iseg >= 0; iseg--) {
    float *h_src = h_checkpoint_buf + iseg * 5 * nax_local * naz;
    cudaMemcpyAsync(d_vx,  h_src,              field_bytes, cudaMemcpyHostToDevice, stream_io);
    cudaMemcpyAsync(d_vz,  h_src + 1*fld_len,  field_bytes, cudaMemcpyHostToDevice, stream_io);
    // ... wait, then re-propagate segment forward while computing adjoint
}
```

#### Memory Footprint per GPU (FWI, production)
```
Resident device memory:
  Fwd wavefields (5):    5 × nax_local × naz × 4     ≈ 280 MB
  Material params (5):   5 × nax_local × naz × 4     ≈ 280 MB
  Adj wavefields (5):    5 × nax_local × naz × 4     ≈ 280 MB
  Gradients (3):         3 × nax_local × naz × 4     ≈ 168 MB
  PML aux:               ~10 MB (only on boundary ranks)
  Halo buffers:          ~1 MB (tiny)
  Total device:          ~1.0 GB per GPU  ✓

Host pinned memory (checkpoints):
  nsnap × 5 × nax_local × naz × 4 ≈ 112 GB (400 snaps)
  Fits in 128-256 GB node RAM       ✓
```

### 2.7 FWI-Specific CUDA Design (Domain-Decomposed)

#### Forward Pass (per shot, all ndom ranks collaborate)
```c
// Each domain rank executes:
for (int it = 0; it < nt; it++) {
    // --- Phase 1: Velocity update ---
    update_velocity_kernel<<<interior_grid, block, smem, stream_compute>>>(
        d_vx, d_vz, d_txx, d_tzz, d_txz, d_rox, d_roz, ...);
    halo_exchange_async(d_vx, d_vz, domain_comm, stream_halo);  // 2 fields
    update_velocity_kernel<<<border_grid, block, smem, stream_compute>>>(...);  // border
    if (dom_bnd.has_physical_left || dom_bnd.has_physical_right)
        pml_velocity_kernel<<<pml_grid, pml_block>>>(d_vx, d_vz, ...);
    pml_velocity_kernel_topbot<<<>>>(...);  // top/bot PML on all ranks

    // --- Phase 2: Stress update ---
    update_stress_kernel<<<interior_grid, block, smem, stream_compute>>>(
        d_vx, d_vz, d_txx, d_tzz, d_txz, d_l2m, d_lam, d_muu, ...);
    halo_exchange_async(d_txx, d_tzz, d_txz, domain_comm, stream_halo);  // 3 fields
    update_stress_kernel<<<border_grid, block, smem, stream_compute>>>(...);
    if (dom_bnd.has_physical_left || dom_bnd.has_physical_right)
        pml_stress_kernel<<<pml_grid, pml_block>>>(d_txx, d_tzz, d_txz, ...);
    pml_stress_kernel_topbot<<<>>>(...);

    // --- Source injection (only on rank that owns the source) ---
    if (i_own_source)
        inject_source_kernel<<<1,1>>>(d_txx, d_tzz, src_ix_local, ...);

    // --- Receiver extraction (only for local receivers) ---
    if (nrec_local > 0)
        extract_receivers_kernel<<<rec_grid,256>>>(d_vx, d_vz, ..., d_rec, it);

    // --- Checkpoint (async to host) ---
    if (it % skipdt == 0)
        checkpoint_async_d2h(it / skipdt, stream_io);
}

// Gather receiver traces from all domain ranks
MPI_Gatherv(d_rec_local, ..., rec_buf_global, ..., 0, domain_comm);
// Domain rank 0 writes .su file
```

#### Adjoint Pass (per shot, domain-decomposed)
```c
for (int iseg = nsnap-1; iseg >= 0; iseg--) {
    checkpoint_async_h2d(iseg, stream_io);  // load fwd checkpoint
    cudaStreamSynchronize(stream_io);

    for (int it = chk_time[iseg+1]-1; it >= chk_time[iseg]; it--) {
        // Re-propagate forward (same domain-decomposed kernels + halo exchange)
        forward_one_step(d_fwd_vx, d_fwd_vz, d_fwd_txx, ..., domain_comm);

        // Adjoint propagation (domain-decomposed, same halo pattern)
        adjoint_one_step(d_adj_vx, d_adj_vz, d_adj_txx, ..., domain_comm);

        // Inject adjoint source (only on rank with receivers)
        if (nrec_local > 0)
            inject_adjoint_source_kernel<<<>>>(d_adj_txx, ..., residuals, it);

        // Cross-correlation gradient (local subdomain only)
        crosscorr_gradient_kernel<<<grid, block>>>(
            d_fwd_vx, d_fwd_vz, d_adj_txx, d_adj_tzz, d_adj_txz,
            d_adj_vx, d_adj_vz, d_grad_lam, d_grad_mu, d_grad_rho, dt, ...);
    }
}
```

#### Two-Level Gradient Reduction
```c
// After all shots assigned to this shot group:
// Step 1: Each domain rank has gradient for its LOCAL subdomain, summed over its shots
// Step 2: Reduce across shot groups (ranks at same domain position)
MPI_Allreduce(MPI_IN_PLACE, d_grad_lam, nx_local*naz, MPI_FLOAT, MPI_SUM, shot_comm);
MPI_Allreduce(MPI_IN_PLACE, d_grad_mu,  nx_local*naz, MPI_FLOAT, MPI_SUM, shot_comm);
MPI_Allreduce(MPI_IN_PLACE, d_grad_rho, nx_local*naz, MPI_FLOAT, MPI_SUM, shot_comm);

// Step 3: Optimizer runs on rank 0 of domain_comm within shot_group 0
// Need to gather full gradient to global rank 0:
if (shot_group_id == 0) {
    MPI_Gatherv(local_grad, nx_local*naz, MPI_FLOAT,
                global_grad, ..., 0, domain_comm);
    if (domain_rank == 0) {
        // Call Fortran optimizer: minimization_(...)
        // Get model update
    }
    // Scatter updated model back to domain ranks
    MPI_Scatterv(global_model, ..., local_model, nx_local*naz, MPI_FLOAT, 0, domain_comm);
}
// Broadcast model to all shot groups
MPI_Bcast(local_model, nx_local*naz, MPI_FLOAT, 0, shot_comm);
// Upload to device
cudaMemcpy(d_model, local_model, nx_local*naz*sizeof(float), cudaMemcpyHostToDevice);
```

#### Misfit Kernel (distributed)
```c
// Each domain rank computes local misfit from its local receivers
float local_misfit = 0.0f;
compute_residual_kernel<<<grid,block>>>(d_syn, d_obs, d_rsq, nrec_local, nt);
cub::DeviceReduce::Sum(d_temp, temp_bytes, d_rsq, d_local_misfit, nrec_local*nt);
cudaMemcpy(&local_misfit, d_local_misfit, sizeof(float), cudaMemcpyDeviceToHost);

// Sum across domain ranks (receivers distributed)
float shot_misfit;
MPI_Allreduce(&local_misfit, &shot_misfit, 1, MPI_FLOAT, MPI_SUM, domain_comm);

// Sum across shots (after all shots processed)
MPI_Allreduce(&total_misfit_local, &total_misfit, 1, MPI_FLOAT, MPI_SUM, shot_comm);
```

### 2.8 CUDA Streams and Async Execution

#### Stream Architecture (per GPU)
```c
cudaStream_t stream_compute;    // stencil kernels (interior)
cudaStream_t stream_border;     // stencil kernels (border, after halo recv)
cudaStream_t stream_halo;       // halo pack/unpack kernels
cudaStream_t stream_io;         // checkpoint D2H / H2D transfers
```

#### Overlapping Strategy
```
Timeline for one time step (velocity phase):

stream_compute: |===== interior velocity kernel =====|
stream_halo:    |pack|--MPI send/recv--|unpack|
stream_border:                                 |= border kernel =|
stream_io:      |========= checkpoint D2H (if needed) =========|

Key overlaps:
1. Interior computation overlaps with halo exchange
2. Checkpoint D2H overlaps with next time step computation
3. Border kernel waits only on halo recv, not interior
```

#### Event-Based Profiling
```c
cudaEvent_t ev_step_start, ev_vel_interior, ev_halo_done, ev_vel_border, ev_stress_done;
// Record events at kernel boundaries → measure per-phase latency
```

### 2.9 Build System

#### Makefile Integration
```makefile
# New flags in Make_include
NVCC = nvcc
MPICC = mpicc
CUDA_ARCH = -arch=sm_70          # V100; -arch=sm_80 for A100; -arch=sm_90 for H100
CUDA_FLAGS = -O3 --use_fast_math -lineinfo -Xcompiler -fopenmp
CUDA_LIBS = -lcudart

# Build targets:
#   make                  → CPU-only serial
#   make USE_MPI=1        → CPU MPI (shot-parallel, existing)
#   make USE_CUDA=1       → single-GPU CUDA
#   make USE_CUDA=1 USE_MPI=1 → multi-GPU CUDA + MPI (domain decomp + shots)

ifdef USE_CUDA
  CFLAGS += -DUSE_CUDA
  ifdef USE_MPI
    CFLAGS += -DUSE_MPI -DUSE_DOMAIN_DECOMP
    CC = $(MPICC)
    NVCCFLAGS += -Xcompiler -DUSE_MPI -Xcompiler -DUSE_DOMAIN_DECOMP
  endif
  CUDA_OBJS = fdelmodc_cuda.o fdelfwi_cuda.o domain_decomp.o
endif
```

#### Targets
```makefile
# CPU-only (default)
fwi_inversion: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# CUDA + MPI (production)
fwi_gpu_inversion: $(OBJS) $(CUDA_OBJS)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LIBS) $(CUDA_LIBS)

# .cu compilation rule
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_ARCH) $(NVCCFLAGS) -c $< -o $@
```

#### File Organization
```
fdelfwi/
  ├── fdelmodc_cuda.cu      # Forward modeling CUDA kernels
  ├── fdelfwi_cuda.cu       # FWI CUDA kernels (adjoint, gradient, misfit)
  ├── domain_decomp.c       # MPI domain decomposition setup, halo exchange
  ├── domain_decomp.h       # Domain decomp structs and function declarations
  ├── cuda_utils.h          # CUDA_CHECK macro, stream setup, device query
  ├── cuda_kernels.h        # Kernel declarations shared between .cu files
  ├── elastic4.c            # Original CPU code (unchanged)
  ├── elastic4_adj.c        # Original CPU code (unchanged)
  └── ...
```

### 2.10 Validation and Testing Strategy

#### Unit Tests (per kernel)
1. **Stencil kernel**: Run CPU elastic4() and GPU update_velocity/stress_kernel on identical input; compare output arrays element-wise. Tolerance: relative L2 < 1e-6 (single precision).
2. **PML kernel**: Same — compare PML auxiliary arrays and boundary wavefield values.
3. **Gradient kernel**: Single time step cross-correlation; compare g_lam, g_mu, g_rho.
4. **Misfit reduction**: Compare CUB reduction vs CPU sum. Tolerance: relative < 1e-5.

#### Integration Tests
1. **Forward wavefield snapshot**: Run full forward pass CPU vs GPU; compare snapshot at t=tmax. Relative L2 norm of difference < 1e-5.
2. **Receiver traces**: Compare synthetic .su files trace-by-trace. Max relative error < 1e-5.
3. **Single-shot gradient**: Run forward+adjoint on CPU and GPU; compare gradient arrays. Relative L2 < 1e-4 (accumulated single-precision rounding).
4. **Dot product test**: Run test_wave_op_dp with GPU kernels; verify <Jx,y> = <x,J^Ty> to machine precision.

#### Domain Decomposition Tests
1. **Halo exchange correctness**: Run 1-GPU vs 2-GPU (ndom=2) forward pass on small grid; compare full wavefield snapshots. Must be bitwise identical (same floating-point order).
2. **Scaling test**: 1 GPU vs 2 vs 4 vs 8 GPUs (ndom); measure wall-clock for 100 time steps. Expect near-linear speedup (halo is tiny vs compute).
3. **Source/receiver mapping**: Verify source injection produces identical wavefield regardless of which subdomain owns the source. Verify gathered receiver traces match single-GPU output.
4. **Gradient consistency**: Single-shot gradient with ndom=1 vs ndom=4 must match to relative L2 < 1e-6.

#### Regression Tests
1. **Full FWI iteration**: 1 L-BFGS iteration with 5 shots; CPU vs GPU gradient norms must match to 4+ significant digits.
2. **Convergence test**: 5 L-BFGS iterations; cost reduction trajectory must match qualitatively (same convergence rate).
3. **Multi-component**: Test with comp=_rvx,_rvz,_rp; verify misfit and gradient match CPU.
4. **Multi-GPU FWI**: Full iteration with ndom=4, nshot_groups=2 (8 GPUs total); gradient must match serial CPU reference.

#### Profiling Baseline
- Record CPU wall-clock for: forward pass, adjoint pass, full FWI iteration
- Record GPU kernel times via cudaEvents
- Measure halo exchange overhead vs compute time (expect < 5%)
- Compare and report speedup: 1 GPU vs CPU, and scaling efficiency across ndom GPUs
- Strong scaling: fix problem size, increase ndom → measure time reduction
- Weak scaling: fix local subdomain size, increase ndom → measure constant time

---

## STEP 3: IMPLEMENTATION (PENDING)

## STEP 4: PERFORMANCE TUNING (PENDING)
