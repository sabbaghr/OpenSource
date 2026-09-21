# CUDA Implementation Log

Tracks every implementation step, decisions, and file changes for the CUDA port.

---

## Session 2026-03-15: Step 3.1 — Foundation Files

### Files Created

#### 1. `fdelfwi/cuda_utils.h` — CUDA foundation utilities

**Purpose:** Shared header for all CUDA files. Provides error checking, stream management, timers, GPU binding, and helper macros.

**Contents:**

| Section | What | Details |
|:--------|:-----|:--------|
| `CUDA_CHECK(call)` | Error-checking macro | Wraps every CUDA API call; prints file:line + error string, exits on failure |
| `d_fd_coeff[4]` | `__constant__` memory | FD stencil coefficients (c1-c4); extern declaration, defined in fdelmodc_cuda.cu |
| `cudaStreamSet` | Stream management struct | 4 streams: compute, border, halo, io |
| `cuda_streams_create/destroy` | Stream lifecycle | Inline functions |
| `cudaTimer` | Event-based timing | start/stop/elapsed_ms for kernel profiling |
| `cuda_bind_gpu(local_rank)` | GPU binding | Maps MPI local rank → GPU device ID (round-robin) |
| `cuda_print_device_info` | Diagnostics | Prints GPU name, SM version, VRAM, SMs, smem/const sizes |
| `BLOCK_X=16, BLOCK_Z=16` | Block dimensions | Standard 256-thread blocks for 2D stencil kernels |
| `cuda_halo_width(iorder)` | Halo calculator | Returns iorder/2 (2 for 4th, 3 for 6th, 4 for 8th) |
| `cuda_pad_naz(naz)` | Z-padding | Pads to next multiple of 32 for 128-byte aligned columns |
| `cuda_alloc_pinned/free_pinned` | Pinned host memory | Wrappers around cudaMallocHost/cudaFreeHost |

**Design decisions:**
- All functions are `static inline` → no link-time dependencies, just include the header
- `d_fd_coeff` is `extern __constant__` — defined once in fdelmodc_cuda.cu, visible from all .cu files
- Block size 16×16 = 256 threads balances occupancy and shared memory usage across GPU architectures

---

#### 2. `fdelfwi/domain_decomp.h` — Domain decomposition header

**Purpose:** Declares the two-level MPI parallelism structures and all public API functions.

**Key structures:**

| Struct | Purpose | Key fields |
|:-------|:--------|:-----------|
| `domainPar` | Per-rank decomposition info | `ndom`, `domain_rank`, `shot_group_id`, `domain_comm`, `shot_comm`, `nx_local`, `ix_global_start`, `nax_local`, `pad_left/right`, stagger offsets `io*/ie*` |
| `haloBuf` | Device halo exchange buffers | `send_left/right`, `recv_left/right`, `nfields`, `halo`, `naz`, `buf_size` |

**Public API functions:**

| Function | Purpose |
|:---------|:--------|
| `domain_decomp_init` | Create communicators, partition grid, bind GPU |
| `domain_decomp_free` | Release communicators |
| `domain_compute_stagger_offsets` | Map global io*/ie* to local subdomain |
| `halo_buf_create/destroy` | Allocate/free device halo buffers |
| `halo_exchange_velocity` | Exchange vx, vz halos (2 fields) |
| `halo_exchange_stress` | Exchange txx, tzz, txz halos (3 fields) |
| `domain_owns_source` | Check if source falls on this rank, return local ix |
| `domain_map_receivers` | Build local receiver list for this rank |
| `domain_scatter_model` | Distribute global model to local subdomains |
| `domain_gather_gradient` | Assemble local gradients into global array |
| `domain_reduce_gradient_shots` | MPI_Allreduce across shot groups |
| `domain_broadcast_model` | Broadcast model update to all ranks |
| `domain_checkpoint_size` | Bytes per checkpoint snapshot (local) |

---

#### 3. `fdelfwi/domain_decomp.c` — Domain decomposition implementation

**Purpose:** Implements all domain decomposition logic.

**MPI communicator setup (domain_decomp_init):**
```
world_rank / ndom → shot_group_id
world_rank % ndom → domain_rank

MPI_Comm_split(WORLD, shot_group_id, domain_rank) → domain_comm  (halo exchange)
MPI_Comm_split(WORLD, domain_rank, shot_group_id) → shot_comm    (gradient reduce)
```

**Grid partitioning:**
- 1D along X: `nx_local = nx_global/ndom + remainder distribution`
- Left padding: `ioPx` (PML) if rank 0, else `halo` (ghost)
- Right padding: `nax-iePx` if rank ndom-1, else `halo`
- Validates world_size % ndom == 0

**Stagger offset mapping (domain_compute_stagger_offsets):**
- Interior ranks: all offsets start at pad_left
- Boundary ranks: inherit global offsets from modPar
- Z-offsets: unchanged (no z-decomposition)
- Extra Vx/Txz column handled on rightmost rank

**Halo exchange implementation:**
- Pack: copy halo columns from device field to contiguous device buffer
- MPI: non-blocking Isend/Irecv with GPU-aware MPI pointers
- Unpack: copy received buffer into ghost columns
- Tags: 100+f / 200+f for velocity, 300+f / 400+f for stress (avoid collisions)
- Current pack/unpack: cudaMemcpy loops (will become CUDA kernels in Step 3.2)

**Source/receiver mapping:**
- `domain_owns_source`: simple range check on global x-position
- `domain_map_receivers`: loop over all global receivers, filter by local x-range

**Model scatter (placeholder):**
- Currently each rank reads full model and extracts local slice
- Production: will use MPI_Scatterv from rank 0

**Gradient gather:**
- Pack interior-only columns into contiguous send buffer
- MPI_Gatherv to rank 0 of domain_comm
- Strips padding (ioPz, pad_left) during packing

---

### Design Notes

**Why 1D decomposition (X-only):**
- Production domain: 120 km × 12 km → X is 10-25× longer than Z
- Z (1200-3000 points) fits in one GPU
- 1D = one left neighbor + one right neighbor → simple topology
- 2D decomposition reserved for Z > ~8000 points

**Halo exchange cost (4th order, dx=5m, naz=2432):**
```
Per exchange: 2 columns × 2432 floats × 4 bytes = 19 KB per field
Velocity phase: 2 fields × 2 directions × 19 KB = 76 KB
Stress phase: 3 fields × 2 directions × 19 KB = 114 KB
Total per time step: ~190 KB — negligible vs compute
```

**What's NOT yet implemented (deferred to later steps):**
- CUDA pack/unpack kernels (currently using cudaMemcpy loops)
- Interior/border split kernel launch (overlap compute with halo exchange)
- Proper MPI_Scatterv for model distribution
- GPU-aware MPI detection and fallback to pinned staging

---

### Next Steps (Step 3.2) — COMPLETED

---

## Session 2026-03-15: Step 3.2 — Forward Modeling CUDA Kernels

### File Created: `fdelfwi/fdelmodc_cuda.cu` (~750 lines)

**Purpose:** GPU implementations of all forward elastic FD operations.

### Kernel Inventory

| # | Kernel | CPU Source | Purpose |
|:--|:-------|:----------|:--------|
| 1 | `update_velocity_kernel<4/6/8>` | elastic4/6/8.c vel loops | Vx,Vz stencil (4th/6th/8th order, templated) |
| 2 | `update_stress_kernel<4/6/8>` | elastic4/6/8.c stress loops | Txx,Tzz,Txz stencil (templated) |
| 3 | `taper_top_velocity_kernel<4/6/8>` | boundaries.c:563+ | Top taper: FD + taper for Vx,Vz |
| 4 | `taper_bot_velocity_kernel<4/6/8>` | boundaries.c:730+ | Bottom taper for Vx,Vz |
| 5 | `taper_left_velocity_kernel<4/6/8>` | boundaries.c elastic left | Left taper for Vx,Vz |
| 6 | `taper_right_velocity_kernel<4/6/8>` | boundaries.c elastic right | Right taper for Vx,Vz |
| 7 | `taper_top_stress_kernel<4/6/8>` | boundaries.c (boundariesV) | Top taper for Txx,Tzz,Txz |
| 8 | `taper_bot_stress_kernel<4/6/8>` | " | Bottom taper for stress |
| 9 | `taper_left_stress_kernel<4/6/8>` | " | Left taper for stress |
| 10 | `taper_right_stress_kernel<4/6/8>` | " | Right taper for stress |
| 11 | `free_surface_mirror_kernel` | boundaries.c:52-60 | Vz mirror at free surface |
| 12 | `inject_source_kernel` | applySource.c:151-267 | Stress/force source injection |
| 13 | `extract_receivers_kernel` | getRecTimes.c | Multi-component receiver recording |

**Note:** Each templated kernel compiles to 3 variants (ORDER=4,6,8). Total compiled kernels: 10×3 + 2 = 32.
Device helper functions (4) are `__forceinline__` and shared across all kernels.

### Device Data Structures

| Struct | Contents | Lifecycle |
|:-------|:---------|:----------|
| `deviceWfl` | vx, vz, txx, tzz, txz | Allocated per shot, zeroed between shots |
| `deviceMod` | l2m, lam, muu, rox, roz | Persistent across shots, updated per FWI iteration |
| `deviceBnd` | tapx, tapz, tapxz, surface | Persistent, uploaded once |
| `deviceRec` | rec_vx/vz/txx/tzz/p, rec_ix/iz | Allocated per shot |

### Memory Management Functions

| Function | Purpose |
|:---------|:--------|
| `cuda_alloc_wfl/free/zero` | Allocate/free/zero 5 wavefield arrays |
| `cuda_alloc_mod/free/upload` | Allocate/free material params, H2D upload |
| `cuda_alloc_bnd/free` | Allocate/upload taper + surface arrays |
| `cuda_alloc_rec/free` | Allocate receiver recording buffers |
| `cuda_set_fd_coefficients` | Upload FD stencil coefficients to `__constant__` memory |

### Kernel Launch Wrappers (extern "C")

| Function | Kernels Launched |
|:---------|:----------------|
| `cuda_launch_velocity_update` | `update_velocity_4_kernel` |
| `cuda_launch_stress_update` | `update_stress_4_kernel` |
| `cuda_launch_taper_velocity` | Top/bot/left/right taper velocity kernels (conditional) |
| `cuda_launch_taper_stress` | Top/bot/left/right taper stress kernels (conditional) |
| `cuda_launch_free_surface` | `free_surface_mirror_kernel` |
| `cuda_launch_inject_source` | `inject_source_kernel` |
| `cuda_launch_extract_receivers` | `extract_receivers_kernel` |

### Design Decisions

**Multi-order support via C++ templates:**
- `template<int ORDER>` generates 3 compiled kernel variants (ORDER=4, 6, 8) from a single source
- Compiler eliminates dead `if (ORDER >= 6/8)` branches at compile time → zero runtime overhead
- Explicit template instantiations at bottom of kernel section
- Runtime dispatch via `DISPATCH_ORDER(iorder, call4, call6, call8)` macro
- All launch wrappers accept `int iorder` parameter and select the right variant

**Device helper functions for stencil computation:**
- `stencil_Dmx_txx_Dpz_txz<ORDER>()` — Vx stencil (D-x txx + D+z txz)
- `stencil_Dpx_txz_Dmz_tzz<ORDER>()` — Vz stencil (D+x txz + D-z tzz)
- `stencil_stress_dvx_dvz<ORDER>()` — stress dvx/dvz computation
- `stencil_txz_update<ORDER>()` — Txz stencil (D-z vx + D-x vz)
- Declared `__device__ __forceinline__` — zero function call overhead
- Used by both interior kernels and taper kernels (no code duplication)

**Stencil implementation — register-based, no shared memory (for now):**
- 4th order stencil radius = 2 → only 5 points per direction
- 8th order stencil radius = 4 → 9 points per direction — still fits L1 cache well
- L1 cache (128 KB on modern GPUs) handles this well without explicit tiling
- Shared memory tiling deferred to Step 4 (performance tuning) if profiling shows benefit
- This keeps the code simpler and easier to validate

**Taper boundaries vs PML:**
- Elastic scheme uses taper boundaries (bnd.type==4), NOT PML (bnd.type==2)
- PML (split-field) is acoustic-only in the current codebase
- Taper = apply FD stencil in taper zone, then multiply by cosine weight
- 8 taper kernels: 4 sides × (velocity + stress)
- Corner tapers not yet implemented (will add in Step 3.3 if needed)

**Source injection:**
- Single-thread kernel (source is 1 grid point) — negligible cost
- Supports: stress monopole (type=1), dipole variants (orient=2-5), txz (type=2), force-x (type=6), force-z (type=7)
- Source amplitude pre-computed on host (wavelet interpolation + scaling)

**Receiver extraction:**
- One thread per receiver, 256 threads/block
- Records directly to device buffer [nrec × nt]
- Supports multi-component: vx, vz, txx, tzz, hydrophone (0.5*(txx+tzz))
- NULL pointer check per component — only active components are recorded

**What's NOT yet implemented (deferred):**
- 6th/8th order kernels (currently only 4th order) — template or separate kernels
- Corner taper kernels (top-left, top-right, bot-left, bot-right)
- boundariesV full free-surface stress update (currently only Vz mirror)
- storeSourceOnSurface / reStoreSourceOnSurface
- Shared memory tiling (deferred to Step 4 performance tuning)

### Next Steps (Step 3.3) — COMPLETED

---

## Session 2026-03-15: Step 3.3 — FWI CUDA Kernels

### File Created: `fdelfwi/fdelfwi_cuda.cu` (~600 lines)

**Purpose:** GPU implementations of all FWI-specific operations: adjoint stencils, gradient cross-correlation, adjoint source injection, and checkpoint management.

### Kernel Inventory

| # | Kernel | CPU Source | Purpose |
|:--|:-------|:----------|:--------|
| 1 | `adj_update_velocity_kernel<4/6/8>` | elastic4/6/8_adj.c vel | True adjoint velocity: material INSIDE derivative |
| 2 | `adj_update_stress_kernel<4/6/8>` | elastic4/6/8_adj.c stress | True adjoint stress: buoyancy INSIDE derivative |
| 3 | `adj_free_surface_txz_kernel<4/6/8>` | elastic4/6/8_adj.c surface | Free-surface txz correction (order-dependent terms) |
| 4 | `gradient_lambda_mu_P_kernel<4/6/8>` | fwi_gradient.c:210-287 | Lambda + mu normal-stress gradient at P grid |
| 5 | `gradient_mu_shear_kernel<4/6/8>` | fwi_gradient.c:299-334 | Mu shear gradient at Txz grid → scatter to P |
| 6 | `gradient_rho_kernel` | fwi_gradient.c:354-396 | Density gradient from velocity time derivative |
| 7 | `inject_adjoint_source_kernel` | applyAdjointSource.c | Multi-type adjoint source injection (P, Fx, Fz, Txz) |

### Checkpoint Management

| Function | Purpose |
|:---------|:--------|
| `cuda_checkpoint_create` | Allocate pinned host buffer for all snapshots |
| `cuda_checkpoint_destroy` | Free pinned buffer |
| `cuda_checkpoint_save` | Async D2H of 5 wavefield arrays to pinned host |
| `cuda_checkpoint_load` | Async H2D from pinned host to device |

Buffer layout: `h_buf[isnap * 5 * nax * naz + field * nax * naz + ix * naz + iz]`

### Design Decisions

**Adjoint stencil — faithful translation:**
- Material parameters (l2m, lam, mul, rox, roz) are inside the derivative, exactly matching the CPU adjoint
- `+=` sign (not `-=`) — true adjoint convention
- Free-surface txz correction is order-dependent: ORDER=4 has 2 rows, ORDER=6 has 3, ORDER=8 has 4
- Each correction row adds progressively fewer terms (triangular pattern)

**Gradient kernels — split into 3 for different grids:**
- `gradient_lambda_mu_P`: lambda + mu normal-stress at P grid (no atomics needed)
- `gradient_mu_shear`: mu shear at Txz grid → scattered to 4 P neighbors via `atomicAdd`
- `gradient_rho`: velocity time derivative → scattered to 2 P neighbors via `atomicAdd`
- atomicAdd is necessary because multiple Txz/Vx/Vz points scatter to overlapping P-grid points

**Adjoint source injection:**
- One thread per adjoint source (receiver), 256 threads/block
- Phase filtering in kernel (phase=1 for force, phase=2 for stress)
- Supports P (→ 0.5*txx + 0.5*tzz), Fx (→ vx), Fz (→ vz), Txz (→ txz), Tzz (→ tzz), Txx (→ txx)

**Checkpoint to pinned host (not VRAM):**
- Production grids: 400 snapshots × 280 MB each = 112 GB → must be in host RAM
- Async D2H/H2D via stream_io overlaps with compute
- Pinned memory (cudaMallocHost) enables full DMA bandwidth

### Launch Wrappers (extern "C")

| Function | Dispatches to |
|:---------|:-------------|
| `cuda_launch_adj_velocity_update` | `adj_update_velocity_kernel<4/6/8>` |
| `cuda_launch_adj_stress_update` | `adj_update_stress_kernel<4/6/8>` |
| `cuda_launch_adj_free_surface_txz` | `adj_free_surface_txz_kernel<4/6/8>` |
| `cuda_launch_gradient_lambda_mu_P` | `gradient_lambda_mu_P_kernel<4/6/8>` |
| `cuda_launch_gradient_mu_shear` | `gradient_mu_shear_kernel<4/6/8>` |
| `cuda_launch_gradient_rho` | `gradient_rho_kernel` |
| `cuda_launch_inject_adjoint_source` | `inject_adjoint_source_kernel` |

### Next Steps (Step 3.4) — COMPLETED (fdelmodc GPU path)

---

## Session 2026-03-17: Step 3.4a — fdelmodc GPU Integration

**Priority change:** User wants GPU forward modeling for observed data generation first, before FWI integration.

### Files Created/Modified

| File | Action | Purpose |
|:-----|:-------|:--------|
| `fdelmodc/fdelmodc_gpu.h` | NEW | Header declaring GPU init/elastic/cleanup functions |
| `fdelmodc/fdelmodc_gpu.cu` | NEW | GPU driver: time loop, source, receiver, I/O orchestration |
| `fdelmodc/fdelmodc_cuda.cu` | SYMLINK → fdelfwi/ | Shared CUDA kernels (stencils, taper, source, receiver) |
| `fdelmodc/cuda_utils.h` | SYMLINK → fdelfwi/ | Shared CUDA utilities |
| `fdelmodc/fdelmodc.c` | MODIFIED | Added `#ifdef USE_CUDA` branch in shot loop |
| `fdelmodc/Makefile` | MODIFIED | Added `fdelmodc_gpu` target with nvcc rules |

### Architecture

```
fdelmodc.c (main)
  ├── getParameters, readModel, defineSource  [unchanged]
  ├── fdelmodc_gpu_init()                     [NEW: upload model to GPU, once]
  ├── Shot loop:
  │     ├── #ifdef USE_CUDA && ischeme==3:
  │     │     └── fdelmodc_gpu_elastic()      [NEW: GPU time loop]
  │     │           ├── cuda_zero_wfl()
  │     │           ├── for (it = it0..it1):
  │     │           │     ├── cuda_launch_velocity_update()
  │     │           │     ├── cuda_launch_inject_source()     [force sources]
  │     │           │     ├── cuda_launch_taper_velocity()
  │     │           │     ├── cuda_launch_free_surface()
  │     │           │     ├── cuda_launch_stress_update()
  │     │           │     ├── cuda_launch_inject_source()     [stress sources]
  │     │           │     ├── cuda_launch_taper_stress()
  │     │           │     ├── cuda_launch_extract_receivers()
  │     │           │     └── writeSnapTimes() [D2H + CPU I/O, if enabled]
  │     │           ├── cudaMemcpy D2H: rec_vx, rec_vz, rec_p, etc.
  │     │           └── cuda_free_rec()
  │     └── goto gpu_post_shot → writeRec()   [CPU .su file output]
  └── fdelmodc_gpu_cleanup()                  [NEW: free GPU arrays, once]
```

### Key Design Decisions

**Single GPU, no domain decomposition:**
- fdelmodc is for generating observed data — single GPU is sufficient for typical synthetic models
- Model must fit in GPU VRAM (production grids need domain decomp → use fdelfwi path)

**Shared CUDA kernels via symlinks:**
- `fdelmodc/fdelmodc_cuda.cu` → `fdelfwi/fdelmodc_cuda.cu` (symlink)
- Same compiled kernels used by both fdelmodc_gpu and future fdelfwi GPU
- No code duplication

**Model stays on GPU across shots:**
- `fdelmodc_gpu_init()` called once → uploads rox, roz, l2m, lam, mul, taper, surface
- `cuda_zero_wfl()` called per shot (only zero wavefields, not model)
- `fdelmodc_gpu_cleanup()` called at exit

**Source amplitude pre-computed on host:**
- Wavelet interpolation done on CPU (trivial cost)
- l2m scaling for stress sources uses host l2m at source point
- Amplitude passed as scalar to `inject_source_kernel<<<1,1>>>`

**Receiver recording on GPU:**
- `extract_receivers_kernel` runs every rec.skipdt steps
- All traces stored in GPU buffer [nrec × nt]
- Single bulk D2H transfer after time loop completes

**Snapshot output via D2H:**
- If sna.nsnap enabled: sync GPU, download 5 fields, call CPU writeSnapTimes
- This is slow but snapshots are typically sparse (every 100+ time steps)

### Build

```bash
# CPU-only (default, unchanged)
make

# GPU (new target)
make USE_CUDA=1

# Produces: fdelmodc_gpu binary
# Usage identical to fdelmodc — same parameters, same .su output
```

### What's NOT yet done

- fdelmodc_gpu only handles `ischeme==3` (elastic). Acoustic/visco falls through to CPU.
- No storeSourceOnSurface / reStoreSourceOnSurface on GPU (free surface source correction)
- No getRecTimes interpolation modes (int_p=2,3) — GPU uses grid-point extraction only
- No beam output on GPU (beams still require CPU path)
- FWI integration (fdelfwi) still pending (Step 3.4b)

### Next Steps

1. Test compile: `make USE_CUDA=1` to verify build succeeds
2. Validate: compare GPU vs CPU receiver traces on small test model

---

## Session 2026-04-24: Step 3.4b — fdelfwi GPU Integration

### Files Created

#### 1. `fdelfwi/fdelfwi_gpu.h` — GPU FWI driver interface

**Purpose:** Declares functions for GPU-accelerated FWI workflow.

**Public API:**

| Function | Purpose |
|:---------|:--------|
| `fdelfwi_gpu_init` | One-time GPU setup: allocate device arrays, upload model/boundaries, create streams, allocate pinned checkpoint buffer |
| `fdelfwi_gpu_upload_model` | Re-upload model after optimizer updates mod->rox/roz/l2m/lam/muu |
| `fdfwimodc_gpu` | GPU forward modeling with checkpointing (drop-in for fdfwimodc) |
| `adj_shot_gpu` | GPU adjoint backpropagation with gradient (replaces adj_shot) |
| `fdelfwi_gpu_cleanup` | Free all GPU resources |

---

#### 2. `fdelfwi/fdelfwi_gpu.cu` (~650 lines) — GPU FWI driver implementation

**Purpose:** GPU-accelerated forward modeling + adjoint/gradient for FWI.

**Architecture:**
```
fdelfwi_gpu_init()        → once: upload model, alloc wavefields/gradients/checkpoints
  ↓
Per FWI iteration:
  ├── fdfwimodc_gpu()     → GPU time loop, checkpoint to pinned host, D2H receivers
  │     └── gpu_forward_one_step()  → vel→src→taper→surface→stress→src→taper
  │     └── cuda_checkpoint_save()  → async D2H to pinned host
  │     └── writeRec()              → CPU .su output
  ├── [CPU: computeResidual, readResidual]
  └── adj_shot_gpu()      → GPU adjoint with gradient
        └── For each segment (reverse):
              ├── cuda_checkpoint_load()  → pinned host → device
              ├── gpu_forward_one_step()  → re-propagate, buffer vx/vz
              ├── gpu_adjoint_one_step()  → adj vel→taper→surface→adj stress→taper
              ├── cuda_launch_gradient_*  → lambda, mu, rho cross-correlation
              └── cuda_launch_inject_adjoint_source() → residual injection
        └── D2H gradient download, add to host arrays
  ↓
fdelfwi_gpu_upload_model() → after optimizer updates model
  ↓
fdelfwi_gpu_cleanup()     → once at exit
```

**Persistent GPU state (static variables):**

| Variable | Purpose | Lifecycle |
|:---------|:--------|:----------|
| `s_wfl_fwd` | Forward wavefield (vx,vz,txx,tzz,txz) | Zeroed per shot |
| `s_wfl_adj` | Adjoint wavefield | Zeroed per shot |
| `s_dmod` | Material parameters (l2m,lam,muu,rox,roz) | Updated per FWI iter |
| `s_dbnd` | Boundary tapers + surface | Persistent |
| `s_d_grad_*` | Gradient arrays (lam,muu,rho) | Zeroed per shot |
| `s_d_buf_vx/vz` | Forward vx/vz buffer for one segment | Resized if needed |
| `s_gpu_chk` | Pinned host checkpoint buffer | Persistent |
| `s_streams` | CUDA streams (compute, border, halo, io) | Persistent |

**Key design decisions:**
- Single GPU, no domain decomposition (production domain decomp would wrap these)
- Checkpoints go to pinned host memory (async D2H), not disk
- Forward vx/vz buffer for re-propagation stays on GPU (avoids D2H/H2D per step)
- Gradient accumulated on GPU, bulk D2H at end of shot
- I/O (writeRec, computeResidual, readResidual) stays on CPU — not hot path
- Adjoint source traces uploaded to GPU once per shot
- `gpu_forward_one_step()` shared by forward pass and adjoint re-propagation
- Stress source l2m scaling: single-element D2H from device (cached in future)
- Hydrophone computation and .su file I/O handled by existing CPU code

---

### Files Modified

#### 3. `fdelfwi/fwi_inversion.c` — Added `#ifdef USE_CUDA` GPU paths

6 insertion points:

| Location | What |
|:---------|:-----|
| After includes | `#include "fdelfwi_gpu.h"` |
| After model load | `fdelfwi_gpu_init()` with computed nsnap |
| Print summary | GPU/MPI banner variants |
| Forward pass | `fdfwimodc_gpu()` replaces `fdfwimodc()` |
| Adjoint pass | `adj_shot_gpu()` replaces `adj_shot()` |
| After model update | `fdelfwi_gpu_upload_model()` |
| Before MPI_Finalize | `fdelfwi_gpu_cleanup()` |

All CPU code paths preserved unchanged when `USE_CUDA` is not defined.

#### 4. `fdelfwi/Makefile` — Added GPU build targets

**New targets:**

| Target | Command | What |
|:-------|:--------|:-----|
| `fwi_gpu_inversion` | `make gpu_inversion` | Single-GPU FWI inversion |
| `fwi_mpi_gpu_inversion` | `make mpi_gpu_inversion` | MPI + single-GPU-per-rank FWI |

**Build configuration:**
- `NVCC`, `CUDA_ARCH` (default sm_70), `CUDA_FLAGS`, `CUDA_LIBS` defined at top
- GPU targets compile: fdelmodc_cuda.cu, fdelfwi_cuda.cu, fdelfwi_gpu.cu with nvcc
- fwi_inversion.c compiled with `-DUSE_CUDA` (and `-DUSE_MPI` for MPI variant)
- Links with existing CPU objects + CUDA objects + `-lcudart -lstdc++`
- Clean target updated to remove GPU .o files and binaries

---

### What's NOT yet implemented (deferred)

- Yang pseudo-Hessian accumulation on GPU (adj_shot_gpu skips hess_* arrays)
- D_σ density gradient (exact stress-based rho gradient, uses dv/dt fallback)
- TRN/Born Hessian-vector product on GPU (hess_shot stays CPU)
- Shared memory tiling (deferred to Step 4 performance tuning)
- Preconditioner shot (precond_shot) on GPU

### Next Steps

1. Test compile on GPU node
2. Validate: compare GPU vs CPU gradient on small test model (3-anomaly)
3. Performance benchmarking: GPU vs CPU wall-clock per shot
4. Step 4: Performance tuning (shared memory, stream overlap, profiling)

---

## Session 2026-04-24: Step 3.5 — Domain Decomposition Integration

### Overview

Wired the existing domain decomposition infrastructure (`domain_decomp.h/.c`) into the GPU FWI workflow. This enables multi-GPU acceleration where multiple GPUs share one shot via 1D X-axis domain decomposition with halo exchange every time step, combined with shot-parallel MPI across shot groups.

### Architecture

```
Total MPI ranks = ndom × nshot_groups

  ndom = 1:  Single GPU per shot (backward compatible)
  ndom > 1:  Domain decomposition along X-axis

Example: 4 GPUs/shot × 5 shot groups = 20 MPI ranks
  Shot group 0: ranks 0-3  → halo exchange, share shot 0, 5, 10, ...
  Shot group 1: ranks 4-7  → halo exchange, share shot 1, 6, 11, ...
  ...

Two MPI communicators:
  domain_comm: ranks sharing one shot (halo exchange every time step)
  shot_comm:   same domain position across shot groups (gradient reduction)

Gradient reduction strategy:
  Each rank accumulates gradient in its LOCAL subdomain positions of a
  GLOBAL-sized array (other positions zero). MPI_Allreduce on MPI_COMM_WORLD
  correctly sums across shots AND assembles subdomains.
```

### Files Modified

#### 1. `fdelfwi/fdelfwi_gpu.h` — Added domain-decomposed API

| Function | Purpose |
|:---------|:--------|
| `fdelfwi_gpu_init_domain` | GPU init with local subdomain dimensions + halo buffers |
| `fdelfwi_gpu_upload_model_domain` | Extract and upload local model slice |
| `fdfwimodc_gpu_domain` | Domain-decomposed forward modeling with halo exchange |
| `adj_shot_gpu_domain` | Domain-decomposed adjoint + gradient with halo exchange |

#### 2. `fdelfwi/fdelfwi_gpu.cu` — Added ~450 lines for domain decomposition

**New static state:**
- `s_dom` — pointer to domainPar for current run
- `s_halo_vel`, `s_halo_stress` — halo exchange buffers (2 fields, 3 fields)
- `s_local_l2m/lam/muu/rox/roz` — host-side local model slices

**New functions:**

| Function | Key difference from single-GPU version |
|:---------|:--------------------------------------|
| `gpu_forward_one_step_domain` | Uses dom->io*/ie* offsets; halo exchange vx/vz after velocity, txx/tzz/txz after stress; source injection only on owning rank |
| `gpu_adjoint_one_step_domain` | Halo exchange for adjoint velocity and stress phases |
| `fdfwimodc_gpu_domain` | Local receiver mapping via domain_map_receivers; MPI_Gatherv to assemble full receiver traces on domain rank 0; writeRec on rank 0 only |
| `adj_shot_gpu_domain` | Gradient download maps LOCAL positions → GLOBAL positions in padded array; skips param=2 conversion (done once by extractGradientVector) |
| `fdelfwi_gpu_init_domain` | Allocates local-sized arrays, creates halo buffers, extracts model slices |
| `fdelfwi_gpu_upload_model_domain` | Re-extracts local model slice after optimizer update |

**Receiver gather protocol:**
1. Each domain rank extracts local receivers via `domain_map_receivers`
2. After GPU time loop: download local receiver traces
3. `MPI_Gatherv` on `domain_comm` assembles full traces on rank 0
4. Rank 0 calls `writeRec` to produce `.su` files

**Gradient download with global mapping:**
Local gradient at column `ix_l` maps to global column `ix_g`:
- Physical-left rank: `ix_g = ix_l` (PML columns at array start)
- Interior/right ranks: `ix_g = ioPx + ix_global_start + ix_l - pad_left`

#### 3. `fdelfwi/fwi_inversion.c` — Domain decomp integration

**New parameter:** `ndom=` (default 1, no domain decomp)

**Changes:**
- File-scoped `s_ndom_fwi`, `s_dom_fwi` for access from `compute_fwi_gradient`
- `ndom` parsing after optimization parameters
- `domain_decomp_init()` after model load when ndom>1
- `mpi_rank` overridden to `shot_group_id`, `mpi_size` to `nshot_groups`
- Forward/adjoint calls dispatch to `_domain` variants when ndom>1
- Model upload dispatches to `_domain` variant when ndom>1
- `domain_decomp_free()` at cleanup
- Print summary shows ndom, nshot_groups

#### 4. `fdelfwi/Makefile` — Updated GPU targets

- `domain_decomp.c` compiled with `-DUSE_CUDA` (single GPU) or `-DUSE_CUDA -DUSE_MPI` (MPI+GPU)
- `domain_decomp.o` linked into both GPU targets
- Clean target includes `domain_decomp.o`

### Usage

```bash
# Build
make mpi_gpu_inversion

# Single GPU per shot (ndom=1, backward compatible):
mpirun -np 4 fwi_mpi_gpu_inversion ... ndom=1
# → 4 shot groups, 1 GPU each, round-robin shots

# 2 GPUs per shot, 4 shot groups (8 GPUs total):
mpirun -np 8 fwi_mpi_gpu_inversion ... ndom=2
# → Each shot: 2 GPUs with halo exchange
# → 4 concurrent shot groups

# 4 GPUs per shot, 2 shot groups (8 GPUs total):
mpirun -np 8 fwi_mpi_gpu_inversion ... ndom=4
# → Each shot: 4 GPUs with halo exchange (faster per shot)
# → 2 concurrent shot groups (fewer concurrent shots)
```

### What's NOT yet implemented

- Interior/border kernel split for halo-compute overlap
- GPU-aware MPI detection (currently assumes GPU-aware MPI)
- Halo pack/unpack CUDA kernels (currently uses cudaMemcpy loops)
- 2D domain decomposition (for very deep models, Z > 8000 pts)
- Adjoint source filtering by local receivers (currently uploads all, kernel skips OOB)
