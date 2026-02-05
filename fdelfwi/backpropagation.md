# Backpropagation Module for Elastic FWI Gradient Computation

## Overview

This document describes the design and current implementation of the backpropagation module for elastic Full Waveform Inversion (FWI). The module backpropagates data residuals through the elastic wave equation using **dedicated adjoint FD kernels** and computes gradients via zero-lag cross-correlation with re-propagated forward wavefield snapshots. The design supports **multicomponent OBN data** (hydrophone + vx + vz) with simultaneous injection of all residual types in a single backward pass.

### Design Principles

1. **Forward kernels are untouched** -- `elastic4.c`, `elastic6.c`, `applySource.c`, etc. remain unmodified to avoid introducing bugs in working code.
2. **Dedicated adjoint kernels** -- Separate `elastic4_adj.c` and `applyAdjointSource.c` handle multicomponent residual injection at physically correct staggered-grid timing.
3. **Disk-based re-propagation checkpointing** -- The forward wavefield is checkpointed to disk at coarse intervals. During the adjoint pass, each segment is re-propagated from its checkpoint, and the gradient is accumulated at every time step for exact accuracy.
4. **No separate `fwd_shot.c`** -- The existing `fdfwimodc()` function is extended with an optional `checkpointPar *chk` parameter to save checkpoints during forward propagation.

---

## 1. Architecture

The backpropagation module consists of the following C files:

| File | Role | Status |
|------|------|--------|
| `readResidual.c` | Reads residual traces from SU file, determines per-source type from TRID header, maps to staggered grid | **Implemented** |
| `applyAdjointSource.c` | Injects multicomponent residuals into adjoint wavefield at correct phase points (force vs stress) | **Implemented** |
| `elastic4_adj.c` | 4th-order elastic adjoint FD kernel; calls `applyAdjointSource` at two injection points | **Implemented** |
| `adj_shot.c` | Backward time-stepping loop with re-propagation: reconstructs forward wavefield from checkpoints, propagates adjoint, accumulates gradient | **Implemented** |
| `checkpoint.c` | Disk-based checkpoint I/O: `initCheckpoints()`, `writeCheckpoint()`, `readCheckpoint()`, `cleanCheckpoints()` | **Implemented** |
| `fdfwimodc.c` | Forward modeling function, extended with `checkpointPar *chk` for saving checkpoints | **Modified** |

### Files NOT Modified

| File | Usage |
|------|-------|
| `elastic4.c`, `elastic6.c` | Forward FD kernels (used as-is for re-propagation in `adj_shot.c`) |
| `applySource.c` | Forward source injection (used during re-propagation) |
| `boundaries.c` | Boundary conditions (same for forward and adjoint) |
| `readModel.c` | Model I/O |
| `acoustic4.c`, `acoustic6.c` | Acoustic forward kernels (untouched; acoustic adjoint kernels are TODO) |

### Workflow for One Shot

```
1. fdfwimodc()    -->  Forward propagation with disk checkpoint saving
                       Input:  model (Vp, Vs, rho), source, checkpointPar *chk
                       Output: synthetic receiver data (.su) + checkpoint files on disk

2. Compute residual:  r = d_obs - d_syn  (done externally or by driver)

3. readResidual()  --> Parse residual SU file into adjSrcPar
                       Decodes TRID to per-source type/orientation
                       Maps receiver positions to staggered grid indices

4. adj_shot()      --> Re-propagation backward pass
                       For each segment (reverse order):
                         a. Load checkpoint, re-propagate forward storing vx/vz
                         b. Propagate adjoint backward (elastic4_adj)
                         c. Cross-correlate fwd/adj at every time step
                       Output: gradient arrays (g_l2m, g_lam, g_muu, g_rho)
```

---

## 2. readResidual.c

### Purpose

Reads residual (adjoint source) traces from an SU file and determines the source injection type from the TRID header, following the same convention as `fdacrtmc/readRcvWav.c`.

### TRID-to-Type Mapping

```c
/* TRID-to-Type Mapping (from readRcvWav.c):
 *   typ    = (hdr.trid - 1) % 8 + 1;     // type 1-8
 *   orient = (hdr.trid - 1) / 8 + 1;     // orientation 1-3
 *
 * Monopole (orient=1):       1=P  2=Txz  3=Tzz  4=Txx  5=S-pot  6=Fx  7=Fz  8=P-pot
 * Vertical Dipole (orient=2):  9=P 10=Txz 11=Tzz 12=Txx 13=S-pot 14=Fx 15=Fz 16=P-pot
 * Horizontal Dipole (orient=3): 17=P 18=Txz 19=Txz 20=Txx 21=S-pot 22=Fx 23=Fz 24=P-pot
 */
```

### Multicomponent OBN Data Mapping

For Ocean Bottom Node (OBN) data, each residual trace maps to a specific adjoint source:

| Observed Data | SU Component File | TRID | adj.typ | Injection Target |
|---------------|-------------------|------|---------|-----------------|
| Hydrophone (P) | `residual_rp.su` | 1 | 1 | `tzz += r`, `txx += r` (elastic) |
| Geophone Vx | `residual_rvx.su` | 6 | 6 | `vx += r` |
| Geophone Vz | `residual_rvz.su` | 7 | 7 | `vz += r` |

All three types are loaded into a single `adjSrcPar` structure and injected simultaneously in one backward pass.

### adjSrcPar Structure (fdelfwi.h)

```c
typedef struct _adjSrcPar {
    int     nsrc;       /* Number of adjoint source traces (receivers) */
    int     nt;         /* Number of time samples per trace */
    size_t *xi;         /* Horizontal grid indices (staggered-grid aware) */
    size_t *zi;         /* Vertical grid indices (staggered-grid aware) */
    int    *typ;        /* Source type per trace (1-8, from TRID) */
    int    *orient;     /* Source orientation per trace (1-3, from TRID) */
    float  *x;          /* Horizontal positions (physical coordinates) */
    float  *z;          /* Vertical positions (physical coordinates) */
    float  *wav;        /* Residual waveforms [nsrc * nt], column-major */
} adjSrcPar;
```

### Grid Position Mapping (staggered grid offsets)

```c
/* Horizontal index */
if (typ == 2 || typ == 6) xi = mod->ioXx + grid_ix;  // Vx grid
else                      xi = mod->ioPx + grid_ix;   // P/Txx/Tzz grid

/* Vertical index */
if (typ == 2 || typ == 7) zi = mod->ioZz + grid_iz;  // Vz grid
else                      zi = mod->ioPz + grid_iz;   // P/Txx/Tzz grid
```

### Key Difference from Forward srcPar

The forward `srcPar` struct has a **single** `type` and `orient` for ALL sources. The adjoint `adjSrcPar` has **per-source** `typ[]` and `orient[]` arrays to support simultaneous injection of multiple component types (e.g., P + Vx + Vz residuals in one backward pass).

---

## 3. Adjoint FD Kernels

### Why Separate Adjoint Kernels?

The forward kernels (`elastic4.c`, etc.) call `applySource()` which has two limitations for multicomponent FWI:

1. **Single source type**: `src.type` is a scalar -- all sources must be the same type. In multicomponent FWI, hydrophone residuals (type 1) and geophone residuals (types 6, 7) must be injected simultaneously.

2. **Amplitude scaling**: `applySource()` scales by `(1.0/mod.dx)*l2m[ix*n1+iz]`, which is forward-modeling scaling. The adjoint operator (transpose of receiver extraction) requires **raw residual injection** without material property scaling.

### elastic4_adj.c (Implemented)

Same staggered-grid FD stencils as `elastic4.c`, but replaces `applySource` with `applyAdjointSource`:

```
elastic4_adj() flow:
  1. Velocity update: vx (identical stencils to elastic4)
  2. Velocity update: vz (identical stencils)
  3. Inject FORCE adjoint sources: applyAdjointSource(phase=1)
     -> Fx (type 6) into vx, Fz (type 7) into vz
  4. Boundary conditions: boundariesP()
  5. Stress update: Txx, Tzz (identical stencils)
  6. Stress update: Txz (identical stencils)
  7. Inject STRESS adjoint sources: applyAdjointSource(phase=2)
     -> P (type 1) into tzz+txx, Txz (2) into txz, Tzz (3) into tzz, Txx (4) into txx
  8. Boundary conditions: boundariesV()
```

```c
int elastic4_adj(modPar mod, adjSrcPar adj, bndPar bnd, int itime,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    float *rox, float *roz, float *l2m, float *lam, float *mul,
    int rec_delay, int rec_skipdt, int verbose);
```

### applyAdjointSource.c (Implemented)

Per-source type dispatch with phase-correct timing:

```c
int applyAdjointSource(modPar mod, adjSrcPar adj, int itime,
    float *vx, float *vz, float *tzz, float *txx, float *txz,
    int rec_delay, int rec_skipdt, int phase, int verbose);
```

**Phase filtering**:
- `phase == 1`: Only inject types > 5 (force sources: Fx=6, Fz=7)
- `phase == 2`: Only inject types <= 5 (stress sources: P=1, Txz=2, Tzz=3, Txx=4)

**Time alignment**: Checks `itime` against `rec_delay` and `rec_skipdt` to ensure residuals are injected at the correct receiver sampling times.

**No amplitude scaling**: Raw `src_ampl = adj.wav[isrc * adj.nt + isam]` is injected directly. This is the correct adjoint (transpose) of the receiver extraction operator: `field[ix,iz] += residual`.

**Orientation support**: Handles monopole (orient=1), vertical dipole (orient=2), and horizontal dipole (orient=3) for stress sources.

### Kernels Still To Implement

| Kernel | Status |
|--------|--------|
| `elastic4_adj.c` | **Implemented** (4th-order elastic) |
| `elastic6_adj.c` | TODO (6th-order elastic) |
| `acoustic4_adj.c` | TODO (4th-order acoustic FWI) |
| `acoustic6_adj.c` | TODO (6th-order acoustic FWI) |

---

## 4. Forward Checkpointing

### Approach: Disk-Based Re-Propagation

The forward simulation saves the **complete wavefield state** (vx, vz, txx, tzz, txz) at coarse intervals to disk. During the adjoint pass, each segment between checkpoints is re-propagated from the saved state, and the gradient is computed at every intermediate time step. This trades compute (2x forward propagation) for memory (only one segment's vx/vz buffer in RAM at a time).

### checkpointPar Structure (fdelfwi.h)

```c
typedef struct _checkpointPar {
    int    nsnap;           /* Total number of stored checkpoints */
    int    skipdt;          /* Time steps between checkpoints */
    int    delay;           /* First checkpoint time step (usually 0) */
    int    naz;             /* Vertical array dimension (= mod->naz) */
    int    nax;             /* Horizontal array dimension (= mod->nax) */
    int    ischeme;         /* Wave equation type (1=acoustic, 3=elastic) */
    char   file_vx[1024];  /* Path for vx checkpoint file */
    char   file_vz[1024];  /* Path for vz checkpoint file */
    char   file_tzz[1024]; /* Path for tzz checkpoint file */
    char   file_txx[1024]; /* Path for txx (elastic only) */
    char   file_txz[1024]; /* Path for txz (elastic only) */
    int    it;              /* Running snapshot index (for adjoint sync) */
} checkpointPar;
```

### checkpoint.c Functions

```c
int  initCheckpoints(checkpointPar *chk, modPar *mod, int skipdt);
int  writeCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl);
int  readCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl);
void cleanCheckpoints(checkpointPar *chk);
```

Each checkpoint writes 5 binary files (acoustic: 3 files). Total disk usage per checkpoint:
- Elastic: `5 * nax * naz * 4 bytes`
- Acoustic: `3 * nax * naz * 4 bytes`

### Integration with fdfwimodc.c

The forward modeling function accepts an optional `checkpointPar *chk`:

```c
/* Inside fdfwimodc() time loop: */
if (chk && (it % chk->skipdt == 0) && (it >= chk->delay)) {
    int isnap = (it - chk->delay) / chk->skipdt;
    writeCheckpoint(chk, isnap, &wfl);
}
```

When `chk == NULL` (as in `test_fdfwimodc.c`), no checkpoints are written and the function behaves identically to before.

---

## 5. adj_shot.c -- Adjoint Backpropagation

### Purpose

Performs the complete adjoint pass for one shot: re-propagates the forward wavefield from checkpoints, propagates the adjoint wavefield backward using dedicated adjoint kernels, and accumulates the gradient via zero-lag cross-correlation at every time step.

### Interface

```c
int adj_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
             recPar *rec, adjSrcPar *adj,
             int ixsrc, int izsrc, float **src_nwav,
             checkpointPar *chk,
             float *grad_l2m, float *grad_lam,
             float *grad_muu, float *grad_rho,
             int verbose);
```

### Internal Dispatch Functions

**`callKernel()`** -- Dispatches to forward FD kernels for re-propagation:
```c
static void callKernel(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
                       int it, int ixsrc, int izsrc, float **src_nwav,
                       wflPar *wfl, int verbose)
```
Calls `acoustic4/6()` or `elastic4/6()` based on `mod->ischeme` and `mod->iorder`.

**`callAdjKernel()`** -- Dispatches to adjoint FD kernels:
```c
static void callAdjKernel(modPar *mod, adjSrcPar *adj, bndPar *bnd,
                          int it, wflPar *wfl,
                          int rec_delay, int rec_skipdt, int verbose)
```
Currently dispatches to `elastic4_adj()` for ischeme 3/5 with iorder 4. Has TODO placeholders for other kernels.

**`accumGradient()`** -- Cross-correlates forward/adjoint wavefields:
```c
static void accumGradient(modPar *mod,
                          float *fwd_vx, float *fwd_vz,
                          wflPar *wfl_adj, float dt,
                          float *grad_l2m, float *grad_lam,
                          float *grad_muu, float *grad_rho)
```
Supports both 4th-order and 6th-order FD stencils for gradient computation.

### Algorithm: Segment-Based Re-Propagation

```
adj_shot():
  1. Allocate forward and adjoint wavefields
  2. Allocate forward vx/vz buffer for one segment (max_nsteps * sizem)
  3. For each segment k = nsnap-1 down to 0:
     a. Read checkpoint k -> wfl_fwd
     b. Store wfl_fwd.vx/vz as buf[0]
     c. Re-propagate forward through segment:
        for j = 1 .. nsteps-1:
          callKernel(forward)  --> wfl_fwd
          store wfl_fwd.vx/vz as buf[j]
     d. Sweep adjoint backward through segment:
        for j = nsteps-1 down to 0:
          callAdjKernel(adjoint)  --> wfl_adj
          accumGradient(buf[j], wfl_adj) --> grad
  4. Free all allocations
```

Memory requirement: `2 * max_nsteps * nax * naz * sizeof(float)` for the forward velocity buffer, plus two full wavefields (forward and adjoint).

---

## 6. Gradient Kernels

### Elastic FWI Gradient Formulation

For the velocity-stress formulation with parameters `(l2m, lam, muu, rho)`, the gradient kernels are implemented in `accumGradient()` within `adj_shot.c`.

#### Gradient w.r.t. lambda+2*mu (P-wave modulus)

```
g_{l2m}(x,z) = -dt * SUM_t [ div_fwd(x,z,t) * div_adj(x,z,t) ]

where:
  div = dvx/dx + dvz/dz (computed with 4th or 6th order FD stencils)
```

Computed at P/Txx/Tzz grid positions (`ioPx:iePx, ioPz:iePz`).

#### Gradient w.r.t. lambda

```
g_{lam}(x,z) = -dt * SUM_t [ dvx/dx_fwd * dvz/dz_adj + dvz/dz_fwd * dvx/dx_adj ]
```

Computed at P/Txx/Tzz grid positions.

#### Gradient w.r.t. mu (shear modulus)

```
g_{muu}(x,z) = -dt * SUM_t [ (dvx/dz + dvz/dx)_fwd * (dvx/dz + dvz/dx)_adj ]
```

Computed at Txz grid positions (`ioTx:ieTx, ioTz:ieTz`). Only for elastic (ischeme > 2).

#### Gradient w.r.t. density (optional)

```
g_{rho}(x,z) = +dt * SUM_t [ vx_fwd * vx_adj + vz_fwd * vz_adj ]
```

Computed at Vx and Vz grid positions respectively.

### Parameterization Conversion

The gradients above are w.r.t. FD parameters `(l2m, lam, muu, rho)`. For Vp/Vs/rho parameterization:

| Target | Chain Rule |
|--------|-----------|
| Vp     | `g_Vp = 2*rho*Vp * g_l2m` |
| Vs     | `g_Vs = 2*rho*Vs * (g_muu - 2*g_lam)` |
| rho    | `g_rho_full = Vp^2 * g_l2m + (Vp^2 - 2*Vs^2) * g_lam + Vs^2 * g_muu + g_rho` |

---

## 7. Staggered Grid Layout

The elastic staggered grid layout (from `elastic4.c`) is critical for correct gradient computation:

```
  | txz vz| txz vz  txz vz
  |       |
  | vx  T | vx  T   vx  T     T = Txx/Tzz/P grid point
   -------
    txz vz  txz vz  txz vz
    vx  T   vx  T   vx  T
```

| Component | Grid Position | Loop Bounds | Array Index |
|-----------|--------------|-------------|-------------|
| Txx, Tzz  | (ix, iz)     | `ioPx:iePx, ioPz:iePz` | `ix*n1+iz` |
| Txz       | (ix+1/2, iz+1/2) | `ioTx:ieTx, ioTz:ieTz` | `ix*n1+iz` |
| Vx        | (ix+1/2, iz) | `ioXx:ieXx, ioXz:ieXz` | `ix*n1+iz` |
| Vz        | (ix, iz+1/2) | `ioZx:ieZx, ioZz:ieZz` | `ix*n1+iz` |

The gradient kernels use the correct grid positions:
- `g_l2m` and `g_lam` computed at P/Txx/Tzz positions
- `g_muu` computed at Txz positions
- `g_rho` has contributions at both Vx and Vz positions

---

## 8. Boundary Conditions During Backpropagation

The adjoint kernel (`elastic4_adj.c`) calls the same boundary condition functions as the forward kernel:
- `boundariesP()` after velocity update + force source injection
- `boundariesV()` after stress update + stress source injection

### Absorbing Boundaries

Taper (`bnd.top/bot/lef/rig == 4`) and PML boundaries are applied identically in forward and adjoint directions.

### Free Surface

If the forward model uses a free surface (`bnd.top == 1`), the same free surface condition is applied during adjoint propagation.

---

## 9. Integration with FWI Driver

### Usage Pattern

```c
/* In the FWI driver, for each shot: */

checkpointPar chk;
adjSrcPar adj;
size_t sizem = mod.nax * mod.naz;

/* Pre-allocate gradient arrays (accumulate across shots) */
float *grad_l2m = (float *)calloc(sizem, sizeof(float));
float *grad_lam = (float *)calloc(sizem, sizeof(float));
float *grad_muu = (float *)calloc(sizem, sizeof(float));
float *grad_rho = (float *)calloc(sizem, sizeof(float));

/* 1. Initialize checkpointing */
initCheckpoints(&chk, &mod, skipdt);

/* 2. Forward propagation (writes checkpoints to disk) */
fdfwimodc(&mod, &src, &wav, &bnd, &rec, &sna,
          ixsrc, izsrc, src_nwav,
          ishot, nshots, fileno, &chk, verbose);

/* 3. Compute residual: r = d_obs - d_syn (external) */

/* 4. Read multicomponent residual data */
readResidual("residual.su", &adj, &mod, &bnd);

/* 5. Adjoint backpropagation + gradient */
adj_shot(&mod, &src, &wav, &bnd, &rec, &adj,
         ixsrc, izsrc, src_nwav, &chk,
         grad_l2m, grad_lam, grad_muu, grad_rho, verbose);

/* 6. Accumulate per-shot gradient into global gradient (MPI_Allreduce) */

/* 7. Clean up */
freeResidual(&adj);
cleanCheckpoints(&chk);
```

### MPI Integration (Phase 5, Not Yet Implemented)

```c
for (ishot = rank; ishot < nshots; ishot += nprocs) {
    /* Forward + Adjoint for this shot */
    /* Accumulate local gradient */
}
MPI_Allreduce(local_grad, global_grad, sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
```

---

## 10. Implementation Phases

### Phase 1: readResidual.c -- COMPLETED
- Reads residual SU files with per-source TRID decoding
- Maps receiver positions to staggered grid indices
- Populates `adjSrcPar` struct added to `fdelfwi.h`

### Phase 2: Forward Checkpointing -- COMPLETED
- `checkpoint.c` with disk-based checkpoint I/O
- `fdfwimodc.c` extended with `checkpointPar *chk` parameter
- `test_fdfwimodc.c` passes `NULL` for backward compatibility

### Phase 3: Adjoint Backpropagation -- COMPLETED
- `adj_shot.c` with re-propagation segment loop
- `callKernel()` for forward re-propagation
- `callAdjKernel()` for adjoint kernel dispatch
- `accumGradient()` with 4th/6th-order FD stencils

### Phase 4: Adjoint FD Kernels -- COMPLETED (elastic 4th-order)
- `applyAdjointSource.c` -- per-source type injection with phase-correct timing
- `elastic4_adj.c` -- 4th-order elastic adjoint kernel
- TODO: `elastic6_adj.c`, `acoustic4_adj.c`, `acoustic6_adj.c`

### Phase 5: MPI Driver -- NOT STARTED
- `fwi_driver.c` -- Shot distribution, gradient accumulation
- `fwi_interface.h` -- C API for Fortran optimizer binding
- `fwi_wrapper.f90` -- ISO_C_BINDING to SEISCOPE optimization toolbox

---

## 11. Testing Strategy

### Test 1: Forward Modeling Verification -- PASS
- Quantitative comparison between `fdelmodc` (reference) and `test_fdfwimodc`
- Acoustic test: receiver threshold 0.001%, PASS
- Elastic with topography: receiver threshold 1.0%, PASS
- Snapshot comparison: txx/tzz at 5 time slices, threshold 3.0%

### Test 2: Adjoint Dot Product Test (TODO)
Verify the forward/adjoint operator pair satisfies:
```
<d, F*m> = <F^T*d, m>
```

### Test 3: Gradient Accuracy via Finite Differences (TODO)
For a single parameter perturbation `dm`:
```
g_fd = [J(m + eps*dm) - J(m - eps*dm)] / (2*eps)
```
Compare `g_fd` with the adjoint-computed gradient.

### Test 4: Homogeneous Model Gradient (TODO)
Verify banana-doughnut pattern for P-wave and ring pattern for S-wave.

---

## 12. Files Summary

### New Files Created

| File | Description |
|------|-------------|
| `readResidual.c` | Read residual SU file, parse TRID, map to staggered grid |
| `checkpoint.c` | Disk-based checkpoint I/O (init/write/read/clean) |
| `adj_shot.c` | Re-propagation backward pass with gradient accumulation |
| `applyAdjointSource.c` | Per-source type adjoint injection (phase 1: force, phase 2: stress) |
| `elastic4_adj.c` | 4th-order elastic adjoint FD kernel |

### Modified Files

| File | Change |
|------|--------|
| `fdelfwi.h` | Added `checkpointPar`, `adjSrcPar` struct definitions |
| `fdfwimodc.c` | Added `checkpointPar *chk` parameter, checkpoint writes in time loop |
| `Makefile` | Added all new source files to `SRCC` list |
| `test_fdfwimodc.c` | Passes `NULL` for `chk` (backward compatible) |

### Existing Files Used Without Modification

| File | Usage |
|------|-------|
| `elastic4.c`, `elastic6.c` | Forward FD kernels (re-propagation in adj_shot) |
| `acoustic4.c`, `acoustic6.c` | Acoustic forward kernels |
| `applySource.c` | Forward source injection (re-propagation) |
| `boundaries.c` | Boundary conditions (forward and adjoint) |
| `readModel.c` | Model I/O |
