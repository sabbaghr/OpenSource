# Elastic FWI Integration Guide

## Tool Coupling: OpenSource_SL10 + TOOLBOX_OPTIMIZATION

This document describes the architecture and implementation strategy for integrating **OpenSource_SL10** (finite-difference wave propagation) with **TOOLBOX_OPTIMIZATION** (Fortran optimization algorithms) to perform **Elastic Full Waveform Inversion (FWI)**.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Capabilities](#2-repository-capabilities)
3. [Architecture](#3-architecture)
4. [Implementation Details](#4-implementation-details)
5. [C/Fortran Binding Layer](#5-cfortran-binding-layer)
6. [Data Flow and File Formats](#6-data-flow-and-file-formats)
7. [Build System Integration](#7-build-system-integration)
8. [Implementation Status](#8-implementation-status)
9. [Testing Strategy](#9-testing-strategy)

---

## 1. Overview

### Goal
Implement elastic FWI using:
- **Forward modeling**: `fdelfwi` elastic kernels (self-contained fork of `fdelmodc`)
- **Adjoint/Gradient**: Dedicated adjoint kernels with multicomponent OBN support
- **Optimization**: SEISCOPE L-BFGS/CG algorithms (TOOLBOX_OPTIMIZATION)

### Key Design Principles
- **Forward kernels untouched**: No modifications to `elastic4.c`, `applySource.c`, etc.
- **Dedicated adjoint kernels**: Separate `elastic4_adj.c` + `applyAdjointSource.c` for multicomponent residual injection
- **Disk-based re-propagation checkpointing**: Forward wavefield saved at coarse intervals; re-propagated during adjoint pass for exact gradients
- **Multicomponent OBN support**: Simultaneous backpropagation of hydrophone, Vx, and Vz residuals in one pass
- **MPI parallelism**: Shot-level distribution across ranks (Phase 5)
- **ISO_C_BINDING**: Fortran optimizer calls C physics engine

---

## 2. Repository Capabilities

### OpenSource_SL10/fdelfwi

| Component | File | Status |
|-----------|------|--------|
| Forward FD kernels | `elastic4.c`, `elastic6.c` | Existing, unmodified |
| Acoustic FD kernels | `acoustic4.c`, `acoustic6.c` | Existing, unmodified |
| Forward source injection | `applySource.c` | Existing, unmodified |
| Forward modeling driver | `fdfwimodc.c` | Modified (checkpoint support) |
| Boundary conditions | `boundaries.c` | Existing, unmodified |
| Checkpoint I/O | `checkpoint.c` | **New** |
| Residual reader | `readResidual.c` | **New** |
| Adjoint source injection | `applyAdjointSource.c` | **New** |
| Adjoint elastic kernel | `elastic4_adj.c` | **New** |
| Adjoint backpropagation | `adj_shot.c` | **New** |
| Data structures | `fdelfwi.h` | Modified (adjSrcPar, checkpointPar) |

### TOOLBOX_OPTIMIZATION

| Algorithm | Location | Best For |
|-----------|----------|----------|
| **L-BFGS** | `LBFGS/kernel/src/LBFGS.f90` | Large-scale FWI (recommended) |
| **PNLCG** | `PNLCG/kernel/src/PNLCG.f90` | Alternative first-order method |
| **TRN** | `TRN/kernel/src/TRN.f90` | Newton-based (needs Hessian-vector) |

**Reverse Communication Protocol**:
```
INIT -> GRAD -> [linesearch] -> NSTE -> GRAD -> ... -> CONV/FAIL
```

---

## 3. Architecture

```
+---------------------------------------------------------------------+
|                     TOOLBOX_OPTIMIZATION                            |
|                   (Fortran L-BFGS/CG Driver)                        |
|                                                                     |
|  optim%xk = [Vp, Vs, rho]    optim%grad = [dJ/dVp, dJ/dVs, dJ/drho]|
+---------------------------------------------------------------------+
                              | ISO_C_BINDING
                              v
+---------------------------------------------------------------------+
|                      fwi_wrapper.f90 (Phase 5)                      |
|              (Fortran wrapper with C bindings)                      |
+---------------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------+
|                       fwi_driver.c (Phase 5)                        |
|                   (C/MPI Shot Orchestrator)                         |
|                                                                     |
|  - Distribute shots across MPI ranks                                |
|  - Call fdfwimodc() and adj_shot() for each shot                    |
|  - MPI_Allreduce to accumulate gradients                            |
+---------------------------------------------------------------------+
                    +---------+---------+
                    v                   v
+--------------------------+  +--------------------------+
|     fdfwimodc.c          |  |      adj_shot.c          |
|   (Forward Modeling)     |  |   (Adjoint + Gradient)   |
|                          |  |                          |
| - Existing forward code  |  | - Re-propagate from      |
| - elastic4/6 kernels     |  |   disk checkpoints       |
| - applySource injection  |  | - elastic4_adj kernel    |
| - Writes disk checkpoints|  | - applyAdjointSource     |
| - Writes synthetic .su   |  |   (multicomponent)       |
|                          |  | - accumGradient at every |
|                          |  |   time step              |
+--------------------------+  +--------------------------+
          |                             |
          v                             v
   +-------------+              +--------------+
   | checkpoint  |              | readResidual |
   | files (disk)|              | (.su files)  |
   +-------------+              +--------------+
```

### Key Data Flow

```
For each shot:

  1. fdfwimodc(chk)  -->  synthetic.su  +  checkpoint files on disk
  2. residual = d_obs - d_syn  (external)
  3. readResidual("residual.su")  -->  adjSrcPar (per-source types)
  4. adj_shot()  -->  grad_l2m, grad_lam, grad_muu, grad_rho
     For each segment (reverse):
       a. readCheckpoint() --> wfl_fwd
       b. Re-propagate forward: callKernel() storing vx/vz in buffer
       c. Sweep adjoint backward: callAdjKernel() using elastic4_adj
       d. Cross-correlate: accumGradient() at every time step
```

---

## 4. Implementation Details

### 4.1 Forward Modeling: fdfwimodc.c (Modified)

The existing `fdfwimodc()` function is extended with an optional `checkpointPar *chk` parameter:

```c
/* Signature (modified) */
int fdfwimodc(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
              recPar *rec, snaPar *sna,
              int ixsrc, int izsrc, float **src_nwav,
              int ishot, int nshots, int fileno,
              checkpointPar *chk,  /* NEW: NULL = no checkpoints */
              int verbose);
```

When `chk != NULL`, writes all 5 wavefield components (vx, vz, txx, tzz, txz) to disk at each checkpoint time step.

### 4.2 Adjoint Engine: adj_shot.c (New)

```c
int adj_shot(modPar *mod, srcPar *src, wavPar *wav, bndPar *bnd,
             recPar *rec, adjSrcPar *adj,
             int ixsrc, int izsrc, float **src_nwav,
             checkpointPar *chk,
             float *grad_l2m, float *grad_lam,
             float *grad_muu, float *grad_rho,
             int verbose);
```

Internal dispatch:
- `callKernel()` -- Forward re-propagation using unmodified elastic4/6 kernels
- `callAdjKernel()` -- Adjoint propagation using elastic4_adj
- `accumGradient()` -- Zero-lag cross-correlation with 4th/6th-order FD stencils

### 4.3 Adjoint Source Injection: applyAdjointSource.c (New)

Handles multicomponent OBN residuals with per-source type dispatch:

| OBN Component | adj.typ | Injection Target | Phase |
|---------------|---------|-----------------|-------|
| Hydrophone (P) | 1 | `tzz += r`, `txx += r` | 2 (after stress update) |
| Geophone Vx | 6 | `vx += r` | 1 (after velocity update) |
| Geophone Vz | 7 | `vz += r` | 1 (after velocity update) |

All types injected simultaneously in one backward pass -- no efficiency penalty for multicomponent data.

### 4.4 Adjoint FD Kernel: elastic4_adj.c (New)

Same FD stencils as `elastic4.c` but with:
- Two calls to `applyAdjointSource()` at correct phase points
- No `applySource()` calls (no forward source)
- No `storeSourceOnSurface()` / `reStoreSourceOnSurface()` calls
- No material property scaling on injected residuals

### 4.5 Checkpoint I/O: checkpoint.c (New)

Disk-based storage of complete wavefield states:

```c
int  initCheckpoints(checkpointPar *chk, modPar *mod, int skipdt);
int  writeCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl);
int  readCheckpoint(checkpointPar *chk, int isnap, wflPar *wfl);
void cleanCheckpoints(checkpointPar *chk);
```

### 4.6 Gradient Formulas

| Parameter | Formula | Grid Position |
|-----------|---------|--------------|
| l2m (P-wave modulus) | `-dt * div_fwd * div_adj` | P/Txx/Tzz |
| lam (first Lame) | `-dt * (dvx/dx_fwd * dvz/dz_adj + dvz/dz_fwd * dvx/dx_adj)` | P/Txx/Tzz |
| muu (shear modulus) | `-dt * curl_fwd * curl_adj` | Txz |
| rho (density) | `+dt * (vx_fwd * vx_adj + vz_fwd * vz_adj)` | Vx, Vz |

Chain rule for Vp/Vs/rho parameterization:
- `g_Vp = 2*rho*Vp * g_l2m`
- `g_Vs = 2*rho*Vs * (g_muu - 2*g_lam)`
- `g_rho_full = Vp^2 * g_l2m + (Vp^2 - 2*Vs^2) * g_lam + Vs^2 * g_muu + g_rho`

---

## 5. C/Fortran Binding Layer

### C Header for Fortran (`fwi_interface.h`, Phase 5)

```c
#ifndef FWI_INTERFACE_H
#define FWI_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

int fwi_init(int nz, int nx, float dz, float dx, float dt, int nt,
             const char *data_path, const char *model_path);

int compute_gradient(int n, float *x, float *fcost, float *grad);

int fwi_finalize(void);

#ifdef __cplusplus
}
#endif
#endif
```

### Fortran Wrapper (`fwi_wrapper.f90`, Phase 5)

```fortran
module fwi_interface
    use, intrinsic :: iso_c_binding
    implicit none

    interface
        function compute_gradient_c(n, x, fcost, grad) bind(C, name='compute_gradient')
            import :: c_int, c_float, c_ptr
            integer(c_int), value :: n
            type(c_ptr), value :: x, fcost, grad
            integer(c_int) :: compute_gradient_c
        end function
    end interface

contains
    subroutine compute_fwi_gradient(n, x, fcost, grad)
        integer, intent(in) :: n
        real, intent(in), target :: x(n)
        real, intent(out), target :: fcost
        real, intent(out), target :: grad(n)
        type(c_ptr) :: x_ptr, fcost_ptr, grad_ptr
        integer :: ierr

        x_ptr = c_loc(x(1))
        fcost_ptr = c_loc(fcost)
        grad_ptr = c_loc(grad(1))
        ierr = compute_gradient_c(n, x_ptr, fcost_ptr, grad_ptr)
    end subroutine
end module fwi_interface
```

---

## 6. Data Flow and File Formats

### SU File Conventions

**Observed/Synthetic Data**:
```
Format: SEG-Y (SU header + float32 traces)
Key headers: ns, dt, gx, gelev, sx, selev, scalco, trid
```

**Multicomponent Residual Files** (OBN):
```
residual_rp.su   -- Hydrophone (trid=1)
residual_rvx.su  -- Geophone Vx (trid=6)
residual_rvz.su  -- Geophone Vz (trid=7)

Combined into single adjSrcPar with per-source typ[] arrays.
```

**TRID Encoding**:
```
typ    = (trid - 1) % 8 + 1    // source type 1-8
orient = (trid - 1) / 8 + 1    // orientation 1-3
```

### Checkpoint Files (Disk-Based)

```
chk_vx_NNNN.bin   -- Velocity x-component at checkpoint NNNN
chk_vz_NNNN.bin   -- Velocity z-component
chk_tzz_NNNN.bin  -- Stress tzz (or pressure for acoustic)
chk_txx_NNNN.bin  -- Stress txx (elastic only)
chk_txz_NNNN.bin  -- Stress txz (elastic only)

Size per checkpoint: 5 * nax * naz * 4 bytes (elastic)
```

---

## 7. Build System Integration

### Current Directory Structure

```
OpenSource_SL10/
├── fdelfwi/                      # FWI forward + adjoint module
│   ├── fdelfwi.h                 # All data structures
│   ├── fdfwimodc.c               # Forward modeling (modified: +chk)
│   ├── test_fdfwimodc.c          # Test driver
│   ├── Makefile                  # Build system
│   │
│   ├── elastic4.c                # Forward 4th-order elastic kernel
│   ├── elastic6.c                # Forward 6th-order elastic kernel
│   ├── acoustic4.c               # Forward 4th-order acoustic kernel
│   ├── applySource.c             # Forward source injection
│   ├── boundaries.c              # Boundary conditions
│   ├── readModel.c               # Model I/O
│   ├── ... (other existing files)
│   │
│   ├── readResidual.c            # NEW: Read residual SU -> adjSrcPar
│   ├── checkpoint.c              # NEW: Disk checkpoint I/O
│   ├── adj_shot.c                # NEW: Adjoint backpropagation
│   ├── applyAdjointSource.c      # NEW: Per-source type adjoint injection
│   ├── elastic4_adj.c            # NEW: 4th-order elastic adjoint kernel
│   │
│   ├── backpropagation.md        # Detailed adjoint module design
│   └── demo/
│       ├── test_topo.sh          # Elastic topography test with verification
│       └── ...
│
├── fdelmodc/                     # Reference forward modeling code
├── fdacrtmc/                     # RTM (reference for adjoint patterns)
├── ELASTIC_FWI_INTEGRATION.md    # This file
└── Make_include                  # Shared build config
```

### Makefile (fdelfwi/Makefile)

All new files are added to the `SRCC` list and compile with zero warnings:
```makefile
SRCC = test_fdfwimodc.c \
       fdfwimodc.c \
       ... (existing files) ...
       checkpoint.c \
       adj_shot.c \
       applyAdjointSource.c \
       elastic4_adj.c \
       ... (existing files) ...
```

---

## 8. Implementation Status

### Completed

| Phase | Component | Files | Status |
|-------|-----------|-------|--------|
| 1 | Residual Reader | `readResidual.c`, `fdelfwi.h` (adjSrcPar) | DONE |
| 2 | Forward Checkpointing | `checkpoint.c`, `fdfwimodc.c` (modified) | DONE |
| 3 | Adjoint Backpropagation | `adj_shot.c` (callKernel, callAdjKernel, accumGradient) | DONE |
| 4 | Adjoint FD Kernels | `elastic4_adj.c`, `applyAdjointSource.c` | DONE (elastic 4th-order) |

### In Progress / TODO

| Phase | Component | Files | Status |
|-------|-----------|-------|--------|
| 4+ | Additional adjoint kernels | `elastic6_adj.c`, `acoustic4_adj.c`, `acoustic6_adj.c` | TODO |
| 5 | MPI Driver | `fwi_driver.c` | NOT STARTED |
| 5 | C/Fortran Binding | `fwi_interface.h`, `fwi_wrapper.f90` | NOT STARTED |
| 5 | Fortran Main | `fwi_main.f90` | NOT STARTED |
| -- | Adjoint Dot Product Test | test scripts | TODO |
| -- | Finite-Difference Gradient Test | test scripts | TODO |

### Build Status

All code compiles with **zero errors and zero warnings**. Forward modeling tests pass:
- Acoustic test: PASS (0.001% error vs reference `fdelmodc`)
- Elastic with topography: PASS (<1.0% error)
- Snapshot comparison: PASS (<3.0% error, 5 time slices x 2 components)

---

## 9. Testing Strategy

### 9.1 Forward Modeling Verification (PASS)

Quantitative comparison between `fdelmodc` (reference) and `test_fdfwimodc` using `test_topo.sh`:
- Receiver data: relative error threshold 1.0%
- Snapshot data: relative error threshold 3.0% (5 time slices, txx and tzz)

### 9.2 Adjoint Dot Product Test (TODO)

```
<d, F*m> = <F^T*d, m>
```

Where F is the forward operator and F^T is the adjoint. The ratio should be 1.0 to machine precision.

### 9.3 Gradient Accuracy Test (TODO)

Finite-difference gradient approximation:
```
g_fd = [J(m + eps*dm) - J(m - eps*dm)] / (2*eps)
```

Compare with adjoint-computed gradient. Agreement within 1-5% for reasonable `eps`.

### 9.4 Synthetic Inversion Test (TODO)

1. Create true model with anomaly
2. Generate synthetic observed data
3. Start from smooth initial model
4. Run FWI, verify convergence and model recovery

---

## Appendix A: Common Issues and Solutions

### Issue 1: Gradient Sign Convention
- **Problem**: Misfit increases instead of decreasing
- **Solution**: Verify sign in gradient kernel; descent direction is `-gradient`

### Issue 2: Boundary Artifacts in Gradient
- **Problem**: Strong gradients at model boundaries
- **Solution**: Extend taper to gradient computation; mask boundary region

### Issue 3: Source/Receiver Positioning
- **Problem**: Mismatch between forward and adjoint source locations
- **Solution**: Use same coordinate system; verify `scalco` header handling; ensure staggered-grid offsets match between `readResidual` (adjSrcPar) and `getRecTimes` (forward recording)

### Issue 4: Multicomponent Type Mismatch
- **Problem**: Wrong residual injected into wrong wavefield component
- **Solution**: Verify TRID headers in residual SU files match the component types; typ 1=P (-> tzz+txx), typ 6=Fx (-> vx), typ 7=Fz (-> vz)

### Issue 5: Checkpoint Segment Boundaries
- **Problem**: Gradient artifacts at checkpoint boundaries
- **Solution**: The re-propagation approach avoids this by computing the gradient at every time step within each segment. Verify that `callKernel()` during re-propagation includes the source wavelet injection.

---

*Document version 2.0 -- Updated to reflect implemented architecture (Phases 1-4 complete)*
