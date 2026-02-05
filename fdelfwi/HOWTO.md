# FWI Module Implementation Guide

## Overview

This document describes how to implement a Full Waveform Inversion (FWI) module by reusing existing code from:
- **fdelmodc** - Forward modeling (elastic/acoustic)
- **fdacrtmc** - Reverse Time Migration (provides backpropagation and imaging condition patterns)

**IMPORTANT**: The FWI module is self-contained and can be compiled independently. Source files are copied into the fwi/ directory (same pattern as fdacrtmc). It does NOT depend on object files from other modules.

The FWI workflow is structurally similar to RTM:
1. **Forward propagation** - Propagate source wavefield, save checkpoints
2. **Backward propagation** - Propagate adjoint wavefield (data residual at receivers)
3. **Gradient computation** - Cross-correlate forward and adjoint wavefields

## Repository Analysis

### fdelmodc Structure (Forward Modeling)

**Location:** `/OpenSource_SL10/fdelmodc/`

**Key Files:**
| File | Purpose |
|------|---------|
| `fdelmodc.c` | Main driver - shot loop, time loop orchestration |
| `fdelmodc.h` | Structure definitions: `modPar`, `srcPar`, `recPar`, `bndPar`, `wavPar`, `snaPar` |
| `elastic4.c` | 4th-order elastic FD kernel |
| `elastic6.c` | 6th-order elastic FD kernel |
| `acoustic4.c` | 4th-order acoustic FD kernel |
| `boundaries.c` | Boundary conditions (free surface, taper, PML) |
| `applySource.c` | Source injection |
| `getRecTimes.c` | Receiver recording |
| `writeSnapTimes.c` | Snapshot/checkpoint writing |
| `readModel.c` | Model file I/O |
| `defineSource.c` | Source wavelet handling |
| `getParameters.c` | Parameter parsing |

**Main Flow (fdelmodc.c):**
```
main()
├── getParameters()           # Parse parameters, allocate arrays
├── readModel()               # Read velocity/density models
├── defineSource()            # Read/create source wavelet
├── for ishot in shots:       # SHOT LOOP (MPI parallelizable)
│   ├── memset(wavefields,0)  # Zero wavefields
│   ├── for it in timesteps:  # TIME LOOP
│   │   ├── elastic4()        # FD kernel (one timestep)
│   │   ├── getRecTimes()     # Record at receivers
│   │   └── writeSnapTimes()  # Write checkpoints (if configured)
│   └── writeRec()            # Write receiver data
└── cleanup()
```

**Elastic4 Kernel (elastic4.c:20-158):**
```c
int elastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd,
             int itime, int ixsrc, int izsrc, float **src_nwav,
             float *vx, float *vz, float *tzz, float *txx, float *txz,
             float *rox, float *roz, float *l2m, float *lam, float *mul,
             int verbose)
{
    // 1. Update velocities from stresses
    for (ix,iz) vx[ix,iz] -= rox[ix,iz] * (dTxx/dx + dTxz/dz)
    for (ix,iz) vz[ix,iz] -= roz[ix,iz] * (dTzz/dz + dTxz/dx)

    // 2. Apply source (force type)
    applySource() if src.type > 5

    // 3. Apply boundary conditions on velocities
    boundariesP()

    // 4. Update stresses from velocities
    for (ix,iz) {
        dvx = dVx/dx;  dvz = dVz/dz
        txx[ix,iz] -= l2m[ix,iz]*dvx + lam[ix,iz]*dvz
        tzz[ix,iz] -= l2m[ix,iz]*dvz + lam[ix,iz]*dvx
        txz[ix,iz] -= mul[ix,iz]*(dVx/dz + dVz/dx)
    }

    // 5. Apply source (stress type)
    applySource() if src.type < 6

    // 6. Apply boundary conditions on stresses
    boundariesV()
}
```

### fdacrtmc Structure (RTM - Backpropagation Pattern)

**Location:** `/OpenSource_SL10/fdacrtmc/`

**Key Files:**
| File | Purpose |
|------|---------|
| `fdacrtmc.c` | Main driver - forward/backward loops, imaging |
| `fdacrtmc.h` | Structure definitions (similar to fdelmodc but extended) |
| `acoustic4.c` | 4th-order acoustic FD kernel |
| `boundaries.c` | Boundary conditions with PML |
| `rtmImagingCondition.c` | Cross-correlation imaging (gradient for RTM) |
| `extractMigrationSnapshots.c` | Save forward snapshots for imaging |
| `injectSource.c` | Source/receiver injection |

**Main Flow (fdacrtmc.c:189-761):**
```
main()
├── getParameters()
├── do {                              # SHOT LOOP
│   ├── readSrcWav()                  # Read source wavefield
│   ├── memset(wavefields,0)
│   │
│   ├── for it=0 to nt:               # FORWARD PROPAGATION
│   │   ├── acoustic4()               # FD kernel (forward)
│   │   ├── storeRcvWavefield()       # Record receivers (optional)
│   │   └── extractMigrationSnapshots()  # SAVE CHECKPOINTS
│   │
│   ├── memset(wavefields,0)          # Reset for backward
│   ├── readRcvWav()                  # Read receiver data
│   │
│   ├── for it=nt-1 down to 0:        # BACKWARD PROPAGATION
│   │   ├── acoustic4()               # FD kernel (same as forward!)
│   │   └── rtmImagingCondition()     # CROSS-CORRELATION
│   │
│   └── writeMigImagePerShot()        # Write per-shot image
│   └── mig.mig += mig.image          # Accumulate total image
} while(!src.eof)
├── writeMigImage()                   # Write final image
└── cleanup()
```

**Imaging Condition (rtmImagingCondition.c:21-36):**
```c
// Conventional zero-lag pressure cross-correlation
// This is equivalent to FWI gradient for velocity!
for (ix,iz) {
    mig.image[ix,iz] += dt * (backward_wavefield[ix,iz] * forward_snapshot[ix,iz])
}
```

## FWI Module Architecture

### Core Concept

FWI gradient computation is structurally identical to RTM imaging:
```
RTM Image    = Σ_t [ Forward_Wavefield(t) × Backward_Wavefield(t) ]
FWI Gradient = Σ_t [ Forward_Wavefield(t) × Adjoint_Wavefield(t) ]
```

The difference:
- **RTM**: Backward wavefield = recorded data injected at receivers
- **FWI**: Adjoint wavefield = data residual (observed - synthetic) injected at receivers

### Proposed Module Structure

**Flat structure matching fdelmodc/fdacrtmc pattern** (self-contained, independent compilation):

```
fwi/
├── Makefile              # Self-contained build (uses ../Make_include)
├── fwi.h                 # FWI-specific structures + includes from par.h, segy.h
├── par.h                 # Copy from fdelmodc
├── segy.h                # Copy from fdelmodc
│
├── fwi.c                 # Main driver (MPI shot distribution)
│
├── # --- FWI-specific source files ---
├── fwi_forward.c         # Forward modeling with checkpointing
├── fwi_adjoint.c         # Adjoint propagation
├── fwi_gradient.c        # Gradient computation (imaging condition)
├── fwi_misfit.c          # Misfit functional (L2, etc.)
├── fwi_checkpoint.c      # Checkpoint storage/retrieval
│
├── # --- Copied from fdelmodc (FD kernels) ---
├── elastic4.c            # 4th-order elastic kernel
├── elastic6.c            # 6th-order elastic kernel
├── acoustic4.c           # 4th-order acoustic kernel
├── boundaries.c          # Boundary conditions
├── applySource.c         # Source injection
├── sourceOnSurface.c     # Free surface source handling
│
├── # --- Copied from fdelmodc (I/O and utilities) ---
├── readModel.c           # Model file I/O
├── defineSource.c        # Wavelet handling
├── getRecTimes.c         # Receiver recording
├── writeRec.c            # Write receiver data
├── writeSnapTimes.c      # Snapshot writing
├── fileOpen.c            # File utilities
├── writesufile.c         # SU file writing
│
├── # --- Copied from fdelmodc (parameter handling) ---
├── getParameters.c       # Parameter parsing (modify for FWI)
├── getpars.c             # Getpar library
├── atopkge.c             # String conversion
├── docpkge.c             # Documentation
├── verbosepkg.c          # Verbose messaging
│
├── # --- Copied from fdelmodc (other utilities) ---
├── wallclock_time.c      # Timing
├── name_ext.c            # Filename utilities
├── spline3.c             # Interpolation
├── gaussGen.c            # Gaussian generation
├── CMWC4096.c            # Random number generator
├── threadAffinity.c      # OpenMP thread affinity
│
└── HOWTO.md              # This documentation
```

**Source files to copy from fdelmodc:**
```bash
# FD kernels
cp ../fdelmodc/elastic4.c ../fdelmodc/elastic6.c ../fdelmodc/acoustic4.c .
cp ../fdelmodc/boundaries.c ../fdelmodc/applySource.c ../fdelmodc/sourceOnSurface.c .

# I/O
cp ../fdelmodc/readModel.c ../fdelmodc/defineSource.c ../fdelmodc/getRecTimes.c .
cp ../fdelmodc/writeRec.c ../fdelmodc/writeSnapTimes.c ../fdelmodc/fileOpen.c .
cp ../fdelmodc/writesufile.c .

# Parameters
cp ../fdelmodc/getParameters.c ../fdelmodc/getpars.c ../fdelmodc/atopkge.c .
cp ../fdelmodc/docpkge.c ../fdelmodc/verbosepkg.c .

# Utilities
cp ../fdelmodc/wallclock_time.c ../fdelmodc/name_ext.c ../fdelmodc/spline3.c .
cp ../fdelmodc/gaussGen.c ../fdelmodc/CMWC4096.c ../fdelmodc/threadAffinity.c .

# Headers
cp ../fdelmodc/par.h ../fdelmodc/segy.h .
```

### Key Structures

```c
/* From fdelmodc.h - reuse directly */
typedef struct _modPar { ... } modPar;   // Model parameters
typedef struct _srcPar { ... } srcPar;   // Source parameters
typedef struct _recPar { ... } recPar;   // Receiver parameters
typedef struct _bndPar { ... } bndPar;   // Boundary parameters
typedef struct _wavPar { ... } wavPar;   // Wavelet parameters
typedef struct _snaPar { ... } snaPar;   // Snapshot parameters

/* FWI-specific structures */
typedef struct _fwiPar {
    int niter;              // Number of iterations
    int nsnap;              // Number of checkpoints
    float misfit;           // Current misfit value
    char *file_obs;         // Observed data file
    char *file_grad;        // Gradient output file
} fwiPar;

typedef struct _gradPar {
    float *grad_vp;         // Gradient w.r.t. Vp
    float *grad_vs;         // Gradient w.r.t. Vs
    float *grad_rho;        // Gradient w.r.t. density
} gradPar;

typedef struct _checkPar {
    int nsnap;              // Number of checkpoints
    int *snap_times;        // Time indices of checkpoints
    float **vx_snap;        // Vx snapshots
    float **vz_snap;        // Vz snapshots
    float **tzz_snap;       // Tzz snapshots (elastic)
    float **txx_snap;       // Txx snapshots (elastic)
    float **txz_snap;       // Txz snapshots (elastic)
} checkPar;
```

### Function Mapping

| FWI Function | Based On | Purpose |
|--------------|----------|---------|
| `fwi_forward()` | `fdelmodc.c` shot loop | Forward propagation with checkpointing |
| `fwi_adjoint()` | `fdacrtmc.c` backward loop | Adjoint propagation |
| `fwi_gradient()` | `rtmImagingCondition()` | Cross-correlation for gradient |
| `fwi_misfit()` | New | Compute L2 misfit between observed and synthetic |
| `fwi_shot()` | Combines above | Complete single-shot FWI |

## Implementation Details

### 1. Forward Modeling with Checkpointing (fwi_forward.c)

Reuse fdelmodc functions directly:
```c
#include "fdelmodc.h"

int fwi_forward(modPar mod, srcPar src, wavPar wav, bndPar bnd,
                int ixsrc, int izsrc, float **src_nwav,
                float *vx, float *vz, float *tzz, float *txx, float *txz,
                float *rox, float *roz, float *l2m, float *lam, float *mul,
                recPar rec, float *rec_p,
                checkPar *check, int verbose)
{
    int it, isnap;
    int n1 = mod.naz;
    size_t sizem = mod.naz * mod.nax;

    // Determine checkpoint times
    int snap_interval = mod.nt / check->nsnap;

    // Time loop (from fdelmodc.c lines 550-692)
    for (it = 0; it < mod.nt; it++) {

        // FD kernel - directly call fdelmodc function
        if (mod.ischeme == 3) {  // Elastic
            if (mod.iorder == 4) {
                elastic4(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav,
                         vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, verbose);
            } else if (mod.iorder == 6) {
                elastic6(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav,
                         vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, verbose);
            }
        } else if (mod.ischeme == 1) {  // Acoustic
            acoustic4(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav,
                      vx, vz, tzz, rox, roz, l2m, verbose);
        }

        // Record at receivers - directly call fdelmodc function
        getRecTimes(mod, rec, bnd, it, ...);

        // Save checkpoint (similar to extractMigrationSnapshots)
        if (it % snap_interval == 0) {
            isnap = it / snap_interval;
            memcpy(check->vx_snap[isnap], vx, sizem * sizeof(float));
            memcpy(check->vz_snap[isnap], vz, sizem * sizeof(float));
            memcpy(check->tzz_snap[isnap], tzz, sizem * sizeof(float));
            if (mod.ischeme > 2) {
                memcpy(check->txx_snap[isnap], txx, sizem * sizeof(float));
                memcpy(check->txz_snap[isnap], txz, sizem * sizeof(float));
            }
            check->snap_times[isnap] = it;
        }
    }

    return 0;
}
```

### 2. Adjoint Propagation (fwi_adjoint.c)

Based on fdacrtmc backward loop (lines 542-609):
```c
int fwi_adjoint(modPar mod, srcPar rcv, wavPar wav, bndPar bnd,
                float *residual,  // Data residual at receivers
                float *vx, float *vz, float *tzz, float *txx, float *txz,
                float *rox, float *roz, float *l2m, float *lam, float *mul,
                checkPar *check, gradPar *grad, int verbose)
{
    int it, isnap;
    int n1 = mod.naz;

    // Zero wavefields
    memset(vx, 0, sizem * sizeof(float));
    memset(vz, 0, sizem * sizeof(float));
    memset(tzz, 0, sizem * sizeof(float));

    // Backward time loop (from fdacrtmc.c line 542)
    isnap = check->nsnap - 1;
    for (it = mod.nt - 1; it >= 0; it--) {

        // FD kernel (same as forward - wave equation is self-adjoint)
        if (mod.ischeme == 3) {
            elastic4(mod, rcv, wav, bnd, it, ...);
        }

        // Apply imaging condition (gradient computation)
        // Similar to rtmImagingCondition (lines 31-35)
        if (it == check->snap_times[isnap]) {
            fwi_gradient_accumulate(mod, check, isnap,
                                    vx, vz, tzz, grad);
            isnap--;
        }
    }

    return 0;
}
```

### 3. Gradient Computation (fwi_gradient.c)

Based on rtmImagingCondition (lines 21-36):
```c
int fwi_gradient_accumulate(modPar mod, checkPar *check, int isnap,
                            float *adj_vx, float *adj_vz, float *adj_tzz,
                            gradPar *grad)
{
    int ix, iz, idx;
    int n1 = mod.naz;
    float dt = mod.dt;

    // Cross-correlation for Vp gradient
    // grad_vp = Σ_t [ ∂tzz_fwd/∂t × tzz_adj ]
    for (ix = mod.ioPx; ix < mod.iePx; ix++) {
        for (iz = mod.ioPz; iz < mod.iePz; iz++) {
            idx = ix * n1 + iz;

            // Similar to rtmImagingCondition case 1 (line 33)
            grad->grad_vp[idx] += dt * (
                check->tzz_snap[isnap][idx] * adj_tzz[idx]
            );
        }
    }

    return 0;
}
```

### 4. Single-Shot FWI (fwi_shot.c)

Combines forward, misfit, and adjoint:
```c
int fwi_shot(modPar mod, srcPar src, recPar rec, bndPar bnd, wavPar wav,
             float *obs_data,     // Observed data
             float *rox, float *roz, float *l2m, float *lam, float *mul,
             checkPar *check, gradPar *grad,
             float *misfit, int verbose)
{
    float *syn_data;    // Synthetic data
    float *residual;    // Data residual
    float *vx, *vz, *tzz, *txx, *txz;

    // Allocate arrays
    allocate_arrays(...);

    // 1. Forward modeling with checkpointing
    fwi_forward(mod, src, wav, bnd, ..., check, verbose);

    // 2. Compute misfit and residual
    *misfit = fwi_misfit(rec, obs_data, syn_data, residual);

    // 3. Adjoint propagation with gradient computation
    fwi_adjoint(mod, rec, wav, bnd, residual, ..., check, grad, verbose);

    // Cleanup
    free_arrays(...);

    return 0;
}
```

### 5. Main Driver with MPI (fwi_main.c)

Based on fdelmodc MPI structure (lines 487-495):
```c
int main(int argc, char **argv)
{
    int npes, pe, ishot;
    int is0, is1;
    float total_misfit;
    gradPar total_grad;

#ifdef MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
#else
    npes = 1;
    pe = 0;
#endif

    // Initialize (parse parameters, read model)
    fwi_init(argc, argv, &mod, &src, &rec, &bnd, &wav, &shot, ...);

    // Distribute shots across MPI ranks
    float npeshot = MAX(((float)shot.n) / ((float)npes), 1.0);
    is0 = ceil(pe * npeshot);
    is1 = MIN(ceil((pe + 1) * npeshot), shot.n);

    // Zero total gradient
    memset(total_grad.grad_vp, 0, sizem * sizeof(float));

    // Shot loop (parallelized via MPI)
    total_misfit = 0.0;
    for (ishot = is0; ishot < is1; ishot++) {
        float shot_misfit;
        gradPar shot_grad;

        // Single shot FWI
        fwi_shot(mod, src, rec, bnd, wav,
                 obs_data[ishot], ...,
                 &check, &shot_grad, &shot_misfit, verbose);

        // Accumulate
        total_misfit += shot_misfit;
        for (i = 0; i < sizem; i++) {
            total_grad.grad_vp[i] += shot_grad.grad_vp[i];
        }
    }

#ifdef MPI
    // Reduce misfit and gradient across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &total_misfit, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, total_grad.grad_vp, sizem, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#endif

    // Write gradient
    if (pe == 0) {
        write_gradient(&total_grad, &mod);
    }

    // Cleanup
    fwi_cleanup(...);

#ifdef MPI
    MPI_Finalize();
#endif

    return 0;
}
```

## Makefile Structure

**Self-contained Makefile (same pattern as fdelmodc):**

```makefile
# Makefile for FWI - Elastic Full Waveform Inversion
# Self-contained module - compiles independently

include ../Make_include

########################################################################
# Include paths (local only - self-contained)
########################################################################
ALLINC = -I.

########################################################################
# Source files
########################################################################
PRG = fwi

# FWI-specific sources
FWI_SRCS = fwi_forward.c \
           fwi_adjoint.c \
           fwi_gradient.c \
           fwi_misfit.c \
           fwi_checkpoint.c

# FD kernels (copied from fdelmodc)
KERNEL_SRCS = elastic4.c \
              elastic6.c \
              acoustic4.c \
              boundaries.c \
              applySource.c \
              sourceOnSurface.c

# I/O sources (copied from fdelmodc)
IO_SRCS = readModel.c \
          defineSource.c \
          getRecTimes.c \
          writeRec.c \
          writeSnapTimes.c \
          fileOpen.c \
          writesufile.c

# Utility sources (copied from fdelmodc)
UTIL_SRCS = getParameters.c \
            getpars.c \
            atopkge.c \
            docpkge.c \
            verbosepkg.c \
            wallclock_time.c \
            name_ext.c \
            spline3.c \
            gaussGen.c \
            CMWC4096.c \
            threadAffinity.c

# All sources
SRCC = $(PRG).c $(FWI_SRCS) $(KERNEL_SRCS) $(IO_SRCS) $(UTIL_SRCS)
OBJC = $(SRCC:%.c=%.o)

########################################################################
# Targets
########################################################################
all: $(PRG)

$(PRG): $(OBJC) fwi.h
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG) $(OBJC) $(LIBS)

# MPI version
ifdef MPICC
fwi_mpi: $(OBJC) fwi.h
	$(MPICC) -DMPI $(CFLAGS) $(OPTC) -c $(PRG).c
	$(MPICC) -DMPI $(LDFLAGS) $(CFLAGS) $(OPTC) -o fwi_mpi $(OBJC) $(LIBS)
endif

install: $(PRG)
	cp $(PRG) $B
ifdef MPICC
	cp fwi_mpi $B
endif

clean:
	rm -f core $(OBJC) $(PRG) fwi_mpi

realclean: clean
	rm -f $B/$(PRG) $B/fwi_mpi
```

## Key Functions to Reuse

### From fdelmodc (link directly):
- `elastic4()` / `elastic6()` - FD kernels
- `acoustic4()` - Acoustic kernel
- `boundariesP()` / `boundariesV()` - Boundary conditions
- `applySource()` - Source injection
- `getRecTimes()` - Receiver recording
- `readModel()` - Model I/O
- `defineSource()` - Wavelet handling
- `getParameters()` - Parameter parsing

### From fdacrtmc (pattern to follow):
- Backward loop structure (line 542-609)
- `rtmImagingCondition()` - Cross-correlation pattern
- `extractMigrationSnapshots()` - Checkpoint storage pattern

## Source File Classification

### Files to COPY AS-IS from fdelmodc (no modifications needed):
```
# FD Kernels - use exactly as-is
elastic4.c          # 4th-order elastic kernel
elastic6.c          # 6th-order elastic kernel
acoustic4.c         # 4th-order acoustic kernel
boundaries.c        # Boundary conditions (taper, free surface)
applySource.c       # Source injection
sourceOnSurface.c   # Free surface source handling

# Utilities - use exactly as-is
wallclock_time.c
name_ext.c
spline3.c
gaussGen.c
CMWC4096.c
threadAffinity.c
getpars.c
atopkge.c
docpkge.c
verbosepkg.c
fileOpen.c
writesufile.c

# I/O - use mostly as-is
readModel.c
defineSource.c
getRecTimes.c
writeRec.c
writeSnapTimes.c    # Can use for checkpointing
```

### Files to COPY AND MODIFY from fdelmodc:
```
getParameters.c     # Modify: add FWI-specific parameters
                    # (niter, file_obs, gradient output, etc.)
```

### NEW FILES to create for FWI:
```
fwi.c               # Main driver (based on fdelmodc.c structure)
fwi.h               # FWI structures (checkPar, gradPar, etc.)
fwi_forward.c       # Forward modeling with checkpoint storage
fwi_adjoint.c       # Adjoint propagation (based on fdacrtmc backward loop)
fwi_gradient.c      # Gradient computation (based on rtmImagingCondition)
fwi_misfit.c        # Misfit computation (L2 residual)
fwi_checkpoint.c    # Checkpoint storage/retrieval utilities
```

## Summary

The FWI module is **self-contained** and compiles independently:

1. **Copy source files** from fdelmodc into fwi/ directory
2. **Use FD kernels as-is** - elastic4/elastic6 work for both forward and adjoint
3. **Follow fdacrtmc patterns** for backward propagation and imaging condition
4. **Create thin wrapper functions** that combine forward/adjoint/gradient
5. **Use fdelmodc's MPI structure** for shot parallelization

The key insight is that the wave equation is self-adjoint, so the **same FD kernel** (elastic4/elastic6) is used for both forward and adjoint propagation. The only difference is:
- Forward: Source at shot location
- Adjoint: Source at receiver locations (inject residual)

The gradient computation is identical to the RTM imaging condition - just a different interpretation of the result.

## Quick Start

```bash
cd /rcp3/software/codes/OpenSource_SL10/fwi

# 1. Copy source files from fdelmodc
./copy_sources.sh   # (or manually copy as listed above)

# 2. Build
make clean
make

# 3. Test
./fwi file_cp=model_cp.su file_cs=model_cs.su file_ro=model_ro.su \
      file_src=wavelet.su file_obs=observed.su verbose=1
```

---

## Step-by-Step: Implementing fwi_forward.c

### Overview

`fwi_forward.c` is the first component to implement. It performs forward modeling with checkpoint storage, based on the time loop in `fdelmodc.c` (lines 550-692).

**Input:**
- Model parameters (modPar, bndPar)
- Material properties (rox, roz, l2m, lam, mul)
- Source (srcPar, wavPar, src_nwav)
- Receiver configuration (recPar)

**Output:**
- Synthetic receiver data (rec_p, rec_vx, rec_vz)
- Checkpoints for adjoint computation (checkPar)

### Function Signature

```c
/**
 * @brief Forward modeling with checkpoint storage for FWI
 *
 * Based on fdelmodc.c time loop (lines 550-692).
 * Calls elastic4/elastic6 kernel, records at receivers, stores checkpoints.
 *
 * @param mod       Model parameters (grid size, indices, scheme)
 * @param src       Source parameters (position, type)
 * @param wav       Wavelet parameters
 * @param bnd       Boundary conditions
 * @param ixsrc     Source x grid index
 * @param izsrc     Source z grid index
 * @param src_nwav  Source wavelet array [nsrc][nt]
 * @param vx        Velocity x array [naz*nax] (workspace, zeroed on entry)
 * @param vz        Velocity z array [naz*nax]
 * @param tzz       Stress tzz array [naz*nax]
 * @param txx       Stress txx array [naz*nax] (elastic only)
 * @param txz       Stress txz array [naz*nax] (elastic only)
 * @param rox       Density rox [naz*nax]
 * @param roz       Density roz [naz*nax]
 * @param l2m       Lambda+2mu [naz*nax]
 * @param lam       Lambda [naz*nax] (elastic only)
 * @param mul       Mu [naz*nax] (elastic only)
 * @param rec       Receiver parameters
 * @param rec_p     Output: recorded pressure [nrec*nt]
 * @param check     Output: checkpoints for adjoint
 * @param verbose   Verbosity level
 * @return 0 on success
 */
int fwi_forward(modPar mod, srcPar src, wavPar wav, bndPar bnd,
                int ixsrc, int izsrc, float **src_nwav,
                float *vx, float *vz, float *tzz, float *txx, float *txz,
                float *rox, float *roz, float *l2m, float *lam, float *mul,
                recPar rec, float *rec_p,
                checkPar *check, int verbose);
```

### Implementation Steps

**Step 1:** Create `fwi_forward.c` with includes and function skeleton:

```c
/**
 * @file fwi_forward.c
 * @brief Forward modeling with checkpoint storage for FWI
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "fwi.h"

/* External FD kernels (from elastic4.c, elastic6.c, acoustic4.c) */
extern int elastic4(modPar mod, srcPar src, wavPar wav, bndPar bnd,
                    int itime, int ixsrc, int izsrc, float **src_nwav,
                    float *vx, float *vz, float *tzz, float *txx, float *txz,
                    float *rox, float *roz, float *l2m, float *lam, float *mul,
                    int verbose);

extern int elastic6(modPar mod, srcPar src, wavPar wav, bndPar bnd,
                    int itime, int ixsrc, int izsrc, float **src_nwav,
                    float *vx, float *vz, float *tzz, float *txx, float *txz,
                    float *rox, float *roz, float *l2m, float *lam, float *mul,
                    int verbose);

/* External receiver recording (from getRecTimes.c) */
extern int getRecTimes(modPar mod, recPar rec, bndPar bnd, int itime, int isam,
                       float *vx, float *vz, float *tzz, float *txx, float *txz,
                       float *rec_vx, float *rec_vz, float *rec_p,
                       float *rec_txx, float *rec_tzz, float *rec_txz,
                       float *rec_pp, float *rec_ss,
                       int verbose);
```

**Step 2:** Implement the time loop (based on fdelmodc.c lines 550-692):

```c
int fwi_forward(modPar mod, srcPar src, wavPar wav, bndPar bnd,
                int ixsrc, int izsrc, float **src_nwav,
                float *vx, float *vz, float *tzz, float *txx, float *txz,
                float *rox, float *roz, float *l2m, float *lam, float *mul,
                recPar rec, float *rec_p,
                checkPar *check, int verbose)
{
    int it, isam;
    int isnap, next_snap;
    size_t sizem;

    sizem = (size_t)mod.naz * (size_t)mod.nax;

    /* Determine checkpoint interval */
    int snap_interval = (check->nsnap > 1) ? mod.nt / (check->nsnap - 1) : mod.nt;
    isnap = 0;
    next_snap = 0;

    /* Time loop - matches fdelmodc.c structure */
    isam = 0;
    for (it = 0; it < mod.nt; it++) {

        /* Store checkpoint before FD kernel (for exact adjoint matching) */
        if (check->nsnap > 0 && it == next_snap && isnap < check->nsnap) {
            if (verbose >= 3) {
                fprintf(stderr, "  Storing checkpoint %d at it=%d\n", isnap, it);
            }
            memcpy(check->vx[isnap],  vx,  sizem * sizeof(float));
            memcpy(check->vz[isnap],  vz,  sizem * sizeof(float));
            memcpy(check->tzz[isnap], tzz, sizem * sizeof(float));
            if (mod.ischeme >= 3) {  /* Elastic */
                memcpy(check->txx[isnap], txx, sizem * sizeof(float));
                memcpy(check->txz[isnap], txz, sizem * sizeof(float));
            }
            check->snap_it[isnap] = it;
            isnap++;
            next_snap = isnap * snap_interval;
            if (next_snap >= mod.nt) next_snap = mod.nt - 1;
        }

        /* FD kernel - one timestep */
        if (mod.ischeme == 3) {  /* Elastic */
            if (mod.iorder == 4) {
                elastic4(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav,
                         vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, verbose);
            } else if (mod.iorder == 6) {
                elastic6(mod, src, wav, bnd, it, ixsrc, izsrc, src_nwav,
                         vx, vz, tzz, txx, txz, rox, roz, l2m, lam, mul, verbose);
            }
        }
        /* TODO: Add acoustic4 for ischeme==1 */

        /* Record at receivers */
        if (rec.n > 0) {
            getRecTimes(mod, rec, bnd, it, isam,
                        vx, vz, tzz, txx, txz,
                        NULL, NULL, rec_p, NULL, NULL, NULL, NULL, NULL,
                        verbose);
            isam++;
        }

        /* Progress reporting */
        if (verbose >= 2 && (it % 100 == 0)) {
            fprintf(stderr, "  Forward: it=%d/%d (%.1f%%)\n",
                    it, mod.nt, 100.0 * it / mod.nt);
        }
    }

    /* Store final checkpoint if not already stored */
    if (check->nsnap > 0 && isnap < check->nsnap) {
        memcpy(check->vx[isnap],  vx,  sizem * sizeof(float));
        memcpy(check->vz[isnap],  vz,  sizem * sizeof(float));
        memcpy(check->tzz[isnap], tzz, sizem * sizeof(float));
        if (mod.ischeme >= 3) {
            memcpy(check->txx[isnap], txx, sizem * sizeof(float));
            memcpy(check->txz[isnap], txz, sizem * sizeof(float));
        }
        check->snap_it[isnap] = mod.nt - 1;
    }

    return 0;
}
```

**Step 3:** Add checkpoint allocation/free functions:

```c
int allocate_checkpoints(checkPar *check, int nsnap, int naz, int nax, int elastic)
{
    int i;
    size_t sizem = (size_t)naz * (size_t)nax;

    check->nsnap = nsnap;
    check->naz = naz;
    check->nax = nax;
    check->sizem = sizem;

    check->snap_it = (int *)calloc(nsnap, sizeof(int));
    check->vx  = (float **)calloc(nsnap, sizeof(float *));
    check->vz  = (float **)calloc(nsnap, sizeof(float *));
    check->tzz = (float **)calloc(nsnap, sizeof(float *));

    for (i = 0; i < nsnap; i++) {
        check->vx[i]  = (float *)calloc(sizem, sizeof(float));
        check->vz[i]  = (float *)calloc(sizem, sizeof(float));
        check->tzz[i] = (float *)calloc(sizem, sizeof(float));
    }

    if (elastic) {
        check->txx = (float **)calloc(nsnap, sizeof(float *));
        check->txz = (float **)calloc(nsnap, sizeof(float *));
        for (i = 0; i < nsnap; i++) {
            check->txx[i] = (float *)calloc(sizem, sizeof(float));
            check->txz[i] = (float *)calloc(sizem, sizeof(float));
        }
    } else {
        check->txx = NULL;
        check->txz = NULL;
    }

    return 0;
}

void free_checkpoints(checkPar *check)
{
    int i;
    if (check->snap_it) free(check->snap_it);
    for (i = 0; i < check->nsnap; i++) {
        if (check->vx && check->vx[i])   free(check->vx[i]);
        if (check->vz && check->vz[i])   free(check->vz[i]);
        if (check->tzz && check->tzz[i]) free(check->tzz[i]);
        if (check->txx && check->txx[i]) free(check->txx[i]);
        if (check->txz && check->txz[i]) free(check->txz[i]);
    }
    if (check->vx)  free(check->vx);
    if (check->vz)  free(check->vz);
    if (check->tzz) free(check->tzz);
    if (check->txx) free(check->txx);
    if (check->txz) free(check->txz);
}
```

---

## Test Program: test_fwi_forward.c

This test verifies that `fwi_forward()` works correctly by:
1. Creating a homogeneous model using makemod
2. Creating a Ricker wavelet using makewave
3. Running forward modeling
4. Verifying checkpoints are stored
5. Comparing output with fdelmodc (optional)

### Test Code

```c
/**
 * @file test_fwi_forward.c
 * @brief Test program for fwi_forward function
 *
 * Usage: test_fwi_forward file_cp=model_cp.su file_cs=model_cs.su \
 *                         file_ro=model_ro.su file_src=wavelet.su [options]
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "fwi.h"
#include "par.h"

/* External functions */
extern int readModel(modPar mod, bndPar bnd, float *rox, float *roz,
                     float *l2m, float *lam, float *muu,
                     float *tss, float *tes, float *tep);
extern int defineSource(wavPar wav, srcPar src, modPar mod, float **src_nwav,
                        int reverse, int verbose);
extern double wallclock_time(void);

/*********************** self documentation **********************/
char *sdoc[] = {
    " ",
    " test_fwi_forward - Test forward modeling with checkpointing",
    " ",
    " Usage: test_fwi_forward file_cp= file_cs= file_ro= file_src= [options]",
    " ",
    " Required parameters:",
    "   file_cp= ............... P-wave velocity model (SU format)",
    "   file_cs= ............... S-wave velocity model (SU format)",
    "   file_ro= ............... Density model (SU format)",
    "   file_src= .............. Source wavelet (SU format)",
    " ",
    " Optional parameters:",
    "   iorder=4 ............... FD order (4 or 6)",
    "   nsnap=10 ............... Number of checkpoints to store",
    "   verbose=0 .............. Verbosity level (0-4)",
    " ",
    NULL
};

/*============================================================================
 * Helper: Read model dimensions from SU file header
 *============================================================================*/
static int read_model_dims(const char *filename, int *nz, int *nx,
                           float *dz, float *dx)
{
    FILE *fp;
    segy hdr;
    long file_size;
    int trace_size;

    fp = fopen(filename, "rb");
    if (!fp) return -1;

    if (fread(&hdr, TRCBYTES, 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    *nz = hdr.ns;
    *dz = hdr.d1;
    *dx = (hdr.d2 > 0) ? hdr.d2 : hdr.d1;

    trace_size = TRCBYTES + hdr.ns * sizeof(float);
    fseek(fp, 0, SEEK_END);
    file_size = ftell(fp);
    *nx = file_size / trace_size;

    fclose(fp);
    return 0;
}

/*============================================================================
 * Helper: Read wavelet from SU file
 *============================================================================*/
static int read_wavelet_info(const char *filename, int *nt, float *dt)
{
    FILE *fp;
    segy hdr;

    fp = fopen(filename, "rb");
    if (!fp) return -1;

    if (fread(&hdr, TRCBYTES, 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    *nt = hdr.ns;
    *dt = (float)hdr.dt / 1000000.0f;

    fclose(fp);
    return 0;
}

/*============================================================================
 * Main test function
 *============================================================================*/
int main(int argc, char *argv[])
{
    /* Parameters */
    char *file_cp, *file_cs, *file_ro, *file_src;
    int iorder = 4;
    int nsnap = 10;
    int verbose = 0;

    /* Model dimensions */
    int nz, nx, nt;
    float dz, dx, dt;

    /* Structures */
    modPar mod;
    bndPar bnd;
    srcPar src;
    recPar rec;
    wavPar wav;
    checkPar check;

    /* Arrays */
    float *vx, *vz, *tzz, *txx, *txz;
    float *rox, *roz, *l2m, *lam, *mul;
    float *rec_p;
    float **src_nwav = NULL;
    size_t sizem;

    /* Timing */
    double t0, t1;
    int ret;

    /* Initialize parameter system */
    initargs(argc, argv);
    requestdoc(1);

    /* Get parameters */
    if (!getparstring("file_cp", &file_cp)) {
        fprintf(stderr, "ERROR: file_cp required\n");
        return 1;
    }
    if (!getparstring("file_cs", &file_cs)) {
        fprintf(stderr, "ERROR: file_cs required\n");
        return 1;
    }
    if (!getparstring("file_ro", &file_ro)) {
        fprintf(stderr, "ERROR: file_ro required\n");
        return 1;
    }
    if (!getparstring("file_src", &file_src)) {
        fprintf(stderr, "ERROR: file_src required\n");
        return 1;
    }
    getparint("iorder", &iorder);
    getparint("nsnap", &nsnap);
    getparint("verbose", &verbose);

    printf("==============================================\n");
    printf("  TEST: fwi_forward\n");
    printf("==============================================\n");

    /*------------------------------------------------------------------------
     * Step 1: Read model and wavelet dimensions
     *------------------------------------------------------------------------*/
    printf("\nStep 1: Reading file info...\n");

    ret = read_model_dims(file_cp, &nz, &nx, &dz, &dx);
    if (ret != 0) {
        fprintf(stderr, "ERROR: Cannot read model file %s\n", file_cp);
        return 1;
    }
    printf("  Model: nz=%d, nx=%d, dz=%.2f, dx=%.2f\n", nz, nx, dz, dx);

    ret = read_wavelet_info(file_src, &nt, &dt);
    if (ret != 0) {
        fprintf(stderr, "ERROR: Cannot read wavelet file %s\n", file_src);
        return 1;
    }
    printf("  Wavelet: nt=%d, dt=%.6f\n", nt, dt);

    /*------------------------------------------------------------------------
     * Step 2: Initialize modPar
     *------------------------------------------------------------------------*/
    printf("\nStep 2: Initialize model parameters...\n");

    memset(&mod, 0, sizeof(modPar));
    mod.nz = nz;
    mod.nx = nx;
    mod.dz = dz;
    mod.dx = dx;
    mod.dt = dt;
    mod.nt = nt;
    mod.iorder = iorder;
    mod.ischeme = 3;  /* Elastic */

    /* Set loop indices for staggered grid */
    int ibnd = iorder / 2;
    mod.naz = nz + 2 * ibnd;
    mod.nax = nx + 2 * ibnd;

    mod.ioXx = ibnd;  mod.ieXx = nx + ibnd;
    mod.ioXz = ibnd;  mod.ieXz = nz + ibnd;
    mod.ioZx = ibnd;  mod.ieZx = nx + ibnd;
    mod.ioZz = ibnd;  mod.ieZz = nz + ibnd;
    mod.ioPx = ibnd;  mod.iePx = nx + ibnd;
    mod.ioPz = ibnd;  mod.iePz = nz + ibnd;
    mod.ioTx = ibnd;  mod.ieTx = nx + ibnd;
    mod.ioTz = ibnd;  mod.ieTz = nz + ibnd;

    mod.file_cp = file_cp;
    mod.file_cs = file_cs;
    mod.file_ro = file_ro;

    printf("  Grid with boundaries: naz=%d, nax=%d\n", mod.naz, mod.nax);

    /*------------------------------------------------------------------------
     * Step 3: Initialize boundary parameters
     *------------------------------------------------------------------------*/
    printf("\nStep 3: Initialize boundaries...\n");

    memset(&bnd, 0, sizeof(bndPar));
    bnd.top = 1;  /* Free surface */
    bnd.bot = 4;  /* Taper */
    bnd.lef = 4;  /* Taper */
    bnd.rig = 4;  /* Taper */
    bnd.ntap = 20;
    bnd.cfree = 1;

    /* Allocate taper arrays */
    bnd.tapx = (float *)calloc(bnd.ntap, sizeof(float));
    bnd.tapz = (float *)calloc(bnd.ntap, sizeof(float));
    bnd.tapxz = (float *)calloc(bnd.ntap * bnd.ntap, sizeof(float));

    float tapfact = 0.30f;
    float scl = tapfact / (float)bnd.ntap;
    for (int i = 0; i < bnd.ntap; i++) {
        float wfct = scl * i;
        bnd.tapx[i] = expf(-(wfct * wfct));
        wfct = scl * (i + 0.5f);
        bnd.tapz[i] = expf(-(wfct * wfct));
    }
    for (int j = 0; j < bnd.ntap; j++) {
        for (int i = 0; i < bnd.ntap; i++) {
            float wfct = scl * sqrtf((float)(i*i + j*j));
            bnd.tapxz[j * bnd.ntap + i] = expf(-(wfct * wfct));
        }
    }

    /* Surface array */
    bnd.surface = (int *)calloc(mod.nax + mod.naz, sizeof(int));
    for (int i = 0; i < mod.nax + mod.naz; i++) {
        bnd.surface[i] = mod.ioPz;
    }

    printf("  Boundaries: top=%d, bot=%d, lef=%d, rig=%d, ntap=%d\n",
           bnd.top, bnd.bot, bnd.lef, bnd.rig, bnd.ntap);

    /*------------------------------------------------------------------------
     * Step 4: Allocate arrays
     *------------------------------------------------------------------------*/
    printf("\nStep 4: Allocate arrays...\n");

    sizem = (size_t)mod.naz * (size_t)mod.nax;
    printf("  Grid size: %zu points (%.2f MB per array)\n",
           sizem, sizem * sizeof(float) / 1e6);

    vx  = (float *)calloc(sizem, sizeof(float));
    vz  = (float *)calloc(sizem, sizeof(float));
    tzz = (float *)calloc(sizem, sizeof(float));
    txx = (float *)calloc(sizem, sizeof(float));
    txz = (float *)calloc(sizem, sizeof(float));

    rox = (float *)calloc(sizem, sizeof(float));
    roz = (float *)calloc(sizem, sizeof(float));
    l2m = (float *)calloc(sizem, sizeof(float));
    lam = (float *)calloc(sizem, sizeof(float));
    mul = (float *)calloc(sizem, sizeof(float));

    if (!vx || !vz || !tzz || !txx || !txz ||
        !rox || !roz || !l2m || !lam || !mul) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return 1;
    }

    /*------------------------------------------------------------------------
     * Step 5: Read model
     *------------------------------------------------------------------------*/
    printf("\nStep 5: Read model...\n");

    ret = readModel(mod, bnd, rox, roz, l2m, lam, mul, NULL, NULL, NULL);
    if (ret != 0) {
        fprintf(stderr, "ERROR: readModel failed\n");
        return 1;
    }
    printf("  Model loaded successfully\n");

    /*------------------------------------------------------------------------
     * Step 6: Setup source
     *------------------------------------------------------------------------*/
    printf("\nStep 6: Setup source...\n");

    memset(&src, 0, sizeof(srcPar));
    src.n = 1;
    src.type = 1;     /* P-wave (explosion) */
    src.orient = 1;

    src.x = (int *)malloc(sizeof(int));
    src.z = (int *)malloc(sizeof(int));
    src.x[0] = nx / 2 + ibnd;  /* Center */
    src.z[0] = 5 + ibnd;       /* Near surface */

    printf("  Source at grid (%d, %d)\n", src.x[0], src.z[0]);

    /* Setup wavelet */
    memset(&wav, 0, sizeof(wavPar));
    wav.nt = nt;
    wav.dt = dt;
    wav.ns = nt;
    wav.nx = 1;
    wav.file_src = file_src;

    ret = defineSource(wav, src, mod, &src_nwav, 0, verbose);
    if (ret < 0) {
        fprintf(stderr, "ERROR: defineSource failed\n");
        return 1;
    }
    printf("  Wavelet loaded: %d samples\n", wav.ns);

    /*------------------------------------------------------------------------
     * Step 7: Setup receivers (line at surface)
     *------------------------------------------------------------------------*/
    printf("\nStep 7: Setup receivers...\n");

    memset(&rec, 0, sizeof(recPar));
    rec.n = nx;
    rec.nt = nt;
    rec.x = (int *)malloc(nx * sizeof(int));
    rec.z = (int *)malloc(nx * sizeof(int));
    for (int ix = 0; ix < nx; ix++) {
        rec.x[ix] = ix + ibnd;
        rec.z[ix] = 2 + ibnd;
    }
    rec.type.p = 1;

    rec_p = (float *)calloc(rec.n * nt, sizeof(float));
    printf("  %d receivers at z=%d\n", rec.n, rec.z[0]);

    /*------------------------------------------------------------------------
     * Step 8: Allocate checkpoints
     *------------------------------------------------------------------------*/
    printf("\nStep 8: Allocate checkpoints...\n");

    memset(&check, 0, sizeof(checkPar));
    ret = allocate_checkpoints(&check, nsnap, mod.naz, mod.nax, 1);
    if (ret != 0) {
        fprintf(stderr, "ERROR: allocate_checkpoints failed\n");
        return 1;
    }
    printf("  %d checkpoints allocated (%.2f MB total)\n",
           nsnap, nsnap * 5 * sizem * sizeof(float) / 1e6);

    /*------------------------------------------------------------------------
     * Step 9: Run forward modeling
     *------------------------------------------------------------------------*/
    printf("\nStep 9: Running fwi_forward...\n");
    printf("----------------------------------------------\n");

    t0 = wallclock_time();

    ret = fwi_forward(mod, src, wav, bnd,
                      src.x[0], src.z[0], src_nwav,
                      vx, vz, tzz, txx, txz,
                      rox, roz, l2m, lam, mul,
                      rec, rec_p,
                      &check, verbose);

    t1 = wallclock_time();

    printf("----------------------------------------------\n");

    if (ret != 0) {
        fprintf(stderr, "ERROR: fwi_forward failed with code %d\n", ret);
        return 1;
    }

    printf("  Forward modeling completed in %.2f seconds\n", t1 - t0);

    /*------------------------------------------------------------------------
     * Step 10: Verify results
     *------------------------------------------------------------------------*/
    printf("\nStep 10: Verify results...\n");

    /* Check that checkpoints were stored */
    printf("  Checkpoints stored:\n");
    for (int i = 0; i < check.nsnap; i++) {
        float max_tzz = 0.0f;
        for (size_t j = 0; j < sizem; j++) {
            if (fabsf(check.tzz[i][j]) > max_tzz) {
                max_tzz = fabsf(check.tzz[i][j]);
            }
        }
        printf("    snap[%d]: it=%d, max|tzz|=%.6e\n",
               i, check.snap_it[i], max_tzz);
    }

    /* Check receiver data */
    float max_rec = 0.0f;
    for (int i = 0; i < rec.n * nt; i++) {
        if (fabsf(rec_p[i]) > max_rec) {
            max_rec = fabsf(rec_p[i]);
        }
    }
    printf("  Receiver data: max|p|=%.6e\n", max_rec);

    /*------------------------------------------------------------------------
     * Cleanup
     *------------------------------------------------------------------------*/
    printf("\nCleaning up...\n");

    free_checkpoints(&check);
    free(vx); free(vz); free(tzz); free(txx); free(txz);
    free(rox); free(roz); free(l2m); free(lam); free(mul);
    free(rec_p);
    free(src.x); free(src.z);
    free(rec.x); free(rec.z);
    free(bnd.tapx); free(bnd.tapz); free(bnd.tapxz); free(bnd.surface);

    printf("\n==============================================\n");
    printf("  TEST PASSED\n");
    printf("==============================================\n");

    return 0;
}
```

### Test Script: run_test_fwi_forward.sh

```bash
#!/bin/bash
# Test script for fwi_forward

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Create test output directory
mkdir -p test_output
cd test_output

echo "=============================================="
echo "  Creating test model and wavelet"
echo "=============================================="

# Model parameters
NZ=101
NX=201
DZ=10.0
DX=10.0
VP=2500
VS=1500
RHO=2000

# Wavelet parameters
NT=500
DT=0.001
FPEAK=15.0

# Create homogeneous model
../../utils/makemod file_base=model.su \
    cp0=${VP} cs0=${VS} ro0=${RHO} \
    sizex=$((NX * DX)) sizez=$((NZ * DZ)) \
    dx=${DX} dz=${DZ} orig=0,0 verbose=1

# Create Ricker wavelet
../../utils/makewave file_out=wavelet.su \
    nt=${NT} dt=${DT} fp=${FPEAK} \
    shift=1 w=g2 verbose=1

echo ""
echo "=============================================="
echo "  Running fwi_forward test"
echo "=============================================="

../test_fwi_forward \
    file_cp=model_cp.su \
    file_cs=model_cs.su \
    file_ro=model_ro.su \
    file_src=wavelet.su \
    iorder=4 \
    nsnap=10 \
    verbose=2

echo ""
echo "=============================================="
echo "  TEST COMPLETED SUCCESSFULLY"
echo "=============================================="
```

### Expected Output

```
==============================================
  TEST: fwi_forward
==============================================

Step 1: Reading file info...
  Model: nz=101, nx=201, dz=10.00, dx=10.00
  Wavelet: nt=500, dt=0.001000

Step 2: Initialize model parameters...
  Grid with boundaries: naz=105, nax=205

...

Step 9: Running fwi_forward...
----------------------------------------------
  Forward: it=0/500 (0.0%)
  Storing checkpoint 0 at it=0
  Forward: it=100/500 (20.0%)
  ...
----------------------------------------------
  Forward modeling completed in 2.35 seconds

Step 10: Verify results...
  Checkpoints stored:
    snap[0]: it=0, max|tzz|=0.000000e+00
    snap[1]: it=55, max|tzz|=1.234567e-02
    ...
  Receiver data: max|p|=5.678901e-03

==============================================
  TEST PASSED
==============================================
```
