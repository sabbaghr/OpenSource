# Implementation Plan: Three New Features for Elastic FWI

## Code Review Summary

### Architecture Overview

The EFWI framework is organized as a modular C codebase with a Fortran-translated optimization library:

| Layer | Key Files | Role |
|:------|:----------|:-----|
| **Optimizer** | `optimization/lbfgs.c`, `trn.c`, `optim.h` | Reverse-communication optimizer (L-BFGS, TRN, Enriched, etc.) |
| **Inversion driver** | `fdelfwi/fwi_inversion.c` | Main loop: parses parameters, calls optimizer, dispatches gradient/Hessian computations |
| **Gradient engine** | `fdelfwi/compute_fwi_gradient()` (in fwi_inversion.c:309) | Per-iteration: distributes shots, calls forward + residual + adjoint, MPI reduction |
| **Misfit** | `fdelfwi/computeResidual.c` | L2 misfit and adjoint source computation per shot, multi-component |
| **Adjoint** | `fdelfwi/adj_shot.c` | Checkpoint-based adjoint backpropagation + gradient cross-correlation |
| **Model bridge** | `fdelfwi/updateModel.c` | `extractModelVector`, `injectModelVector`, `extractGradientVector`, chain-rule conversion |
| **Scaling** | `fdelfwi/scaling.c` | Brossier/Yang parameter normalization |
| **Preconditioner** | `fdelfwi/yang_precond.c` | Yang pseudo-Hessian block preconditioner |
| **Forward modeling** | `fdelfwi/fdfwimodc.c`, `elastic4.c` | Staggered-grid finite differences with checkpointing |

### Key Design Patterns Already in Place

1. **Reverse communication loop**: The optimizer returns flags (OPT_GRAD, OPT_HESS, OPT_PREC, OPT_NSTE, OPT_CONV, OPT_FAIL) and the main loop responds. This cleanly separates physics from optimization.

2. **Flat model/gradient vectors**: Model parameters are stored as flat vectors `x[nvec]` where `nvec = nparam * nmodel`, laid out as `[param1(nx*nz) | param2(nx*nz) | param3(nx*nz)]`.

3. **Parameterization abstraction**: `param=1` (Lame: lambda, mu, rho) vs `param=2` (velocity: Vp, Vs, rho). Gradients always accumulate in Lame space internally; conversion happens once in `extractGradientVector`.

4. **Misfit type enum**: `fdelfwi.h:242` already defines `misfitType` with `MISFIT_L2=0` and a placeholder `MISFIT_CORRELATION`. The `computeResidual` function takes this enum but currently only implements L2 (line 163-164 has an explicit error for non-L2).

5. **Multi-component**: Supports comma-separated components (`comp=_rvx,_rvz,_rp`) with per-component Brossier weighting.

6. **Box constraints**: The optimizer already supports optional `lb[n]`/`ub[n]` bounds via `optim_project()`.

### Identified Integration Points

| Feature | Primary integration point | Secondary touchpoints |
|:--------|:--------------------------|:----------------------|
| Frequency bands | `fwi_inversion.c` main loop (wrap iterations in band loop) | `computeResidual.c` (filter before misfit), new `bandpass_filter.c` |
| Cross-correlation misfit | `computeResidual.c` (new misfit branch) | `hess_shot.c` (Born residual for TRN), `fwi_inversion.c` (parameter parsing) |
| Selectable parameters | `fwi_inversion.c` (gradient masking), `updateModel.c` (model update masking) | `scaling.c` (per-parameter), `yang_precond.c` (block preconditioner sizing) |

---

## Feature 1: Frequency Band Scheduling (Multiscale FWI)

### Theoretical Background

Multiscale FWI (Bunks et al., 1995; Sirgue & Pratt, 2004) mitigates cycle-skipping by inverting low frequencies first, where the misfit landscape is smoother and has a wider basin of attraction. The inversion proceeds through a sequence of frequency bands:

```
Stage 0: [f_lo, f_1]    (e.g., 2-5 Hz)   — long wavelengths, smooth updates
Stage 1: [f_lo, f_2]    (e.g., 2-10 Hz)  — intermediate detail
Stage 2: [f_lo, f_3]    (e.g., 2-20 Hz)  — full resolution
```

At each stage, both observed and synthetic data are bandpass-filtered before computing the misfit and adjoint source. The model from the previous stage serves as the starting model for the next. The optimizer state (L-BFGS history, step length) is typically reset between stages.

The bandpass filter must be applied in the time domain (since the modeling engine is time-domain FD). A zero-phase Butterworth filter (applied forward + reverse = `filtfilt`) preserves waveform timing, which is critical for FWI.

### Affected Files and Functions

| File | Changes |
|:-----|:--------|
| `fdelfwi/fwi_inversion.c` | Outer band loop wrapping the optimizer loop; parse band schedule; reset optimizer between bands |
| `fdelfwi/computeResidual.c` | Apply bandpass filter to observed and synthetic traces before computing residual |
| **NEW: `fdelfwi/bandpass_filter.c`** | Zero-phase Butterworth bandpass filter (time-domain, `filtfilt` style) |
| `fdelfwi/fdelfwi.h` | Declare filter functions; add `bandPar` structure |
| `fdelfwi/Makefile` | Add `bandpass_filter.o` to link |

### Data Structures

```c
/* fdelfwi.h — Band schedule for multiscale FWI */
#define MAX_BANDS 20

typedef struct _bandPar {
    int    nbands;                  /* Number of frequency bands (0 = no filtering) */
    float  flo[MAX_BANDS];          /* Low corner frequency per band (Hz) */
    float  fhi[MAX_BANDS];          /* High corner frequency per band (Hz) */
    int    niter_per_band[MAX_BANDS]; /* Max optimizer iterations per band */
    int    filter_order;            /* Butterworth order (default 4) */
} bandPar;
```

### API Design

```c
/* bandpass_filter.c */

/**
 * Apply zero-phase Butterworth bandpass filter to a trace in-place.
 *
 * @param data    Trace data [ns samples], filtered in-place
 * @param ns      Number of samples
 * @param dt_sec  Sample interval in seconds
 * @param flo     Low corner frequency (Hz), 0 = lowpass only
 * @param fhi     High corner frequency (Hz), 0 = highpass only
 * @param order   Filter order (typically 2-6)
 */
void bandpass_filter_trace(float *data, int ns, float dt_sec,
                           float flo, float fhi, int order);

/**
 * Apply bandpass filter to all traces in an SU file, writing to output file.
 * Can filter in-place (infile == outfile).
 *
 * @param infile   Input SU file path
 * @param outfile  Output SU file path (can be same as infile)
 * @param flo      Low corner frequency (Hz)
 * @param fhi      High corner frequency (Hz)
 * @param order    Butterworth order
 */
void bandpass_filter_sufile(const char *infile, const char *outfile,
                            float flo, float fhi, int order);
```

### Pseudocode: Main Loop Integration

```c
/* fwi_inversion.c::main() — after parameter parsing */

/* Parse band schedule */
bandPar bands;
parseBandSchedule(&bands);  /* reads freq_lo=, freq_hi=, niter_band= */

if (bands.nbands == 0) {
    /* No frequency scheduling — single-band mode (current behavior) */
    bands.nbands = 1;
    bands.flo[0] = 0.0f;   /* no filter */
    bands.fhi[0] = 0.0f;
    bands.niter_per_band[0] = niter;
}

/* Outer band loop */
for (int iband = 0; iband < bands.nbands; iband++) {

    if (mpi_rank == 0)
        vmess("=== Frequency band %d/%d: [%.1f, %.1f] Hz, %d iterations ===",
              iband+1, bands.nbands, bands.flo[iband], bands.fhi[iband],
              bands.niter_per_band[iband]);

    /* Reset optimizer for new band (keep model, clear L-BFGS history) */
    if (iband > 0) {
        optim_finalize(&opt);
        memset(&opt, 0, sizeof(optim_type));
        opt.niter_max = bands.niter_per_band[iband];
        opt.conv = conv;
        opt.l = lbfgs_mem;
        opt.nls_max = nls_max;
        /* ... re-initialize other opt fields ... */
        flag = OPT_INIT;
    } else {
        opt.niter_max = bands.niter_per_band[0];
    }

    /* Store current band frequencies for use in compute_fwi_gradient */
    float current_flo = bands.flo[iband];
    float current_fhi = bands.fhi[iband];

    /* Compute initial gradient for this band */
    fcost = compute_fwi_gradient_filtered(..., current_flo, current_fhi, bands.filter_order);
    /* ... extract, scale, precondition gradient ... */

    /* Inner optimizer loop (existing while loop) */
    while (flag != OPT_CONV && flag != OPT_FAIL) {
        /* ... existing optimizer logic ... */
        if (flag == OPT_GRAD) {
            fcost = compute_fwi_gradient_filtered(..., current_flo, current_fhi, bands.filter_order);
            /* ... rest of gradient processing ... */
        }
    }

    if (mpi_rank == 0)
        vmess("Band %d/%d complete: misfit=%.6e", iband+1, bands.nbands, fcost);
}
```

### Pseudocode: Filtering in computeResidual

```c
/* computeResidual.c — inside the per-component loop, after reading traces */

/* Apply bandpass filter to both observed and synthetic */
if (fhi > 0.0f || flo > 0.0f) {
    bandpass_filter_trace(buf_obs, ns, dt_sec, flo, fhi, filter_order);
    bandpass_filter_trace(buf_syn, ns, dt_sec, flo, fhi, filter_order);
}

/* Then proceed with residual computation as before */
for (isamp = 0; isamp < ns; isamp++) {
    float r = buf_syn[isamp] - buf_obs[isamp];
    misfit += 0.5f * w2 * r * r;
    buf_obs[isamp] = w2 * r;
}
```

**Important**: The filtering must happen inside `computeResidual` (not outside) because synthetic data changes every iteration. Observed data could be pre-filtered once per band for efficiency, but filtering both together is simpler and avoids file management complexity.

### Command-Line Interface

```bash
# Single band (current behavior, no change):
fwi_inversion ... niter=20

# Three-band multiscale:
fwi_inversion ... \
    freq_lo=2,2,2 \
    freq_hi=5,10,20 \
    niter_band=5,10,15 \
    filter_order=4

# Two-band with overlap:
fwi_inversion ... \
    freq_lo=1,1 \
    freq_hi=8,25 \
    niter_band=10,20
```

### Risks and Gotchas

1. **Adjoint source consistency**: The adjoint source must be the derivative of the *filtered* misfit. Since we filter both d_obs and d_syn before computing r = d_syn - d_obs, the adjoint source ψ = w²·r is correct — the filter is self-adjoint (zero-phase Butterworth applied forward+backward).

2. **Data weights per band**: Brossier weights `w_c = 1/rms(d_obs_c)` should be recomputed for each band, since filtering changes the RMS. The code currently computes weights once from shot 0 (line 1219-1231). This needs to move inside the band loop.

3. **Hessian-vector products (TRN)**: Born data must also be filtered at the same band before computing the adjoint of the Born residual in `hess_shot.c`. This is a secondary but important change.

4. **Filter stability**: Butterworth IIR filters can be numerically unstable for high orders. Use second-order sections (SOS) cascade for orders > 4. Alternatively, implement as FFT-based bandpass for guaranteed stability.

5. **Optimizer reset**: Between bands, the L-BFGS history (sk, yk pairs) from the previous band is invalid because the objective function changes. Must call `optim_finalize` and re-initialize. The model vector `x` should be preserved.

---

## Feature 2: Cross-Correlation Objective Function

### Theoretical Background

The L2 waveform misfit `J = 0.5 * Σ (d_syn - d_obs)²` is sensitive to amplitude errors and prone to cycle-skipping when the starting model is far from the true model. The normalized cross-correlation misfit measures waveform similarity independent of amplitude:

**Global correlation misfit** (per trace):

```
C(τ) = Σ_t d_obs(t) · d_syn(t - τ)
J_cc = 1 - C(0) / √(Σ d_obs² · Σ d_syn²)
```

This is equivalent to `J_cc = 1 - cos(θ)` where θ is the angle between the observed and synthetic waveforms viewed as vectors.

**Adjoint source** (van Leeuwen & Mulder, 2010; Choi & Alkhalifah, 2012):

For the normalized cross-correlation misfit at zero lag, the adjoint source for trace `i` is:

```
ψ_i(t) = (1/N_obs) · [ d_syn,i(t)/||d_syn,i|| - cos(θ_i) · d_obs,i(t)/||d_obs,i|| ]
```

where:
```
cos(θ_i) = <d_obs,i, d_syn,i> / (||d_obs,i|| · ||d_syn,i||)
N_obs = ||d_obs,i||
```

More precisely, we can write:

```
J_cc,i = 1 - <d_obs,i, d_syn,i> / (||d_obs,i|| · ||d_syn,i||)

∂J_cc,i/∂d_syn,i(t) = -d_obs,i(t) / (||d_obs,i|| · ||d_syn,i||)
                       + <d_obs,i, d_syn,i> · d_syn,i(t) / (||d_obs,i|| · ||d_syn,i||³)

= -(1/||d_syn,i||) · [ d_obs,i(t)/||d_obs,i|| - cos(θ_i) · d_syn,i(t)/||d_syn,i|| ]
```

This adjoint source has several desirable properties:
- **Amplitude-insensitive**: Only depends on the angle between waveforms, not magnitudes
- **No cycle-skipping for small phase errors**: The gradient pushes toward phase alignment
- **Self-normalizing**: No need for Brossier weighting since the misfit is already normalized

### Affected Files and Functions

| File | Changes |
|:-----|:--------|
| `fdelfwi/computeResidual.c` | Add `MISFIT_CORRELATION` branch computing cross-correlation misfit and adjoint source |
| `fdelfwi/fdelfwi.h` | Already has `MISFIT_CORRELATION` in `misfitType` enum (line 244) |
| `fdelfwi/fwi_inversion.c` | Parse `misfit=` parameter (0=L2, 1=correlation); pass to `computeResidual` |
| `fdelfwi/hess_shot.c` | For TRN: Born residual misfit must use same objective |

### API Design

The existing `computeResidual` signature already accepts `misfitType mtype`:

```c
float computeResidual(int ncomp, const char **obs_files, const char **syn_files,
                      const char *res_file, misfitType mtype,
                      const float *comp_weights, int verbose);
```

No signature change needed — just implement the `MISFIT_CORRELATION` case.

### Pseudocode: Cross-Correlation Misfit in computeResidual

```c
/* computeResidual.c — inside the per-trace loop */

if (mtype == MISFIT_CORRELATION) {
    /* Step 1: Compute norms and inner product for this trace */
    double norm_obs2 = 0.0, norm_syn2 = 0.0, dot_os = 0.0;
    for (isamp = 0; isamp < ns; isamp++) {
        norm_obs2 += (double)buf_obs[isamp] * (double)buf_obs[isamp];
        norm_syn2 += (double)buf_syn[isamp] * (double)buf_syn[isamp];
        dot_os    += (double)buf_obs[isamp] * (double)buf_syn[isamp];
    }
    double norm_obs = sqrt(norm_obs2);
    double norm_syn = sqrt(norm_syn2);

    /* Guard against zero-amplitude traces */
    if (norm_obs < 1.0e-30 || norm_syn < 1.0e-30) {
        /* Skip this trace — write zeros as adjoint source */
        memset(buf_obs, 0, ns * sizeof(float));
    } else {
        double cos_theta = dot_os / (norm_obs * norm_syn);

        /* Misfit: J = 1 - cos(theta), summed over traces */
        misfit += (float)(w2 * (1.0 - cos_theta));

        /* Adjoint source: dJ/dd_syn(t) */
        for (isamp = 0; isamp < ns; isamp++) {
            double adj = -(buf_obs[isamp] / (norm_obs * norm_syn))
                         + cos_theta * buf_syn[isamp] / norm_syn2;
            buf_obs[isamp] = (float)(w2 * adj);
        }
    }

} else {
    /* Existing L2 misfit */
    for (isamp = 0; isamp < ns; isamp++) {
        float r = buf_syn[isamp] - buf_obs[isamp];
        misfit += 0.5f * w2 * r * r;
        buf_obs[isamp] = w2 * r;
    }
}
```

### Command-Line Interface

```bash
# L2 misfit (default, current behavior):
fwi_inversion ... misfit=0

# Cross-correlation misfit:
fwi_inversion ... misfit=1
```

### Hessian-Vector Product Consistency (TRN)

For TRN algorithms, `hess_shot.c` computes `J^T J dm` using Born-modeled data as "synthetic" and the original residual objective. When using cross-correlation misfit, the Born residual adjoint source must also use the correlation formula. This requires:

1. Passing `mtype` through to `hess_shot.c`
2. In `hess_shot.c`, after Born modeling generates `born_syn` data, compute the Born residual using the same misfit type

**Change in `hess_shot.c`**:
```c
/* Currently: L2 residual of Born data */
/* Change to: use computeResidual with mtype parameter */
misfit = computeResidual(ncomp, syn_arr, born_arr, born_res_file,
                         mtype, comp_weights, 0);
```

This is already almost the case — `hess_shot` calls `computeResidual` for the Born data, and we just need to pass the correct `mtype`.

### Risks and Gotchas

1. **Initial step length**: The cross-correlation misfit has very different magnitude than L2 (order 1 vs order 10^6). The initial step length `alpha = 1/||g||` (line 1317) handles this automatically, but the convergence tolerance `conv` may need adjustment. Document recommended `conv` values for each misfit type.

2. **Zero-amplitude traces**: Must guard against division by zero when traces are muted or contain only zeros (e.g., after windowing). The pseudocode above handles this.

3. **Multi-component weighting**: For cross-correlation, each trace is self-normalized, so Brossier weighting is less critical. However, `comp_weights` can still be used to weight the relative contribution of different components. The user should set `data_weight=0` when using correlation misfit.

4. **Gradient sign convention**: The adjoint source sign must be consistent with the existing convention. Currently, `adj_shot.c` negates the residual at injection (line 351). The cross-correlation adjoint source already has the correct sign (negative of dJ/dd_syn), so this should work with the existing negation.

5. **Cycle-skipping regime**: Cross-correlation is more robust but has a narrower convergence basin for very large phase errors (> half period). It's best used in combination with frequency band scheduling (Feature 1).

---

## Feature 3: Selectable Parameter Updates

### Theoretical Background

In elastic FWI, simultaneously inverting for all parameters (Vp, Vs, rho) can lead to trade-offs and cross-talk between parameters, especially when the data has limited sensitivity to some of them (e.g., density from surface seismic). A common practice is to:

1. **First invert Vp only** (freeze Vs and rho) to get the dominant velocity structure
2. **Then add Vs** (freeze rho) to resolve shear-wave information
3. **Finally release all parameters** for fine-tuning

This "hierarchical" or "cascaded" approach is especially important for elastic FWI where parameter trade-offs are severe (Operto et al., 2013).

Implementation requires:
- Zeroing the gradient components for frozen parameters
- Skipping model updates for frozen parameters
- Maintaining the frozen parameters in the forward model (they still affect wave propagation)

### Affected Files and Functions

| File | Changes |
|:-----|:--------|
| `fdelfwi/fwi_inversion.c` | Parse `active_params=`, create parameter mask, apply mask to gradient after extraction, apply mask after model injection |
| `fdelfwi/updateModel.c` | `extractGradientVector`: optionally zero frozen parameter gradients |
| `fdelfwi/scaling.c` | Scaling should only apply to active parameters |
| `fdelfwi/yang_precond.c` | Preconditioner should handle reduced parameter sets |
| `fdelfwi/fdelfwi.h` | Declare mask-related utilities |
| `fdelfwi/hess_shot.c` | Hessian-vector product: zero perturbation for frozen params |

### Data Structures

```c
/* fdelfwi.h — Parameter activity mask */
typedef struct _paramMask {
    int update_p1;   /* 1 = active, 0 = frozen (lambda or Vp) */
    int update_p2;   /* 1 = active, 0 = frozen (mu or Vs) */
    int update_p3;   /* 1 = active, 0 = frozen (rho) */
} paramMask;
```

### Design: Two Approaches

**Approach A (Recommended): Full-vector with masking**
- Keep `nvec = nparam * nmodel` (always 3*nmodel for elastic)
- After extracting gradient, zero the components for frozen parameters
- After optimizer updates x, restore frozen parameter values from initial model
- Pros: Minimal changes to optimizer, scaling, preconditioner
- Cons: Optimizer wastes some memory/computation on zero gradient components

**Approach B: Reduced vector**
- Set `nvec` = (number of active params) * nmodel
- Pack/unpack only active parameters into optimizer vector
- Pros: Faster optimizer (smaller vectors)
- Cons: Requires rewriting extractModelVector, injectModelVector, extractGradientVector, scaling, preconditioner — very invasive

**Approach A is recommended** because it requires changes in only a few locations and doesn't risk breaking the optimizer's internal state management.

### Pseudocode: Implementation

```c
/* fwi_inversion.c::main() — parameter parsing */

/* Parse active parameters */
paramMask pmask = {1, 1, 1};  /* Default: all active */
char *active_str;
if (getparstring("active_params", &active_str)) {
    pmask.update_p1 = 0;
    pmask.update_p2 = 0;
    pmask.update_p3 = 0;

    /* Parse comma-separated list: "vp", "vs", "rho", "lam", "mu" */
    char abuf[256];
    strncpy(abuf, active_str, sizeof(abuf)-1);
    char *tok = strtok(abuf, ",");
    while (tok) {
        if (strcmp(tok, "vp") == 0 || strcmp(tok, "lam") == 0 || strcmp(tok, "lambda") == 0)
            pmask.update_p1 = 1;
        else if (strcmp(tok, "vs") == 0 || strcmp(tok, "mu") == 0)
            pmask.update_p2 = 1;
        else if (strcmp(tok, "rho") == 0)
            pmask.update_p3 = 1;
        tok = strtok(NULL, ",");
    }

    if (mpi_rank == 0) {
        const char *p1name = (param == 1) ? "lambda" : "Vp";
        const char *p2name = (param == 1) ? "mu" : "Vs";
        vmess("Active parameters: %s=%s  %s=%s  rho=%s",
              p1name, pmask.update_p1 ? "UPDATE" : "FROZEN",
              p2name, pmask.update_p2 ? "UPDATE" : "FROZEN",
              pmask.update_p3 ? "UPDATE" : "FROZEN");
    }
}

/* Save initial model for restoring frozen parameters */
float *x_frozen = NULL;
if (!pmask.update_p1 || !pmask.update_p2 || !pmask.update_p3) {
    x_frozen = (float *)malloc(nvec * sizeof(float));
    memcpy(x_frozen, x, nvec * sizeof(float));
}
```

```c
/* Apply mask to gradient — after extractGradientVector, before passing to optimizer */

static void applyParamMask(float *g, int nmodel, int nparam,
                           const paramMask *pmask)
{
    if (!pmask->update_p1)
        memset(g, 0, nmodel * sizeof(float));
    if (nparam >= 2 && !pmask->update_p2)
        memset(g + nmodel, 0, nmodel * sizeof(float));
    if (nparam >= 3 && !pmask->update_p3)
        memset(g + 2*nmodel, 0, nmodel * sizeof(float));
}

/* In the gradient processing section (after extractGradientVector): */
extractGradientVector(grad_vec, grad1, grad2, grad3, &mod, &bnd, param);
applyParamMask(grad_vec, nmodel, nparam, &pmask);  /* <-- NEW */
if (scaling > 0) scaling_scale_gradient(grad_vec, nmodel, nparam, m0);
```

```c
/* Restore frozen parameters — after optimizer modifies x, before injectModelVector */

static void restoreFrozenParams(float *x, const float *x_frozen,
                                int nmodel, int nparam,
                                const paramMask *pmask,
                                const float *m0, const float *m_shift,
                                int scaling)
{
    /* x and x_frozen are both in the same space (normalized or physical) */
    if (!pmask->update_p1)
        memcpy(x, x_frozen, nmodel * sizeof(float));
    if (nparam >= 2 && !pmask->update_p2)
        memcpy(x + nmodel, x_frozen + nmodel, nmodel * sizeof(float));
    if (nparam >= 3 && !pmask->update_p3)
        memcpy(x + 2*nmodel, x_frozen + 2*nmodel, nmodel * sizeof(float));
}

/* In OPT_GRAD branch, before injectModelVector: */
if (scaling > 0) scaling_denormalize(x, nmodel, nparam, m0, m_shift);
restoreFrozenParams(x, x_frozen, nmodel, nparam, &pmask, m0, m_shift, scaling);
```

```c
/* TRN Hessian-vector product: zero perturbation for frozen params */

/* In OPT_HESS branch, after denormalizing opt.d: */
applyParamMask(opt.d, nmodel, nparam, &pmask);  /* Zero perturbation for frozen params */
/* ... compute H*d ... */
applyParamMask(opt.Hd, nmodel, nparam, &pmask); /* Zero Hessian output for frozen params */
```

### Command-Line Interface

```bash
# All parameters active (default):
fwi_inversion ...

# Vp only:
fwi_inversion ... active_params=vp

# Vp and Vs (freeze rho):
fwi_inversion ... active_params=vp,vs

# Lame parameterization, lambda and mu only:
fwi_inversion ... param=1 active_params=lam,mu

# Density only (unusual but valid):
fwi_inversion ... active_params=rho
```

### Interaction with Preconditioner

When parameters are frozen, the Yang pseudo-Hessian block preconditioner must handle the reduced system. Two options:

**Option 1 (Simple)**: Zero the rows and columns of frozen parameters in the 3x3 block matrix, set diagonal to 1. This means `P^{-1}g = g` for frozen parameters (which is zero anyway after masking).

**Option 2 (Correct)**: Build a reduced block preconditioner. For example, if only Vp is active, use a 1x1 preconditioner `P11^{-1}`. If Vp+Vs, use a 2x2 block.

Option 1 is recommended for simplicity — since the gradient is already zeroed for frozen parameters, the preconditioner output for those components is irrelevant.

### Risks and Gotchas

1. **L-BFGS history pollution**: If a parameter is frozen (gradient = 0), the L-BFGS sk/yk pairs will contain zeros in those components. This doesn't break the algorithm but wastes memory. For large models where this matters, consider Approach B (reduced vector).

2. **Initial step length**: `alpha = 1/||g||` is computed from the full gradient vector. With frozen parameters, the active gradient norm may be much smaller than the full gradient would be, giving a reasonable step.

3. **Box constraints**: Bounds for frozen parameters are irrelevant but harmless. The optimizer may try to project frozen components, but since they're immediately restored, this is fine.

4. **Scaling**: Brossier/Yang scaling computes `m0` from the initial model vector. For frozen parameters, `m0` is still computed but the scaled gradient component is zero, so no issue.

5. **Model output**: Intermediate model files should still show all parameters (including frozen ones at their initial values) for consistency.

6. **Switching active parameters mid-inversion**: Not supported in a single run. To do hierarchical inversion (Vp first, then Vp+Vs), use separate runs or combine with Feature 1's band loop to also switch active parameters per stage.

---

## Implementation Order and Shared Infrastructure

### Recommended Order

```
1. Feature 3 (Selectable Parameters)   — Simplest, no new files, minimal risk
2. Feature 2 (Cross-Correlation)       — Self-contained in computeResidual.c
3. Feature 1 (Frequency Bands)         — Most invasive, requires new file + loop restructuring
```

### Rationale

- **Feature 3** is purely additive (masking) and touches no core algorithms. It can be tested immediately with existing L2 misfit and serves as a useful tool while developing the other features.

- **Feature 2** is self-contained in `computeResidual.c` with a clean switch. The adjoint source derivation is well-known and can be verified with the existing Taylor test infrastructure (`test_taylor`).

- **Feature 1** requires restructuring the main loop (adding an outer band loop), introducing a new source file (bandpass filter), and modifying the flow of data through `computeResidual`. It should be done last because the other two features will be tested within the existing single-band framework first.

### Shared Infrastructure

1. **`applyParamMask()` utility** (from Feature 3) is also useful for Feature 1 if the user wants to change active parameters per frequency band in a future extension.

2. **`misfitType` enum** (already in `fdelfwi.h`) is shared between Features 1 and 2. Feature 2 implements the correlation case; Feature 1 passes it through to the filtered misfit computation.

3. **`computeResidual` signature**: Features 1 and 2 both modify this function. Feature 2 adds the correlation branch. Feature 1 adds filter parameters. The combined signature becomes:

```c
float computeResidual(int ncomp, const char **obs_files, const char **syn_files,
                      const char *res_file, misfitType mtype,
                      const float *comp_weights,
                      float flo, float fhi, int filter_order,  /* Feature 1 */
                      int verbose);
```

When `flo == 0 && fhi == 0`, no filtering is applied (backward compatible).

4. **Test infrastructure**: All three features can be verified with:
   - `test_taylor.c`: Verify gradient accuracy by Taylor expansion (should give O(h²) convergence)
   - `test_hessian_dp.c`: Verify adjoint consistency (for Feature 2's new adjoint source)
   - Demo scripts: Extended versions of existing `test_lbfgs_inversion_velocity.sh`

### Testing Plan

| Feature | Test | Expected Result |
|:--------|:-----|:----------------|
| 3 (Params) | Run L-BFGS with `active_params=vp` | Only Vp updates; Vs and rho unchanged |
| 3 (Params) | Taylor test with frozen params | O(h²) convergence for active params only |
| 2 (CC misfit) | Taylor test with `misfit=1` | O(h²) convergence of correlation misfit |
| 2 (CC misfit) | L-BFGS with `misfit=1` | Convergence (slower than L2 for well-posed problems) |
| 1 (Bands) | 3-band L-BFGS | Final model matches or improves on single-band result |
| 1 (Bands) | Single band with `freq_lo=2 freq_hi=10` | Equivalent to pre-filtered data |
| All | Combined: 3-band + CC misfit + Vp-only first band | Production-ready workflow |
