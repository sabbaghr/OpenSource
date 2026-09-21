# TOY2DAC V2.6 — Complete Implementation Analysis
## For Time-Domain Acoustic/Viscoacoustic FWI (fdelfwi)

**Purpose**: Extract every implementation detail from TOY2DAC to guide the construction
of a time-domain acoustic FWI that reproduces identical inversion results.

---

## 1. Forward Problem

### 1.1 Wave Equation

TOY2DAC solves the **variable-density viscoacoustic Helmholtz equation** in the frequency domain:

```
S(m) p = f
```

where `S` is the impedance matrix, `p` is the pressure wavefield, and `f` is the source vector.

The equation discretized is:

```
∇ · (b ∇p) + ω²/κ̃ · p = f/h²
```

where:
- `b = 1/ρ` (buoyancy)
- `κ̃ = ρ ṽ²` (complex bulk modulus)
- `ṽ = Vp × (1 - i/(2Q))` (complex velocity for constant-Q attenuation)
- `ω_c = ω + i σ` (complex frequency for Laplace damping)

**Source**: `sub_v2mu.f90:22-27` — the `imq=1` model (hardcoded):
```fortran
vc = v(i1,i2) * (1. - 0.5*ci/q(i1,i2))
mu(i1,i2) = rho(i1,i2) * vc * vc
```
So `mu` in the code = `κ̃ = ρ Vp² (1 - i/(2Q))²`

**Buoyancy**: `sub_rho2b.f90` — simply `b = 1/ρ` with zero-density guard.

### 1.2 Mixed-Grid Stencil (Jo/Shin Optimization)

**Source**: `sub_evalmatrix.f90`

The impedance matrix uses a 9-point mixed-grid finite-difference stencil with optimized weights
(Stekl & Pratt, 1998; Jo et al., 1996):

**Mass matrix weights** (for ω²/κ̃ term):
```
wm1 = 0.6287326    (center node)
wm2 = 0.3712667/4  (edge-adjacent nodes, ×4)
wm3 = (1-wm1-wm2×4)/4  ≈ 0.000000175/4  (corner nodes, ×4)
```
Note: wm2 is stored as 0.3712667 then multiplied by 0.25 in use.

**Stiffness matrix weights** (for ∇·(b∇p) term):
```
w1 = 0.4382634     (rotated 45° stencil, anti-lumped)
w2 = 1 - w1 = 0.5617366  (standard 5-point stencil)
```

The stencil combines:
- **w2 fraction**: Standard 5-point Laplacian with averaged buoyancy
- **w1 fraction**: Rotated 45° Laplacian with corner-averaged buoyancy

### 1.3 PML Implementation

**Source**: `sub_PML.f90`, called via `subdamp()`

PML via complex coordinate stretching. The damping functions `d1p, d1pb, d2p, d2pb`
modify the spatial derivatives in the impedance matrix. Parameters:
- `apml` = 90 (PML strength, from `fdfd_input`)
- `npml` = 10 (PML thickness in grid points)
- `itd` = 2 (quadratic profile)

PML is applied by multiplying spatial derivative operators by complex damping coefficients.

### 1.4 Free Surface

Controlled by `pbdir%free_surface` (0 or 1). When enabled, `sub_evalmatrix_fs` is called
instead of `sub_evalmatrix`, which modifies the stencil at the top boundary.

### 1.5 Hicks Interpolation

When `pbdir%hicks=1`, sources and receivers use Hicks interpolation for off-grid positioning:
- `sub_fill_RHS_SOURCES_HICKS()` for source injection
- `Hicks_extraction()` for receiver extraction
- `sub_Hicks_constant()` precomputes interpolation coefficients

### 1.6 Source Types

- `itypeso=0`: Point pressure source (monopole) — `f/h²` at source node
- `itypeso=1`: Dipole source (vertical force) — requires Hicks interpolation

### 1.7 Model Extension

Models are extended by `npml` nodes on each side:
```
n1e = n1 + 2*npml
n2e = n2 + 2*npml
```
via `sub_modext()` which extrapolates boundary values into the PML region.

---

## 2. Cost Function

### 2.1 Exact Formula

**Source**: `sub_FDFD.f90:sub_pbdirect_FDFD_FCOST` (lines 650-761)

```
For each (source, receiver) pair k:

  weight(k) = exp(σ × t0(k)) × weight_data(k)

  residual(k) = (d_obs(k) - cc1 × d_cal(k)) × weight(k) / |cc1|

  fcost += 0.5 × |residual(k)|²     (complex modulus squared)

  residual_for_adjoint(k) = residual(k) × weight(k)   ← DOUBLE WEIGHTING
```

**CRITICAL DETAIL**: The residual stored for the adjoint source has `weight²` (double weighting):
```fortran
inv%residual(k) = (inv%obs_data(k,iw) - data_cal(k)) * weight / ABS(inv%cc1(isrc,iw))
inv%fcost = inv%fcost + 0.5 * (inv%residual(k) * CONJG(inv%residual(k)))
inv%residual(k) = inv%residual(k) * weight    ! ← SECOND weight multiplication
```

### 2.2 Data Weighting

```
weight(k) = exp(slaplace × t0(isrc,irec)) × weight_data(k)
```

Where:
- `slaplace` = Laplace damping parameter (from `fdfd_input`, default 0)
- `t0(isrc,irec)` = first-arrival traveltime for source-receiver pair
- `weight_data(k)` = offset-dependent weight from file, linearly interpolated

The weight file has `n_data_weight` values at spacing `dx_data_weight`, indexed by source-receiver offset.

**Source estimation options** (`inv%src_estim`):
- 3 or 4: weight uses only `exp(slaplace × t0)` WITHOUT `weight_data`
- 1 or 2: weight uses both `exp(slaplace × t0)` AND `weight_data`

### 2.3 Source Estimation (cc1)

**Source**: `sub_FDFD.f90:sub_pbdirect_FDFD_FCOST` (lines 717-734)

```
cc1 = Σ_k conj(d_cal(k) × weight(k)) × d_obs(k) × weight(k)
     / Σ_k conj(d_cal(k) × weight(k)) × d_cal(k) × weight(k)
```

This is a complex scalar (in frequency domain).

Options:
- `src_estim = 0`: No estimation, `cc1 = 1`
- `src_estim = 1`: One **global** cc1 for all sources (accumulated across all source gathers)
- `src_estim = 2`: One cc1 **per source** gather (reset valnum/valden per source)
- `src_estim = 3,4`: Same as 1,2 but without `weight_data` in the weight

**Timing**: Computed at `firstgrad == 0` (first gradient evaluation only), then fixed for all subsequent iterations.

### 2.4 Scaling Factor

**Source**: `sub_modeling.f90:sub_modeling_FWI_grad` (lines 247-261)

Computed once at the first gradient evaluation:

```fortran
inv%scalingfactor = 10e-5 * SQRT(MAX(SUM(model(:,:,1)²), 1e3*n1*n2) / SUM(gradient(:,:,1)²))
inv%scalingfactor = 1.0 / inv%scalingfactor
```

**CRITICAL**: In Fortran, `10e-5 = 10 × 10⁻⁵ = 10⁻⁴`, NOT `10⁻⁵`!

So the effective computation is:
```
S_temp = 1e-4 × sqrt(max(||m₁||², 1e3×N) / ||g₁||²)
scalingfactor = 1 / S_temp
```

Applied at EVERY iteration (not just the first):
```fortran
inv%fcost = inv%fcost / inv%scalingfactor
inv%gradient(:,:,:) = inv%gradient(:,:,:) / inv%scalingfactor
```

For epsilon/delta parameters, different constants are used:
```fortran
inv%scalingfactor = 5e-3 * SQRT(MAX(SUM(model²), 0.1*N) / SUM(gradient²))
```

**Key**: Scaling uses only the FIRST parameter's model/gradient norms (`model(:,:,1)`, `gradient(:,:,1)`).

---

## 3. Adjoint Source

### 3.1 Construction

**Source**: `sub_fill_RHS_RECEIVERS.f90`

The adjoint source injected at receiver positions is:

```
adjoint_src(k) = conj(residual(k)) / h²
```

Where `residual(k)` has the DOUBLE weight (see Section 2.1).

The transpose solve `A^T λ = -conj(RHS)/h²` gives the adjoint wavefield. Note the conjugation
and sign: `ICNTL(9) = 0` for transpose solve.

### 3.2 Time-Domain Equivalent

In time domain, the adjoint source at receiver `k` is:
```
adj_src(k, t) = [d_obs(k,t) - cc1·d_cal(k,t)] × weight²(k) / |cc1|
```
time-reversed and injected from `t = T` to `t = 0`.

---

## 4. Gradient Formulas

### 4.1 Vp Gradient

**Source**: `sub_gradient_for_base_param.f90:gradient_vp` (lines 6-18)

```
g_Vp(x) = -Re[ 2ω²/(ρ Vp³ (1 - i/(2Q))²) × p_inc(x) × p_adj(x) × cc1/|cc1| ]
```

Summed over all sources and frequencies.

### 4.2 Qp Gradient

**Source**: `sub_gradient_for_base_param.f90:gradient_qp` (lines 21-33)

```
g_Qp(x) = -Re[ iω²/(ρ Vp² (1 - i/(2Q))³ Q²) × p_inc(x) × p_adj(x) × cc1/|cc1| ]
```

### 4.3 Density Gradient

**Source**: `sub_gradient_for_base_param.f90:sub_gradient_rho` (lines 230-305)

The density gradient involves spatial derivatives (via `sub_dAdmi`):
```
g_ρ(x) = Re[ p_adj^T × (∂A/∂ρ) × p_inc × cc1/|cc1| ]
```

Where `∂A/∂ρ` has contributions from:
- Buoyancy term: `∂b/∂ρ = -1/ρ²` (5-point stencil, b00 and neighbors)
- Mass term: `∂(1/κ̃)/∂ρ = -1/(ρ² ṽ²)`

The mode=2 call sets:
```
mu1(0,0) = 1/(ρ² ṽ²)     ! mass derivative
b(0,0) = 1/ρ²             ! buoyancy derivative
```

### 4.4 Log-Parameterization Gradients

For `log(Vp)` (invpar=10 or 11):
```
g_{log(Vp)} = Vp × g_Vp = -Re[2ω²/(ρ Vp² (1-i/(2Q))²) × p_inc × p_adj × cc1/|cc1|]
```
Note: The `Vp³` becomes `Vp²` because of the chain rule `d/d(log Vp) = Vp × d/dVp`.

For `log(Vp/Vp₀)` (invpar=31): Same gradient as `log(Vp)` since `Vp₀` is constant.

### 4.5 Time-Domain Gradient Equivalents

The frequency-domain `ω²` factor corresponds to `∂²/∂t²` in time domain:

**Vp gradient (time domain)**:
```
g_Vp(x) = -2/(ρ Vp³) × Σ_t [∂²p_fwd/∂t²](x,t) × p_adj(x,T-t) × dt
```

For acoustic constant-Q, additional terms arise from the `(1-i/(2Q))²` factor
which corresponds to relaxation mechanisms in time domain.

**For pure acoustic (Q → ∞)**:
```
g_Vp(x) = -2/(ρ Vp³) × Σ_t [∂²p_fwd/∂t²](x,t) × p_adj(x,T-t) × dt
```

---

## 5. Shin Pseudo-Hessian Preconditioner

### 5.1 Diagonal Computation

**Source**: `sub_Shin_preco.f90` (lines 7-125)

**For Vp** (invpar=1):
```
H_diag(x) = Σ_src Σ_freq 4ω⁴ / (ρ² Vp⁶ |1-i/(2Q)|⁴) × |p_inc(x)|⁴
```

**For Qp** (invpar=3):
```
H_diag(x) = Σ_src Σ_freq ω⁴ / (ρ² Vp⁴ |1-i/(2Q)|⁶ Q⁴) × |p_inc(x)|⁴
```

**For ρ** (invpar=2): Uses `sub_drho5_fld_preco()` — involves spatial derivatives of the
incident wavefield (5-point stencil contribution).

### 5.2 Threshold and Application

```fortran
threshold = inv%coeff_damping_preco   ! e.g., 1e-4
max = maxval(preco(:,:,:))            ! global max across all params

! Apply threshold
preco(i1,i2,ipar) = 1.0 / (preco(i1,i2,ipar) + threshold * max)

! Apply preconditioner
v_preco(k) = preco(i1,i2,ipar) * v(k)

! Norm-preserving rescaling
scal_preco = ||grad|| / ||grad_preco||     ! computed BEFORE calling
v_preco(:) = v_preco(:) * scal_preco
```

**Key implementation detail**: The scaling factor `scal_preco` is computed as a two-step process:
1. First call with `scal_preco = 1` to compute `||grad_preco||`
2. Compute `scal_preco = ||grad|| / ||grad_preco||`
3. Second call applies the actual preconditioner with correct scaling

**Source**: `sub_FWI_OPTIM.f90` (lines 88-96):
```fortran
inv%scal_preco = 1.
call Shin_preconditioning(nn, ..., grad, grad_preco)
call normL2(nn, grad_preco, norm_grad_preco)
call normL2(nn, grad, norm_grad)
inv%scal_preco = norm_grad / norm_grad_preco
call Shin_preconditioning(nn, ..., grad, grad_preco)
```

### 5.3 Time-Domain Equivalent

For Vp:
```
H_diag(x) = 4/(ρ² Vp⁶) × Σ_src Σ_t [∂²p_fwd/∂t²]⁴(x,t) × dt
```

The `ω⁴ |p|⁴` in frequency domain corresponds to `|∂²p/∂t²|⁴` in time domain
(by Parseval's theorem, power spectrum of 4th order).

For pure acoustic (Q → ∞, |1-i/(2Q)| → 1):
```
H_diag(x) = 4/(ρ² Vp⁶) × Σ_src Σ_t [∂²p_fwd/∂t²]⁴(x,t)
```

---

## 6. SEISCOPE Optimization Toolbox Interface

### 6.1 Flag Protocol

**Source**: `sub_FWI_OPTIM.f90` (lines 156-264)

```
FLAG = 'INIT'  →  Provide initial fcost and gradient
                   Optimizer returns with FLAG = 'GRAD' or 'CONV'

FLAG = 'GRAD'  →  Recompute fcost and gradient at new model point x
                   (line search trial step)

FLAG = 'NSTE'  →  Nonlinear step accepted
                   Write intermediate model, check model-change convergence

FLAG = 'PREC'  →  Apply preconditioner to optimizer's internal vector
                   For PLBFGS: optim%q_plb
                   For PTRN: optim%residual → optim%residual_preco

FLAG = 'HESS'  →  Compute Hessian-vector product (truncated Newton)
                   Input: optim%d, Output: optim%Hd

FLAG = 'CONV'  →  Converged (exit loop)
FLAG = 'FAIL'  →  Line search failed (exit loop)
```

### 6.2 Optimization Methods

| opt_meth | Method | Call | Preconditioned? |
|----------|--------|------|-----------------|
| 1 | STD (Steepest Descent) | `PSTD(nn,x,fcost,grad,grad_preco,optim,FLAG)` | No (grad_preco=grad) |
| 2 | PSTD | `PSTD(...)` | Yes |
| 3 | NLCG | `PNLCG(...)` | No |
| 4 | PNLCG | `PNLCG(...)` | Yes |
| 5 | LBFGS | `LBFGS(nn,x,fcost,grad,optim,FLAG)` | No |
| 6 | PLBFGS | `PLBFGS(nn,x,fcost,grad,grad_preco,optim,FLAG)` | Yes |
| 7 | TGN (Gauss-Newton) | `TRN(nn_tilde,x_tilde,fcost,grad_tilde,optim,FLAG)` | No |
| 8 | TRN (Exact Newton) | `TRN(...)` | No |
| 9 | PTGN | `PTRN(...)` | Yes |
| 10 | PTRN | `PTRN(...)` | Yes |

### 6.3 Initialization

**Source**: `sub_init_optim.f90`

```fortran
optim%conv = inv%convergence_criterion       ! e.g., 1e-4
optim%niter_max = inv%niter_nonlin           ! e.g., 10
optim%print_flag = 1                         ! print convergence info
optim%debug = .false.                        ! (or .true. from debug_option)
optim%bound = inv%bound                      ! 0 or 1
optim%lb(:) = inv%lb(ipar)                   ! per-parameter lower bound
optim%ub(:) = inv%ub(ipar)                   ! per-parameter upper bound
optim%threshold = inv%threshold              ! bound tolerance
optim%l = inv%lbfgs_m                        ! L-BFGS memory (for methods 5,6)
optim%niter_max_CG = inv%niter_CG            ! inner CG iterations (for methods 7-10)
```

### 6.4 Model Vector Layout

The model vector `x` has length `nn = n1 × n2 × npar` with column-major ordering:

```
x[k] = model(i1, i2, ipar)
k = i1 + (i2-1)*n1 + (ipar-1)*n1*n2
```

Where `i1` = depth (fast), `i2` = horizontal (medium), `ipar` = parameter (slow).

---

## 7. Regularization

### 7.1 Tikhonov Regularization

**Source**: `sub_Tikhonov.f90`

Activated when `lambda(ipar) > 0`.

**Cost function addition** (before scaling):
```
f_tik_z = 0.5 × λ_z/h² × Σ_ipar λ(ipar) × Σ_{i1,i2} (m(i1+1,i2,ipar) - m(i1,i2,ipar))²
f_tik_x = 0.5 × λ_x/h² × Σ_ipar λ(ipar) × Σ_{i1,i2} (m(i1,i2+1,ipar) - m(i1,i2,ipar))²
fcost += f_tik_z + f_tik_x
```

**Gradient addition**:
Standard second-order FD Laplacian:
```
g_tik(i1,i2) = λ(ipar)/h² × [λ_z × (-m(i1+1) + 2m(i1) - m(i1-1))
                              + λ_x × (-m(i2+1) + 2m(i2) - m(i2-1))]
```
With one-sided stencils at boundaries.

**Note**: Tikhonov is applied BEFORE scaling, then divided by scalingfactor with everything else.

### 7.2 Gradient Smoothing (Implicit Regularization)

When `lambda(1) < 0`, regularization mode switches to **gradient smoothing** (`inv%regul=2`):
```fortran
CALL sub_precond2(inv, pbdir%vp, inv%gradient, n1, n2, h, freq_max, npar, -1)
```
This applies a 2D smoothing operator to the gradient. The `lambda_x` and `lambda_z` values
are pre-normalized by `h²`.

### 7.3 Prior Model Regularization

**Source**: `sub_prior.f90`

Activated when `lambda_prior0 > 0`.

```
f_prior = 0.5 × λ_prior × Σ_{i1,i2} (m(i1,i2,1) - m_prior(i1,i2))²
g_prior(i1,i2) = λ_prior × (m(i1,i2,1) - m_prior(i1,i2))
```

Only applies to the FIRST parameter.

**Hessian-vector product**: `Hv += λ_prior × v` (identity contribution).

---

## 8. Parameterization

### 8.1 Options

**Source**: `sub_converter.f90:sub_modelinv2pbdir` and `sub_setmodelpbdir2inv`

| invpar | Description | Forward: x → model | Gradient chain rule |
|--------|-------------|---------------------|---------------------|
| 1 | Vp (raw) | Vp = x | g(x) = g(Vp) |
| 2 | ρ (raw) | ρ = x | g(x) = g(ρ) |
| 3 | Qp (raw) | Qp = x | g(x) = g(Qp) |
| 10,11 | log(Vp) | Vp = exp(x) | g(x) = Vp × g(Vp) |
| 12 | log(ρ) | ρ = exp(x) | g(x) = ρ × g(ρ) |
| 31 | log(Vp/Vp₀) | Vp = Vp₀ × exp(x) | g(x) = Vp × g(Vp) |
| 32 | log(ρ/ρ₀) | ρ = ρ₀ × exp(x) | g(x) = ρ × g(ρ) |
| 33 | log(Qp/Qp₀) | Qp = Qp₀ × exp(x) | g(x) = Qp × g(Qp) |
| 34 | log(1+ε) | ε = exp(x)-1 | g(x) = (1+ε) × g(ε) |
| 35 | log(1+δ) | δ = exp(x)-1 | g(x) = (1+δ) × g(δ) |

Reference values (`Vp₀`, `ρ₀`, `Qp₀`) are set from the initial model lower bounds.

### 8.2 Bound Constraints

```fortran
optim%lb(k) = inv%lb(ipar)    ! same bound for all grid points of parameter ipar
optim%ub(k) = inv%ub(ipar)
optim%threshold = inv%threshold
```

For log parameterization, bounds should be in log space:
```
optim%lb = log(physical_lower_bound)
optim%ub = log(physical_upper_bound)
```

---

## 9. Multi-Frequency Strategy

**Source**: `sub_read_freqdom_freq_list.f90`, `freq_management` file

The frequency file specifies:
```
nfreq
freq_1 freq_2 ... freq_nfreq
```

All frequencies are processed within a single inversion step (summed in cost/gradient).

For multiscale, you re-run with different `freq_management` files containing
progressively higher frequencies. Each run reads the final model from the previous run
as its starting model.

---

## 10. Bathymetry and Dead Zone

### 10.1 Bathymetry

- `ibathy(i2)` = seafloor depth index for column i2
- Gradient is zeroed above bathymetry: `i1 < ibathy(i2)`
- In frozen-coefficient mode (Newton), vectors are reduced to exclude above-bathymetry points

### 10.2 Dead Zone

- `ideadzone` = additional depth offset below bathymetry
- Gradient starts from `ibathy(i2) + ideadzone`

---

## 11. Convergence Criteria

Three stopping criteria:
1. **Cost reduction**: `fcost/f0 < convergence_criterion` (handled by toolbox via `optim%conv`)
2. **Model change**: `||x - x_prev|| / ||x_init|| < convergence_criterion_model` (checked at NSTE)
3. **Max iterations**: `niter_nonlin` (handled by toolbox via `optim%niter_max`)

Plus toolbox internal: line search failure → FLAG='FAIL'

---

## 12. Hessian-Vector Products (Truncated Newton)

### 12.1 Gauss-Newton (opt_meth=7,9)

**Source**: `sub_modeling.f90:sub_modeling_FWI_GN` (lines 296-448)

1. Compute Born wavefield: `α = S⁻¹ × (∂S/∂m × v) × p_inc` via `sub_fill_RHS_Jv`
2. Extract at receivers: `R α` via `sub_restriction`
3. Solve adjoint: `λ = (S^T)⁻¹ × (-conj(Rα)/h²)`
4. Compute Hv: same gradient formula but with `p_inc` and `λ` instead of `p_inc` and `p_adj`

### 12.2 Exact Newton (opt_meth=8,10)

**Source**: `sub_modeling.f90:sub_modeling_FWI_EN` (lines 454-750)

Additional second-order terms involving:
- `α × p_adj` cross-correlation (second-order adjoint)
- `∂²A/∂m²` terms for parameter cross-coupling

---

## 13. Ball Template Test Case

**Source**: `run_ball_template/`

- Grid: 101×101, h=20m → 2000×2000m domain
- True model: `vp_ball` — homogeneous with velocity perturbation ball
- Starting model: `vp_homogeneous`
- ρ, Qp: constant (files `rho`, `qp`)
- Frequency: single frequency at 3 Hz
- Acquisition: `acqui_full` (full coverage)
- PML: 10 nodes, strength 90
- Hicks interpolation: ON
- Free surface: OFF
- Laplace damping: 0
- Inversion: PLBFGS (opt_meth=6), 10 iterations, lbfgs_m=5
- Preconditioner threshold: 1e-4
- Bounds: [1000, 4000] m/s
- Source estimation: OFF (cc1=1)
- Regularization: OFF (lambda=0)

---

## 14. Key Differences: Frequency Domain vs Time Domain

| Aspect | TOY2DAC (freq domain) | fdelfwi (time domain) |
|--------|----------------------|----------------------|
| Forward solve | LU factorization + back-substitution | Explicit time stepping |
| Adjoint solve | Same LU (transpose solve) | Reverse time stepping |
| ω² in gradient | Direct multiplication | ∂²/∂t² (2nd time derivative) |
| cc1 | Complex scalar per frequency | Real scalar (amplitude only) |
| Wavefield storage | All sources × all freqs in memory | Checkpoint + re-propagation |
| PML | Complex coordinate stretching | CPML (convolutional PML) |
| Q-attenuation | Complex velocity ṽ = Vp(1-i/(2Q)) | Relaxation mechanisms (τ_σ, τ_ε) |
| Multi-frequency | Sum over frequencies in one solve | Bandpass filtering of time data |

---

## 15. Implementation Priority for fdelfwi Acoustic Mode

### Phase 2A: Minimum Viable Acoustic FWI
1. **Acoustic forward/adjoint** — already exists (ischeme=1 in fdelmodc)
2. **Cost function** — adapt `computeResidual.c` to match TOY2DAC's weight²/cc1 formula
3. **Gradient** — modify `accumGradient` for acoustic: `-2/(ρVp³) × ∂²p/∂t² × p_adj × dt`
4. **Scaling** — implement TOY2DAC's `10e-5` scaling (replace current Brossier scaling)
5. **SEISCOPE toolbox** — already integrated (LBFGS, PLBFGS, TRN all working)
6. **Shin preconditioner** — adapt from elastic to acoustic formula

### Phase 2B: Full Feature Parity
7. Source estimation (cc1)
8. Tikhonov regularization
9. Prior model regularization
10. Log/normalized-log parameterization
11. Bathymetry/dead zone
12. Data weighting with offset-dependent weights
13. Free surface

### Phase 2C: Validation
14. Finite-difference gradient test
15. Ball template comparison vs TOY2DAC
16. Marmousi comparison vs TOY2DAC

---

## 16. Existing fdelfwi Code Mapping

| TOY2DAC Component | fdelfwi Equivalent | Status |
|---|---|---|
| Forward solver | `fdfwimodc.c` | ✅ Working (acoustic mode) |
| Adjoint solver | `adj_shot.c` | ✅ Working (elastic, needs acoustic simplification) |
| Cost function | `computeResidual.c` | ⚠️ L2 only, no cc1/weight² |
| Gradient (Vp) | `fwi_gradient.c:accumGradient` | ⚠️ Elastic only, needs acoustic kernel |
| Gradient (ρ) | `fwi_gradient.c` | ⚠️ Elastic only |
| Scaling | `scaling.c` | ❌ Range normalization, NOT TOY2DAC scaling |
| Preconditioner | `preconditioner.c` | ⚠️ Yang P4, NOT Shin diagonal |
| Optimization | `fwi_inversion.c` + `optim.h` | ✅ LBFGS/PLBFGS/TRN working |
| Regularization | — | ❌ Not implemented |
| Source estimation | — | ❌ Not implemented |
| Parameterization | — | ❌ Raw only |
| Bound constraints | `fwi_inversion.c` | ✅ Working |
| Checkpointing | `checkpoint.c` | ✅ Working |
| MPI | `fwi_inversion.c` (#ifdef USE_MPI) | ✅ Working |
| Multiscale | `bandpass_filter.c` | ✅ Working |
