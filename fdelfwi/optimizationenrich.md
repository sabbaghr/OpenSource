# Enriched Optimization for Elastic FWI

## References

- **Morales & Nocedal (2002)**: "Enriched Methods for Large-Scale Unconstrained Optimization", Comput. Optim. Appl., 21, 143-154
- **Métivier, Brossier, Operto, Virieux (2017)**: "Full Waveform Inversion and the Truncated Newton Method", SIAM Review, 59(1), 153-195

---

## 1. Motivation

The FWI optimization currently supports standalone L-BFGS and standalone TRN:

| Method | Cost per iteration | Strength | Weakness |
|--------|-------------------|----------|----------|
| **L-BFGS** | 1 gradient (fwd + adj per source) | Cheap, good for smooth problems | Stalls on ill-conditioned / high-contrast models; curvature approximation degrades when problem is strongly nonlinear |
| **TRN** | gradient + N_cg × Hessian-vector products (each = Born + adj per source) | Handles multiscattering, parameter coupling, ill-conditioning | Expensive; N_cg can be large without good preconditioning |

**Design philosophy**: L-BFGS is the workhorse. TRN is called sparingly — only when L-BFGS stalls. When TRN does run, maximize its effectiveness with the best preconditioning (the L-BFGS curvature history), so each expensive Hessian-vector product counts.

**Key innovation — start with TRN**: The first iteration(s) use TRN. The inner CG produces pairs `(d_j, H·d_j)` which are direct Hessian curvature samples. These are stored in the L-BFGS history buffer as `(s,y)` pairs (they satisfy `y^T·s > 0` since `d^T·Hd > 0` for positive curvature directions). When we switch to L-BFGS, it starts with a rich second-order curvature approximation instead of building from scratch. This dramatically improves L-BFGS effectiveness from the very first iteration.

---

## 2. Algorithm Overview

### Unified Framework

Both L-BFGS and TN are special cases of Preconditioned CG (Morales & Nocedal):

```
L-BFGS iteration  =  PCG with maxcg = 1  →  returns p = -H(m)·g
TN iteration       =  PCG with maxcg > 1  →  approximately solves H·p = -g
```

They share the same L-BFGS preconditioner `H(m)` (built from `(s,y)` history). The ADJUST procedure dynamically switches between modes based on step quality.

### Flow

```
[t₀ * TRN] ──H(m)──→ [l * L-BFGS] ──H(m)──→ [t * TRN] ──H(m)──→ [l * L-BFGS] → ...
     ↑                                                                        ↑
  Seed L-BFGS buffer                                           Curvature keeps
  with (d, Hd) pairs                                           improving
  from inner CG
```

Where `l` and `t` are dynamically adjusted by the ADJUST procedure.

### Startup Phase

1. **First TRN iteration(s)** (t₀ = 2, with maxcg = 5 as cautious start):
   - Forward + adjoint → gradient
   - Inner CG: each iteration computes `Hd_j` via Born + adjoint
   - Store each positive-curvature CG pair `(d_j, Hd_j)` into L-BFGS buffer as `(s,y)` pair
   - After CG converges, linesearch → accepted step → store outer `(s,y)` pair too
   - **Result**: L-BFGS buffer seeded with ~5-10 true curvature pairs per TRN step

2. **Switch to L-BFGS** (l = 20 default):
   - L-BFGS two-loop recursion now uses the TRN-seeded curvature
   - Each L-BFGS step adds its own outer `(s,y)` pair
   - Cheap iterations with good curvature approximation

3. **If L-BFGS stalls** → ADJUST switches back to TRN with L-BFGS preconditioner
   - Inner CG is preconditioned by `z = H(m)·r` using the accumulated L-BFGS history
   - Fewer inner CG iterations needed thanks to preconditioning

---

## 3. Detailed Algorithm

### 3.1 Enriched Algorithm (Main Loop)

```
ENRICHED ALGORITHM
  Choose starting point x, memory parameter m, initial t₀ = 2
  Set method ← 'HFN'; first ← true; enr_l = 20

  While convergence test is not satisfied:
    Repeat
      compute p: call PCG(B, p, g, method, status, maxcg)
      compute α: call LNSRCH(α)
      compute x₊ = x + α·p
      store s = x₊ - x  and  y = g₊ - g
      update H(m) with (s, y) pair
      call ADJUST(l, t, α, method, status, first, maxcg)
    End repeat
  End while
```

### 3.2 Preconditioned CG (PCG)

```
PROCEDURE PCG(B, p, g, method, status, maxcg)
  set r⁽⁰⁾ ← g
  compute z⁽⁰⁾ = H(m)·r⁽⁰⁾    [L-BFGS two-loop recursion]

  if method = 'L-BFGS':
    set p ← -z⁽⁰⁾              [single preconditioned step]
    return

  [HFN mode: multiple preconditioned CG iterations]
  set p⁽⁰⁾ ← 0
  set v⁽¹⁾ ← -z⁽⁰⁾
  ρ₀ = <r⁽⁰⁾, z⁽⁰⁾>

  for j = 1, 2, ... maxcg:
    compute Bv⁽ʲ⁾                [Hessian-vector product via Born+adjoint]
    dHd = <v⁽ʲ⁾, Bv⁽ʲ⁾>

    if dHd ≤ 0:                  [negative curvature]
      status ← 'Indefinite'
      if j = 1: p ← v⁽¹⁾        [fallback to preconditioned steepest descent]
      return

    [Store CG pair in L-BFGS buffer for curvature enrichment]
    store (v⁽ʲ⁾, Bv⁽ʲ⁾) into H(m) as bonus (s,y) pair

    αᶜᵍ = ρⱼ₋₁ / dHd
    p⁽ʲ⁾ = p⁽ʲ⁻¹⁾ + αᶜᵍ · v⁽ʲ⁾
    r⁽ʲ⁾ = r⁽ʲ⁻¹⁾ + αᶜᵍ · Bv⁽ʲ⁾

    z⁽ʲ⁾ = H(m)·r⁽ʲ⁾           [precondition with L-BFGS]
    ρⱼ = <r⁽ʲ⁾, z⁽ʲ⁾>
    β = ρⱼ / ρⱼ₋₁
    v⁽ʲ⁺¹⁾ = -z⁽ʲ⁾ + β · v⁽ʲ⁾

    [Eisenstat-Walker stopping criterion]
    if ||r⁽ʲ⁾|| ≤ η · ||g||:
      status ← 'Converged'
      return
  end for

  set p ← p⁽ᵐᵃˣᶜᵍ⁾
  return
```

### 3.3 ADJUST Procedure (Dynamic Mode Switching)

Biased toward L-BFGS: long L-BFGS cycles, short TRN cycles.

```
PROCEDURE ADJUST(l, t, α, method, status, first, maxcg)
  k ← k + 1

  if method = 'L-BFGS':
    if k ≥ l:                    [L-BFGS cycle complete]
      if first:
        maxcg ← 5; t ← 2; force2 ← false; first ← false
      end if
      method ← 'HFN'; k ← 0; profit ← 0
    end if

  else:                          [method = 'HFN']
    if status = 'Indefinite Hessian':
      t ← 1; force2 ← false
      l ← min(3·l/2, 30)        [lengthen L-BFGS cycle]
      method ← 'L-BFGS'; k ← 0
      return

    if α ≥ 0.8:                  [profitable Newton step]
      profit ← profit + 1
    else:                        [unprofitable: small step]
      if force2 and k = 1:
        return                   [give second chance]
      else:
        t ← max(2, k - 1)
        method ← 'L-BFGS'; k ← 0
        return

    if k ≥ t:                    [HFN cycle complete]
      if profit = k: t ← t + 1  [all profitable → extend next TRN cycle]
      if profit ≥ 2: force2 ← true  else  force2 ← false
      method ← 'L-BFGS'; k ← 0
    end if
  end if
```

### 3.4 Eisenstat-Walker Forcing Term (η_k,1)

Best choice for FWI per Métivier et al.:

```
η_k = ||∇f(m_k) - ∇f(m_{k-1}) - α_{k-1} H(m_{k-1}) Δm_{k-1}|| / ||∇f(m_{k-1})||
```

Approximated as:
```
eisenvect = grad_new - residual_from_last_CG
η = ||eisenvect|| / ||grad_prev||
```

Safeguard (Eisenstat & Walker):
```
if η_{k-1}^φ > 0.1:  η_k = max(η_k, η_{k-1}^φ)    where φ = (1+√5)/2
if η_k > 1.0:        η_k = 0.9
```

Initial value: η₀ = 0.9

### 3.5 Curvature Seeding from Inner CG

During TRN inner CG, each iteration produces `(d_j, Hd_j)` where `d^T·Hd > 0` (we stop on negative curvature). These pairs satisfy the L-BFGS secant condition `y ≈ H·s`, so they can be stored in the L-BFGS buffer:

```
For each CG iteration j with dHd > 0:
  optim_save_lbfgs_pair(n, opt, d_j, Hd_j)
```

This is a lightweight addition (just memcpy into the circular buffer) but seeds the preconditioner with true Hessian information. The L-BFGS buffer has `m` slots (default 20); if TRN produces more pairs than slots, oldest are discarded (circular buffer behavior).

**Important**: Only store pairs where `<d, Hd> > 0` and `||d|| > 0` (positive curvature, non-degenerate).

---

## 4. Computational Cost Analysis

| Phase | Cost per outer iteration | Typical iterations |
|-------|------------------------|--------------------|
| **Initial TRN** | 1 grad + N_cg Hess-vec = (2 + 2·N_cg) fwd problems per source | t₀ = 2 iterations, N_cg ≈ 5 |
| **L-BFGS** | 1 grad = 2 fwd problems per source | l ≈ 20 iterations |
| **TRN (preconditioned)** | 1 grad + N_cg Hess-vec, N_cg small due to preconditioning | t ≈ 2-5 iterations, N_cg ≈ 3-5 |

For a 100-iteration inversion:
- **Pure L-BFGS**: ~100 gradient evaluations
- **Enriched (typical)**: ~80 L-BFGS + ~15 TRN (with ~4 CG each) = ~80 + 15×(1+4×2) = ~215 equivalent fwd problems
- **Pure TRN**: ~30 outer × (1+5×2) = ~330 equivalent fwd problems

The enriched method uses ~30% more compute than pure L-BFGS but with much better convergence on difficult problems. It uses ~35% less compute than pure TRN.

---

## 5. Implementation Plan

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `optimization/enriched.c` | **CREATE** | Enriched optimizer (~500 lines) |
| `optimization/optim.h` | **MODIFY** | Add ALG_ENRICHED, enriched state fields, public two-loop fn |
| `optimization/lbfgs.c` | **MODIFY** | Extract two-loop recursion as public `optim_lbfgs_apply()` |
| `fdelfwi/fwi_inversion.c` | **MODIFY** | Hook up algorithm=5, parse `enr_l` param |
| `fdelfwi/Makefile` | **MODIFY** | Add `enriched.o` to optimization objects |

### Step 1: Make L-BFGS two-loop recursion public

In `lbfgs.c`: rename `descent_lbfgs()` to `optim_lbfgs_apply()` and make it public.
```c
// Computes output = -H(m) * input using L-BFGS two-loop recursion
void optim_lbfgs_apply(int n, optim_type *opt, const float *input, float *output);
```

In `optim.h`: add declaration.

The existing `lbfgs_run()` calls the new public function instead of the old static one.

### Step 2: Add enriched state to `optim.h`

```c
ALG_ENRICHED  /* in optAlg enum */

/* In optim_type struct: */
int     enr_method;     /* 0=L-BFGS, 1=HFN */
int     enr_l;          /* L-BFGS cycle length */
int     enr_t;          /* HFN cycle length */
int     enr_k;          /* step counter within current cycle */
int     enr_profit;     /* consecutive profitable HFN steps */
int     enr_force2;     /* force ≥2 HFN iterations */
int     enr_first;      /* first time in HFN (startup phase) */
int     enr_maxcg;      /* max CG iterations for current cycle */
float  *enr_precond_z;  /* preconditioner workspace [n] */

/* Declaration: */
void enriched_run(int n, float *x, float fcost, float *grad,
                  optim_type *opt, optFlag *flag);
```

### Step 3: Create `optimization/enriched.c`

Internal functions:
- `init_enriched()` — allocate arrays, set initial state (method=HFN for startup)
- `pcg_enriched()` — preconditioned CG with curvature seeding
- `adjust_enriched()` — dynamic mode switching
- `forcing_term_enriched()` — Eisenstat-Walker η_{k,1}
- `print_info_enriched()` — write `iterate_ENR.dat`
- `finalize_enriched()` — free arrays
- `enriched_run()` — main reverse-communication dispatcher

### Step 4: Hook up in `fwi_inversion.c`

- `algorithm=5` → enriched
- Parse `enr_l=20` parameter
- `case 5: enriched_run(nvec, x, fcost, grad_vec, &opt, &flag);`
- `keep_chk = 1` when `algorithm == 5`
- OPT_HESS handling identical to existing TRN case

### Step 5: Update Makefile

Add `enriched.o` to OPTIM_OBJ.

---

## 6. Reverse Communication State Machine

```
                    ┌──────────────────────────────────────────┐
                    │               OPT_INIT                    │
                    │  init → start HFN (startup)               │
                    │  CG_INIT → need Hd                        │
                    └──────────┬───────────────────────────────┘
                               │ OPT_HESS
                               ↓
                    ┌──────────────────────────────────────────┐
                    │            ENR_DESC (inner CG)            │
                    │  process Hd, do one CG step               │
                    │  store (d, Hd) in L-BFGS buffer           │
                    │  if CG done → start linesearch            │
                    └──────┬──────────────┬────────────────────┘
                           │              │
                  OPT_HESS │     OPT_GRAD │ (CG done, need LS)
                  (more CG)│              │
                           ↓              ↓
                    ┌──────────────────────────────────────────┐
                    │            ENR_NSTE (linesearch)          │
                    │  Wolfe linesearch                          │
                    │  if LS_NEW_STEP:                          │
                    │    update (s,y), ADJUST, forcing term      │
                    │    if L-BFGS: descent=-H(m)g → LS         │
                    │    if HFN: start CG → OPT_HESS            │
                    │  if LS_NEW_GRAD: → OPT_GRAD               │
                    └──────┬──────────────┬───────────────┬────┘
                           │              │               │
                  OPT_NSTE │     OPT_GRAD │      OPT_HESS│
                  (accepted)│    (LS eval) │    (new CG)  │
                           ↓              ↓               ↓
                    ┌──────────────────────────────────────────┐
                    │         Caller handles flag               │
                    │  OPT_GRAD: compute gradient               │
                    │  OPT_HESS: compute Hd = Born+adj          │
                    │  OPT_NSTE: write model, continue          │
                    │  OPT_CONV / OPT_FAIL: stop                │
                    └──────────────────────────────────────────┘
```

---

## 7. Convergence File Format

`iterate_ENR.dat`:
```
**********************************************************************
                    ENRICHED OPTIMIZATION ALGORITHM
**********************************************************************
     Convergence criterion  :   1.00e-06
     Niter_max              :      100
     Initial cost is        :   2.34e+05
     Initial norm_grad is   :   1.00e+00
     L-BFGS cycle length    :        20
     Memory parameter m     :        20
**********************************************************************
   Niter        fk      ||gk||      fk/f0         alpha  method   nls   nit_CG         eta    ngrad   nhess
       0  2.34e+05  1.00e+00  1.00e+00  0.00e+00    HFN      0        5  9.00e-01        1       5
       1  1.87e+05  8.40e-01  7.99e-01  8.20e-03    HFN      2        4  7.10e-01        4      10
       2  1.52e+05  7.10e-01  6.50e-01  1.00e+00    LB       1        0  6.30e-01        5      10
       3  1.38e+05  6.50e-01  5.90e-01  1.00e+00    LB       1        0  5.80e-01        6      10
     ...
```
