# The ∂A(m)/∂m Operator for Elastic FWI Hessian-Vector Products

## Overview

This document derives the operator ∂A(m)/∂m for the elastic wave equation as implemented
in the staggered-grid finite-difference code. This operator is the key ingredient for
computing the Gauss-Newton Hessian-vector product H·v = J^T(J·v) used in Truncated Newton
optimization of elastic Full Waveform Inversion.

The Hessian-vector product requires propagating an auxiliary wavefield μ₁ that satisfies
the **same wave equation** as the forward wavefield u, but driven by a distributed virtual
source −(∂A/∂m · δm) u at every grid point and every time step.

---

## 1. The Wave Equation in the Solver

The code implements the first-order velocity-stress system in **buoyancy form**:

```
∂vx/∂t = (1/ρ) · (∂txx/∂x + ∂txz/∂z)         (momentum, x-component)
∂vz/∂t = (1/ρ) · (∂txz/∂x + ∂tzz/∂z)         (momentum, z-component)
∂txx/∂t = (λ+2μ) · ∂vx/∂x + λ · ∂vz/∂z       (Hooke's law, txx)
∂tzz/∂t = λ · ∂vx/∂x + (λ+2μ) · ∂vz/∂z       (Hooke's law, tzz)
∂txz/∂t = μ · (∂vx/∂z + ∂vz/∂x)               (Hooke's law, txz)
```

Written compactly as **A(m) u = f**, where:
- u = (vx, vz, txx, tzz, txz) is the wavefield state vector
- m = (λ, μ, ρ) or (Vp, Vs, ρ) are the model parameters
- f is the source term

**Key:** The momentum equations use **1/ρ** (buoyancy), not ρ directly. This affects
how the density virtual source is formulated.

---

## 2. The μ₁ Wavefield (Second-Order Adjoint State)

From the paper (eq 23), μ₁ satisfies:

```
A(m) μ₁ = −(∂A/∂m · δm) u
```

This is a **standard forward propagation** using the same wave equation A(m) and same
boundary conditions. The only difference from the original forward problem A(m)u = f is
the source:

- Original forward: point source f (e.g., Ricker wavelet at shot location)
- μ₁ propagation: distributed virtual source −(∂A/∂m · δm) u at every grid point, every time step

The virtual source depends on:
1. The forward wavefield u (read from checkpoints at each time step)
2. The model perturbation direction δm (from the optimizer's CG direction opt->d)

---

## 3. ∂A/∂m for Lamé Parameterization (λ, μ, ρ)

### 3.1 ∂A/∂ρ — Density

ρ appears **only** in the momentum equations through the buoyancy term 1/ρ:

```
∂/∂ρ [(1/ρ) · ∂σ_ij/∂x_j] = −(1/ρ²) · ∂σ_ij/∂x_j
```

The virtual source −(∂A/∂ρ · δρ) u for the μ₁ equation:

```
vsrc_vx = +(δρ/ρ²) · (∂txx_fwd/∂x + ∂txz_fwd/∂z)
vsrc_vz = +(δρ/ρ²) · (∂txz_fwd/∂x + ∂tzz_fwd/∂z)
vsrc_txx = 0
vsrc_tzz = 0
vsrc_txz = 0
```

**Properties:**
- Force-like source (injected into velocity equations)
- Computed from spatial derivatives of the forward **stress** field
- ρ does not appear in Hooke's law → no stress virtual source
- Physical meaning: denser regions reduce buoyancy, opposing acceleration

### 3.2 ∂A/∂λ — First Lamé Parameter

λ appears in the normal-stress Hooke's law (txx and tzz equations), multiplying ∇·v:

```
∂/∂λ [(λ+2μ)·∂vx/∂x + λ·∂vz/∂z] = ∂vx/∂x + ∂vz/∂z = ∇·v
```

The virtual source −(∂A/∂λ · δλ) u:

```
vsrc_vx  = 0
vsrc_vz  = 0
vsrc_txx = +δλ · (∂vx_fwd/∂x + ∂vz_fwd/∂z)
vsrc_tzz = +δλ · (∂vx_fwd/∂x + ∂vz_fwd/∂z)
vsrc_txz = 0
```

**Properties:**
- Stress-like source (injected into stress equations)
- Computed from spatial derivatives of the forward **velocity** field
- Isotropic: same source in txx and tzz (no shear)
- Physical meaning: λ perturbation scatters from volumetric strain (∇·v)
- Pure P-wave scattering

### 3.3 ∂A/∂μ — Shear Modulus

μ appears in all three stress equations:

```
∂(txx equation)/∂μ = 2 · ∂vx/∂x           (from the λ+2μ term)
∂(tzz equation)/∂μ = 2 · ∂vz/∂z           (from the λ+2μ term)
∂(txz equation)/∂μ = ∂vx/∂z + ∂vz/∂x     (full shear strain)
```

The virtual source −(∂A/∂μ · δμ) u:

```
vsrc_vx  = 0
vsrc_vz  = 0
vsrc_txx = +2δμ · ∂vx_fwd/∂x
vsrc_tzz = +2δμ · ∂vz_fwd/∂z
vsrc_txz = +δμ  · (∂vx_fwd/∂z + ∂vz_fwd/∂x)
```

**Properties:**
- Stress-like source (all three stress components)
- txx and tzz: deviatoric normal stress (not isotropic — note different derivatives)
- txz: full shear strain
- Physical meaning: μ perturbation scatters from both deviatoric and shear strain
- Generates P-to-P, P-to-S, and S-to-S mode conversions

### 3.4 Summary Table — Lamé Parameterization

| Parameter | vsrc_vx | vsrc_vz | vsrc_txx | vsrc_tzz | vsrc_txz | Scatters from |
|-----------|---------|---------|----------|----------|----------|---------------|
| δρ | +(δρ/ρ²)·∂σ_xj/∂x_j | +(δρ/ρ²)·∂σ_zj/∂x_j | 0 | 0 | 0 | Inertia (∇·σ) |
| δλ | 0 | 0 | +δλ·(∇·v) | +δλ·(∇·v) | 0 | Volumetric strain |
| δμ | 0 | 0 | +2δμ·∂vx/∂x | +2δμ·∂vz/∂z | +δμ·ε_xz | Deviatoric + shear |

where ε_xz = ∂vx/∂z + ∂vz/∂x is the engineering shear strain.

---

## 4. ∂A/∂m for Velocity Parameterization (Vp, Vs, ρ)

Using: **λ = ρ(Vp² − 2Vs²)** and **μ = ρVs²**

The chain rule converts the Lamé virtual sources.

### 4.1 ∂A/∂Vp — P-wave Velocity

Jacobian of the parameter mapping:
```
∂λ/∂Vp = 2ρVp       ∂μ/∂Vp = 0
```

The virtual source −(∂A/∂Vp · δVp) u:

```
vsrc_vx  = 0
vsrc_vz  = 0
vsrc_txx = +2ρVp·δVp · (∂vx_fwd/∂x + ∂vz_fwd/∂z)
vsrc_tzz = +2ρVp·δVp · (∂vx_fwd/∂x + ∂vz_fwd/∂z)
vsrc_txz = 0
```

**Properties:**
- Only isotropic stress sources (txx = tzz, no txz)
- Scatters exclusively from divergence ∇·v (compressional strain)
- Pure P-wave scattering — no S-wave coupling
- No force sources (Vp does not affect buoyancy)

### 4.2 ∂A/∂Vs — S-wave Velocity

Jacobian:
```
∂λ/∂Vs = −4ρVs      ∂μ/∂Vs = 2ρVs
```

Combining the δλ and δμ contributions:

For txx: −4ρVs·δVs · (∂vx/∂x + ∂vz/∂z) + 2·(2ρVs·δVs) · ∂vx/∂x

Simplifying:

```
vsrc_vx  = 0
vsrc_vz  = 0
vsrc_txx = −4ρVs·δVs · ∂vz_fwd/∂z
vsrc_tzz = −4ρVs·δVs · ∂vx_fwd/∂x
vsrc_txz = +2ρVs·δVs · (∂vx_fwd/∂z + ∂vz_fwd/∂x)
```

**Derivation of txx simplification:**
```
txx = (−4ρVs·δVs)·(∂vx/∂x + ∂vz/∂z) + (4ρVs·δVs)·∂vx/∂x
    = (−4ρVs·δVs)·∂vz/∂z
```

**Properties:**
- Cross-terms: txx depends on ∂vz/∂z, tzz depends on ∂vx/∂x
- Shear component from μ term
- Generates S-wave scattering and P-to-S mode conversions
- No force sources (Vs does not affect buoyancy)

### 4.3 ∂A/∂ρ — Density (Velocity Parameterization)

Density affects **three** things simultaneously:

```
∂(1/ρ)/∂ρ = −1/ρ²        (buoyancy)
∂λ/∂ρ = Vp² − 2Vs²       (first Lamé parameter)
∂μ/∂ρ = Vs²               (shear modulus)
```

The virtual source has three additive contributions:

**From buoyancy (direct density effect on momentum):**
```
vsrc_vx = +(δρ/ρ²) · (∂txx_fwd/∂x + ∂txz_fwd/∂z)
vsrc_vz = +(δρ/ρ²) · (∂txz_fwd/∂x + ∂tzz_fwd/∂z)
```

**From λ chain rule (density's effect on compressibility):**
```
vsrc_txx += +(Vp²−2Vs²)·δρ · (∂vx_fwd/∂x + ∂vz_fwd/∂z)
vsrc_tzz += +(Vp²−2Vs²)·δρ · (∂vx_fwd/∂x + ∂vz_fwd/∂z)
```

**From μ chain rule (density's effect on rigidity):**
```
vsrc_txx += +2Vs²·δρ · ∂vx_fwd/∂x
vsrc_tzz += +2Vs²·δρ · ∂vz_fwd/∂z
vsrc_txz += +Vs²·δρ  · (∂vx_fwd/∂z + ∂vz_fwd/∂x)
```

**Properties:**
- Only parameter that generates BOTH force AND stress virtual sources
- Force sources from buoyancy (inertia change)
- Stress sources from stiffness coupling (λ = ρ(Vp²−2Vs²), μ = ρVs²)
- This dual nature makes density the hardest parameter to resolve in FWI
- Its signature overlaps with both Vp and Vs

### 4.4 Summary Table — Velocity Parameterization

| Parameter | vsrc_vx | vsrc_vz | vsrc_txx | vsrc_tzz | vsrc_txz |
|-----------|---------|---------|----------|----------|----------|
| δVp | 0 | 0 | +2ρVp·δVp·(∇·v) | +2ρVp·δVp·(∇·v) | 0 |
| δVs | 0 | 0 | −4ρVs·δVs·∂vz/∂z | −4ρVs·δVs·∂vx/∂x | +2ρVs·δVs·ε_xz |
| δρ | +(δρ/ρ²)·∂σ_xj/∂x_j | +(δρ/ρ²)·∂σ_zj/∂x_j | (Vp²−2Vs²+2Vs²·∂vx/∂x...)·δρ | (similar) | +Vs²·δρ·ε_xz |

---

## 5. Mapping to Discrete FD Stencils

### 5.1 Forward Operator (elastic4.c)

The discrete time step in `elastic4.c`:

```
Phase F1 (velocity update):
  vx -= rox · [D⁻ₓ(txx) + D⁺z(txz)]
  vz -= roz · [D⁺ₓ(txz) + D⁻z(tzz)]

Phase F2 (stress update):
  txx -= l2m · D⁺ₓ(vx) + lam · D⁺z(vz)
  tzz -= lam · D⁺ₓ(vx) + l2m · D⁺z(vz)
  txz -= mul · [D⁻z(vx) + D⁻ₓ(vz)]
```

FD coefficients related to physical parameters:

```
rox = dt/(dx·ρ)             buoyancy (stagger-averaged in x)
roz = dt/(dx·ρ)             buoyancy (stagger-averaged in z)
l2m = (dt/dx)·(λ + 2μ)     P-wave modulus = (dt/dx)·ρ·Vp²
lam = (dt/dx)·λ             first Lamé = (dt/dx)·ρ·(Vp² − 2Vs²)
mul = (dt/dx)·μ             shear modulus = (dt/dx)·ρ·Vs²
```

FD stencil operators (4th order, c1 = 9/8, c2 = −1/24):

```
D⁻ₓ(f)[ix] = c1·(f[ix] − f[ix−1]) + c2·(f[ix+1] − f[ix−2])
D⁺ₓ(f)[ix] = c1·(f[ix+1] − f[ix]) + c2·(f[ix+2] − f[ix−1])
D⁻z(f)[iz] = c1·(f[iz] − f[iz−1]) + c2·(f[iz+1] − f[iz−2])
D⁺z(f)[iz] = c1·(f[iz+1] − f[iz]) + c2·(f[iz+2] − f[iz−1])
```

For 6th order add c3 = 3/640; for 8th order add c3 = 49/5120, c4 = −5/7168.

### 5.2 Linearizing the Discrete Operator

The virtual source is obtained by linearizing each time-step update with respect to the
FD coefficients. For any term `field -= coeff · D(other_field)`:

```
Perturbed:  (field + δfield) -= (coeff + δcoeff) · D(other_field + δother_field)
Linearized: δfield -= coeff · D(δother_field) + δcoeff · D(other_field_fwd)
                      ─────────────────────     ─────────────────────────────
                      wave operator on δu       virtual source from δm
```

The first term is the wave operator acting on the μ₁ wavefield (handled by elastic4).
The second term is the virtual source that we inject.

### 5.3 FD Coefficient Perturbations

**From Lamé perturbation (param=1): δm = (δλ, δμ, δρ)**

```
δl2m = (dt/dx) · (δλ + 2δμ)
δlam = (dt/dx) · δλ
δmul = (dt/dx) · δμ
δrox = −(rox/ρ) · δρ = −dt/(dx·ρ²) · δρ
δroz = −(roz/ρ) · δρ = −dt/(dx·ρ²) · δρ     (with stagger averaging)
```

**From velocity perturbation (param=2): δm = (δVp, δVs, δρ)**

First apply chain rule to get Lamé perturbations:
```
δλ = 2ρVp · δVp − 4ρVs · δVs + (Vp² − 2Vs²) · δρ
δμ = 2ρVs · δVs + Vs² · δρ
```

Then convert to FD coefficients:
```
δl2m = (dt/dx) · (δλ + 2δμ) = (dt/dx) · [2ρVp · δVp + Vp² · δρ]
δlam = (dt/dx) · [2ρVp · δVp − 4ρVs · δVs + (Vp² − 2Vs²) · δρ]
δmul = (dt/dx) · [2ρVs · δVs + Vs² · δρ]
δrox = −(rox/ρ) · δρ
δroz = −(roz/ρ) · δρ
```

Note: δl2m = (dt/dx)·∂(ρVp²)/∂(Vp,ρ) — Vs drops out because l2m = (dt/dx)ρVp².

### 5.4 Virtual Source Injection in FD Code

At each time step of the μ₁ propagation:

**Step 1:** Propagate μ₁ using elastic4 (same wave operator, no physical source)

**Step 2:** Inject Phase F1 virtual source (velocity, from δρ):

```c
// 4th order — same stencils as elastic4 Phase F1 applied to forward stress
for (ix = mod.ioXx; ix < mod.ieXx; ix++) {
    for (iz = mod.ioXz; iz < mod.ieXz; iz++) {
        // Spatial derivatives of forward stress (same as Phase F1)
        Dtxx = c1*(txx_fwd[ix*n1+iz]     - txx_fwd[(ix-1)*n1+iz])
             + c2*(txx_fwd[(ix+1)*n1+iz] - txx_fwd[(ix-2)*n1+iz]);
        Dtxz = c1*(txz_fwd[ix*n1+iz+1]   - txz_fwd[ix*n1+iz])
             + c2*(txz_fwd[ix*n1+iz+2]   - txz_fwd[ix*n1+iz-1]);

        // Inject: born_vx -= δrox · D(σ_fwd)
        born_vx[ix*n1+iz] -= delta_rox[ix*n1+iz] * (Dtxx + Dtxz);
    }
}

// Similarly for vz:
for (ix = mod.ioZx; ix < mod.ieZx; ix++) {
    for (iz = mod.ioZz; iz < mod.ieZz; iz++) {
        Dtxz = c1*(txz_fwd[(ix+1)*n1+iz] - txz_fwd[ix*n1+iz])
             + c2*(txz_fwd[(ix+2)*n1+iz] - txz_fwd[(ix-1)*n1+iz]);
        Dtzz = c1*(tzz_fwd[ix*n1+iz]     - tzz_fwd[ix*n1+iz-1])
             + c2*(tzz_fwd[ix*n1+iz+1]   - tzz_fwd[ix*n1+iz-2]);

        born_vz[ix*n1+iz] -= delta_roz[ix*n1+iz] * (Dtxz + Dtzz);
    }
}
```

**Step 3:** Inject Phase F2 virtual source (stress, from δλ and δμ):

```c
// Normal stress (txx, tzz) — same stencils as elastic4 Phase F2 applied to forward velocity
for (ix = mod.ioTx; ix < mod.ieTx; ix++) {
    for (iz = mod.ioTz; iz < mod.ieTz; iz++) {
        // Spatial derivatives of forward velocity (same as Phase F2)
        dvx = c1*(vx_fwd[(ix+1)*n1+iz] - vx_fwd[ix*n1+iz])
            + c2*(vx_fwd[(ix+2)*n1+iz] - vx_fwd[(ix-1)*n1+iz]);
        dvz = c1*(vz_fwd[ix*n1+iz]     - vz_fwd[ix*n1+iz-1])
            + c2*(vz_fwd[ix*n1+iz+1]   - vz_fwd[ix*n1+iz-2]);

        // Inject: born_txx -= δl2m·D⁺ₓ(vx_fwd) + δlam·D⁺z(vz_fwd)
        born_txx[ix*n1+iz] -= delta_l2m[ix*n1+iz]*dvx + delta_lam[ix*n1+iz]*dvz;
        born_tzz[ix*n1+iz] -= delta_lam[ix*n1+iz]*dvx + delta_l2m[ix*n1+iz]*dvz;
    }
}

// Shear stress (txz)
for (ix = mod.ioSx; ix < mod.ieSx; ix++) {
    for (iz = mod.ioSz; iz < mod.ieSz; iz++) {
        dvx_dz = c1*(vx_fwd[ix*n1+iz]   - vx_fwd[ix*n1+iz-1])
                + c2*(vx_fwd[ix*n1+iz+1] - vx_fwd[ix*n1+iz-2]);
        dvz_dx = c1*(vz_fwd[ix*n1+iz]   - vz_fwd[(ix-1)*n1+iz])
                + c2*(vz_fwd[(ix+1)*n1+iz] - vz_fwd[(ix-2)*n1+iz]);

        // Inject: born_txz -= δmul·(D⁻z(vx_fwd) + D⁻ₓ(vz_fwd))
        born_txz[ix*n1+iz] -= delta_mul[ix*n1+iz] * (dvx_dz + dvz_dx);
    }
}
```

**Step 4:** Record μ₁ at receivers using getRecTimes()

### 5.5 Complete μ₁ Time Step Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ One time step of μ₁ propagation                                 │
│                                                                  │
│ 1. elastic4(born_vx, born_vz, born_txx, born_tzz, born_txz,    │
│             rox, roz, l2m, lam, mul)                             │
│    → propagates μ₁ with same wave operator (no physical source) │
│                                                                  │
│ 2. born_vx  -= δrox · D(σ_fwd)       ← δρ force virtual source │
│    born_vz  -= δroz · D(σ_fwd)                                  │
│                                                                  │
│ 3. born_txx -= δl2m·D⁺ₓ(vx_fwd) + δlam·D⁺z(vz_fwd)           │
│    born_tzz -= δlam·D⁺ₓ(vx_fwd) + δl2m·D⁺z(vz_fwd)           │
│    born_txz -= δmul·(D⁻z(vx_fwd) + D⁻ₓ(vz_fwd))              │
│    ← δλ, δμ stress virtual sources                              │
│                                                                  │
│ 4. getRecTimes(born_vx, born_vz, ...)  ← record μ₁ at receivers│
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. From μ₁ to Hessian-Vector Product

### 6.1 The Full H·v Computation

The Gauss-Newton Hessian-vector product H_GN · δm = J^T(J · δm) requires:

```
Step 1: Forward propagation                  A(m) u = f
        → save checkpoints                  [already done for gradient]

Step 2: μ₁ propagation (this document)       A(m) μ₁ = −(∂A/∂m · δm) u
        → record μ₁ at receivers: d_μ₁ = R μ₁

Step 3: μ₂ propagation (second adjoint)      A†(m) μ₂ = −R† d_μ₁
        → same adjoint operator as gradient  [reuses adj_shot]

Step 4: Imaging condition                    (H·δm)_i = −(∂A/∂m_i · u)^T · μ₂
        → same cross-correlation as gradient [reuses accumGradient]
```

The gradient and H·v use the **same imaging condition** (accumGradient). The only
difference is which adjoint wavefield is cross-correlated with the forward:
- **Gradient:** cross-correlate u with λ (adjoint of data residual)
- **H·v:** cross-correlate u with μ₂ (adjoint of μ₁ receiver data)

### 6.2 Computational Cost

| Operation | Propagations per shot |
|-----------|-----------------------|
| Gradient (forward u + adjoint λ) | 2 |
| One H·v (μ₁ forward + μ₂ adjoint) | 2 |
| Truncated Newton with k CG steps | 2 + 2k |
| L-BFGS iteration | 2 |

### 6.3 Where δm Comes From

In the Truncated Newton framework:
- The optimizer (`trn_run`) maintains CG direction vector `opt->d`
- When it returns `OPT_HESS`, it asks: compute `opt->Hd = H · opt->d`
- `opt->d` is a flat vector with the same layout as the model vector:
  ```
  opt->d[0..nint-1]          = δλ (or δVp)
  opt->d[nint..2*nint-1]     = δμ (or δVs)
  opt->d[2*nint..3*nint-1]   = δρ
  ```
- The function `perturbFDcoefficients()` converts this to (δl2m, δlam, δmul, δrox, δroz)

---

## 7. Sign Convention Summary

The forward operator in the code uses the convention:

```
field -= coeff · D(other_field)
```

Linearizing: `δfield -= coeff · D(δother_field) + δcoeff · D(other_field_fwd)`

The virtual source injection follows the same `−=` sign:

```
born_field -= δcoeff · D(field_fwd)
```

This is consistent: the virtual source term has the same sign as the wave operator term
because it comes from the same linearization. No additional negation is needed for the
μ₁ propagation.

For the μ₂ (second adjoint) propagation, the sign convention matches `adj_shot` with
the `negate_source=0` flag (Born data is NOT negated, unlike the gradient residual which
requires negation for the Lagrangian convention).
