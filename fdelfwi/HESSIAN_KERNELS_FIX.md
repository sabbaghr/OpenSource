# Pseudo-Hessian Kernel Fix: Code vs Paper Equations

## Reference: Liu et al. (2022), Appendix A, eq A20-A22

The Shin pseudo-Hessian at grid point x is (eq A20):

    H(x,x) = ∫ [(∂C/∂m_i) Dv(x)] · [(∂C/∂m_j) Dv(x)] dt

where C is the stiffness matrix, D is the spatial derivative operator,
v is the source (forward) wavefield particle velocity.

## Stiffness matrix C and its derivatives (2D isotropic, Voigt notation)

    C = [λ+2μ   λ      0  ]
        [λ       λ+2μ   0  ]
        [0       0      μ  ]

    Dv = [∂ₓvₓ, ∂zvz, ∂ₓvz + ∂zvₓ]ᵀ = [exx, ezz, exz]ᵀ

    ∂C/∂λ = [1  1  0]     ∂C/∂μ = [2  0  0]
             [1  1  0]              [0  2  0]
             [0  0  0]              [0  0  1]

## Correct Hessian kernels (Lamé parameterization)

### K_λ (stress-space vector)

    K_λ = (∂C/∂λ) · Dv = [div(v), div(v), 0]ᵀ

where div(v) = ∂ₓvₓ + ∂zvz.

    H_λλ = |K_λ|² = 2 · div(v)²

### K_μ (stress-space vector)

    K_μ = (∂C/∂μ) · Dv = [2·∂ₓvₓ, 2·∂zvz, ∂ₓvz + ∂zvₓ]ᵀ = [2exx, 2ezz, exz]ᵀ

    H_μμ = |K_μ|² = 4·exx² + 4·ezz² + exz²

### K_ρ (velocity-space vector)

ρ appears in the velocity equation: ρ·∂ₜv = Dᵀσ.
Virtual source: (∂A/∂ρ)u acts on the velocity equation with source = ∂ₜv.

    K_ρ = [∂ₜvₓ, ∂ₜvz]ᵀ

    H_ρρ = |K_ρ|² = (∂ₜvₓ)² + (∂ₜvz)²

### Cross-terms in Lamé space

    K_λ · K_μ = div·2exx + div·2ezz + 0·exz
              = 2·div·(exx + ezz) = 2·div·div = 2·div²
              = H_λλ   (ALWAYS identical!)

    K_λ · K_ρ = 0   (K_λ is in stress-space, K_ρ is in velocity-space → orthogonal)
    K_μ · K_ρ = 0   (same reason)

So the Lamé block Hessian has structure:

    H_Lamé = [H_λλ    H_λλ    0    ]
             [H_λλ    H_μμ    0    ]
             [0        0      H_ρρ ]

The λ-ρ and μ-ρ blocks are exactly zero. The λ-μ cross-term equals H_λλ.

## What our code currently computes (WRONG)

### K_λ (line 239-240):
```c
K_lam_tmp[ig] = (fwd_txx[ig] + fwd_tzz[ig]) * div_f;
```
This is `(σ_xx + σ_zz) · div(v)` — a SCALAR.
**Wrong:** includes forward stress. Should be just `div(v)` (or `sqrt(2)·div(v)` for norm).

### K_μ normal (line 242):
```c
K_muu_tmp[ig] = 2.0f*(fwd_txx[ig]*dvxdx_f + fwd_tzz[ig]*dvzdz_f);
```
This is `2(σ_xx·exx + σ_zz·ezz)` — a SCALAR mixing stress and strain.
**Wrong:** should be just `4exx² + 4ezz²` (the squared norm of the vector [2exx, 2ezz]).

### K_μ shear (line 328):
```c
K_muu_S = fwd_txz[ig] * (dvxdz_f + dvzdx_f);
```
This is `σ_xz · exz` — stress × strain.
**Wrong:** should be `exz²`.

### K_ρ (lines 372-394):
```c
K_rho_vx = fwd_vx[ig] * dvx_dt;  // vx · ∂ₜvx
K_rho_tmp += 0.5f * K_rho_vx / rho;
```
This is `(1/ρ) · vx · ∂ₜvx` — velocity × acceleration / ρ.
**Wrong:** should be `(∂ₜvx)²`. No velocity factor, no 1/ρ.

### Off-diagonal products (lines 416-418):
```c
hess_lam_muu[ig] += Kl * Km;   // scalar × scalar
hess_lam_rho[ig] += Kl * Kr;   // scalar × scalar ≠ 0 (wrong!)
hess_muu_rho[ig] += Km * Kr;   // scalar × scalar ≠ 0 (wrong!)
```
With the wrong scalar K values, these are all nonzero. But physically:
- H_λρ = 0 (stress-space vs velocity-space → orthogonal)
- H_μρ = 0 (same)
- H_λμ = H_λλ (proven above)

## What the code should compute

### Directly accumulate H_ii products (no K_tmp arrays needed):

At P grid (λ and μ normal part):
```c
float div_f = dvxdx_f + dvzdz_f;
hess_lam[ig]     += 2.0f * div_f * div_f;              // 2·div²
hess_muu[ig]     += 4.0f*(dvxdx_f*dvxdx_f + dvzdz_f*dvzdz_f);  // 4exx² + 4ezz²
hess_lam_muu[ig] += 2.0f * div_f * div_f;              // = H_λλ (always)
// H_λρ and H_μρ: don't accumulate (always zero in Lamé space)
```

At Txz grid (μ shear part), scattered to P:
```c
float exz = dvxdz_f + dvzdx_f;
float exz_sq = exz * exz;
hess_muu[P neighbors] += 0.25f * exz_sq;  // scatter exz² to P grid
```

At Vx/Vz grid (ρ), scattered to P:
```c
float dvx_dt = (fwd_vx - fwd_vx_prev) / dt;
hess_rho[P neighbors] += 0.5f * dvx_dt * dvx_dt;  // scatter (∂ₜvx)²

float dvz_dt = (fwd_vz - fwd_vz_prev) / dt;
hess_rho[P neighbors] += 0.5f * dvz_dt * dvz_dt;  // scatter (∂ₜvz)²
```

### Wavefield energy for P1/P2 (precond=2):

Paper eq 5: `H₀ = ∫ ∂ₜv · ∂ₜv dt` (acceleration²).
Our code: `Ws += vx² + vz²` (velocity²).
**Should be:** `Ws += (∂ₜvx)² + (∂ₜvz)²` (= H_ρρ, same quantity).

## Verification against paper eq A21 (velocity parameterization)

After chain rule (∂λ/∂Vp = 2ρVp, ∂μ/∂Vp = 0):

    H_VpVp = (2ρVp)² · H_λλ = (2ρα)² · 2·div² = 8ρ²α²·div²

This matches eq A21: H_αα = 8ρ²α² ∫ (∂ₓvₓ + ∂zvz)² dt ✓

For Vs (∂λ/∂Vs = -4ρVs, ∂μ/∂Vs = 2ρVs):

    H_VsVs = (-4ρβ)²·H_λλ + 2·(-4ρβ)·(2ρβ)·H_λμ + (2ρβ)²·H_μμ
           = 16ρ²β²·2div² + 2·(-8ρ²β²)·2div² + 4ρ²β²·(4exx²+4ezz²+exz²)
           = 32ρ²β²div² - 32ρ²β²div² + 4ρ²β²·(4exx²+4ezz²+exz²)
           = 16ρ²β²(exx²+ezz²) + 4ρ²β²·exz²

The paper eq A21 gives:
    H_ββ = 16ρ²β² ∫ [(∂ₓvₓ)²+(∂zvz)²] dt + 4ρ²β² ∫ (exz)² dt

This matches! ✓

For ρ: H_ρρ = (∂ₜvₓ)² + (∂ₜvz)², matches eq A21 ✓
