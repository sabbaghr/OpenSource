# Pseudo-Hessian Preconditioner: Paper References vs Code Implementation

## 1. Paper Formulations

### 1.1 Métivier et al. (2014) — Geophysical Prospecting, 62, 1353–1375

**Context:** Single-parameter (acoustic Vp), frequency domain, preconditioner for TRN inner CG loop and l-BFGS.

**Pseudo-Hessian definition (eq 38):**

The Jacobian column α_j(m) = ∂u/∂m_j satisfies A(m)α_j = −(∂A/∂m_j)u (eq 37). Shin's approximation replaces A(m) with the identity I, giving:

    H̃_ij(m) = ( (∂A/∂m_i) u(m) )^T ( (∂A/∂m_j) u(m) ),   i,j = 1,...,M     (eq 38)

This is the correlation of virtual sources (radiation patterns × incident wavefield), NOT the correlation of scattered wavefields (which would be the true Hessian).

**Diagonal preconditioner (eq 39):**

    P_k = diag( 1 / H̃_ii(m_k) ),   i = 1,...,M

**Damped preconditioner (eq 40-41):**

Because of fast wavefield decay with depth, small values appear on H̃ for deep parameters. Direct use of P_k causes numerical instabilities. Fix:

    C_k = max_j( H̃_jj(m_k) )                                               (eq 40)
    P^θ_k = diag( 1 / ( H̃_ii(m_k) + θ·C_k ) ),   i = 1,...,M              (eq 41)

where θ ∈ ℝ is a threshold parameter. C_k is the **global maximum** over all grid points (single-parameter case, so also over all M unknowns).

**Norm preservation (eq 42-43):**

    P^{ν,θ}_k = ν · P^θ_k                                                    (eq 42)

    ν = ||∇f(m_k)|| / ||P^θ_k ∇f(m_k)||                                     (eq 43)

**Justification:** "Since the stopping criterion for the linear system is based on the reduction of the linear residuals with respect to the norm of the gradient, it appears natural that the preconditioner conserves the gradient norm."

**Usage:** P^{ν,θ}_k is used:
- As preconditioner of the TRN inner CG loop (preconditioned linear system)
- As preconditioner of the nonlinear conjugate gradient method
- As preconditioner of the l-BFGS method

**Practical θ value:** θ = 10⁻² chosen as "an average value yielding a preconditioner with good properties for the four minimization methods."

---

### 1.2 Métivier et al. (2017) — SIAM Review, 59(1), 153–195

**Context:** Single-parameter (acoustic Vp), frequency domain, comprehensive review paper. Extends Métivier 2014.

**Section 4.5 Preconditioning (p. 172):**

Refers to the same Shin pseudo-Hessian strategy. For TRN, the preconditioner is used inside the inner CG loop. The paper discusses two l-BFGS preconditioning strategies:

1. **Outer l-BFGS (Algorithm 2, p. 174):** An l-BFGS approximation Q_k is built from previous gradients. The CG inner loop solves Q_k H(m_k) Δm_k = −Q_k ∇f(m_k). If only 1 CG iteration: Δm_k = −Q_k ∇f(m_k), which is the standard l-BFGS descent direction.

2. **Inner l-BFGS (Algorithm 3, p. 175):** An l-BFGS approximation Q_2 is built from the CG residuals during the inner loop, and is used to precondition the CG iterations. A separate outer Q_1 preconditions the initial residual. The preconditioners Q_1 and Q_2 are swapped between outer iterations.

**Key observation (Algorithm 2 vs 3):** In Algorithm 2, the pseudo-Hessian preconditioner is NOT explicitly mentioned — the l-BFGS Q_k itself serves as the preconditioner. In Algorithm 3, the inner l-BFGS Q_2 preconditions the CG loop, which is where the pseudo-Hessian could be incorporated as the initial H₀ for Q_2.

**No explicit mention of norm preservation in this paper** — it appears only in Métivier 2014.

---

### 1.3 Yang et al. (2018) — SIAM J. Sci. Comput., 40(4), B1101–B1130

**Context:** Multi-parameter (Vp, ρ, Q_inv), time-domain visco-acoustic, TRN with SEISCOPE toolbox.

**Section 4. Preconditioning (p. B1114):**

**Pseudo-Hessian definition (eq 45):**

    H̃_ij = ∫₀ᵀ dt ( w^† (∂A/∂m_i) w )^† ( w^† (∂A/∂m_j) w )

where w is the incident (forward) wavefield. This is the time-domain equivalent of Métivier's eq 38.

**Multiparameter block structure (eq 46):**

For N_par parameter classes, the pseudo-Hessian has a block structure:

    P = [ diag H̃₁₁  diag H̃₁₂  diag H̃₁₃ ]⁻¹
        [ diag H̃₂₁  diag H̃₂₂  diag H̃₂₃ ]
        [ diag H̃₃₁  diag H̃₃₂  diag H̃₃₃ ]

Each block diag H̃_ij is a diagonal matrix (one value per grid point). The diagonal blocks (H̃₁₁, H̃₂₂, H̃₃₃) are autocorrelations of sensitivity kernels. The off-diagonal blocks (H̃₁₂, etc.) are cross-correlations describing inter-parameter coupling.

**ε-regularization:** "It is necessary to add a regularization parameter ε in the diagonal blocks in order to solve the matrix inversion stably." (No specific formula given for ε — deferred to Appendix B.)

**Application:** 3×3 linear system solved per grid point (Cramer's rule or direct inversion).

**Section 5. Parameter scaling (eq 47-48):**

Unity-based normalization to [0,1]:

    m̄_i = (m_i − m_i^min) / (m_i^max − m_i^min)                           (eq 47)

**Additional tunable scaling (eq 48):**

    P = [ s₁s₁ diag H̃₁₁   s₁s₂ diag H̃₁₂   s₁s₃ diag H̃₁₃ ]⁻¹
        [ s₁s₂ diag H̃₁₂   s₂s₂ diag H̃₂₂   s₂s₃ diag H̃₂₃ ]
        [ s₁s₃ diag H̃₁₃   s₂s₃ diag H̃₂₃   s₃s₃ diag H̃₃₃ ]

where s_i are "tunable scaling factors associated with the parameter m_i ∈ m." The s_i are O(1) values tuned by the user to "promote the features from specific parameters whose physical sensitivity is weaker."

**Practical s_i values (p. B1120):** s₁ = 1 (Vp), s₂ = 2.5 (ρ), s₃ = 10 (Q_inv). These are applied AFTER unity normalization (eq 47), so they act on parameters already in [0,1].

**No norm preservation mentioned** in Yang et al. (2018). The full block preconditioner inherently produces well-scaled output because the 3×3 solve balances inter-parameter coupling.

---

## 2. Current Code Implementation

### 2.1 File: `yang_precond.c`

**`buildBlockPrecond()`** — Builds the preconditioner diagonal:

1. **Extract** padded Hessian diagonal arrays (H_λλ, H_μμ, H_ρρ) → flat arrays
2. **Chain rule** (if param=2): convert diagonal from Lamé to velocity space:
   - H_VpVp = (2ρVp)² · H_λλ
   - H_VsVs = (4ρVs)² · H_λλ + (2ρVs)² · H_μμ
   - H_ρρ = (Vp²−2Vs²)² · H_λλ + Vs⁴ · H_μμ + H_ρρ
3. **m₀² scaling**: H̃_ii = m0_i² · H_ii (chain rule for normalized space)
4. **Global damping** (Métivier eq 40-41): C_k = max(H̃_ii), P_ii = H̃_ii + θ·C_k
5. **Store** P_ii in P11, P22, P33. Off-diagonals P12, P13, P23 = 0.

**`applyBlockPrecond()`** — Applies the preconditioner with norm preservation:

1. **Compute** ||g||² (before the solve, in double precision)
2. **Divide** z_i = g_i / P_ii (in double precision to avoid underflow)
3. **Norm preservation** (Métivier eq 42-43): ν = ||g|| / ||z||, output = ν · z

### 2.2 File: `scaling.c`

Three normalization modes:
- **scaling=1** (Brossier): m0 = mean(|m_init|), m* = m/m0
- **scaling=2** (Yang eq 47): m0 = Δm = m_max − m_min, m* = (m − m_min)/Δm
- **scaling=3** (Ludovic/range): m0 = Δm = m_max − m_min, m* = m/Δm (no shift)

For all modes: g* = g · m0 (chain rule), H* = m0² · H (chain rule).

### 2.3 File: `fwi_inversion.c`

- Preconditioner arrays allocated when `precond=1`
- `buildBlockPrecond` called with s_i = m0_i (from scaling)
- `applyBlockPrecond` called on gradient for PLBFGS (algorithm=2), PNLCG (algorithm=3), PTRN (algorithm=6), PEnriched (algorithm=7)
- For PLBFGS: also called on `opt.q_plb` between L-BFGS two-loop recursion halves (OPT_PREC flag)

---

## 3. Detailed Comparison: Papers vs Code

### 3.1 Pseudo-Hessian Computation

| Aspect | Métivier 2014 | Yang 2018 | Our Code |
|:-------|:-------------|:----------|:---------|
| Domain | Frequency | Time | Time |
| Physics | Acoustic | Visco-acoustic | Elastic |
| Parameters | 1 (Vp) | 3 (Vp,ρ,Q) | 3 (λ,μ,ρ) or (Vp,Vs,ρ) |
| H̃_ij formula | (∂A/∂m_i · u)^† (∂A/∂m_j · u) | Same structure, time integral | K_i · K_j summed over t |
| Blocks used | Diagonal only (1 param) | Full 3×3 block | **Diagonal only** |
| Off-diagonal | N/A (1 param) | Yes (eq 46, 48) | **No** (zeroed) |

**Key difference:** Yang 2018 uses the full 3×3 block pseudo-Hessian including off-diagonal coupling terms (H̃₁₂, H̃₁₃, H̃₂₃). Our code currently uses only the diagonal (H̃₁₁, H̃₂₂, H̃₃₃). This means we do NOT account for inter-parameter coupling in the preconditioner. Per Ludovic's email recommendation: "You might remain first with a diagonal scaling, acting only on diagonal blocks associated with a single parameter... Once this works, you can add the diagonal part of the off-diagonal blocks."

### 3.2 Regularization / Damping

| Aspect | Métivier 2014 | Yang 2018 | Our Code |
|:-------|:-------------|:----------|:---------|
| Formula | θ · max_j(H̃_jj) | ε in diagonal blocks | θ · max(H̃_ii) globally |
| Scope of max | All grid points (1 param) | Per block? (unclear) | All params × all grid points |
| θ value | 10⁻² | Not specified | User param `precond_eps` |
| Applied to | Diagonal elements | Diagonal blocks before inversion | Diagonal elements |

**Match:** Our implementation matches Métivier 2014 eq 40-41 exactly. The global max C_k is taken over all three parameters AND all grid points, then β = θ·C_k is added to every diagonal element.

**Yang's ε is different:** Yang adds ε to the diagonal blocks before the 3×3 matrix inversion, which is a per-block regularization. Since we don't use the block structure, this is not applicable.

### 3.3 Parameter Scaling

| Aspect | Yang 2018 | Ludovic email | Our Code |
|:-------|:----------|:-------------|:---------|
| Normalization | (m−m_min)/(m_max−m_min) → [0,1] | m/(m_max−m_min), no shift | scaling=2 (Yang) or scaling=3 (range) |
| s_i factors | O(1) tunable: 1, 2.5, 10 | Not needed (range handles it) | s_i = m0_i from scaling |
| s_i applied to | Preconditioner only (eq 48) | N/A | H̃_ii = m0² · H_ii |
| Purpose of s_i | Promote weak parameters (Q) | — | Chain rule consistency |

**Critical distinction:** In Yang 2018, the s_i are **user-tuned O(1) factors** applied AFTER unity normalization. They do NOT equal m0 — they are small adjustments (1, 2.5, 10) to rebalance parameters whose physical sensitivity is inherently weaker (like Q_inv vs Vp). In our code, we pass s_i = m0_i and multiply H by m0², which is the chain rule transformation to normalized space — a different purpose. The tunable s_i from Yang could be added as additional factors on top of our m0² scaling, but this is not currently implemented.

### 3.4 Norm Preservation

| Aspect | Métivier 2014 | Yang 2018 | Our Code |
|:-------|:-------------|:----------|:---------|
| Norm preservation | Yes (eq 42-43) | No | **Yes** |
| Formula | ν = \|\|∇f\|\| / \|\|P^θ ∇f\|\| | — | ν = \|\|g\|\| / \|\|P_θ·g\|\| |
| Justification | Stopping criterion compatibility | — | Same + line search compatibility |
| Precision | Not specified | — | Double precision |

**Match:** Our implementation exactly follows Métivier 2014 eq 42-43. The norm preservation ensures ||P^{ν,θ} g|| = ||g||, so the preconditioner only changes direction, not magnitude.

**Yang 2018 does not use norm preservation** because the full 3×3 block inversion inherently produces well-scaled output (the matrix inverse naturally balances the magnitudes).

### 3.5 Chain Rule for Parameterization Change

| Aspect | Yang 2018 | Our Code |
|:-------|:----------|:---------|
| Native parameterization | (ρ, κ_inv, Q_inv) | (λ, μ, ρ) |
| Target parameterization | (Vp, ρ, Q_inv) via chain rule | (Vp, Vs, ρ) via chain rule |
| Gradient chain rule | g_target = J^T · g_native | g_vel = J^T · g_Lamé (in `extractGradientVector`) |
| Hessian chain rule | H_target = J^T · H_native · J (full) | H_vel_pp = Σ_k J_kp² · H_Lamé_kk (**diagonal only**) |
| When applied | Before preconditioning | Before preconditioning |

**Key simplification in our code:** Since we only use the diagonal of H, the chain rule reduces to H_vel_pp = Σ_k J_kp² · H_Lamé_kk (sum of squared Jacobian elements times diagonal Hessian). This is cheaper than the full J^T H J transformation but loses off-diagonal coupling information.

### 3.6 Integration with L-BFGS / PLBFGS

| Aspect | Métivier 2014 | Métivier 2017 | Our Code |
|:-------|:-------------|:-------------|:---------|
| L-BFGS usage | P used as H₀ in l-BFGS | Q_k built from gradient history | P applied via OPT_PREC flag |
| PLBFGS flow | Not detailed | Algorithms 2 & 3 | SEISCOPE PLBFGS: first loop → apply P to q_plb → second loop |
| Initial step | γ₀ computed so model update ~ hundreds of m/s | Not specified | α₀ = 1/\|\|grad_vec\|\| |
| H₀ = | P^{ν,θ} | l-BFGS Q_k | P applied between two-loop halves, γ also applied |

**Our PLBFGS integration:** The SEISCOPE PLBFGS splits the two-loop recursion. After the first loop produces q_plb, our code applies `applyBlockPrecond(q_plb, q_plb, ...)` (in-place). Then `descent2_plbfgs` applies the γ = s^T y / ||y||² scaling and completes the second loop. The effective H₀ is γ · P^{ν,θ}.

---

## 4. What is Implemented vs What Could Be Added

### Currently Implemented ✓
- Diagonal pseudo-Hessian (Shin 2001 / Métivier 2014)
- Global damping β = θ · max(H̃_ii) (Métivier 2014, eq 40-41)
- Norm preservation ν (Métivier 2014, eq 42-43)
- Chain rule Lamé → velocity (diagonal only)
- Three scaling modes: Brossier (scaling=1), Yang (scaling=2), Range (scaling=3)
- Double precision arithmetic in apply step
- PLBFGS integration via SEISCOPE OPT_PREC mechanism

### Not Yet Implemented (future work)
- **Off-diagonal blocks** (Yang 2018, eq 46): H̃₁₂, H̃₁₃, H̃₂₃ for inter-parameter coupling. Per Ludovic: add these incrementally after diagonal works.
- **Tunable s_i factors** (Yang 2018, eq 48): additional O(1) scaling on top of m0² to promote weak parameters. Would be useful for elastic FWI where ρ sensitivity is much weaker than Vp/Vs.
- **Full 3×3 block inversion** (Yang 2018, eq 46): Cramer's rule per grid point. Was previously implemented but removed due to conditioning issues. Could be revisited with better regularization.
- **Inner l-BFGS preconditioning** (Métivier 2017, Algorithm 3): build l-BFGS approximation from CG residuals during TRN inner loop.
- **Depth-dependent blending** (Ludovic email): interpolate smooth depth scaling near acquisition with wavefield-based preconditioning at depth.

---

## 5. Ludovic Métivier — Practical Implementation Guidance (Email)

The following points are from direct correspondence with Ludovic Métivier regarding integration with the SEISCOPE optimization toolbox. These clarify practical details not covered in the published papers:

### 5.1 Normalization First

> "Usually we know within which bounds we are looking for the model parameters, let say m is between m_1 and m_2, then we work with m* = m / (m_2-m_1) (pointwise)"

**Key detail:** He writes `m* = m / (m_2 - m_1)`, NOT `m* = (m - m_1) / (m_2 - m_1)`. This is division by the range only, no shift. This corresponds to our `scaling=3`.

> "This is a kind of block diagonal preconditioning, based on which you can further introduce a wavefield based preconditioning"

The range normalization is itself a preconditioner (diagonal scaling by 1/Δm per parameter class). The wavefield preconditioner is a second layer on top.

> "This should already help you in managing not too big or too small number for machine representation."

This applies to the model and gradient vectors seen by the optimizer. The pseudo-Hessian values remain large (O(10²⁸–10³⁰)) because the raw kernels K² are inherently large — the normalization does not fix the Hessian magnitude. The norm preservation (Métivier 2014, eq 42-43) handles that.

### 5.2 SEISCOPE Toolbox Usage

> "You need to provide preconditioned version of l-BFGS and TrN with both the gradient and the preconditioned gradient at first call. Then simply apply the preconditioner on the good quantities each time the flag PREC is returned."

This confirms:
- At `OPT_INIT`: provide both `grad_vec` (unpreconditioned) and `grad_preco_vec` (preconditioned)
- At `OPT_PREC`: apply preconditioner to `opt.q_plb` (the intermediate vector in the L-BFGS two-loop recursion)

### 5.3 Instabilities Near Acquisition

> "Sometimes wavefield based preconditioner can be a little bit instable, for instance if you have a poor sampling in source or receiver. This might lead to instabilities nearby the acquisition. In this case, we interpolate a linear or polynomial depth scaling near the acquisition (usually near the surface) with the wavefield based preconditioning in deeper regions."

This is more sophisticated than our current `grad_taper` (cosine taper around sources/receivers). Ludovic suggests a **blended approach**: smooth depth-dependent scaling near the surface, transitioning to wavefield-based preconditioning at depth. This would avoid the sharp taper artifacts while still handling the source/receiver singularity.

### 5.4 Multi-Parameter Strategy

> "You might remain first with a diagonal scaling, acting only on diagonal blocks associated with a single parameter. This is what we call illumination based preconditioner. Once this works, you can add the diagonal part of the off-diagonal blocks, dealing with parameter couplings. You will probably have to tune again the weight of these off-diagonal blocks with respect to the diagonal blocks to get something stable."

This is exactly our implementation path:
1. **Current:** Diagonal only (H̃₁₁, H̃₂₂, H̃₃₃) — illumination preconditioner ✓
2. **Next step:** Add off-diagonal coupling (H̃₁₂, H̃₁₃, H̃₂₃) with tunable weights
3. **Future:** Full 3×3 block inversion (Yang eq 46)

---

## 6. References

1. **Shin, C., Jang, S., and Min, D.J. (2001).** Improved amplitude preservation for prestack depth migration by inverse scattering theory. Geophysics, 66(6), 1895–1903.

2. **Métivier, L., Bretaudeau, F., Brossier, R., Operto, S., and Virieux, J. (2014).** Full waveform inversion and the truncated Newton method: quantitative imaging of complex subsurface structures. Geophysical Prospecting, 62, 1353–1375.

3. **Métivier, L., Brossier, R., Operto, S., and Virieux, J. (2017).** Full Waveform Inversion and the Truncated Newton Method. SIAM Review, 59(1), 153–195.

4. **Yang, P., Brossier, R., Métivier, L., Virieux, J., and Zhou, W. (2018).** A time-domain preconditioned truncated Newton approach to visco-acoustic multiparameter full waveform inversion. SIAM J. Sci. Comput., 40(4), B1101–B1130.

5. **Ludovic Métivier (2025).** Personal communication (email) regarding SEISCOPE optimization toolbox integration and range-based normalization.
