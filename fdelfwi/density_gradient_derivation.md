# Density Gradient Derivation for Elastic FWI

## 1. The Forward Velocity Update

From `elastic4.c`, the velocity update (Step F1) is:

$$
v_x^{n+1} = v_x^n - \text{rox} \cdot \text{FD}_\sigma(\sigma^n)
$$

$$
v_z^{n+1} = v_z^n - \text{roz} \cdot \text{FD}_\sigma(\sigma^n)
$$

where `rox` and `roz` are **pre-scaled buoyancies** defined at staggered grid positions.

From `readModel.c`, the averaging to construct rox at the Vx grid point $(ix, iz)$:

$$
\text{rox}(ix, iz) = \frac{dt}{dx} \cdot \frac{1}{2}\left[\frac{1}{\rho(ix-1, iz)} + \frac{1}{\rho(ix, iz)}\right]
$$

This is the **arithmetic average of buoyancy** ($1/\rho$) at the two neighboring P-grid points.  The Vx grid point at $(ix, iz)$ straddles the P-grid points $(ix-1, iz)$ and $(ix, iz)$ in the $x$-direction.

Similarly for roz at the Vz grid point $(ix, iz)$:

$$
\text{roz}(ix, iz) = \frac{dt}{dx} \cdot \frac{1}{2}\left[\frac{1}{\rho(ix, iz-1)} + \frac{1}{\rho(ix, iz)}\right]
$$

The Vz grid point straddles P-grid points $(ix, iz-1)$ and $(ix, iz)$ in the $z$-direction.


## 2. Sensitivity of rox to Physical $\rho$

We need the derivative of rox at a Vx point with respect to the physical density $\rho$ at a P-grid point.

For a Vx point at $(ix, iz)$, the rox depends on $\rho$ at two P-grid neighbors: $(ix-1, iz)$ and $(ix, iz)$.

Taking the derivative with respect to $\rho$ at P-grid point $(ix_0, iz_0)$:

$$
\frac{\partial\,\text{rox}(ix, iz)}{\partial\,\rho(ix_0, iz_0)} =
\begin{cases}
\displaystyle -\frac{1}{2} \cdot \frac{dt}{dx \cdot \rho(ix_0, iz_0)^2} & \text{if } ix_0 = ix-1 \text{ or } ix_0 = ix, \text{ and } iz_0 = iz \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

Defining the "local buoyancy" at P-grid:

$$
\text{rox}_P(ix_0, iz_0) = \frac{dt}{dx \cdot \rho(ix_0, iz_0)}
$$

we can write:

$$
\frac{\partial\,\text{rox}(ix)}{\partial\,\rho(ix_0)} = -\frac{\text{rox}_P(ix_0)}{2\,\rho(ix_0)}
$$


## 3. The Gradient from the Lagrangian

### Lagrangian setup

The F1 constraint at time step $n$ is:

$$
C_1^n: \quad v_x^{n+1} - v_x^n + \text{rox} \cdot \text{FD}_\sigma(\sigma^n) = 0
$$

The Lagrangian includes the term $\langle \psi_{v_x}^n,\, C_1^n \rangle$ for each time step.

### Gradient expression

The gradient of the Lagrangian with respect to $\rho$ at P-grid point $(ix_0, iz_0)$:

$$
g_\rho(ix_0, iz_0) = \sum_n \sum_{\substack{\text{Vx neighbors} \\ \text{of } ix_0}} \psi_{v_x}^n(ix, iz) \cdot \frac{\partial\,\text{rox}(ix)}{\partial\,\rho(ix_0)} \cdot \text{FD}_\sigma(ix)
\quad + \quad (\text{same for } v_z)
$$

### Substituting the velocity change

From the F1 constraint: $\text{rox} \cdot \text{FD}_\sigma = -(v_x^{n+1} - v_x^n) = -\Delta v_x$

So: $\text{FD}_\sigma = -\Delta v_x / \text{rox}$

Substituting into the gradient:

$$
\frac{\partial\,\text{rox}}{\partial\,\rho(ix_0)} \cdot \text{FD}_\sigma
= \left(-\frac{\text{rox}_P(ix_0)}{2\,\rho(ix_0)}\right) \cdot \left(\frac{-\Delta v_x}{\text{rox}}\right)
= \frac{\text{rox}_P(ix_0)}{2\,\rho(ix_0)} \cdot \frac{\Delta v_x}{\text{rox}}
$$


## 4. Simplification for Uniform (or Slowly Varying) $\rho$

For **uniform density** $\rho = \rho_0$:

$$
\text{rox}_P = \text{rox} = \frac{dt}{dx \cdot \rho_0}
$$

so:

$$
\frac{\text{rox}_P(ix_0)}{\rho(ix_0) \cdot \text{rox}(ix)} = \frac{1}{\rho_0}
$$

### Final formula (uniform $\rho$)

$$
\boxed{
g_\rho(ix_0, iz_0) = \frac{1}{\rho} \sum_n \left[
\frac{1}{2}\,\psi_{v_x}(ix_0)\,\Delta v_x(ix_0)
+ \frac{1}{2}\,\psi_{v_x}(ix_0\!+\!1)\,\Delta v_x(ix_0\!+\!1)
+ \frac{1}{2}\,\psi_{v_z}(iz_0)\,\Delta v_z(iz_0)
+ \frac{1}{2}\,\psi_{v_z}(iz_0\!+\!1)\,\Delta v_z(iz_0\!+\!1)
\right]
}
$$

where:

- $\Delta v_x = v_x^{n+1} - v_x^n$ is the forward velocity change at each time step
- The $\frac{1}{2}$ weights come from the **scatter** (each Vx point straddles two P-grid points)
- The $\frac{1}{\rho}$ comes from the **chain rule** $\partial\text{rox}/\partial\rho$
- $\psi_{v_x}$, $\psi_{v_z}$ are the adjoint velocity wavefields


### General formula (non-uniform $\rho$)

For non-uniform density, the $1/\rho$ factor is evaluated at the **target P-grid point**:

$$
g_\rho(ix_0) \mathrel{+}= \frac{1}{2} \cdot \frac{\text{rox}_P(ix_0)}{\rho(ix_0) \cdot \text{rox}(ix)} \cdot \psi_{v_x}(ix) \cdot \Delta v_x(ix)
$$

for each Vx neighbor $(ix)$ of P-point $(ix_0)$.


## 5. Comparison: Current Code vs Correct Formula

### Current code (`fwi_gradient.c`) — WRONG

```c
dvx_dt = (fwd_vx[ix*n1+iz] - fwd_vx_prev[ix*n1+iz]) * sdt;
vx_contrib = dt * wfl_adj->vx[ix*n1+iz] * dvx_dt;    // = ψ_vx · Δvx
grad_rho[(ix-1)*n1+iz] += 0.5f * vx_contrib;           // scatter
grad_rho[ix*n1+iz]     += 0.5f * vx_contrib;           // scatter
```

This computes:

$$
g_\rho^{\text{code}} = \sum_n \text{scatter}\left(\tfrac{1}{2}\,\psi_v \cdot \Delta v\right)
$$

**Missing: the $1/\rho$ factor.**

### Correct code

```c
dvx_dt = (fwd_vx[ix*n1+iz] - fwd_vx_prev[ix*n1+iz]) * sdt;
vx_contrib = dt * wfl_adj->vx[ix*n1+iz] * dvx_dt;
// 1/ρ at each P-grid neighbor
float inv_rho_left  = 1.0f / rho_P[(ix-1)*n1+iz];
float inv_rho_right = 1.0f / rho_P[ix*n1+iz];
grad_rho[(ix-1)*n1+iz] += 0.5f * inv_rho_left  * vx_contrib;
grad_rho[ix*n1+iz]     += 0.5f * inv_rho_right * vx_contrib;
```


## 6. Numerical Verification

From the diagnostic output of the Claerbout dot product test with $\rho = 1800$:

| Component | Current code | With $1/\rho$ correction |
|-----------|-------------|-------------------------|
| $g_\lambda \cdot \delta\lambda$ | $-5.77 \times 10^{-7}$ | (unchanged) |
| $g_\mu \cdot \delta\mu$ | $-1.63 \times 10^{-6}$ | (unchanged) |
| $g_\rho \cdot \delta\rho$ | $+1.034 \times 10^{-3}$ | $\approx 5.7 \times 10^{-7}$ |
| **Total** | dominates, ratio $\gg 1$ | all terms comparable |

After the $1/\rho$ correction, the density contribution becomes $O(10^{-7})$, comparable to the $\lambda$ and $\mu$ contributions — which is physically expected since all three sensitivities should be of similar magnitude for a well-conditioned problem.


## 7. Physical Interpretation

The factor $1/\rho$ arises because the forward equation uses **buoyancy** $b = 1/\rho$, not density directly:

$$
\frac{\partial v}{\partial t} = b \cdot \nabla \cdot \sigma
$$

The sensitivity of buoyancy to density is:

$$
\frac{\partial b}{\partial \rho} = -\frac{1}{\rho^2}
$$

Combined with the averaging (factor $1/2$) and the definition $\text{rox} = (dt/dx) \cdot b_{\text{avg}}$, the net effect is a $1/\rho$ scaling on the gradient at each P-grid point.
