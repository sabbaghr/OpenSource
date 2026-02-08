# Wave Operator Adjoint Dot Product Test

## 1. What and Why

The wave operator adjoint dot product test verifies the fundamental identity:

```
<Ax, y> = <x, A^T y>
```

where:
- **A** is the elastic wave propagation operator (maps source wavelet to receiver data)
- **A^T** is the numerical adjoint (transpose) of A
- **x** is a random source wavelet
- **y** is random receiver data
- **Ax** = forward-propagated data recorded at receivers
- **A^T y** = adjoint wavefield extracted at the source position

This is a more fundamental test than the model-parameter Jacobian dot product test
(which verifies the gradient computation). Here we test that the *wave equation solver
itself* (`elastic4.c`) and its adjoint (`elastic4_adj.c`) correctly implement the
transpose relationship. If this test fails, the FWI gradient cannot be correct.

The test uses random vectors x and y to avoid accidental cancellations.
A passing test (epsilon < 1e-02) confirms adjointness to within the expected
tolerance set by absorbing boundary layers (which are not exactly self-adjoint).


## 2. Operator Decomposition

The full wave operator A decomposes as:

```
A = R . P . S
```

| Symbol | Name | Role |
|--------|------|------|
| **S** | Source injection | Injects wavelet x into the wavefield at the source position |
| **P** | Wave propagation | Advances the wavefield through all time steps |
| **R** | Receiver extraction | Extracts data d = Ax at receiver positions |

The adjoint reverses the chain:

```
A^T = S^T . P^T . R^T
```

| Symbol | Name | Role |
|--------|------|------|
| **R^T** | Adjoint receiver | Injects random data y into the wavefield at receiver positions |
| **P^T** | Adjoint propagation | Propagates the wavefield backward in time |
| **S^T** | Adjoint source | Extracts the adjoint wavefield at the source position |

In the code:
- **S** = `applySource()` in `elastic4()`
- **P** = the FD stencil loop in `elastic4()`
- **R** = manual wavefield read at `getRecTimes` grid positions
- **R^T** = `applyAdjointSource()` in `elastic4_adj()`
- **P^T** = the FD stencil loop in `elastic4_adj()` (same stencils!)
- **S^T** = manual wavefield read at source grid position


## 3. The Metivier & Brossier (2024) Result

Equation (90) in Metivier & Brossier (2024, Geophysical Journal International)
establishes a key property of the elastic adjoint on staggered grids:

> The same forward FD code can be used to compute a solution of the adjoint
> equations, up to a posterior change of sign to obtain the adjoint stress field.

Mathematically, if `elastic4` computes the forward and `elastic4_adj` uses
**identical FD stencils**, then:

| Field | Relationship | Meaning |
|-------|-------------|---------|
| Velocity (vx, vz) | lambda_v_1 = lambda_v_2 | Output velocity = correct adjoint velocity |
| Stress (txx, tzz, txz) | lambda_sigma_1 = -lambda_sigma_2 | Output stress = **negative** of true adjoint stress |

**Consequence for source extraction**: When computing A^T y at the source position:
- For velocity source types (Fz, Fx): extract directly from the adjoint velocity field
- For stress source types (explosive): extract from the adjoint stress and **negate**


## 4. Three Gotchas (Why the Naive Implementation Fails)

The initial test implementation used `fdfwimodc` for forward recording and
`readResidual` for adjoint source setup, producing a ratio of 23.2 (should be 1.0).
Three bugs were identified:

### 4.1 Time-Sample Mapping Mismatch

`fdfwimodc` records using a complex time mapping:
```c
isam = (it + NINT(mod.t0/mod.dt)) / skipdt + 1   // fdfwimodc
```

But `applyAdjointSource` reads with a simpler mapping:
```c
isam = (itime - rec_delay) / rec_skipdt            // applyAdjointSource
```

With `t0 = -0.10` and `dt = 0.001`, these differ by `NINT(t0/dt) + 1 = -99`
samples, causing the adjoint to inject the wrong data at each time step.

**Fix**: Bypass `fdfwimodc`. Record manually from the wavefield arrays at every
time step, and set `rec_delay=0`, `rec_skipdt=1` for the adjoint.


### 4.2 Grid Position Mismatch (Vz Stagger)

`getRecTimes` records Vz with a +1 stagger shift:
```c
rec_vz[...] += vz[ix*n1 + iz+1]   // iz+1 for Vz stagger
```

But `readResidual` sets adjoint positions as:
```c
adj->zi = mod->ioZz + grid_z       // MISSING the +1 shift!
```

This injects the adjoint source one grid point away from the recording position,
breaking the transpose relationship.

**Fix**: Build `adjSrcPar` manually with `zi = ibndz + rec.z[ir] + 1` to exactly
match `getRecTimes`.


### 4.3 Stress Sign (Metivier Eq. 90)

For explosive sources (`src_type=1`), the forward injects into `txx` and `tzz`.
The adjoint extraction must account for the stress sign flip from Eq. (90):

```c
// WRONG: alpha = +l2m/dx
// RIGHT: alpha = -l2m/dx  (negate for stress sign flip)
```

**Fix**: Set `alpha = -l2m[src_ig] / dx` for explosive source extraction.


## 5. Solution: Direct Wavefield Access

The working test bypasses `fdfwimodc` and `readResidual` entirely:

1. **Forward**: Call `elastic4()` directly for each time step. Record from the
   wavefield arrays at the exact grid positions used by `getRecTimes`.

2. **Adjoint source setup**: Build `adjSrcPar` manually with:
   - Grid positions matching the forward recording positions exactly
   - Correct adjoint injection types (Fz for Vz, Fx for Vx, etc.)
   - Sign compensation for stress types (applyAdjointSource uses `-=`)

3. **Adjoint propagation**: Call `elastic4_adj()` with `rec_delay=0`,
   `rec_skipdt=1` for clean 1:1 time mapping.

4. **Source extraction**: Read the adjoint wavefield at the source position,
   scale by alpha (the adjoint of the source injection scaling).

5. **RHS computation**: Compute `<x, A^T y>` using the same wavelet
   interpolation formula as `applySource` (line 111) to match exactly.


## 6. Recording Component Table

The `rec_comp` parameter selects which wavefield component to record in the
forward and inject in the adjoint:

| rec_comp | Forward field | Grid position (relative to P grid) | Adj type | Adj wav sign |
|----------|--------------|-------------------------------------|----------|-------------|
| `vz` | `vz[ix*n1 + iz+1]` | (ibndx+rx, ibndz+rz+1) | 7 (Fz) | +y |
| `vx` | `vx[(ix+1)*n1 + iz]` | (ibndx+rx+1, ibndz+rz) | 6 (Fx) | +y |
| `txx` | `txx[ix*n1 + iz]` | (ibndx+rx, ibndz+rz) | 4 (Txx) | +y |
| `tzz` | `tzz[ix*n1 + iz]` | (ibndx+rx, ibndz+rz) | 3 (Tzz) | +y |

**Why +y for ALL types (no sign flip)?** The adjoint wav data is always stored
as +y (unmodified) because `applyAdjointSource` already handles the sign
correctly for both velocity and stress:

- **Velocity types** (vz, vx): `applyAdjointSource` uses `field += src_ampl`.
  This is the direct adjoint of `d = field[pos]`.

- **Stress types** (txx, tzz): `applyAdjointSource` uses `field -= src_ampl`.
  This accounts for the Metivier sign convention: `elastic4_adj` tracks
  `lambda_tilde_sigma = -lambda_sigma`. The true adjoint needs
  `lambda_sigma += y`, i.e., `lambda_tilde_sigma -= y`, which is exactly
  what `txx -= src_ampl` achieves with `src_ampl = y`.


## 7. Source Extraction Scaling (Alpha)

The source extraction scaling alpha is the adjoint of the forward source
injection formula in `applySource.c`. The common scaling factor (line 143)
is `(1/dx) * l2m[ig]`, then each type applies a type-specific factor:

| src_type | Forward injection | Net effect | Alpha |
|----------|------------------|------------|-------|
| 7 (Fz) | `vz += src_ampl * roz/l2m` | `vz += wavelet * roz/dx` | `+roz[ig]/dx` |
| 6 (Fx) | `vx += 0.5 * src_ampl * rox/l2m` | `vx += wavelet * 0.5*rox/dx` | `+0.5*rox[ig]/dx` |
| 1 (explosive) | `txx += src_ampl; tzz += src_ampl` | `txx,tzz += wavelet * l2m/dx` | `-l2m[ig]/dx` |

The negative alpha for explosive sources accounts for the Metivier stress
sign flip (Section 3). The 0.5 factor for Fx comes from `applySource.c`
line 154.

The extraction field at the source position depends only on src_type:
- src_type=7: `z_adj[it] = alpha * adj_vz[src_ig]`
- src_type=6: `z_adj[it] = alpha * adj_vx[src_ig]`
- src_type=1: `z_adj[it] = alpha * (adj_txx[src_ig] + adj_tzz[src_ig])`


## 8. Test Combinations

The test supports any combination of `src_type` and `rec_comp`:

| Test | src_type | rec_comp | Description |
|------|----------|----------|-------------|
| 1 | 7 (Fz) | vz | Force source, velocity recording (baseline) |
| 2 | 7 (Fz) | txx | Force source, Txx stress recording |
| 3 | 7 (Fz) | tzz | Force source, Tzz stress recording |
| 4 | 6 (Fx) | vx | Fx force source, Vx recording |
| 5 | 6 (Fx) | txx | Fx force source, Txx recording |
| 6 | 6 (Fx) | tzz | Fx force source, Tzz recording |


## 9. Expected Results

The absorbing boundary layers (PML/taper) are not exactly self-adjoint,
introducing small errors. Typical epsilon values:

- Homogeneous model, all absorbing boundaries: **epsilon ~ 1e-05 to 1e-03**
- Free surface (top=1): slightly larger epsilon due to free surface implementation
- Larger models with more interior receivers: smaller epsilon

The test uses `epsilon < 1e-02` as the PASS/FAIL threshold.


## 10. Usage

```bash
# Build
cd fdelfwi && make test_wave_op_dp

# Run (Fz source + Vz recording, default)
./test_wave_op_dp \
    file_cp=model_cp.su file_cs=model_cs.su file_den=model_ro.su \
    file_src=wave.su file_rcv=syn file_snap=snap \
    ischeme=3 iorder=4 src_type=7 \
    rec_type_vz=1 dtrcv=0.001 tmod=0.5 verbose=1 \
    xrcv1=100 xrcv2=400 zrcv1=350 zrcv2=350 dxrcv=10 \
    xsrc=250 zsrc=100 ntaper=50 \
    left=4 right=4 top=4 bottom=4 \
    seed=42 rec_comp=vz

# Run with Tzz recording
./test_wave_op_dp ... src_type=7 rec_comp=tzz

# Run with Fx source and Vx recording
./test_wave_op_dp ... src_type=6 rec_comp=vx
```


## 11. References

- Metivier, L. & Brossier, R. (2024). "A review of the adjoint-state method
  for computing the gradient of a functional with geophysical applications."
  *Geophysical Journal International*, 238(2), 939-983. Eq. (90).

- Claerbout, J.F. (1992). "Earth soundings analysis: Processing versus
  inversion." Chapter on the adjoint dot product test.

- Mora, P. (1987). "Nonlinear two-dimensional elastic inversion of
  multioffset seismic data." Appendix B.
