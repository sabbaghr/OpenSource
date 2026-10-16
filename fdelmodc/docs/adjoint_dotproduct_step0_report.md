# Step 0 Report — Elastic Adjoint Dot-Product Test (fdelmodc + SU only)

Scope: read-only code/doc analysis plus a small empirical SU/fdelmodc check (no source
changes, no Steps 1-4 implementation). All line numbers refer to the working tree at
`/Users/rsabbagh/OpenSource/fdelmodc` as of 2026-09-29.

**Build note.** `git status` shows uncommitted changes to `fdelmodc.c`, `fdelmodc.h`,
`getParameters.c`, `writeSnapTimes.c`, `Makefile` (all for the unrelated wavefield-reconstruction
/ PML work — `wfl_recon`, `wfl_tshift`, CPML). The routines this test actually depends on —
`applySource.c`, `defineSource.c`, `elastic4.c`, `getRecTimes.c`, `recvPar.c`, `readModel.c` —
are **not** in that modified set, so they are at their last-committed state. The empirical checks
below used the already-built binary `/Users/rsabbagh/OpenSource/bin/fdelmodc` (built 2026-09-16,
i.e. predates the dirty files above). Since the dirty files gate new/unrelated features behind
`mod.wfl_recon` (default 0), this does not affect the standard `src_type`/`rec_type` path used
here — but flagging it since it means the *currently built* PML additions are not what Steps 1-4
would run against; rebuild only if you want the PML sub-case in Step 4 and even then, see §6.

---

## 1. Source injection (per src_type): exact scale factor S_c

`src_type` is a single scalar for the whole run (`getParameters.c:233`), so **one run cannot mix
src_types** — confirms the task's assumption for item 4: run once per type and sum by linearity.

Mapping (`fdelmodc.c:138`): `1=P 2=Txz 3=Tzz 4=Txx 5=S-pot 6=Fx 7=Fz 8=P-pot 9=double-couple
10=Fz-by-P 11=moment-tensor 12=Fa 13=S-Zibi`.

### 1.1 Universal pre-scale (applies to every src_type, before any type-specific code)

`applySource.c:148`:
```c
src_ampl *= (1.0/mod.dx) * l2m[ix*n1+iz];
```
`l2m[node]` is **not** the physical modulus; it is the code array built in `readModel.c:248`
(`l2m[...] = fac*lamda2mu`, `fac = dt/dx`, `readModel.c:51`), i.e.
`l2m_arr = (dt/dx)·(λ+2μ)` evaluated *at that exact grid node* (no averaging — `readModel.c:241,248`).
So after this line, for every type, `src_ampl = ampl_file · dt·(λ+2μ)/dx²` at the node.

### 1.2 Type-specific code and net S_c

| src_type | file:line | code | net array increment (per unit file amplitude) | S_c |
|---|---|---|---|---|
| 4 Txx | `applySource.c:279-281` | `txx += src_ampl` | `ampl_file·dt·(λ+2μ)/dx²` | `dt·(λ+2μ)/dx²` |
| 3 Tzz | `applySource.c:275-277` | `tzz += src_ampl` | same | `dt·(λ+2μ)/dx²` |
| 2 Txz | `applySource.c:249-257` | `txz += src_ampl` (away from a top free surface) | same | `dt·(λ+2μ)/dx²` |
| 1 P, orient=1 (monopole) | `applySource.c:214-219` | `txx += src_ampl; tzz += src_ampl` (**same** value added to both) | same, into **both** components | `dt·(λ+2μ)/dx²` per component |
| 6 Fx | `applySource.c:158-162` | `vx += 0.5·src_ampl·rox/l2m` | `0.5·ampl_file·rox_arr/dx = 0.5·ampl_file·dt/(dx²ρ)` | `0.5·dt/(dx²ρ)` |
| 7 Fz | `applySource.c:163-169` | `vz += src_ampl·roz/l2m` | `ampl_file·dt/(dx²ρ)` | `dt/(dx²ρ)` |

`rox_arr = fac/bx = dt/(dx·ρ_x-avg)` (`readModel.c:246`), `roz_arr` analogous — i.e. `rox`, `roz`
already carry `dt/dx` and `1/ρ`, matching the sign convention below.

**Two findings that must be verified/handled before Steps 1-4, both confirmed by direct code
reading, not speculation:**

- **Txz is scaled with `l2m` (λ+2μ), not `mul` (μ).** `applySource.c:148` is unconditional, and
  the Txz branch (`:256`) just does `txz += src_ampl` with that same `src_ampl`. So the
  file-amplitude→array-increment conversion factor for a Txz source uses **λ0+2μ0 = 1.8e10**,
  *not* μ0 = 4.5e9. This is almost certainly not what a naive reading of the physics would
  predict, and the diagnostic matrix (Step 3, block "τ only / v only") will show a constant
  `b/a` offset of `(λ0+2μ0)/μ0 = 4` if this is missed.
- **Fx carries an extra explicit factor 0.5 that Fz does not** (`applySource.c:159` vs `:164`).
  This is in the current code as-is (comment references a "stable implementation" alternative
  that isn't taken); I did not find a corresponding compensating factor elsewhere. Recommend
  treating this as an open item to be caught empirically by the v-only/v-only diagnostic block
  (a Fx-sourced test should show `b/a` off by a clean factor of 2 if the pre-scaling is built
  from the naive symmetric assumption).

### 1.3 `src_injectionrate` — required, easy to miss

`defineSource.c:162-170`: unless `src.injectionrate==1`, the wavelet read from `file_src` is
**pre-integrated** in the frequency domain by `-1/(jω)` before being handed to `applySource`.
`getParameters.c:743` defaults `src_injectionrate=0`; it is auto-forced to 1 only for
`src_type=6,7,12` (`getParameters.c:998-999`). **For this test, `src_injectionrate=1` must be set
explicitly for every src_type used (1,2,3,4 as well as 6,7)**, or the stress-type sources will
inject the time-integral of the intended wavelet instead of the wavelet itself, breaking the
direct correspondence between `ampl_file(t)` and `s_τ(t)`/`s_v(t)` that the S_c table above
assumes.

### 1.4 Phase-1 literal numbers (ρ0=2000, Vp0=3000, Vs0=1500 ⇒ μ0=4.5e9, λ0=9.0e9, λ0+2μ0=1.8e10)

With `dx=5` (from the task spec) these reduce to (dt still TBD in Step 1 — note only the
velocity-source scales depend on dt, the stress ones don't once expressed per unit *physical*
forcing, see §2 below):

- `S_c(Txx) = S_c(Tzz) = S_c(Txz) = S_c(P, each component) = dt·1.8e10/25`
- `S_c(Fz) = dt/(25·2000) = dt/50000`
- `S_c(Fx) = 0.5·dt/50000 = dt/100000`

---

## 2. Adjoint source pre-scaling (`1/S_c`, in terms of `sugain scale=`)

The recorded field `y` at a receiver is itself a physical quantity (stress/velocity), and
`applySource` adds `ampl_file · S_c` to the array once per timestep, which — since `S_c` already
carries one factor of `dt` — represents exactly `dt · s_physical` where `s_physical =
ampl_file · S_c/dt`. Solving for the file amplitude needed to realize a desired physical forcing
`s̃` (i.e. the components of `M^{-1}y` from the task spec):

- `ampl_file(Fz) = s̃_vz · dx²·ρ0`
- `ampl_file(Fx) = 2 · s̃_vx · dx²·ρ0` (compensates the 0.5 in §1.2)
- `ampl_file(Txx) = s̃_txx · dx²/(λ0+2μ0)`
- `ampl_file(Tzz) = s̃_tzz · dx²/(λ0+2μ0)`
- `ampl_file(Txz) = s̃_txz · dx²/(λ0+2μ0)` — **uses λ0+2μ0, not μ0**, per the §1.2 finding.

Note `dt` cancels entirely — the pre-scale factors only need `dx` and the phase-1 moduli, so they
can be hardcoded as literal `sugain scale=` values without knowing `dt` in advance. With `dx=5`,
`dx²=25`, `λ0+2μ0=1.8e10`, `ρ0=2000`:

- `scale(Fz-adjoint-source) = 25·2000 = 50000`
- `scale(Fx-adjoint-source) = 100000`
- `scale(Txx/Tzz/Txz-adjoint-source) = 25/1.8e10 = 1.388889e-9`

(all before the `M^{-1}` step itself, which is plain `sugain scale=` / `suop2 op=sum` combinations
using `1/ρ0=5e-4`, and the `C^{-1}` numbers below.)

`C^{-1}` numeric blocks (matches the task spec exactly):
`det = 4·μ0·(λ0+μ0) = 4·4.5e9·1.35e10 = 2.43e20`
`normal block = 1/det·[[λ0+2μ0, −λ0],[−λ0, λ0+2μ0]] = [[7.407407e-11, −3.703704e-11],[−3.703704e-11, 7.407407e-11]]`
`shear = 1/μ0 = 2.222222e-10`

---

## 3. Receiver extraction (per rec_type)

Default path (`rec.int_p=rec.int_vx=rec.int_vz=0`, `getRecTimes.c:161-329`, "else read values
directly from grid points"): every component is read straight off its own native staggered node
of the reference cell `(ix,iz) = (rec.x[irec]+ibndx, rec.z[irec]+ibndz)` with **no interpolation**:

- `vx` → `vx[(ix+1)*n1+iz]` (`:327`) — one cell to the right of the reference T-node
- `vz` → `vz[ix*n1+(iz+1)]` (`:291`) — one cell below
- `txx`, `tzz` → `txx[ix*n1+iz]`, `tzz[ix*n1+iz]` (`:227-228`) — at the reference T-node itself
- `txz` → `txz[(ix+1)*n1+(iz+1)]` (`:244`)
- `p` → `tzz[ix*n1+iz]` directly when `rec.int_p==0` (`:224`) — not used in Phase 1.

This matches source injection exactly: `applySource` places Fx/Fz/Txx/Tzz/Txz on the same
staggered offsets relative to `(ixsrc,izsrc)` that `getRecTimes` reads relative to
`(rec.x,rec.z)`, so placing a source and a receiver at the *same* `(x,z)` request puts every
component pair on the same physical node — required for injection/extraction to be exact
transposes. Use `rec_int_vx=0 rec_int_vz=0 rec_int_p=0` explicitly (all default 0, but the task
asks it be stated).

`rec_delay`, `rec_int_*=1/2/3` all invoke interpolation across time/space and must be avoided
for Phase 1.

---

## 4. Time staggering, source→sample index map, and time reversal

### 4.1 Where v and τ live relative to the forward loop index `it`

`elastic4.c:82-153`, called once per `it` from the main loop (`fdelmodc.c:697`):
1. `vx`, `vz` updated first (`:85,98`) using the previous τ.
2. Force source (`src.type>5`) added directly to `vx`/`vz` (`:107-109`) — same call as (1).
3. `txx/tzz` then `txz` updated (`:122-148`) using the just-updated v.
4. Stress source (`src.type<6`) added directly (`:151-153`).

With `vx=vz=0` at `it=-1` (i.e. continuous time `t=-dt/2`) and `txx=tzz=txz=0` at continuous
time `t=0`, standard staggered-leapfrog bookkeeping gives, **after** the call for loop index `it`:
`v` holds `v^{it+1/2}` (continuous time `(it+1/2)·dt`), `τ` holds `τ^{it+1}` (continuous time
`(it+1)·dt`).

`getRecTimes` is called once per `it` right after this (`fdelmodc.c:772`), so it reads exactly
these values. The sample index used is (`fdelmodc.c:769`, with `rec.delay=0`, `mod.t0=0`,
`rec.skipdt=1`, `itwritten=0` — i.e. `dtrcv=dt`, no decimation, single output file, as needed for
Phase 1): `isam = it + 1`.

**Net time map (empirically confirmed exact, see §4.3, and matching the code above):**
- `rec_vx[isam]`, `rec_vz[isam]` ↔ `v^{isam − 1/2}` (continuous time `(isam−1/2)·dt`)
- `rec_txx[isam]`, `rec_tzz[isam]`, `rec_txz[isam]` ↔ `τ^{isam}` (continuous time `isam·dt`)

I.e. **for the same output sample index, the recorded velocity components are half a sample
*earlier* than the recorded stress components.** This must be compensated when building the
adjoint source / re-aligning after time-reversal (exactly the kind of half-step issue the
project's existing `wfl_tshift` flag was built to correct for the boundary-injection
reconstruction path — see `fdelmodc.h:77` — good independent confirmation the issue is real in
this codebase, though that flag lives in a different code path (`injectBoundaryBwd.c`) and is not
directly reusable here).

On the source side, with `tbeg=0`, `t0=0`, and the source SU file built at exactly `mod.dt`
(no FFT resampling — `defineSource.c` only resamples if `wav.dt != mod.dt`): `applySource.c:106-120`
gives `id1=floor(itime)=itime` exactly, i.e. **source file sample `j` (0-indexed) is injected
during forward-loop iteration `it=j`**, landing on `v^{j+1/2}` if it's a force type or `τ^{j+1}`
if it's a stress type (by the same reasoning as above, since the source add happens inside the
same call).

### 4.2 Practical consequence for the reversed-injection / time-reversed-output step

Because τ lives on the plain integer-`dt` grid, reversing the recorded τ arrays with a pure
sample-index flip is exact and requires no additional shift. Because v is offset by `−dt/2`
relative to τ's grid, the same pure flip of a v-trace reverses it about the *wrong* center by
one half sample relative to τ — the reversed v needs a one-sample realignment relative to the
reversed τ to restore correct relative timing (analogous to what `wfl_tshift` does in the other
code path). This is exactly what Step 3's `eps ~ 1e-3` / "half-step time map" diagnostic is
designed to catch; I have not tried to derive the sign/direction of the needed one-sample shift
from first principles beyond this — it should be pinned down empirically in Step 3 against the
`v-only/v-only` vs `τ-only/τ-only` blocks, which will isolate it cleanly.

### 4.3 SU time-reversal command — verified

`suflip flip=3` ("flip top-to-bottom") is CWP's per-trace time-reversal (source:
`suflip.c` self-doc). I verified it does an **exact, shift-free, zero-based reversal**
`reversed[j] = original[N-1-j]` on a 60-sample trace built with `suspike`:
```
suspike nt=60 ntr=1 dt=0.0005 nspk=1 ix1=1 it1=11 offset=0 > spike_src.su
suflip < spike_src.su flip=3 > spike_src_rev.su
```
Spike at 1-indexed sample 11 → after flip, 1-indexed sample 50 (`= 60+1-11`), confirmed by
`suascii` inspection. This is the tool to use for the "time-reverse" step everywhere in Steps 1-4;
any one-sample correction from §4.2 should be layered on top of it (e.g. via `sushw`+a
single-sample pad/trim, to be worked out in Step 1) — no other SU program name qualifies as
"the" time-reversal tool (there is no `sureverse` in this SU install).

---

## 5. Multiple sources / arbitrary points

- Arbitrary point positions for sources: `xsrca=`/`zsrca=` (parsed `getParameters.c:663-687,
  862-863`), analogous to receivers' `xrcva=`/`zrcva=` (`recvPar.c:394-410`). Both bypass the
  linear-array logic and place points at exactly the given `(x,z)`, snapped to the nearest grid
  node via `NINT((x-sub_x0)/dx)` (`recvPar.c:403-404`, `getParameters.c:862-863`) — i.e. **nearest
  node, not interpolated**, matching the Phase 1 requirement.
- Different simultaneous wavelets per source: `src_multiwav=1` with `file_src` holding one trace
  per `xsrca` position (`applySource.c:93-96,118-120`: with `src.multiwav`, position comes from
  `src.x[isrc]/src.z[isrc]` and the wavelet from `src_nwav[isrc]` — a distinct trace per source).
- One run = one `src_type` (see §1). Mixing types requires separate runs summed by linearity
  (`suop2 op=sum`), as assumed in the task.
- Receivers: linear arrays, `xrcva=/zrcva=`, a text file (`rcv_txt=`), or a circle/ellipse
  (`recvPar.c`); output goes to `<file_rcv>_r{vx,vz,txx,tzz,txz,...}.su` (`writeRec.c:118-123`),
  one SU file per component, one trace per receiver in `rec.x`/`rec.z` order.

---

## 6. Boundary options

Codes (`getParameters.c:399`): `1=free 2=pml 3=rigid 4=taper`, set independently per side via
`top=`, `left=`, `right=`, `bottom=`. Rigid: `boundaries.c:62,77,93,109` (zeroes velocities at the
edge). Free surface: `boundaries.c:48,1781,1832` (mirrors stresses to enforce zero-traction).
Taper: cosine taper table built in `getParameters.c:422-462` and applied to the field inside the
`npml`/`ntaper`-wide edge strip.

**Correction (previous version of this report was wrong here):** elastic CFS-CPML **is**
implemented. `boundaries.c:354-480` (full `vx`/`vz` CPML update, `mod.ischeme==3`) and
`boundaries.c:1542-1600+` (the matching `txx`/`tzz` direct stress update, same guard) are separate
from the acoustic branches at `:229` and `:1403`; my first pass grepped/scanned only up to the
acoustic block and missed these. Both are already-committed code — `git diff` on `boundaries.c`
is empty; they were added in `ad7d4a2` ("optimizing PML for acoustic scheme, elastic todo"),
which is *not* among the files with uncommitted changes. `demo/fdelmodc_pmlnewel.scr` runs
exactly this path (`ischeme=3`, `left=right=top=bottom=2`).

I ran a quick empirical spot-check rather than take this on faith: 151×151 elastic grid
(dx=2, cp0=2000, cs0=1500, ro0=2000), a spike Txx-ish (`src_type=1`) source at the center, a
`vx` receiver 60 m from the left edge, comparing `left=3` (rigid) against `left=2`
(PML, `npml=20 R=1e-5 m=2.0`) with every other boundary held at `3` (rigid) in both runs so only
the left wall differs. Both runs complete cleanly (no NaN/Inf). In the reflected-arrival time
window, the PML run's peak amplitude was **roughly 3-4× smaller** than the rigid run's — real,
measurable attenuation, not a no-op — but well short of the near-total (~R=1e-5) suppression a
fully tuned CPML would ideally give. That could be a genuine tuning/completeness gap (consistent
with "elastic todo" in the commit message) or simply this test geometry's grazing-incidence
content, which is known to degrade any CPML; I have not done the systematic incidence-angle /
`npml`-sweep testing needed to tell those apart, and did not try to find or fix a cause since that
would mean editing `boundaries.c`, out of scope for this dot-product task per your instructions.

**Decision (2026-09-29):** include PML in Step 4 as a fourth boundary configuration, run as-is
(no tuning/fixing of `boundaries.c` first). Its `eps` will be reported alongside rigid/free+rigid/
free+taper, with the caveat above noted in the interpretation — a worse `eps` for the PML row is
not automatically evidence of an adjoint-construction bug, since this PML implementation's own
absorption already measured weaker than ideal in the §6 spot-check.

---

## 7. Staggered-grid averaging of ρ, λ, μ — patch size needed for exact phase-1 constants

From `readModel.c:221-338` (elastic branch), all computed **before** boundary extension:

- `l2m[T-node] = fac·(λ+2μ)`, `lam[T-node] = fac·λ` — computed **directly from `cp,cs,ro` at that
  single base-grid node** (`:240-249`), no spatial averaging at all.
- `rox[Vx-node] = fac/[0.5·(ρ(ix)+ρ(ix+1))]` (`:244,246`) — needs the **two horizontally
  adjacent** base-grid points uniform.
- `roz[Vz-node] = fac/[0.5·(ρ(ix,iz)+ρ(ix,iz+1))]` (`:245,247`) — needs the **two vertically
  adjacent** points uniform.
- `mul[Txz-node] = fac·[harmonic mean of μ at the 4 surrounding corners]` (`:312-320`,
  `mul = 4/(Σ 1/(cs²ρ) over the 4 corners)`) — needs a full **2×2 corner block** uniform.

Since a `getRecTimes`/`applySource` "point" `(ix,iz)` actually touches T at `(ix,iz)`, Vx at
`(ix+1,iz)`, Vz at `(ix,iz+1)`, Txz at `(ix+1,iz+1)` (§3), the full set of base-grid points
touched by any one of the five components at one nominal point spans `(ix-1..ix+1, iz-1..iz+1)` —
a **3×3 patch of `(ρ0,Vp0,Vs0)` centered on the point is sufficient** to make every one of
`rox,roz,l2m,lam,mul` exactly the phase-1 constant value at every node used by that point (a 5×5
patch, as the task allows, adds a safety margin against the 4th-order spatial stencils used
elsewhere in `elastic4.c` reaching `±2` cells, which matters for wave-propagation accuracy away
from the point but not for the exactness of the injection/extraction operators themselves).

---

## 8. SU recipes verified

All of the following were run against the actual SU install at `/Users/rsabbagh/cwp/bin` (no
Python/awk/bc used):

**Model + homogeneous background (stand-in for "smooth heterogeneous + patches", trivial case):**
```
makemod file_base=mod.su cp0=3000 cs0=1500 ro0=2000 sizex=500 sizez=500 dx=5 dz=5 orig=0,0
```
produced `mod_cp.su, mod_cs.su, mod_ro.su` correctly (`makemod` self-doc confirmed via
`utils/makemod.c`).

**Spike source + fdelmodc run** (`src_type=4`, i.e. Txx, at (250,250), receiver via
`xrcva=300 zrcva=250`, `left=right=top=bottom=3` (rigid), `rec_int_*=0`, `dtrcv=dt` to disable
decimation):
```
suspike nt=60 ntr=1 dt=0.0005 nspk=1 ix1=1 it1=11 offset=0 > spike_src.su
fdelmodc file_cp=mod_cp.su file_cs=mod_cs.su file_den=mod_ro.su file_src=spike_src.su \
  file_rcv=rec_txx4 ischeme=3 dt=0.0005 tmod=0.03 dtrcv=0.0005 \
  src_type=4 src_orient=1 xsrc=250 zsrc=250 xrcva=300 zrcva=250 \
  rec_type_p=0 rec_type_vx=1 rec_type_vz=1 rec_type_txx=1 rec_type_tzz=1 rec_type_txz=1 \
  rec_int_vx=0 rec_int_vz=0 rec_int_p=0 left=3 right=3 top=3 bottom=3
```
ran without error and produced `rec_txx4_r{vx,vz,txx,tzz,txz}.su`, confirming the parameter names
above are all accepted and wired as documented. **Caveat:** a single-sample spike is far from
"band-limited well below grid dispersion" (Step 1's actual requirement) — the recorded traces
showed smooth-looking energy ramping up several samples *before* the naive geometric P-arrival
time (50 m / 3000 m/s ≈ 33 samples after the spike), which is numerical-dispersion precursor
noise from the unresolved high-frequency content of a true delta function, not a sign of an
indexing bug. This is exactly why Step 1 specifies band-limited random signals; the spike here
was used only to pin down discrete index bookkeeping (§4) and SU-tool mechanics (below), not
propagation-time accuracy.

**Time reversal:** `suflip flip=3` — verified exact index map, §4.3.

**Scaling:** `sugain scale=<value>` — verified exact linear scaling (`X → 2.5·X`, every sample,
every trace) on a small synthetic file.

**Dot product recipe — verified end-to-end on a hand-checkable tiny case:**
Built `X = [[0,1,0,0],[0,1,0,0]]`, `Y = [[0,1,0,0],[0,0,0,0]]` (2 traces × 4 samples each) with
`suspike`. Hand value: `Σ X·Y = 1`.
```
suop2 X.su Y.su op=prod > prod.su
sushw < prod.su key=cdp a=1 > prod_h.su
sustack < prod_h.su key=cdp normpow=0 > stack1.su      # sum over traces (receivers)
suflip < stack1.su flip=0 > tr.su                       # transpose: samples -> traces
sushw < tr.su key=cdp a=1 > tr_h.su
sustack < tr_h.su key=cdp normpow=0 > total.su          # sum over (former) samples
suascii < total.su bare=1                               # -> 1.00000000
```
Reproduced the hand value exactly. **`normpow=0` is required** on both `sustack` calls — the
default `normpow=1` divides by the count of nonzero-valued inputs, which silently turns the sum
into something else whenever not every trace/sample is nonzero. `suflip flip=0` (transpose) was
confirmed to carry the `cdp` header through, so the second `sushw` is defensive/optional but
cheap insurance. `transp` is the other CWP tool named for this and should work identically; I
used `suflip flip=0` since it was already in hand from the time-reversal check.

---

## Summary of open items to resolve in Step 1

1. Fix `dx` (task says 5 m) and choose a stable `dt` for the actual smooth-heterogeneous model
   (Courant number using the model's `cp_max`).
2. Set `src_injectionrate=1` explicitly for **every** src_type used (§1.3) — easy to forget for
   1/2/3/4, since it's only auto-set for 6/7/12.
3. Use `l2m0=λ0+2μ0=1.8e10` (not `μ0`) when pre-scaling a Txz adjoint source (§1.2).
4. Carry the extra ×2 factor for Fx relative to Fz through the pre-scaling, and treat it as a
   specific, testable hypothesis in the Step 3 diagnostic matrix rather than an assumed constant
   (§1.2).
5. Work out and test the one-sample v/τ realignment after `suflip flip=3` (§4.2) — expect this to
   show up as the `eps~1e-3` "half-step" signature in Step 3 if missed.
6. Elastic PML exists (§6, corrected) and can be included in Step 4, but its absorption looked
   weaker than expected (~3-4x vs rigid, not ~1e-5) in a quick spot-check — decide up front
   whether a PML `eps` failure in Step 4 should be read as an adjoint bug or a PML-tuning issue.
