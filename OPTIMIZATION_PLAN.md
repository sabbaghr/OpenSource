# Plan: C L-BFGS Optimizer + FWI Inversion Driver

## Context

The FWI gradient computation is working (`fdelfwi/fwi_driver.c`): forward modeling, residual, adjoint, gradient accumulation, MPI reduction. But it does a **single gradient pass** only. To run actual inversion, we need an optimization loop that iteratively updates the model using the gradient.

The SEISCOPE TOOLBOX_OPTIMIZATION (Fortran, at `/rcp3/software/codes/TOOLBOX_OPTIMIZATION/`) provides L-BFGS with Wolfe linesearch. Rather than writing a fragile C/Fortran wrapper, we'll **translate the L-BFGS to pure C** (~700 lines) and build a new inversion driver that wraps the existing gradient machinery.

**Why C translation instead of Fortran wrapper:**
- Single language → simpler debugging (GDB), no `libgfortran` dependency
- No opaque-pointer dance for Fortran allocatable arrays
- No character-string marshalling across C/Fortran boundary
- We own the code → can modify, extend, and maintain freely

## Design Decisions

- **Parameterization**: Both Lamé (λ,μ,ρ) and velocity (Vp,Vs,ρ) supported via `param=` flag
- **Algorithms**: L-BFGS (primary) + steepest descent (baseline/debug)
- **Code structure**: Optimizer library in `optimization/` (repo root), inversion driver in `fdelfwi/`
- **Bound constraints**: Included from the start (box projection)
- **Model vector**: Interior-only (nx×nz per parameter), not padded grid
- **Precision**: `float` throughout (matches existing codebase)

---

## Directory Layout

```
OpenSource_SL10/
├── optimization/              ← NEW: standalone optimizer library
│   ├── Makefile               ← builds liboptim.a
│   ├── optim.h                ← public header (struct + API)
│   ├── lbfgs.c                ← L-BFGS algorithm (C translation)
│   ├── steepest_descent.c     ← steepest descent algorithm
│   ├── optim_common.c         ← shared: linesearch, convergence, print, project
│   └── test_rosenbrock.c      ← unit test
├── fdelfwi/
│   ├── fwi_inversion.c        ← NEW: inversion driver (optimization loop)
│   ├── updateModel.c          ← NEW: model vector ↔ FD coefficients
│   ├── fwi_driver.c           ← UNCHANGED: single gradient pass
│   ├── Makefile               ← MODIFIED: add inversion targets, link liboptim.a
│   └── ...                    ← existing FWI files unchanged
```

---

## Files to Create

### 1. `optimization/optim.h` — Public optimizer header

```c
#ifndef OPTIM_H
#define OPTIM_H

/* Optimizer state flags (replace Fortran string-based FLAG) */
typedef enum { OPT_INIT, OPT_GRAD, OPT_CONV, OPT_FAIL, OPT_NSTE } optFlag;

/* Algorithm selector */
typedef enum { ALG_SD, ALG_LBFGS } optAlg;

/* Optimizer state structure (replaces Fortran optim_type) */
typedef struct {
    /* Algorithm */
    optAlg  algorithm;
    int     debug;
    int     print_flag;     /* 1=write convergence file */

    /* Dimensions */
    int     n;              /* model vector length */

    /* Iteration control */
    int     niter_max;
    int     cpt_iter;
    float   conv;           /* convergence tolerance: fcost/f0 < conv */
    float   f0;             /* initial cost */

    /* Linesearch state */
    int     first_ls;
    int     nls_max;
    int     cpt_ls;
    int     nfwd_pb;        /* forward problem counter */
    float   fk;             /* previous cost */
    float   m1, m2;         /* Wolfe parameters (default 1e-4, 0.9) */
    float   mult_factor;    /* bracket expansion factor (default 10) */
    float   alpha_L, alpha_R, alpha;
    float   q0, q;          /* directional derivatives */

    /* Work arrays (size n) */
    float  *xk;             /* saved iterate for linesearch */
    float  *grad;           /* saved gradient */
    float  *descent;        /* descent direction */

    /* L-BFGS history (sk, yk: flat n*l, column-major) */
    int     l;              /* max history pairs */
    int     cpt_lbfgs;      /* current history count */
    float  *sk;             /* step differences [n*l] */
    float  *yk;             /* gradient differences [n*l] */

    /* Bound constraints */
    int     bound;          /* 0=off, 1=on */
    float   threshold;      /* bound tolerance */
    float  *lb;             /* lower bounds [n] (user-allocated) */
    float  *ub;             /* upper bounds [n] (user-allocated) */
} optim_type;

/* Main API */
void lbfgs_run(int n, float *x, float fcost, float *grad,
               optim_type *opt, optFlag *flag);
void steepest_descent_run(int n, float *x, float fcost, float *grad,
                          optim_type *opt, optFlag *flag);
void optim_finalize(optim_type *opt);

/* Common utilities (also usable standalone) */
float optim_norm_l2(int n, const float *x);
float optim_dot(int n, const float *x, const float *y);

#endif /* OPTIM_H */
```

### 2. `optimization/optim_common.c` — Shared routines (~200 lines)

Translates these Fortran files from TOOLBOX_OPTIMIZATION/COMMON/src/:

| Fortran source | C function | Description |
|---|---|---|
| `normL2.f90` | `float optim_norm_l2(int n, const float *x)` | L2 norm |
| `scalL2.f90` | `float optim_dot(int n, const float *x, const float *y)` | dot product |
| `project.f90` | `void optim_project(int n, optim_type *opt, float *x)` | box constraint clipping |
| `std_test_conv.f90` | `int optim_test_conv(optim_type *opt, float fcost)` | convergence check |
| `print_info.f90` | `void optim_print_info(int n, const char *tag, optim_type *opt, float fcost, optFlag flag)` | write iterate_XX.dat |
| `std_linesearch.f90` | `void optim_wolfe_linesearch(int n, float *x, float fcost, float *grad, optim_type *opt)` | Wolfe linesearch |

**Internal linesearch state** (replaces Fortran `character*8 task`):
- `ls_task=0` (NEW_GRAD): need next cost/gradient evaluation
- `ls_task=1` (NEW_STEP): linesearch accepted step
- `ls_task=2` (FAILURE): linesearch failed

**Wolfe linesearch algorithm** (from `std_linesearch.f90`):
1. First call (`first_ls=true`): save `xk=x`, compute `q0=<grad,descent>`, step `x=xk+alpha*descent`
2. Subsequent calls: check Wolfe conditions
   - **Condition 1** (sufficient decrease): `fcost <= fk + m1*alpha*q0` (m1=1e-4)
   - **Condition 2** (curvature): `q >= m2*q0` where `q=<grad,descent>` (m2=0.9)
   - Both satisfied → accept step (NEW_STEP)
   - Condition 1 violated → shrink: `alpha_R=alpha`, `alpha=(alpha_L+alpha_R)/2`
   - Condition 2 violated → expand: `alpha_L=alpha`, `alpha=mult_factor*alpha` (or bisect if bracketed)
3. Max LS iterations with cost decrease → forced accept
4. Max LS iterations without decrease → FAILURE

### 3. `optimization/lbfgs.c` — L-BFGS algorithm (~300 lines)

Translates these Fortran files from TOOLBOX_OPTIMIZATION/LBFGS/kernel/src/:

| Fortran source | C function |
|---|---|
| `init_LBFGS.f90` | `static void init_lbfgs(int n, float *x, float fcost, float *grad, optim_type *opt)` |
| `descent_LBFGS.f90` | `static void descent_lbfgs(int n, optim_type *opt, float *grad)` |
| `save_LBFGS.f90` | `static void save_lbfgs(int n, optim_type *opt, float *x, float *grad)` |
| `update_LBFGS.f90` | `static void update_lbfgs(int n, optim_type *opt, float *x, float *grad)` |
| `LBFGS.f90` | `void lbfgs_run(...)` — main reverse-communication dispatcher |
| `finalize_LBFGS.f90` | `void optim_finalize(...)` |

**Key translation details:**
- Fortran 1-based → C 0-based indexing
- 2D `sk(n,l)` → flat `sk[j*n + i]` (column-major preserved; column j starts at `&sk[j*n]`)
- String FLAG → `optFlag` enum
- `descent_lbfgs`: two-loop recursion (Nocedal Algorithm 7.5) with safeguard fallback to steepest descent
- `save_lbfgs`: circular buffer with `memmove` for column shift when full

**Two-loop recursion** (`descent_lbfgs`, from Nocedal Algorithm 7.5):
```
Safeguard: if ||sk_newest|| == 0 or ||yk_newest|| == 0, use d = -grad

First loop (backward through history):
  q = grad
  for j = newest..oldest:
    rho[j] = 1 / dot(yk[:,j], sk[:,j])
    alpha[j] = rho[j] * dot(sk[:,j], q)
    q = q - alpha[j] * yk[:,j]

Scale: gamma = dot(sk_newest, yk_newest) / ||yk_newest||^2
       d = gamma * q

Second loop (forward through history):
  for j = oldest..newest:
    beta = rho[j] * dot(yk[:,j], d)
    d = d + (alpha[j] - beta) * sk[:,j]

descent = -d
```

**History management** (`save_lbfgs` + `update_lbfgs`):
- `save_lbfgs`: stores current x,grad at position `cpt_lbfgs` (or shifts all left and stores at end when full)
- `update_lbfgs`: converts stored values to differences: `sk[:,j] = x_new - x_old`, `yk[:,j] = grad_new - grad_old`
- Typical sequence per iteration: update(converts old→diff) → descent(uses diffs) → save(stores new)

**Main dispatcher control flow** (`lbfgs_run`):
```
flag=OPT_INIT → init_lbfgs → wolfe_linesearch → print_info → flag=OPT_GRAD
flag=OPT_GRAD → wolfe_linesearch
  ls_task=NEW_STEP → test_conv → if converged: flag=OPT_CONV, finalize
                                  else: update + descent + save → flag=OPT_NSTE
  ls_task=NEW_GRAD → flag=OPT_GRAD (continue linesearch)
  ls_task=FAILURE  → flag=OPT_FAIL, finalize
```

### 4. `optimization/steepest_descent.c` — SD algorithm (~80 lines)

Same reverse-communication pattern as L-BFGS but simpler:
- No history arrays (sk, yk)
- Descent direction = `-grad` at each new iteration (no two-loop recursion)
- Reuses same `wolfe_linesearch`, `test_conv`, `print_info`
- Writes `iterate_SD.dat`

### 5. `optimization/test_rosenbrock.c` — Unit test (~100 lines)

Validates both algorithms against the Rosenbrock function (same test as `TOOLBOX_OPTIMIZATION/LBFGS/test/src/test_LBFGS.f90`):
```c
/* f(x,y) = (1-x)^2 + 100*(y-x^2)^2
 * grad[0] = -2*(1-x) - 400*x*(y-x^2)
 * grad[1] = 200*(y-x^2)
 *
 * Minimum at (1,1), f=0. Start at (1.5, 1.5).
 * L-BFGS expected: ~30-50 iters with l=20.
 * SD expected: converges but much slower.
 *
 * Also tests bound constraints: bounds=[0,2]x[0,2] should not affect convergence.
 */
```

### 6. `optimization/Makefile` — Library build

```makefile
include ../Make_include

OBJS = optim_common.o lbfgs.o steepest_descent.o
LIB  = liboptim.a

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFUNCT) $(LIB) $(OBJS)

test: test_rosenbrock
	./test_rosenbrock

test_rosenbrock: test_rosenbrock.o $(LIB)
	$(CC) $(LDFLAGS) -o $@ test_rosenbrock.o $(LIB) -lm

%.o: %.c optim.h
	$(CC) $(CFLAGS) $(OPTC) -c $<

clean:
	rm -f *.o $(LIB) test_rosenbrock
```

---

## Files to Create in fdelfwi/

### 7. `fdelfwi/updateModel.c` — Model vector utilities (~250 lines)

The critical bridge between the optimizer's flat vector and the FD code's padded grid arrays.

```c
#include "fdelfwi.h"

/* Recompute FD coefficients (l2m, lam, muu, rox, roz) from mod->cp/cs/rho.
 * In-memory equivalent of readModel.c lines 215-400.
 * Handles staggered-grid averaging, edge cases, boundary extensions. */
void recomputeFDcoefficients(modPar *mod, bndPar *bnd);

/* Extract interior (nx*nz) model parameters into flat vector x[nparam*nx*nz].
 * param=1 (Lame): extracts physical [lambda, mu, rho]
 *   lambda = lam[padded] / fac, mu = (l2m[padded]-lam[padded])/(2*fac)
 * param=2 (velocity): extracts [Vp, Vs, rho] from mod->cp/cs/rho */
void extractModelVector(float *x, modPar *mod, bndPar *bnd, int param);

/* Inject flat vector x back into mod arrays and recompute FD coefficients.
 * param=1: x=[lambda,mu,rho] -> convert to Vp/Vs/rho -> update cp/cs/rho -> recompute
 * param=2: x=[Vp,Vs,rho] -> update cp/cs/rho directly -> recompute */
void injectModelVector(float *x, modPar *mod, bndPar *bnd, int param);

/* Extract gradients: stripBoundary each component, optionally apply chain rule
 * (Lame->velocity via convertGradientToVelocity if param=2),
 * concatenate into flat vector g[nparam*nx*nz] */
void extractGradientVector(float *g, float *g1, float *g2, float *g3,
                           modPar *mod, bndPar *bnd, int param);
```

**`recomputeFDcoefficients` logic** (from `readModel.c` lines 215-400):

For each interior point (ix=0..nx-2, iz=0..nz-2):
```c
fac = dt/dx;
cp2 = cp[ix*nz+iz] * cp[ix*nz+iz];
cs2 = cs[ix*nz+iz] * cs[ix*nz+iz];
mu  = cs2 * rho[ix*nz+iz];
l2m_val = cp2 * rho[ix*nz+iz];
lam_val = l2m_val - 2*mu;

/* Harmonic average for muu at Txz stagger */
cs11 = cs2*ro[ix,iz]; cs12 = cs2b*ro[ix,iz+1];
cs21 = cs2a*ro[ix+1,iz]; cs22 = cs2c*ro[ix+1,iz+1];
mul = 4.0/(1/cs11 + 1/cs12 + 1/cs21 + 1/cs22);

/* Arithmetic average for buoyancy */
bx = 0.5*(rho[ix] + rho[ix+1]);
bz = 0.5*(rho[iz] + rho[iz+1]);

/* Store with FD scaling */
l2m[padded] = fac * l2m_val;
lam[padded] = fac * lam_val;
muu[padded] = fac * mul;
rox[padded] = fac / bx;
roz[padded] = fac / bz;
```

Edge handling (ix=nx-1, iz=nz-1): same patterns as `readModel.c`.
Boundary extension: copy nearest interior value into taper/PML zones (from `readModel.c` lines 400+).

### 8. `fdelfwi/fwi_inversion.c` — Inversion driver (~500 lines)

New `main()` that wraps existing gradient computation in an optimization loop.

**Overall structure:**
```c
int main(int argc, char **argv) {
    /* ===== SETUP (reuse from fwi_driver.c lines 226-358) ===== */
    /* MPI init, parse params, getParameters, readModel, defineSource */
    /* Sinking receivers, boundary surface detection */

    /* ===== NEW: Optimizer parameters ===== */
    /* Parse: niter, conv, algorithm, lbfgs_mem, nls_max, bounds */
    int nparam = (mod.ischeme > 2) ? 3 : 2;
    int nvec = nparam * mod.nx * mod.nz;
    float *x = calloc(nvec, sizeof(float));
    float *grad_vec = calloc(nvec, sizeof(float));

    /* Extract initial model into optimizer vector */
    extractModelVector(x, &mod, &bnd, param);

    /* Configure optimizer */
    optim_type opt;
    memset(&opt, 0, sizeof(optim_type));
    opt.algorithm = algorithm;
    opt.niter_max = niter;
    opt.conv = conv;
    opt.l = lbfgs_mem;
    opt.nls_max = nls_max;
    opt.print_flag = 1;
    opt.debug = (verbose > 1);
    set_bounds(&opt, &mod, param, nvec);  /* physical bounds */

    /* ===== INITIAL GRADIENT COMPUTATION ===== */
    fcost = compute_fwi_gradient(&mod, &src, &wav, &bnd, &rec, &sna,
                                 &shot, src_nwav, &chk, param,
                                 grad1, grad2, grad3, ...);
    extractGradientVector(grad_vec, grad1, grad2, grad3, &mod, &bnd, param);

    /* ===== OPTIMIZATION LOOP ===== */
    optFlag flag = OPT_INIT;
    void (*opt_run)(int, float*, float, float*, optim_type*, optFlag*);
    opt_run = (algorithm == ALG_LBFGS) ? lbfgs_run : steepest_descent_run;

    while (flag != OPT_CONV && flag != OPT_FAIL) {
        opt_run(nvec, x, fcost, grad_vec, &opt, &flag);

        if (flag == OPT_GRAD) {
            /* Inject updated model into FD arrays (rank 0 only) */
            if (mpi_rank == 0)
                injectModelVector(x, &mod, &bnd, param);

#ifdef USE_MPI
            /* Broadcast 5 FD coefficient arrays to all ranks */
            MPI_Bcast(mod.l2m, sizem, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(mod.lam, sizem, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(mod.muu, sizem, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(mod.rox, sizem, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(mod.roz, sizem, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
            /* Recompute cost + gradient with updated model */
            /* Zero out gradient arrays first */
            memset(grad1, 0, sizem * sizeof(float));
            memset(grad3, 0, sizem * sizeof(float));
            if (grad2) memset(grad2, 0, sizem * sizeof(float));

            fcost = compute_fwi_gradient(&mod, ...);
            extractGradientVector(grad_vec, grad1, grad2, grad3, &mod, &bnd, param);
        }
        if (flag == OPT_NSTE && mpi_rank == 0) {
            write_iteration_model(opt.cpt_iter, x, &mod, param);
        }
    }

    /* ===== WRITE FINAL MODEL + CLEANUP ===== */
    if (mpi_rank == 0) {
        write_final_model(x, &mod, param);
    }
    optim_finalize(&opt);
    /* ... free arrays, MPI_Finalize ... */
}
```

**`compute_fwi_gradient()`** — Extracted from `fwi_driver.c` lines 414-559:
- Shot distribution: round-robin across MPI ranks
- Per-shot pipeline:
  1. `initCheckpoints` → `fdfwimodc` (forward with checkpointing)
  2. `computeResidual` (L2 misfit)
  3. `readResidual` → `applyCosineTaper` (parse residual as adjoint source)
  4. `adj_shot` (adjoint backpropagation + gradient cross-correlation)
  5. `cleanCheckpoints`
- Accumulate shot gradients into local arrays
- `MPI_Allreduce` → global gradient + global misfit
- Returns total misfit (scalar float)

---

## Files to Modify

### 9. `fdelfwi/Makefile` — Add inversion targets

```makefile
# Add to existing variable definitions
PRG8 = fwi_inversion
PRG9 = fwi_mpi_inversion

OPTIM_DIR = ../optimization
OPTIM_LIB = $(OPTIM_DIR)/liboptim.a
OPTIM_INC = -I$(OPTIM_DIR)

# New convenience targets
inversion: $(PRG8)
mpi_inversion: $(PRG9)

# Build optimizer library if needed
$(OPTIM_LIB):
	$(MAKE) -C $(OPTIM_DIR)

# Serial inversion driver
$(PRG8): $(OBJC) $(OPTIM_LIB) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) $(OPTIM_INC) -c updateModel.c
	$(CC) $(CFLAGS) $(OPTC) $(OPTIM_INC) -c fwi_inversion.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(LDFLAGS) -o $(PRG8) fwi_inversion.o updateModel.o \
	    $(OBJC) fileOpen.o writeRec.o $(OPTIM_LIB) $(LIBS)

# MPI inversion driver
$(PRG9): $(OBJC) $(OPTIM_LIB) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) $(OPTIM_INC) -c updateModel.c
	$(MPICC) -c $(CFLAGS) $(OPTC) $(OPTIM_INC) -DUSE_MPI \
	    -o fwi_inversion_mpi.o fwi_inversion.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(MPICC) $(LDFLAGS) -o $(PRG9) fwi_inversion_mpi.o updateModel.o \
	    $(OBJC) fileOpen.o writeRec.o $(OPTIM_LIB) $(LIBS)
```

Add new `.o` files to `clean` target.

### 10. `fdelfwi/fdelfwi.h` — Add function prototypes

Add after existing declarations:
```c
/* updateModel.c */
void recomputeFDcoefficients(modPar *mod, bndPar *bnd);
void extractModelVector(float *x, modPar *mod, bndPar *bnd, int param);
void injectModelVector(float *x, modPar *mod, bndPar *bnd, int param);
void extractGradientVector(float *g, float *g1, float *g2, float *g3,
                           modPar *mod, bndPar *bnd, int param);

/* fwi_gradient.c (already implemented, add header declaration) */
void convertGradientToVelocity(float *grad1, float *grad2, float *grad3,
                               float *cp, float *cs, float *rho, size_t sizem);
```

---

## Implementation Order

### Phase 1: L-BFGS C translation + unit test
1. Create `optimization/optim.h`
2. Create `optimization/optim_common.c` (norm, dot, project, convergence, linesearch, print_info)
3. Create `optimization/lbfgs.c` (init, descent, save, update, finalize, main dispatcher)
4. Create `optimization/steepest_descent.c`
5. Create `optimization/test_rosenbrock.c`
6. Create `optimization/Makefile`
7. **Verify**: `cd optimization && make test` → Rosenbrock converges to (1,1) with f≈0

### Phase 2: Model update utilities
8. Create `fdelfwi/updateModel.c`
9. Add prototypes to `fdelfwi/fdelfwi.h`
10. **Verify**: compile `updateModel.o`; round-trip test (extract → inject → compare FD coefficients)

### Phase 3: Inversion driver
11. Create `fdelfwi/fwi_inversion.c`
12. Update `fdelfwi/Makefile` with new targets
13. **Verify**: `make inversion` compiles; run 2-3 iters on small test, check misfit decreases

### Phase 4: Integration test
14. Run full inversion on demo case (5-10 iterations)
15. Compare iteration-0 gradient against `fwi_driver` output (must match exactly)
16. Check `iterate_LB.dat` convergence file

---

## Model Vector Layout

| Parameterization | Vector layout | Size |
|---|---|---|
| `param=1` (Lamé) | `x = [λ_interior \| μ_interior \| ρ_interior]` | `3*nx*nz` |
| `param=2` (velocity) | `x = [Vp_interior \| Vs_interior \| ρ_interior]` | `3*nx*nz` |
| Acoustic (ischeme≤2) | `x = [Vp_interior \| ρ_interior]` (no shear) | `2*nx*nz` |

Interior = physical domain only (nx×nz), excluding boundary padding.

## MPI Synchronization Pattern

```
Rank 0: optimizer updates x
  → injectModelVector(x) updates mod->cp/cs/rho then recomputes l2m/lam/muu/rox/roz
  → MPI_Bcast(l2m, lam, muu, rox, roz) — 5 arrays of sizem floats
All ranks: compute_fwi_gradient() — shot loop with MPI_Allreduce gradient
Rank 0: extractGradientVector → call optimizer → next iteration
```

## New Command-Line Parameters

| Parameter | Default | Description |
|---|---|---|
| `niter=` | 20 | Max optimization iterations |
| `conv=` | 1e-6 | Convergence tolerance (fcost/f0) |
| `algorithm=` | 1 | 0=steepest descent, 1=L-BFGS |
| `lbfgs_mem=` | 20 | L-BFGS history pairs |
| `nls_max=` | 20 | Max linesearch iterations per step |
| `vp_min=, vp_max=` | from model | Velocity bounds (also derives Lamé bounds) |
| `vs_min=, vs_max=` | from model | Velocity bounds |
| `rho_min=, rho_max=` | from model | Density bounds |
| `write_iter=` | 1 | Write model every N iterations |

All existing `fwi_driver` parameters (file_obs, comp, chk_skipdt, res_taper, etc.) remain unchanged.

---

## Reference: Fortran Source Files Translated

| Fortran file (TOOLBOX_OPTIMIZATION) | C target |
|---|---|
| `COMMON/include/optim_type.h` | `optimization/optim.h` (struct definition) |
| `COMMON/src/normL2.f90` | `optimization/optim_common.c` |
| `COMMON/src/scalL2.f90` | `optimization/optim_common.c` |
| `COMMON/src/project.f90` | `optimization/optim_common.c` |
| `COMMON/src/std_test_conv.f90` | `optimization/optim_common.c` |
| `COMMON/src/std_linesearch.f90` | `optimization/optim_common.c` |
| `COMMON/src/print_info.f90` | `optimization/optim_common.c` |
| `LBFGS/kernel/src/LBFGS.f90` | `optimization/lbfgs.c` |
| `LBFGS/kernel/src/init_LBFGS.f90` | `optimization/lbfgs.c` |
| `LBFGS/kernel/src/descent_LBFGS.f90` | `optimization/lbfgs.c` |
| `LBFGS/kernel/src/save_LBFGS.f90` | `optimization/lbfgs.c` |
| `LBFGS/kernel/src/update_LBFGS.f90` | `optimization/lbfgs.c` |
| `LBFGS/kernel/src/finalize_LBFGS.f90` | `optimization/lbfgs.c` |
| `LBFGS/test/src/test_LBFGS.f90` | `optimization/test_rosenbrock.c` |
