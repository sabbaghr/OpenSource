/*********************************************************************
 *
 * fdelfwi_offload.h -- OpenMP target offload layer for fdelfwi.
 *
 * ONE SOURCE, THREE TARGETS.  The finite-difference kernels are
 * annotated once; which hardware they run on is a build-time choice:
 *
 *   CPU   (default)  : gcc/icx -fopenmp          -- unchanged behaviour
 *   H100  (h100 q)   : nvc  -mp=gpu -gpu=cc90    -DUSE_OMP_TARGET
 *   PVC   (pvc  q)   : icx  -fiopenmp -fopenmp-targets=spir64 -DUSE_OMP_TARGET
 *
 * WHY OpenMP TARGET AND NOT CUDA/SYCL.  The tree already contains a
 * CUDA port (fdelmodc_cuda.cu, fdelfwi_cuda.cu, fdelfwi_gpu.cu).  It was
 * last touched 2026-09-21 and has since fallen out of parity with the
 * CPU physics -- it knows nothing about CFS-CPML, the lag-extended
 * gradients, energy_ic, data_weight=2 or misfit=1.  A second GPU
 * implementation drifts from the first the moment anybody adds physics.
 * Annotating the existing .c files means CPML and everything after it
 * exist exactly once, and the CPU build remains the reference.
 *
 *
 * ------------------------------------------------------------------
 * THE TWO RULES THAT KEEP THIS CORRECT
 * ------------------------------------------------------------------
 *
 * 1. NEVER let a struct cross into a target region.
 *
 *    modPar/bndPar are passed BY VALUE and contain pointers (see the
 *    stale-object ABI hazard already documented for bndPar).  Mapping
 *    them would copy host pointers onto the device, where they are
 *    meaningless.  Every kernel therefore hoists the scalars it needs
 *    into plain locals FIRST:
 *
 *        const int n1 = mod.naz, ioXx = mod.ioXx, ieXx = mod.ieXx;
 *        FD_LOOP_IJ
 *        for (ix = ioXx; ix < ieXx; ix++) ...
 *
 *    Scalars referenced in a target region are implicitly firstprivate,
 *    so nothing has to be mapped at all.
 *
 * 2. Kernels carry NO map clauses.
 *
 *    The wavefield and material arrays are mapped ONCE PER SHOT by the
 *    enter/exit data region in fdfwimodc.c, and the time loop never
 *    touches the host.  A map clause on a kernel would either re-copy
 *    (ruinous) or, worse, silently allocate a second device buffer.
 *    This "map once, many unmapped kernels" pattern is exactly what
 *    omp_offload_probe.c TEST2 verifies before any physics is trusted.
 *
 *
 * ------------------------------------------------------------------
 * CPU PATH IS BIT-IDENTICAL
 * ------------------------------------------------------------------
 * Without -DUSE_OMP_TARGET every macro below expands to the pragma that
 * was already there -- same `omp for`, same schedule(guided,1), same
 * nowait placement, same `#pragma simd`.  The arithmetic is untouched,
 * so the default build produces the same bits as before this header
 * existed.  That is deliberate: the CPU result is the reference the
 * offload build is validated against.
 *
 * Validation runs the OFFLOAD BUILD with OMP_TARGET_OFFLOAD=DISABLED,
 * so host and device execute literally the same annotated source.  Any
 * difference is then a real bug -- unlike the CUDA path, where
 * --use_fast_math makes a bit-comparison against the CPU impossible.
 *
 *********************************************************************/

#ifndef FDELFWI_OFFLOAD_H
#define FDELFWI_OFFLOAD_H

#ifdef USE_OMP_TARGET

/* ---------------------------------------------------------------
 * fdOffloadActive -- "are the arrays on the device right now?"
 *
 * elastic4() has exactly ONE annotated body, but it is reached from
 * several drivers and only some of them have opened a device data
 * region.  fdfwimodc() maps the wavefields per shot (slice 1a);
 * adj_shot(), born_shot() and the dot-product tests do NOT yet.  If an
 * unmapped caller hit an unconditional `omp target`, the kernel would
 * dereference host pointers on the device -- a segfault, or worse,
 * silent garbage.
 *
 * So every target construct carries `if(target: fdOffloadActive)`.
 * When the flag is 0 the construct runs on the HOST, in the host data
 * environment, on the host's own copy of the arrays -- which is
 * precisely what an unmapped caller wants.  The whole binary therefore
 * keeps working while the port advances one driver at a time.
 *
 * COST, STATED PLAINLY: a host-side fallback execution is effectively
 * single-threaded, because the host `omp parallel` regions that used to
 * supply the team are compiled out in this build (see FD_HOST_PARALLEL).
 * In the offload build adj_shot is therefore SLOW, not wrong.  It stops
 * being slow in slice 2, when adj_shot opens its own data region.
 *
 * The definition lives in fdfwimodc.c, next to the data region that is
 * the only thing allowed to raise it.
 * --------------------------------------------------------------- */
extern int fdOffloadActive;

/* ---------------------------------------------------------------
 * GPU: each loop nest becomes its own kernel launch.
 *
 * collapse(2) flattens the (ix,iz) nest into one iteration space so
 * the whole grid is available for parallelism -- the outer x-loop
 * alone is far too short to fill an H100 or a PVC tile.  Loop
 * variables of a collapse-associated nest are predetermined private,
 * so no private() clause is needed (and specifying one would be
 * ill-formed).
 *
 * `nowait` is meaningless here and is intentionally dropped: a target
 * region is synchronous by default, which supplies exactly the
 * ordering the CPU path gets from the implicit barrier at the end of
 * its non-nowait `omp for`.
 * --------------------------------------------------------------- */

#define FD_LOOP_IJ      _Pragma("omp target teams distribute parallel for collapse(2) if(target: fdOffloadActive)")
#define FD_LOOP_IJ_NW   _Pragma("omp target teams distribute parallel for collapse(2) if(target: fdOffloadActive)")
#define FD_SIMD         /* consumed by collapse(2) */

/* Host-side `omp parallel` is not used in the offload build: the
 * kernels launch themselves.  The braces of the former parallel
 * region simply become an ordinary block.
 *
 * THIS IS NOT COSMETIC -- IT IS REQUIRED FOR CORRECTNESS.  A `target`
 * construct encountered inside a host `omp parallel` is encountered
 * ONCE PER THREAD, so each thread would launch its own kernel over the
 * full grid and they would race on `vx[i] -= ...`.  Deleting the host
 * team is what makes the single annotated body safe for every caller.
 *
 * The orphaned `omp master` / `omp barrier` inside boundaries.c and
 * boundaries_cpml_adj.c need no change: with no enclosing parallel
 * region they bind to the initial thread's implicit team of one, where
 * `master` executes and `barrier` is a no-op. */
#define FD_HOST_PARALLEL
#define FD_HOST_BARRIER
#define FD_HOST_MASTER

/* ---------------------------------------------------------------
 * TEMPORARY SCAFFOLDING -- slice 1a only.  DELETE IN SLICE 1b.
 *
 * applySource(), boundariesP() and boundariesV() still execute on the
 * host, so the five wavefields have to be round-tripped around them.
 * These two macros make that cost visible at every call site instead
 * of burying it.
 *
 * NAME CAPTURE, DELIBERATE: they refer to vx/vz/txx/tzz/txz and sizem
 * by name, because _Pragma() takes a string literal and macro
 * arguments cannot be interpolated into it.  Every elastic kernel
 * already uses exactly these names, so the capture is consistent --
 * but it does mean these macros are usable ONLY inside the elastic FD
 * kernels, and `size_t sizem` must be in scope.
 *
 * Expected cost while this exists: 10 full-grid transfers per time
 * step.  On a demo-sized grid (400x160) that is ~2.5 MB/step, which is
 * tolerable for correctness work and intolerable for anything else.
 * --------------------------------------------------------------- */
#define OFFLOAD_WFL_FROM_DEVICE \
    _Pragma("omp target update if(fdOffloadActive) from(vx[0:sizem], vz[0:sizem], txx[0:sizem], tzz[0:sizem], txz[0:sizem])")
#define OFFLOAD_WFL_TO_DEVICE \
    _Pragma("omp target update if(fdOffloadActive) to(vx[0:sizem], vz[0:sizem], txx[0:sizem], tzz[0:sizem], txz[0:sizem])")

/* A useful side effect of the round-trips above: when elastic4 returns,
 * the HOST copies are current (the last thing it did was push the
 * host-side boundary result back to the device).  getRecTimes,
 * writeSnapTimes and writeCheckpoint in fdfwimodc.c therefore need no
 * update of their own in slice 1a.  That stops being true the moment
 * the boundary kernels move to the device -- slice 1b must add an
 * explicit `target update from` before each of those three. */

#else /* ---------------- CPU (default, unchanged) ---------------- */

#define FD_LOOP_IJ      _Pragma("omp for private(ix,iz) schedule(guided,1)")
#define FD_LOOP_IJ_NW   _Pragma("omp for private(ix,iz) nowait schedule(guided,1)")
#define FD_SIMD         _Pragma("simd")

#define FD_HOST_PARALLEL _Pragma("omp parallel default(shared)")
#define FD_HOST_BARRIER  _Pragma("omp barrier")
#define FD_HOST_MASTER   _Pragma("omp master")

#define OFFLOAD_WFL_FROM_DEVICE
#define OFFLOAD_WFL_TO_DEVICE

#endif /* USE_OMP_TARGET */

#endif /* FDELFWI_OFFLOAD_H */
