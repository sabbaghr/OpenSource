# Makefile for fdelfwi - self-contained FWI forward modeling module

include ../Make_include

########################################################################
# define general include and system library
ALLINC  = -I.

# MPI compiler (override with MPICC=... on command line if needed)
MPICC ?= mpicc

# Optimization library
OPTIM_DIR = ../optimization
OPTIM_LIB = $(OPTIM_DIR)/liboptim.a
OPTIM_INC = -I$(OPTIM_DIR)

# CUDA settings (enable with USE_CUDA=1 on command line)
NVCC ?= nvcc
# MIO: P100 (sm_60), Wendian: V100 (sm_70) / A100 (sm_80)
# Override on command line: make ... CUDA_ARCH="-arch=sm_70"
CUDA_ARCH ?= -arch=sm_60
CUDA_FLAGS = -O3 --use_fast_math -lineinfo $(CUDA_ARCH) -Xcompiler "$(OPTC)"
CUDA_LIBS = -lcudart -lstdc++

########################################################################
# OpenMP target offload -- ONE SOURCE, every NVIDIA card and Intel PVC.
#
#   make OFFLOAD=p100 fwd     nvc, NVIDIA cc60   (MIO gpu004, module libs/nvhpc/21.3)
#   make OFFLOAD=h100 fwd     nvc, NVIDIA cc90   (Stampede3, module nvidia/26.1)
#   make OFFLOAD=pvc  fwd     icx, Intel spir64  (Stampede3, module intel/26.0)
#   make OFFLOAD=host fwd     gcc, target regions forced to the host
#   make              fwd     unchanged gcc CPU build (the reference)
#
# This overrides only the toolchain variables, so EVERY existing target
# gains an offload build and there is no per-program rule to keep in sync.
#
# Three things must change together, not just the compiler:
#
#  1. -DUSE_OMP_TARGET flips fdelfwi_offload.h from `omp for` to
#     `omp target teams distribute parallel for collapse(2)`.
#
#  2. The OpenMP RUNTIME.  Make_include links -lmkl_gnu_thread and -lgomp.
#     Pulling libgomp into an nvc or icx binary puts TWO OpenMP runtimes in
#     one process, which deadlocks or silently serialises.  MKL is used here
#     only for FFTs, never in the FD hot loop, so the sequential layer is
#     substituted: it has no threading runtime of its own and so cannot
#     conflict.  Costs nothing that matters.
#
#  3. -march=native -ffast-math are gcc spellings and are dropped.  Dropping
#     fast-math is DELIBERATE, not an oversight: the offload build is
#     validated by bit-comparison against the gcc CPU reference, and that is
#     exactly what --use_fast_math makes impossible for the legacy CUDA path.
########################################################################
# Only the NON-gcc toolchains need the MKL runtime swap.  The gcc-based
# offload modes (host, gcc-nvptx) use libgomp themselves, so Make_include's
# -lmkl_gnu_thread/-lgomp is already the right pairing for them.
ifneq ($(filter $(OFFLOAD),nvidia p100 v100 a100 h100 intel pvc),)
  MKL_SEQ_LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential \
                 -lmkl_core -lpthread -lm -ldl
  LIBS = -L$L -lgenfft $(MKL_SEQ_LIBS) -lgfortran
endif

# NVIDIA / NVHPC.  Compute capability is the ONLY thing that changes between
# cards, so the mode is "nvidia" and the card names are just shorthands:
#   OFFLOAD=p100     cc60    DOES NOT BUILD -- see below
#   OFFLOAD=v100     cc70    Wendian
#   OFFLOAD=a100     cc80
#   OFFLOAD=h100     cc90    Stampede3
#   OFFLOAD=nvidia           you set GPU_ARCH= yourself
#
# OFFLOAD=p100 is kept only to document that it is a DEAD END with nvc.
# Measured on MIO 2026-11-14 with nvhpc 21.3:
#   nvc-Fatal-OpenMP GPU Offload is available only on systems with NVIDIA
#   GPUs with compute capability '>= cc70'
# NVHPC supports Pascal for OpenACC but not for OpenMP target.  MIO's only
# GPU node (gpu004) is a Tesla P100, so on MIO use OFFLOAD=gcc-nvptx below;
# gcc's nvptx offload does emit sm_60.
ifeq ($(OFFLOAD),p100)
  GPU_ARCH ?= cc60
endif
ifeq ($(OFFLOAD),v100)
  GPU_ARCH ?= cc70
endif
ifeq ($(OFFLOAD),a100)
  GPU_ARCH ?= cc80
endif
ifeq ($(OFFLOAD),h100)
  GPU_ARCH ?= cc90
endif
ifneq ($(filter $(OFFLOAD),nvidia p100 v100 a100 h100),)
  CC        = nvc
  GPU_ARCH ?= cc70
  OPTC      = -O3 -mp=gpu -gpu=$(GPU_ARCH) -DUSE_OMP_TARGET -Minfo=mp
  LDFLAGS  += -mp=gpu -gpu=$(GPU_ARCH)
endif

# Intel / oneAPI.  "pvc" is a shorthand for "intel".  spir64 is JIT'd for
# whatever Level Zero device is present, so there is no arch flag to set.
ifneq ($(filter $(OFFLOAD),intel pvc),)
  CC       = icx
  OPTC     = -O3 -fiopenmp -fopenmp-targets=spir64 -DUSE_OMP_TARGET
  LDFLAGS += -fiopenmp -fopenmp-targets=spir64
endif

# Host-fallback validation build.  Same annotated source, no device code
# generated, target regions execute on the host.  This is what makes the
# port testable on a workstation with no GPU toolchain at all -- and on
# Stampede3 it is the build you compare against before spending queue time.
ifeq ($(OFFLOAD),host)
  CC       = gcc
  OPTC     = -O3 -fopenmp -foffload=disable -DUSE_OMP_TARGET
endif

# gcc + nvptx offload.  The ONLY way to reach a Pascal card from OpenMP
# target, since NVHPC refuses anything below cc70 (see the note above).
# Needs a gcc built with --enable-offload-targets=nvptx-none, i.e. the
# distro package gcc-offload-nvptx / libgomp-plugin-nvptx.  Probe with:
#   gcc -fopenmp -foffload=nvptx-none -O1 probe.c -o probe
# PTX_ISA is overridable; gcc's default is old enough to run anywhere, and
# sm_60 is what MIO's P100 actually is.
ifeq ($(OFFLOAD),gcc-nvptx)
  CC       = gcc
  PTX_ISA ?= sm_60
  OPTC     = -O3 -fopenmp -foffload=nvptx-none \
             -foffload-options=nvptx-none=-misa=$(PTX_ISA) -DUSE_OMP_TARGET
  LDFLAGS += -fopenmp -foffload=nvptx-none
endif

all: fwi_inversion

# FWI driver targets
driver: fwi_driver
mpi: fwi_mpi_driver
inversion: fwi_inversion
mpi_inversion: fwi_mpi_inversion
test_wave_op_dp: $(PRG5)
test_dotproduct: $(PRG6)
test_taylor: $(PRG7)
test_hessian_dp: $(PRG10)
test_gradient_visual: $(PRG12)
test_recop_dp: $(PRG14)
test_lag_gradient: $(PRG17)
test_lag_cone: $(PRG18)
test_space_lag_gradient: $(PRG19)
test_cpml_dp: $(PRG21)
test_fs_col: $(PRG22)
test_petro: $(PRG20)
petro_gradient: $(PRG24)

PRG3 = fwi_driver
PRG4 = fwi_mpi_driver
PRG5 = test_wave_op_dp
PRG6 = test_dotproduct
PRG7 = test_taylor
PRG8 = fwi_inversion
PRG9 = fwi_mpi_inversion
PRG10 = test_hessian_dp
PRG12 = test_gradient_visual
PRG13 = fdelfwi_fwd
PRG14 = test_recop_dp
PRG15 = test_ei_planewave
PRG16 = test_ei_modesep
PRG17 = test_lag_gradient
PRG18 = test_lag_cone
PRG19 = test_space_lag_gradient
PRG20 = test_petro
PRG21 = test_cpml_dp
PRG22 = test_fs_col
PRG24 = petro_gradient

# Alias must come AFTER PRG13 is defined: make expands prerequisites
# immediately when the rule is read.
fwd: $(PRG13)

# Library sources (no test drivers -- those have main())
SRCC	= fdfwimodc.c \
		acoustic2.c \
		acoustic4.c \
		acousticSH4.c \
		acoustic4_qr.c \
		acoustic6.c \
		acoustic16.c \
		viscoacoustic4.c \
		elastic4.c \
		elastic4dc.c \
		elastic6.c \
		elastic8.c \
		viscoelastic4.c \
		defineSource.c  \
		getParameters.c  \
		getWaveletInfo.c  \
		getModelInfo.c  \
		applySource.c  \
		getRecTimes.c  \
		getBeamTimes.c  \
		writeSnapTimes.c  \
		writeSrcRecPos.c  \
		decomposition.c  \
		recvPar.c  \
		readModel.c  \
		readResidual.c  \
		checkpoint.c  \
		adj_shot.c  \
		fwi_gradient.c  \
		applyAdjointSource.c  \
		acoustic4_adj.c  \
		acoustic_gradient.c  \
		elastic4_adj.c  \
		elastic6_adj.c  \
		elastic8_adj.c  \
		computeResidual.c  \
		energy_weights.c  \
		energy_imaging.c  \
		lag_gradient.c  \
		lag_modesep.c  \
		space_lag_gradient.c  \
		lag_cone_filter.c  \
		lag_cone_pspi.c  \
		kh_misfit.c  \
		aperture_taper.c  \
		residual_taper.c  \
		grad_taper_mask.c  \
		computeHydrophone.c  \
		born_vsrc.c  \
		born_shot.c  \
		hess_shot.c  \
		preconditioner.c  \
		illum_precond.c  \
		taper_source.c  \
		precond_shot.c  \
		smooth2d.c  \
		scaling.c  \
		toy2dac_scaling.c  \
		source_estimation.c  \
		regularization.c  \
		petro_constraint.c  \
		shin_precond_acoustic.c  \
		bandpass_filter.c  \
		sourceOnSurface.c  \
		getWaveletHeaders.c  \
		boundaries.c  \
		boundaries_adj.c  \
		cpml_coef.c  \
		cpml_state.c  \
		boundaries_cpml_adj.c  \
		verbosepkg.c  \
		writesufile.c  \
		gaussGen.c  \
		spline3.c  \
		CMWC4096.c  \
		wallclock_time.c  \
		name_ext.c  \
		atopkge.c \
		docpkge.c \
		threadAffinity.c \
		getpars.c

OBJC	= $(SRCC:%.c=%.o)

# fdelfwi.h defines the shared structs (modPar, recPar, adjSrcPar, ...).
# Without this dependency an edit to the header only triggers a relink, so
# objects keep the old struct layout and the binary segfaults silently.
$(OBJC): fdelfwi.h

# Serial FWI driver (no MPI)
$(PRG3):	$(OBJC) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c fwi_driver.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG3) fwi_driver.o $(OBJC) fileOpen.o writeRec.o $(LIBS)

# Wave operator dot product test
$(PRG5):	$(OBJC) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_wave_op_dp.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG5) test_wave_op_dp.o $(OBJC) fileOpen.o writeRec.o $(LIBS)

# Claerbout dot product test (model-parameter Jacobian)
$(PRG6):	$(OBJC) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_dotproduct.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG6) test_dotproduct.o $(OBJC) fileOpen.o writeRec.o $(LIBS)

# Taylor gradient verification test
$(PRG7):	$(OBJC) fdelfwi.h test_taylor.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_taylor.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG7) test_taylor.o $(OBJC) fileOpen.o writeRec.o $(LIBS)

# MPI driver: only fwi_driver.c needs MPI (-DUSE_MPI).
# All other sources (elastic4, boundaries, adj_shot, etc.) have no MPI calls --
# MPI parallelization is over shots in the driver, not within a single shot.
# We reuse the serial $(OBJC) objects and link with mpicc to pull in MPI libs.
$(PRG4):	$(OBJC) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -DUSE_MPI -o fwi_driver_mpi.o fwi_driver.c
	$(MPICC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG4) fwi_driver_mpi.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Build optimizer library if needed
$(OPTIM_LIB):
	$(MAKE) -C $(OPTIM_DIR)

# Serial FWI inversion (optimization loop + gradient computation)
# fwi_inversion.c and updateModel.c are compiled by the recipe but are not in
# $(OBJC), so they must be listed here or an edit to either leaves a STALE
# binary that make reports as "up to date".
$(PRG8):	$(OBJC) $(OPTIM_LIB) fdelfwi.h fwi_inversion.c updateModel.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(OPTIM_INC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(CC) $(OPTIM_INC) $(CFLAGS) $(OPTC) -c fwi_inversion.c
	$(CC) $(LDFLAGS) $(OPTIM_INC) $(CFLAGS) $(OPTC) -o $(PRG8) fwi_inversion.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(OPTIM_LIB) $(LIBS)

# MPI FWI inversion
$(PRG9):	$(OBJC) $(OPTIM_LIB) fdelfwi.h fwi_inversion.c updateModel.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(OPTIM_INC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(MPICC) -c $(OPTIM_INC) $(CFLAGS) $(OPTC) -DUSE_MPI -o fwi_inversion_mpi.o fwi_inversion.c
	$(MPICC) $(LDFLAGS) $(OPTIM_INC) $(CFLAGS) $(OPTC) -o $(PRG9) fwi_inversion_mpi.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(OPTIM_LIB) $(LIBS)

# GPU FWI inversion (single GPU, no MPI)
gpu_inversion: fwi_gpu_inversion
fwi_gpu_inversion:	$(OBJC) $(OPTIM_LIB) fdelfwi.h fdelfwi_gpu.h domain_decomp.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(OPTIM_INC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(CC) $(OPTIM_INC) $(CFLAGS) $(OPTC) -DUSE_CUDA -c -o fwi_inversion_gpu.o fwi_inversion.c
	$(NVCC) $(CUDA_FLAGS) -DUSE_CUDA -x cu -c -o domain_decomp.o domain_decomp.c
	$(NVCC) $(CUDA_FLAGS) -DUSE_CUDA -c -o fdelmodc_cuda.o fdelmodc_cuda.cu
	$(NVCC) $(CUDA_FLAGS) -DUSE_CUDA -c -o fdelfwi_cuda.o fdelfwi_cuda.cu
	$(NVCC) $(CUDA_FLAGS) -DUSE_CUDA -c -o fdelfwi_gpu.o fdelfwi_gpu.cu
	$(CC) $(LDFLAGS) $(OPTIM_INC) $(CFLAGS) $(OPTC) -o fwi_gpu_inversion \
		fwi_inversion_gpu.o fdelfwi_gpu.o fdelmodc_cuda.o fdelfwi_cuda.o \
		domain_decomp.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(OPTIM_LIB) $(LIBS) $(CUDA_LIBS)

# GPU+MPI FWI inversion (shot-parallel MPI + optional domain decomposition)
# Usage: mpirun -np N fwi_mpi_gpu_inversion ... ndom=K
#   N = K * nshot_groups (total ranks = GPUs/shot × concurrent shot groups)
#   ndom=1: single GPU per shot, N = nshot_groups (backward compatible)
#   ndom=K>1: K GPUs per shot with 1D X-axis domain decomposition + halo exchange
mpi_gpu_inversion: fwi_mpi_gpu_inversion
fwi_mpi_gpu_inversion:	$(OBJC) $(OPTIM_LIB) fdelfwi.h fdelfwi_gpu.h domain_decomp.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(OPTIM_INC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(MPICC) -c $(OPTIM_INC) $(CFLAGS) $(OPTC) -DUSE_CUDA -DUSE_MPI -o fwi_inversion_mpi_gpu.o fwi_inversion.c
	$(NVCC) $(CUDA_FLAGS) -DUSE_CUDA -DUSE_MPI -c -o domain_decomp.o domain_decomp.c
	$(NVCC) $(CUDA_FLAGS) -DUSE_CUDA -DUSE_MPI -c -o fdelmodc_cuda.o fdelmodc_cuda.cu
	$(NVCC) $(CUDA_FLAGS) -DUSE_CUDA -DUSE_MPI -c -o fdelfwi_cuda.o fdelfwi_cuda.cu
	$(NVCC) $(CUDA_FLAGS) -DUSE_CUDA -DUSE_MPI -c -o fdelfwi_gpu.o fdelfwi_gpu.cu
	$(MPICC) $(LDFLAGS) $(OPTIM_INC) $(CFLAGS) $(OPTC) -o fwi_mpi_gpu_inversion \
		fwi_inversion_mpi_gpu.o fdelfwi_gpu.o fdelmodc_cuda.o fdelfwi_cuda.o \
		domain_decomp.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(OPTIM_LIB) $(LIBS) $(CUDA_LIBS)

# Hessian symmetry dot product test
$(PRG10):	$(OBJC) fdelfwi.h test_hessian_dp.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(CC) $(CFLAGS) $(OPTC) -c test_hessian_dp.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG10) test_hessian_dp.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Gradient visualization test (raw + Brossier scaled SU output).
# Depends on its own source so that editing test_gradient_visual.c actually
# triggers a relink -- without it make reports "up to date" and silently
# leaves a stale binary in place.
$(PRG12):	$(OBJC) fdelfwi.h test_gradient_visual.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(CC) $(CFLAGS) $(OPTC) -c test_gradient_visual.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG12) test_gradient_visual.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Forward-only modeling with the fdelfwi engine (observed-data generation).
# Unlike the other rules this one also depends on its own source file, so that
# editing fdelfwi_fwd.c actually triggers a relink.
$(PRG13):	$(OBJC) fdelfwi.h fdelfwi_fwd.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c fdelfwi_fwd.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG13) fdelfwi_fwd.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Receiver restriction/injection dot product test (pure algebra, no
# time stepping).  Depends on its own source so edits trigger a relink.
$(PRG14):	$(OBJC) fdelfwi.h test_recop_dp.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_recop_dp.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG14) test_recop_dp.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Analytic plane-wave calibration of the gamma=0 energy-imaging accumulators.
# No time stepping and no I/O: it builds one plane wave, pairs it with itself
# and checks rms(P)/rms(K) == 1, which is exact for BOTH P and S.  P exercises
# only the div path of V_0, S only the curl path, so this is the one test that
# can distinguish a curl mis-scaling from correct conversion-nulling.
$(PRG15):	$(OBJC) fdelfwi.h test_ei_planewave.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_ei_planewave.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG15) test_ei_planewave.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Analytic TWO-plane-wave angle calibration of the EI_MODESEP accumulators.
# test_ei_planewave pairs one wave with itself, which is the single point
# psi = 0 and so cannot see the angle dependence at all.  This one sweeps the
# opening angle and checks V = cos(psi)*K pointwise and a_tomo = cos^2(psi/2),
# both exact and both free of Vp/Vs -- the claim EI_MODESEP is built on.  It
# also measures the P-S blind spot rather than asserting it.
$(PRG16):	$(OBJC) fdelfwi.h test_ei_modesep.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_ei_modesep.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG16) test_ei_modesep.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Standalone time-lag gradient test.  Deliberately NOT part of
# test_gradient_visual: that driver always builds and applies the block
# preconditioner and writes the pseudo-Hessian, none of which the lag test
# wants.  This one accumulates in Lame and converts once to (Vp, Vs, rho), so
# the tau=0 slice of _lagvp.su / _lagvs.su can be compared against
# _vp_raw.su / _vs_raw.su directly.
# Depends on its own source so edits trigger a relink.
$(PRG17):	$(OBJC) fdelfwi.h test_lag_gradient.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(CC) $(CFLAGS) $(OPTC) -c test_lag_gradient.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG17) test_lag_gradient.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# test_lag_cone -- cone-filter a lag cube that already exists on disk.  No
# wave equation, so it links only the library objects; its purpose is to let
# the C filter be compared sample by sample against the python reference
# demo/gradient/lag_slope_filter.py before the filter is trusted inside an
# inversion.
$(PRG18):	$(OBJC) fdelfwi.h test_lag_cone.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_lag_cone.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG18) test_lag_cone.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Standalone space-lag gradient test, the sibling of $(PRG17).  Separate
# program for the same reason: the lag cubes are read straight out of
# space_lag_gradient.c's own accumulators and must not be perturbed by the
# preconditioner, scaling or smoothing that test_gradient_visual builds.  Its
# h=0 check is EXACT (no band limiting, unlike the time-lag tau=0 check), and
# the driver makes it numerically and returns non-zero on failure, so this
# target is usable as a regression test.
$(PRG19):	$(OBJC) fdelfwi.h test_space_lag_gradient.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(CC) $(CFLAGS) $(OPTC) -c test_space_lag_gradient.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG19) test_space_lag_gradient.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# test_petro -- J_P is a pure model-domain functional, so it is checked
# with no wave equation at all and runs in well under a second.  Worth
# having standalone because inside fwi_inversion J_P is buried under
# scaling, masking and a chain rule, where a sign or exponent error would
# surface only as an unexplained linesearch failure.  Returns non-zero on
# failure, so this target is usable as a regression test.
$(PRG20):	$(OBJC) fdelfwi.h test_petro.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_petro.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG20) test_petro.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# petro_gradient -- adds a*g_P (model-only) to an existing data gradient.
# No wave propagation; same petro_constraint.c code fwi_inversion uses.
$(PRG24):	$(OBJC) fdelfwi.h petro_gradient.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c petro_gradient.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG24) petro_gradient.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Bare-propagator dot product test: <F^N S0, R> = <S0, (F^T)^N R> over the
# full padded wavefield, with no source and no receivers.  This is the test
# that isolates the boundary transpose -- CFS-CPML in particular -- from the
# source injection and receiver registration that test_wave_op_dp also
# exercises.  Returns non-zero on failure.
$(PRG21):	$(OBJC) fdelfwi.h test_cpml_dp.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_cpml_dp.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG21) test_cpml_dp.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

# Brute-force extraction of ONE column of F^T at a free-surface txx slot,
# compared entry by entry against the adjoint the code computes.  A dot
# product returns a single scalar and can only say "wrong"; this names the
# exact (component, ix, iz) entry that is wrong.
$(PRG22):	$(OBJC) fdelfwi.h test_fs_col.c
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_fs_col.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG22) test_fs_col.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

install: $(PRG8) $(PRG9)
	cp $(PRG8) $B
	cp $(PRG9) $B

clean:
		rm -f core $(OBJC) *_mpi.o fileOpen.o writeRec.o updateModel.o \
			fwi_driver.o fwi_driver_mpi.o \
			fwi_inversion.o fwi_inversion_mpi.o \
			fwi_inversion_gpu.o fwi_inversion_mpi_gpu.o \
			fdelmodc_cuda.o fdelfwi_cuda.o fdelfwi_gpu.o domain_decomp.o \
			test_wave_op_dp.o test_dotproduct.o test_taylor.o test_hessian_dp.o \
			test_gradient_visual.o fdelfwi_fwd.o test_recop_dp.o \
			test_ei_planewave.o test_ei_modesep.o test_lag_gradient.o \
			test_space_lag_gradient.o test_petro.o petro_gradient.o test_cpml_dp.o \
			test_fs_col.o \
			$(PRG3) $(PRG4) $(PRG5) $(PRG6) $(PRG7) $(PRG8) $(PRG9) $(PRG10) $(PRG12) $(PRG13) $(PRG14) $(PRG15) $(PRG16) $(PRG17) $(PRG19) $(PRG20) $(PRG21) $(PRG22) \
			fwi_gpu_inversion fwi_mpi_gpu_inversion

realclean:
		rm -f core $(OBJC) fileOpen.o writeRec.o updateModel.o \
			fwi_driver.o $(PRG3) $B/$(PRG3)
		rm -f *_mpi.o fwi_driver_mpi.o fwi_inversion_mpi.o $(PRG4) $(PRG9) \
			$B/$(PRG4) $B/$(PRG9)
		rm -f fwi_inversion.o $(PRG8) $B/$(PRG8)

print:	Makefile $(SRC)
	$(PRINT) $?
	@touch print

count:
	@wc $(SRC)

tar:
	@tar cf fdelfwi.tar Makefile $(SRC) && compress fdelfwi.tar
