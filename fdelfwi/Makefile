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

all: test_fdfwimodc test_fwi_gradient

# FWI driver targets
driver: fwi_driver
mpi: fwi_mpi_driver
inversion: fwi_inversion
mpi_inversion: fwi_mpi_inversion
test_wave_op_dp: $(PRG5)
test_dotproduct: $(PRG6)
test_taylor: $(PRG7)
test_hessian_dp: $(PRG10)

PRG1 = test_fdfwimodc
PRG2 = test_fwi_gradient
PRG3 = fwi_driver
PRG4 = fwi_mpi_driver
PRG5 = test_wave_op_dp
PRG6 = test_dotproduct
PRG7 = test_taylor
PRG8 = fwi_inversion
PRG9 = fwi_mpi_inversion
PRG10 = test_hessian_dp

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
		elastic4_adj.c  \
		elastic6_adj.c  \
		elastic8_adj.c  \
		computeResidual.c  \
		born_vsrc.c  \
		born_shot.c  \
		hess_shot.c  \
		sourceOnSurface.c  \
		getWaveletHeaders.c  \
		boundaries.c  \
		boundaries_adj.c  \
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

$(PRG1):	$(OBJC) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_fdfwimodc.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG1) test_fdfwimodc.o $(OBJC) fileOpen.o writeRec.o $(LIBS)

$(PRG2):	$(OBJC) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c test_fwi_gradient.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG2) test_fwi_gradient.o $(OBJC) fileOpen.o writeRec.o $(LIBS)

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
$(PRG7):	$(OBJC) fdelfwi.h
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
$(PRG8):	$(OBJC) $(OPTIM_LIB) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(OPTIM_INC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(CC) $(OPTIM_INC) $(CFLAGS) $(OPTC) -c fwi_inversion.c
	$(CC) $(LDFLAGS) $(OPTIM_INC) $(CFLAGS) $(OPTC) -o $(PRG8) fwi_inversion.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(OPTIM_LIB) $(LIBS)

# MPI FWI inversion
$(PRG9):	$(OBJC) $(OPTIM_LIB) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(OPTIM_INC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(MPICC) -c $(OPTIM_INC) $(CFLAGS) $(OPTC) -DUSE_MPI -o fwi_inversion_mpi.o fwi_inversion.c
	$(MPICC) $(LDFLAGS) $(OPTIM_INC) $(CFLAGS) $(OPTC) -o $(PRG9) fwi_inversion_mpi.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(OPTIM_LIB) $(LIBS)

# Hessian symmetry dot product test
$(PRG10):	$(OBJC) fdelfwi.h
	$(CC) $(CFLAGS) $(OPTC) -c fileOpen.c
	$(CC) $(CFLAGS) $(OPTC) -c writeRec.c
	$(CC) $(CFLAGS) $(OPTC) -c updateModel.c
	$(CC) $(CFLAGS) $(OPTC) -c test_hessian_dp.c
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG10) test_hessian_dp.o updateModel.o \
		$(OBJC) fileOpen.o writeRec.o $(LIBS)

install: $(PRG1) $(PRG2) $(PRG4) $(PRG8) $(PRG9)
	cp $(PRG1) $B
	cp $(PRG2) $B
	cp $(PRG4) $B
	cp $(PRG8) $B
	cp $(PRG9) $B

clean:
		rm -f core $(OBJC) *_mpi.o fileOpen.o writeRec.o updateModel.o \
			test_fdfwimodc.o test_fwi_gradient.o fwi_driver.o fwi_driver_mpi.o \
			fwi_inversion.o fwi_inversion_mpi.o \
			test_wave_op_dp.o test_dotproduct.o test_taylor.o test_hessian_dp.o \
			$(PRG1) $(PRG2) $(PRG3) $(PRG4) $(PRG5) $(PRG6) $(PRG7) $(PRG8) $(PRG9) $(PRG10)

realclean:
		rm -f core $(OBJC) fileOpen.o writeRec.o updateModel.o \
			test_fdfwimodc.o test_fwi_gradient.o fwi_driver.o $(PRG1) $(PRG2) $(PRG3) \
			$B/$(PRG1) $B/$(PRG2) $B/$(PRG3)
		rm -f *_mpi.o fwi_driver_mpi.o fwi_inversion_mpi.o $(PRG4) $(PRG9) \
			$B/$(PRG4) $B/$(PRG9)
		rm -f fwi_inversion.o $(PRG8) $B/$(PRG8)

print:	Makefile $(SRC)
	$(PRINT) $?
	@touch print

count:
	@wc $(SRC)

tar:
	@tar cf $(PRG1).tar Makefile $(SRC) && compress $(PRG1).tar
