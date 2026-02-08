# Makefile for fdelfwi - self-contained FWI forward modeling module

include ../Make_include

########################################################################
# define general include and system library
ALLINC  = -I.

# MPI compiler (override with MPICC=... on command line if needed)
MPICC ?= mpicc

all: test_fdfwimodc test_fwi_gradient

# FWI driver targets
driver: fwi_driver
mpi: fwi_mpi_driver
test_wave_op_dp: $(PRG5)
test_dotproduct: $(PRG6)

PRG1 = test_fdfwimodc
PRG2 = test_fwi_gradient
PRG3 = fwi_driver
PRG4 = fwi_mpi_driver
PRG5 = test_wave_op_dp
PRG6 = test_dotproduct

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
		applyAdjointSource.c  \
		elastic4_adj.c  \
		elastic6_adj.c  \
		elastic8_adj.c  \
		computeResidual.c  \
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

# MPI driver: recompile sources with MPI compiler and -DUSE_MPI
# Object files are named *_mpi.o to avoid conflict with serial objects
OBJC_MPI = $(SRCC:%.c=%_mpi.o)

%_mpi.o: %.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o $@ $<

$(PRG4):	fdelfwi.h
	$(MPICC) -c $(CFLAGS) $(OPTC) -o fileOpen_mpi.o fileOpen.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o writeRec_mpi.o writeRec.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -DUSE_MPI -o fwi_driver_mpi.o fwi_driver.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o fdfwimodc_mpi.o fdfwimodc.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o acoustic2_mpi.o acoustic2.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o acoustic4_mpi.o acoustic4.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o acousticSH4_mpi.o acousticSH4.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o acoustic4_qr_mpi.o acoustic4_qr.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o acoustic6_mpi.o acoustic6.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o acoustic16_mpi.o acoustic16.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o viscoacoustic4_mpi.o viscoacoustic4.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o elastic4_mpi.o elastic4.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o elastic4dc_mpi.o elastic4dc.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o elastic6_mpi.o elastic6.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o elastic8_mpi.o elastic8.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o viscoelastic4_mpi.o viscoelastic4.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o defineSource_mpi.o defineSource.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o getParameters_mpi.o getParameters.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o getWaveletInfo_mpi.o getWaveletInfo.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o getModelInfo_mpi.o getModelInfo.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o applySource_mpi.o applySource.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o getRecTimes_mpi.o getRecTimes.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o getBeamTimes_mpi.o getBeamTimes.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o writeSnapTimes_mpi.o writeSnapTimes.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o writeSrcRecPos_mpi.o writeSrcRecPos.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o decomposition_mpi.o decomposition.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o recvPar_mpi.o recvPar.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o readModel_mpi.o readModel.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o readResidual_mpi.o readResidual.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o checkpoint_mpi.o checkpoint.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o adj_shot_mpi.o adj_shot.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o applyAdjointSource_mpi.o applyAdjointSource.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o elastic4_adj_mpi.o elastic4_adj.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o elastic6_adj_mpi.o elastic6_adj.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o elastic8_adj_mpi.o elastic8_adj.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o computeResidual_mpi.o computeResidual.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o sourceOnSurface_mpi.o sourceOnSurface.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o getWaveletHeaders_mpi.o getWaveletHeaders.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o boundaries_mpi.o boundaries.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o boundaries_adj_mpi.o boundaries_adj.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o verbosepkg_mpi.o verbosepkg.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o writesufile_mpi.o writesufile.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o gaussGen_mpi.o gaussGen.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o spline3_mpi.o spline3.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o CMWC4096_mpi.o CMWC4096.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o wallclock_time_mpi.o wallclock_time.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o name_ext_mpi.o name_ext.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o atopkge_mpi.o atopkge.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o docpkge_mpi.o docpkge.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o threadAffinity_mpi.o threadAffinity.c
	$(MPICC) -c $(CFLAGS) $(OPTC) -o getpars_mpi.o getpars.c
	$(MPICC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $(PRG4) fwi_driver_mpi.o \
		fdfwimodc_mpi.o acoustic2_mpi.o acoustic4_mpi.o acousticSH4_mpi.o \
		acoustic4_qr_mpi.o acoustic6_mpi.o acoustic16_mpi.o viscoacoustic4_mpi.o \
		elastic4_mpi.o elastic4dc_mpi.o elastic6_mpi.o elastic8_mpi.o viscoelastic4_mpi.o \
		defineSource_mpi.o getParameters_mpi.o getWaveletInfo_mpi.o getModelInfo_mpi.o \
		applySource_mpi.o getRecTimes_mpi.o getBeamTimes_mpi.o writeSnapTimes_mpi.o \
		writeSrcRecPos_mpi.o decomposition_mpi.o recvPar_mpi.o readModel_mpi.o \
		readResidual_mpi.o checkpoint_mpi.o adj_shot_mpi.o applyAdjointSource_mpi.o \
		elastic4_adj_mpi.o elastic6_adj_mpi.o elastic8_adj_mpi.o computeResidual_mpi.o sourceOnSurface_mpi.o \
		getWaveletHeaders_mpi.o boundaries_mpi.o boundaries_adj_mpi.o verbosepkg_mpi.o writesufile_mpi.o \
		gaussGen_mpi.o spline3_mpi.o CMWC4096_mpi.o wallclock_time_mpi.o \
		name_ext_mpi.o atopkge_mpi.o docpkge_mpi.o threadAffinity_mpi.o getpars_mpi.o \
		fileOpen_mpi.o writeRec_mpi.o $(LIBS)

install: $(PRG1) $(PRG2) $(PRG4)
	cp $(PRG1) $B
	cp $(PRG2) $B
	cp $(PRG4) $B

clean:
		rm -f core $(OBJC) *_mpi.o fileOpen.o writeRec.o test_fdfwimodc.o test_fwi_gradient.o fwi_driver.o fwi_driver_mpi.o test_wave_op_dp.o test_dotproduct.o $(PRG1) $(PRG2) $(PRG3) $(PRG4) $(PRG5) $(PRG6)

realclean:
		rm -f core $(OBJC) fileOpen.o writeRec.o test_fdfwimodc.o test_fwi_gradient.o fwi_driver.o $(PRG1) $(PRG2) $(PRG3) $B/$(PRG1) $B/$(PRG2) $B/$(PRG3)
		rm -f *_mpi.o fwi_driver_mpi.o $(PRG4) $B/$(PRG4)

print:	Makefile $(SRC)
	$(PRINT) $?
	@touch print

count:
	@wc $(SRC)

tar:
	@tar cf $(PRG1).tar Makefile $(SRC) && compress $(PRG1).tar
