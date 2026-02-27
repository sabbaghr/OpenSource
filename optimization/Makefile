# Makefile for optimization library (C translation of SEISCOPE Toolbox)

include ../Make_include

ALLINC = -I.

SRCS = optim_common.c lbfgs.c steepest_descent.c pnlcg.c plbfgs.c trn.c enriched.c
OBJS = $(SRCS:%.c=%.o)
LIB  = liboptim.a

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJS)

test: test_rosenbrock
	./test_rosenbrock

test_rosenbrock: test_rosenbrock.o $(LIB)
	$(CC) $(LDFLAGS) $(CFLAGS) $(OPTC) -o $@ test_rosenbrock.o $(LIB) -lm

%.o: %.c optim.h
	$(CC) $(CFLAGS) $(OPTC) $(ALLINC) -c $<

clean:
	rm -f *.o $(LIB) test_rosenbrock iterate_*.dat

.PHONY: all test clean
