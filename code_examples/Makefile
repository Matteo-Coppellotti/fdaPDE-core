# Make sure to copy the Makefile.inc file used for the mumps installation in the Makefile.inc file in this directory
include Makefile.inc

EIGENDIR = /home/matteo/Desktop/eigen
MUMPSDIR = /home/matteo/Desktop/MUMPS-openMPI-noBLACS
CXXDIR = /usr/lib/gcc/x86_64-linux-gnu/13

INCMUMPS = $(MUMPSDIR)/include
LIBSDMUMPS = -L$(MUMPSDIR)/lib -ldmumps -lmumps_common
LIBCXX = -L$(CXXDIR) -lstdc++

BUILDDIR = build_makefile

##############################################################################

.PHONY: all clean run mk_dir

.DEFAULT_GOAL: all

all: example_code

example_code: example_code.o
	mpif77 -o $(BUILDDIR)/example_code $(OPTL) $(BUILDDIR)/example_code.o $(LIBSDMUMPS) $(LORDERINGS) $(LIBS) $(LIBBLAS) $(LIBOTHERS) $(LIBCXX) -lmpi_cxx

example_code.o: mk_dir src/example_code.cpp
	mpicxx -std=c++20 $(OPTC) $(CDEFS) -I$(EIGENDIR) -I$(INCMUMPS) -c src/example_code.cpp $(OUTC) $(BUILDDIR)/example_code.o 

run: all
	mpirun --use-hwthread-cpus $(BUILDDIR)/example_code

mk_dir:
	mkdir -p $(BUILDDIR)

clean:
	$(RM) -r $(BUILDDIR)