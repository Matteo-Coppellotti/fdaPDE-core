LSCOTCHDIR = /usr/lib
LSCOTCH   = -L$(LSCOTCHDIR) -lptesmumps -lptscotch -lptscotcherr

PORDDIR = $(HOME)/Desktop/MUMPS-openMPI-noBLACS/PORD
LPORDDIR = $(PORDDIR)/lib/
LPORD    = -L$(LPORDDIR) -lpord

LMETISDIR = /usr/lib 
LMETIS    = -L$(LMETISDIR) -lparmetis -lmetis

LORDERINGS = $(LMETIS) $(LPORD) $(LSCOTCH)

LIBOTHERS = -lpthread

LIBCXX = -L/usr/lib/gcc/x86_64-linux-gnu/13/ -lstdc++


mkdir build
cd build

g++ -std=c++23 -O -fopenmp -c src/gen_data_mesh.cpp -o mumps_h_test.o -I$(spack location -i eigen)/include/eigen3 -I$(spack location -i mumps)/include -I$(spack location -i openmpi)/include
gfortran -o mumps_h_test -O -fopenmp mumps_h_test.o -L$(spack locaton -i mumps)/lib -ldmumps -lmumps_common -L$(spack location -i gcc)/lib64 -lstdc++ -L$(spack location -i openmpi)/lib -lmpifort -lmpi -lmpi_cxx 