#!/bin/bash

#SBATCH --job-name=fdaPDE_mumps_mesh
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=4
#SBATCH --partition=g100_usr_prod
#SBATCH --mail-user=matteo.coppellotti@mail.polimi.it
#SBATCH --mail-type=ALL
#SBATCH --output=gen_data_mesh.out

WORKING_DIR=/g100/home/userexternal/mcoppell/mumps/fdaPDE-core/mumps_testing/computation_times/build
cd $WORKING_DIR

export LD_LIBRARY_PATH="$(spack location -i mumps)/lib:$(spack location -i openmpi)/lib:$(spack location -i netlib-scalapack)/lib:$(spack location -i netlib-lapack)/lib64:$(spack location -i scotch)/lib:$(spack location -i metis)/lib:$(spack location -i parmetis)/lib"

mpirun ./gen_data_mesh
