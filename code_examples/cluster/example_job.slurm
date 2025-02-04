#!/bin/bash

#SBATCH --job-name=example_job
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --partition=g100_usr_prod
#SBATCH --mem=1GB

WORKING_DIR=/g100/home/userexternal/mcoppell/mumps/fdaPDE-core/code_examples/cluster/build
cd $WORKING_DIR

export LD_LIBRARY_PATH="$(spack location -i mumps)/lib:$(spack location -i openmpi)/lib:$(spack location -i netlib-scalapack)/lib:$(spack location -i netlib-lapack)/lib64:$(spack location -i scotch)/lib:$(spack location -i metis)/lib:$(spack location -i parmetis)/lib"

mpirun ./code_example
