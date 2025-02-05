# Using the MUMPS wrapper
In order to use the mumps wrapper the following steps are needed:  
* Install the MUMPS library (can be found [here](https://mumps-solver.org/index.php)) along with its dependancies.
* All required fdaPDE core dependencies can be found in the `README.md`.

## Code examples
Compilation examples can be found in the `code_examples` directory. Make sure to follow the instructions provided in the `Readme.md` file present there. 
In this directory a subdirectory named `cluster` contains a compilation example for use on the G100@Cineca HPC cluster (istructions in a `Readme.md` file).

## Code tests
In the `mumps_testing` directory it is possible to test the correctness of the wrapper installation. Follow the instructions in the `Readme.md` file found there.