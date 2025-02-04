# Example code
In this directory an example code using the MUMPS wrapper can be found. With it there are an example of a makefile and an example of a cmake needed for compilation. Here are some instructions to follow to configure them correctly for you MUMPS installation. Source installation for the MUMPS library is assumed (according to the project report installation guide).

## Makefile
To configure the Makefile correctly first copy the `Makefile.inc` file used for the MUMPS source installation in this directory, then open the `Makefile` and change the following path variables according to the user setup: `EIGENDIR`, `MUMPSDIR`, `CXXDIR`.  

To compile the code simply run in the terminal:
```
make
```
To run the example code use:
```
make run
```  

## Cmake
To configure the cmake the user should change the path vriables declared in the `user_config.cmake` file to match the ones used in the MUMPS source installation and according to the user setup.  

To compile the code use the bash script `compile_code.sh`:
```
./compile_code.sh
```
To run the code example use the bash script `run_code.sh` (it is possible to select the numer of MPI processes by adding the desired number as a script argument, if omitted the flag `--use-hwthread-cpus` will be used):
```
./run_code.sh
```
or
```
./run_code.sh 4
```