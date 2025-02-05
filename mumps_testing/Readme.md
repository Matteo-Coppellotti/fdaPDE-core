# Running tests
It is possible to run a batch of tests to ensure that the wrapper is working correctly by following these instructions.  

## Required libraries
To run the tests make sure to have the `gtest` library installed (hosted at [GitHub](https://github.com/google/googletest/)).

## Configuration
The tests use Cmake for compiling, it is required that the user chaanges the contents of the `user_config.cmake` file according to the paths used in the `Makefile.inc` file used in the MUMPS library installation.

## Running the tests
After configuration, it is possible to run the tests by using the following script:
```
./run_tests.sh
```
This will run a comprehensive set of tests, the terminal output will be also saved in a `test_output.log` file.