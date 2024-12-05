#include <gtest/gtest.h>

// #include <unsupported/Eigen/SparseExtra>

#include "../../../fdaPDE/linear_algebra/mumps.h"
// #include "utils/rand_indices.h"

using namespace fdapde::mumps;
using namespace Eigen;

// #include "../../test/src/utils/constants.h"
// using fdapde::testing::DOUBLE_TOLERANCE;

#include <chrono>
#include <fstream>
#include <random>
using namespace std::chrono;

constexpr bool Seeded = true;
constexpr int Seed = 42;

SparseMatrix<double> generateFullRankMatrix(int size, double density, std::mt19937& gen) {
    SparseMatrix<double> A(size, size);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::vector<Triplet<double>> tripletList;
    int nnz = static_cast<int>(density * size * size);   // Approximate non-zeros
    for (int k = 0; k < nnz; ++k) {
        int i = gen() % size;
        int j = gen() % size;
        double value = dis(gen);
        tripletList.push_back(Triplet<double>(i, j, value));
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // Ensure full rank by adding a small value to the diagonal
    for (int i = 0; i < size; ++i) { A.coeffRef(i, i) += 1e-5; }

    return A;
}

SparseMatrix<double> generateSpdMatrix(int size, double density, std::mt19937& gen) {
    SparseMatrix<double> A(size, size);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::vector<Triplet<double>> tripletList;
    int nnz = static_cast<int>(density * size * size / 2);   // Approximate non-zeros
    for (int k = 0; k < nnz; ++k) {
        int i = gen() % size;
        int j = gen() % size;
        double value = dis(gen);
        if (i <= j) {
            tripletList.push_back(Triplet<double>(i, j, value));
            if (i != j) {
                tripletList.push_back(Triplet<double>(j, i, value));   // Ensure symmetry
            }
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // Make it symmetric positive definite by adding a multiple of the identity (large enough to ensure positive
    // definiteness)
    for (int i = 0; i < size; ++i) { A.coeffRef(i, i) += size; }

    return A;
}

void printSparsityPattern(const SparseMatrix<double>& A, std::string filename) {
    std::ofstream file(filename);
    file << "i,j\n";
    if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            if (A.coeff(i, j) != 0) { file << i << "," << j << "\n"; }
        }
    }
}

void saveCSV(std::string dst_name) {
    std::ifstream src("../data/temp.csv");
    std::ofstream dst(dst_name);
    dst << src.rdbuf();
    dst.close();
}

int main(int argc, char* argv[]) {
    std::ofstream file;
    std::string filename = "../data/temp.csv";
    int Iterations = 1;
    std::vector<int> N = {10, 50, 100, 500, 1000, 5000};
    std::vector<double> dens = {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005};

    std::vector<std::string> solvers = {"SparseLU", "MumpsLU", "MumpsBLR"};
    std::vector<std::string> spd_solvers = {"SimplicialLLT", "MumpsLDLT"};

    SparseMatrix<double> A;
    VectorXd x, b;

    std::mt19937 gen;
    if (Seeded) {
        gen.seed(Seed);
    } else {
        gen.seed(std::random_device()());
    }

    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix<SparseMatrix<double>, Dynamic, Dynamic> Matrices(N.size(), dens.size());
    for (size_t i = 0; i < N.size(); ++i) {
        for (size_t j = 0; j < dens.size(); ++j) {
            Matrices(i, j) = generateFullRankMatrix(N[i], dens[j], gen);
            if (rank == 0) {
                std::cout << "Generated matrix of size " << N[i] << "x" << N[i] << " with density " << dens[j] << "\n";
                printSparsityPattern(
                  Matrices(i, j), "../data/sparsity_patterns/sparsity_pattern_" + std::to_string(N[i]) + "_" +
                                    std::to_string(dens[j]) + ".csv");
            }
        }
    }

    Matrix<SparseMatrix<double>, Dynamic, Dynamic> MatricesSpd(N.size(), dens.size());
    for (size_t i = 0; i < N.size(); ++i) {
        for (size_t j = 0; j < dens.size(); ++j) {
            MatricesSpd(i, j) = generateSpdMatrix(N[i], dens[j], gen);
            if (rank == 0) {
                std::cout << "Generated SPD matrix of size " << N[i] << "x" << N[i] << " with density " << dens[j]
                          << "\n";
                printSparsityPattern(
                  MatricesSpd(i, j), "../data/spd_sparsity_patterns/spd_sparsity_pattern_" + std::to_string(N[i]) +
                                       "_" + std::to_string(dens[j]) + ".csv");
            }
        }
    }

    // file.open(filename);
    // if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
    // file << "Solver,N,density,time\n";

    // for (auto solver : solvers) {
    //     for (size_t i = 0; i < N.size(); ++i) {
    //         for (size_t j = 0; j < dens.size(); ++j) {
    //             A = Matrices(i, j);
    //             b = VectorXd::Ones(N[i]);

    //             for (int n = 0; n < Iterations; ++n) {
    //                 auto t1 = high_resolution_clock::now();
    //                 if (solver == "MumpsLU") {
    //                     MumpsLU mumps(A);
    //                     x = mumps.solve(b);
    //                 }
    //                 if (solver == "MumpsBLR") {
    //                     MumpsBLR mumps(A);
    //                     x = mumps.solve(b);
    //                 }
    //                 if (solver == "SparseLU") {
    //                     SparseLU<SparseMatrix<double>> solver(A);
    //                     x = solver.solve(b);
    //                 }
    //                 auto t2 = high_resolution_clock::now();
    //                 duration<double> duration = t2 - t1;
    //                 MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    //                 if (rank == 0) {
    //                     std::cout << solver << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
    //                     file << solver << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
    //                 }
    //             }
    //         }
    //     }
    // }
    // file.close();

    // saveCSV("../data/data.csv");

    // file.open(filename);
    // if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
    // file << "Solver,N,density,time\n";

    // for (auto solver : spd_solvers) {
    //     for (size_t i = 0; i < N.size(); ++i) {
    //         for (size_t j = 5; j < 6/*dens.size()*/; ++j) {
    //             A = MatricesSpd(i, j);
    //             b = VectorXd::Ones(N[i]);

    //             for (int n = 0; n < Iterations; ++n) {
    //                 auto t1 = high_resolution_clock::now();
    //                 if (solver == "MumpsLDLT") {
    //                     MumpsLDLT mumps(A);
    //                     // mumps.mumpsIcntl()[13] = 100;
    //                     x = mumps.solve(b);
    //                 }
    //                 if (solver == "SimplicialLLT") {
    //                     SimplicialLLT<SparseMatrix<double>> solver(A);
    //                     x = solver.solve(b);
    //                 }
    //                 auto t2 = high_resolution_clock::now();
    //                 duration<double> duration = t2 - t1;
    //                 MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    //                 if (rank == 0) {
    //                     std::cout << solver << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
    //                     file << solver << "," << N[i] << "," << dens[j] << "," << duration.count() << "\n";
    //                 }
    //             }
    //         }
    //     }
    // }
    // file.close();

    // saveCSV("../data/spd_data.csv");

    file.open(filename);
    if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
    file << "Solver,N,density,schur_size,time\n";

    std::string schur_solver = "MumpsSchur";
    for (size_t i = 0; i < N.size(); ++i) {
        std::vector<int> schur_size;
        schur_size.reserve(i + 1);
        schur_size.push_back(N[i] / 5);
        for (int k = 0; k <= i; ++k) { schur_size.push_back(N[i] / N[k]); }
        for (size_t j = 0; j < 1/*dens.size()*/; ++j) {
            A = Matrices(i, j);
            b = VectorXd::Ones(N[i]);

            for (int k = 0; k < schur_size.size(); ++k) {
                for (int n = 0; n < Iterations; ++n) {
                    auto t1 = high_resolution_clock::now();
                    MumpsSchur mumps(A, schur_size[k], WorkingHost);
                    x = mumps.solve(b);
                    auto t2 = high_resolution_clock::now();
                    duration<double> duration = t2 - t1;
                    MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                    if (rank == 0) {
                        std::cout << schur_solver << "," << N[i] << "," << dens[j] << "," << schur_size[k] << "," << duration.count() << "\n";
                        file << schur_solver << "," << N[i] << "," << dens[j] << "," << schur_size[k] << "," << duration.count() << "\n";
                    }
                }
            }
        }
    }
    file.close();

    saveCSV("../data/schur_data.csv");

    MPI_Finalize();
    return 0;
}
