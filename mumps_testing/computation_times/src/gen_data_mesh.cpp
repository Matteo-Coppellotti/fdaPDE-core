#include <chrono>
#include <filesystem>
#include <fstream>

#include "../../../fdaPDE/fields.h"
#include "../../../fdaPDE/finite_elements.h"
#include "../../../fdaPDE/geoframe/csv.h"
#include "../../../fdaPDE/geometry.h"
#include "../../../fdaPDE/linear_algebra/mumps.h"
#include "utils/mesh_generation.h"

using namespace Eigen;
using namespace fdapde;
using namespace fdapde::mumps;
using namespace std::chrono;

void printSparsityPattern(const SparseMatrix<double>& A, std::string filename);
void saveCSV(std::string input_filename, std::string output_filename);

int main() {
    /**
     * GENERATE SCHUR TIMES?
     */
    bool generate_schur_times = true;

    /**
     * SET THE OUTPUT DIRECTORY
     */
    std::string output_directory = "../data";

    /**
     * SET THE TEMPORARY FILENAME
     *
     * Make sure that the name in not already used as it will be overwritten.
     */
    std::string temp_filename = "temp.csv";

    /**
     * SET THE OUTPUT FILENAMES
     */
    std::string output_filename = "computation_times.csv";
    std::string output_filename_spd = "computation_times_spd.csv";
    std::string output_filename_schur = "computation_times_schur.csv";

    /**
     * SET THE PROBLEM SIZE
     */
    std::vector<int> N = {10, 100, 1000, 10000, 100000, 1000000};

    /**
     * SET THE SOLVERS TO BE USED
     *
     * Available solvers:
     * - SparseLU
     * - MumpsLU
     * - MumpsBLR
     *
     * Available symmetric solvers:
     * - SimplicialLLT
     * - MumpsLDLT
     */
    std::vector<std::string> solvers = {"SparseLU", "MumpsLU", "MumpsBLR"};
    std::vector<std::string> solvers_spd = {"SimplicialLLT", "MumpsLDLT"};

    /**
     * SET THE NUMBER OF ITERATIONS
     */
    int n_iter = 100;

    /**
     * SET THE PROBLEMS TO BE SOLVED
     *
     * Available problems:
     * - mass (SPD)
     * - laplacian (SPD)
     * - diffusion_transport (non-symmetric)
     */
    std::vector<std::string> problems = {"mass", "laplacian", "diffusion_transport"};
    std::vector<std::string> problems_spd = {"mass", "laplacian"};

    // data structures
    std::vector<SparseMatrix<double>> mass_matrices;
    std::vector<SparseMatrix<double>> laplacian_matrices;
    std::vector<SparseMatrix<double>> diffusion_transport_matrices;
    std::vector<VectorXd> forces;

    // MPI initialization + rank (for output)
    MPI_Init(NULL, NULL);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) std::cout << "MPI initialized with" << size << "processes." << std::endl;

    // create output directory
    if (rank == 0) {
        if (!std::filesystem::exists(output_directory)) { std::filesystem::create_directory(output_directory); }
    }

    // fill data structures
    for (int n : N) {
        /**
         * Available mesh generators:
         * - meshInterval(a, b, n_nodes)
         * - meshUnitInterval(n_nodes)
         * - meshRectangle(a_x, b_x, a_y, b_y, n_nodes_x, n_nodes_y)
         * - meshSquare(a, b, n_nodes)
         * - meshUnitSquare(n_nodes)
         * - meshParallelepipiped(a_x, b_x, a_y, b_y, a_z, b_z, n_nodes_x, n_nodes_y, n_nodes_z)
         * - meshCube(a, b, n_nodes)
         * - meshUnitCube(n_nodes)
         */
        auto mesh = meshUnitSquare(n);

        if (rank == 0) std::cout << "Generated mesh with " << n << " nodes" << std::endl;

        Triangulation</* tangent_space = */ 2, /* embedding_space = */ 2> unit_square(
          mesh.nodes, mesh.elements, mesh.boundary);

        FeSpace Vh(unit_square, P1<1>);
        TrialFunction u(Vh);
        TestFunction v(Vh);
        if (rank == 0) {
            auto a1 = integral(unit_square)(u * v);                   // mass matrix: SPD
            auto a2 = integral(unit_square)(dot(grad(u), grad(v)));   // laplacian discretization (Poisson): SPD

            Matrix<double, 2, 1> b(0.2, 0.2);

            auto a3 = integral(unit_square)(
              dot(grad(u), grad(v)) + dot(b, grad(u)) * v);   // diffusion-transport: non simmetrica

            mass_matrices.push_back(SparseMatrix<double>(a1.assemble()));
            laplacian_matrices.push_back(SparseMatrix<double>(a2.assemble()));
            diffusion_transport_matrices.push_back(SparseMatrix<double>(a3.assemble()));
            if (rank == 0) std::cout << "Generated matrices for " << n << " nodes" << std::endl;

            // linear system: R1 * u = f
        } else {
            mass_matrices.push_back(SparseMatrix<double>(n, n));
            laplacian_matrices.push_back(SparseMatrix<double>(n, n));
            diffusion_transport_matrices.push_back(SparseMatrix<double>(n, n));
        }
        // forcing term
        ScalarField<2, decltype([]([[maybe_unused]] const Vector2d& p) { return 1; })> f;
        auto F = integral(unit_square)(f * v);

        forces.push_back(VectorXd(F.assemble()));
        if (rank == 0) std::cout << "Generated force for " << n << " nodes" << std::endl;
    }

    // print sparsity patterns of the matrices
    if (rank == 0) {
        std::cout << "Printing sparsity patterns" << std::endl;
        std::string sp_patterns_directory = output_directory + "/sparsity_patterns";
        if (!std::filesystem::exists(sp_patterns_directory)) {
            std::filesystem::create_directory(sp_patterns_directory);
        }
        if (!std::filesystem::exists(sp_patterns_directory + "/mass")) {
            std::filesystem::create_directory(sp_patterns_directory + "/mass");
        }
        if (!std::filesystem::exists(sp_patterns_directory + "/laplacian")) {
            std::filesystem::create_directory(sp_patterns_directory + "/laplacian");
        }
        if (!std::filesystem::exists(sp_patterns_directory + "/diffusion_transport")) {
            std::filesystem::create_directory(sp_patterns_directory + "/diffusion_transport");
        }
        for (int i = 0; i < N.size(); ++i) {
            printSparsityPattern(
              mass_matrices[i], sp_patterns_directory + "/mass/mass_" + std::to_string(N[i]) + ".csv");
            printSparsityPattern(
              laplacian_matrices[i], sp_patterns_directory + "/laplacian/laplacian_" + std::to_string(N[i]) + ".csv");
            printSparsityPattern(
              diffusion_transport_matrices[i],
              sp_patterns_directory + "/diffusion_transport/diffusion_transport_" + std::to_string(N[i]) + ".csv");
        }
    }

    // file manager for output
    std::ofstream file;
    std::string temp_path = output_directory + "/" + temp_filename;
    std::string output_path = output_directory + "/" + output_filename;
    std::string output_path_spd = output_directory + "/" + output_filename_spd;
    std::string output_path_schur = output_directory + "/" + output_filename_schur;

    if (rank == 0) {
        std::cout << "Solving problems" << std::endl;
        file.open(temp_path);
        if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
        file << "N,problem,solver,time\n";
    }
    for (size_t i = 0; i < N.size(); ++i) {
        SparseMatrix<double> A;
        for (const auto& problem : problems) {
            if (problem == "mass") { A = mass_matrices[i]; }
            if (problem == "laplacian") { A = laplacian_matrices[i]; }
            if (problem == "diffusion_transport") { A = diffusion_transport_matrices[i]; }
            for (const auto& solver : solvers) {
                for (int iter = 0; iter < n_iter; ++iter) {
                    auto t1 = high_resolution_clock::now();
                    if (solver == "SparseLU" && rank == 0) {
                        Eigen::SparseLU<SparseMatrix<double>> solver(A);
                        VectorXd x = solver.solve(forces[i]);
                    }
                    if (solver == "MumpsLU") {
                        MumpsLU solver(A);
                        VectorXd x = solver.solve(forces[i]);
                    }
                    if (solver == "MumpsBLR") {
                        MumpsBLR solver(A);
                        VectorXd x = solver.solve(forces[i]);
                    }
                    auto t2 = high_resolution_clock::now();
                    duration<double> duration = t2 - t1;
                    MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                    if (rank == 0) {
                        std::cout << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                        file << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                    }
                }
            }
        }
    }
    if (rank == 0) {
        file.close();
        saveCSV(temp_path, output_path);
    }

    if (rank == 0) {
        std::cout << "Solving SPD problems" << std::endl;
        file.open(temp_path);
        if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
        file << "N,problem,solver,time\n";
    }
    for (size_t i = 0; i < N.size(); ++i) {
        SparseMatrix<double> A;
        for (const auto& problem : problems_spd) {
            if (problem == "mass") { A = mass_matrices[i]; }
            if (problem == "laplacian") { A = laplacian_matrices[i]; }
            for (const auto& solver : solvers_spd) {
                for (int iter = 0; iter < n_iter; ++iter) {
                    auto t1 = high_resolution_clock::now();
                    if (solver == "SimplicialLLT" && rank == 0) {
                        Eigen::SimplicialLLT<SparseMatrix<double>> solver(A);
                        VectorXd x = solver.solve(forces[i]);
                    }
                    if (solver == "MumpsLDLT") {
                        MumpsLDLT solver(A);
                        VectorXd x = solver.solve(forces[i]);
                    }
                    auto t2 = high_resolution_clock::now();
                    duration<double> duration = t2 - t1;
                    MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                    if (rank == 0) {
                        std::cout << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                        file << N[i] << "," << problem << "," << solver << "," << duration.count() << "\n";
                    }
                }
            }
        }
    }
    if (rank == 0) {
        file.close();
        saveCSV(temp_path, output_path_spd);
    }

    if (generate_schur_times) {
        if (rank == 0) {
            std::cout << "Solving Schur problems" << std::endl;
            file.open(temp_path);
            if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
            file << "N,problem,schur_size,time\n";
        }

        for (size_t i = 0; i < N.size(); ++i) {
            for (const auto& problem : problems) {
                for (int schur_size = N[i] / 10; schur_size < N[i]; schur_size += N[i] / 10) {   // change step ???
                    for (int iter = 0; iter < n_iter; ++iter) {
                        auto t1 = high_resolution_clock::now();
                        MumpsSchur solver(mass_matrices[i], schur_size);
                        VectorXd x = solver.solve(forces[i]);
                        auto t2 = high_resolution_clock::now();
                        duration<double> duration = t2 - t1;
                        MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                        if (rank == 0) {
                            std::cout << N[i] << "," << problem << "," << schur_size << "," << duration.count() << "\n";
                            file << N[i] << "," << problem << "," << schur_size << "," << duration.count() << "\n";
                        }
                    }
                }
            }
        }
        if (rank == 0) {
            file.close();
            saveCSV(temp_path, output_path_schur);
        }
    }

    // delete temporary file
    if (rank == 0) std::filesystem::remove(temp_path);

    MPI_Finalize();
    return 0;
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
    file.close();
}

void saveCSV(std::string input_filename, std::string output_filename) {
    std::ifstream input_file(input_filename);
    std::ofstream output_file(output_filename);
    if (!input_file.is_open()) { throw std::runtime_error("Unable to open file for reading."); }
    if (!output_file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
    output_file << input_file.rdbuf();
    output_file.close();
}