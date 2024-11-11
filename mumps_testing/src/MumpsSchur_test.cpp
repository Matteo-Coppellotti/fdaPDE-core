#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <set>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

#include "../../fdaPDE/linear_algebra/mumps.h"

using namespace fdapde::mumps;
using namespace Eigen;

#include "../../test/src/utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;

constexpr bool Seeded = true;
constexpr int Seed = 42;

void randomIndices(std::vector<int>& vec, int rows) {
    std::mt19937 gen;
    if (Seeded) {
        gen.seed(Seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::uniform_int_distribution<> dis(0, rows - 1);

    // Determine the random number of elements to add to the vector (between 1 and rows)
    std::uniform_int_distribution<> sizeDis(1, rows / 10);   // Random number of elements from 1 to rows/10
    int numElements = sizeDis(gen);

    std::set<int> uniqueNumbers;   // Set to ensure uniqueness

    // Fill the set with unique numbers
    while (uniqueNumbers.size() < numElements) { uniqueNumbers.insert(dis(gen)); }

    // Convert the set to a vector (which will automatically be sorted in increasing order)
    vec.assign(uniqueNumbers.begin(), uniqueNumbers.end());
}

TEST(MumpsSchur_test, split_analyze_factorize) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    MumpsSchur<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

    std::vector<int> schur_indices;
    randomIndices(schur_indices, A.rows());
    solver.setSchurIndices(schur_indices);

    solver.analyzePattern(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());

    solver.factorize(A);
    EXPECT_TRUE(solver.info() == Success);
    double det = solver.determinant();
    EXPECT_TRUE(det != 0);

    VectorXd x, b;
    b = VectorXd::Ones(A.rows());
    x = solver.solve(b);

    MatrixXd X, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X = solver.solve(B);

    MatrixXd complement = solver.complement();
    EXPECT_TRUE(complement.rows() == schur_indices.size());
    EXPECT_TRUE(complement.cols() == schur_indices.size());
}

TEST(MumpsSchur_test, compute) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    MumpsSchur<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

    std::vector<int> schur_indices;
    randomIndices(schur_indices, A.rows());
    solver.setSchurIndices(schur_indices);

    solver.compute(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();
    EXPECT_TRUE(det != 0);

    VectorXd x, b;
    b = VectorXd::Ones(A.rows());
    x = solver.solve(b);

    MatrixXd X, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X = solver.solve(B);

    MatrixXd complement = solver.complement();
    EXPECT_TRUE(complement.rows() == schur_indices.size());
    EXPECT_TRUE(complement.cols() == schur_indices.size());
}

TEST(MumpsSchur_test, type_deduction) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    std::vector<int> schur_indices;
    randomIndices(schur_indices, A.rows());

    MumpsSchur solver(A, schur_indices);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();
    EXPECT_TRUE(det != 0);

    VectorXd x, b;
    b = VectorXd::Ones(A.rows());
    x = solver.solve(b);

    MatrixXd X, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X = solver.solve(B);

    MatrixXd complement = solver.complement();
    EXPECT_TRUE(complement.rows() == schur_indices.size());
    EXPECT_TRUE(complement.cols() == schur_indices.size());
}

TEST(MumpsSchur_test, flags) {
    MumpsSchur<SparseMatrix<double>> solver(NoDeterminant);
    EXPECT_TRUE(solver.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver.mumpsRawStruct().par == 0);

    MumpsSchur<SparseMatrix<double>> solver2(Verbose);
    if (solver2.getProcessRank() == 0) {
        EXPECT_TRUE(solver2.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver2.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver2.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver2.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver2.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver2.mumpsRawStruct().par == 0);

    MumpsSchur<SparseMatrix<double>> solver3(WorkingHost);
    EXPECT_TRUE(solver3.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver3.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver3.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver3.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver3.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver3.mumpsRawStruct().par == 1);

    MumpsSchur<SparseMatrix<double>> solver4(NoDeterminant | Verbose | WorkingHost);
    if (solver4.getProcessRank() == 0) {
        EXPECT_TRUE(solver4.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver4.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver4.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver4.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver4.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver4.mumpsRawStruct().par == 1);

    MumpsSchur<SparseMatrix<double>> solver5(NoDeterminant | Verbose);
    if (solver5.getProcessRank() == 0) {
        EXPECT_TRUE(solver5.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver5.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver5.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver5.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver5.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver5.mumpsRawStruct().par == 0);

    MumpsSchur<SparseMatrix<double>> solver6(NoDeterminant | WorkingHost);
    EXPECT_TRUE(solver6.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver6.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver6.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver6.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver6.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver6.mumpsRawStruct().par == 1);

    MumpsSchur<SparseMatrix<double>> solver7(Verbose | WorkingHost);
    if (solver7.getProcessRank() == 0) {
        EXPECT_TRUE(solver7.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver7.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver7.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver7.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver7.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver7.mumpsRawStruct().par == 1);

    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");
    std::vector<int> schur_indices;
    randomIndices(schur_indices, A.rows());

    MumpsSchur solver8(A, schur_indices, NoDeterminant);
    EXPECT_TRUE(solver8.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver8.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver8.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver8.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver8.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver8.mumpsRawStruct().par == 0);

    MumpsSchur solver9(A, schur_indices, Verbose);
    if (solver9.getProcessRank() == 0) {
        EXPECT_TRUE(solver9.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver9.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver9.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver9.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver9.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver9.mumpsRawStruct().par == 0);

    MumpsSchur solver10(A, schur_indices, WorkingHost);
    EXPECT_TRUE(solver10.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver10.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver10.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver10.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver10.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver10.mumpsRawStruct().par == 1);

    MumpsSchur solver11(A, schur_indices, NoDeterminant | Verbose | WorkingHost);
    if (solver11.getProcessRank() == 0) {
        EXPECT_TRUE(solver11.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver11.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver11.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver11.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver11.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver11.mumpsRawStruct().par == 1);

    MumpsSchur solver12(A, schur_indices, NoDeterminant | Verbose);
    if (solver12.getProcessRank() == 0) {
        EXPECT_TRUE(solver12.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver12.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver12.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver12.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver12.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver12.mumpsRawStruct().par == 0);

    MumpsSchur solver13(A, schur_indices, NoDeterminant | WorkingHost);
    EXPECT_TRUE(solver13.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver13.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver13.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver13.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver13.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver13.mumpsRawStruct().par == 1);

    MumpsSchur solver14(A, schur_indices, Verbose | WorkingHost);
    if (solver14.getProcessRank() == 0) {
        EXPECT_TRUE(solver14.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver14.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver14.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver14.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver14.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver14.mumpsRawStruct().par == 1);
}

TEST(MumpsSchur_test, colmajor_vs_rowmajor) {
    SparseMatrix<double> A_colmajor;
    SparseMatrix<double, RowMajor> A_rowmajor;
    Eigen::loadMarket(A_colmajor, "../data/matrix_fullrank.mtx");
    Eigen::loadMarket(A_rowmajor, "../data/matrix_fullrank.mtx");

    std::vector<int> schur_indices;
    randomIndices(schur_indices, A_colmajor.rows());

    MumpsSchur solver_colmajor(A_colmajor, schur_indices);
    MumpsSchur solver_rowmajor(A_rowmajor, schur_indices);

    EXPECT_TRUE(solver_colmajor.info() == Success);
    EXPECT_TRUE(solver_rowmajor.info() == Success);
    EXPECT_TRUE(solver_colmajor.rows() == A_colmajor.rows());
    EXPECT_TRUE(solver_rowmajor.rows() == A_rowmajor.rows());
    EXPECT_TRUE(solver_colmajor.cols() == A_colmajor.cols());
    EXPECT_TRUE(solver_rowmajor.cols() == A_rowmajor.cols());
    double det_colmajor = solver_colmajor.determinant();
    double det_rowmajor = solver_rowmajor.determinant();
    EXPECT_TRUE(det_colmajor != 0);
    EXPECT_TRUE(det_rowmajor != 0);

    VectorXd x_colmajor, x_rowmajor, b;
    b = VectorXd::Ones(A_colmajor.rows());
    x_colmajor = solver_colmajor.solve(b);
    x_rowmajor = solver_rowmajor.solve(b);

    Matrix<double, Dynamic, Dynamic> X_colmajor_colcol, X_colmajor_rowrow, X_colmajor_colrow, X_colmajor_rowcol,
      B_colmajor;
    Matrix<double, Dynamic, Dynamic, RowMajor> X_rowmajor_colcol, X_rowmajor_rowrow, X_rowmajor_colrow,
      X_rowmajor_rowcol, B_rowmajor;
    B_colmajor = MatrixXd::Ones(A_colmajor.rows(), 3);
    B_rowmajor = MatrixXd::Ones(A_rowmajor.rows(), 3);
    X_colmajor_colcol = solver_colmajor.solve(B_colmajor);
    X_colmajor_rowrow = solver_rowmajor.solve(B_rowmajor);
    X_colmajor_colrow = solver_colmajor.solve(B_rowmajor);
    X_colmajor_rowcol = solver_rowmajor.solve(B_colmajor);
    X_rowmajor_colcol = solver_colmajor.solve(B_colmajor);
    X_rowmajor_rowrow = solver_rowmajor.solve(B_rowmajor);
    X_rowmajor_colrow = solver_colmajor.solve(B_rowmajor);
    X_rowmajor_rowcol = solver_rowmajor.solve(B_colmajor);

    EXPECT_TRUE(X_colmajor_colcol.isApprox(X_colmajor_colrow, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_colmajor_colrow.isApprox(X_colmajor_rowcol, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_colmajor_rowcol.isApprox(X_colmajor_rowrow, DOUBLE_TOLERANCE));

    EXPECT_TRUE(X_rowmajor_colcol.isApprox(X_rowmajor_colrow, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_rowmajor_colrow.isApprox(X_rowmajor_rowcol, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_rowmajor_rowcol.isApprox(X_rowmajor_rowrow, DOUBLE_TOLERANCE));

    EXPECT_TRUE(X_colmajor_colcol.isApprox(X_rowmajor_colcol, DOUBLE_TOLERANCE));

    Matrix<double, Dynamic, Dynamic> complement_colmajor_col = solver_colmajor.complement();
    Matrix<double, Dynamic, Dynamic> complement_colmajor_row = solver_rowmajor.complement();
    EXPECT_TRUE(complement_colmajor_col.rows() == schur_indices.size());
    EXPECT_TRUE(complement_colmajor_col.cols() == schur_indices.size());
    EXPECT_TRUE(complement_colmajor_row.rows() == schur_indices.size());
    EXPECT_TRUE(complement_colmajor_row.cols() == schur_indices.size());
    EXPECT_TRUE(complement_colmajor_col.isApprox(complement_colmajor_row, DOUBLE_TOLERANCE));

    Matrix<double, Dynamic, Dynamic, RowMajor> complement_rowmajor_col = solver_colmajor.complement();
    Matrix<double, Dynamic, Dynamic, RowMajor> complement_rowmajor_row = solver_rowmajor.complement();
    EXPECT_TRUE(complement_rowmajor_col.rows() == schur_indices.size());
    EXPECT_TRUE(complement_rowmajor_col.cols() == schur_indices.size());
    EXPECT_TRUE(complement_rowmajor_row.rows() == schur_indices.size());
    EXPECT_TRUE(complement_rowmajor_row.cols() == schur_indices.size());
    EXPECT_TRUE(complement_rowmajor_col.isApprox(complement_rowmajor_row, DOUBLE_TOLERANCE));
}

TEST(MumpsSchur_test, icntl_check) {
    MumpsSchur<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.mumpsIcntl()[18] == 3);
}
