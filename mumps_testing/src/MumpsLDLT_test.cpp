#include <gtest/gtest.h>

#include <unsupported/Eigen/SparseExtra>

#include "../../fdaPDE/linear_algebra/mumps.h"
#include "utils/rand_indices.h"

using namespace fdapde::mumps;
using namespace Eigen;

#include "../../test/src/utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;

TEST(MumpsLDLT_test, split_analyze_factorize) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_spd.mtx");

    MumpsLDLT<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.mumpsRawStruct().sym == 1);
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

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
    VectorXd Ax = A * x;
    EXPECT_TRUE(Ax.isApprox(b, DOUBLE_TOLERANCE));

    MatrixXd X, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X = solver.solve(B);
    MatrixXd AX = A * X;
    EXPECT_TRUE(AX.isApprox(B, DOUBLE_TOLERANCE));
}

TEST(MumpsLDLT_test, compute) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_spd.mtx");

    MumpsLDLT<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.mumpsRawStruct().sym == 1);
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

    solver.compute(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();
    EXPECT_TRUE(det != 0);

    VectorXd x, b;
    b = VectorXd::Ones(A.rows());
    x = solver.solve(b);
    VectorXd Ax = A * x;
    EXPECT_TRUE(Ax.isApprox(b, DOUBLE_TOLERANCE));

    MatrixXd X, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X = solver.solve(B);
    MatrixXd AX = A * X;
    EXPECT_TRUE(AX.isApprox(B, DOUBLE_TOLERANCE));
}

TEST(MumpsLDLT_test, type_deduction) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_spd.mtx");

    MumpsLDLT solver(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.mumpsRawStruct().sym == 1);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();
    EXPECT_TRUE(det != 0);

    VectorXd x, b;
    b = VectorXd::Ones(A.rows());
    x = solver.solve(b);
    VectorXd Ax = A * x;
    EXPECT_TRUE(Ax.isApprox(b, DOUBLE_TOLERANCE));

    MatrixXd X, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X = solver.solve(B);
    MatrixXd AX = A * X;
    EXPECT_TRUE(AX.isApprox(B, DOUBLE_TOLERANCE));
}

TEST(MumpsLDLT_test, base_flags) {
    MumpsLDLT<SparseMatrix<double>> solver(NoDeterminant);
    EXPECT_TRUE(solver.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver.mumpsRawStruct().par == 0);

    MumpsLDLT<SparseMatrix<double>> solver2(Verbose);
    if (solver2.getProcessRank() == 0) {
        EXPECT_TRUE(solver2.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver2.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver2.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver2.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver2.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver2.mumpsRawStruct().par == 0);

    MumpsLDLT<SparseMatrix<double>> solver3(WorkingHost);
    EXPECT_TRUE(solver3.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver3.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver3.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver3.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver3.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver3.mumpsRawStruct().par == 1);

    MumpsLDLT<SparseMatrix<double>> solver4(NoDeterminant | Verbose | WorkingHost);
    if (solver4.getProcessRank() == 0) {
        EXPECT_TRUE(solver4.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver4.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver4.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver4.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver4.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver4.mumpsRawStruct().par == 1);

    MumpsLDLT<SparseMatrix<double>> solver5(NoDeterminant | Verbose);
    if (solver5.getProcessRank() == 0) {
        EXPECT_TRUE(solver5.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver5.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver5.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver5.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver5.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver5.mumpsRawStruct().par == 0);

    MumpsLDLT<SparseMatrix<double>> solver6(NoDeterminant | WorkingHost);
    EXPECT_TRUE(solver6.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver6.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver6.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver6.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver6.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver6.mumpsRawStruct().par == 1);

    MumpsLDLT<SparseMatrix<double>> solver7(Verbose | WorkingHost);
    if (solver7.getProcessRank() == 0) {
        EXPECT_TRUE(solver7.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver7.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver7.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver7.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver7.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver7.mumpsRawStruct().par == 1);

    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_spd.mtx");
    std::vector<int> schur_indices = {0, 1};

    MumpsLDLT solver8(A, NoDeterminant);
    EXPECT_TRUE(solver8.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver8.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver8.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver8.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver8.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver8.mumpsRawStruct().par == 0);

    MumpsLDLT solver9(A, Verbose);
    if (solver9.getProcessRank() == 0) {
        EXPECT_TRUE(solver9.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver9.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver9.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver9.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver9.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver9.mumpsRawStruct().par == 0);

    MumpsLDLT solver10(A, WorkingHost);
    EXPECT_TRUE(solver10.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver10.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver10.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver10.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver10.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver10.mumpsRawStruct().par == 1);

    MumpsLDLT solver11(A, NoDeterminant | Verbose | WorkingHost);
    if (solver11.getProcessRank() == 0) {
        EXPECT_TRUE(solver11.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver11.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver11.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver11.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver11.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver11.mumpsRawStruct().par == 1);

    MumpsLDLT solver12(A, NoDeterminant | Verbose);
    if (solver12.getProcessRank() == 0) {
        EXPECT_TRUE(solver12.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver12.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver12.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver12.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver12.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver12.mumpsRawStruct().par == 0);

    MumpsLDLT solver13(A, NoDeterminant | WorkingHost);
    EXPECT_TRUE(solver13.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver13.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver13.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver13.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver13.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver13.mumpsRawStruct().par == 1);

    MumpsLDLT solver14(A, Verbose | WorkingHost);
    if (solver14.getProcessRank() == 0) {
        EXPECT_TRUE(solver14.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver14.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver14.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver14.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver14.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver14.mumpsRawStruct().par == 1);
}

TEST(MumpsLDLT_test, colmajor_vs_rowmajor) {
    SparseMatrix<double> A_colmajor;
    SparseMatrix<double, RowMajor> A_rowmajor;
    Eigen::loadMarket(A_colmajor, "../data/matrix_spd.mtx");
    Eigen::loadMarket(A_rowmajor, "../data/matrix_spd.mtx");

    MumpsLDLT solver_colmajor(A_colmajor);
    MumpsLDLT solver_rowmajor(A_rowmajor);

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
    VectorXd Ax_colmajor = A_colmajor * x_colmajor;
    VectorXd Ax_rowmajor = A_rowmajor * x_rowmajor;
    EXPECT_TRUE(Ax_colmajor.isApprox(b, DOUBLE_TOLERANCE));
    EXPECT_TRUE(Ax_rowmajor.isApprox(b, DOUBLE_TOLERANCE));
    EXPECT_TRUE(x_colmajor.isApprox(x_rowmajor, DOUBLE_TOLERANCE));

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
    Matrix<double, Dynamic, Dynamic> AX_colmajor_colcol = A_colmajor * X_colmajor_colcol;
    Matrix<double, Dynamic, Dynamic> AX_colmajor_rowrow = A_colmajor * X_colmajor_rowrow;
    Matrix<double, Dynamic, Dynamic> AX_colmajor_colrow = A_colmajor * X_colmajor_colrow;
    Matrix<double, Dynamic, Dynamic> AX_colmajor_rowcol = A_colmajor * X_colmajor_rowcol;
    Matrix<double, Dynamic, Dynamic, RowMajor> AX_rowmajor_colcol = A_rowmajor * X_rowmajor_colcol;
    Matrix<double, Dynamic, Dynamic, RowMajor> AX_rowmajor_rowrow = A_rowmajor * X_rowmajor_rowrow;
    Matrix<double, Dynamic, Dynamic, RowMajor> AX_rowmajor_colrow = A_rowmajor * X_rowmajor_colrow;
    Matrix<double, Dynamic, Dynamic, RowMajor> AX_rowmajor_rowcol = A_rowmajor * X_rowmajor_rowcol;

    EXPECT_TRUE(AX_colmajor_colcol.isApprox(B_colmajor, DOUBLE_TOLERANCE));
    EXPECT_TRUE(AX_colmajor_rowrow.isApprox(B_rowmajor, DOUBLE_TOLERANCE));
    EXPECT_TRUE(AX_colmajor_colrow.isApprox(B_rowmajor, DOUBLE_TOLERANCE));
    EXPECT_TRUE(AX_colmajor_rowcol.isApprox(B_colmajor, DOUBLE_TOLERANCE));
    EXPECT_TRUE(AX_rowmajor_colcol.isApprox(B_colmajor, DOUBLE_TOLERANCE));
    EXPECT_TRUE(AX_rowmajor_rowrow.isApprox(B_rowmajor, DOUBLE_TOLERANCE));
    EXPECT_TRUE(AX_rowmajor_colrow.isApprox(B_rowmajor, DOUBLE_TOLERANCE));
    EXPECT_TRUE(AX_rowmajor_rowcol.isApprox(B_colmajor, DOUBLE_TOLERANCE));

    EXPECT_TRUE(X_colmajor_colcol.isApprox(X_colmajor_colrow, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_colmajor_colrow.isApprox(X_colmajor_rowcol, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_colmajor_rowcol.isApprox(X_colmajor_rowrow, DOUBLE_TOLERANCE));

    EXPECT_TRUE(X_rowmajor_colcol.isApprox(X_rowmajor_colrow, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_rowmajor_colrow.isApprox(X_rowmajor_rowcol, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_rowmajor_rowcol.isApprox(X_rowmajor_rowrow, DOUBLE_TOLERANCE));

    EXPECT_TRUE(X_colmajor_colcol.isApprox(X_rowmajor_colcol, DOUBLE_TOLERANCE));
}

TEST(MumpsLDLT_test, upper_lower) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_spd.mtx");

    MumpsLU solver(A);
    MumpsLDLT<SparseMatrix<double>, Upper> solver_upper(A);
    MumpsLDLT<SparseMatrix<double>, Lower> solver_lower(A);

    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver_upper.info() == Success);
    EXPECT_TRUE(solver_lower.info() == Success);
    EXPECT_TRUE(solver.mumpsRawStruct().sym == 0);
    EXPECT_TRUE(solver_upper.mumpsRawStruct().sym == 1);
    EXPECT_TRUE(solver_lower.mumpsRawStruct().sym == 1);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver_upper.rows() == A.rows());
    EXPECT_TRUE(solver_lower.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    EXPECT_TRUE(solver_upper.cols() == A.cols());
    EXPECT_TRUE(solver_lower.cols() == A.cols());
    double det = solver.determinant();
    double det_upper = solver_upper.determinant();
    double det_lower = solver_lower.determinant();
    EXPECT_TRUE(det != 0);
    EXPECT_TRUE(det_upper != 0);
    EXPECT_TRUE(det_lower != 0);

    VectorXd x_upper, x_lower, b;
    b = VectorXd::Ones(A.rows());
    x_upper = solver_upper.solve(b);
    x_lower = solver_lower.solve(b);
    VectorXd Ax_upper = A * x_upper;
    VectorXd Ax_lower = A * x_lower;
    EXPECT_TRUE(Ax_upper.isApprox(b, DOUBLE_TOLERANCE));
    EXPECT_TRUE(Ax_lower.isApprox(b, DOUBLE_TOLERANCE));
    EXPECT_TRUE(x_upper.isApprox(x_lower, DOUBLE_TOLERANCE));

    MatrixXd X_upper, X_lower, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X_upper = solver_upper.solve(B);
    X_lower = solver_lower.solve(B);
    MatrixXd AX_upper = A * X_upper;
    MatrixXd AX_lower = A * X_lower;
    EXPECT_TRUE(AX_upper.isApprox(B, DOUBLE_TOLERANCE));
    EXPECT_TRUE(AX_lower.isApprox(B, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_upper.isApprox(X_lower, DOUBLE_TOLERANCE));
}

TEST(Mumps_LDLT, inverse_elements) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_spd.mtx");

    MumpsLDLT<SparseMatrix<double>> solver(A);
    // std::vector<std::pair<int, int>> elements;
    std::set<std::pair<int, int>> elements;
    randomInvIndices(elements, A.rows());
    std::vector<Triplet<double>> inv_elements = solver.inverseElements(elements);
    // std::set<Triplet<double>> inv_elements = solver.inverseElements(elements);
    EXPECT_TRUE(inv_elements.size() == elements.size());

    SparseMatrix<double> A_inv;
    Eigen::loadMarket(A_inv, "../data/matrix_spd_inv.mtx");

    SparseMatrix<double> A_inv_test(A.rows(), A.cols());
    A_inv_test.setFromTriplets(inv_elements.begin(), inv_elements.end());
    SparseMatrix<double> A_inv_expected(A.rows(), A.cols());
    for (auto& element : inv_elements) {
        A_inv_expected.insert(element.row(), element.col()) = A_inv.coeff(element.row(), element.col());
    }
    EXPECT_TRUE(A_inv_test.isApprox(A_inv_expected, 1e-4));   // HIGHER TOLERANCE, im computing the inverse in 2
                                                              // different ways, so the results are not exactly the same
}

TEST(MumpsLDLT_test, split_analyze_factorize_sparse) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_spd.mtx");

    MumpsLDLT<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

    solver.analyzePattern(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());

    solver.factorize(A);
    EXPECT_TRUE(solver.info() == Success);
    double det = solver.determinant();
    EXPECT_TRUE(det != 0);

    SparseMatrix<double> x, b;
    b = VectorXd::Ones(A.rows()).sparseView();
    x = solver.solve(b);
    SparseMatrix<double> Ax = A * x;
    EXPECT_TRUE(Ax.isApprox(b, DOUBLE_TOLERANCE));

    SparseMatrix<double> X, B;
    B = MatrixXd::Ones(A.rows(), 3).sparseView();
    X = solver.solve(B);
    SparseMatrix<double> AX = A * X;
    EXPECT_TRUE(AX.isApprox(B, DOUBLE_TOLERANCE));
}

TEST(MumpsLDLT_test, compute_sparse) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_spd.mtx");

    MumpsLDLT<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

    solver.compute(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();
    EXPECT_TRUE(det != 0);

    SparseVector<double> x, b;
    b = VectorXd::Ones(A.rows()).sparseView();
    x = solver.solve(b);
    SparseVector<double> Ax = A * x;
    EXPECT_TRUE(Ax.isApprox(b, DOUBLE_TOLERANCE));

    SparseMatrix<double> X, B;
    B = MatrixXd::Ones(A.rows(), 3).sparseView();
    X = solver.solve(B);
    SparseMatrix<double> AX = A * X;
    EXPECT_TRUE(AX.isApprox(B, DOUBLE_TOLERANCE));
}

TEST(MumpsLDLT_test, type_deduction_sparse) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_spd.mtx");

    MumpsLDLT solver(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();
    EXPECT_TRUE(det != 0);

    SparseVector<double> x, b;
    b = VectorXd::Ones(A.rows()).sparseView();
    x = solver.solve(b);
    SparseVector<double> Ax = A * x;
    EXPECT_TRUE(Ax.isApprox(b, DOUBLE_TOLERANCE));

    SparseMatrix<double> X, B;
    B = MatrixXd::Ones(A.rows(), 3).sparseView();
    X = solver.solve(B);
    SparseMatrix<double> AX = A * X;
    EXPECT_TRUE(AX.isApprox(B, DOUBLE_TOLERANCE));
}
