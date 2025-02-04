#include "../../fdaPDE/linear_algebra/mumps.h"

#include <Eigen/Sparse>

using fdapde::mumps::MumpsLU;

int main() {

    Eigen::SparseMatrix<double> A(3, 3);
    A.insert(0, 0) = 1;
    A.insert(1, 1) = 2;
    A.insert(2, 2) = 3;
    A.makeCompressed();

    Eigen::VectorXd b(3);
    b << 1, 2, 3;

    MumpsLU mumps(A);
    Eigen::VectorXd x = mumps.solve(b);

    if (mumps.getProcessRank() == 0) {
        std::cout << "Solution: " << x.transpose() << std::endl;
    }

    return 0;
}