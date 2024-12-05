#include "../../../fdaPDE/fields.h"
#include "../../../fdaPDE/geometry.h"
#include "../../../fdaPDE/finite_elements.h"
#include "../../../fdaPDE/geoframe/csv.h"

using namespace fdapde;

int main() {

    Eigen::Matrix<double, Dynamic, Dynamic> nodes = read_csv<double>("../../../test/mesh/unit_square/points.csv").as_matrix();
    Eigen::Matrix<int, Dynamic, Dynamic> cells    = read_csv<int>("../../../test/mesh/unit_square/elements.csv").as_matrix();
    Eigen::Matrix<int, Dynamic, Dynamic> boundary = read_csv<int>("../../../test/mesh/unit_square/boundary.csv").as_matrix();

    Triangulation</* tangent_space = */ 2, /* embedding_space = */ 2> unit_square(nodes, cells, boundary); // need to generate square mesh

    FeSpace Vh(unit_square, P1<1>);
    TrialFunction u(Vh);
    TestFunction  v(Vh);

    auto a1 = integral(unit_square)(u * v); // mass matrix: SPD
    auto a2 = integral(unit_square)(dot(grad(u), grad(v))); // laplacian discretization (Poisson): SPD 

    Eigen::Matrix<double, 2, 1> b(0.2, 0.2);

    auto a3 = integral(unit_square)(dot(grad(u), grad(v)) + dot(b, grad(u)) * v); // diffusion-transport: non simmetrica

    SpMatrix<double> R1 = a1.assemble();
    SpMatrix<double> R2 = a2.assemble();
    SpMatrix<double> R3 = a3.assemble();

    // forcing term
    ScalarField<2, decltype([]([[maybe_unused]] const Eigen::Matrix<double, 2, 1>& p) { return 1; })> f;
    auto F = integral(unit_square)(f * v);

    Eigen::Matrix<double, Dynamic, 1> force = F.assemble();

    std::cout << force.topRows(10) << std::endl;

    // linear system: R1 * u = f
    //MumpsLU<SpMatrix<double>> solver;
    //solver.compute(R1);
    //solver.solve(force); // booom!!



}