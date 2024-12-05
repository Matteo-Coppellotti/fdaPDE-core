#include "../../../fdaPDE/fields.h"
#include "../../../fdaPDE/geometry.h"
#include "../../../fdaPDE/finite_elements.h"
#include "../../../fdaPDE/geoframe/csv.h"

using namespace fdapde;

// std::tuple<Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<int, Dynamic, Dynamic>, Eigen::Matrix<int, Dynamic, Dynamic>> generate_unit_square_mesh(int N) {
//     Eigen::Matrix<double, Dynamic, Dynamic> nodes(2, (N + 1) * (N + 1));
//     Eigen::Matrix<int, Dynamic, Dynamic> cells(3, 2 * N * N);
//     Eigen::Matrix<int, Dynamic, Dynamic> boundary(2, 4 * N);

//     double h = 1.0 / N;
//     int node_id = 0;
//     for (int j = 0; j <= N; ++j) {
//         for (int i = 0; i <= N; ++i) {
//             nodes(0, node_id) = i * h;
//             nodes(1, node_id) = j * h;
//             node_id++;
//         }
//     }

//     int cell_id = 0;

//     for (int j = 0; j < N; ++j) {
//         for (int i = 0; i < N; ++i) {
//             cells(0, cell_id) = i + j * (N + 1);
//             cells(1, cell_id) = i + 1 + j * (N + 1);
//             cells(2, cell_id) = i + 1 + (j + 1) * (N + 1);
//             cell_id++;

//             cells(0, cell_id) = i + j * (N + 1);
//             cells(1, cell_id) = i + 1 + (j + 1) * (N + 1);
//             cells(2, cell_id) = i + (j + 1) * (N + 1);
//             cell_id++;
//         }
//     }

//     int boundary_id = 0;

//     for (int i = 0; i < N; ++i) {
//         boundary(0, boundary_id) = i;
//         boundary(1, boundary_id) = i + 1;
//         boundary_id++;

//         boundary(0, boundary_id) = i + N * (N + 1);
//         boundary(1, boundary_id) = i + 1 + N * (N + 1);
//         boundary_id++;

//         boundary(0, boundary_id) = i * (N + 1);
//         boundary(1, boundary_id) = (i + 1) * (N + 1);
//         boundary_id++;

//         boundary(0, boundary_id) = i + N * (N + 1);
//         boundary(1, boundary_id) = i + (N - 1) * (N + 1);
//         boundary_id++;
//     }
    
//     return {nodes, cells, boundary};
// }

int main() {
    // int N = 10;

    Eigen::Matrix<double, Dynamic, Dynamic> nodes = read_csv<double>("../../../test/mesh/unit_square/points.csv").as_matrix();
    Eigen::Matrix<int, Dynamic, Dynamic> cells    = read_csv<int>("../../../test/mesh/unit_square/elements.csv").as_matrix();
    Eigen::Matrix<int, Dynamic, Dynamic> boundary = read_csv<int>("../../../test/mesh/unit_square/boundary.csv").as_matrix();
    // [nodes, cells, boundary] = generate_unit_square_mesh(N);

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