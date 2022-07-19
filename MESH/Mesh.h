#ifndef __MESH_H__
#define __MESH_H__

#include <Eigen/Core>
#include <type_traits>
#include <vector>
#include <memory>
#include <array>

#include "Element.h"
#include "CSVReader.h"
#include "../utils/Symbols.h"
#include "MeshUtils.h"

namespace fdaPDE{
namespace core{
namespace MESH{

  // Mesh is the access point to mesh informations (triangulated domains). It offers an abstraction layer allowing to reason
  // on the mesh from a geometrical perspective, i.e. without considering its internal representation in memory. The class is able to transparently
  // handle manifold and non manifold meshes, exposing the same interface in any case. Two non-type template parameters are used:
  //     * M: local dimension of the mesh (dimension of the space to which a mesh element belongs)
  //     * N: embeddding dimension of the mesh (dimension of the space where the whole mesh lies)
  // if M != N the mesh is a manifold. Currently are implemented:
  //     * 1.5D meshes (linear newtorks,    M=1, N=2)
  //     * 2D meshes   (planar domains,     M=2, N=2)
  //     * 2.5D meshes (surfaces,           M=2, N=3)
  //     * 3D meshes   (volumetric domains, M=3, N=3)

  // NB about internal implementaiton: special care is needed in the development of linear networks, since differently from any other case the number
  // of neighboing elements is not known at compile time. This implies the usage of specialized data structures wrt any other case

  // trait to detect if the mesh is a manifold
  template <unsigned int M, unsigned int N>
  struct is_manifold{
    static constexpr bool value = (M != N);
  };
  
  // trait to detect if the mesh is a linear network
  template <unsigned int M, unsigned int N>
  struct is_linear_network{
    static constexpr bool value = std::conditional<
      (M == 1 && N == 2), std::true_type, std::false_type
      >::type::value;
  };

  // trait to select a proper neighboring storage structure depending on the type of mesh. In case of linear networks this information is stored as
  // a sparse matrix where entry (i,j) is set to 1 if and only if elements i and j are neighbors
  template <unsigned int M, unsigned int N>
  struct neighboring_structure{
    using type = typename std::conditional<
      is_linear_network<M, N>::value, SpMatrix<int>, DMatrix<int>
      >::type;
  };
  
  template <unsigned int M, unsigned int N>
  class Mesh{
  private:
    // coordinates of points constituting the vertices of mesh elements
    DMatrix<double> points_;
    unsigned int numNodes = 0;
    // matrix of edges. Each row of the matrix contains the row numbers in points_ matrix
    // of the points which form the edge
    DMatrix<int> edges_;
    // matrix of triangles in the triangulation. Each row of the matrix contains the row
    // numbers in points_ matrix of the points which form the triangle
    DMatrix<int> triangles_;
    unsigned int numElements = 0;
    // in case of non linear-networks neighbors_ is a dense matrix where row i contains the indexes as row number in triangles_ matrix of the
    // neighboring triangles to triangle i (all triangles in the triangulation which share an edge with i). In case of linear-newtorks neighbors_
    // is a sparse matrix where entry (i,j) is set to 1 iff i and j are neighbors
    typename neighboring_structure<M, N>::type neighbors_;
    // store boundary informations. This is a vector of binary coefficients such that, if element j is 1
    // then mesh node j is on boundary, otherwise 0
    DMatrix<int> boundaryMarkers_;
    
    // store min-max values for each dimension of the mesh
    std::array<std::pair<double, double>, N> meshRange;
    // is often required to access just to the minimum value along each dimension and to the quantity
    // 1/(max[dim] - min[dim]) = 1/(meshRange[dim].second - meshRange[dim].first). Compute here once and cache results for efficiency
    std::array<double, N> minMeshRange;
    std::array<double, N> kMeshRange; // kMeshRange[dim] = 1/(meshRange[dim].second - meshRange[dim].first)
  
  public:
    // constructor from .csv files
    Mesh(const std::string& pointsFile,    const std::string& edgesFile, const std::string& trianglesFile,
	 const std::string& neighborsFile, const std::string& boundaryMarkersFile);

    // construct an element object given its ID (its row number in the triangles_ matrix) from raw (matrix-like) informations
    std::shared_ptr<Element<M,N>> requestElementById(unsigned int ID) const;

    // allow range-for loop over mesh elements
    struct iterator{
    private:
      friend Mesh;
      const Mesh* meshContainer; // pointer to mesh object
      int index;           // keep track of current iteration during for-loop
      // constructor
      iterator(const Mesh* container_, int index_) : meshContainer(container_), index(index_) {}; 
    public:
      // just increment the current iteration and return this iterator
      iterator& operator++() {
	++index;
	return *this;
      }
      // dereference the iterator means to create Element object at current index
      std::shared_ptr<Element<M,N>> operator*() {
	return meshContainer->requestElementById(index);
      }
      // two iterators are different when their indexes are different
      friend bool operator!=(const iterator& lhs, const iterator& rhs) {
	return lhs.index != rhs.index;
      }

      // const version to enable const auto& syntax
      std::shared_ptr<Element<M,N>> operator*() const { return meshContainer->requestElementById(index); }
    };
    // provide begin() and end() methods
    iterator begin() const { return iterator(this, 0); }
    iterator end()   const { return iterator(this, triangles_.rows()); }

    // getters
    unsigned int getNumberOfElements() const { return numElements; }
    unsigned int getNumberOfNodes() const { return numNodes; }
    std::array<std::pair<double, double>, N> getMeshRange() const { return meshRange; }

    // return true if the given node is on boundary, false otherwise
    bool isOnBoundary(size_t j) const { return boundaryMarkers_(j) == 1; }
  };

  // export some aliases
  using Mesh2D = Mesh<2,2>;
  using Mesh3D = Mesh<3,3>;
  // manifold cases
  using SurfaceMesh = Mesh<2,3>;
  using LinearNetworkMesh  = Mesh<1,2>;

#include "Mesh.tpp"
}}}
  
#endif // __MESH_H__
