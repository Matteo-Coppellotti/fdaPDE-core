#ifndef __MUMPS_H__
#define __MUMPS_H__

#include <Mumps/dmumps_c.h>
#include <mpi.h>

#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "../utils/assert.h"

using namespace Eigen;

namespace fdapde {

namespace mumps {

// concepts
template <typename T>
concept isEigenSparseMatrix = std::derived_from<T, SparseMatrixBase<T>>;
// Integer set concept used in MumpsSchur
template <typename Set>
concept isIntSet = std::same_as<Set, std::set<int, typename Set::key_compare, typename Set::allocator_type>>;
// Pair of integers set concept used in inverseElements
template <typename Set>
concept isPairIntIntSet =
  std::same_as<Set, std::set<std::pair<int, int>, typename Set::key_compare, typename Set::allocator_type>>;

struct OrderByColumns {   // Used to sort the elements in the inverseElements method
    bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) const {
        if (a.second != b.second) {
            return a.second < b.second;   // Compare by second element first
        }
        return a.first < b.first;   // If second elements are equal, compare by first element
    }
};

// SINGLETON class for MPI initialization and finalization
// [MSG for A.Palummo] Ho dovuto aggiungere questa classe perchè ho giustamente riscontrato problemi con lo scoping dei
// solver: se dichiaravo il solver in uno scope e poi ne uscivo veniva chiamato il distruttore che finalizzava MPI,
// purtroppo così facendo non era più possibile dichiarare solver nello stesso codice perchè avrebbe chiamato
// automaticamente MPI_Init(), cosa che MPI non permetter dopo una call a MPI_Finalize(), oltre a creare problemi con i
// getter delle flag MPI_Initialized() e MPI_Finalized(). Con questa classe posso quindi garantire che MPI_Init() e
// MPI_Finalize() vengano chiamate una sola volta e che MPI_Init() venga chiamata prima di qualsiasi altra funzione MPI.
// Ovviamente, se MPI_Init() e MPI_Finalize() vengono chiamate in modo esplicito nel codice, questa classe non ha alcun
// effetto.
class MPI_Manager {
   public:
    static MPI_Manager& getInstance() {
        static MPI_Manager instance;
        return instance;
    }

    MPI_Manager(const MPI_Manager&) = delete;
    MPI_Manager& operator=(const MPI_Manager&) = delete;
    MPI_Manager(MPI_Manager&&) = delete;
    MPI_Manager& operator=(MPI_Manager&&) = delete;

    bool isMPIinitialized() const { return mpi_initialized; }
   private:
    bool mpi_initialized;

    MPI_Manager() : mpi_initialized(false) {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(NULL, NULL);
            mpi_initialized = true;
        }
    }

    ~MPI_Manager() {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized && mpi_initialized) { MPI_Finalize(); }
    }
};

template <typename T, int Size> class MumpsParameterArray {
   public:
    MumpsParameterArray() : mp_data(nullptr) { }
    MumpsParameterArray(T* data_ptr) : mp_data(data_ptr) { }

    inline T* data() { return mp_data; }
    inline const T* data() const { return mp_data; }
    inline size_t size() const { return Size; }

    inline T& operator[](size_t i) {
        fdapde_assert(i >= 0 && "Index must be positive");
        fdapde_assert(i < Size && "Index out of bounds");
        return mp_data[i];
    }

    inline const T& operator[](size_t i) const {
        fdapde_assert(i >= 0 && "Index must be positive");
        fdapde_assert(i < Size && "Index out of bounds");
        return mp_data[i];
    }
   protected:
    T* mp_data;
};

// MUMPS FLAGS

// Base class flags
constexpr unsigned int NoDeterminant = (1 << 0);
constexpr unsigned int Verbose = (1 << 1);
constexpr unsigned int WorkingHost = (1 << 2);

// BLR flags
constexpr unsigned int UFSC = (1 << 3);   // default
constexpr unsigned int UCFS = (1 << 4);
constexpr unsigned int Compressed = (1 << 5);

// FORWARD DECLARATIONS
template <isEigenSparseMatrix MatrixType_> class MumpsLU;
template <isEigenSparseMatrix MatrixType_, int Options = Upper> class MumpsLDLT;
template <isEigenSparseMatrix MatrixType_> class MumpsBLR;
template <isEigenSparseMatrix MatrixType_> class MumpsRR;
template <isEigenSparseMatrix MatrixType_> class MumpsSchur;

namespace internal {

template <class Derived> struct mumps_traits;

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsLU<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsLDLT<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsBLR<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsRR<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsSchur<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

}   // namespace internal

// MUMPS BASE CLASS
template <class Derived> class MumpsBase : public SparseSolverBase<Derived> {
   protected:
    using Base = SparseSolverBase<Derived>;
    using Base::derived;
    using Base::m_isInitialized;

    using Traits = internal::mumps_traits<Derived>;
   public:
    using Base::_solve_impl;

    using MatrixType = typename Traits::MatrixType;
    using Scalar = typename Traits::Scalar;
    using RealScalar = typename Traits::RealScalar;
    using StorageIndex = typename Traits::StorageIndex;
    using MumpsIcntlArray = MumpsParameterArray<int, 60>;
    using MumpsCntlArray = MumpsParameterArray<double, 15>;
    using MumpsInfoArray = MumpsParameterArray<int, 80>;
    using MumpsRinfoArray = MumpsParameterArray<double, 40>;
    enum {
        ColsAtCompileTime = MatrixType::ColsAtCompileTime,
        MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

    inline Index cols() const { return m_size; }
    inline Index rows() const { return m_size; }

    inline int getProcessRank() const { return m_mpiRank; }
    inline int getProcessSize() const { return m_mpiSize; }

    inline MumpsIcntlArray& mumpsIcntl() { return m_mumpsIcntl; }
    inline MumpsCntlArray& mumpsCntl() { return m_mumpsCntl; }

    inline const MumpsIcntlArray& mumpsIcntl() const { return m_mumpsIcntl; }
    inline const MumpsCntlArray& mumpsCntl() const { return m_mumpsCntl; }
    inline const MumpsInfoArray& mumpsInfo() const { return m_mumpsInfo; }
    inline const MumpsInfoArray& mumpsInfog() const { return m_mumpsInfog; }
    inline const MumpsRinfoArray& mumpsRinfo() const { return m_mumpsRinfo; }
    inline const MumpsRinfoArray& mumpsRinfog() const { return m_mumpsRinfog; }

    // setter & getter for the raw mumps struct, tinkering with it is unadvides unless the user has an already good
    // understanding of mumps and this wrapper
    // [MSG for A.Palummo] Ho deciso di aggiungere questi getter in modo da permettere a persone più esperte di poter
    // accedere direttamente alla struttura di MUMPS, in modo da poter settare parametri non accessibili tramite i
    // metodi pubblici della classe. Questo permette di accedere a membri della struttura che non sono utilizzati da
    // questa classe che potrebbero però essere utili nel caso vengano modificati manualmente alcuni icntl o cntl.
    inline DMUMPS_STRUC_C& mumpsRawStruct() { return m_mumps; }
    inline const DMUMPS_STRUC_C& mumpsRawStruct() const { return m_mumps; }

    Derived& analyzePattern(const MatrixType& matrix) {
        define_matrix(matrix);
        analyzePattern_impl();
        return derived();
    }

    Derived& factorize(const MatrixType& matrix) {
        define_matrix(matrix);
        factorize_impl();
        return derived();
    }

    Derived& compute(const MatrixType& matrix) {
        define_matrix(matrix);
        compute_impl();
        return derived();
    }

    template <typename BDerived, typename XDerived>
    void _solve_impl(const MatrixBase<BDerived>& b, MatrixBase<XDerived>& x) const {
        fdapde_assert(m_factorizationIsOk && "The matrix must be factorized with factorize() before calling solve()");

        if (b.derived().data() == x.derived().data()) {   // inplace solve
            fdapde_assert((BDerived::Flags & RowMajorBit) == 0 && "Inplace solve doesn't support row-major rhs");
            if (getProcessRank() == 0) {
                m_mumps.nrhs = b.cols();
                m_mumps.lrhs = b.rows();
                m_mumps.rhs = const_cast<Scalar*>(b.derived().data());
            }
            mumps_execute(3);   // 3: solve
        }

        else {
            Matrix<Scalar, Dynamic, Dynamic, ColMajor> buff;   // mumps requires the rhs to be ColMajor
            if (getProcessRank() == 0) {
                buff = b;
                m_mumps.nrhs = buff.cols();
                m_mumps.lrhs = buff.rows();
                m_mumps.rhs = const_cast<Scalar*>(buff.data());
            }
            mumps_execute(3);   // 3: solve
            x.derived().resizeLike(b);
            if (getProcessRank() == 0) { x.derived() = buff; }
        }

        MPI_Bcast(x.derived().data(), x.size(), MPI_DOUBLE, 0, m_mpiComm);
    }

    // [MSG for A.Palummo for A.Palummo] Questo overloading è stato aggiunto per mermettere la risoluzione di sistemi
    // con sparse rhs, per quanto ho capito da Solve.h, Eigen non supporta il caso in cui rhs è sparso ma dest è denso.
    // L'unico problema con questo overloading è che, al meglio della mia comprensione, Solve.h nega l'implementazione
    // di Dense2Dense nel caso in cui dest sia RowMajor. Purtroppo non posso inserire assert per negare l'accesso a
    // questo overloading nel caso in cui dest sia RowMajor, in quanto il check delle option credo sia fatto
    // direttamente da Solve.h. COSA SUCCEDE SE USO solve() CON DEST SPARSA RowMajor? Solve.h non riconosce questo
    // overloading e cercca di usare il metodo per matrici dense, che ovviamente porta a errori dovuti al fatto che i
    // metodi forniti dalle due classi sono diversi. L'unica idea che mi viene è quella di creare un metodo solveSparse
    // che prenda in input sia dest che rhs sparse, crei una copia converita in ColMajor, passi queste copie a solve e
    // poi riconverta le copie in RowMajor e le assegni a dest. Questo metodo però è molto poco elegante e soprattutto
    // devia dalla norma per quanto riguarda l'interazoine dell'utente con il metoto solve delle classi di Eigen. Lo
    // lascio commentato dopo il metodo _solve_impl e mi affido al suo giudizio
    template <typename BDerived, typename XDerived>
    void _solve_impl(const SparseMatrix<BDerived>& b, SparseMatrixBase<XDerived>& x) const {
        fdapde_assert(m_factorizationIsOk && "The matrix must be factorized with factorize() before calling solve()");

        SparseMatrix<Scalar, ColMajor> loc_b;

        std::vector<StorageIndex> irhs_ptr;   // Stores the indices of the non zeros in valuePts that start a new column
                                              // like innerIndexPtr does but requires 1-based indexing and an extra
                                              // element at the end equal to the number of nonzeros + 1
        std::vector<StorageIndex> irhs_sparse;   // Stores the row indices of the nonzeros like outerIndexPtr does, but
                                                 // requires 1-based indexing

        Matrix<Scalar, Dynamic, Dynamic, ColMajor> rhs(b.rows(), b.cols());

        if (getProcessRank() == 0) {
            loc_b = b;
            loc_b.makeCompressed();

            irhs_ptr.reserve(loc_b.cols() + 1);
            for (StorageIndex i = 0; i < loc_b.cols(); ++i) {   // Already shifts to 1-based indexing
                irhs_ptr.push_back(loc_b.outerIndexPtr()[i] + 1);
            }
            irhs_ptr.push_back(loc_b.nonZeros() + 1);

            irhs_sparse.reserve(loc_b.nonZeros());
            for (StorageIndex i = 0; i < loc_b.nonZeros(); ++i) {   // Already shifts to 1-based indexing
                irhs_sparse.push_back(loc_b.innerIndexPtr()[i] + 1);
            }

            m_mumps.nz_rhs = loc_b.nonZeros();
            m_mumps.nrhs = loc_b.cols();
            m_mumps.lrhs = loc_b.rows();
            m_mumps.rhs_sparse = loc_b.valuePtr();
            m_mumps.irhs_sparse = irhs_sparse.data();
            m_mumps.irhs_ptr = irhs_ptr.data();
            m_mumps.rhs = const_cast<Scalar*>(rhs.data());
        }

        m_mumps.icntl[19] = 1;   // 1: sparse right-hand side
        mumps_execute(3);
        m_mumps.icntl[19] = 0;   // reset to default

        MPI_Bcast(rhs.data(), rhs.size(), MPI_DOUBLE, 0, m_mpiComm);

        x = rhs.sparseView();   // Unfortunately Mumps only outputs the solution in dense format, i need to make it
                                // sparse to adhere to Solve.h (only Dense2Dense or Sparse2Sparse are allowed)
    }

    // template <typename BDerived, typename XDerived>
    // void solveSparse(const SparseMatrix<BDerived>& b, SparseMatrixBase<XDerived>& x) const {
    //     SparseMatrix<Scalar, ColMajor> loc_b;
    //     SparseMatrix<Scalar, ColMajor> loc_x;
    //     if (BDerived::Flags & RowMajorBit || XDerived::Flags & RowMajorBit) {
    //         loc_b = b;
    //         loc_b.makeCompressed();
    //         loc_x = x;
    //         loc_x.makeCompressed();
    //         loc_x = solve(loc_b);
    //         x = loc_x;
    //     } else {
    //         x = solve(b);
    //     }
    // }

    Scalar determinant() {
        fdapde_assert(m_computeDeterminant && "The determinant computation must be enabled");
        fdapde_assert(
          m_factorizationIsOk && "The matrix must be factoried with factorize_impl() before calling determinant()");
        if (mumpsInfog()[27] != 0) {
            return 0;
        }   // infog[27] stores the number of singularities, if it's not 0 the
            // matrix is singular and the determinant is 0 (this is needed for RR since if not the determinant computed
            // wouldn't be the actual determinant but determinant computed on a matrix with missing rows/columns [see
            // MUMPS documentation])
        return mumpsRinfog()[11] *
               std::pow(2, mumpsInfog()[33]);   // Keep in mind that if infog[33] is very high std::pow will return inf
    }

    // [MSG for A.Palummo] Questo metodo permette di calcolare l'incverso di un set di elementi della matrice. L'attuale
    // implementazione prende in input un set di coordinate per gerantirne l'unicità e poi inserisco gli elementi in un
    // mio set con custom ordering che ordina automaticamente gli indici per colonna per rendere la creazione delle
    // strutture richieste da mumps più efficienti. Ho lasciato come output un vector di triplet, ma se preferisce
    // impostare acnhe l'output come set di triplet non ci sono problemi, ho lasciato il codice necessario per farlo
    // commentato. Inoltre, nel caso reputi che ricevere il set in input sia eccessivmente limitante, ho lasciato anche
    // il codice con input vector di pair e output vector di triplet commentato di seguito alla funzione.
    // Nel caso decida di cambiare il tipo di input/output, i test associati dovrebbero fallire ma ho lsciato nei file
    // di testing i test per gli altri casi commentati in modo analogo. (anche utils.h)

    // split into template + overloading to allow any key_compare in the input set but still ensure that my function
    // recieves the ordering i need (in this case OrderByColumns)
    template <isPairIntIntSet SetType> std::vector<Triplet<Scalar>> inverseElements(const SetType& elements) {
        std::set<std::pair<int, int>, OrderByColumns> el(elements.begin(), elements.end());
        return inverseElements(el);
    }

    // // se si vuole usare un set di triplet (the require can be replaced with a concept)
    // template <isPairIntIntSet SetType, typename OutSet>
    //     requires std::same_as<
    //       OutSet, std::vector<Triplet<Scalar>>, typename OutSet::key_compare, typename OutSet::allocator_type>
    // OutSet inverseElements(SetType& elements) {
    //     std::set<std::pair<int, int>, OrderByColumns> el(elements.begin(), elements.end());
    //     return OutSet(inverseElements(el).begin(), inverseElements(el).end());
    // }

    // se si vuole usare un set di triplet
    // std::set<Triplet<Scalar>> inverseElements(std::set<std::pair<int,int>, OrderByColumns>& el) {
    std::vector<Triplet<Scalar>> inverseElements(const std::set<std::pair<int, int>, OrderByColumns>& elements) {
        fdapde_assert(
          mumpsIcntl()[18] == 0 &&
          "Incompatible with Schur complement");   // The Mumps icntls 18 and 29 are incompatible
        fdapde_assert(
          m_factorizationIsOk && "The matrix must be factorized with factorize() before calling inverseElements()");
        fdapde_assert(elements.size() > 0 && "The set of elements must be non-empty");
        for (const auto& elem : elements) {
            fdapde_assert(elem.first >= 0 && elem.first < m_size && "The selected rows must be within the matrix size");
            fdapde_assert(
              elem.second >= 0 && elem.second < m_size && "The selected columns must be within the matrix size");
        }

        std::vector<Triplet<Scalar>> inv_elements;
        inv_elements.reserve(elements.size());
        // std::set<Triplet<Scalar>> inv_elements;

        std::vector<int> irhs_ptr;   // Stores the indices of the fisrt requested element of each column, requires
                                     // 1-based indexing and an extra element at the end equal to the number of columns
                                     // + 1. It should have size = number of columns + 1
                                     // In general, if a column doesn't have any requested element, the respective entry
                                     // should be equal to the next column's first requested element (if this is empty
                                     // as well the process should be repeated until a column with requested elements is
                                     // found and that index is used for the empty columns preceeding it as well)
        std::vector<int> irhs_sparse;     // Stores the row indices of the requested elements, requires 1-based indexing
        std::vector<Scalar> rhs_sparse;   // Stores the inverse of the requested elements
        rhs_sparse.resize(elements.size());

        if (getProcessRank() == 0) {
            irhs_ptr.reserve(m_size + 1);
            for (int i = 0; i <= elements.begin()->second; ++i) {
                irhs_ptr.push_back(1);
            }   // i nedd to fill the vector with 1 until the column of the first requested element (included)
            auto prev = elements.begin();
            int k = 1;   // position in the set (the for cycle starts from the second element)
            for (auto it = std::next(elements.begin()); it != elements.end(); ++it) {
                if (it->second != prev->second) {
                    for (int j = 0; j < it->second - prev->second; ++j) { irhs_ptr.push_back(k + 1); }
                }
                prev = it;
                ++k;
            }   // When the column changes, I need to insert into the vector the index of the first requested element of
                // the new column, if there are empty columns I need to fill the vector with the index of the first
                // requested element of the next column, with as many elements as the number of empty columns before the
                // next column with requested elements (alredy shifted to 1-based indexing)
            for (int i = prev->second; i <= m_size; ++i) {
                irhs_ptr.push_back(elements.size() + 1);
            }   // Adding the last entry as mumps requires (number of columns + 1 in position of the last column + 1)

            irhs_sparse.reserve(elements.size());
            for (const auto& elem : elements) {
                irhs_sparse.push_back(elem.first + 1);
            }   // already scales the row indices to 1-based indexing

            m_mumps.nz_rhs = elements.size();
            m_mumps.nrhs = m_size;
            m_mumps.lrhs = m_size;
            m_mumps.irhs_ptr = irhs_ptr.data();
            m_mumps.irhs_sparse = irhs_sparse.data();
            m_mumps.rhs_sparse = rhs_sparse.data();
        }

        mumpsIcntl()[29] = 1;   // 1: compute selected elements of the inverse
        mumps_execute(3);
        mumpsIcntl()[29] = 0;   // reset to default

        MPI_Bcast(rhs_sparse.data(), rhs_sparse.size(), MPI_DOUBLE, 0, m_mpiComm);

        int i = 0;
        for (const auto& elem : elements) {
            // se si vuole usare un set di triplet
            // inv_elements.emplace(elem.first, elem.second, rhs_sparse[i++]);
            inv_elements.emplace_back(elem.first, elem.second, rhs_sparse[i++]);
        }   // the elements indices were not changed, so I can just copy them from the input vector

        return inv_elements;
    }

    // std::vector<Triplet<Scalar>> inverseElements(std::vector<std::pair<int, int>> elements) {
    //     fdapde_assert(
    //       mumpsIcntl()[18] == 0 &&
    //       "Incompatible with Schur complement");   // The Mumps icntls 18 and 29 are incompatible
    //     fdapde_assert(
    //       m_factorizationIsOk && "The matrix must be factorized with factorize() before calling inverseElements()");

    //     // reored elements by col number
    //     std::sort(elements.begin(), elements.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
    //         return a.second < b.second;
    //     });

    //     fdapde_assert(   // check if the selected elements are unique (the adjacent_find could be done outside the
    //                      // assert and save the ... == ... to a bool to then pun into the assert, I don't know
    //                      wheteher
    //                      // it would be peferred)
    //       std::adjacent_find(
    //         elements.begin(), elements.end(),
    //         [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
    //             return a.first == b.first && a.second == b.second;
    //         }) == elements.end() &&
    //       "The selected elements must be unique");

    //     for (auto& elem : elements) {
    //         fdapde_assert(elem.first >= 0 && elem.first < m_size && "The selected rows must be within the matrix
    //         size"); fdapde_assert(
    //           elem.second >= 0 && elem.second < m_size && "The selected columns must be within the matrix size");
    //     }

    //     std::vector<Triplet<Scalar>> inv_elements;
    //     inv_elements.reserve(elements.size());

    //     std::vector<int> irhs_ptr;   // Stores the indices of the fisrt requested element of each column, requires
    //                                  // 1-based indexing and an extra element at the end equal to the number of
    //                                  columns
    //                                  // + 1. It should have size = number of columns + 1
    //                                  // In general, if a column doesn't have any requested element, the respective
    //                                  entry
    //                                  // should be equal to the next column's first requested element (if this is
    //                                  empty
    //                                  // as well the process should be repeated until a column with requested elements
    //                                  is
    //                                  // found and that index is used for the empty columns preceeding it as well)
    //     std::vector<int> irhs_sparse;     // Stores the row indices of the requested elements, requires 1-based
    //     indexing std::vector<Scalar> rhs_sparse;   // Stores the inverse of the requested elements
    //     rhs_sparse.resize(elements.size());

    //     if (getProcessRank() == 0) {
    //         irhs_ptr.reserve(elements.size() + 1);
    //         for (int i = 0; i <= elements[0].second; ++i) {
    //             irhs_ptr.push_back(1);
    //         }   // i nedd to fill the vector with 1 until the column of the first requested element (included)
    //         for (size_t i = 1; i < elements.size(); ++i) {
    //             if (elements[i].second != elements[i - 1].second) {
    //                 for (int j = 0; j < elements[i].second - elements[i - 1].second; ++j) { irhs_ptr.push_back(i +
    //                 1); }
    //             }
    //         }   // When the column changes, I need to insert into the vector the index of the first requested element
    //         of
    //             // the new column, if there are empty columns I need to fill the vector with the index of the first
    //             // requested element of the next column, with as many elements as the number of empty columns before
    //             the
    //             // next column with requested elements (alredy shifted to 1-based indexing)
    //         for (int i = elements.back().second; i <= m_size; ++i) {
    //             irhs_ptr.push_back(elements.size() + 1);
    //         }   // Adding the last entry as mumps requires (number of columns + 1 in position of the last column + 1)

    //         irhs_sparse.reserve(elements.size());
    //         for (const auto& elem : elements) {
    //             irhs_sparse.push_back(elem.first + 1);
    //         }   // already scales the row indices to 1-based indexing

    //         m_mumps.nz_rhs = elements.size();
    //         m_mumps.nrhs = m_size;
    //         m_mumps.lrhs = m_size;
    //         m_mumps.irhs_ptr = irhs_ptr.data();
    //         m_mumps.irhs_sparse = irhs_sparse.data();
    //         m_mumps.rhs_sparse = rhs_sparse.data();
    //     }

    //     mumpsIcntl()[29] = 1;   // 1: compute selected elements of the inverse
    //     mumps_execute(3);
    //     mumpsIcntl()[29] = 0;   // reset to default

    //     MPI_Bcast(rhs_sparse.data(), rhs_sparse.size(), MPI_DOUBLE, 0, m_mpiComm);

    //     for (size_t i = 0; i < elements.size(); ++i) {
    //         inv_elements.emplace_back(elements[i].first, elements[i].second, rhs_sparse[i]);
    //     }   // the elements indices were not changed, so I can just copy them from the input vector

    //     return inv_elements;
    // }

    ComputationInfo info() const {
        fdapde_assert(m_isInitialized && "Decomposition is not initialized.");
        return m_info;
    }
   protected:
    MumpsBase(MPI_Comm comm, unsigned int flags) :
        m_size(0),
        m_mpiComm(comm),
        m_info(InvalidInput),
        m_matrixDefined(false),
        m_analysisIsOk(false),
        m_factorizationIsOk(false),
        m_mumpsIcntl(m_mumps.icntl),
        m_mumpsCntl(m_mumps.cntl),
        m_mumpsInfo(m_mumps.info),
        m_mumpsInfog(m_mumps.infog),
        m_mumpsRinfo(m_mumps.rinfo),
        m_mumpsRinfog(m_mumps.rinfog) {
        m_verbose = (flags & Verbose) ? true : false;                     // default is non verbose
        m_computeDeterminant = !(flags & NoDeterminant) ? true : false;   // default is to compute the determinant
        m_mumps.par = (flags & WorkingHost) ? 1 : 0;                      // default is delegating host (par = 0)

        MPI_Manager::getInstance();   // initialize MPI if not already done, getInstance() will create th singleton only
                                      // if it doesn't exist (static function (always available even if the class
                                      // doesn't have an instance yet) that creates a static object)

        MPI_Comm_rank(m_mpiComm, &m_mpiRank);
        MPI_Comm_size(m_mpiComm, &m_mpiSize);

        m_mumps.comm_fortran = (MUMPS_INT)MPI_Comm_c2f(m_mpiComm);

        m_isInitialized = false;
    }

    // DESTRUCTOR
    virtual ~MumpsBase() { }

    // [MSG for A.Palummo] Ho aggiunto un check per evitare di ridefinire la matrice se è già stata definita. Questo è
    // risultato necessario perchè l'interfaccia richiesta da SparseSolverBase prevede che i metodi analyzePattern e
    // factorize vengano chiamati entrambi con la matrice come argument. Se la matrice è la setssa evito di ridefinire
    // tutte le strutture di mumps mentre se viene cambiata voglio segnalare a factorize che la nuova matrice non è
    // passata per analyzePattern. Questo mi permette anche di effettuare il reset di tutte le flag necessarie che prima
    // erano spacchettate nei vari metodi.
    // Per quanto riguarda la versione "iper-parallelizzata" la trova commantata dopo il metodo define_matrix, ho fatto
    // un paio di prove veloci con delle matrici prese da matrix market ma non ho riscontrato particolari differenze in
    // termini di velocità di computazione.
    virtual void define_matrix(const MatrixType& matrix) {
        fdapde_assert(matrix.rows() == matrix.cols() && "The matrix must be square");
        fdapde_assert(matrix.rows() > 0 && "The matrix must be non-empty");

        if (m_matrixDefined == true) {
            int exit_code = 0;
            if (getProcessRank() == 0) {
                if (m_size == matrix.rows() && m_matrix.isApprox(matrix)) { exit_code = 1; }
            }
            MPI_Bcast(&exit_code, 1, MPI_INT, 0, m_mpiComm);
            if (exit_code == 1) { return; }
        }

        m_matrixDefined = false;
        m_isInitialized = false;
        m_analysisIsOk = false;
        m_factorizationIsOk = false;

        m_size = matrix.rows();
        if (getProcessRank() == 0) {
            m_matrix = matrix;
            m_matrix.makeCompressed();

            m_colIndices.reserve(m_matrix.nonZeros());
            m_rowIndices.reserve(m_matrix.nonZeros());

            for (int k = 0; k < m_matrix.outerSize(); ++k) {   // already scales to 1-based indexing
                for (typename MatrixType::InnerIterator it(m_matrix, k); it; ++it) {
                    m_rowIndices.push_back(it.row() + 1);
                    m_colIndices.push_back(it.col() + 1);
                }
            }

            // defining the problem on the host
            m_mumps.n = m_matrix.rows();
            m_mumps.nnz = m_matrix.nonZeros();
            m_mumps.irn = m_rowIndices.data();
            m_mumps.jcn = m_colIndices.data();

            m_mumps.a = const_cast<Scalar*>(m_matrix.valuePtr());
        }
        m_matrixDefined = true;
    }

    // virtual void define_matrix(const MatrixType &matrix) {
    //   if (m_matrixDefined == true) {
    //       int exit_code = 0;
    //         if (getProcessRank() == 0) {
    //           if (m_size == matrix.rows() && m_matrix.isApprox(matrix)) { exit_code = 1; }
    //       }
    //       MPI_Bcast(&exit_code, 1, MPI_INT, 0, m_mpiComm);
    //       if (exit_code == 1) { return; }
    //   }

    //   m_matrixDefined = false;
    //   m_isInitialized = false;
    //   m_analysisIsOk = false;
    //   m_factorizationIsOk = false;
    //   MatrixType temp = matrix;
    //   temp.makeCompressed();
    //   m_size = temp.rows();
    //   if (getProcessRank() == 0) {
    //     m_matrix = temp;
    //   }

    //   std::vector<int> loc_row_indices;
    //   std::vector<int> loc_col_indices;

    //   int loc_start;
    //   int loc_end;

    //   int loc_size = temp.outerSize() / getProcessSize();
    //   int loc_size_remainder = temp.outerSize() % getProcessSize();

    //   if (getProcessRank() < loc_size_remainder) {
    //     loc_start = getProcessRank() * (loc_size + 1);
    //     loc_end = loc_start + loc_size + 1;
    //   } else {
    //     loc_start = getProcessRank() * loc_size + loc_size_remainder;
    //     loc_end = loc_start + loc_size;
    //   }

    //   loc_row_indices.reserve(temp.nonZeros());
    //   loc_col_indices.reserve(temp.nonZeros());

    //   for (int k = loc_start; k < loc_end; ++k) {
    //     for (typename MatrixType::InnerIterator it(temp, k); it; ++it) {
    //       loc_row_indices.push_back(it.row() + 1);
    //       loc_col_indices.push_back(it.col() + 1);
    //     }
    //   }

    //   // Gather local sizes first
    //   int local_size = loc_row_indices.size();
    //   std::vector<int> all_sizes(getProcessSize());
    //   MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, m_mpiComm);

    //   // Calculate displacements for gathering
    //   std::vector<int> displacements(getProcessSize());
    //   if (getProcessRank() == 0) {
    //     displacements[0] = 0;
    //     for (int i = 1; i < getProcessSize(); ++i) {
    //       displacements[i] = displacements[i - 1] + all_sizes[i - 1];
    //     }
    //     m_rowIndices.resize(std::accumulate(all_sizes.begin(), all_sizes.end(), 0));
    //     m_colIndices.resize(std::accumulate(all_sizes.begin(), all_sizes.end(), 0));
    //   }

    //   // Gather all local indices into the global vectors on the root process
    //   MPI_Gatherv(loc_row_indices.data(), local_size, MPI_INT, m_rowIndices.data(), all_sizes.data(),
    //               displacements.data(), MPI_INT, 0, m_mpiComm);

    //   MPI_Gatherv(loc_col_indices.data(), local_size, MPI_INT, m_colIndices.data(), all_sizes.data(),
    //               displacements.data(), MPI_INT, 0, m_mpiComm);

    //   if (getProcessRank() == 0) {
    //     m_mumps.n = m_matrix.rows();
    //     m_mumps.nnz = m_matrix.nonZeros();
    //     m_mumps.irn = m_rowIndices.data();
    //     m_mumps.jcn = m_colIndices.data();

    //     m_mumps.a = const_cast<Scalar *>(m_matrix.valuePtr());
    //   }
    // }

    void mumps_execute(const int job) const {
        if (m_mumps.job == -2 && job != -1) {
            if (getProcessRank() == 0) {
                std::cerr << "MUMPS is already finalized. You cannot execute any job." << std::endl;
            }
            exit(1);
        }
        m_mumps.job = job;
        dmumps_c(&m_mumps);

        if (job == -1) {
            if (getProcessRank() == 0) {
                m_mumps.icntl[0] = (m_verbose) ? 6 : -1;   // 6: verbose output, -1: no output
                m_mumps.icntl[1] = (m_verbose) ? 6 : -1;   // 6: verbose output, -1: no output
                m_mumps.icntl[2] = (m_verbose) ? 6 : -1;   // 6: verbose output, -1: no output
                m_mumps.icntl[3] = (m_verbose) ? 4 : 0;    // 4: verbose output, 0: no output
            } else {
                m_mumps.icntl[0] = -1;
                m_mumps.icntl[1] = -1;
                m_mumps.icntl[2] = -1;
                m_mumps.icntl[3] = 0;
            }

            m_mumps.icntl[32] =
              (m_computeDeterminant) ? 1 : 0;   // 1: compute determinant, 0: do not compute determinant
        }

        errorCheck();
    }

    virtual void analyzePattern_impl() {
        m_info = InvalidInput;
        mumps_execute(1);   // 1: analyze
        m_info = Success;
        m_isInitialized = true;
        m_analysisIsOk = true;
        // m_factorizationIsOk = false;  // I don't think this is necessary, the factorization should be invalidated
        // only if the matrix is changed (should i allow thE user to call analyzePattern, factorize and then
        // analyzePattern again all with the same matrix and have the factorization be valid for the solve?)
    }

    void factorize_impl() {
        fdapde_assert(m_analysisIsOk && "The matrix must be analyzed with analyzePattern() before calling factorize()");
        m_info = NumericalIssue;
        mumps_execute(2);   // 2: factorize
        m_info = Success;
        m_factorizationIsOk = true;
    }

    void compute_impl() {
        analyzePattern_impl();
        factorize_impl();
    }

    // METHODS FOR ERROR CHECKING (EXPLICIT ERROR MESSAGES)
    void errorCheck() const {
        if (mumpsInfog()[0] < 0) {
            if (!m_verbose && getProcessRank() == 0) {
                std::string job_name;
                switch (m_mumps.job) {
                case -2:
                    job_name = "Finalization";
                    break;
                case -1:
                    job_name = "Initialization";
                    break;
                case 1:
                    job_name = "Analysis";
                    break;
                case 2:
                    job_name = "Factorization";
                    break;
                case 3:
                    job_name = "Solve";
                    break;
                default:
                    job_name = "Unknown";
                }
                std::cerr << "Error occured during " + job_name + " phase." << std::endl;
                std::cerr << "MUMPS error mumpsInfog()[0]: " << mumpsInfog()[0] << std::endl;
                std::cerr << "MUMPS error mumpsInfog()[1]: " << mumpsInfog()[1] << std::endl;
            }
            exit(1);
        }
    }
   protected:
    mutable DMUMPS_STRUC_C m_mumps;

    // matrix data
    MatrixType m_matrix;
    Index m_size;   // only matrix related member defined on all processes
    std::vector<int> m_colIndices;
    std::vector<int> m_rowIndices;

    // mpi
    int m_mpiRank;
    int m_mpiSize;
    MPI_Comm m_mpiComm;

    // flags
    bool m_verbose;
    bool m_computeDeterminant;

    // computation flags
    bool m_matrixDefined;
    bool m_analysisIsOk;
    bool m_factorizationIsOk;

    mutable ComputationInfo m_info;

    // parameter wrappers
    MumpsIcntlArray m_mumpsIcntl;
    MumpsCntlArray m_mumpsCntl;
    MumpsInfoArray m_mumpsInfo;
    MumpsInfoArray m_mumpsInfog;
    MumpsRinfoArray m_mumpsRinfo;
    MumpsRinfoArray m_mumpsRinfog;
};

// MUMPS LU SOLVER
template <isEigenSparseMatrix MatrixType> class MumpsLU : public MumpsBase<MumpsLU<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsLU<MatrixType>>;
   public:
    // CONSTRUCTORS
    MumpsLU() : MumpsLU(MPI_COMM_WORLD, 0) { }
    explicit MumpsLU(MPI_Comm comm) : MumpsLU(comm, 0) { }
    explicit MumpsLU(unsigned int flags) : MumpsLU(MPI_COMM_WORLD, flags) { }

    MumpsLU(MPI_Comm comm, unsigned int flags) : Base(comm, flags) {
        // initialize MUMPS
        Base::m_mumps.sym = 0;
        Base::mumps_execute(-1);   // -1: initialization
    }

    explicit MumpsLU(const MatrixType& matrix) : MumpsLU(matrix, MPI_COMM_WORLD, 0) { }
    MumpsLU(const MatrixType& matrix, MPI_Comm comm) : MumpsLU(matrix, comm, 0) { }
    MumpsLU(const MatrixType& matrix, unsigned int flags) : MumpsLU(matrix, MPI_COMM_WORLD, flags) { }

    MumpsLU(const MatrixType& matrix, MPI_Comm comm, unsigned int flags) : MumpsLU(comm, flags) {
        Base::compute(matrix);
    }

    // DESTRUCTOR
    ~MumpsLU() override {
        // finalize MUMPS
        Base::mumps_execute(-2);   // -2: finalization
    }
};

// MUMPS LDLT SOLVER
template <isEigenSparseMatrix MatrixType, int Options> class MumpsLDLT : public MumpsBase<MumpsLDLT<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsLDLT<MatrixType>>;
   public:
    // CONSTUCTORS
    MumpsLDLT() : MumpsLDLT(MPI_COMM_WORLD, 0) { }
    explicit MumpsLDLT(MPI_Comm comm) : MumpsLDLT(comm, 0) { }
    explicit MumpsLDLT(unsigned int flags) : MumpsLDLT(MPI_COMM_WORLD, flags) { }

    MumpsLDLT(MPI_Comm comm, unsigned int flags) : Base(comm, flags) {
        // initialize MUMPS
        Base::m_mumps.sym = 1;     // symmetric and positive definite
        Base::mumps_execute(-1);   // -1: initialization
    }

    explicit MumpsLDLT(const MatrixType& matrix) : MumpsLDLT(matrix, MPI_COMM_WORLD, 0) { }
    MumpsLDLT(const MatrixType& matrix, MPI_Comm comm) : MumpsLDLT(matrix, comm, 0) { }
    MumpsLDLT(const MatrixType& matrix, unsigned int flags) : MumpsLDLT(matrix, MPI_COMM_WORLD, flags) { }

    MumpsLDLT(const MatrixType& matrix, MPI_Comm comm, unsigned int flags) : MumpsLDLT(comm, flags) {
        Base::compute(matrix);
    }

    // DESTRUCTOR
    ~MumpsLDLT() override {
        // finalize MUMPS
        Base::mumps_execute(-2);   // -2: finalization
    }

    // METHOD FOR DEFINING THE MATRIX ON THE HOST PROCESS (UPPER/LOWER TRIANGULAR)
    void define_matrix(const MatrixType& matrix) override {
        Base::define_matrix(matrix.template triangularView<Options>());
    }
};

// MUMPS BLR SOLVER

// USAGE: flags = [UFSC | UCFS] | [uncompressed_CB | compressed_CB] ---- (flags can be omitted and default will be used)

// estimated compression rate of LU factors
// mumpsIcntl()[37] = ...
// between 0 and 1000 (default: 600)
// mumpsIcntl()[37]/10 is a percentage representing the typical compression of the compressed factors factor matrices in
// BLR fronts

// estimated compression rate of contribution blocks
// mumpsIcntl()[38] = ...
// between 0 and 1000 (default: 500)
// mumpsIcntl()[38]/10 is a percentage representing the typical compression of the compressed CB CB in BLR fronts

template <isEigenSparseMatrix MatrixType> class MumpsBLR : public MumpsBase<MumpsBLR<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsBLR<MatrixType>>;
   public:
    // CONSTRUCTORS
    MumpsBLR() : MumpsBLR(MPI_COMM_WORLD, 0, 0) { }
    explicit MumpsBLR(MPI_Comm comm) : MumpsBLR(comm, 0, 0) { }
    explicit MumpsBLR(unsigned int flags) : MumpsBLR(MPI_COMM_WORLD, flags, 0) { }
    explicit MumpsBLR(double dropping_parameter) : MumpsBLR(MPI_COMM_WORLD, 0, dropping_parameter) { }
    MumpsBLR(MPI_Comm comm, unsigned int flags) : MumpsBLR(comm, flags, 0) { }
    MumpsBLR(unsigned int flags, double dropping_parameter) : MumpsBLR(MPI_COMM_WORLD, flags, dropping_parameter) { }
    MumpsBLR(MPI_Comm comm, double dropping_parameter) : MumpsBLR(comm, 0, dropping_parameter) { }

    MumpsBLR(MPI_Comm comm, unsigned int flags, double dropping_parameter) : Base(comm, flags) {
        fdapde_assert(!(flags & UFSC && flags & UCFS) && "UFSC and UCFS cannot be set at the same time");

        // initialize MUMPS
        Base::m_mumps.sym = 0;
        Base::mumps_execute(-1);   // -1: initialization

        // activate the BLR feature & set the BLR parameters
        Base::mumpsIcntl()[34] = 1;   // 1: automatic choice of the BLR memory management strategy
        Base::mumpsIcntl()[35] =
          (flags & UCFS) ? 1 : 0;   // set mumpsIcntl()[35] = 1 if UCFS is set, otherwise it remains 0 (default -> UFSC)
        Base::mumpsIcntl()[36] = (flags & Compressed) ? 1 : 0;   // set mumpsIcntl()[36] = 1 if compressed_CB is set,
                                                                 // otherwise it remains 0 (default -> uncompressed_CB)
        Base::mumpsCntl()[6] = dropping_parameter;               // set the dropping parameter
    }

    explicit MumpsBLR(const MatrixType& matrix) : MumpsBLR(matrix, MPI_COMM_WORLD, 0, 0) { }
    MumpsBLR(const MatrixType& matrix, MPI_Comm comm) : MumpsBLR(matrix, comm, 0, 0) { }
    MumpsBLR(const MatrixType& matrix, unsigned int flags) : MumpsBLR(matrix, MPI_COMM_WORLD, flags, 0) { }
    MumpsBLR(const MatrixType& matrix, double dropping_parameter) :
        MumpsBLR(matrix, MPI_COMM_WORLD, 0, dropping_parameter) { }
    MumpsBLR(const MatrixType& matrix, MPI_Comm comm, unsigned int flags) : MumpsBLR(matrix, comm, flags, 0) { }
    MumpsBLR(const MatrixType& matrix, MPI_Comm comm, double dropping_parameter) :
        MumpsBLR(matrix, comm, 0, dropping_parameter) { }
    MumpsBLR(const MatrixType& matrix, unsigned int flags, double dropping_parameter) :
        MumpsBLR(matrix, MPI_COMM_WORLD, flags, dropping_parameter) { }

    MumpsBLR(const MatrixType& matrix, MPI_Comm comm, unsigned int flags, double dropping_parameter) :
        MumpsBLR(comm, flags, dropping_parameter) {
        Base::compute(matrix);
    }

    // DESTRUCTOR
    ~MumpsBLR() override {
        // finalize MUMPS
        Base::mumps_execute(-2);   // -2: finalization
    }
};

// MUMPS RANK REVEALING SOLVER
template <isEigenSparseMatrix MatrixType> class MumpsRR : public MumpsBase<MumpsRR<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsRR<MatrixType>>;
   public:
    using Scalar = typename Base::Scalar;

    using Base::solve;

    // CONSTRUCTOR
    MumpsRR() : MumpsRR(MPI_COMM_WORLD, 0) { }
    explicit MumpsRR(MPI_Comm comm) : MumpsRR(comm, 0) { }
    explicit MumpsRR(unsigned int flags) : MumpsRR(MPI_COMM_WORLD, flags) { }

    MumpsRR(MPI_Comm comm, unsigned int flags) : Base(comm, flags) {
        // initialize MUMPS
        Base::m_mumps.sym = 0;
        Base::mumps_execute(-1);   // -1: initialization

        Base::mumpsIcntl()[23] = 1;   // 1: null pivot detection
        Base::mumpsIcntl()[55] = 1;   // 1: perform rank revealing factorization
    }

    explicit MumpsRR(const MatrixType& matrix) : MumpsRR(matrix, MPI_COMM_WORLD, 0) { }
    MumpsRR(const MatrixType& matrix, MPI_Comm comm) : MumpsRR(matrix, comm, 0) { }
    MumpsRR(const MatrixType& matrix, unsigned int flags) : MumpsRR(matrix, MPI_COMM_WORLD, flags) { }

    MumpsRR(const MatrixType& matrix, MPI_Comm comm, unsigned int flags) : MumpsRR(comm, flags) {
        Base::compute(matrix);
    }

    // DESTRUCTOR
    ~MumpsRR() override {
        // finalize MUMPS
        Base::mumps_execute(-2);
    }

    // NULL SPACE SIZE METHODS
    int nullSpaceSize() {
        fdapde_assert(
          Base::m_factorizationIsOk && "The matrix must be factorized with factorize() before calling nullSpaceSize()");
        return Base::mumpsInfog()[27];
    }

    // RANK METHODS
    int rank() {
        fdapde_assert(
          Base::m_factorizationIsOk && "The matrix must be factorized with factorize() before calling rank()");
        return Base::m_size - Base::mumpsInfog()[27];
    }

    // NULL SPACE BASIS METHODS)
    MatrixXd nullSpaceBasis() {
        fdapde_assert(
          Base::m_factorizationIsOk &&
          "The matrix must be factorized with factorize() before calling nullSpaceBasis()");

        int null_space_size_ = Base::mumpsInfog()[27];

        if (null_space_size_ == 0) { return MatrixXd::Zero(Base::m_size, 0); }

        // allocate memory for the null space basis
        std::vector<Scalar> buff;
        buff.resize(Base::m_size * null_space_size_);

        if (Base::getProcessRank() == 0) {
            Base::m_mumps.nrhs = null_space_size_;
            Base::m_mumps.lrhs = Base::m_size;
            Base::m_mumps.rhs = buff.data();
        }

        Base::mumpsIcntl()[24] = -1;   // -1: perform a null space basis computation step
        Base::mumps_execute(3);        // 3 : solve
        Base::mumpsIcntl()[24] = 0;    // reset mumpsIcntl()[24] to 0 -> a normal solve can be called

        MPI_Bcast(buff.data(), buff.size(), MPI_DOUBLE, 0, Base::m_mpiComm);

        // --> matrix columns are the null space basis vectors
        return Map<MatrixXd>(buff.data(), Base::m_size, null_space_size_);
    }
};

// CLASS FOR COMPUTING THE SHUR COMPLEMENT
template <isEigenSparseMatrix MatrixType> class MumpsSchur : public MumpsBase<MumpsSchur<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsSchur<MatrixType>>;
   public:
    using Scalar = typename Base::Scalar;

    // CONSTRUCTOR
    MumpsSchur() : MumpsSchur(MPI_COMM_WORLD, 0) { }
    explicit MumpsSchur(MPI_Comm comm) : MumpsSchur(comm, 0) { }
    explicit MumpsSchur(unsigned int flags) : MumpsSchur(MPI_COMM_WORLD, flags) { }

    MumpsSchur(MPI_Comm comm, unsigned int flags) : Base(comm, flags), m_schurIndicesSet(false) {
        // initialize MUMPS
        Base::m_mumps.sym = 0;
        Base::mumps_execute(-1);   // -1: initialization

        Base::mumpsIcntl()[18] = 3;   // 3: distributed by columns

        // changing parameters to centralize it according to Mumps documentation
        Base::m_mumps.nprow = 1;
        Base::m_mumps.npcol = 1;
        Base::m_mumps.mblock = 100;
        Base::m_mumps.nblock = 100;
    }

    // [MSG for A.Palummo] Ho cambiato gli schur indices da vector a set per funzionare con la nuova interfaccia di
    // setSchurIndices. Ho lasciato la versione con il vector commentata nel caso preferisca usare quella.
    template <isIntSet SetType>
    MumpsSchur(const MatrixType& matrix, const SetType& schur_indices) :
        MumpsSchur(matrix, schur_indices, MPI_COMM_WORLD, 0) { }
    template <isIntSet SetType>
    MumpsSchur(const MatrixType& matrix, const SetType& schur_indices, MPI_Comm comm) :
        MumpsSchur(matrix, schur_indices, comm, 0) { }
    template <isIntSet SetType>
    MumpsSchur(const MatrixType& matrix, const SetType& schur_indices, unsigned int flags) :
        MumpsSchur(matrix, schur_indices, MPI_COMM_WORLD, flags) { }
    template <isIntSet SetType>
    MumpsSchur(const MatrixType& matrix, const SetType& schur_indices, MPI_Comm comm, unsigned int flags) :
        MumpsSchur(comm, flags) {
        setSchurIndices(schur_indices);
        Base::compute(matrix);
    }

    // MumpsSchur(const MatrixType& matrix, const std::vector<int>& schur_indices) :
    //     MumpsSchur(matrix, schur_indices, MPI_COMM_WORLD, 0) { }
    // MumpsSchur(const MatrixType& matrix, const std::vector<int>& schur_indices, MPI_Comm comm) :
    //     MumpsSchur(matrix, schur_indices, comm, 0) { }
    // MumpsSchur(const MatrixType& matrix, const std::vector<int>& schur_indices, unsigned int flags) :
    //     MumpsSchur(matrix, schur_indices, MPI_COMM_WORLD, flags) { }

    // MumpsSchur(const MatrixType& matrix, const std::vector<int>& schur_indices, MPI_Comm comm, unsigned int flags) :
    //     MumpsSchur(comm, flags) {
    //     setSchurIndices(schur_indices);
    //     Base::compute(matrix);
    // }

    // DESTRUCTOR
    ~MumpsSchur() override {
        // finalize MUMPS
        Base::mumps_execute(-2);   // -2: finalization
    }

    // [MSG for A.Palummo] Ho cambiato l'input da vector a set per evitare che l'utente possa inserire indici duplicati
    // e per garantire che siano ordinati in modo crescente. Questo mi permette di evitare di fare controlli con assert.
    // Devo però mantenere m_schurIndices come vector perchè ho bisogno di posizioni contigue in memmoria per mumps, che
    // vuole come input il puntatore c al primo elemento. Le lascio la versione con il vector in input commentata se
    // decide di voler usare quella. (i test e utils.h devono essere cambiati di conseguenza, ho lasciato il codice
    // commantato nei test e in utils.h nel caso si voglia usare il vector)

    // split into template + overloading to allow any key_compare in the input set but still ensure that my function
    // recieves the ordering i need (in this case std::less<int>)
    // overloading of course is preferred to template if the SetType is std::set<int>
    template <isIntSet SetType> MumpsSchur<MatrixType>& setSchurIndices(const SetType& schur_indices) {
        std::set<int> idx(schur_indices.begin(), schur_indices.end());
        return setSchurIndices(idx);
    }

    MumpsSchur<MatrixType>& setSchurIndices(const std::set<int>& schur_indices) {
        fdapde_assert(schur_indices.size() > 0 && "The Schur complement size must be greater than 0");
        fdapde_assert(*schur_indices.begin() >= 0 && "The Schur indices must be positive");

        Base::m_mumps.size_schur = schur_indices.size();

        m_schurIndices.reserve(Base::m_mumps.size_schur);
        m_schurIndices.assign(schur_indices.begin(), schur_indices.end());
        std::for_each(m_schurIndices.begin(), m_schurIndices.end(), [](int& idx) { idx += 1; });

        m_schurBuff.resize(Base::m_mumps.size_schur * Base::m_mumps.size_schur);
        Base::m_mumps.listvar_schur = m_schurIndices.data();

        // I need to define these on the first working processor: PAR=1 -> rank 0, PAR=0 -> rank 1
        if (Base::getProcessRank() == (Base::m_mumps.par + 1) % 2) {
            Base::m_mumps.schur_lld = Base::m_mumps.size_schur;
            Base::m_mumps.schur = m_schurBuff.data();
        }

        m_schurIndicesSet = true;

        return *this;
    }

    // MumpsSchur<MatrixType>& setSchurIndices(const std::vector<int>& schur_indices) {
    //     eigen_assert(schur_indices.size() > 0 && "The Schur complement size must be greater than 0");
    //     eigen_assert(
    //       std::is_sorted(schur_indices.begin(), schur_indices.end()) &&
    //       "The Schur indices must be sorted in ascending order");
    //     eigen_assert(
    //       std::adjacent_find(schur_indices.begin(), schur_indices.end()) == schur_indices.end() &&
    //       "The Schur indices must be unique");
    //     eigen_assert(schur_indices.front() >= 0 && "The Schur indices must be positive");

    //     Base::m_mumps.size_schur = schur_indices.size();

    //     m_schurIndices = schur_indices;
    //     std::for_each(m_schurIndices.begin(), m_schurIndices.end(), [](int& idx) { idx += 1; });
    //     m_schurBuff.resize(Base::m_mumps.size_schur * Base::m_mumps.size_schur);
    //     Base::m_mumps.listvar_schur = m_schurIndices.data();

    //     // I need to define these on the first working processor: PAR=1 -> rank 0, PAR=0 -> rank 1
    //     if (Base::getProcessRank() == (Base::m_mumps.par + 1) % 2) {
    //         Base::m_mumps.schur_lld = Base::m_mumps.size_schur;
    //         Base::m_mumps.schur = m_schurBuff.data();
    //     }

    //     m_schurIndicesSet = true;

    //     return *this;
    // }

    MatrixXd complement() {
        fdapde_assert(
          Base::m_factorizationIsOk && "The matrix must be factorized with factorize() before calling complement()");
        MPI_Bcast(m_schurBuff.data(), m_schurBuff.size(), MPI_DOUBLE, (Base::m_mumps.par + 1) % 2, Base::m_mpiComm);
        return Map<MatrixXd>(m_schurBuff.data(), Base::m_mumps.size_schur, Base::m_mumps.size_schur);
    }
   protected:
    void define_matrix(const MatrixType& matrix) override {
        fdapde_assert(
          Base::m_mumps.size_schur < matrix.rows() && "The Schur complement size must be smaller than the matrix size");
        fdapde_assert(
          Base::m_mumps.size_schur < matrix.cols() && "The Schur complement size must be smaller than the matrix size");
        fdapde_assert(
          m_schurIndices.back() <= matrix.rows() &&
          "The Schur indices must be within the matrix size");   // m_schurIndices is 1-based
        fdapde_assert(
          m_schurIndices.back() <= matrix.cols() &&
          "The Schur indices must be within the matrix size");   // m_schurIndices is 1-based

        Base::define_matrix(matrix);
    }

    void analyzePattern_impl() override {
        fdapde_assert(
          m_schurIndicesSet && "The Schur indices must be set with setSchurIndices() before calling analyzePattern()");
        Base::analyzePattern_impl();
    }
   protected:
    std::vector<int> m_schurIndices;
    std::vector<Scalar> m_schurBuff;

    bool m_schurIndicesSet;
};

}   // namespace mumps

}   // namespace fdapde

#endif   // __MUMPS_H__
