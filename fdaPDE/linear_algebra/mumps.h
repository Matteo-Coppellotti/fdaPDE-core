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
template <typename T>
concept isEigenDenseMatrix = std::derived_from<T, MatrixBase<T>>;

class MPI_Manager {   // SINGLETON CLASS
   public:
    static MPI_Manager& getInstance() {
        static MPI_Manager instance;   // Initialized once
        return instance;
    }

    // Delete copy and move constructors and assignment operators
    MPI_Manager(const MPI_Manager&) = delete;
    MPI_Manager& operator=(const MPI_Manager&) = delete;
    MPI_Manager(MPI_Manager&&) = delete;
    MPI_Manager& operator=(MPI_Manager&&) = delete;

    // Public method to check MPI state, could be useful
    bool isMPIinitialized() const { return mpi_initialized; }
   private:
    bool mpi_initialized;

    // Private constructor initializes MPI
    MPI_Manager() : mpi_initialized(false) {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(NULL, NULL);
            mpi_initialized = true;
        }
    }

    // Private destructor finalizes MPI
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
        requires isEigenDenseMatrix<BDerived> && isEigenDenseMatrix<XDerived>
    void _solve_impl(const MatrixBase<BDerived>& b, MatrixBase<XDerived>& x) const {
        fdapde_assert(
          m_factorizationIsOk && "The matrix must be factorized with factorize(matrix) before calling solve(rhs)");

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

    Scalar determinant() {
        fdapde_assert(m_computeDeterminant && "The determinant computation must be enabled");
        fdapde_assert(
          m_factorizationIsOk && "The matrix must be factoried with factorize_impl() before calling determinant()");
        if (mumpsInfog()[27] != 0) { return 0; }
        return mumpsRinfog()[11] * std::pow(2, mumpsInfog()[33]);
    }

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

        MPI_Manager::getInstance();   // initialize MPI

        MPI_Comm_rank(m_mpiComm, &m_mpiRank);
        MPI_Comm_size(m_mpiComm, &m_mpiSize);

        m_mumps.comm_fortran = (MUMPS_INT)MPI_Comm_c2f(m_mpiComm);

        m_isInitialized = false;
    }

    // DESTRUCTOR
    virtual ~MumpsBase() { }

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
    //     for (SparseMatrix<double>::InnerIterator it(temp, k); it; ++it) {
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
        fdapde_assert(
          m_matrixDefined &&
          "The matrix must be defined first with private method define_matrix() before calling analyze()");
        m_info = InvalidInput;
        mumps_execute(1);   // 1: analyze
        m_info = Success;
        m_isInitialized = true;
        m_analysisIsOk = true;
        m_factorizationIsOk = false;
    }

    void factorize_impl() {
        fdapde_assert(
          m_analysisIsOk &&
          "The matrix pattern must be analyzed first with analyzePattern() before calling factorize()");
        m_info = NumericalIssue;
        mumps_execute(2);   // 2: factorize
        m_info = Success;
        m_factorizationIsOk = true;
    }

    virtual void compute_impl() {
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
        Base::mumps_execute(-1);
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
        Base::mumps_execute(-2);
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
        Base::m_mumps.sym = 1;   // -> symmetric and positive definite
        Base::mumps_execute(-1);
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
        Base::mumps_execute(-2);
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
        Base::mumps_execute(-1);

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
        Base::mumps_execute(-2);
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
        Base::mumps_execute(-1);

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
          Base::m_factorizationIsOk &&
          "The matrix must be factorize_impld with factorize_impl() before calling nullSpaceSize()");
        return Base::mumpsInfog()[27];
    }

    // RANK METHODS
    int rank() {
        fdapde_assert(
          Base::m_factorizationIsOk &&
          "The matrix must be factorize_impld with factorize_impl() before calling rank()");
        return Base::m_size - Base::mumpsInfog()[27];
    }

    // NULL SPACE BASIS METHODS)
    MatrixXd nullSpaceBasis() {
        fdapde_assert(
          Base::m_factorizationIsOk &&
          "The matrix must be factorize_impld with factorize_impl() before calling nullSpaceBasis()");

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
        Base::m_mumps.sym = 0;
        Base::mumps_execute(-1);   // initialize MUMPS

        Base::mumpsIcntl()[18] = 3;   // 3: distributed by columns (changing parameters to cenralize it)
        Base::m_mumps.nprow = 1;
        Base::m_mumps.npcol = 1;
        Base::m_mumps.mblock = 100;
        Base::m_mumps.nblock = 100;
    }

    MumpsSchur(const MatrixType& matrix, const std::vector<int>& schur_indices) :
        MumpsSchur(matrix, schur_indices, MPI_COMM_WORLD, 0) { }
    MumpsSchur(const MatrixType& matrix, const std::vector<int>& schur_indices, MPI_Comm comm) :
        MumpsSchur(matrix, schur_indices, comm, 0) { }
    MumpsSchur(const MatrixType& matrix, const std::vector<int>& schur_indices, unsigned int flags) :
        MumpsSchur(matrix, schur_indices, MPI_COMM_WORLD, flags) { }

    MumpsSchur(const MatrixType& matrix, const std::vector<int>& schur_indices, MPI_Comm comm, unsigned int flags) :
        MumpsSchur(comm, flags) {
        setSchurIndices(schur_indices);
        Base::compute(matrix);
    }

    // DESTRUCTOR
    ~MumpsSchur() override {
        Base::mumps_execute(-2);   // finalize MUMPS
    }

    MumpsSchur<MatrixType>& setSchurIndices(const std::vector<int>& schur_indices) {
        fdapde_assert(schur_indices.size() > 0 && "The Schur complement size must be greater than 0");
        fdapde_assert(
          std::is_sorted(schur_indices.begin(), schur_indices.end()) &&
          "The Schur indices must be sorted in ascending order");
        fdapde_assert(
          std::adjacent_find(schur_indices.begin(), schur_indices.end()) == schur_indices.end() &&
          "The Schur indices must be unique");
        fdapde_assert(schur_indices.front() >= 0 && "The Schur indices must be positive");

        Base::m_mumps.size_schur = schur_indices.size();

        m_schurIndices = schur_indices;
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

    MatrixXd complement() {
        fdapde_assert(
          Base::m_factorizationIsOk && "The matrix must be factorized with factorize() before calling complement()");
        MPI_Bcast(m_schurBuff.data(), m_schurBuff.size(), MPI_DOUBLE, (Base::m_mumps.par + 1) % 2, Base::m_mpiComm);
        return Map<MatrixXd>(m_schurBuff.data(), Base::m_mumps.size_schur, Base::m_mumps.size_schur);
    }
   protected:
    void define_matrix(const MatrixType& matrix) override {
        fdapde_assert(
          matrix.rows() > Base::m_mumps.size_schur && "The Schur complement size must be smaller than the matrix size");
        fdapde_assert(
          matrix.cols() > Base::m_mumps.size_schur && "The Schur complement size must be smaller than the matrix size");
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

    void compute_impl() override {
        fdapde_assert(
          m_schurIndicesSet && "The Schur indices must be set with setSchurIndices() before calling compute()");
        Base::compute_impl();
    }
   protected:
    std::vector<int> m_schurIndices;
    std::vector<Scalar> m_schurBuff;

    bool m_schurIndicesSet;
};

}   // namespace mumps

}   // namespace fdapde

#endif   // __MUMPS_H__
