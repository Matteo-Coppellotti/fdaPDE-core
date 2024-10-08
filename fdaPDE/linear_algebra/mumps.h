#ifndef __MUMPS_H__
#define __MUMPS_H__

#include <Eigen/Sparse>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <concepts>

#include <mpi.h>
#include <Mumps/dmumps_c.h>

using namespace Eigen;
using namespace internal;


// CONCEPTS
template <typename T>
concept isEigenSparseMatrix = std::derived_from<T, SparseMatrixBase<T>>;

template <typename T>
concept isVector = requires(T t) {
    { t.size() }
    ->std::convertible_to<int>;
    { t.data() }
    ->std::convertible_to<double *>;
};

template <typename T>
concept isScalar = std::is_arithmetic<T>::value;


// FORWARD DECLARATIONS
template <isEigenSparseMatrix MatrixType_>
class MumpsLU;
template <isEigenSparseMatrix MatrixType_, int Options = Upper>
class MumpsLDLT;
template <isEigenSparseMatrix MatrixType_>
class MumpsBLR;
template <isEigenSparseMatrix MatrixType_>
class MumpsRankRevealing;
template <isEigenSparseMatrix MatrixType_>
class MumpsSchurComplement;


// MUMPS BASE CLASS
template <isEigenSparseMatrix MatrixType_> //--> CRTP???? (with this as base class) ---> concept ????
class MumpsBase {

    using Scalar = typename MatrixType_::Scalar;
    using StorageIndex = typename MatrixType_::StorageIndex;

public:

    // SETTERS AND GETTERS WITH SCALED INDEXES S.T. IT REFLECTS THE MUMPS DOCUMENTATION

    // SETTERS
    inline int& ICNTL(int index) { return mumps_.icntl[index - 1]; }
    inline double& CNTL(int index) { return mumps_.cntl[index - 1]; }

    // GETTERS
    inline int ICNTL(int index) const { return mumps_.icntl[index - 1]; }
    inline double CNTL(int index) const { return mumps_.cntl[index - 1]; }
    inline int INFO(int index) const { return mumps_.info[index - 1]; }
    inline double RINFO(int index) const { return mumps_.rinfo[index - 1]; }
    inline int INFOG(int index) const { return mumps_.infog[index - 1]; }
    inline double RINFOG(int index) const { return mumps_.rinfog[index - 1]; }


    // COMPUTE METHOD
    void compute(const MatrixType_ &matrix) {

        ICNTL(1) = (verbose_) ? 6 : -1; // 6: verbose output, -1: no output
        ICNTL(2) = (verbose_) ? 6 : -1; // 6: verbose output, -1: no output
        ICNTL(3) = (verbose_) ? 6 : -1; // 6: verbose output, -1: no output
        ICNTL(4) = (verbose_) ? 4 : 0; // 4: verbose output, 0: no output

        eigen_assert(matrix.rows() == matrix.cols() && "The matrix must be square");

        define_matrix(matrix);

        mumps_execute(1); // 1: analyze

        ICNTL(33) = (compute_determinant_) ? 1 : 0; // 1: compute determinant, 0: do not compute determinant

        mumps_execute(2); // 2: factorize

        determinant_computed_ = compute_determinant_;

        matrix_computed_ = true;
    }

    // GENERAL SOLVE METHOD
    template <isVector V>
    V solve(const V &rhs) {
        //fdapde_assert(rhs.size() == n_);
        eigen_assert(matrix_computed_ && "The matrix must be computed with compute(Matrix) before calling solve(rhs)");
        eigen_assert(rhs.size() == n_ && "The size of the right-hand side vector must match the size of the matrix");

        V buff = rhs;

        // defining the problem on the host
        if (rank_ == 0) {
            mumps_.rhs = const_cast<Scalar *>(buff.data());
        }

        mumps_execute(3); // 3: solve

        MPI_Bcast(buff.data(), buff.size(), MPI_DOUBLE, 0, comm_);

        return buff;
    }

    // DETERMINANT COMPUTATION METHOD
    Scalar determinant() {
        eigen_assert(compute_determinant_ && "The determinant computation must be enabled");
        eigen_assert(matrix_computed_ && "The matrix must be computed with compute(Matrix) before calling determinant()");
        eigen_assert(determinant_computed_ && "The determinant computation must be enabled before calling compute(Matrix)");
        return RINFOG(12) * pow(2, INFOG(34));
    }

    // TOGGLE MUMPS MESSAGES
    // CHECK IF THE FORTRAN OUTPUT IS REDIRECTED TO THE C++ OUTPUT -----!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    void toggle_mumps_messages() { // off by default
        verbose_ = !verbose_;
        if (rank_ == 0) std::cout << "MUMPS messages are now " << (verbose_ ? "ON" : "OFF") << std::endl;
    }

    // TOGGLE DETERMINANT COMPUTATION
    void toggle_determinant_computation() { // on by default
        compute_determinant_ = !compute_determinant_;
        if (rank_ == 0) std::cout << "Determinant computation is now " << (compute_determinant_ ? "ON" : "OFF") << std::endl;
    }


protected:

    static constexpr int JOB_INIT = -1;
    static constexpr int JOB_END = -2;

    static constexpr int StorageOrder = MatrixType_::IsRowMajor ? RowMajor : ColMajor;

    // mumps struct
    DMUMPS_STRUC_C mumps_;

    // matrix useful info (available on all processes, the mumps info for matrix and rhs should be available only on the host)
    int n_;
    std::vector<int> col_indices_;
    std::vector<int> row_indices_;

    // mpi
    MPI_Comm comm_;
    int mpi_initialized_;
    int rank_;

    // flags
    bool verbose_ = false;
    bool matrix_computed_ = false;
    bool compute_determinant_ = true;
    bool determinant_computed_ = false;

    // CONSTRUCTORS --> protected to prevent instantiation of the base class
    MumpsBase(): MumpsBase(MPI_COMM_WORLD) {}
    explicit MumpsBase(MPI_Comm comm) : comm_(comm) {
        // if MPI isn't initialized, initialize it
        MPI_Initialized(&mpi_initialized_);  // --> CONSDER ADDING mpi_err = AT THE START OF ALL THE MPI CALLS TO CHECK FOR ERRORS
        if (!mpi_initialized_) {
            MPI_Init(NULL, NULL);
        }

        MPI_Comm_rank(comm_, &rank_);

        mumps_.comm_fortran = (MUMPS_INT) MPI_Comm_c2f(comm_);
    }

    // DESTRUCTOR
    virtual ~MumpsBase() {
        // if MPI was initialized by this class, finalize it
        if (!mpi_initialized_) {
            MPI_Finalize();
        }
    }

    // INDEX SCALING FOR MUMPS (MUMPS uses 1-based indexing)
    std::vector<int> mumps_index_scaling(const StorageIndex* begin, const StorageIndex* end) const {
        std::vector<int> indices;
        indices.reserve(end - begin);
        std::copy(begin, end, std::back_inserter(indices));
        std::for_each(indices.begin(), indices.end(), [](int &idx)
                        { idx += 1; });
        return indices;
    }

    // DEFINING THE MATRIX ON THE HOST PROCESS
    virtual void define_matrix(const MatrixType_ &matrix) {

        n_ = matrix.rows();

        if (rank_ == 0) {
        // scaling the indexes for MUMPS
        col_indices_ = (StorageOrder == ColMajor) ? 
            mumps_index_scaling(matrix.innerIndexPtr(), matrix.innerIndexPtr() + matrix.nonZeros())
            : mumps_index_scaling(matrix.outerIndexPtr(), matrix.outerIndexPtr() + matrix.nonZeros());
        row_indices_ = (StorageOrder == ColMajor) ? 
            mumps_index_scaling(matrix.outerIndexPtr(), matrix.outerIndexPtr() + matrix.nonZeros())
            : mumps_index_scaling(matrix.innerIndexPtr(), matrix.innerIndexPtr() + matrix.nonZeros());

        // defining the problem on the host
            mumps_.n = matrix.rows();
            mumps_.nnz = matrix.nonZeros();
            mumps_.irn = row_indices_.data();
            mumps_.jcn = col_indices_.data();

            mumps_.a = const_cast<Scalar *>(matrix.valuePtr());
        }
    }

    // METHOD FOR EXECUTING MUMPS JOBS
    void mumps_execute(const int job) {
        mumps_.job = job;
        dmumps_c(&mumps_);
        if (rank_ == 0)
            errorCheck();
    }

    // METHODS FOR ERROR CHECKING (EXPLICIT ERROR MESSAGES)
    void errorCheck() {
        if (INFOG(1) < 0) {
            std::string job_name;
            switch (mumps_.job) {
            case JOB_INIT:
                job_name = "initialization";
                break;
            case 1:
                job_name = "analysis";
                break;
            case 2:
                job_name = "factorization";
                break;
            case 3:
                job_name = "solve";
                break;
            case JOB_END:
                job_name = "finalization";
                break;
            default:
                job_name = "unknown";
            }
            std::cerr << "Error occured during " + job_name + " phase" << std::endl;
            std::cerr << "MUMPS error INFOG(1): " << INFOG(1) << std::endl;
            std::cerr << "MUMPS error INFOG(2): " << INFOG(2) << std::endl;
            exit(1);
        }
    }
};



// MUMPS LU SOLVER
template <isEigenSparseMatrix MatrixType_>
class MumpsLU : public MumpsBase< MatrixType_> {

    using Scalar = typename MatrixType_::Scalar;

public:

    // CONSTRUCTOR
    MumpsLU() : MumpsLU(MPI_COMM_WORLD) {}
    explicit MumpsLU(MPI_Comm comm): MumpsBase<MatrixType_>(comm) {
        // initialize MUMPS
        this->mumps_.par = 0;
        this->mumps_.sym = 0;
        this->mumps_execute(this->JOB_INIT);
    }

    // DESTRUCTOR
    ~MumpsLU() override {
        // finalize MUMPS
        this->mumps_execute(this->JOB_END);
    }
};



// MUMPS LDLT SOLVER
template <isEigenSparseMatrix  MatrixType_, int Options>
class MumpsLDLT : public MumpsBase< MatrixType_> {

    using Scalar = typename MatrixType_::Scalar;

public:

    // CONSTUCTOR
    MumpsLDLT(): MumpsLDLT(MPI_COMM_WORLD) {}
    explicit MumpsLDLT(MPI_Comm comm): MumpsBase< MatrixType_>(comm) {
        // initialize MUMPS
        this->mumps_.par = 0;
        this->mumps_.sym = 1; // -> symmetric and positive definite
        this->mumps_execute(this->JOB_INIT);
    }

    // DESTRUCTOR
    ~MumpsLDLT() override {
        // finalize MUMPS
        this->mumps_execute(this->JOB_END);
    }

protected:

    MatrixType_ t_matrix_;

    // METHOD FOR DEFINING THE MATRIX ON THE HOST PROCESS (UPPER/LOWER TRIANGULAR)
    void define_matrix(const MatrixType_ &matrix) override {
        eigen_assert(matrix.rows() == matrix.cols() && "The matrix must be square");

        this->n_ = matrix.rows();

        if (this->rank_ == 0) {
            t_matrix_ = matrix.template selfadjointView<Options>();

            // scaling the indexes for MUMPS
            this->col_indices_ = (this->StorageOrder == ColMajor) ? 
                this->mumps_index_scaling(t_matrix_.innerIndexPtr(), t_matrix_.innerIndexPtr() + t_matrix_.nonZeros())
                : this->mumps_index_scaling(t_matrix_.outerIndexPtr(), t_matrix_.outerIndexPtr() + t_matrix_.nonZeros());
            this->row_indices_ = (this->StorageOrder == ColMajor) ? 
                this->mumps_index_scaling(t_matrix_.outerIndexPtr(), t_matrix_.outerIndexPtr() + t_matrix_.nonZeros())
                : this->mumps_index_scaling(t_matrix_.innerIndexPtr(), t_matrix_.innerIndexPtr() + t_matrix_.nonZeros());


            // defining the problem on the host
            this->mumps_.n = t_matrix_.rows();
            this->mumps_.nnz = t_matrix_.nonZeros();
            this->mumps_.irn = this->row_indices_.data();
            this->mumps_.jcn = this->col_indices_.data();

            this->mumps_.a = const_cast<Scalar *>(t_matrix_.valuePtr());
        }
    }
};


// MUMPS BLR SOLVER

// BLR factorization variant
static constexpr int UFSC = (1 << 0); // default
static constexpr int UCFS = (1 << 1);

// compression of the contribution blocks (CB)
static constexpr int uncompressed_CB = (1 << 2); // default 
static constexpr int compressed_CB = (1 << 3);

//USAGE: flags = [UFSC | UCFS] | [uncompressed_CB | compressed_CB] ---- (flags can be omitted and default will be used)

// estimated compression rate of LU factors
//ICNTL(38) = ... 
// between 0 and 1000 (default: 600)
// ICNTL(38)/10 is a percentage representing the typical compression of the compressed factors factor matrices in BLR fronts

// estimated compression rate of contribution blocks
//ICNTL(39) = ...
// between 0 and 1000 (default: 500)
// ICNTL(39)/10 is a percentage representing the typical compression of the compressed CB CB in BLR fronts


template <isEigenSparseMatrix  MatrixType_>
class MumpsBLR : public MumpsBase< MatrixType_>
{
    using Scalar = typename MatrixType_::Scalar;

public:

    // CONSTRUCTOR
    MumpsBLR(): MumpsBLR(MPI_COMM_WORLD) {}
    explicit MumpsBLR(MPI_Comm comm): MumpsBase< MatrixType_>(comm) {
        // initialize MUMPS
        this->mumps_.par = 0;
        this->mumps_.sym = 0;
        this->mumps_execute(this->JOB_INIT);

        // activate the BLR feature
        this->ICNTL(35) = 1; // 1: automatic choice of the BLR memory management strategy
    }
    MumpsBLR(MPI_Comm comm, int flags) : MumpsBLR(comm) {
        eigen_assert(!(flags & UFSC && flags & UCFS) && "UFSC and UCFS cannot be set at the same time");
        eigen_assert(!(flags & uncompressed_CB && flags & compressed_CB) && "uncompressed_CB and compressed_CB cannot be set at the same time");
        this->ICNTL(36) = flags & UCFS; // set ICNTL(36) = 1 if UCFS is set, otherwise it remains 0 (default -> UFSC)
        this->ICNTL(37) = flags & compressed_CB; // set ICNTL(37) = 1 if compressed_CB is set, otherwise it remains 0 (default -> uncompressed_CB)
    }
    MumpsBLR(MPI_Comm comm, int flags, double dropping_parameter) : MumpsBLR(comm, flags) {
        eigen_assert(dropping_parameter >= 0.0 && "The dropping parameter must be non-negative");
        this->CNTL(7) = dropping_parameter;
    }

    // DESTRUCTOR
    ~MumpsBLR() override {
        // finalize MUMPS
        this->mumps_execute(this->JOB_END);
    }
};



// MUMPS RANK REVEALING SOLVER
template <isEigenSparseMatrix  MatrixType_>
class MumpsRankRevealing : public MumpsBase< MatrixType_> {

    using Scalar = typename MatrixType_::Scalar;

public:

    // CONSTRUCTOR
    MumpsRankRevealing(): MumpsRankRevealing(MPI_COMM_WORLD) {}
    explicit MumpsRankRevealing(MPI_Comm comm): MumpsBase< MatrixType_>(comm) {
        // initialize MUMPS
        this->mumps_.par = 0;
        this->mumps_.sym = 0;
        this->mumps_execute(this->JOB_INIT);

        // set ICNTL(56) = 1 to perform rank revealing factorization
        this->ICNTL(56) = 1;
    }

    // DESTRUCTOR
    ~MumpsRankRevealing() override {
        // finalize MUMPS
        this->mumps_execute(this->JOB_END);
    }

    // NULL SPACE SIZE METHODS
    int nullSpaceSize() {
        eigen_assert(this->matrix_computed_ && "The matrix must be computed with compute(Matrix) before calling nullSpaceSize()");
        return this->INFOG(28);
    }

    // RANK METHODS
    int rank() {
        eigen_assert(this->matrix_computed_ && "The matrix must be computed with compute(Matrix) before calling rank()");
        return this->n_ - this->INFOG(28);
    }

    // NULL SPACE BASIS METHODS
    MatrixXd nullSpaceBasis() {
        eigen_assert(this->matrix_computed_ && "The matrix must be computed with compute(Matrix) before calling nullSpaceBasis()");

        int null_space_size_ = nullSpaceSize();

        if(null_space_size_ == 0) {
            std::cout << "The matrix is full rank, the null space is empty" << std::endl;
            return MatrixXd::Zero(this->n_, 0);
        }

        // allocate memory for the null space basis
        std::vector<Scalar> buff;
        buff.resize(this->n_ * null_space_size_);

        if(this->rank_ == 0) {
            this->mumps_.rhs = buff.data();
            this->mumps_.nrhs = null_space_size_;
        }

        this->ICNTL(25) = 1; // 1: perform a null space basis computation step
        this->mumps_execute(3); // 3 : solve
        this->ICNTL(25) = 0; // reset ICNTL(25) to 0 -> a noral solve can be called

        MPI_Bcast(buff.data(), buff.size(), MPI_DOUBLE, 0, this->comm_);

        return Map<MatrixXd>(buff.data(), this->n_, null_space_size_); // --> matrix columns are the null space basis vectors
    }
};


// // CLASS FOR COMPUTING THE SHUR COMPLEMENT
// template <isEigenSparseMatrix  MatrixType_>
// class MumpsSchurComplement : public MumpsBase< MatrixType_> {

//     using Scalar = typename MatrixType_::Scalar;

//     using MumpsBase<MatrixType_>::mumps_;
//     using MumpsBase<MatrixType_>::JOB_INIT;
//     using MumpsBase<MatrixType_>::JOB_END;

// protected:
//     int schur_size_;
//     std::vector<Scalar> schur_buff_;
//     using MumpsBase<MatrixType_> :: compute; // redefining base class method as private to prevent the user from accessing this version of compute

// public:

//     // CONSTRUCTOR
//     MumpsSchurComplement() : MumpsSchurComplement(MPI_COMM_WORLD) {}
//     explicit MumpsSchurComplement(MPI_Comm comm): MumpsBase< MatrixType_>(comm) {
//         mumps_.par = 0;
//         mumps_.sym = 0;
//         mumps_execute(JOB_INIT); // initialize MUMPS

//         ICNTL(19) = 3; // 3: distributed by columns (changing parameters to cenralize it)
//         mumps_.nprow = mumps_.npcol = 1;
//         mumps_.mblock = mumps_.nblock = 100;
//     }

//     // DESTRUCTOR
//     ~MumpsSchurComplement() override {
//         mumps_execute(JOB_END); // finalize MUMPS
//     }

//     void compute (const MatrixType_ &matrix, int schur_size) {
        
//         eigen_assert(matrix.rows() == matrix.cols() && "The matrix must be square");
//         eigen_assert(schur_size < matrix.rows() && "The Schur complement size must be smaller than the matrix size");

//         schur_size_ = schur_size;

//         schur_buff.resize(schur_size_ * schur_size_)

//         mumps_.size_schur = schur_size_;
//         mumps_.listvar_schur = // ???????????????????????????

//         if (rank_ == (mumps_.par + 1)%2) { // I need to define these on the first working processor: PAR=1 -> rank 0, PAR=0 -> rank 1
//             mumps_.schur_lld = schur_size;
//             mumps_.schur = schur_buff.data();
//         }

//         compute(matrix)
//     }

//     // SCHUR COMPLEMENT COMPUTATION METHODS
//     MatrixXd complement() {
//         eigen_assert(matrix_computed_ && "The matrix must be computed with compute(Matrix, schur_size) before calling complement()");
//         MPI_Bcast(schur_buff.data(), schur_buff.size(), MPI_DOUBLE, (mumps_.par + 1)%2, comm_);
//         return Map<MatrixXd>(buff.data(), schur_size_, schur_size_);
//     }
// };

#endif // __MUMPS_H__
