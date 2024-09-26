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

// defining macros for mumps
#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

// macro s.t. indices match documentation
#define ICNTL(I) icntl[(I) - 1]
#define CNTL(I) cntl[(I) - 1]
#define INFO(I) info[(I) - 1]
#define INFOG(I) infog[(I) - 1]
#define RINFO(I) rinfo[(I) - 1]
#define RINFOG(I) rinfog[(I) - 1]

using namespace Eigen;
using namespace internal;

// namespace fdapde// fdapde
// {

// PardisoLU eigen

// MumpsBase -> gestire comunicazione via MPI + utilità comuni tipo set_ictl, code_error, code_warning

// MumpsLU : pubilc MumpsBase {
// solve()

// }

// MumpsBLR
// solve()
// Mumps {

    // MumpsBase ...; MumpsCommunicator

// }

    // concepts: get in a matrix of any type and convert it to an eigen sparse -> in MumpsBase put the convesrsion method

    // LIST OF CONCEPTS
    // 1. SparseMatrixType
    // 2. VectorType
    // 3. ScalarType
    // 4. IndexType

    // LIST OF CLASSES
    // 1. MumpsBase
    // 2. MumpsLU
    // 3. MumpsLDLT
    // 4. MumpsBLR
    // 5. MumpsRankRevealing
    // 6. MumpsDeterminant

    // CONCEPTS IMPLEMENTATION
    /*
    template <typename T>
    concept SparseMatrixType = requires(T t)
    {
        { t.rows() }
        ->std::same_as<int>;
        { t.cols() }
        ->std::same_as<int>;
        { t.nonZeros() }
        ->std::same_as<int>;
        { t.innerIndexPtr() }
        ->std::same_as<const int *>;
        { t.outerIndexPtr() }
        ->std::same_as<const int *>;
    };

    template <typename T>
    concept VectorType = requires(T t)
    {
        { t.size() }
        ->std::same_as<int>;
        { t.data() }
        ->std::same_as<double *>;
    };

    template <typename T>
    concept ScalarType = std::is_arithmetic<T>::value;
    */

    // FORWARD DECLARATIONS
    template <typename MatrixType_>
    class MumpsLU;
    template <typename MatrixType_>
    class MumpsLDLT;
    template <typename MatrixType_>
    class MumpsBLR;
    template <typename MatrixType_>
    class MumpsRankRevealing;
    template <typename MatrixType_>
    class MumpsDeterminant;

    // MUMPS BASE CLASS
    template <typename MatrixType_> //--> CRTP???? (with this as base class) ---> concept ????
    class MumpsBase
    {

        using Scalar = MatrixType_::Scalar;
        using StorageIndex = MatrixType_::StorageIndex;
        using VectorType = Matrix<Scalar, Dynamic, 1>;

    //template <typename Derived>
    //class MumpsBase : public SparseSolverBase<Derived> // ???????????????? -> check what i'm inheriting from SparseSolverBase, 
                                                        // do i implement solve or _solve_impl???? -> chech the code of the public methods, 
                                                        // PardisoLU defines only _solve_impl
        // protected: mumps, matrix information, mpi stuff
        // public: setIcntl, setCntl, getIcntl, getCntl, errorCheck, handleError, handleWarning, getRinfo, getRinfog, getIinfo, getInfog, warning/error codes
        // constructor, virtual destructor -> let derived classes initialize and terminate mumps

    public:

        explicit MumpsBase(MPI_Comm comm = MPI_COMM_WORLD) : comm_(comm)
        {
            // if MPI isn't initialized, initialize it
            MPI_Initialized(&mpi_initialized_);  // --> CONSDER ADDING mpi_err = AT THE START OF ALL THE MPI CALLS TO CHECK FOR ERRORS
            if (!mpi_initialized_)
            {
                MPI_Init(NULL, NULL);
            }

            MPI_Comm_rank(comm_, &rank_);

            // if (comm_ == MPI_COMM_WORLD)  // IS THIS ALLOWED? (look online for a better way of doing it if there is one)
            //     mumps_.comm_fortran = USE_COMM_WORLD;
            // else
                mumps_.comm_fortran = (MUMPS_INT) MPI_Comm_c2f(comm_);
        }

        virtual ~MumpsBase()
        {
            // if MPI was initialized by this class, finalize it
            if (!mpi_initialized_)
            {
                MPI_Finalize();
            }
        }

        // setters and getters for ICNTL and CNTL
        void set_ICNTL(int index, int value)
        {
            mumps_.ICNTL(index) = value;
        }

        void set_CNTL(int index, double value)
        {
            mumps_.CNTL(index) = value;
        }

        int get_ICNTL(int index) const
        {
            return mumps_.ICNTL(index);
        }

        double get_CNTL(int index) const
        {
            return mumps_.CNTL(index);
        }

        int get_INFO(int index) const
        {
            return mumps_.INFO(index);
        }

        int get_INFOG(int index) const
        {
            return mumps_.INFOG(index);
        }

        double get_RINFO(int index) const
        {
            return mumps_.RINFO(index);
        }      

        double get_RINFOG(int index) const
        {
            return mumps_.RINFOG(index);
        }

        // defining the matrix
        void define_matrix(const MatrixType &matrix) // -> CONSIDER MAKING THIS TAKE A CONCEPT AND AUTOMATICALLY CONVERT IT TO AN EIGEN SPARSE MATRIX
        {   
            // ASSERTS -> square matrix mainly fdapde_assert(matrix.rows() == matrix.cols() /*&& ... FDAPDE_COMMA*/);
            fdapde_assert(matrix.rows() == matrix.cols());

            // scaling the indexes for MUMPS
            row_indices_ = mumpps_index_scaling(matrix.innerIndexPtr(), matrix.innerIndexPtr() + matrix.nonZeros());
            col_indices_ = mumpps_index_scaling(matrix.outerIndexPtr(), matrix.outerIndexPtr() + matrix.nonZeros());

            // defining the problem on the host
            if (rank_ == 0)
            {
                mumps_.n = matrix.rows();
                mumps_.nnz = matrix.nonZeros();
                mumps_.irn = row_indices_.data(); // DO I NEED A CONST CAST TO A C POINTER? (like i need to do for rhs)
                mumps_.jcn = col_indices_.data();
            }

            matrix_defined_ = true;
        }

        // error / warning handling & error / warnings CODE GETTERS???

        void errorCheck(bool warning_check) const;
        const std::string handleError(int error_1, double error_2) const;
        const std::string handleWarning(int warning_1, double warning_2) const;


    protected:

        // mumps struct
        DMUMPS_STRUC_C mumps_;

        // matrix info
        std::vector<int> col_indices_;
        std::vector<int> row_indices_;

        // mpi
        MPI_Comm comm_;
        int mpi_initialized_;
        int rank_;

        // flag
        bool matrix_defined_ = false;  

        // Index scaling for MUMPS (MUMPS uses 1-based indexing)
        std::vector<int> mumps_index_scaling(const StorageIndex* begin, const StorageIndex* end) // -> assuming EigenSparseMatrix
        {
            std::vector<int> indices;
            indices.reserve(end - begin);
            std::copy(begin, end, std::back_inserter(indices));
            std::for_each(indices.begin(), indices.end(), [](int &idx)
                          { idx += 1; });
            return indices;
        }

        void mumps_execute(bool error_check = true, bool warning_check = false)
        {
            dmumps_c(&mumps_);

            if (rank == 0 && error_check)
            {
                errorCheck(warning_check);
            }
        }

        // ?????
        // oid mumps_execute(int job, bool error_check = true)
        // {
        //     mumps_.job = job;
        //     dmumps_c(&mumps_);

        //     if (rank == 0 && error_check)
        //     {
        //         errorCheck();
        //     }
        // }
    };



    template <typename MatrixType_>
    class MumpsLU : protected MumpsBase< MatrixType_>
    // template <typename MatrixType_>
    // class MumpsLU : protected MumpsBase< MumpsLU<typename MatrixType_>>
    {
        using Scalar = MatrixType_::Scalar;
        using StorageIndex = MatrixType_::StorageIndex;
        using VectorType = Matrix<Scalar, Dynamic, 1>;
        // Solver with LU factorization
        // protected: i think nothing
        // public: solve() -> check on EigenSparseSolver how to implement it

    public:

        explicit MumpsLU(MPI_Comm comm = MPI_COMM_WORLD): MumpsBase< MatrixType_>(comm)
        //MumpsLU(MPI_Comm comm): MumpsBase< MumpsLU<typename MatrixType_>>(comm)
        {
            // initialize MUMPS
            mumps_.par = 1;
            mumps_.sym = 0;
            mumps_.job = JOB_INIT;
            mumps_execute();

            // set default values for ICNTL
            mumps_.ICNTL(1) = -1; // output stream for error messages
            mumps_.ICNTL(2) = -1; // output stream for diagnostic printing
            mumps_.ICNTL(3) = -1; // output stream for global information
            mumps_.ICNTL(4) = 0;  // level of printing
        }
        
        ~MumpsLU() override
        {
            // finalize MUMPS
            mumps_.job = JOB_END;
            mumps_execute();
        }

        // solve method
        VectorType solve(VectorType rhs) //Mumps saves the solution in the rhs vector -> passing by copy
        {
            // check if the matrix has been defined -> throw an error if not
            if (!matrix_defined_)
            {
                if (rank_ == 0)
                {
                    std::cerr << "You need to call define_matrix() before calling solve(rhs)"
                                << " or alternatively call solve(matrix, rhs)" << std::endl;
                }
                exit(1);
            }

            // defining the problem on the host
            if (rank_ == 0)
            {
                mumps_.n = rhs.size();
                mumps_.rhs = const_cast<Scalar *>(rhs.data());
            }

            mumps_.job = 6; // 6: analyze + factorize + solve
            mumps_execute();
            
            // if (rank_ == 0)
            // {
            //     errorCheck();
            // }

            MPI_Bcast(rhs.data(), rhs.size(), MPI_DOUBLE, 0, comm_); // -> consider letting mumps base handle all mpi commands

            return rhs;
        }

        VectorType solve(const MatrixType &matrix, VectorType rhs) //Mumps saves the solution in the rhs vector -> passing by copy
        {
            define_matrix(matrix);
            return solve(rhs); // -> does this create 2 copies???
        }
    };



    template <typename MatrixType_>
    class MumpsLDLT : protected MumpsBase< MatrixType_>
    {
        using Scalar = MatrixType_::Scalar;
        using StorageIndex = MatrixType_::StorageIndex;
        using VectorType = Matrix<Scalar, Dynamic, 1>;

        // Solver with LDLT factorization -> symmetric and positive definite matrix only (only pass upper or lower triangular matrix)
        // protected: i think nothing
        // public: solve() -> check on EigenSparseSolver how to implement it

    public:

        explicit MumpsLDLT(MPI_Comm comm = MPI_COMM_WORLD): MumpsBase< MatrixType_>(comm)
        {
            // initialize MUMPS
            mumps_.par = 1;
            mumps_.sym = 1; // -> symmetric and positive definite
            mumps_.job = JOB_INIT;
            mumps_execute();

            // set default values for ICNTL
            mumps_.ICNTL(1) = -1; // output stream for error messages
            mumps_.ICNTL(2) = -1; // output stream for diagnostic printing
            mumps_.ICNTL(3) = -1; // output stream for global information
            mumps_.ICNTL(4) = 0;  // level of printing
        }

        ~MumpsLDLT() override
        {
            // finalize MUMPS
            mumps_.job = JOB_END;
            mumps_execute();
        }

        VectorType solve(VectorType rhs)
        {
            // check if the matrix has been defined -> throw an error if not
            if (!matrix_defined_)
            {
                if (rank_ == 0)
                {
                    std::cerr << "You need to call define_matrix() before calling solve(rhs)"
                                << " or alternatively call solve(matrix, rhs)" << std::endl;
                }
                exit(1);
            }

            // defining the problem on the host
            if (rank_ == 0)
            {
                mumps_.n = rhs.size();
                mumps_.rhs = const_cast<Scalar *>(rhs.data());
            }

            mumps_.job = 6; // 6: analyze + factorize + solve
            mumps_execute();

            // if (rank_ == 0)
            // {
            //     errorCheck();
            // }

            MPI_Bcast(rhs.data(), rhs.size(), MPI_DOUBLE, 0, comm_);

            return rhs;
        }

        VectorType solve(const MatrixType &matrix, VectorType rhs)
        {
            define_matrix(matrix);
            return solve(rhs);
        }
    };



    template <typename MatrixType_>
    class MumpsBLR : protected MumpsBase< MatrixType_>
    {
        using Scalar = MatrixType_::Scalar;
        using StorageIndex = MatrixType_::StorageIndex;
        using VectorType = Matrix<Scalar, Dynamic, 1>;

        // Solver with BLR factorization
        // protected: i think nothing
        // public: solve() -> check on EigenSparseSolver how to implement it

    public:

        explicit MumpsBLR(MPI_Comm comm = MPI_COMM_WORLD): MumpsBase< MatrixType_>(comm) // -> ADD THE ICNTL FOR BLR
        {
            // initialize MUMPS
            mumps_.par = 1;
            mumps_.sym = 0;
            mumps_.job = JOB_INIT;
            mumps_execute();

            // set default values for ICNTL  --> CONSIDER CHANGING THESE FOR THE BLR STATISTICS OUTPUT
            mumps_.ICNTL(1) = -1; // output stream for error messages
            mumps_.ICNTL(2) = -1; // output stream for diagnostic printing
            mumps_.ICNTL(3) = -1; // output stream for global information
            mumps_.ICNTL(4) = 0;  // level of printing

            // activate the BLR feature
            mumps_.ICNTL(35) = 1; // 1: automatic choice of the BLR memory management strategy

            //dropping parameter controlling the accuracy of the Block Low-Rank approximations
            //mumps_.CNTL(7) = ...
            // 0.0: full precision approximation (default)
            // >0.0: the dropping parameter is CNTL(7)

            // choice of the BLR factorization variant
            //mumps_.ICNTL(36) = ... 
            // 0: Standard UFSC variant with low-rank updates accumulation (LUA) (default)
            // 1: UCFS variant with low-rank updates accumulation (LUA) 
            //    (performing the compression earlier in order to further reduce the number of operations)

            // compression of the contribution blocks (CB)
            //mumps_.ICNTL(37) = ... 
            // 0: blocks are not compressed (default)
            // 1: blocks are compressed

            // estimated compression rate of LU factors
            //mumps_.ICNTL(38) = ... 
            // between 0 and 1000 (default: 600)
            // ICNTL(38)/10 is a percentage representing the typical compression of the compressed factors factor matrices in BLR fronts

            // estimated compression rate of contribution blocks
            //mumps_.ICNTL(39) = ...
            // between 0 and 1000 (default: 500)
            // ICNTL(39)/10 is a percentage representing the typical compression of the compressed CB CB in BLR fronts

        ~MumpsBLR() override
        {
            // finalize MUMPS
            mumps_.job = JOB_END;
            mumps_execute();
        }

        VectorType solve(VectorType rhs)
        {
            // check if the matrix has been defined -> throw an error if not
            if (!matrix_defined_)
            {
                if (rank_ == 0)
                {
                    std::cerr << "You need to call define_matrix() before calling solve(rhs)"
                                << " or alternatively call solve(matrix, rhs)" << std::endl;
                }
                exit(1);
            }

            // defining the problem on the host
            if (rank_ == 0)
            {
                mumps_.n = rhs.size();
                mumps_.rhs = const_cast<Scalar *>(rhs.data());
            }

            mumps_.job = 6; // 6: analyze + factorize + solve
            mumps_execute();

            // if (rank_ == 0)
            // {
            //     errorCheck();
            // }

            MPI_Bcast(rhs.data(), rhs.size(), MPI_DOUBLE, 0, comm_);

            return rhs;
        }

        VectorType solve(const MatrixType &matrix, VectorType rhs)
        {
            define_matrix(matrix);
            return solve(rhs);
        }
    };



    template <typename MatrixType_>
    class MumpsRankRevealing : protected MumpsBase< MatrixType_>
    {
        using Scalar = MatrixType_::Scalar;
        using StorageIndex = MatrixType_::StorageIndex;
        using VectorType = Matrix<Scalar, Dynamic, 1>;

        // Solver with rank revealing factorization
        // protected: i think nothing
        // public: solve() -> check on EigenSparseSolver how to implement it

    protected:
        int null_space_size_ = -1;
        //VectorType null_space_basis_;
        //bool null_space_basis_computed_ = false;

    public:

        explicit MumpsRankRevealing(MPI_Comm comm = MPI_COMM_WORLD): MumpsBase< MatrixType_>(comm)
        {
            // initialize MUMPS
            mumps_.par = 1;
            mumps_.sym = 0;
            mumps_.job = JOB_INIT;
            mumps_execute();

            // set default values for ICNTL
            mumps_.ICNTL(1) = -1; // output stream for error messages
            mumps_.ICNTL(2) = -1; // output stream for diagnostic printing
            mumps_.ICNTL(3) = -1; // output stream for global information
            mumps_.ICNTL(4) = 0;  // level of printing

            // set ICNTL(56) = 1 to perform rank revealing factorization
            mumps_.ICNTL(56) = 1;
        }

        ~MumpsRankRevealing() override
        {
            // finalize MUMPS
            mumps_.job = JOB_END;
            mumps_execute();
        }

        VectorType solve(VectorType rhs)
        {
            // check if the matrix has been defined -> throw an error if not
            if (!matrix_defined_)
            {
                if (rank_ == 0)
                {
                    std::cerr << "You need to call define_matrix() before calling solve(rhs)"
                                << " or alternatively call solve(matrix, rhs)" << std::endl;
                }
                exit(1);
            }

            // defining the problem on the host
            if (rank_ == 0)
            {
                mumps_.n = rhs.size();
                mumps_.rhs = const_cast<Scalar *>(rhs.data());
            }

            mumps_.ICNTL(25) = 0; // 0: perform a regular solution step 
            // (if the matrix is found singular during factorization then one of the possible solutions is returned)

            mumps_.job = 6; // 6: analyze + factorize + solve
            mumps_execute();

            // if (rank_ == 0)
            // {
            //     errorCheck();
            // }

            MPI_Bcast(rhs.data(), rhs.size(), MPI_DOUBLE, 0, comm_);

            return rhs;
        }

        VectorType solve(const MatrixType &matrix, VectorType rhs)
        {
            define_matrix(matrix);
            return solve(rhs);
        }

        int nullSpaceSize(bool recompute = false)
        {
            if(!matrix_defined_)
            {
                if (rank_ == 0)
                {
                    std::cerr << "You need to call define_matrix() before calling nullSpaceSize()" << std::endl;
                }
                exit(1);
            }

            if (null_space_size_ >= 0 && !recompute)
            {
                return null_space_size_;
            }
            
            mumps_.job = 4;
            mumps_execute();

            null_space_size_ = mumps_.INFOG(28);

            return null_space_size_;
        }

        int nullSpaceSize(const MatrixType &matrix)
        {
            define_matrix(matrix);
            return nullSpaceSize(recompute = true);
        }

        VectorType nullSpaceBasis(bool recompute = false)
        {
            if(!matrix_defined_)
            {
                if (rank_ == 0)
                {
                    std::cerr << "You need to call define_matrix() before calling nullSpaceBasis()" << std::endl;
                }
                exit(1);
            }

            if(null_space_size_ == -1)
            {
                nullSpaceSize();
            }

            if(null_space_size_ == 0)
            {
                std::cout << "The matrix is full rank, the null space is empty" << std::endl;
                return VectorType::Zero(mumps_.n);
            }

            //if(null_space_basis_computed_ && !recompute)
            //{
            //    return null_space_basis_;
            //}

            mumps_.ICNTL(25) = 1; // 1: perform a null space basis computation step
            
            // allocate memory for the null space basis
            VectorType null_space_basis(null_space_size_ * mumps_.n);

            if(rank_ == 0)
            {
                mumps_.rhs = null_space_basis.data();
                mumps_.nrhs = null_space_size_;
            }

            mumps_.job = 6;
            mumps_execute();

            // covert null_space_basis to a matrix <------------------------------
            //null_space_basis = MatrixType::Map(null_space_basis.data(), mumps_.n, null_space_size_);

            //null_space_basis_= null_space_basis;
            //null_space_basis_computed_ = true;
            return null_space_basis;
        }

        VectorType nullSpaceBasis(const MatrixType &matrix)
        {
            define_matrix(matrix);
            return nullSpaceBasis(recompute = true);
        }

        int rank()
        {
            if(!matrix_defined_)
            {
                if (rank_ == 0)
                {
                    std::cerr << "You need to call define_matrix() before calling rank()" << std::endl;
                }
                exit(1);
            }
            return mumps_.n - nullSpaceSize();
        }

        int rank(const MatrixType &matrix)
        {
            define_matrix(matrix);
            return rank();
        }
    };



    template <typename MatrixType_>
    class MumpsDeterminant : protected MumpsBase< MatrixType_>
    {
        using Scalar = MatrixType_::Scalar;
        using StorageIndex = MatrixType_::StorageIndex;
        using VectorType = Matrix<Scalar, Dynamic, 1>;

        // determinant computation
        // protected: i think nothing
        // public: solve() -> check on EigenSparseSolver how to implement it

    protected:
        // Scalar determinant_;
        // bool determinant_computed_ = false;

    public:

        explicit MumpsDeterminant(MPI_Comm comm = MPI_COMM_WORLD): MumpsBase< MatrixType_>(comm)
        {
            // initialize MUMPS
            mumps_.par = 1;
            mumps_.sym = 0;
            mumps_.job = JOB_INIT;
            mumps_execute();

            // set default values for ICNTL
            mumps_.ICNTL(1) = -1; // output stream for error messages
            mumps_.ICNTL(2) = -1; // output stream for diagnostic printing
            mumps_.ICNTL(3) = -1; // output stream for global information
            mumps_.ICNTL(4) = 0;  // level of printing

            // set ICNTL(33) = 1 to compute the determinant
            mumps_.ICNTL(33) = 1;
        }

        ~MumpsDeterminant() override
        {
            // finalize MUMPS
            mumps_.job = JOB_END;
            mumps_execute();
        }

        Scalar solve(const MatrixType &matrix)
        {
            define_matrix(matrix);
            return solve(/*recompute = true*/); 
        }

        Scalar solve(/*bool recompute = false*/)
        {
            // check if the matrix has been defined -> throw an error if not
            if (!matrix_defined_)
            {
                if (rank_ == 0)
                {
                    std::cerr << "You need to call define_matrix() before calling solve()" << std::endl;
                }
                exit(1);
            }

            // if (determinant_computed_ && !recompute)
            // {
            //     return determinant_;
            // }

            mumps_.job = 4; // 4: analyze + factorize
            mumps_execute();

            // if (rank_ == 0)
            // {
            //     errorCheck();
            // }

            // determinant_ = mumps_.RINFOG(12) * pow(2, mumps_.INFOG(34));
            // determinant_computed_ = true;

            return determinant_;
        }

    };


    template <typename MatrixType_>
    class MumpsShurComplement : protected MumpsBase< MatrixType_>
    {
        // Shur complement computation
        };


// template <typename IteratorType>
// vector<...> mumps_align_indexes_(IteratorType begin, IteratorType end) 


    // ERROR HANDLING
    template <typename MatrixType_>
    void MumpsBase<MatrixType_> :: errorCheck(bool warning_check) const
        {
            if (mumps_.INFOG(1) < 0)
            {
                std::cerr << "ERROR in MUMPS:\n"
                          << handleError(mumps_.INFOG(1), mumps_.INFOG(2)) << std::endl;
                exit(1);
            }
            if (warning_check && mumps_.INFOG(1) > 0)
            {
                std::cerr << "WARNING in MUMPS:\n"
                          << handleWarning(mumps_.INFOG(1), mumps_.INFOG(2)) << std::endl;
            }
        }

    template <typename MatrixType_>
    const std::string MumpsBase<MatrixType_> :: handleError(int error_1, double error_2) const
    {
        if (error_1 == 0)
        {
            return "No error";
        }
        switch (error_1)
        {
        case -1:
            return "An error occurred on processor " + std::to_string(error_2);
        case -2:
            return "Number of nonzeros is out of range. Current value: " + std::to_string(error_2);
        case -3:
            return "Mumps was called with an invalid JOB parameter, " +
                    "operations were out of order (analyzePattern -> factorize -> solve) or JOB had different values on different processors";
        case -4:
            return "Error in user-provided permutation array perm_in at position " + std::to_string(error_2);
        case -5:
            return "Problem of real workspace allocation of size " + std::to_string(error_2) + " during analysis";
        case -6:    
            return "Matrix is singular in structure. Structural rank: " + std::to_string(error_2);
        case -7:
            return "Problem of integer workspace allocation of size " + std::to_string(error_2) + " during analysis";
        case -8:
            return "Main internal integer workarray IS too small for factorization";
        case -9:
            return "Main internal real/complex workarray S too small. " +
                            "The number of entries that are missing in S at the moment when the error is raised is: " +
                            (error_2 >= 0)
                        ? std::to_string(error_2)
                        : std::to_string(abs(error_2) * pow(10, 6));
        case -10:
            return "Numerically singular matrix or zero pivot encountered. Number of eliminated pivots: " + std::to_string(error_2);
        case -11:
            return "Internal real/complex workarray S or LWK_USER too small for solution. " +
                            "The number of entries that are missing in S/LMK_USER at the moment when the error is raised is: " +
                            (error_2 >= 0)
                        ? std::to_string(error_2)
                        : "unknown";
        case -12:
            return "Internal real/complex workarray S too small for iterative refinement";
        case -13:
            return "Problem of workspace allocation of size " +
                            (error_2 >= 0)
                        ? std::to_string(error_2)
                        : std::to_string(abs(error_2) * pow(10, 6)) +
                                " during the factorization or solve steps";
        case -14:
            return "Internal integer workarray IS too small for solution";
        case -15:
            return "Integer workarray IS too small for iterative refinement and/or error analysis";
        case -16:
            return "N is out of range. Current N: " + std::to_string(error_2);
        case -17:
            return "The internal send buffer that was allocated dynamically by MUMPS on the processor is too small";
        case -18:
            return "The blocking size for multiple RHS (ICNTL(27)) is too large and may lead to an integer overflow. " +
                    "Estimate of the maximum value of ICNTL(27) that should be used: " + std::to_string(error_2);
        case -19:
            return "The maximum allowed size of working memory ICNTL(23) is too small to run the factorization phase and should be increased. " +
                            "The number of entries that are missing at the moment when the error is raised is: " +
                            (error_2 >= 0)
                        ? std::to_string(error_2)
                        : std::to_string(abs(error_2) * pow(10, 6));
        case -20:
            return "The internal reception buffer that was allocated dynamically by MUMPS is too small. " +
                    "The minimum size of the reception buffer required is: " + std::to_string(error_2);
        case -21:
            return "Value of PAR=0 is not allowed because only one processor is available";
        case -22:
            std::string add_str = "";
            switch (error_2)
            {
            case 1:
                add_str = "IRN or ELTPTR";
            case 2:
                add_str = "JCN or ELTVAR";
            case 3:
                add_str = "PERM IN";
            case 4:
                add_str = "A or A_ELT";
            case 5:
                add_str = "ROWSCA";
            case 6:
                add_str = "COLSCA";
            case 7:
                add_str = "RHS";
            case 8:
                add_str = "LISTVAR_SCHUR";
            case 9:
                add_str = "SCHUR";
            case 10:
                add_str = "RHS_SPARSE";
            case 11:
                add_str = "IRHS_SPARSE";
            case 12:
                add_str = "IRHS_PTR";
            case 13:
                add_str = "ISOL_loc";
            case 14:
                add_str = "SOL_loc";
            case 15:
                add_str = "REDRHS";
            case 16:
                add_str = "IRN_loc, JCN_loc or A_loc";
            case 17:
                add_str = "IRHS_loc";
            case 18:
                add_str = "RHS_loc";
            defualt:
                add_str = "'unknown'";
            };
            return "The pointer array " + add_str +
                    " provided by the user is either:\n" +
                    "   - not associated, or\n" +
                    "   - has insufficient size, or\n" +
                    "   - is associated and should not be associated";
        case -23:
            return "MPI was not initialized by the user prior to a call to MUMPS with JOB= -1";
        case -24:
            return "NELT is out of range. Current NELT: " + std::to_string(error_2);
        case -25:
            return "A problem has occurred in the initialization of the BLACS";
        case -26:
            return "LRHS is out of range. Current LRHS: " + std::to_string(error_2);
        case -27:
            return "NZ_RHS and IRHS_PTR(NRHS+1) do not match. Current IRHS_PTR(NRHS+1): " + std::to_string(error_2);
        case -28:
            return "IRHS_PTR(1) is not equal to 1. Current IRHS_PTR(1): " + std::to_string(error_2);
        case -29:
            return "LSOL_loc is too small. Current LSOL_loc: " + std::to_string(error_2);
        case -30:
            return "SCHUR_LLD is out of range. Current SCHUR_LLD: " + std::to_string(error_2);
        case -31:
            return "A 2D block cyclic symmetric (SYM=1 or 2) Schur complement is required with the option ICNTL(19)=3, " +
                    "but the user has provided a process grid that does not satisfy the constraint MBLOCK=NBLOCK. " +
                    "Current MBLOCK-NBLOCK: " + std::to_string(error_2);
        case -32:
            return "Incompatible values of NRHS and ICNTL(25). " +
                    "Either ICNTL(25) was set to -1 and NRHS is different from INFOG(28); " +
                    "or ICNTL(25) was set to i, 1 ≤ i ≤ INFOG(28) and NRHS is different from 1. " +
                    "Current value of NRHS: " + std::to_string(error_2);
        case -33:
            return "ICNTL(26) was asked for during solve phase (or during the factorization - see ICNTL(32)) " +
                    "but the Schur complement was not asked for at the analysis phase (ICNTL(19)). " +
                    "Current value of ICNTL(26): " + std::to_string(error_2);
        case -34:
            return "LREDRHS is out of range. Current value of LREDRHS: " + std::to_string(error_2);
        case -35:
            return "The expansion phase of the Schur feature is called (ICNTL(26)=2) but reduction phase (ICNTL(26)=1) was not called before. " +
                    "Current value of ICNTL(26): " + std::to_string(error_2);
        case -36:
            return "Incompatible values of ICNTL(25) and INFOG(28). Current value of ICNTL(25): " + std::to_string(error_2);
        case -37:
            return "Value of ICNTL(25) incompatible with ICNTL(" + std::to_string(error_2) + ")";
        case -38:
            return "Parallel analysis was set (i.e., ICNTL(28)=2) but PT-SCOTCH or ParMetis were not provided";
        case -39:
            return "Incompatible values for ICNTL(28) and ICNTL(5) and/or ICNTL(19) and/or ICNTL(6)";
        case -40:
        return "The matrix was indicated to be positive definite (SYM=1) by the user but " +
                "a negative or null pivot was encountered during the processing of the root by ScaLAPACK. SYM=2 should be used." case -41:
        return "Incompatible value of LWK USER between factorization and solution phases" case -42:
            return "ICNTL(32) was set to 1 (forward during factorization), " +
                    "but the value of NRHS on the host processor is incorrect: " +
                    "either the value of NRHS provided at analysis is negative or zero, " +
                    "or the value provided at factorization or solve is different from the value provided at analysis. " +
                    "The value of NRHS provided at analysis is: " + std::to_string(error_2);
        case -43:
            return "Incompatible values of ICNTL(32) and ICNTL(" + std::to_string(error_2) + ")";
        case -44:
            return "The solve phase (JOB= 3) cannot be performed because the factors or part of the factors are not available. " +
                    "Current value of ICNTL(31): " + std::to_string(error_2);
        case -45:
            return "NRHS <= 0. Current value of NRHS: " + std::to_string(error_2);
        case -46:
            return "NZ_RHS <= 0. This is currently not allowed with ICNTL(26)=1 and in case entries of A^-1 are equested (ICNTL(30)=1). " +
                    "Current value of NZ_RHS: " + std::to_string(error_2);
        case -47:
            return "Entries of A^-1 were requested during the solve phase (JOB= 3, ICNTL(30)=1) but the constraint NRHS=N is not respected. " +
                    "Current value of NRHS: " + std::to_string(error_2);
        case -48:
            return "A^-1 Incompatible values of ICNTL(30) and ICNTL(" + std::to_string(error_2) + ")";
        case -49:
            return "SIZE_SCHUR has an incorrect value: SIZE_SCHUR < 0 or SIZE SCHUR >=N, " +
                    "or SIZE_SCHUR was modified on the host since the analysis phase. " +
                    "Current value of SIZE_SCHUR: " + std::to_string(error_2);
        case -50:
            return "An error occurred while computing the fill-reducing ordering during the analysis phase";
        case -51:
            return "An external ordering (Metis/ParMetis, SCOTCH/PT-SCOTCH, PORD), with 32-bit default integers, " +
                            "is invoked to processing a graph of size larger than 2^31 - 1. " +
                            "Size required to store the graph as a number of integer values: " +
                            (error_2 >= 0)
                        ? std::to_string(error_2)
                        : std::to_string(abs(error_2) * pow(10, 6));
        case -52:
            std::string add_str = "";
            switch (error_2)
            {
            case 1:
                add_str = "Metis/ParMetis";
            case 2:
                add_str = "SCOTCH/PT-SCOTCH";
            case 3:
                add_str = "PORD";
            default:
                add_str = "'unknown'";
            };
            return "When default Fortran integers are 64 bit (e.g. Fortran compiler flag -i8 -fdefault-integer-8 or something equivalent depending on your compiler) " +
                    "then external ordering libraries should also have 64-bit default integers. " +
                    "Invoked library that reaide the error: " + add_str;
        case -53:
            return "Internal error that could be due to inconsistent input data between two consecutive calls";
        case -54:
        return "The analysis phase (JOB= 1) was called with ICNTL(35)=0 but the factorization phase was called with ICNTL(35)=1, 2 or 3" case -55:
            return "During a call to MUMPS including the solve phase with distributed right-hand side, " +
                    "either LRHS_loc was detected to be smaller than Nloc_RHS (LRHS_loc = " + std::to_string(error_2) +
                    "), or Nloc_RHS was not equal to zero on the non working host (PAR=0) (Nloc_RHS = " + std::to_string(-error_2) + ")";
        case -56:
            return "During a call to MUMPS including the solve phase with distributed right-hand side and distributed solution, " +
                    "RHS_loc and SOL_loc point to the same workarray but LRHS_loc < LSOL_loc. " +
                    "Current LRHS_loc: " + std::to_string(error_2);
        case -57:
            std::string add_str = "";
            switch (error_2)
            {
            case 1:
                add_str = "NBLK is incorrect (or not compatible with BLKPTR size), or -ICNTL(15) is not compatible with N";
            case 2:
                add_str = "BLKPTR is not provided or its content is incorrect";
            case 3:
                add_str = "BLKVAR if provided should be of size N";
            default:
                add_str = "'unknown'";
            };
            return "During a call to MUMPS analysis phase with a block format (ICNTL(15) != 0), " +
                    "the following error in the interface provided by the user was detected:\n" + add_str;
        case -58:
            std::string add_str = "";
            if (error_2 == 0)
            {
                add_str = "ICNTL(48) was equal to 1 at analysis, but compilation is without OpenMP enabled";
            }
            else if (error_2 > 0)
            {
                add_str = "ICNTL(48) is active but the number of threads available for the current phase (factorization or solve) " +
                            " is different from " + std::to_string(error_2) + ", the number of threads available at analysis.";
            }
            else if (error_2 < 0)
            {
                add_str = "ICNTL(48) is active but the number of threads effectively created during the main parallel region " +
                            "of the factorization that exploits multithreaded tree parallelism is different from the one obtained with omp_get_max_threads(), " +
                            "and used during analysis to prepare the work. The number of threads effectively obtained in the parallel region, " +
                            "retrieved with omp_get_num_threads() is: " + std::to_string(-error_2 - 100);
            }
            else
            {
                add_str = "Unknown error";
            }
            return "During a call to MUMPS with ICNTL(48)=1, the followinf error occurred:\n" + add_str;
        case -69:
            return "The size of the default Fortran INTEGER datatype does not match the size of MUMPS_INT. " +
                    "The size of MUMPS_INT is " + std::to_string(error_2);
        case -70:
            return "During a call to MUMPS with JOB= 7, the file specified to save the current instance, " +
                    "as derived from SAVE_DIR and/or SAVE_PREFIX, already exists. Before saving an instance into this file, " +
                    "it should be first suppressed (see JOB= -3). Otherwise, a different file should be specified by changing " +
                    "the values of SAVE_DIR and/or SAVE_PREFIX.";
        case -71:
            return "An error has occurred during the creation of one of the files needed to save MUMPS data (JOB= 7)";
        case -72:
            return "Error while saving data (JOB= 7); a write operation did not succeed (e.g., disk full, I/O error, ...). " +
                            "The size that should have been written during that operation is: " +
                            (error_2 >= 0)
                        ? std::to_string(error_2)
                        : std::to_string(abs(error_2) * pow(10, 6));
        case -73:
            std::string add_str = "";
            switch (error_2)
            {
            case 1:
                add_str = "'fortran version (after/before 2003)'";
            case 2:
                add_str = "'integer size(32/64 bit)'";
            case 3:
                add_str = "'saved instance not compatible over MPI processes'";
            case 4:
                add_str = "'number of MPI processes'";
            case 5:
                add_str = "'arithmetic'";
            case 6:
                add_str = "SYM";
            case 7:
                add_str = "PAR";
            default:
                add_str = "'unknown'";
            };
            return "During a call to MUMPS with JOB= 8, the parameter " + add_str +
                    " of the current instance is not compatible with the corresponding one in the saved instance.";
        case -74:
            return "The file resulting from the setting of SAVE_DIR and SAVE_PREFIX could not be opened for restoring data (JOB= 8). " +
                    "The rank of the process (in the communicator COMM) on which the error was detected is: " + std::to_string(error_2);
        case -75:
            return "Error while restoring data (JOB= 8); a read operation did not succeed (e.g., end of file reached, I/O error, ...). " +
                            "The size still to be read is: " +
                            (error_2 >= 0)
                        ? std::to_string(error_2)
                        : std::to_string(abs(error_2) * pow(10, 6));
        case -76:
            return "Error while deleting the files (JOB= -3); some files to be erased were not found or could not be suppressed. " +
                    "The rank of the process (in the communicator COMM) on which the error was detected is: " + std::to_string(error_2);
        case -77:
            std::string add_str = "";
            if (error_2 == 0)
            {
                add_str = "Neither SAVE_DIR nor the environment variable MUMPS_SAVE_DIR are defined";
            }
            else
            {
                add_str = "The environment variable MUMPS_SAVE_DIR is defined but " +
                            "its length is larger than the maximum authorized length, which is: " + std::to_string(error_2);
            }
            else if (error_2 < 0)
            {
                add_str = "The environment variable MUMPS_SAVE_PREFIX is defined but " +
                            "its length is larger that the maximum authorized length, which is: " + std::to_string(-error_2);
            }
            else
            {
                add_str = "'unknown'";
            }
            return "Problem with SAVE_DIR and/or SAVE_PREFIX:\n" + add_str;
        case -78:
            return "Problem of workspace allocation during the restore step. The size still to be allocated is: " + std::to_string(error_2);
        case -79:
            std::string add_str = "";
            switch (error_2)
            {
            case 1:
                add_str = "the problem occurs in the analysis phase, when attempting to find a free Fortran unit for the WRITE_PROBLEM feature";
            case 2:
                add_str = "the problem occurs during a call to MUMPS with JOB= 7, 8 or -3 (save-restore feature)";
            default:
                add_str = "'unknown'";
            };
            return "MUMPS could not find a Fortran file unit to perform I/O's: " + add_str;
        case -88:
            return "An error occurred during SCOTCH ordering. The error number returned by SCOTCH is: " + std::to_string(error_2);
        case -89:
            return "An error occurred during SCOTCH kway-partitioning in SCOTCHFGRAPHPART. " +
                    "The error code returned by SCOTCH is: " + std::to_string(error_2);
        case -90:
            return "Error in out-of-core management";
        case -800:
            return "Temporary error associated to the current MUMPS release, subject to change or disappearance in the future" +
                            (error_2 == 5)
                        ? ". This error is due to the fact that the elemental matrix format (ICNTL(5)=1) is currently incompatible " +
                                "with a BLR factorization (ICNTL(35)!=0)"
                        : "";
        case default:
            return "Unknown error";
        }
    }

    template <typename MatrixType_>
    const std::string MumpsBase<MatrixType_> :: handleWarning(int warning_1, double warning_2) const
    {
        if (warning_1 == 0)
        {
            return "No warning";
        }

        std::string output = "There were warnings on " + std::to_string(warning_2) + " processors\n";

        std::string binary_warning = std::bitset<5>(warning_1).to_string();

        int sum = 0;
        for (int i = 0; i < 5; i++)
        {
            sum += binary_warning[i] - '0';
        }

        output+= "There were " + sum + " different warnings:\n";

        if(binary_warning[4] == '1')
            output += "Index (in IRN or JCN) out of range. Action taken by subroutine is to ignore any such entries and continue\n";
        if(binary_warning[3] == '1')
            output += "During error analysis the max-norm of the computed solution is close to zero. " +
                    "In some cases, this could cause difficulties in the computation of RINFOG(6)\n";
        if(binary_warning[2] == '1')
            output += "ICNTL(49)=1,2 and not enough memory to compact S at the end of the factorization\n";
        if(binary_warning[1] == '1')
            output += "Warning return from the iterative refinement routine. More than ICNTL(10) iterations are required\n";
        if(binary_warning[0] == '1')
            output += "Warning return from rank-revealing feature (ICNTL(56)). " +
                    "The values of the inertia (INFOG(12)) and/or the determinant (ICNTL(33)) might not be consistent with the number of singularities\n";
        
        return output;
        }

    };
// }; // namespace Eigen

#endif // __MUMPS_H__

