#include <LinAlg/blas.hpp>

extern "C" {

double ddot_(const int *N,
             const double *X,
             const int *incX,
             const double *Y,
             const int *incY);

double dnrm2_(const int *N,
              const double *X,
              const int *incX);

double dasum_(const int *N,
              const double *X,
              const int *incX);

size_t idamax_(const int *N,
               const double *X,
               const int *incX);

//---- Level 1 blas (vector * vector)

void dcopy_(const int *N,
            const double *X,
            const int *incX,
            double *Y,
            const int *incY);

void daxpy_(const int *N,
            const double *alpha,
            const double *X,
            const int *incX,
            double *Y,
            const int *incY);

void drot_(const int *N,
           double *X,
           const int *incX,
           double *Y,
           const int *incY,
           const double *c,
           const double *s);

void dscal_(const int *N,
            const double *alpha,
            double *X,
            const int *incX);

//----- Level 2 blas (matrix * vector)

void dgemv_(const char * TransA,
            const int *M,
            const int *N,
            const double *alpha,
            const double *A,
            const int *lda,
            const double *X,
            const int *incX,
            const double *beta,
            double *Y,
            const int *incY);

void dtrmv_(const char *Uplo,
            const char *TransA,
            const char *Diag,
            const int *N,
            const double *A,
            const int *lda,
            double *X,
            const int *incX);

void dtrsv_(const char * Uplo,
            const char * TransA,
            const char * Diag,
            const int *N,
            const double *A,
            const int *lda,
            double *X,
            const int *incX);

void dsymv_(const char* Uplo,
            const int *N,
            const double *alpha,
            const double *A,
            const int *lda,
            const double *X,
            const int *incX,
            const double *beta,
            double *Y,
            const int *incY);

void dger_(const int *M,
           const int *N,
           const double *alpha,
           const double *X,
           const int *incX,
           const double *Y,
           const int *incY,
           double * A,
           const int *lda);

void dsyr_(const char *Uplo,
          const int *N,
          const double *alpha,
          const double *X,
          const int *incX,
          double *A,
          const int *lda);

void dsyr2_(const char *Uplo,
           const int *N,
           const double *alpha,
           const double *X,
           const int *incX,
           const double *Y,
           const int *incY,
           double *A,
           const int *lda);

//------ Level 3 blas (matrix * matrix)

void dgemm_(const char* TransA,
            const char* TransB,
            const int *M,
            const int *N,
            const int *K,
            const double *alpha,
            const double *A,
            const int *lda,
            const double *B,
            const int *ldb,
            const double *beta,
            double *C,
            const int *ldc);

void dsymm_(const char *Side,
            const char *Uplo,
            const int *M,
            const int *N,
            const double *alpha,
            const double *A,
            const int *lda,
            const double *B,
            const int *ldb,
            const double *beta,
            double *C,
            const int *ldc);

void dsyrk_(const char * Uplo,
            const char * Trans,
            const int *N,
            const int *K,
            const double *alpha,
            const double *A,
            const int *lda,
            const double *beta,
            double *C,
            const int *ldc);

void dsyr2k_(const char *Uplo,
             const char *Trans,
             const int *N,
             const int *K,
             const double *alpha,
             const double *A,
             const int *lda,
             const double *B,
             const int *ldb,
             const double *beta,
             double *C,
             const int *ldc);

void dtrmm_(const char * Side,
            const char * Uplo,
            const char * TransA,
            const char * Diag,
            const int *M,
            const int *N,
            const double *alpha,
            const double *A,
            const int *lda,
            double *B,
            const int *ldb);

void dtrsm_(const char * Side,
            const char * Uplo,
            const char * TransA,
            const char * Diag,
            const int *M,
            const int *N,
            const double *alpha,
            const double *A,
            const int *lda,
            double *B,
            const int *ldb);
}

namespace BOOM {
  namespace blas {
    const char TransposeChar[2][2] = {"N", "T"};
    const char UploChar[2][2]  = {"U", "L"};
    const char SideChar[2][2] = {"L", "R"};
    const char DiagChar[2][2] = {"N", "U"};

    double ddot(const int N,
                const double *X,
                const int incX,
                const double *Y,
                const int incY) {
      return ddot_(&N, X, &incX, Y, &incY);
    }

    double dnrm2(const int N,
                 const double *X,
                 const int incX) {
      return dnrm2_(&N, X, &incX);
    }

    double dasum(const int N,
                 const double *X,
                 const int incX) {
      return dasum_(&N, X, &incX);
    }

    size_t idamax(const int N,
                  const double *X,
                  const int incX) {
      return idamax_(&N, X, &incX);
    }

    void dcopy(const int N,
               const double *X,
               const int incX,
               double *Y,
               const int incY) {
      dcopy_(&N, X, &incX, Y, &incY);
    }

    void daxpy(const int N,
               const double alpha,
               const double *X,
               const int incX,
               double *Y,
               const int incY) {
      daxpy_(&N, &alpha, X, &incX, Y, &incY);
    }

    void drot(const int N,
              double *X,
              const int incX,
              double *Y,
              const int incY,
              const double c,
              const double s) {
      drot_(&N, X, &incX, Y, &incY, &c, &s);
    }

    void dscal(const int N,
               const double alpha,
               double *X,
               const int incX) {
      dscal_(&N, &alpha, X, &incX);
    }

    void dgemv(const TRANSPOSE TransA,
               const int M,
               const int N,
               const double alpha,
               const double *A,
               const int lda,
               const double *X,
               const int incX,
               const double beta,
               double *Y,
               const int incY) {
      dgemv_(TransposeChar[TransA], &M, &N, &alpha, A, &lda,
             X, &incX, &beta, Y, &incY);
    }

    void dtrmv(const UPLO Uplo,
               const TRANSPOSE TransA,
               const DIAG Diag,
               const int N,
               const double *A,
               const int lda,
               double *X,
               const int incX) {
      dtrmv_(UploChar[Uplo], TransposeChar[TransA], DiagChar[Diag],
             &N, A, &lda, X, &incX);
    }

    void dtrsv(const UPLO Uplo,
               const TRANSPOSE TransA,
               const DIAG Diag,
               const int N,
               const double *A,
               const int lda,
               double *X,
               const int incX) {
      dtrsv_(UploChar[Uplo], TransposeChar[TransA], DiagChar[Diag],
             &N, A, &lda, X, &incX);
    }


    void dsymv(const UPLO Uplo,
               const int N,
               const double alpha,
               const double *A,
               const int lda,
               const double *X,
               const int incX,
               const double beta,
               double *Y,
               const int incY) {
      dsymv_(UploChar[Uplo], &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY);
    }

    void dger(const int M,
              const int N,
              const double alpha,
              const double *X,
              const int incX,
              const double *Y,
              const int incY,
              double * A,
              const int lda) {
      dger_(&M, &N, &alpha, X, &incX, Y, &incY, A, &lda);
    }

    void dsyr(const UPLO Uplo,
              const int N,
              const double alpha,
              const double *X,
              const int incX,
              double *A,
              const int lda) {
      dsyr_(UploChar[Uplo], &N, &alpha, X, &incX, A, &lda);
    }

    void dsyr2(const UPLO Uplo,
               const int N,
               const double alpha,
               const double *X,
               const int incX,
               const double *Y,
               const int incY,
               double *A,
               const int lda) {
      dsyr2_(UploChar[Uplo], &N, &alpha, X, &incX, Y, &incY, A, &lda);
    }


    void dgemm(const TRANSPOSE TransA,
               const TRANSPOSE TransB,
               const int M,
               const int N,
               const int K,
               const double alpha,
               const double *A,
               const int lda,
               const double *B,
               const int ldb,
               const double beta,
               double *C,
               const int ldc) {
      dgemm_(TransposeChar[TransA], TransposeChar[TransB],
             &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    }

    void dsymm(const SIDE Side,
               const UPLO Uplo,
               const int M,
               const int N,
               const double alpha,
               const double *A,
               const int lda,
               const double *B,
               const int ldb,
               const double beta,
               double *C,
               const int ldc) {
      dsymm_(SideChar[Side], UploChar[Uplo], &M, &N, &alpha,
             A, &lda, B, &ldb, &beta, C, &ldc);
    }

    void dsyrk(const UPLO Uplo,
               const TRANSPOSE Trans,
               const int N,
               const int K,
               const double alpha,
               const double *A,
               const int lda,
               const double beta,
               double *C,
               const int ldc) {
      dsyrk_(UploChar[Uplo], TransposeChar[Trans],
             &N, &K, &alpha, A, &lda, &beta, C, &ldc);
    }

    void dsyr2k(const UPLO Uplo,
                const TRANSPOSE Trans,
                const int N,
                const int K,
                const double alpha,
                const double *A,
                const int lda,
                const double *B,
                const int ldb,
                const double beta,
                double *C,
                const int ldc) {
      dsyr2k_(UploChar[Uplo], TransposeChar[Trans], &N, &K, &alpha, A,
              &lda, B, &ldb, &beta, C, &ldc);
    }

    void dtrmm(const SIDE Side,
               const UPLO Uplo,
               const TRANSPOSE TransA,
               const DIAG Diag,
               const int M,
               const int N,
               const double alpha,
               const double *A,
               const int lda,
               double *B,
               const int ldb) {
      dtrmm_(SideChar[Side], UploChar[Uplo],
             TransposeChar[TransA], DiagChar[Diag],
             &M, &N, &alpha, A, &lda, B, &ldb);
    }

    void dtrsm(const SIDE Side,
               const UPLO Uplo,
               const TRANSPOSE TransA,
               const DIAG Diag,
               const int M,
               const int N,
               const double alpha,
               const double *A,
               const int lda,
               double *B,
               const int ldb) {
      dtrsm_(SideChar[Side], UploChar[Uplo],
             TransposeChar[TransA], DiagChar[Diag],
             &M, &N, &alpha, A, &lda, B, &ldb);
    }

    void initialize_blas_globals() {
      double A, B, C, X, Y = 1.0;
      dsymv(UPLO::Upper,
            1,
            1.0,
            &A,
            1,
            &X,
            1,
            1.0,
            &Y,
            1);
      dtrsv(UPLO::Upper,
            TRANSPOSE::NoTrans,
            DIAG::Unit,
            1,
            &A,
            1,
            &X,
            1);
      dtrmv(UPLO::Upper,
            TRANSPOSE::NoTrans,
            DIAG::Unit,
            1,
            &A,
            1,
            &X,
            1);
      dgemv(TRANSPOSE::NoTrans,
            1,
            1,
            1.0,
            &A,
            1,
            &X,
            1,
            1.0,
            &Y,
            1);
      dtrsm(SIDE::Left,
            UPLO::Upper,
            TRANSPOSE::NoTrans,
            DIAG::Unit,
            1,
            1,
            1.0,
            &A,
            1,
            &B,
            1);
      dtrmm(SIDE::Left,
            UPLO::Upper,
            TRANSPOSE::NoTrans,
            DIAG::Unit,
            1,
            1,
            1.0,
            &A,
            1,
            &B,
            1);
      dsyrk(UPLO::Upper,
            TRANSPOSE::NoTrans,
            1,
            1,
            1.0,
            &A,
            1,
            1.0,
            &C,
            1);
      dsyr(UPLO::Upper,
           1,
           1.0,
           &X,
           1,
           &A,
           1);
    }

  }  // namespace blas
}  // namespace BOOM
