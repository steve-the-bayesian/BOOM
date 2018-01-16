/*
  Copyright (C) 2005-2014 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BOOM_BLAS_WRAPPER_HPP_
#define BOOM_BLAS_WRAPPER_HPP_

#include <cstddef>  // for size_t

// This is a wrapper around the blas functions used by BOOM.  BOOM
// does not implement its own BLAS, but having BLAS functions wrapped
// makes it relatively easy to switch BLAS implementations.

namespace BOOM {
  namespace blas {
    enum TRANSPOSE {
      NoTrans = 0,
      Trans
    };

    enum UPLO {
      Upper = 0,
      Lower
    };

    enum SIDE {
      Left = 0,
      Right
    };

    enum DIAG {
      NonUnit = 0,
      Unit
    };

    double ddot(const int N,
                const double *X,
                const int incX,
                const double *Y,
                const int incY);

    double dnrm2(const int N,
                 const double *X,
                 const int incX);

    double dasum(const int N,
                 const double *X,
                 const int incX);

    size_t idamax(const int N,
                  const double *X,
                  const int incX);

    //----- Level 1 blas (vector * vector)
    void dcopy(const int N,
               const double *X,
               const int incX,
               double *Y,
               const int incY);

    void daxpy(const int N,
               const double alpha,
               const double *X,
               const int incX,
               double *Y,
               const int incY);

    void drot(const int N,
              double *X,
              const int incX,
              double *Y,
              const int incY,
              const double c,
              const double s);

    void dscal(const int N,
               const double alpha,
               double *X,
               const int incX);

    //----- Level 2 blas (matrix * vector)

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
               const int incY);

    void dtrmv(const UPLO Uplo,
               const TRANSPOSE TransA,
               const DIAG Diag,
               const int N,
               const double *A,
               const int lda,
               double *X,
               const int incX);

    void dtrsv(const UPLO Uplo,
               const TRANSPOSE TransA,
               const DIAG Diag,
               const int N,
               const double *A,
               const int lda,
               double *X,
               const int incX);

    void dsymv(const UPLO Uplo,
               const int N,
               const double alpha,
               const double *A,
               const int lda,
               const double *X,
               const int incX,
               const double beta,
               double *Y,
               const int incY);

    void dger(const int M,
              const int N,
              const double alpha,
              const double *X,
              const int incX,
              const double *Y,
              const int incY,
              double * A,
              const int lda);

    void dsyr(const UPLO Uplo,
              const int N,
              const double alpha,
              const double *X,
              const int incX,
              double *A,
              const int lda);

    void dsyr2(const UPLO Uplo,
               const int N,
               const double alpha,
               const double *X,
               const int incX,
               const double *Y,
               const int incY,
               double *A,
               const int lda);

    //----- Level 3 blas (matrix * matrix)

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
               const int ldc);

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
               const int ldc);

    void dsyrk(const UPLO Uplo,
               const TRANSPOSE Trans,
               const int N,
               const int K,
               const double alpha,
               const double *A,
               const int lda,
               const double beta,
               double *C,
               const int ldc);

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
                const int ldc);

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
               const int ldb);

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
               const int ldb);

    // Initialization function to allow for thread-safe use of routines that
    // construct static lookup tables on first use.

    void initialize_blas_globals();

  }  // namespace blas
}  // namespace BOOM
#endif // BOOM_BLAS_WRAPPER_HPP_
