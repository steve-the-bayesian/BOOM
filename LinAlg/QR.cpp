/*
  Copyright (C) 2005 Steven L. Scott

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
#include <LinAlg/QR.hpp>
#include <LinAlg/blas.hpp>
#include <LinAlg/SubMatrix.hpp>
#include <cpputil/report_error.hpp>
#include <cstring>

extern "C"{
  void dgeqrf_(int *, int *, double *, int *, double *,
               double *, int *, int *);
  void dorgqr_(int *, int *, int *, double *, int *, const double *,
               const double *, const int *, int *);
  void dormqr_(const char *, const char *, int *, int *, int *, const double *,
               int *, const double *, double *, int *, double *, int *, int *);
  void dtrtrs_(const char *,const char *,const char *, int *, int *,
               const double *, int *, double *, int *, int *);
}

namespace BOOM{
  using namespace blas;

  QR::QR(const Matrix &mat)
    : dcmp(mat),
      tau(std::min(mat.nrow(), mat.ncol())),
      work(mat.ncol())
  {
    int m = mat.nrow();
    int n = mat.ncol();
    int info=0;
    lwork = -1;
    // have LAPACK compute optimal lwork...
    dgeqrf_(&m, &n, dcmp.data(), &m, tau.data(), work.data(), &lwork, &info);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    // compute the decomposition with the optimal value...
    dgeqrf_(&m, &n, dcmp.data(), &m, tau.data(), work.data(), &lwork, &info);
  }

  Matrix QR::getQ()const{
    Matrix ans(dcmp);
    int m = ans.nrow();
    int n = ans.ncol();
    // If *this = Q * R is wider than it is tall, then R is actually a
    // trapezoidal matrix (zero entries below the diagonal), and Q is
    // square, with dimension nrow(ans).  If *this is taller and
    // skinny then the dimensions of Q match those of *this, and R is
    // square with dimension ncol(ans).

    // k is the number "elementary reflectors" determining the matrix
    // Q.  The documentation for dgeqrf specifies k = min(#rows, #cols).
    int k = std::min(m,n);
    int info=0;
    dorgqr_(&m,    // number of rows in Q
            &k,    // number of columns in Q.
            &k,    // number of elementary reflections defining Q
            ans.data(),  // Data from the QR decomposition.
            &m,          // leading dimension of ans
            tau.data(),  // scalar factors of the elementary
                         // reflectors returned by dgeqrf
            work.data(), // workspace
            &lwork,      // dimension of workspace
            &info);      // info
    if (n > m) {
      // If the number of columns is greater than the nubmer of rows,
      // then we just want the square part of the result.
      ans = SubMatrix(ans, 0, m-1, 0, m-1);
    }
    return ans;
  }

  Matrix QR::getR()const{
    uint m = dcmp.nrow();
    uint n = dcmp.ncol();
    uint k = std::min(m,n);
    Matrix ans(k,n, 0.0);
    if(m>=n){  // usual case
      for(uint i=0; i<n; ++i)
        std::copy(dcmp.col_begin(i), dcmp.col_begin(i)+i+1,
            ans.col_begin(i));
    }else{
      for(uint i=0; i<m; ++i)
        std::copy(dcmp.col_begin(i),     // triangular part
            dcmp.col_begin(i)+i+1,
            ans.col_begin(i));
      for(uint i=m; i<n; ++i)
        std::copy(dcmp.col_begin(i),     // rectangular part
            dcmp.col_begin(i)+m,
            ans.col_begin(i));
    }
    return ans;
  }

  Matrix QR::solve(const Matrix &B)const{
    Matrix ans(B);
    int m = dcmp.nrow();
    int n = dcmp.ncol(); // same as B.nrow
    int k = std::min(m,n);
    int ncol_b = ans.ncol();
    Vector work(1);
    int lwork = -1;
    int info=0;

    // set ans = Q^T*B
    dormqr_("L", "T", &n, &ncol_b, &k, dcmp.data(), &m, tau.data(),
            ans.data(), &n, work.data(), &lwork, &info);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    dormqr_("L", "T", &n, &ncol_b, &k, dcmp.data(), &m, tau.data(),
            ans.data(), &n, work.data(), &lwork, &info);

    // set ans = R^{-1} * ans
    dtrtrs_("U", "N", "N", &k, &ncol_b, dcmp.data(), &m,
            ans.data(), &n, &info);
    return ans;
  }

  Vector QR::Qty(const Vector &y)const{
    if (length(y) != dcmp.nrow()) {
      report_error("Wrong size argument y passed to QR::Qty.");
    }
    Vector ans(y);
    const char * side="L";
    const char * trans="T";
    int m = y.size();
    int ldc = y.size();
    int n = 1;
    //    int k = std::min(dcmp.nrow(), dcmp.ncol());
    int k = length(tau);
    const double * a = dcmp.data();
    Vector work(1);
    int lwork = -1;
    int info=0;

    // set ans = Q^T*y          comments show LAPACK argument names
    dormqr_(side,               // side
            trans,              // trans
            &m,                 // m   nrow(y)
            &n,                 // n   ncol(y) := 1
            &k,                 // k   dcmp.nrow()
            a,                  // a
            &m,                 // lda
            tau.data(),         // tau
            ans.data(),         // C
            &ldc,               // ldc
            work.data(),        // work
            &lwork,             // lwork
            &info);             // info
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    dormqr_(side,
            trans,
            &m,
            &n,
            &k,
            a,
            &m,
            tau.data(),
            ans.data(),
            &ldc,
            work.data(),
            &lwork,
            &info);
    if (dcmp.ncol() < dcmp.nrow()) {
      ans.resize(dcmp.ncol());
    }
    return ans;
  }

  Matrix QR::QtY(const Matrix &Y)const{
    Matrix ans(Y);
    int m = ans.nrow();
    int n = ans.ncol();
    int k = tau.size();
    int lda = dcmp.nrow();
    int ldc = Y.nrow();
    Vector work(1);
    int lwork = -1;
    int info=0;

    // set ans = Q^T*Y
    dormqr_("L",          // Side
            "T",          // Trans
            &m,           // M
            &n,           // N
            &k,           // K
            dcmp.data(),  // A
            &lda,         // LDA
            tau.data(),   // TAU
            ans.data(),   // C
            &ldc,         // LDC
            work.data(),  // WORK
            &lwork,       // LWORK
            &info);       // INFO
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    dormqr_("L",
            "T",
            &m,
            &n,
            &k,
            dcmp.data(),
            &lda,
            tau.data(),
            ans.data(),
            &ldc,
            work.data(),
            &lwork,
            &info);
    if (dcmp.nrow() > dcmp.ncol()) {
      Matrix ans2;
      ans2 = SubMatrix(ans, 0, dcmp.ncol() - 1, 0, ans.ncol() - 1);
      return ans2;
    }
    return ans;
  }

  Vector QR::solve(const Vector &B)const{
    Vector ans(B);
    int m = dcmp.nrow();
    int n = dcmp.ncol(); // same as B.nrow
    int k = std::min(m,n);
    int ncol_b = 1;
    Vector work(1);
    int lwork = -1;
    int info=0;

    // set ans = Q^T*B
    dormqr_("L", "T", &n, &ncol_b, &k, dcmp.data(), &m, tau.data(),
            ans.data(), &n, work.data(), &lwork, &info);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    dormqr_("L", "T", &n, &ncol_b, &k, dcmp.data(), &m, tau.data(),
            ans.data(), &n, work.data(), &lwork, &info);

    // set ans = R^{-1} * ans
    dtrtrs_("U", "N", "N", &k, &ncol_b, dcmp.data(), &m,
            ans.data(), &n, &info);
    return ans;
  }

  double QR::det()const{
    double ans = 1.0;
    uint m = std::min(dcmp.nrow(), dcmp.ncol());
    for(uint i=0; i<m; ++i) ans*= dcmp.unchecked(i,i);
    return ans; }

  void QR::decompose(const Matrix &mat){
    dcmp = mat;
    tau.resize(std::min(dcmp.nrow(), dcmp.ncol()));
    work.resize(dcmp.ncol());
    int m = dcmp.nrow();
    int n = dcmp.ncol();
    int info=0;
    lwork = -1;
    // have LAPACK compute optimal lwork...
    dgeqrf_(&m, &n, dcmp.data(), &m, tau.data(), work.data(), &lwork, &info);
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    // compute the decomposition with the optimal value...
    dgeqrf_(&m, &n, dcmp.data(), &m, tau.data(), work.data(), &lwork, &info);
  }

  void QR::clear(){
    dcmp = Matrix();
    tau = Vector();
    work = Vector();
    lwork = -1;
  }

  Vector QR::Rsolve(const Vector &Qty)const{
    Vector ans(Qty);
    Matrix R(getR());
    dtrsv(Upper, NoTrans, NonUnit, R.nrow(), R.data(), R.nrow(),
          ans.data(), ans.stride());
    return ans;
  }

  Matrix QR::Rsolve(const Matrix & QtY)const{
    Matrix ans(QtY);
    Matrix R(getR());
    int m = ans.nrow();
    int n = ans.ncol();
    dtrsm(Left,
          Upper,
          NoTrans,
          NonUnit,
          m,
          n,
          1.0,
          R.data(),
          R.nrow(),
          ans.data(),
          ans.nrow());
    return ans;
  }

  Vector QR::vectorize()const{
    int total_size = dcmp.size() + 2 + tau.size() + 1 + work.size() + 1 + 1;
    Vector ans(total_size);
    double *dp = ans.data();

    *dp = static_cast<double>(nrow()); ++dp;
    *dp = static_cast<double>(ncol()); ++dp;
    memcpy(dp, dcmp.data(), dcmp.size() * sizeof(double));
    dp += dcmp.size();

    *dp = tau.size(); ++dp;
    memcpy(dp, tau.data(), tau.size() * sizeof(double));
    dp += tau.size();

    *dp = work.size(); ++dp;
    memcpy(dp, work.data(), work.size() * sizeof(double));
    dp += work.size();
    *dp = lwork;

    return ans;
  }

  const double *QR::unvectorize(const double *dp) {
    int nrow = lround(*dp); ++dp;
    int ncol = lround(*dp); ++dp;
    dcmp.resize(nrow, ncol);
    memcpy(dcmp.data(), dp, nrow * ncol * sizeof(double));
    dp += (nrow * ncol);

    int tau_size = lround(*dp); ++dp;
    tau.resize(tau_size);
    memcpy(tau.data(), dp, tau_size * sizeof(double));
    dp += tau_size;

    int work_size = lround(*dp); ++dp;
    work.resize(work_size);
    memcpy(work.data(), dp, work_size * sizeof(double));
    dp += work_size;

    lwork = lround(*dp);
    ++dp;

    return dp;
  }

}  // namespace BOOM
