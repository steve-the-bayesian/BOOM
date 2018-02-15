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

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Cholesky.hpp"
#include "LinAlg/LU.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/blas.hpp"

#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <sstream>

extern "C" {
  void dpotrf_(const char *uplo, const int *n, double *data, const int *lda,
               int *info);
  void dposv_(const char *uplo, const int *n, const int *nrhs, double *spd_a,
              const int *lda, double *rhs_b, const int *b, int *info);
  void dpotri_(const char *, int *, double *, int *, int *);

  void dsyevr_(const char *JOBZ,
               const char * RANGE,
               const char * UPLO,
               const int * N,
               double *A,
               const int *LDA,
               const double *VL,
               const double *VU,
               const int *IL,
               const int *IU,
               const double *abstol,
               const int *M,
               double *evals,
               double *evecs,
               const int *LDZ,
               int * isuppz,
               double *work,
               const int *lwork,
               int * iwork,
               const int *liwork,
               int *info);
  double dlamch_(const char *);

}  // extern "C"

namespace BOOM {
  using namespace blas;

  typedef std::vector<double> dVector;

  SpdMatrix::SpdMatrix() {}

  SpdMatrix::SpdMatrix(uint dim, double x)
      : Matrix(dim, dim)
  {
    if (dim > 0) set_diag(x);
  }

  SpdMatrix::SpdMatrix(uint n, double *x, bool ColMajor)
      : Matrix(n, n, x, ColMajor)
  {}

  SpdMatrix::SpdMatrix(const Vector &v, bool minimal)
  {
    if (v.empty()) return;
    size_t dimension = 0;
    if (minimal) {
      dimension = lround((-1 + sqrt(1 + 8 * v.size())) / 2.0);
      if (dimension * (dimension + 1) != 2 * v.size()) {
        report_error("Wrong size Vector argument to SpdMatrix constructor.");
      }
    } else {
      dimension = lround(sqrt(v.size()));
      if (dimension * dimension != v.size()) {
        report_error("Wrong size Vector argument to SpdMatrix constructor.");
      }
    }
    this->resize(dimension);
    unvectorize(v, minimal);
  }

  SpdMatrix::SpdMatrix(const Matrix &A, bool check)
      : Matrix(A)
  {
    if (check && !A.is_sym()) {
      report_error("Matrix argument to SpdMatrix is not symmetric.");
    }
  }

  SpdMatrix::SpdMatrix(const SubMatrix &rhs, bool check)
  {
    if (check && (rhs.nrow() != rhs.ncol())) {
      report_error("SpdMatrix constructor was supplied a non-square"
                   "SubMatrix argument");
    }
    operator=(rhs);
  }

  SpdMatrix::SpdMatrix(const ConstSubMatrix &rhs, bool check)
  {
    if (check && rhs.nrow() != rhs.ncol()) {
      report_error("SpdMatrix constructor was supplied a non-square"
                   "SubMatrix argument");
    }
    operator=(rhs);
  }

  SpdMatrix & SpdMatrix::operator=(const SubMatrix &rhs) {
    if (rhs.nrow() != rhs.ncol()) {
      report_error("SpdMatrix::operator= called with rectangular "
                   "RHS argument");
    }
    Matrix::operator=(rhs);
    return *this;
  }

  SpdMatrix & SpdMatrix::operator=(const ConstSubMatrix &rhs) {
    if (rhs.nrow() != rhs.ncol()) {
      report_error("SpdMatrix::operator= called with rectangular "
                   "RHS argument");
    }
    Matrix::operator=(rhs);
    return *this;
  }

  SpdMatrix & SpdMatrix::operator=(const Matrix &rhs) {
    assert(rhs.is_sym());
    Matrix::operator=(rhs);
    return *this;
  }

  SpdMatrix & SpdMatrix::operator=(double x) {
    Matrix::operator=(x);
    return *this;
  }

  bool SpdMatrix::operator==(const SpdMatrix &rhs) const {
    return Matrix::operator == (rhs);}

  void SpdMatrix::swap(SpdMatrix &rhs) { Matrix::swap(rhs); }

  SpdMatrix & SpdMatrix::randomize() {
    Matrix::randomize();
    SpdMatrix tmp(nrow());
    dsyrk(Upper, Trans, nrow(), nrow(),
          1.0, data(), nrow(), 0.0, tmp.data(), tmp.nrow());
    swap(tmp);
    reflect();
    return *this;
  }

  uint SpdMatrix::nelem() const {
    uint n = nrow();
    return n * (n + 1) / 2;
  }

  SpdMatrix & SpdMatrix::resize(uint n) {
    Matrix::resize(n, n);
    return *this;
  }

  SpdMatrix & SpdMatrix::set_diag(double x, bool zero) {
    Matrix::set_diag(x, zero);
    return *this; }

  SpdMatrix & SpdMatrix::set_diag(const Vector &v, bool zero) {
    Matrix::set_diag(v, zero);
    return *this; }

  inline void zero_upper(SpdMatrix &V) {
    uint n = V.nrow();
    for (uint i = 0; i < n; ++i) {
      dVector::iterator b = V.col_begin(i);
      dVector::iterator e = b+i;
      std::fill(b, e, 0.0);}}

  Matrix SpdMatrix::chol() const { bool ok = true; return chol(ok);}
  Matrix SpdMatrix::chol(bool &ok) const {
    SpdMatrix ans(*this);
    if (this->nrow() == 0) return ans;
    ans.reflect();
    int n = ans.nrow();
    int info = 0;
    dpotrf_("L", &n, ans.data(), &n, &info);
    ok = (info == 0);
    zero_upper(ans);
    return ans;
  }

  SpdMatrix SpdMatrix::inv() const {bool ok = true; return inv(ok);}
  SpdMatrix SpdMatrix::inv(bool & ok) const {
    int n = nrow();
    int info = 0;
    SpdMatrix LLT(*this);
    SpdMatrix ans(Id());
    if (n == 0) return ans;
    dposv_("U", &n, &n, LLT.data(), &n, ans.data(), &n, &info);
    ok = info == 0;
    return ans;
  }

  double SpdMatrix::det() const {
    Chol L(*this);
    if (L.is_pos_def()) return std::exp(L.logdet());

    LU L2(*this);
    return L2.det();
  }

  double SpdMatrix::logdet() const {
    bool ok(true);
    return logdet(ok);}

  double SpdMatrix::logdet(bool &ok) const {
    ok = true;
    uint n = nrow();
    if (n == 0) {
      return negative_infinity();
    } else if (n == 1) {
      double x = data()[0];
      if (x <= 0) {
        ok = false;
        return negative_infinity();
      } else {
        return std::log(x);
      }
    } else if (n == 2) {
      const double *values(data());
      // If the matrix needs to reflect then the upper triangle is
      // current, but the lower triangle might not be.  In that case
      // prefer looking at values[2], the upper-right element, over
      // values[1], the lower left element.
      double determinant = values[0] * values[3] - values[2] * values[2];
      if (determinant <= 0) {
        ok = false;
        return negative_infinity();
      }
      return std::log(determinant);
    } else {
      Matrix L(chol(ok));
      if (!ok) return BOOM::negative_infinity();
      double ans = 0.0;
      for (uint i = 0; i < n; ++i) ans += std::log(L(i, i));
      ans *= 2;
      return ans;
    }
  }

  Matrix SpdMatrix::solve(const Matrix &rhs) const {
    assert(rhs.nrow() == ncol());
    int n = nrow();
    int nrhs = rhs.ncol();
    int info = 0;
    SpdMatrix LLT(*this);
    Matrix ans(rhs);
    if (n == 0) return ans;
    dposv_("U", &n, &nrhs, LLT.data(), &n, ans.data(), &n, &info);
    if (info != 0) {
      ostringstream msg;
      msg << "Matrix not positive definite in SpdMatrix::solve(Matrix)"
          << std::endl
          << "info = " << info << std::endl
          << "arguments: " << std::endl
          << "n = " << n << "  nrhs = " << nrhs << std::endl
          << "SpdMatrix: " << std::endl
          << *this << std::endl;
      report_error(msg.str());
    }
    return ans;
  }

  Vector SpdMatrix::solve(const Vector &rhs) const {
    bool ok = true;
    Vector ans(this->solve(rhs, ok));
    if (!ok) {
      ostringstream msg;
      msg << "Matrix not positive definite in SpdMatrix::solve(Vector)."
          << std::endl;
      report_error(msg.str());
    }
    return ans;
  }

  Vector SpdMatrix::solve(const Vector &rhs, bool &ok) const {
    assert(rhs.size() == ncol());
    int n = nrow();
    int nrhs = 1;
    int info = 0;
    SpdMatrix LLT(*this);
    Vector ans(rhs);
    if (n == 0) return ans;
    dposv_("U", &n, &nrhs, LLT.data(), &n, ans.data(), &n, &info);
    if (info != 0) {
      ok = false;
      return rhs.zero() + negative_infinity();
    }
    ok = true;
    return ans;
  }

  void SpdMatrix::reflect() {
    uint n = nrow();
    double *d = data();
    for (uint i = 0; i < n; ++i) {
      uint pos = i * n + i;
      double * row = d + pos;          // stride is n, length is n - i
      double * col = d + pos;          // stride is 1, length is n - i
      dcopy(n - i, row, n, col, 1);    // y is column, x is row
    }
  }

  double SpdMatrix::Mdist(const Vector &x, const Vector &y) const {
    return Mdist(x - y);
  }

  double SpdMatrix::Mdist(const Vector &x) const {
    int n = x.size();
    if (n != nrow()) {
      report_error("Wrong size x passed to SpdMatrix::Mdist");
    }
    const double *xdata(x.data());
    const double *thisdata(data());
    double ans = 0;
    for (int j = 0; j < n; ++j) {
      ans += xdata[j] * xdata[j] * thisdata[INDX(j, j)];
      for (int i = j + 1; i < n; ++i) {
        ans += 2 * xdata[j] * xdata[i] * thisdata[INDX(i, j)];
      }
    }
    return ans;
  }

  template <class V>
      void local_add_outer(SpdMatrix &S, const V &v, double w) {
    assert(v.size() == S.nrow());
    if (S.nrow() == 0) return;
    dsyr(Upper, v.size(), w, v.data(), v.stride(),
         S.data(), S.nrow());
  }

  SpdMatrix & SpdMatrix::add_outer(const Vector &v, double w, bool force_sym) {
    local_add_outer<Vector>(*this, v, w);
    if (force_sym) reflect();
    return *this; }

  SpdMatrix & SpdMatrix::add_outer(const VectorView &v, double w,
                                   bool force_sym) {
    local_add_outer<VectorView>(*this, v, w);
    if (force_sym) reflect();
    return *this; }

  SpdMatrix & SpdMatrix::add_outer(const ConstVectorView &v, double w,
                                   bool force_sym) {
    local_add_outer<ConstVectorView>(*this, v, w);
    if (force_sym) reflect();
    return *this; }

  SpdMatrix & SpdMatrix::add_outer(const Matrix &X, double w, bool force_sym) {
    assert(X.nrow() == this->nrow());
    int n = nrow();
    assert(X.ncol() == this->nrow());
    uint k = X.ncol();
    if (n == 0 || k == 0) return *this;
    dsyrk(Upper,     // uplo
          NoTrans,   // trans
          n,              // N      number of rows in *this
          k,              // k      number of columns in X
          w,              // alpha  scale factor for X * X^T
          X.data(),       // A
          n,              // lda
          1.0,            // beta   scale factor for *this
          this->data(),   // C
          n);             // ldc  (number of rows in C)
    if (force_sym) reflect();
    return *this;
  }

  SpdMatrix & SpdMatrix::add_inner(const Matrix &X, const Vector &w,
                                   bool force_sym) {
    assert(X.nrow() == w.size());
    assert(X.ncol() == this->ncol());
    uint n = w.size();
    for (uint i = 0; i < n; ++i) {
      this->add_outer(X.row(i), w[i], false);
    }
    if (force_sym) reflect();
    return *this;
  }


  SpdMatrix & SpdMatrix::add_inner(const Matrix &x, double w) {
    int n = nrow();
    assert(x.ncol() == this->nrow());
    uint k = x.nrow();
    if (n == 0 || k == 0) return *this;
    dsyrk(Upper, Trans, n, k, w, x.data(), k, 1.0, this->data(), n);
    reflect();
    return *this;
  }

  SpdMatrix & SpdMatrix::add_inner2(const Matrix &A, const Matrix &B,
                                    double w) {
    // adds w*(A^TB + B^TA)
    assert(A.ncol() == B.ncol() && A.ncol() == nrow());
    assert(A.nrow() == B.nrow());
    if (nrow() == 0) return *this;
    dsyr2k(Upper,
           Trans,
           nrow(),
           A.nrow(),
           w,
           A.data(),
           A.nrow(),
           B.data(),
           B.nrow(),
           1.0,
           data(),
           nrow());
    reflect();
    return *this;
  }


  SpdMatrix & SpdMatrix::add_outer2(const Matrix &A, const Matrix &B,
                                    double w) {
    // adds w*(AB^T + BA^T)
    assert(A.nrow() == B.nrow()  &&  B.nrow() == nrow());
    assert(B.ncol() == A.ncol());
    if (nrow() == 0) return *this;
    dsyr2k(Upper,
           NoTrans,
           nrow(),
           A.ncol(),
           w,
           A.data(),
           A.nrow(),
           B.data(),
           B.nrow(),
           1.0,
           data(),
           nrow());
    reflect();
    return *this;
  }

  SpdMatrix & SpdMatrix::add_outer2(const Vector &x,
                                    const Vector &y,
                                    double w) {
    assert(x.size() == nrow() && y.size() == ncol());
    if (nrow() == 0) return *this;
    dsyr2(Upper, nrow(), w,
          x.data(), x.stride(),
          y.data(), y.stride(),
          data(), nrow());
    reflect();
    return *this;
  }

  //-------------- multiplication --------------------


  //---------- general_Matrix ---------
  Matrix & SpdMatrix::mult(const Matrix &B, Matrix &ans, double scal) const {
    assert(can_mult(B, ans));
    uint m = nrow();
    uint n = B.ncol();
    if (n == 0 || m == 0) return ans;
    dsymm(Left, Upper, m, n, scal, data(), nrow(), B.data(), B.nrow(),
          0.0, ans.data(), ans.nrow());
    return ans; }

  Matrix & SpdMatrix::Tmult(const Matrix &B, Matrix &ans, double scal) const {
    return mult(B, ans, scal);}

  Matrix & SpdMatrix::multT(const Matrix &B, Matrix & ans, double scal) const {
    return Matrix::multT(B, ans, scal);}

  //---------- SpdMatrix ---------
  Matrix & SpdMatrix::mult(const SpdMatrix &B, Matrix &ans,
                           double scal) const {
    const Matrix &A(B);
    return mult(A, ans, scal);}

  Matrix & SpdMatrix::Tmult(const SpdMatrix &B, Matrix &ans,
                            double scal) const {
    const Matrix &A(B);
    return Tmult(A, ans, scal);}

  Matrix & SpdMatrix::multT(const SpdMatrix &B, Matrix &ans,
                            double scal) const {
    const Matrix &A(B);
    return multT(A, ans, scal);}

  //--------- DiagonalMatrix this and B are both symmetric ---------
  Matrix & SpdMatrix::mult(
      const DiagonalMatrix &B, Matrix &ans, double scal) const {
    return Matrix::mult(B, ans, scal);
  }
  Matrix & SpdMatrix::Tmult(
      const DiagonalMatrix &B, Matrix &ans, double scal) const {
    return Matrix::mult(B, ans, scal);
  }
  Matrix & SpdMatrix::multT(
      const DiagonalMatrix &B, Matrix &ans, double scal) const {
    return Matrix::mult(B, ans, scal);
  }

  //--------- Vector --------------

  Vector & SpdMatrix::mult(const Vector &v, Vector & ans, double scal) const {
    assert(ans.size() == nrow());
    if (size() == 0) return ans;
    dsymv(Upper, nrow(), scal, data(), nrow(), v.data(), v.stride(), 0.0,
          ans.data(), ans.stride());
    return ans;}

  Vector & SpdMatrix::Tmult(const Vector &v, Vector & ans, double scal) const {
    return mult(v, ans, scal);}

  Vector SpdMatrix::vectorize(bool minimal) const {  // copies upper triangle
    uint n = ncol();
    uint ans_size = minimal ? nelem() : n*n;
    Vector ans(ans_size);
    Vector::iterator it = ans.begin();
    for (uint i = 0; i < n; ++i) {
      dVector::const_iterator b = col_begin(i);
      dVector::const_iterator e = minimal ? b+i+1 : b+n;
      it = std::copy(b, e, it);}
    return ans;
  }

  void SpdMatrix::unvectorize(const Vector &x, bool minimal) {
    Vector::const_iterator b(x.begin());
    unvectorize(b, minimal);}

  Vector::const_iterator SpdMatrix::unvectorize
      (Vector::const_iterator &b, bool minimal) {
    uint n = ncol();
    for (uint i = 0; i < n; ++i) {
      Vector::const_iterator e = minimal ? b+i+1 : b+n;
      dVector::iterator dest = col_begin(i);
      std::copy(b, e, dest);
      b = e;
    }
    reflect();
    return b;
  }

  void SpdMatrix::make_symmetric(bool have_upper) {
    uint n = ncol();
    for (uint i = 1; i < n; ++i) {
      for (uint j = 0; j < i; ++j) { // (i, j) is in the lower triangle
        if (have_upper) unchecked(i, j) = unchecked(j, i);
        else  unchecked(j, i) = unchecked(i, j);}}}

  // ================== non member functions ===========================
  SpdMatrix Id(uint p) {
    SpdMatrix ans(p);
    ans.set_diag(1.0);
    return ans;
  }

  SpdMatrix outer(const Vector &v) {
    SpdMatrix ans(v.size(), 0.0);
    ans.add_outer(v);
    return ans;
  }
  SpdMatrix outer(const VectorView &v) {
    SpdMatrix ans(v.size(), 0.0);
    ans.add_outer(v);
    return ans;
  }
  SpdMatrix outer(const ConstVectorView &v) {
    SpdMatrix ans(v.size(), 0.0);
    ans.add_outer(v);
    return ans;
  }

  SpdMatrix LLT(const Matrix &L, double a) {
    SpdMatrix ans(L.nrow());
    int n = L.nrow();
    int k = L.ncol();
    if (n == 0 || k == 0) return ans;
    dsyrk(Upper, NoTrans, n, k, a, L.data(), n, 0.0, ans.data(), n);
    ans.reflect();
    return ans;
  }

  SpdMatrix RTR(const Matrix &R, double a) {
    SpdMatrix ans(R.ncol());
    int n = R.nrow();
    int k = R.ncol();
    if (n == 0 || k == 0) return ans;
    dsyrk(Upper, Trans, n, k, a, R.data(), n, 0.0, ans.data(), n);
    ans.reflect();
    return ans;
  }

  SpdMatrix LTL(const Matrix &L) {
    Matrix ans(L);
    dtrmm(Left, Lower, Trans, NonUnit, L.nrow(), L.ncol(), 1.0,
          L.data(), L.nrow(), ans.data(), ans.nrow());
    return ans;
  }

  Matrix chol(const SpdMatrix &S) { return S.chol();}
  Matrix chol(const SpdMatrix &S, bool & ok) {return S.chol(ok);}

  SpdMatrix chol2inv(const Matrix &L) {
    assert(L.is_square());
    int n = L.nrow();
    SpdMatrix ans(L, false);
    if (n == 0) return ans;
    int info = 0;
    dpotri_("L", &n, ans.data(), &n, &info);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        ans(j, i) = ans(i, j);}}
    return ans;
  }

  SpdMatrix sandwich(const Matrix &A, const SpdMatrix &V) {  // AVA^T
    Matrix tmp(A.nrow(), V.ncol());
    if (A.size() == 0 || V.size() == 0) {
      return SpdMatrix(0);
    }
    dsymm(Right,
          Upper,
          tmp.nrow(),
          tmp.ncol(),
          1.0,
          V.data(),
          V.nrow(),
          A.data(),
          A.nrow(),
          0.0,
          tmp.data(),
          tmp.nrow());
    return matmultT(tmp, A);
  }

  SpdMatrix as_symmetric(const Matrix &A) {
    assert(A.is_square());
    Matrix ans = A.t();
    ans += A;
    ans/=2.0;
    return SpdMatrix(ans, false); // no symmetry check needed
  }

  SpdMatrix sum_self_transpose(const Matrix &A) {
    assert(A.is_square());
    uint n = A.nrow();
    SpdMatrix ans(n, 0.0);
    for (uint i = 0; i < n; ++i) {
      for (uint j = 0; j < i; ++j) {
        ans(i, j) = ans(j, i) = A(i, j) + A(j, i);}}
    return ans;
  }

  Vector eigenvalues(const SpdMatrix &X) {
    SpdMatrix tmp(X);
    int n = tmp.nrow();
    int nfound = 0;
    Vector ans(n);
    if (n == 0) return ans;
    double zero = 0.0;
    double abstol = dlamch_("Safe minimum");
    int lwork(-1);
    Vector work(1);
    std::vector<int> iwork(1);
    int liwork(-1);
    int info(0);
    std::vector<int> isuppz(2*n, 0);
    dsyevr_("N",         // JOBZ
            "A",         // RANGE
            "U",         // UPLO
            &n,          // N... dimension of problem
            tmp.data(),  // A    Matrix to decompose
            &n,          // LDA
            &zero, &zero, // VL, VU... not referenced
            &n, &n,      // IL, IU...  not referenced
            &abstol,     // criterion for eigenvalue convergence
            &nfound,     // M... number of eigenvalues found
            ans.data(),  // W... eigenvalues
            ans.data(), // Z... eigenVectors.. not referenced
            &n,          // LDZ
            &isuppz[0],  //
            work.data(),
            &lwork,
            &iwork[0],
            &liwork,
            &info);

    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    liwork = static_cast<int>(iwork[0]);
    iwork.resize(liwork);

    dsyevr_("N",         // JOBZ
            "A",         // RANGE
            "U",         // UPLO
            &n,          // N... dimension of problem
            tmp.data(),  // A    Matrix to decompose
            &n,          // LDA
            &zero, &zero, // VL, VU... not referenced
            &n, &n,      // IL, IU...  not referenced
            &abstol,     // criterion for eigenvalue convergence
            &nfound,     // M... number of eigenvalues found
            ans.data(),  // W... eigenvalues
            ans.data(),  // Z... eigenVectors.. not referenced
            &n,          // LDZ
            &isuppz[0],  //
            work.data(),
            &lwork,
            &iwork[0],
            &liwork,
            &info);

    return ans;
  }

  Vector eigen(const SpdMatrix &X, Matrix & Z) {
    SpdMatrix tmp(X);
    int n = tmp.nrow();
    Z.resize(n, n);
    int nfound = 0;
    Vector ans(n);
    if (n == 0) return ans;
    double zero = 0.0;
    double abstol = dlamch_("Safe minimum");
    int lwork(-1);
    Vector work(1);
    std::vector<int> iwork(1);
    int liwork(-1);
    int info(0);
    std::vector<int> isuppz(2*n, 0);
    dsyevr_("V",         // JOBZ
            "A",         // RANGE
            "U",         // UPLO
            &n,          // N... dimension of problem
            tmp.data(),  // A    Matrix to decompose
            &n,          // LDA
            &zero, &zero, // VL, VU... not referenced
            &n, &n,      // IL, IU...  not referenced
            &abstol,     // criterion for eigenvalue convergence
            &nfound,     // M... number of eigenvalues found
            ans.data(),  // W... eigenvalues
            Z.data(),    // Z... eigenVectors.. not referenced
            &n,          // LDZ
            &isuppz[0],  //
            work.data(),
            &lwork,
            &iwork[0],
            &liwork,
            &info);

    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    liwork = static_cast<int>(iwork[0]);
    iwork.resize(liwork);

    dsyevr_("V",         // JOBZ
            "A",         // RANGE
            "U",         // UPLO
            &n,          // N... dimension of problem
            tmp.data(),  // A    Matrix to decompose
            &n,          // LDA
            &zero, &zero, // VL, VU... not referenced
            &n, &n,      // IL, IU...  not referenced
            &abstol,     // criterion for eigenvalue convergence
            &nfound,     // M... number of eigenvalues found
            ans.data(),  // W... eigenvalues
            Z.data(),    // Z... eigenvectors.. not referenced
            &n,          // LDZ
            &isuppz[0],  //
            work.data(),
            &lwork,
            &iwork[0],
            &liwork,
            &info);

    return ans;
  }

  double largest_eigenvalue(const SpdMatrix &X) {
    SpdMatrix tmp(X);
    int n = tmp.nrow();
    if (n == 0) return negative_infinity();
    int nfound = 0;
    Vector ans(n);
    double zero = 0.0;
    double abstol = dlamch_("Safe minimum");
    int IL = n;  // Indices of the smallest and largest
    int IU = n;  // eigenvalues to be found
    int lwork(-1);
    Vector work(1);
    std::vector<int> iwork(1);
    int liwork(-1);
    int info(0);
    std::vector<int> isuppz(2*n, 0);
    dsyevr_("N",         // JOBZ
            "I",         // RANGE
            "U",         // UPLO
            &n,          // N... dimension of problem
            tmp.data(),  // A    Matrix to decompose
            &n,          // LDA
            &zero, &zero, // VL, VU... not referenced
            &IL, &IU,
            &abstol,     // criterion for eigenvalue convergence
            &nfound,     // M... number of eigenvalues found
            ans.data(),  // W... eigenvalues
            ans.data(),  // Z... eigenVectors.. not referenced
            &n,          // LDZ
            &isuppz[0],  //
            work.data(),
            &lwork,
            &iwork[0],
            &liwork,
            &info);

    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    liwork = static_cast<int>(iwork[0]);
    iwork.resize(liwork);

    dsyevr_("N",         // JOBZ
            "I",         // RANGE
            "U",         // UPLO
            &n,          // N... dimension of problem
            tmp.data(),  // A    Matrix to decompose
            &n,          // LDA
            &zero, &zero,// VL, VU... not referenced
            &IL,         // Indices of the smallest and largest
            &IU,         //   eigenvalues to be found
            &abstol,     // criterion for eigenvalue convergence
            &nfound,     // M... number of eigenvalues found
            ans.data(),  // W... eigenvalues
            ans.data(),  // Z... eigenVectors.. not referenced
            &n,          // LDZ
            &isuppz[0],  //
            work.data(),
            &lwork,
            &iwork[0],
            &liwork,
            &info);
    return ans[0];
  }

  SpdMatrix operator*(double x, const SpdMatrix &V) {
    SpdMatrix ans(V);
    ans *= x;
    return ans;
  }

  SpdMatrix operator*(const SpdMatrix &V, double x) {
    return x*V;
  }

  SpdMatrix operator/(const SpdMatrix &v, double x) {
    return v*(1.0/x);
  }

  SpdMatrix symmetric_square_root(const SpdMatrix &V) {
    Matrix eigenvectors(V.nrow(), V.nrow());
    Vector eigenvalues = eigen(V, eigenvectors);
    // We want Q^T Lambda^{1/2} Q.  We can get there by taking
    // Lambda^1/4 and pre-multiplying rows of Q.
    for (int i = 0; i < nrow(eigenvectors); ++i) {
      eigenvectors.col(i) *= sqrt(sqrt(eigenvalues[i]));
    }
    return eigenvectors.outer();
  }

  Matrix eigen_root(const SpdMatrix &X) {
    Matrix eigenvectors(X.nrow(), X.nrow());
    Vector eigenvalues = eigen(X, eigenvectors);
    for (int i = 0; i < nrow(eigenvectors); ++i) {
      eigenvectors.col(i) *= sqrt(eigenvalues[i]);
    }
    return eigenvectors.t();
  }

} // namespace BOOM
