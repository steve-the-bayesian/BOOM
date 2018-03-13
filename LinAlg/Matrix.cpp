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

// std library includes
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <functional>
#include <vector>
#include <string>

// other linear algebra
#include <LinAlg/Matrix.hpp>
#include <LinAlg/Vector.hpp>
#include <LinAlg/VectorView.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <LinAlg/SubMatrix.hpp>
#include <LinAlg/DiagonalMatrix.hpp>
#include <LinAlg/LU.hpp>
#include <LinAlg/Cholesky.hpp>
#include <LinAlg/QR.hpp>
#include <LinAlg/blas.hpp>

// other BOOM
#include <distributions.hpp>
#include <cpputil/string_utils.hpp>
#include <cpputil/Split.hpp>
#include <cpputil/report_error.hpp>

// lapack and blas

extern "C"{
  void dgesv_(const int *n, const int *nrhs, double *A, const int *lda,
              int *ipiv, double *ans, const int *ldb, int *info);
  int ilaenv_( const int *, const char *, const char *, const int *,
               const int *, const int *, const int *);
  void dgesdd_(const char *, const int *m, const int *n, double *A,
               const int *lda, double *values,
               double *u, const int *ldu, double *vt, const int *ldvt,
               double *wsp, const int *lwork, int *iwork, int *info);
  void dgeev_(const char *, const char *, const int *, double *,
              const int *, double *, double *, double *, const int *,
              double *, const int *, double *, const int *, int *);
}

typedef std::vector<double> dVector;

namespace BOOM{

  inline bool can_add(const Matrix &A, const Matrix &B) {
    return  (A.nrow() == B.nrow()) && (A.ncol() == B.ncol());
  }

  using namespace std;
  using namespace blas;
  Matrix::Matrix()
    : V(),
      nr_(0),
      nc_(0)
  {}

  Matrix::Matrix(uint nr, uint nc, double x)
    : V(nr*nc, x),
      nr_(nr),
      nc_(nc)
  {}

  Matrix::Matrix(uint nr, uint nc, const double *m,
           bool byrow)
    : V(&m[0], &m[nr*nc]),
      nr_(nr),
      nc_(nc)
  {
    if (byrow) {
      for (uint i=0; i<nr; ++i) {
        for (uint j=0; j<nc; ++j) {
          V[INDX(i,j)]= *m++;}}}
  }

  Matrix::Matrix(uint nr, uint nc, const std::vector<double> &v, bool byrow)
      : V(v),
        nr_(nr),
        nc_(nc)
  {
    if (v.size() != nr * nc) {
      ostringstream err;
      err << "Size of vector (" << v.size()
          << ") does not match dimensions ("
          << nr << ", " << nc << ") in Matrix constructor.";
      report_error(err.str());
    }
    if (byrow) {
      const double *d(v.data());
      for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nc; ++j) {
          V[INDX(i, j)] = *d++;
        }
      }
    }
  }

  Matrix::Matrix(const std::string & s, const std::string &row_delim) {
    BOOM::StringSplitter rowsplit(row_delim);
    std::vector<string> row_strings = rowsplit(s);

    nr_ = row_strings.size();
    std::vector<Vector> v;
    v.reserve(nr_);
    nc_ =0;
    for (uint i=0; i<nr_; ++i) {
      Vector row(row_strings[i]);
      v.push_back(row);
      if (i==0) {
        nc_ = v[0].size();
      } else if (v[i].size()!=nc_) {
        string msg =
            "Attempt to initialize Matrix with rows of differing lengths";
        report_error(msg.c_str());
      }
    }
    V.resize(nr_*nc_);
    for (uint i=0; i<nr_; ++i) set_row(i,v[i]);
  }

  Matrix::Matrix(const SubMatrix &rhs) {
    operator=(rhs);
  }

  Matrix::Matrix(const ConstSubMatrix &rhs) {
    operator=(rhs);
  }

  Matrix & Matrix::operator=(const SubMatrix & rhs) {
    nr_ = rhs.nrow();
    nc_ = rhs.ncol();
    V.resize(nr_*nc_);
    for (uint j = 0; j<nc_; ++j) col(j) = rhs.col(j);
    return *this;
  }

  Matrix & Matrix::operator=(const ConstSubMatrix & rhs) {
    nr_ = rhs.nrow();
    nc_ = rhs.ncol();
    V.resize(nr_*nc_);
    for (uint j = 0; j<nc_; ++j) col(j) = rhs.col(j);
    return *this;
  }

  Matrix & Matrix::operator=(const double &x) {
    if (V.empty()) {
      V.resize(1);
      nr_ = nc_ = 1;
    }
    V.assign(V.size(), x); //
    return *this; }

  bool Matrix::operator==(const Matrix &rhs) const {
    if (nr_ != rhs.nr_ || nc_!=rhs.nc_) return false;
    else return (V== rhs.V); }

  void Matrix::swap(Matrix &rhs) {
    std::swap(nr_, rhs.nr_);
    std::swap(nc_, rhs.nc_);
    std::swap(V, rhs.V); }

  Matrix & Matrix::randomize() {
    uint n = nr_ * nc_;
    for (uint i=0; i<n; ++i) {
      V[i] = runif (0,1);
    }
    return *this;
  }

  Matrix::~Matrix() {}

  bool Matrix::all_finite() const {
    size_t n = V.size();
    const double *d(V.data());
    for (size_t i = 0; i < n; ++i){
      if (!std::isfinite(d[i])) {
        return false;
      }
    }
    return true;
  }

  uint Matrix::size() const { return V.size();}
  uint Matrix::nrow() const { return nr_;}
  uint Matrix::ncol() const { return nc_;}

  bool Matrix::is_sym(double tol) const {
    if (nr_ != nc_) return false;
    double num=0, denom=0;
    for (uint i=0; i<nr_; ++i) {
      for (uint j=0; j<i; ++j) {
        num+= fabs(unchecked(i,j) - unchecked(j,i));
        denom+= fabs(unchecked(i,j)) + fabs(unchecked(j,i));}}
    return (denom < tol || num/denom<=tol) ; }

  bool Matrix::same_dim(const Matrix &A) const {
    return nr_ == A.nr_ && nc_ == A.nc_;
  }

  bool Matrix::is_square() const { return nr_==nc_;}

  bool Matrix::is_pos_def() const {
    if (!is_square())return false;
    Chol choldc(*this);
    return choldc.is_pos_def(); }

  Matrix & Matrix::resize(uint nr, uint nc) {
    V.resize(nr*nc);
    nr_ = nr;
    nc_ = nc;
    return *this;
  }

  Matrix & Matrix::rbind(const Matrix &A) {
    if (nrow()==0) {
      *this = A;
      return *this;
    }
    assert(A.ncol()==ncol());
    uint m = nrow() + A.nrow();
    uint n = ncol();
    Matrix tmp(m,n);
    for (uint i=0; i<n; ++i) {
      dVector::iterator it =
        std::copy(col_begin(i), col_end(i), tmp.col_begin(i));
      std::copy(A.col_begin(i), A.col_end(i), it);
    }
    swap(tmp);
    //      swap(*this, tmp);
    return *this;
  }

  Matrix & Matrix::rbind(const Vector &A) {
    if (nrow()==0) {
      resize(1, A.size());
      row(0) = A;
      return *this;
    }
    assert(A.size()==ncol());
    uint m = nrow() + 1;
    uint n = ncol();
    Matrix tmp(m,n);
    for (uint i=0; i<n; ++i) {
      std::copy(col_begin(i), col_end(i), tmp.col_begin(i));
      tmp(m-1,i) = A[i];
    }
    //      swap(*this, tmp);
    swap(tmp);
    return *this;
  }

  Matrix & Matrix::cbind(const Matrix &A) {
    if (nrow()==0) {
      *this = A;
      return *this;
    }
    assert(A.nrow()==nrow());
    uint nc = nc_;
    nc_+= A.ncol();
    resize(nr_,nc_);
    std::copy(A.begin(), A.end(), col_begin(nc));
    return *this;
  }

  Matrix & Matrix::cbind(const Vector &A) {
    if (nrow()==0) {
      resize(A.size(), 1);
      col(0) = A;
      return *this;
    }
    if (A.size() != nrow()) {
      ostringstream err;
      err << "Improperly sized argument to cbind.  "
          << "The LHS matrix has dimension "
          << nrow() << " x " << ncol()
          << ".  The RHS vector has length "
          << A.size() << endl
          << "LHS = " << *this
          << "RHS = " << A << endl;
      report_error(err.str());
    }
    nc_+= 1;
    resize(nr_,nc_);
    std::copy(A.begin(), A.end(), col_begin(nc_-1));
    return *this;
  }

  double * Matrix::data() {
    return V.data();
  }

  const double * Matrix::data() const {
    return V.data();
  }

  double & Matrix::operator()(uint i, uint j) {
    assert(inrange(i,j));
    return V[INDX(i,j)];}

  const double & Matrix::operator()(uint i, uint j) const {
    assert(inrange(i,j));
    return V[INDX(i,j)];}

  double & Matrix::unchecked(uint i, uint j) {
    return V[INDX(i,j)];}

  const double & Matrix::unchecked(uint i, uint j) const {
    return V[INDX(i,j)];}


  ConstVectorView Matrix::row(uint i) const {
    return ConstVectorView (data()+i, ncol(), nrow());}

  VectorView Matrix::row(uint i) {
    return VectorView(data()+i, ncol(), nrow());}

  void Matrix::set_row(uint i, const Vector &v) {
    assert(v.size()==nc_);
    for (uint j=0; j<nc_; ++j) unchecked(i,j) = v[j];}

  void Matrix::set_row(uint i, const double *v) {
    for (uint j=0; j<nc_; ++j) unchecked(i,j) = v[j];}

  void Matrix::set_row(uint i, const double x) {
    for (uint j=0; j<nc_; ++j) unchecked(i,j) = x;}


  VectorView Matrix::col(uint j) {
    double *start = &(*col_begin(j));
    return VectorView(start, nrow(), 1);}

  const VectorView Matrix::col(uint j) const {
    double *start = const_cast<double *>(&(*col_begin(j)));
    return VectorView(start, nrow(), 1);}

  void Matrix::set_col(uint j, const Vector &v) {
    std::copy(v.begin(), v.end(), col_begin(j)); }

  void Matrix::set_col(uint j, const double *v) {
    std::copy(v, v+nr_, col_begin(j)); }

  void Matrix::set_col(uint j, double x) {
    std::fill(col_begin(j), col_end(j), x); }

  void Matrix::set_rc(uint i,  double x) {
    assert(is_square());
    for (uint k=0; k<nr_; ++k) unchecked(i,k) = unchecked(k,i) = x;}


  ConstVectorView Matrix::diag() const {
    uint m = std::min(nr_,nc_);
    ConstVectorView ans(data(), m, nrow()+1);
    return ans;
  }

  VectorView Matrix::diag() {
    uint m = std::min(nr_,nc_);
    return  VectorView(data(), m, nrow()+1);}

  VectorView Matrix::subdiag(int i) {
    if (i < 0) return superdiag(-i);
    int m = std::min(nr_,nc_);
    return VectorView(data()+i, m - i, nrow() + 1);
  }

  ConstVectorView Matrix::subdiag(int i) const {
    if (i < 0) return superdiag(-i);
    int m = std::min(nr_,nc_);
    return ConstVectorView(data()+i, m - i, nrow() + 1);
  }

  VectorView Matrix::superdiag(int i) {
    if (i < 0) return subdiag(-i);
    int m = std::min(nr_, nc_);
    return VectorView(data() + i*nr_, m - i, nrow() + 1);
  }

  ConstVectorView Matrix::superdiag(int i) const {
    if (i < 0) return subdiag(-i);
    int m = std::min(nr_, nc_);
    return ConstVectorView(data() + i*nr_, m - i, nrow() + 1);
  }

  VectorView Matrix::first_row() {
    return VectorView(data(), ncol(), nrow()); }
  ConstVectorView Matrix::first_row() const {
    return ConstVectorView(data(), ncol(), nrow()); }

  VectorView Matrix::first_col() {
    return VectorView(data(), nrow(), 1);
  }

  ConstVectorView Matrix::first_col() const {
    return ConstVectorView(data(), nrow(), 1);
  }

  VectorView Matrix::last_row() {
    uint nr = nrow();
    return VectorView(data()+nr-1, ncol(), nr);
  }
  ConstVectorView Matrix::last_row() const {
    uint nr = nrow();
    return ConstVectorView(data()+nr-1, ncol(), nr);
  }
  VectorView Matrix::last_col() {
    uint nc = ncol();
    uint nr = nrow();
    return VectorView(data()+(nc-1)*nr, nr, 1);
  }
  ConstVectorView Matrix::last_col() const {
    uint nc = ncol();
    uint nr = nrow();
    return ConstVectorView(data()+(nc-1)*nr, nr, 1);
  }


  Matrix & Matrix::set_diag(double x, bool zero_off) {
    if (zero_off) operator=(0.0);
    diag()=x;
    return *this; }

  Matrix & Matrix::set_diag(const Vector &v, bool zero_off) {
    if (zero_off) operator=(0.0);
    assert(v.size()== std::min(nr_, nc_));
    diag()=v;
    return *this; }

  dVector::iterator Matrix::col_begin(uint j) {
    return V.begin() + j*nr_; }
  dVector::iterator Matrix::col_end(uint j) {
    return V.begin() + (j+1)*nr_; }
  dVector::const_iterator Matrix::col_begin(uint j) const {
    return V.begin() + j*nr_; }
  dVector::const_iterator Matrix::col_end(uint j) const {
    return V.begin() + (j+1)*nr_; }

  dVector::iterator Matrix::begin() {
    return V.begin();}
  dVector::const_iterator Matrix::begin() const {
    return V.begin();}
  dVector::iterator Matrix::end() {
    return V.end();}
  dVector::const_iterator Matrix::end() const {
    return V.end();}

  VectorViewIterator Matrix::dbegin() {
    return VectorViewIterator(&V.front(), &V.front(), ncol()+1); }
  VectorViewIterator Matrix::dend() {  // make it right for rectangular matrices
    uint m = std::min(nr_, nc_);
    uint stride = ncol()+1;
    double * last_ = &(unchecked(m-1, m-1)) + stride;
    return VectorViewIterator(last_, &unchecked(0,0), stride);
  }

  VectorViewConstIterator Matrix::dbegin() const {
    return VectorViewConstIterator(&(V.front()), &(V.back()), ncol()+1); }
  VectorViewConstIterator Matrix::dend() const {
    return VectorViewConstIterator(
        &V.front(),
        &(V.back())+ ncol()+1,
        ncol()+1);
  }

  VectorViewIterator Matrix::row_begin(uint i) {
    double *b = data()+i;
    uint stride = nrow();
    return VectorViewIterator(b,b,stride);}

  VectorViewIterator Matrix::row_end(uint i) {
    double *b = data()+i;
    uint stride = nrow();
    double *e = b + ncol()*stride;
    return VectorViewIterator(e,b,stride);}

  VectorViewConstIterator Matrix::row_begin(uint i) const {
    const double *b = data()+i;
    uint stride = nrow();
    return VectorViewConstIterator(b,b,stride);}

  VectorViewConstIterator Matrix::row_end(uint i) const {
    const double *b = data()+i;
    uint stride = nrow();
    const double *e = b + ncol()*stride;
    return VectorViewConstIterator(e,b,stride);}


  //------------ linear algebra ------------------------

  bool Matrix::can_mult(const Matrix &B, const Matrix &ans) const {
    return nrow()==ans.nrow() && B.ncol()==ans.ncol() && ncol() == B.nrow();}

  bool Matrix::can_Tmult(const Matrix &B, const Matrix &ans) const {
    return ncol()==ans.nrow() && B.ncol()==ans.ncol() && nrow() == B.nrow();}

  bool Matrix::can_multT(const Matrix &B, const Matrix &ans) const {
    return nrow()==ans.nrow() && B.nrow()==ans.ncol() && ncol() == B.ncol();}

  Matrix & Matrix::mult(const Matrix &B, Matrix &ans, double scal) const {
    assert(can_mult(B,ans));
    uint m =ans.nrow();
    uint n=ans.ncol();
    uint k = ncol();
    dgemm(NoTrans, NoTrans, m, n, k, scal, data(),
          nrow(), B.data(), B.nrow(), 0.0, ans.data(), ans.nrow());
    return ans; }

  Matrix & Matrix::Tmult(const Matrix &B, Matrix &ans, double scal) const {
    assert(can_Tmult(B,ans));
    uint m =ans.nrow();
    uint n=ans.ncol();
    uint k =nrow();
    dgemm(Trans, NoTrans, m, n, k, scal, data(),
          nrow(), B.data(), B.nrow(), 0.0, ans.data(), ans.nrow());
    return ans; }

  Matrix & Matrix::multT(const Matrix &B, Matrix &ans, double scal) const {
    assert(can_multT(B,ans));
    uint m =ans.nrow();
    uint n=ans.ncol();
    uint k =ncol();
    dgemm(NoTrans, Trans, m, n, k, scal, data(),
          nrow(), B.data(), B.nrow(), 0.0, ans.data(), ans.nrow());
    return ans; }


  //------- support for spd matrices --------

  Matrix & Matrix::mult(const SpdMatrix &S, Matrix & ans, double scal) const {
    assert(can_mult(S,ans));
    uint m = nrow();
    uint n = S.ncol();
    dsymm(Right, Upper, m, n, scal,
          S.data(), S.nrow(), data(), nrow(),
          0.0, ans.data(), ans.nrow());
    return ans; }


  Matrix & Matrix::Tmult(const SpdMatrix &S, Matrix & ans, double scal) const {
    return Tmult( static_cast<const Matrix &>(S), ans, scal);}

  Matrix & Matrix::multT(const SpdMatrix &S, Matrix & ans, double scal) const {
    return mult(S,ans, scal);}

  //----------- diagonal matrices ----------

  Matrix & Matrix::mult(const DiagonalMatrix &d,
                        Matrix &ans,
                        double scal) const {
    assert(ncol() == d.nrow());
    ans = *this;
    const Vector &diagonal_values(d.diag());
    for (uint i = 0; i < ncol(); ++i) {
      ans.col(i) *= diagonal_values[i] * scal;
    }
    return ans;
  }

  // The diagonal matrix scales the columns of this->transpose(),
  // which is the same as scaling the rows of *this.
  Matrix & Matrix::Tmult(const DiagonalMatrix &d,
                         Matrix &ans,
                         double scal) const {
    assert(nrow() == d.ncol());
    ans.resize(ncol(), nrow());
    const Vector &diagonal_values(d.diag());
    for (int i = 0; i < nrow(); ++i) {
      ans.col(i) = row(i) * (diagonal_values[i] * scal);
    }
    return ans;
  }

  Matrix & Matrix::multT(const DiagonalMatrix &d,
                         Matrix & ans,
                         double scal) const {
    return mult(d, ans, scal);
  }

  template<class MATRIX, class VECTOR_IN, class VECTOR_OUT>
  VECTOR_OUT & matrix_multiply_impl(const MATRIX &m, const VECTOR_IN &v,
                                    VECTOR_OUT &ans, bool transpose) {
    dgemv(transpose ? Trans : NoTrans,
          m.nrow(),
          m.ncol(),
          1.0,
          m.data(),
          m.nrow(),
          v.data(),
          v.stride(),
          0.0,
          ans.data(),
          ans.stride());
    return ans;
  }

  //--------- Vector support
  Vector & Matrix::mult(const Vector &v, Vector &ans, double scal) const {
    assert(ncol()==v.size() && nrow()==ans.size());
    dgemv(NoTrans,nrow(),ncol(), scal, data(), nrow(),
          v.data(), v.stride(), 0.0, ans.data(), ans.stride());
    return ans;
  }

  Vector & Matrix::Tmult(const Vector &v, Vector &ans, double scal) const {
    assert(nrow() == v.size()
           && ncol() == ans.size());
    dgemv(Trans,nrow(),ncol(), scal, data(), nrow(),
          v.data(), v.stride(), 0.0, ans.data(), ans.stride());
    return ans;
  }


  //---------- non-virtual multiplication funcitons -----------

  Matrix Matrix::mult(const Matrix &B) const {
    Matrix ans(nrow(), B.ncol());
    return mult(B, ans);}

  Matrix Matrix::Tmult(const Matrix &B) const {
    Matrix ans(ncol(), B.ncol());
    return Tmult(B, ans);}

  Matrix Matrix::multT(const Matrix &B) const {
    Matrix ans(nrow(), B.nrow());
    return multT(B, ans);}

  Vector Matrix::mult(const Vector &v) const {
    Vector ans(nrow());
    return mult(v,ans);}

  Vector Matrix::Tmult(const Vector &v) const {
    Vector ans(ncol());
    return Tmult(v,ans);}


  Matrix Matrix::Id() const {
    Matrix ans(nr_,nc_);
    ans.set_diag(1.0);
    return ans;}

  Matrix Matrix::t() const {
    Matrix ans(nc_,nr_);
    for (uint i=0; i<nr_; ++i) {
      for (uint j=0; j<nc_; ++j) {
        ans(j,i) = (*this)(i,j);}}
    return ans; }

  Matrix & Matrix::transpose_inplace_square() {
    assert(is_square());
    double *d(data());
    for (uint i = 0; i < nr_; ++i) {
      for (uint j = 0; j < i; ++j) {
        std::swap(d[INDX(i,j)], d[INDX(j,i)]);
      }
    }
    return *this;
  }

  SpdMatrix Matrix::inner() const {  // eventually return type will be SpdMatrix
    SpdMatrix ans(nc_, 0.0);
    dsyrk(Upper, Trans, nc_, nr_, 1.0, data(),
          nr_, 0.0, ans.data(), nc_);
    ans.reflect();
    return ans; }

  SpdMatrix Matrix::outer() const {
    SpdMatrix ans(nr_);
    //
    // C <- alpha * A A^T + beta * C
    // where C is NxN and A is NxK.
    dsyrk(Upper,
          NoTrans,
          nr_,            // N
          nc_,            // K
          1.0,            // alpha
          data(),         // *A
          nr_,            // LDA
          0.0,            // beta
          ans.data(),     // *C
          nr_);           // LDC
    ans.reflect();
    return ans;
  }

  Matrix Matrix::inv() const {
    assert(is_square());
    return solve(Id()); }

  Matrix Matrix::solve(const Matrix &m) const {
    //return this^{-1}*m
    assert(this->is_square());
    assert(m.nrow()==ncol());
    Matrix ans(m);
    Matrix A(*this);
    int n = nrow();
    int nrhs  = ans.ncol();
    int lda = nrow();
    std::vector<int> ipiv(n);
    int ldb = ans.nrow();
    int info=0;
    dgesv_(&n, &nrhs, A.data(), &lda, &(ipiv[0]),
           ans.data(), &ldb, &info);
    return ans;
  }

  Vector Matrix::solve(const Vector &v) const {
    assert(this->is_square());
    Vector ans(v);
    int n = nrow();
    int nrhs = 1;
    int lda= nrow();
    std::vector<int> ipiv(n);
    int ldb  = v.size();
    int info;
    Matrix A(*this);
    dgesv_(&n, &nrhs, A.data(), &lda, &(ipiv[0]),
           ans.data(), &ldb, &info);
    return ans;
  }

  double Matrix::trace() const {
    return accumulate(dbegin(), dend(), 0.0);}

  double Matrix::det() const {
    assert(is_square());
    QR qr(*this);
    return qr.det();
  }

  Vector Matrix::singular_values() const {
    int m =nr_;
    int n =nc_;
    int one=1;
    int info=0;
    int lwork(-1);
    Vector work(1);
    std::vector<int> iwork(8*std::min(m,n));
    Vector values(std::min(m,n));
    double *u=0;
    double *vt=0;

    Matrix tmp(*this);
    dgesdd_("N", &m, &n, tmp.data(), &m, values.data(),
            u, &one, vt, &one, work.data(), &lwork, &(iwork[0]), &info);
    if (info!=0) report_error(
           "error computing optimal work size  with dgesdd_");
    lwork = static_cast<int>(work[0]);
    work.resize(lwork);
    dgesdd_("N", &m, &n, tmp.data(), &m, values.data(),
            u, &one, vt, &one, work.data(), &lwork, &(iwork[0]), &info);
    if (info!=0) report_error(
           "error computing singular values with dgesdd_");
    return values;
  }

  struct greater{
    bool operator()(double a, double b) const { return a>b;}
  };

  uint Matrix::rank(double prop) const {
    Vector s = singular_values();
    double bound =s[0]*prop;
    std::vector<double>::iterator pos =
      lower_bound(s.begin(), s.end(), bound, greater());
    uint k = distance(pos, s.end());
    return s.size()-k;
  }

  Vector Matrix::real_evals() const {
    assert(this->is_square());
    int n = nrow();
    int one = 1;
    Matrix A(*this);
    Vector real_ev(n);
    Vector imag_ev(n);
    Vector work(1);
    int lwork(-1);
    int info=0;
    dgeev_("N", "N", &n, A.data(), &n, real_ev.data(), imag_ev.data(),
           0, &one, 0, &one, work.data(), &lwork, &info);
    lwork = static_cast<uint>(work[0]);
    work.resize(lwork);
    dgeev_("N", "N", &n, A.data(), &n, real_ev.data(), imag_ev.data(),
           0, &one, 0, &one, work.data(), &lwork, &info);
    dVector tmp;
    tmp.reserve(n);
    for (int i=0; i<n; ++i) {
      if (i==n-1) {
        tmp.push_back(real_ev[i]);
      } else if (real_ev[i] == real_ev[i+1] && imag_ev[i] == -imag_ev[i+1]) {
        ++i;
      } else {
        tmp.push_back(real_ev[i]);
      }
    }
    Vector ans(tmp.begin(), tmp.end());
    return ans;
  }

  Matrix & Matrix::add_outer(const Vector &x, const Vector &y, double w) {
    assert(nrow()==x.size() && ncol() == y.size());
    dger(nrow(), ncol(), w, x.data(), x.stride(),
         y.data(), y.stride(), data(), nrow());
    return *this;
  }

  Matrix & Matrix::add_outer(const Vector &x, const VectorView &y, double w) {
    assert(nrow()==x.size() && ncol() == y.size());
    dger(nrow(), ncol(), w, x.data(), x.stride(),
         y.data(), y.stride(), data(), nrow());
    return *this;
  }

  Matrix & Matrix::add_outer(const VectorView &x, const Vector &y, double w) {
    assert(nrow()==x.size() && ncol() == y.size());
    dger(nrow(), ncol(), w, x.data(), x.stride(),
         y.data(), y.stride(), data(), nrow());
    return *this;
  }
  Matrix & Matrix::add_outer(const VectorView &x,
                             const VectorView &y,
                             double w) {
    assert(nrow()==x.size() && ncol() == y.size());
    dger(nrow(), ncol(), w, x.data(), x.stride(),
         y.data(), y.stride(), data(), nrow());
    return *this;
  }
  Matrix & Matrix::add_outer(const ConstVectorView &x,
                             const Vector &y,
                             double w) {
    assert(nrow()==x.size() && ncol() == y.size());
    dger(nrow(), ncol(), w, x.data(), x.stride(),
         y.data(), y.stride(), data(), nrow());
    return *this;
  }
  Matrix & Matrix::add_outer(const Vector &x,
                             const ConstVectorView &y,
                             double w) {
    assert(nrow()==x.size() && ncol() == y.size());
    dger(nrow(), ncol(), w, x.data(), x.stride(),
         y.data(), y.stride(), data(), nrow());
    return *this;
  }
  Matrix & Matrix::add_outer(const ConstVectorView &x,
                             const VectorView &y,
                             double w) {
    assert(nrow()==x.size() && ncol() == y.size());
    dger(nrow(), ncol(), w, x.data(), x.stride(),
         y.data(), y.stride(), data(), nrow());
    return *this;
  }
  Matrix & Matrix::add_outer(const VectorView &x,
                             const ConstVectorView &y,
                             double w) {
    assert(nrow()==x.size() && ncol() == y.size());
    dger(nrow(), ncol(), w, x.data(), x.stride(),
         y.data(), y.stride(), data(), nrow());
    return *this;
  }
  Matrix & Matrix::add_outer(const ConstVectorView &x,
                             const ConstVectorView &y,
                             double w) {
    assert(nrow()==x.size() && ncol() == y.size());
    dger(nrow(), ncol(), w, x.data(), x.stride(),
         y.data(), y.stride(), data(), nrow());
    return *this;
  }

  Matrix & Matrix::operator+=(double x) {
    uint n = size();
    double *d(data());
    for (uint i=0; i<n; ++i) d[i] += x;
    return *this; }

  Matrix & Matrix::operator-=(double x) {
    return this->operator+=(-x);}

  Matrix & Matrix::exp() {
    uint n = size();
    double *d(data());
    for (uint i=0; i<n; ++i) d[i] = std::exp(d[i]);
    return *this;
  }

  Matrix & Matrix::log() {
    uint n = size();
    double *d(data());
    for (uint i=0; i<n; ++i) d[i] = std::log(d[i]);
    return *this;
  }

  Matrix & Matrix::operator*=(double x) {
    int n = size();
    double *d(data());
    dscal(n,x,d,1);
    return *this; }

  Matrix & Matrix::operator/=(double x) {
    return this->operator*=(1.0/x);}

  Matrix & Matrix::operator+= (const Matrix &m) {
    if (!same_dim(m)) {
      ostringstream err;
      err << "Matrix::operator+= wrong dimension:  "
          << "LHS[" <<nrow() << "," << ncol() << "]   RHS["
          << m.nrow() << "," << m.ncol() << ")";
      report_error(err.str());
    }
    V+= m.V;
    return *this;
  }

  Matrix & Matrix::operator+=(const SubMatrix &m) {
    SubMatrix lhs(*this);
    lhs += m;
    return *this;
  }

  Matrix & Matrix::operator+=(const ConstSubMatrix &m) {
    SubMatrix lhs(*this);
    lhs += m;
    return *this;
  }

  Matrix & Matrix::operator-= (const Matrix &m) {
    if (!same_dim(m)) {
      ostringstream err;
      err << "Matrix::operator-= wrong dimension:  "
          << "LHS[" <<nrow() << "," << ncol() << "]   RHS["
          << m.nrow() << "," << m.ncol() << ")";
      report_error(err.str());
    }
    V-= m.V;
    return *this;
  }

  Matrix & Matrix::operator-=(const SubMatrix &m) {
    SubMatrix lhs(*this);
    lhs -= m;
    return *this;
  }

  Matrix & Matrix::operator-=(const ConstSubMatrix &m) {
    SubMatrix lhs(*this);
    lhs -= m;
    return *this;
  }

  ostream & Matrix::write(ostream &out, bool nl) const {
    for (uint i=0; i<nr_; ++i) {
      for (uint j=0; j<nc_; ++j) {
        out << unchecked(i,j) << " ";}}
    if (nl) out << std::endl;
    return out; }

  istream & Matrix::read(istream &in) {
    for (uint i=0; i<nr_; ++i) {
      for (uint j = 0; j<nc_; ++j) {
        in >> unchecked(i,j); }}
    return in; }

  //========== non-member functions =============

  VectorView diag(Matrix &m) {return m.diag();}
  ConstVectorView diag(const Matrix &m) {return m.diag();}

  Matrix diag(const Vector &v) {
    uint n = v.size();
    Matrix ans(n,n,0.0);
    ans.set_diag(v);
    return ans;
  }

  Matrix diag(const VectorView &v) {
    uint n = v.size();
    Matrix ans(n,n,0.0);
    ans.set_diag(v);
    return ans;
  }

  ostream & Matrix::display(ostream & out, int precision) const {
    out << std::setprecision(precision);
    for (uint i = 0; i < nrow(); ++i) {
      for (uint j = 0; j < ncol(); ++j)
        out << std::setw(8) << unchecked(i,j) << " ";
      out << endl;}
    return out; }

  ostream & operator<<(ostream & out, const Matrix &x) {
    return x.display(out, 5);
  }

  void print(const Matrix &m) {
    cout << m << endl;
  }

  istream & operator>>(istream &in, Matrix &m) {
    // reads until a blank line is found or the end of a line

    typedef std::vector<string> svec;

    svec lines;
    while(in) {
      string line;
      getline(in, line);
      if (is_all_white(line)) break;
      lines.push_back(line);
    }
    uint nrows = lines.size();
    StringSplitter split;
    svec splitline(split(lines[0]));
    uint ncols = splitline.size();

    if (m.nrow()!=nrows || m.ncol()!=ncols) {
      m = Matrix(nrows, ncols);}

    for (uint j=0; j<ncols; ++j) {
      istringstream sin(splitline[j]);
      sin >> m(0,j);}

    for (uint i=1; i<nrows; ++i) {
      splitline = split(lines[i]);
      assert(splitline.size()==ncols);
      for (uint j=0; j<ncols; ++j) {
        istringstream sin(splitline[j]);
        sin >> m(i,j);}}
    return in;
  }

  Matrix operator-(const double y, const Matrix &x) {
    Matrix ans = -x;
    ans+=y;
    return ans; }

  Matrix operator/(const double y, const Matrix &x) {
    Matrix ans = x;
    transform(ans.begin(), ans.end(), ans.begin(),
        bind1st(std::divides<double>(), y) );
    return ans;
  }

  inline double mul(double x, double y) {return x*y;}

  Matrix el_mult(const Matrix &A, const Matrix &B) {
    assert(A.same_dim(B));
    Matrix ans(A.nrow(), A.ncol());
    transform(A.begin(), A.end(), B.begin(), ans.begin(), mul);
    return ans;
  }

  double el_mult_sum(const Matrix &A, const Matrix &B) {
    assert(can_add(A,B));
    uint n = A.size();
    return ddot(n, A.data(), 1, B.data(), 1);
  }

  double Matrix::sum() const {
    return accumulate(V.begin(), V.end(), 0.0);}

  double Matrix::abs_norm() const {
    return dasum(size(), data(), 1);}

  double Matrix::sumsq() const {
    return ddot(size(), data(), 1, data(), 1); }

  double Matrix::prod() const {
    return accumulate(V.begin(), V.end(), 1.0, mul);}

  double Matrix::max() const { return *std::max_element(begin(), end()); }

  double Matrix::min() const {
    return *min_element(begin(), end()); }

  double Matrix::max_abs() const {
    int n = size();
    const double *d = data();
    double max = -1;
    for (int i = 0; i < n; ++i) {
      double fd = fabs(d[i]);
      if (fd > max) {
        max = fd;
      }
    }
    return max;
  }

  LabeledMatrix::LabeledMatrix(
      const Matrix &m,
      const std::vector<std::string> &row_names,
      const std::vector<std::string> &col_names)
      : Matrix(m),
        row_names_(row_names),
        col_names_(col_names)
      {
        if (!row_names.empty() &&
            row_names.size() != m.nrow()) {
          report_error("row_names was the wrong size in "
                       "LabeledMatrix constructor");
        }
        if (!col_names.empty() &&
            col_names.size() != m.ncol()) {
          report_error("col_names was the wrong size in "
                       "LabeledMatrix constructor");
        }
      }

  ostream & LabeledMatrix::display(ostream &out, int precision) const {
    int max_row_label = 0;
    bool have_row_names = !row_names_.empty();
    if (have_row_names) {
      for (int i = 0; i < row_names_.size(); ++i) {
        max_row_label = std::max<int>(max_row_label, row_names_[i].size());
      }
      out << std::setw(max_row_label) << " "
          << " ";
    }

    bool have_col_names = !col_names_.empty();
    if (have_col_names) {
      for (int i = 0; i < col_names_.size(); ++i) {
        int col_width = std::max<int>(col_names_[i].size(), 8);
        out << std::setw(col_width) << col_names_[i] << " ";
      }
      out << endl;
    }

    for (int i = 0; i < nrow(); ++i) {
      if (have_row_names) {
        out << std::setw(max_row_label) << std::left << row_names_[i]
            << std::right << " ";
      }
      for (int j = 0; j < ncol(); ++j) {
        int col_width =
            have_col_names ? std::max<int>(col_names_[j].size(), 8) : 8;
        out << std::setw(col_width) << unchecked(i, j) << " ";
      }
      out << endl;
    }
    return out;
  }

  Matrix LabeledMatrix::drop_labels() const {
    return Matrix(*this);
  }

  ArbitraryOffsetMatrix::ArbitraryOffsetMatrix(
      int first_row, uint number_of_rows,
      int first_column, uint number_of_columns,
      double initial_value)
      : data_(number_of_rows, number_of_columns, initial_value),
        row_offset_(first_row),
        column_offset_(first_column)
  {}

  Matrix log(const Matrix &x) {
    Matrix ans(x);
    std::transform(ans.begin(), ans.end(), ans.begin(),
             pointer_to_unary_function<double,double>(std::log));
    return ans; }

  Matrix exp(const Matrix &x) {
    Matrix ans(x);
    std::transform(ans.begin(), ans.end(), ans.begin(),
             pointer_to_unary_function<double,double>(std::exp));
    return ans; }

  Matrix matmult(const Matrix &A, const Matrix &B) {
    return A.mult(B);}

  Matrix matTmult(const Matrix &A, const Matrix &B) {
    return A.Tmult(B);}

  Matrix matmultT(const Matrix &A, const Matrix &B) {
    return A.multT(B);}

  Vector matmult(const Matrix &A, const Vector &v) {
    return A.mult(v);}

  Vector matmult(const Vector &v, const Matrix &A) {
    return v.mult(A);}

  Vector operator*(const Vector &v, const Matrix &m) {
    return v.mult(m);}

  Vector operator*(const Matrix &m, const Vector &v) {
    return m.mult(v);}

  // t(v) %*% m   =   t(m) %*% v
  Vector operator*(const VectorView &v, const Matrix &m) {
    Vector ans(m.ncol());
    assert(v.size() == m.nrow());
    matrix_multiply_impl(m,v,ans, true);
    return ans;
  }

  Vector operator*(const Matrix &m, const VectorView &v) {
    Vector ans(m.nrow());
    assert(v.size() ==  m.ncol());
    matrix_multiply_impl(m,v,ans, false);
    return ans;
  }

  // t(v) %*% m   =   t(m) %*% v
  Vector operator*(const ConstVectorView &v, const Matrix &m) {
    Vector ans(m.ncol());
    assert(v.size() == m.nrow());
    matrix_multiply_impl(m,v,ans, true);
    return ans;
  }

  Vector operator*(const Matrix &m, const ConstVectorView &v) {
    Vector ans(m.nrow());
    assert(v.size() ==  m.ncol());
    matrix_multiply_impl(m,v,ans, false);
    return ans;
  }

  Matrix operator*(const Matrix &A, const Matrix &B) {
    return A.mult(B);}


  //----------- changing Matrix size and layout -------------

  Matrix rbind(const Matrix &m1, const Matrix &m2) {
    Matrix tmp(m1);
    return tmp.rbind(m2); }

  Matrix rbind(const Vector &v, const Matrix &m) {
    Matrix tmp(v.begin(), v.end(), 1u, v.size());
    return tmp.rbind(m); }

  Matrix rbind(const Matrix &m, const Vector &v) {
    Matrix ans(m);
    return ans.rbind(v);}

  Matrix rbind(const Vector &v1, const Vector &v2) {
    Matrix tmp(v1.begin(), v1.end(), 1, v1.size());
    return tmp.rbind(v2); }

  Matrix rbind(double x, const Matrix &m) {
    Vector tmp(m.ncol(), x);
    return rbind(tmp, m); }

  Matrix rbind(const Matrix &m, double x) {
    Vector tmp(m.ncol(), x);
    return rbind(m,tmp);
  }

  Matrix cbind(const Matrix &m1, const Matrix &m2) {
    Matrix ans(m1);
    return ans.cbind(m2); }

  Matrix cbind(const Matrix &m, const Vector &v) {
    Matrix ans(m);
    return ans.cbind(v); }

  Matrix cbind(const Vector &v, const Matrix &m) {
    Matrix ans(v.begin(), v.end(), v.size(), 1);
    return ans.cbind(m);}

  Matrix cbind(const Vector &v1, const Vector &v2) {
    Matrix ans(v1.begin(), v1.end(), v1.size(), 1);
    return ans.cbind(v2); }

  Matrix cbind(const Matrix &m, double x) {
    Vector v(m.nrow(), x);
    Matrix ans(m);
    return ans.cbind(v); }

  Matrix cbind(double x, const Matrix &m) {
    Vector v(m.nrow(), x);
    return cbind(v,m); }

  Matrix drop_col(const Matrix &m, uint j) {
    uint nr = m.nrow();
    uint nc = m.ncol()-1;
    Matrix ans(nr,nc);
    for (uint i=0; i<j; ++i) ans.col(i) = m.col(i);
    for (uint i=j+1; i<nc; ++i) ans.col(i-1) = m.col(i);
    return ans;
  }

  Matrix drop_cols(const Matrix &m, std::vector<uint> indx) {
    std::sort(indx.begin(), indx.end(), std::greater<uint>() );
    uint nr = m.nrow();
    uint nc = m.ncol()-indx.size();
    assert(m.ncol() > indx.size());
    Matrix ans(nr,nc);
    uint I=0;
    for (uint i=0; i<m.ncol(); ++i) {
      if (i== indx.back()) {
        indx.pop_back();
      } else {
        ans.col(I++) = m.col(i);
      }
    }
    return ans;
  }


  Matrix permute_Matrix(const Matrix &Q, const std::vector<uint> &perm) {
    assert(Q.is_square());
    Matrix ans = Q;
    uint n = Q.nrow();
    for (uint i = 0; i<n; ++i) {
      for (uint j = 0; j<n; ++j) {
        ans(i,j) = Q(perm[i], perm[j]);}}
    return ans; }

  double traceAB(const Matrix &A, const Matrix &B) {
  // tr(AB) = sum_i  A.row(i) .dot. B.col(i)
    uint n = A.nrow();
    double ans=0;
    for (uint i = 0; i<n; ++i) ans+= A.row(i).dot(B.col(i));
    return ans;
  }

  double traceAtB(const Matrix &A, const Matrix &B) {
    uint n = A.size();
    assert(n==B.size());
    return ddot(n,A.data(),1,B.data(),1);
  }


  Matrix unpartition(double a, const Vector &v, const Matrix &B) {
    // a v
    // v B
    assert(B.is_square() && B.nrow()==v.size());
    Matrix ans = cbind(v,B);
    Vector tmp = concat(a,v);
    ans= rbind(tmp, ans);
    return ans;
  }

   Matrix unpartition(const Matrix &B, const Vector &v, double a) {
     // B v
     // v a
     assert(B.is_square() && B.nrow()==v.size());
     Matrix ans(B);
     ans.cbind(v);          // ans = B v
     Vector tmp(v);
     ans.rbind(tmp.push_back(a));
     return ans;
   }

  Matrix unpartition(const Matrix &A, const Matrix &Rect, const Matrix &B) {
    assert(A.is_square() && B.is_square() && (A.nrow()==Rect.nrow()));
    Matrix ans = A;
    ans.cbind(Rect);

    Matrix tmp(Rect.t());
    tmp.cbind(B);
    return ans.rbind(tmp);
  }

  Matrix block_diagonal(const Matrix &A, const Matrix &B) {
    assert(A.is_square() && B.is_square());
    uint n1 = A.nrow();
    uint n2 = B.nrow();

    uint n = n1 + n2;
    Matrix ans(n,n,0.0);
    for (uint i=0; i<n1; ++i) {
      std::copy(A.col_begin(i), A.col_end(i),
          ans.col_begin(i));
    }
    for (uint i=n1; i<n; ++i) {
      std::copy(B.col_begin(i-n1), B.col_end(i-n1),
          ans.col_begin(i)+n1);
    }
    return ans;
  }

  // ------------- lower triangular functions --------------

  Matrix LTmult(const Matrix &L, const Matrix &B) {
    assert(L.is_square() && (L.nrow() == B.nrow()));
    Matrix ans(B);

    dtrmm(Left,
          Lower,
          Trans,
          NonUnit,
          L.nrow(),
          L.ncol(),
          1.0,
          L.data(),
          L.nrow(),
          ans.data(),
          ans.nrow());
    return ans;
  }

  Vector Lmult(const Matrix &L, const Vector &y) {
    assert(L.is_square() && L.nrow()==y.size());
    Vector ans(y);
    dtrmv(Lower, NoTrans, NonUnit, L.nrow(), L.data(), L.nrow(),
          ans.data(), 1);
    return ans; }

  Vector& Lsolve_inplace(const Matrix &L, Vector &b) {
    assert(L.is_square() && L.nrow()==b.size());
    dtrsv(Lower, NoTrans, NonUnit, L.nrow(), L.data(), L.nrow(),
          b.data(), 1);
    return b; }

  Vector & LTsolve_inplace(const Matrix &L, Vector &b) {
    assert(L.is_square() && L.nrow()==b.size());
    dtrsv(Lower, Trans, NonUnit, L.nrow(), L.data(), L.nrow(),
          b.data(), 1);
    return b; }


  Vector Lsolve(const Matrix &L, const Vector &b) {
    Vector ans(b);
    return Lsolve_inplace(L,ans);}

  Matrix &Lsolve_inplace(const Matrix &L, Matrix &B) {
    assert(L.is_square() && L.ncol()==B.nrow());
    dtrsm(Left, Lower, NoTrans, NonUnit, B.nrow(), B.ncol(), 1.0,
          L.data(), L.nrow(), B.data(), B.nrow());
    return B; }

  Matrix Lsolve(const Matrix &L, const Matrix &B) {
    Matrix ans = B;
    return Lsolve_inplace(L,ans);}

  Matrix Linv(const Matrix &L) {
    Matrix ans(L.Id());
    return Lsolve_inplace(L, ans); }

  // ------------- upper triangular functions --------------

  Vector Umult(const Matrix &U, const Vector &y) {
    assert(U.is_square() && U.nrow()==y.size());
    Vector ans(y);
    dtrmv(Upper, NoTrans, NonUnit, U.nrow(), U.data(), U.nrow(),
          ans.data(), 1);
    return ans; }

  Matrix Umult(const Matrix &U, const Matrix &B) {
    assert(U.is_square() && U.ncol()==B.nrow());
    Matrix ans(B);
    dtrmm(Left, Upper, NoTrans, NonUnit, B.nrow(), B.ncol(), 1.0, U.data(),
          U.nrow(), ans.data(), ans.nrow());
    return ans;
  }

  Vector& Usolve_inplace(const Matrix &U, Vector &b) {
    assert(U.is_square() && U.nrow()==b.size());
    dtrsv(Upper, NoTrans, NonUnit,
          U.nrow(), U.data(), U.nrow(), b.data(), 1);
    return b; }

  Vector Usolve(const Matrix &U, const Vector &b) {
    Vector ans(b);
    return Usolve_inplace(U,ans);}

  Matrix &Usolve_inplace(const Matrix &U, Matrix &B) {
    assert(U.is_square() && U.ncol()==B.nrow());
    dtrsm(Left, Upper, NoTrans, NonUnit, B.nrow(), B.ncol(), 1.0,
          U.data(), U.nrow(), B.data(), B.nrow());
    return B; }

  Matrix Usolve(const Matrix &U, const Matrix &B) {
    Matrix ans = B;
    return Usolve_inplace(U,ans);}

  Matrix Uinv(const Matrix &U) {
    Matrix ans(U.Id());
    return Usolve_inplace(U, ans); }


}// namespace BOOM
