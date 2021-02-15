// Copyright 2018 Google LLC. All Rights Reserved.
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
#ifndef BOOM_NEWLA_MATRIX_HPP
#define BOOM_NEWLA_MATRIX_HPP
#include <iosfwd>
#include <vector>
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

namespace BOOM {
  class Vector;
  class VectorView;
  class SpdMatrix;
  class DiagonalMatrix;
  class SubMatrix;
  class ConstSubMatrix;
  class Matrix {
   public:
    typedef std::vector<double> dVector;

    Matrix();
    Matrix(const Matrix &rhs) = default;
    Matrix(Matrix &&rhs) = default;

    Matrix(uint nr, uint nc, double x = 0.0);
    Matrix(uint nr, uint nc, const double *m, bool byrow = false);
    Matrix(uint nr, uint nc, const ConstVectorView &v, bool byrow = false);

    // Populate a matrix by a space-separated string, with rows separated by a
    // row delimiter.
    //
    // Example:
    // Matrix A("1 2 | 3 4") gives:
    //   1 2
    //   3 4
    explicit Matrix(const std::string &s, const std::string &row_delim = "|");

    // Build a matrix by stacking rows or columns.
    // Args:
    //   rows_or_cols: The rows or columns to be joined.  All elements must be
    //     the same size.
    //   rows: If true, treat the first argument as a collection of rows.  If
    //     false treat it as a collection of columns.
    explicit Matrix(const std::vector<Vector> &rows_or_cols, bool rows=true);

    template <class FwdIt>
    Matrix(FwdIt Beg, FwdIt End, uint nr, uint nc);

    explicit Matrix(const SubMatrix &rhs);
    explicit Matrix(const ConstSubMatrix &rhs);

    Matrix &operator=(const Matrix &rhs) = default;
    Matrix &operator=(Matrix &&rhs) = default;

    Matrix &operator=(const SubMatrix &);
    Matrix &operator=(const ConstSubMatrix &);
    Matrix &operator=(const double &);

    bool operator==(const Matrix &) const;

    template <class FwdIt>
    FwdIt assign(FwdIt begin, FwdIt end);

    void swap(Matrix &rhs);  // efficient.. swaps pointers and size info

    // Fill matrix entries with U(0,1) random variables.
    // Returns *this.
    virtual Matrix &randomize();
    virtual ~Matrix();

    // Returns true if empty, or if std::isfinite returns 'true' on
    // all elements.  Returns false otherwise.
    bool all_finite() const;

    // ----------- size and shape info ------------------
    uint size() const;  // number of elements in the Matrix
    uint nrow() const;
    uint ncol() const;

    // The largest absolute discrepancy between elements (i, j) and (j, i),
    // relative to the average absolute magnitude of the elements in the matrix.
    // The distance is taken to be zero if all elements are zero, and infinity
    // if the matrix is not square.
    double distance_from_symmetry() const;
    bool is_sym(double tol = 1.0e-4) const;
    bool same_dim(const Matrix &A) const;
    bool is_square() const;
    bool is_pos_def() const;

    //---- change size and shape  -----
    Matrix &resize(uint nr, uint nc);  // invalidates elements if nr changes
    Matrix &rbind(const Matrix &m);
    Matrix &rbind(const Vector &v);
    Matrix &cbind(const Matrix &m);
    Matrix &cbind(const Vector &v);

    double *data();  // for sending data to LAPACK, etc
    const double *data() const;

    //-------- subscripting, range checking can be turned off
    //-------- by defining the macro NDEBUG
    double &operator()(uint r, uint c);
    const double &operator()(uint r, uint c) const;
    double &unchecked(uint r, uint c);
    const double &unchecked(uint r, uint c) const;

    // -------- row and column operations ----------
    VectorView row(uint i);
    ConstVectorView row(uint i) const;
    void set_row(uint i, const Vector &v);
    void set_row(uint i, const double *v);
    void set_row(uint i, double x);

    VectorView col(uint j);
    const VectorView col(uint j) const;
    void set_col(uint j, const Vector &v);
    void set_col(uint j, const double *v);
    void set_col(uint j, double x);

    void set_rc(uint i, double x);  // sets row and column i to x

    VectorView diag();
    ConstVectorView diag() const;
    VectorView subdiag(int i);
    ConstVectorView subdiag(int i) const;
    VectorView superdiag(int i);
    ConstVectorView superdiag(int i) const;
    VectorView first_row();
    ConstVectorView first_row() const;
    VectorView last_row();
    ConstVectorView last_row() const;
    VectorView first_col();
    ConstVectorView first_col() const;
    VectorView last_col();
    ConstVectorView last_col() const;

    Matrix &set_diag(double x, bool zero_offdiag = true);
    Matrix &set_diag(const Vector &v, bool zero_offdiag = true);

    //------  STL iteration -------------
    dVector::iterator begin();
    dVector::iterator end();
    dVector::const_iterator begin() const;
    dVector::const_iterator end() const;

    dVector::iterator col_begin(uint i);
    dVector::iterator col_end(uint i);
    dVector::const_iterator col_begin(uint i) const;
    dVector::const_iterator col_end(uint i) const;

    VectorViewIterator dbegin();
    VectorViewIterator dend();
    VectorViewConstIterator dbegin() const;
    VectorViewConstIterator dend() const;

    VectorViewIterator row_begin(uint i);
    VectorViewIterator row_end(uint i);
    VectorViewConstIterator row_begin(uint i) const;
    VectorViewConstIterator row_end(uint i) const;

    //------ linear algebra -----------------

    bool can_mult(const Matrix &B, const Matrix &Ans) const;
    bool can_Tmult(const Matrix &B, const Matrix &Ans) const;
    bool can_multT(const Matrix &B, const Matrix &Ans) const;

    // scal * this * B
    virtual Matrix &mult(const Matrix &B, Matrix &ans, double scal = 1.0) const;
    // scal *this^T * B
    virtual Matrix &Tmult(const Matrix &B, Matrix &ans,
                          double scal = 1.0) const;
    // scal * this * B^T
    virtual Matrix &multT(const Matrix &B, Matrix &ans,
                          double scal = 1.0) const;

    virtual Matrix &mult(const SpdMatrix &S, Matrix &ans,
                         double scal = 1.0) const;
    virtual Matrix &Tmult(const SpdMatrix &S, Matrix &ans,
                          double scal = 1.0) const;
    virtual Matrix &multT(const SpdMatrix &S, Matrix &ans,
                          double scal = 1.0) const;
    // no BLAS support for this^T * S
    // virtual Matrix & Tmult(const SpdMatrix &S, Matrix & ans) const;

    // this * B
    virtual Matrix &mult(const DiagonalMatrix &B, Matrix &ans,
                         double scal = 1.0) const;
    // this^T * B
    virtual Matrix &Tmult(const DiagonalMatrix &B, Matrix &ans,
                          double scal = 1.0) const;
    Matrix Tmult(const DiagonalMatrix &B) const;
    // this * B^T
    virtual Matrix &multT(const DiagonalMatrix &B, Matrix &ans,
                          double scal = 1.0) const;

    // this * v
    virtual Vector &mult(const Vector &v, Vector &ans, double scal = 1.0) const;
    // this^T * v
    virtual Vector &Tmult(const Vector &v, Vector &ans,
                          double scal = 1.0) const;

    // the following functions are non-virtual, but behave virtually
    // because they call the virtual functions listed above
    Matrix mult(const Matrix &B) const;   // this * B
    Matrix Tmult(const Matrix &B) const;  // this^T * B
    Matrix multT(const Matrix &B) const;  // this * B^T
    Vector mult(const Vector &v) const;
    Vector Tmult(const Vector &v) const;

    Matrix Id() const;
    Matrix transpose() const;            // SpdMatrix and DiagonalMatrix
    Matrix &transpose_inplace_square();  // asserts (is_square())
    Matrix inv() const;
    virtual SpdMatrix inner() const;  // X^T * X
    // Return this->transpose * diag(weights) * this
    SpdMatrix inner(const ConstVectorView &weights) const;
    SpdMatrix outer() const;          // X * X^T

    virtual Matrix solve(const Matrix &mat) const;
    virtual Vector solve(const Vector &v) const;
    double trace() const;
    virtual double det() const;

    // The log of the absolute value of the determinant.
    virtual double logdet() const;

    Vector singular_values() const;  // sorted largest to smallest
    uint rank(double prop = 1e-12) const;
    // 'rank' is the number of singular values at least 'prop' times
    // the largest

    Matrix &add_outer(const Vector &x, const Vector &y, double w = 1.0);
    Matrix &add_outer(const Vector &x, const VectorView &y, double w = 1.0);
    Matrix &add_outer(const Vector &x, const ConstVectorView &y,
                      double w = 1.0);
    Matrix &add_outer(const VectorView &x, const Vector &y, double w = 1.0);
    Matrix &add_outer(const VectorView &x, const VectorView &y, double w = 1.0);
    Matrix &add_outer(const VectorView &x, const ConstVectorView &y,
                      double w = 1.0);
    Matrix &add_outer(const ConstVectorView &x, const Vector &y,
                      double w = 1.0);
    Matrix &add_outer(const ConstVectorView &x, const VectorView &y,
                      double w = 1.0);
    Matrix &add_outer(const ConstVectorView &x, const ConstVectorView &y,
                      double w = 1.0);
    // *this += w*x*y^T

    // Add the result of coefficient * left * right.transpose() to *this, and
    // return the result.
    Matrix &add_outer(const Matrix &left, const Matrix &right,
                      double coefficient);

    //--------  Math
    virtual Matrix &operator+=(double x);
    virtual Matrix &operator*=(double x);
    virtual Matrix &operator-=(double x);
    virtual Matrix &operator/=(double x);

    Matrix &operator+=(const Matrix &m);
    Matrix &operator+=(const SubMatrix &m);
    Matrix &operator+=(const ConstSubMatrix &m);

    Matrix &operator-=(const Matrix &m);
    Matrix &operator-=(const SubMatrix &m);
    Matrix &operator-=(const ConstSubMatrix &m);

    Matrix &operator/=(const Matrix &m);
    Matrix &operator/=(const SubMatrix &m);
    Matrix &operator/=(const ConstSubMatrix &m);

    Matrix &exp();  // in place exponentiation
    Matrix &log();  // in place logarithm

    // The sum of the elements in each column.  The result has size ncol().
    Vector col_sums() const;

    // The sum of the elements in each row.  The result has size nrow().
    Vector row_sums() const;

    virtual double sum() const;
    virtual double abs_norm() const;
    virtual double sumsq() const;
    virtual double prod() const;
    virtual double max() const;
    virtual double min() const;
    // The value of the entry with the largest absolute value.
    double max_abs() const;

    virtual std::ostream &display(std::ostream &out, int precision = 5) const;
    std::ostream &write(std::ostream &, bool nl = true) const;
    std::istream &read(std::istream &);

   protected:
    inline uint INDX(uint i, uint j) const;
    inline bool inrange(uint i, uint j) const;

   private:
    Vector data_;
    uint nr_, nc_;
  };

  //======================================================================
  class LabeledMatrix : public Matrix {
   public:
    // Args:
    //   m:  The matrix to which row and/or column names should be attached.
    //   row_names: Can be an empty vector if no row_names are
    //     desired.  If non-empty, row_names.size() == m.nrow() must
    //     be true.
    //   col_names: Can be an empty vector if no col_names are
    //     desired.  If non-empty, col_names.size() == m.ncol() must
    //     be true.
    LabeledMatrix(const Matrix &m, const std::vector<std::string> &row_names,
                  const std::vector<std::string> &col_names);
    const std::vector<std::string> &row_names() const { return row_names_; }
    const std::vector<std::string> &col_names() const { return col_names_; }

    std::ostream &display(std::ostream &out, int precision = 5) const override;
    Matrix drop_labels() const;

   private:
    std::vector<std::string> row_names_;
    std::vector<std::string> col_names_;
  };

  // Be kind to the British.
  typedef LabeledMatrix LabelledMatrix;

  //======================================================================

  // A "Matrix" with rows and columns counted from numbers other than zero.
  class ArbitraryOffsetMatrix {
   public:
    ArbitraryOffsetMatrix(int first_row, uint number_of_rows, int first_column,
                          uint number_of_columns, double initial_value = 0.0);

    double &operator()(int row, int column) {
      return data_(row - row_offset_, column - column_offset_);
    }

    double operator()(int row, int column) const {
      return data_(row - row_offset_, column - column_offset_);
    }

   private:
    Matrix data_;
    int row_offset_;
    int column_offset_;
  };

  //______________________________________________________________________

  // ---- template constructor --
  template <class FwdIt>
  Matrix::Matrix(FwdIt Beg, FwdIt End, uint nr, uint nc)
      : data_(Beg, End), nr_(nr), nc_(nc) {
    assert(data_.size() == nr * nc);
  }

  template <class FwdIt>
  FwdIt Matrix::assign(FwdIt b, FwdIt e) {
    assert(distance(b, e) == size());
    data_.assign(b, e);
    return e;
  }

  VectorView diag(Matrix &m);
  ConstVectorView diag(const Matrix &m);
  Matrix diag(const Vector &v);
  Matrix diag(const VectorView &v);

  inline uint Matrix::INDX(uint i, uint j) const { return j * nr_ + i; }
  inline bool Matrix::inrange(uint i, uint j) const {
    return i < nr_ && j < nc_;
  }
  std::ostream &operator<<(std::ostream &out, const Matrix &x);

  // Print the matrix to stdout.
  void print(const Matrix &m);
  std::istream &operator>>(std::istream &in, Matrix &m);
  // reads until a blank line is found or the end of a line

  inline double trace(const Matrix &m) { return m.trace(); }

  // Matrix - Matrix element-by-element operations
  inline Matrix operator+(const Matrix &m1, const Matrix &m2) {
    Matrix ans(m1);
    ans += m2;
    return ans;
  }
  inline Matrix operator-(const Matrix &m1, const Matrix &m2) {
    Matrix ans(m1);
    ans -= m2;
    return ans;
  }
  inline Matrix operator/(const Matrix &m1, const Matrix &m2) {
    Matrix ans(m1);
    ans /= m2;
    return ans;
  }

  // Matrix - double Field operators
  inline Matrix operator+(const Matrix &m, double a) {
    Matrix ans(m);
    ans += a;
    return ans;
  }
  inline Matrix operator+(double a, const Matrix &m) {
    return m + a;
  }
  inline Matrix operator-(const Matrix &m, double a) {
    return m + (-a);
  }
  inline Matrix operator-(double a, const Matrix &m) {
    Matrix ans(m.nrow(), m.ncol(), a);
    ans -= m;
    return ans;
  }
  inline Matrix operator*(const Matrix &m, double a) {
    Matrix ans(m);
    ans *= a;
    return ans;
  }
  inline Matrix operator*(double a, const Matrix &m) {
    return m * a;
  }

  Matrix operator/(double a, const Matrix &m);
  inline Matrix operator/(const Matrix &m, double a) {
    Matrix ans(m);
    ans /= a;
    return ans;
  }

  // Unary minus.
  inline Matrix operator-(const Matrix &x) { return -1.0 * x; }

  Matrix el_mult(const Matrix &A, const Matrix &B);
  double el_mult_sum(const Matrix &A, const Matrix &B);

  inline Matrix t(const Matrix &X) { return X.transpose(); }
  inline uint nrow(const Matrix &X) { return X.nrow(); }
  inline uint ncol(const Matrix &X) { return X.ncol(); }

  // Matrix multiplication
  Matrix matmult(const Matrix &A, const Matrix &B);   // A * B
  Matrix matTmult(const Matrix &A, const Matrix &B);  // A^T B
  Matrix matmultT(const Matrix &A, const Matrix &B);  // A B^T
  Vector matmult(const Vector &v, const Matrix &m);   // v^T m
  Vector matmult(const Matrix &m, const Vector &v);   // m * v

  //    Vector operator*(const Vector &v, const Matrix &m);
  Vector operator*(const Vector &v, const Matrix &m);
  Vector operator*(const Matrix &m, const Vector &v);
  Vector operator*(const VectorView &v, const Matrix &m);
  Vector operator*(const Matrix &m, const VectorView &v);
  Vector operator*(const ConstVectorView &v, const Matrix &m);
  Vector operator*(const Matrix &m, const ConstVectorView &v);
  Matrix operator*(const Matrix &a, const Matrix &b);

  // various useful transformations
  Matrix log(const Matrix &x);
  Matrix exp(const Matrix &x);
  inline double max(const Matrix &m) { return m.max(); }
  inline double min(const Matrix &m) { return m.min(); }
  inline double sum(const Matrix &M) { return M.sum(); }
  inline double det(const Matrix &M) { return M.det(); }

  Matrix rbind(const Matrix &m1, const Matrix &m2);
  Matrix rbind(const Vector &v, const Matrix &m);
  Matrix rbind(const Matrix &m, const Vector &v);
  Matrix rbind(const Vector &v1, const Vector &v2);
  Matrix rbind(double x, const Matrix &m);
  Matrix rbind(const Matrix &m, double x);

  Matrix cbind(const Matrix &m1, const Matrix &m2);
  Matrix cbind(const Vector &v, const Matrix &m);
  Matrix cbind(const Matrix &m, const Vector &v);
  Matrix cbind(const Vector &v1, const Vector &v2);
  Matrix cbind(double x, const Matrix &m);
  Matrix cbind(const Matrix &m, double x);

  Matrix drop_col(const Matrix &m, uint j);
  Matrix drop_cols(const Matrix &m, std::vector<uint> indx);

  Matrix permute_Matrix(const Matrix &Q, const std::vector<uint> &perm);
  double traceAB(const Matrix &A, const Matrix &B);
  double traceAtB(const Matrix &A, const Matrix &B);

  Matrix unpartition(double a, const Vector &v, const Matrix &B);
  Matrix unpartition(const Matrix &B, const Vector &v, double a);
  //  a v   or B v
  //  v B      v a

  Matrix unpartition(const Matrix &A, const Matrix &Rect, const Matrix &B);
  // A and B are square. Rect is the upper right rectangle: nrows
  // matches A, ncols matches B:
  // A      Rect
  // Rect^T B

  Matrix block_diagonal(const Matrix &A, const Matrix &B);
  // A and B both square

  //---------------------------------------------------------------------------
  // Routines for lower triangluar matrices.
  Vector Lmult(const Matrix &L, const Vector &y);   // ans = L * y
  Vector LTmult(const Matrix &L, const Vector &y);  // ans = L^T * y
  Vector Lsolve(const Matrix &L, const Vector &b);  // ans = L^{-1}b
  Vector &LTsolve_inplace(const Matrix &L, Vector &b);  // b = L^{T -1} * b
  Vector &Lsolve_inplace(const Matrix &L, Vector &b);   // b = L^{-1} * b
  Matrix Lsolve(const Matrix &L, const Matrix &B);      // ans = L^{-1} * B
  Matrix &Lsolve_inplace(const Matrix &L, Matrix &B);   // B = L^{-1} * B
  Matrix &LTsolve_inplace(const Matrix &L, Matrix &B);  // B = L^{T -1}* B
  // Return the inverse of L (which is upper triangluar).
  Matrix Linv(const Matrix &L);

  Vector Umult(const Matrix &U, const Vector &y);
  Matrix Umult(const Matrix &U, const Matrix &m);
  Vector Usolve(const Matrix &U, const Vector &b);     // ans = U^{-1}b
  Vector &Usolve_inplace(const Matrix &U, Vector &b);  // b = U^{-1}b
  Matrix Usolve(const Matrix &U, const Matrix &B);     // ans = U^{-1}B
  Matrix &Usolve_inplace(const Matrix &U, Matrix &B);  // B = U^{-1}B
  Matrix Uinv(const Matrix &U);

  // Returns the Kronecker product of A and B, which is a A partitioned matrix
  // A(0, 0)*B, A(0, 1)*B, ...
  // A(1, 0)*B, A(1, 1)*B, ...
  Matrix Kronecker(const Matrix &A, const Matrix &B);

  // Vectorization operator 'vec' forms a Vector from A by stacking its columns.
  inline Vector vec(const Matrix &A) {
    return Vector(A.begin(), A.end());
  }

}  // namespace BOOM

#endif  // BOOM_NEWLA_MATRIX_HPP
