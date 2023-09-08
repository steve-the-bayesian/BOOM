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

#ifndef NEW_LA_SPD_MATRIX_H
#define NEW_LA_SPD_MATRIX_H
#include <algorithm>
#include "LinAlg/Matrix.hpp"

namespace BOOM {

  class Selector;

  class SpdMatrix : public Matrix {
    // symmetric, positive definite matrix with 'square' storage
    // (i.e. 0's are stored)
   public:
    SpdMatrix();
    explicit SpdMatrix(uint dim, double diag = 0.0);
    explicit SpdMatrix(uint dim, const double *m, bool ColMajor = true);
    template <class FwdIt>
    explicit SpdMatrix(FwdIt Beg, FwdIt End);

    SpdMatrix(const SpdMatrix &rhs) = default;
    SpdMatrix(SpdMatrix &&rhs) = default;
    SpdMatrix &operator=(const SpdMatrix &rhs) = default;
    SpdMatrix &operator=(SpdMatrix &&rhs) = default;

    // Args:
    //   v: The elements of the matrix.
    //   minimal: If true then v just contains the upper diagonal
    //     elements of the matrix, in column major order.  Otherwise v
    //     contains all matrix elements, and by symmetry the order can
    //     be either column- or row-major.
    explicit SpdMatrix(const Vector &v, bool minimal = false);

    // Args:
    //   m: A Matrix object that happens to be symmetric and positive
    //     definite.
    //   check: If true, then throw an exception if m is not
    //     symmetric.  Skip the check if 'check' is false.
    //
    // cppcheck-suppress noExplicitConstructor
    SpdMatrix(const Matrix &m, bool check = true);
    // cppcheck-suppress noExplicitConstructor
    SpdMatrix(const SubMatrix &m, bool check = true);
    // cppcheck-suppress noExplicitConstructor
    SpdMatrix(const ConstSubMatrix &m, bool check = true);

    SpdMatrix &operator=(const Matrix &);
    SpdMatrix &operator=(const SubMatrix &);
    SpdMatrix &operator=(const ConstSubMatrix &);
    SpdMatrix &operator=(double x);
    bool operator==(const SpdMatrix &) const;

    void swap(SpdMatrix &rhs);
    // Fill entries with U(0,1) random variables, then multiply by
    // self-transpose.
    // Returns *this;
    SpdMatrix &randomize(RNG &rng = GlobalRng::rng) override;

    //-------- size and shape info ----------
    virtual uint nelem() const;  // number of distinct elements
    uint dim() const { return nrow(); }

    //--------- change size and shape ----------
    SpdMatrix &resize(uint n);

    // -------- row and column operations ----------
    SpdMatrix &set_diag(double x, bool zero_offdiag = true);
    SpdMatrix &set_diag(const Vector &v, bool zero_offdiag = true);

    //------------- Linear Algebra -----------
    //      lower_triangular_Matrix chol() const;
    Matrix chol() const;
    Matrix chol(bool &ok) const;
    SpdMatrix inv() const;
    SpdMatrix inv(bool &ok) const;

    // Invert the matrix without allocating extra storage.  Returns the log
    // determinant of the inverted matrix.
    double invert_inplace();

    // Determinant of the matrix.
    double det() const override;
    double logdet() const override;
    double logdet(bool &ok) const;

    // Returns this^{-1} * mat.  Throws an exception if this cannot be
    // inverted.
    Matrix solve(const Matrix &mat) const override;

    // Returns this^{-1} * v and sets ok to true.  If this cannot be
    // inverted ok is set to false and the return value a Vector of
    // the same dimension as rhs filled with negative_infinity.
    Vector solve(const Vector &v, bool &ok) const;

    // Returns this{-1} * v.  Throws an exception if this cannot be
    // inverted.
    Vector solve(const Vector &v) const override;

    // Copy the entries in the upper triangle to the lower triangle.
    void reflect();

    // Average corresponding elements above and below the diagonal to enforce
    // symmetry.
    void fix_near_symmetry();

    // Returns the Mahalanobis distance:  (x - y)^T (*this) (x - y).
    double Mdist(const Vector &x, const Vector &y) const;

    // Mahalanobis distance from 0:  x^T (*this) x
    double Mdist(const Vector &x) const;

    // Increment *this by w * x * x.transpose().
    // Args:
    //   x: The vector whose outer product augments *this.
    //   w: The coefficient (weight) multiplying the outer product.
    //   force_sym: If true then reflect() is called at the end of the
    //     calculation.  Otherwise only the upper triangle is computed.  If many
    //     outer products are to be summed, it is more efficient to call
    //     reflect() at the end.
    //
    //   inc: For overloads that have an 'inc' parameter, only the positions
    //     flagged by 'inc' will be updated.
    SpdMatrix &add_outer(const Vector &x, double w = 1.0,
                         bool force_sym = true);  // *this+= w*x*x^T
    SpdMatrix &add_outer(const Vector &x, const Selector &inc,
                         double w = 1.0, bool force_sym = true);
    SpdMatrix &add_outer(const VectorView &x, double w = 1.0,
                         bool force_sym = true);
    SpdMatrix &add_outer(const VectorView &x, const Selector &inc,
                         double w = 1.0, bool force_sym = true);
    SpdMatrix &add_outer(const ConstVectorView &x, double w = 1.0,
                         bool force_sym = true);
    SpdMatrix &add_outer(const ConstVectorView &x, const Selector &inc,
                         double w = 1.0, bool force_sym = true);

    // Increment *this by w * X * X.transpose().
    SpdMatrix &add_outer(const Matrix &X, double w = 1.0,
                         bool force_sym = true);

    SpdMatrix &add_inner(const Matrix &x, double w = 1.0);
    SpdMatrix &add_inner(const Matrix &X, const Vector &w,
                         bool force_sym = true);  // *this+= X^T w X

    // *this  += w x.t()*y + y.t()*x;
    SpdMatrix &add_inner2(const Matrix &x, const Matrix &y, double w = 1.0);
    // *this  += w x*y.t() + y*x.t();
    SpdMatrix &add_outer2(const Matrix &x, const Matrix &y, double w = 1.0);

    SpdMatrix &add_outer2(const Vector &x, const Vector &y, double w = 1.0);

    //--------- Matrix multiplication ------------

    // Multiply all off diagonal elements by 'scale', then return *this.
    SpdMatrix &scale_off_diagonal(double scale);

    Matrix &mult(const Matrix &B, Matrix &ans,
                 double scal = 1.0) const override;
    Matrix &Tmult(const Matrix &B, Matrix &ans,
                  double scal = 1.0) const override;
    Matrix &multT(const Matrix &B, Matrix &ans,
                  double scal = 1.0) const override;

    Matrix &mult(const SpdMatrix &B, Matrix &ans,
                 double scal = 1.0) const override;
    Matrix &Tmult(const SpdMatrix &B, Matrix &ans,
                  double scal = 1.0) const override;
    Matrix &multT(const SpdMatrix &B, Matrix &ans,
                  double scal = 1.0) const override;

    Matrix &mult(const DiagonalMatrix &B, Matrix &ans,
                 double scal = 1.0) const override;
    Matrix &Tmult(const DiagonalMatrix &B, Matrix &ans,
                  double scal = 1.0) const override;
    Matrix &multT(const DiagonalMatrix &B, Matrix &ans,
                  double scal = 1.0) const override;

    Vector &mult(const Vector &v, Vector &ans,
                 double scal = 1.0) const override;
    Vector &Tmult(const Vector &v, Vector &ans,
                  double scal = 1.0) const override;

    //------------- input/output ---------------
    virtual Vector vectorize(bool minimal = true) const;
    virtual void unvectorize(const Vector &v, bool minimal = true);
    Vector::const_iterator unvectorize(Vector::const_iterator &b,
                                       bool minimal = true);
    ConstVectorView::const_iterator unvectorize(
        ConstVectorView::const_iterator b, bool minimal = true);
    void make_symmetric(bool have_upper_triangle = true);

   private:
    // This function does not really make sense for SpdMatrix.  Its override
    // reports an error.
    SpdMatrix &randomize_gaussian(double mean, double sd,
                                  RNG &rng = GlobalRng::rng) override;
  };

  //______________________________________________________________________
  template <class Fwd>
  SpdMatrix::SpdMatrix(Fwd b, Fwd e) {
    uint n = std::distance(b, e);
    uint m = lround(::sqrt(static_cast<double>(n)));
    assert(m * m == n);
    resize(m);
    std::copy(b, e, begin());
  }

  SpdMatrix operator*(double x, const SpdMatrix &V);
  SpdMatrix operator*(const SpdMatrix &v, double x);
  SpdMatrix operator/(const SpdMatrix &v, double x);

  SpdMatrix Id(uint p);

  SpdMatrix RTR(const Matrix &R, double a = 1.0);  // a * R^T%*%R
  SpdMatrix LLT(const Matrix &L, double a = 1.0);  // a * L%*%L^T

  SpdMatrix outer(const Vector &v);
  SpdMatrix outer(const VectorView &v);
  SpdMatrix outer(const ConstVectorView &v);

  Matrix chol(const SpdMatrix &Sigma);
  Matrix chol(const SpdMatrix &Sigma, bool &ok);

  SpdMatrix Kronecker(const SpdMatrix &A, const SpdMatrix &B);

  inline double logdet(const SpdMatrix &Sigma) { return Sigma.logdet(); }

  SpdMatrix chol2inv(const Matrix &L);
  // Returns A^{-1}, where L is the cholesky factor of A.

  // Args:
  //   A: the outer matrix doing the sandwiching.
  //   V: the inner matrix being sandwiched.
  // Returns:
  //   A * V * A^T
  SpdMatrix sandwich(const Matrix &A, const SpdMatrix &V);
  SpdMatrix sandwich(const Matrix &A, const Vector &V);

  // Args:
  //   X: A matrix to be adjusted by averaging with its own diagonal.
  //   diagonal_shrinkage: A number between 0 and 1 giving the weight assigned
  //     to the diagonal.  If 0 then the original matrix is returned.  If 1 then
  //     the diagonal matrix is returned
  //
  // Returns:
  //  (1 - a) * X + a * diag(X), where a is 'diagonal_shrinkage'.
  SpdMatrix self_diagonal_average(const SpdMatrix &X,
                                  double diagonal_shrinkage);
  void self_diagonal_average_inplace(SpdMatrix &X,
                                     double diagonal_shrinkage);

  // Args:
  //   A: the outer matrix doing the sandwiching.
  //   V: the inner matrix being sandwiched.
  // Returns:
  //   A^T * V * A
  inline SpdMatrix sandwich_transpose(const Matrix &A, const SpdMatrix &V) {
    return A.Tmult(V * A);
  }
  SpdMatrix sandwich_transpose(const Matrix &A, const Vector &V);

  SpdMatrix as_symmetric(const Matrix &A);

  SpdMatrix sum_self_transpose(const Matrix &A);  // A + A.t()

  // Returns the vector of eigenvalues of X, sorted from smallest to
  // largest.
  Vector eigenvalues(const SpdMatrix &X);

  // Args:
  //   V:  The matrix to decompose.
  //   eigenvectors:  On return the columns of 'eigenvectors' are the
  //     eigenvectors coresponding to the eigenvalues in the same
  //     position.
  // Returns: the vector of eigenvalues of V, sorted from smallest to
  //   largest.
  //
  // The relationship is V = Q^T Lambda Q, or Q * V = Lambda * Q,
  // where Q^T = eigenvectors.
  Vector eigen(const SpdMatrix &V, Matrix &eigenvectors);

  // Returns the largest eigenvalue of X.
  double largest_eigenvalue(const SpdMatrix &X);

  // An SpdMatrix X can be written X = Q^T Lambda Q, where the columns
  // of Q^T contain the eigenvectors (i.e. the eigenvectors are the
  // rows of Q), and Lambda is a diagonal matrix containing the
  // eigenvalues.
  //
  // The symmetric square root of X is Q^T Lambda^{1/2} Q.
  SpdMatrix symmetric_square_root(const SpdMatrix &X);

  // An SpdMatrix X can be written X = Q^T Lambda Q, where the columns
  // of Q^T contain the eigenvectors (i.e. the eigenvectors are the
  // rows of Q), and Lambda is a diagonal matrix containing the
  // eigenvalues.  The "eigen_root" is a matrix square root of X
  // defined as Z = Lambda^{1/2} * Q.  It is a matrix square root in
  // the sense that Z^T * Z = Q^T * Lambda^{1/2} * Lambda^{1/2} * Q =
  // X.
  //
  // Note that the eigen_root can be multiplied by any orthogonal
  // matrix A to produce W = A * Lambda^{1/2} * Q, which preserves the
  // relationship W^T * W = X.
  Matrix eigen_root(const SpdMatrix &X);

  // Produce a dense SpdMatrix with 'blocks' as the block diagonal elements.
  SpdMatrix block_diagonal_spd(const std::vector<SpdMatrix> &blocks);

}  // namespace BOOM

#endif  // NEW_LA_SPD_MATRIX_H
