// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#ifndef BOOM_SPARSE_MATRIX_HPP_
#define BOOM_SPARSE_MATRIX_HPP_

#include <map>

#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/Vector.hpp"

#include "Models/ParamTypes.hpp"
#include "Models/SpdParams.hpp"

#include "cpputil/Ptr.hpp"
#include "cpputil/RefCounted.hpp"
#include "cpputil/report_error.hpp"

#include "Models/Glm/GlmCoefs.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"

#include "stats/moments.hpp"

namespace BOOM {

  //======================================================================
  // A SparseKalmanMatrix is a sparse matrix that can be used in the Kalman
  // recursions.  This may get expanded to a more full fledged sparse matrix
  // class later on, if need be.
  class SparseKalmanMatrix : private RefCounted {
   public:
    virtual ~SparseKalmanMatrix() {}

    virtual int nrow() const = 0;
    virtual int ncol() const = 0;

    virtual Vector operator*(const Vector &v) const = 0;
    virtual Vector operator*(const VectorView &v) const = 0;
    virtual Vector operator*(const ConstVectorView &v) const = 0;

    virtual Matrix operator*(const Matrix &rhs) const;

    virtual Vector Tmult(const ConstVectorView &v) const = 0;
    virtual Matrix Tmult(const Matrix &rhs) const;

    // Replace the argument P with
    //   this * P * this.transpose()
    // This only works with square matrices.  Non-square matrices will throw.
    virtual void sandwich_inplace(SpdMatrix &P) const;
    virtual void sandwich_inplace_submatrix(SubMatrix P) const;

    // Replace the argument P with
    //    this->transpose() * P * this
    // This only works with square matrices.  Non-square matrices will throw.
    virtual void sandwich_inplace_transpose(SpdMatrix &P) const;

    // Returns *this * P * this->transpose().
    // This is a valid call, even if *this is non-square.
    virtual SpdMatrix sandwich(const SpdMatrix &P) const;

    // Returns this->Tmult(P) * (*this), which is equivalent to
    // calling this->transpose()->sandwich(P) (or it would be if
    // this->transpose() was defined).  This is a valid call, even if
    // *this is non-square.
    virtual SpdMatrix sandwich_transpose(const SpdMatrix &P) const;

    // Returns this->transpose() * this
    virtual SpdMatrix inner() const = 0;

    // Returns this->transpose * diag(weights) * this
    virtual SpdMatrix inner(const ConstVectorView &weights) const = 0;

    // P += *this
    virtual Matrix &add_to(Matrix &P) const = 0;
    virtual SubMatrix add_to_submatrix(SubMatrix P) const;

    // Returns a dense matrix representation of *this.  Mainly for
    // debugging and testing.
    //
    // The default implementation only works for square matrices.
    // Child classes that can be non-square should override.
    virtual Matrix dense() const;

    // Returns this * rhs.transpose().
    Matrix multT(const Matrix &rhs) const;

    // Checks that nrow() == i.  Reports an error if it does not.
    void conforms_to_rows(int i) const;

    // Checks that ncol() == i.  Reports an error if it does not.
    void conforms_to_cols(int i) const;

    void check_can_add(const SubMatrix &block) const;

    std::ostream & print(std::ostream &out = std::cout) const {
      out << dense();
      return out;
    }

    std::string to_string() const {
      std::ostringstream out;
      print(out);
      return out.str();
    }

   private:
    friend void intrusive_ptr_add_ref(SparseKalmanMatrix *m) { m->up_count(); }
    friend void intrusive_ptr_release(SparseKalmanMatrix *m) {
      m->down_count();
      if (m->ref_count() == 0) delete m;
    }
  };

  Matrix operator*(const Matrix &lhs, const SparseKalmanMatrix &rhs);

  // Returns lhs * rhs.transpose();
  Matrix multT(const SpdMatrix &lhs, const SparseKalmanMatrix &rhs);

  //======================================================================
  // The SparseMatrixBlock classes are SparseKalmanMatrices that can be used as
  // elements in a BlockDiagonalMatrix.
  class SparseMatrixBlock : public SparseKalmanMatrix {
   public:
    ~SparseMatrixBlock() override {}
    virtual SparseMatrixBlock *clone() const = 0;

    // lhs = this * rhs
    virtual void multiply(VectorView lhs, const ConstVectorView &rhs) const = 0;
    Vector operator*(const Vector &v) const override;
    Vector operator*(const VectorView &v) const override;
    Vector operator*(const ConstVectorView &v) const override;
    Matrix operator*(const Matrix &rhs) const override;

    // lhs += this * rhs
    virtual void multiply_and_add(VectorView lhs,
                                  const ConstVectorView &rhs) const = 0;

    // lhs = this.transpose() * rhs
    virtual void Tmult(VectorView lhs, const ConstVectorView &rhs) const = 0;
    Vector Tmult(const ConstVectorView &rhs) const override;
    Matrix Tmult(const Matrix &rhs) const override;

    // Replace x with this * x.  Assumes *this is square.
    virtual void multiply_inplace(VectorView x) const = 0;

    // m = this * m
    virtual void matrix_multiply_inplace(SubMatrix m) const;

    Matrix dense() const override;

    // m = m * this->t();
    virtual void matrix_transpose_premultiply_inplace(SubMatrix m) const;

    // Add *this to block
    virtual void add_to_block(SubMatrix block) const = 0;
    Matrix &add_to(Matrix &P) const override;

    // Returns a solution to y = this * x.  This may raise an error if the
    // solution is not well defined, which is the default.
    virtual Vector left_inverse(const ConstVectorView &x) const;
  };

  //===========================================================================
  // A sparse matrix block that is, itself, a block diagonal matrix.  Blocks in
  // this matrix are sparse, and must be square.
  class BlockDiagonalMatrixBlock : public SparseMatrixBlock {
   public:
    BlockDiagonalMatrixBlock() : dim_(0) {}
    BlockDiagonalMatrixBlock(const BlockDiagonalMatrixBlock &rhs);
    BlockDiagonalMatrixBlock(BlockDiagonalMatrixBlock &&rhs) = default;
    BlockDiagonalMatrixBlock *clone() const override;
    BlockDiagonalMatrixBlock &operator=(const BlockDiagonalMatrixBlock &rhs);
    BlockDiagonalMatrixBlock &operator=(BlockDiagonalMatrixBlock &&rhs) =
        default;

    void add_block(const Ptr<SparseMatrixBlock> &block);

    int nrow() const override { return dim_; }
    int ncol() const override { return dim_; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_inplace(VectorView x) const override;
    void matrix_multiply_inplace(SubMatrix m) const override;
    void matrix_transpose_premultiply_inplace(SubMatrix m) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;

   private:
    // Checks that this can multiply rhs, and that lhs is correctly sized to
    // receive the result.  An error is reported if either check fails.
    void check_can_multiply(const VectorView &lhs,
                            const ConstVectorView &rhs) const;
    std::vector<Ptr<SparseMatrixBlock>> blocks_;
    int dim_;
  };
  //===========================================================================
  // A rectangular matrix formed by stacking a collection of sparse matrices
  // with the same number of columns.
  class StackedMatrixBlock : public SparseMatrixBlock {
   public:
    StackedMatrixBlock() : nrow_(0), ncol_(0) {}
    StackedMatrixBlock(const StackedMatrixBlock &rhs);
    StackedMatrixBlock(StackedMatrixBlock &&rhs) = default;
    StackedMatrixBlock *clone() const override {
      return new StackedMatrixBlock(*this);
    }
    StackedMatrixBlock &operator=(const StackedMatrixBlock &rhs);
    StackedMatrixBlock &operator=(StackedMatrixBlock &&rhs) = default;
    int nrow() const override { return nrow_; }
    int ncol() const override { return ncol_; }

    // Remove all entries from blocks_ and set nrow_ and ncol_ to zero.
    void clear();
    void add_block(const Ptr<SparseMatrixBlock> &block);
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_inplace(VectorView x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;

    Matrix dense() const override;

    // Returns the "best least squares" solution.
    Vector left_inverse(const ConstVectorView &x) const override;

   private:
    // Each block must have the same number of columns, which are determined by
    // the first block added.
    std::vector<Ptr<SparseMatrixBlock>> blocks_;
    int nrow_, ncol_;
  };

  //======================================================================
  // The LocalLinearTrendMatrix is
  //  1 1
  //  0 1
  //  It corresponds to state elements [mu, delta], where mu[t] =
  //  mu[t-1] + delta[t-1] + error[0] and de[ta[t] = delta[t-1] +
  //  error[1].
  class LocalLinearTrendMatrix : public SparseMatrixBlock {
   public:
    LocalLinearTrendMatrix *clone() const override;
    int nrow() const override { return 2; }
    int ncol() const override { return 2; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_inplace(VectorView v) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;
    Matrix dense() const override;
    Vector left_inverse(const ConstVectorView &x) const override {
      Vector ans = x;
      ans[0] -= ans[1];
      return ans;
    }
  };

  //======================================================================
  // A SparseMatrixBlock filled with a DenseMatrix.  I.e. a dense
  // sub-block of a sparse matrix.
  class DenseMatrix : public SparseMatrixBlock {
   public:
    explicit DenseMatrix(const Matrix &m) : m_(m) {}
    DenseMatrix(const DenseMatrix &rhs) : SparseMatrixBlock(rhs), m_(rhs.m_) {}
    DenseMatrix *clone() const override { return new DenseMatrix(*this); }
    void resize(int rows, int cols) { m_.resize(rows, cols); }
    void set(const Matrix &matrix) {m_ = matrix;}
    VectorView col(int i) { return m_.col(i); }
    int nrow() const override { return m_.nrow(); }
    int ncol() const override { return m_.ncol(); }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      lhs = m_ * rhs;
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      lhs += m_ * rhs;
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      lhs = m_.Tmult(rhs);
    }
    void multiply_inplace(VectorView x) const override { x = m_ * x; }
    SpdMatrix inner() const override {return m_.inner();}
    SpdMatrix inner(const ConstVectorView &weights) const override {
      return m_.inner(weights);
    }
    void add_to_block(SubMatrix block) const override { block += m_; }
    Matrix dense() const override { return m_; }

    // Fast access to the underlying matrix without returning a copy.
    const Matrix &matrix() const {return m_;}

    Vector left_inverse(const ConstVectorView &x) const override {
      return m_.solve(x);
    }

   private:
    Matrix m_;
  };

  //======================================================================
  class DenseSpdBase : public SparseMatrixBlock {
   public:
    virtual const SpdMatrix &value() const = 0;
    int nrow() const override { return value().nrow(); }
    int ncol() const override { return value().ncol(); }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      lhs = value() * rhs;
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      lhs += value() * rhs;
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      lhs = value() * rhs;
    }
    void multiply_inplace(VectorView x) const override { x = value() * x; }
    void add_to_block(SubMatrix block) const override { block += value(); }
  };

  // A SparseMatrixBlock filled with a dense SpdMatrix.
  class DenseSpd : public DenseSpdBase {
   public:
    explicit DenseSpd(const SpdMatrix &m) : m_(m) {}
    DenseSpd(const DenseSpd &rhs) : DenseSpdBase(rhs), m_(rhs.m_) {}
    DenseSpd *clone() const override { return new DenseSpd(*this); }
    const SpdMatrix &value() const override { return m_; }
    void set_matrix(const SpdMatrix &m) { m_ = m; }
    SpdMatrix inner() const override {return m_.inner();}
    SpdMatrix inner(const ConstVectorView &weights) const override {
      return m_.inner(weights);
    }
    Vector left_inverse(const ConstVectorView &v) const override {
      return m_.solve(v);
    }

   private:
    SpdMatrix m_;
  };

  class DenseSpdParamView : public DenseSpdBase {
   public:
    explicit DenseSpdParamView(const Ptr<SpdParams> &matrix)
        : matrix_(matrix) {}
    DenseSpdParamView *clone() const override {
      return new DenseSpdParamView(*this);
    }
    const SpdMatrix &value() const override { return matrix_->var(); }
    SpdMatrix inner() const override {return value().inner();}
    SpdMatrix inner(const ConstVectorView &weights) const override {
      return value().inner(weights);
    }

    Vector left_inverse(const ConstVectorView &v) const override {
      return value().solve(v);
    }

   private:
    Ptr<SpdParams> matrix_;
  };

  //======================================================================
  // A component that is is a diagonal matrix (a square matrix with
  // zero off-diagonal components).  The diagonal elements can be
  // changed to arbitrary values after construction.  This class is
  // conceptutally similar to UpperLeftDiagonalMatrix, but it allows
  // different behavior with respect to setting its elements to
  // arbitrary values.
  class DiagonalMatrixBlockBase : public SparseMatrixBlock {
   public:
    virtual const Vector &diagonal_elements() const = 0;
    double operator[](int i) const { return diagonal_elements()[i]; }
    int nrow() const override { return diagonal_elements().size(); }
    int ncol() const override { return diagonal_elements().size(); }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      lhs = diagonal_elements();
      lhs *= rhs;
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      lhs += diagonal_elements() * rhs;
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      multiply(lhs, rhs);
    }
    void multiply_inplace(VectorView x) const override {
      x *= diagonal_elements();
    }
    void matrix_multiply_inplace(SubMatrix m) const override {
      for (int i = 0; i < m.ncol(); ++i) {
        m.col(i) *= diagonal_elements();
      }
    }

    void matrix_transpose_premultiply_inplace(SubMatrix m) const override {
      for (int i = 0; i < m.nrow(); ++i) {
        m.row(i) *= diagonal_elements();
      }
    }

    SpdMatrix inner() const override {
      SpdMatrix ans(nrow(), 0.0);
      ans.diag() = pow(diagonal_elements(), 2);
      return ans;
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      if (weights.size() != nrow()) {
        report_error("Wrong size weight vector.");
      }
      SpdMatrix ans(nrow());
      const Vector &diag(diagonal_elements());
      for (int i = 0; i < ans.nrow(); ++i) {
        ans(i, i) = square(diag[i]) * weights[i];
      }
      return ans;
    }

    void add_to_block(SubMatrix block) const override {
      block.diag() += diagonal_elements();
    }

    Vector left_inverse(const ConstVectorView &x) const override {
      return x / diagonal_elements();
    }
  };

  //----------------------------------------------------------------------------
  // A diagonal matrix with elements to be set manually.
  class DiagonalMatrixBlock : public DiagonalMatrixBlockBase {
   public:
    explicit DiagonalMatrixBlock(int size) : diagonal_elements_(size) {}
    explicit DiagonalMatrixBlock(const Vector &diagonal_elements)
        : diagonal_elements_(diagonal_elements) {}
    DiagonalMatrixBlock *clone() const override {
      return new DiagonalMatrixBlock(*this);
    }
    const Vector &diagonal_elements() const override {
      return diagonal_elements_;
    }
    double &mutable_element(int i) { return diagonal_elements_[i]; }

    void set_elements(const Vector &v) { diagonal_elements_ = v; }
    void set_elements(const VectorView &v) { diagonal_elements_ = v; }
    void set_elements(const ConstVectorView &v) { diagonal_elements_ = v; }

   private:
    Vector diagonal_elements_;
  };

  //----------------------------------------------------------------------------
  // A diagonal matrix obtained by storing a vector of scalar variances in
  // VectorParams.
  class DiagonalMatrixBlockVectorParamView : public DiagonalMatrixBlockBase {
   public:
    explicit DiagonalMatrixBlockVectorParamView(
        const Ptr<VectorParams> &diagonal_elements)
        : diagonal_elements_(diagonal_elements) {}
    DiagonalMatrixBlockVectorParamView *clone() const override {
      return new DiagonalMatrixBlockVectorParamView(*this);
    }
    const Vector &diagonal_elements() const override {
      return diagonal_elements_->value();
    }

   private:
    Ptr<VectorParams> diagonal_elements_;
  };

  //----------------------------------------------------------------------------
  // A diagonal matrix implemented using a vector of univariate variance
  // paramters.
  class DiagonalMatrixParamView : public DiagonalMatrixBlockBase {
   public:
    DiagonalMatrixParamView() : current_(false) {}
    DiagonalMatrixParamView(const DiagonalMatrixParamView &rhs) = default;
    DiagonalMatrixParamView(DiagonalMatrixParamView &&rhs) = default;
    DiagonalMatrixParamView *clone() const override {
      return new DiagonalMatrixParamView(*this);
    }
    DiagonalMatrixParamView &operator=(const DiagonalMatrixParamView &rhs) =
        default;
    DiagonalMatrixParamView &operator=(DiagonalMatrixParamView &&rhs) = default;

    int nrow() const override { return variances_.size(); }
    int ncol() const override { return variances_.size(); }

    const Vector &diagonal_elements() const override {
      ensure_current();
      return diagonal_elements_;
    }

    // Add a variance element to the diagonal, increasing the diagonal dimension
    // by 1.
    void add_variance(const Ptr<UnivParams> &variance);

   private:
    std::vector<Ptr<UnivParams>> variances_;
    mutable Vector diagonal_elements_;
    mutable bool current_;
    void ensure_current() const;
    void set_observer(const Ptr<UnivParams> &variance);
  };

  //============================================================================
  // A diagonal matrix whose elements are stored in a SparseVector.  This class
  // is similar to the other diagonal matrices, but it does not offer a
  // diagonal_elements() member returning a dense vector.
  class SparseDiagonalMatrixBlockParamView : public SparseMatrixBlock {
   public:
    explicit SparseDiagonalMatrixBlockParamView(int dim) : dim_(dim) {}
    SparseDiagonalMatrixBlockParamView *clone() const override;
    void add_element(const Ptr<UnivParams> &element, int position);

    int nrow() const override { return dim_; }
    int ncol() const override { return dim_; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_inplace(VectorView x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;

   private:
    std::vector<Ptr<UnivParams>> elements_;
    std::vector<int> positions_;
    int dim_;
  };

  //======================================================================
  // A seasonal state space matrix describes the state evolution in an
  // dynamic linear model.  Conceptually it looks like this:
  // -1 -1 -1 -1 ... -1
  //  1  0  0  0 .... 0
  //  0  1  0  0 .... 0
  //  0  0  1  0 .... 0
  //  0  0  0  1 .... 0
  // A row of -1's at the top, then an identity matrix with a column
  // of 0's appended on the right hand side.
  class SeasonalStateSpaceMatrix : public SparseMatrixBlock {
   public:
    explicit SeasonalStateSpaceMatrix(int number_of_seasons);
    SeasonalStateSpaceMatrix *clone() const override;
    int nrow() const override;
    int ncol() const override;

    // lhs = (*this) * rhs;
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    // lhs = this->transpose() * rhs
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    // x = (*this) * x;
    void multiply_inplace(VectorView x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;
    Matrix dense() const override;

    Vector left_inverse(const ConstVectorView &x) const override;
   private:
    int number_of_seasons_;
  };

  //======================================================================
  // An AutoRegressionTransitionMatrix is a [p X p] matrix with top
  // row containing a vector of autoregression parameters.  The lower
  // left block is a [p-1 X p-1] identity matrix (i.e. a shift-down
  // operator), and the lower right block is a [p-1 X 1] vector of
  // 0's.
  //
  // phi1 phi2  ....  phip
  // 1    0       0   0
  // 0    1       0   0
  // ...
  // 0    0       0   0
  // 0    0       1   0
  class AutoRegressionTransitionMatrix : public SparseMatrixBlock {
   public:
    explicit AutoRegressionTransitionMatrix(const Ptr<GlmCoefs> &rho);
    AutoRegressionTransitionMatrix(const AutoRegressionTransitionMatrix &rhs);
    AutoRegressionTransitionMatrix *clone() const override;

    int nrow() const override;
    int ncol() const override;
    // lhs = this * rhs
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_inplace(VectorView x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;
    // virtual void matrix_multiply_inplace(SubMatrix m) const;
    // virtual void matrix_transpose_premultiply_inplace(SubMatrix m) const;
    Matrix dense() const override;

    Vector left_inverse(const ConstVectorView &x) const override;
   private:
    Ptr<GlmCoefs> autoregression_params_;
  };

  //======================================================================
  // The [dim x dim] identity matrix
  class IdentityMatrix : public SparseMatrixBlock {
   public:
    explicit IdentityMatrix(int dim) : dim_(dim) {}
    IdentityMatrix *clone() const override { return new IdentityMatrix(*this); }
    int nrow() const override { return dim_; }
    int ncol() const override { return dim_; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      lhs = rhs;
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      lhs += rhs;
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_rows(rhs.size());
      conforms_to_cols(lhs.size());
      lhs = rhs;
    }
    void multiply_inplace(VectorView x) const override {}
    void matrix_multiply_inplace(SubMatrix m) const override {}
    void matrix_transpose_premultiply_inplace(SubMatrix m) const override {}
    void add_to_block(SubMatrix block) const override { block.diag() += 1.0; }
    SpdMatrix inner() const override {return SpdMatrix(nrow(), 1.0);}
    SpdMatrix inner(const ConstVectorView &weights) const override {
      SpdMatrix ans(nrow());
      ans.diag() = weights;
      return ans;
    }

   private:
    int dim_;
  };

  //======================================================================
  // An empty matrix with no rows or columns.  This is useful for
  // models which are all deterministic, with no random component.
  class EmptyMatrix : public SparseMatrixBlock {
   public:
    EmptyMatrix *clone() const override { return new EmptyMatrix(*this); }
    int nrow() const override { return 0; }
    int ncol() const override { return 0; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {}
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {}
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {}
    void multiply_inplace(VectorView x) const override {}
    void matrix_multiply_inplace(SubMatrix m) const override {}
    void matrix_transpose_premultiply_inplace(SubMatrix m) const override {}
    void add_to_block(SubMatrix block) const override {}
    Matrix dense() const override {
      return Matrix(0, 0);
    }
    SpdMatrix inner() const override {return SpdMatrix(0);}
    SpdMatrix inner(const ConstVectorView &weights) const override {
      return SpdMatrix(0);
    }
  };

  //======================================================================
  // A scalar constant times the identity matrix
  class ConstantMatrixBase : public SparseMatrixBlock {
   public:
    explicit ConstantMatrixBase(int dim) : dim_(dim) {}
    virtual double value() const = 0;

    // In most cases the dimension of the matrix will be set in the constructor.
    // If the dimension is data dependent then it can be set dynamically with
    // set_dim.
    void set_dim(int dim) { dim_ = dim; }
    int nrow() const override { return dim_; }
    int ncol() const override { return dim_; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      // Doing this operation in two steps, insted of lhs = rhs *
      // value(), eliminates a temporary that profiliing found to be
      // expensive.
      lhs = rhs;
      lhs *= value();
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      lhs.axpy(rhs, value());
    }

    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_rows(rhs.size());
      conforms_to_cols(lhs.size());
      lhs = rhs * value();
    }
    void multiply_inplace(VectorView x) const override { x *= value(); }
    void matrix_multiply_inplace(SubMatrix x) const override { x *= value(); }
    void matrix_transpose_premultiply_inplace(SubMatrix x) const override {
      x *= value();
    }
    SpdMatrix inner() const override {
      return SpdMatrix(nrow(), square(value()));
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      SpdMatrix ans(nrow());
      ans.diag() = weights * square(value());
      return ans;
    }

    void add_to_block(SubMatrix block) const override { block.diag() += value(); }

   private:
    int dim_;
  };

  class ConstantMatrix : public ConstantMatrixBase {
   public:
    ConstantMatrix(int dim, double value)
        : ConstantMatrixBase(dim), value_(value) {}
    ConstantMatrix *clone() const override { return new ConstantMatrix(*this); }
    void set_value(double value) { value_ = value; }
    double value() const override { return value_; }

   private:
    double value_;
  };

  class ConstantMatrixParamView : public ConstantMatrixBase {
   public:
    ConstantMatrixParamView(int dim, const Ptr<UnivParams> &value)
        : ConstantMatrixBase(dim), value_(value) {}
    ConstantMatrixParamView *clone() const override {
      return new ConstantMatrixParamView(nrow(), value_);
    }
    double value() const override { return value_->value(); }

   private:
    Ptr<UnivParams> value_;
  };

  //======================================================================
  // A square matrix of all zeros.
  class ZeroMatrix : public ConstantMatrix {
   public:
    explicit ZeroMatrix(int dim) : ConstantMatrix(dim, 0.0) {}
    ZeroMatrix *clone() const override { return new ZeroMatrix(*this); }
    void add_to_block(SubMatrix block) const override {}
  };

  // A 'matrix' with zero columns.  This is not really a 'Matrix' in the
  // traditional sense.  Including a NullMatrix in a BlockDiagonalMatrix expands
  // the number of rows (all 0's), but does not expand the number of columns.  A
  // block diagonal matrix with 'regular' blocks B0 and B1 separated by a
  // NullMatrix looks like:
  //
  // [B0 0]
  // [0  0]
  // [0 B1]
  class NullMatrix: public SparseMatrixBlock {
   public:
    // Create a Nullmatrix with 'dim' rows and 0 columns.
    explicit NullMatrix(int dim): dim_(dim) {}
    NullMatrix *clone() const override {return new NullMatrix(*this);}
    int ncol() const override {return 0;}
    int nrow() const override {return dim_;}

    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_rows(lhs.size());
    }

    void multiply_and_add(
        VectorView lhs, const ConstVectorView &rhs) const override { }

    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_rows(rhs.size());
      lhs = 0.0;
    }

    void multiply_inplace(VectorView x) const override {
      report_error("Only square matrices can multiply_inplace.");
    }
    void matrix_multiply_inplace(SubMatrix x) const override {
      report_error("Only square matrices can matrix_multiply_inplace.");
    }
    void matrix_transpose_premultiply_inplace(SubMatrix x) const override {
      report_error("Only square matrices can matrix_transpose_"
                   "premultiply_inplace.");
    }

    SpdMatrix inner() const override {
      return SpdMatrix(nrow(), 0.0);
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      return this->inner();
    }

    void add_to_block(SubMatrix block) const override {
      report_error("A NullMatrix cannot add_to_block.");
    }

   private:
    int dim_;
  };

  //======================================================================
  //  A matrix that is all zeros except for a single nonzero value in
  //  the (0,0) corner.
  class UpperLeftCornerMatrixBase : public SparseMatrixBlock {
   public:
    explicit UpperLeftCornerMatrixBase(int dim) : dim_(dim) {}
    virtual double value() const = 0;
    int nrow() const override { return dim_; }
    int ncol() const override { return dim_; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      lhs = rhs * 0;
      lhs[0] = rhs[0] * value();
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      lhs[0] += rhs[0] * value();
    }

    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      // An upper left corner matrix is symmetric, so Tmult is the
      // same as multiply.
      multiply(lhs, rhs);
    }
    void multiply_inplace(VectorView x) const override {
      double tmp = x[0];
      x = 0;
      x[0] = tmp * value();
    }
    SpdMatrix inner() const override {
      SpdMatrix ans(dim_, 0.0);
      ans(0, 0) = square(value());
      return ans;
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      if (weights.size() != nrow()) {
        report_error("Wrong size weight vector.");
      }
      SpdMatrix ans(dim_, 0.0);
      ans(0, 0) = square(value()) * weights[0];
      return ans;
    }

    void add_to_block(SubMatrix block) const override { block(0, 0) += value(); }

   private:
    int dim_;
  };

  class UpperLeftCornerMatrix : public UpperLeftCornerMatrixBase {
   public:
    UpperLeftCornerMatrix(int dim, double value)
        : UpperLeftCornerMatrixBase(dim), value_(value) {}
    UpperLeftCornerMatrix *clone() const override {
      return new UpperLeftCornerMatrix(*this);
    }
    double value() const override { return value_; }
    void set_value(double value) { value_ = value; }

   private:
    double value_;  // the value in the upper left corner of the matrix
  };

  class UpperLeftCornerMatrixParamView : public UpperLeftCornerMatrixBase {
   public:
    UpperLeftCornerMatrixParamView(int dim, const Ptr<UnivParams> &param)
        : UpperLeftCornerMatrixBase(dim), value_(param) {}
    UpperLeftCornerMatrixParamView *clone() const override {
      return new UpperLeftCornerMatrixParamView(*this);
    }
    double value() const override { return value_->value(); }

   private:
    Ptr<UnivParams> value_;
  };

  //======================================================================
  // An nx1 rectangular matrix with a 1 in the upper left corner and
  // 0's elsewhere.  This is intended to be the "expander matrix" for
  // state errors in models with a single dimension of randomness but
  // multiple dimensions of state.
  //
  // (*this) = [1, 0, 0, ..., 0]^T
  class FirstElementSingleColumnMatrix : public SparseMatrixBlock {
   public:
    explicit FirstElementSingleColumnMatrix(int nrow) : nrow_(nrow) {}
    FirstElementSingleColumnMatrix *clone() const override {
      return new FirstElementSingleColumnMatrix(*this);
    }

    int nrow() const override { return nrow_; };
    int ncol() const override { return 1; }

    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      lhs[0] = rhs[0];
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      lhs[0] += rhs[0];
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      lhs[0] = rhs[0];
    }

    void multiply_inplace(VectorView x) const override {
      report_error("multiply_inplace only works for square matrices.");
    }

    void matrix_multiply_inplace(SubMatrix m) const override {
      report_error("matrix_multiply_inplace only works for square matrices.");
    }

    void matrix_transpose_premultiply_inplace(SubMatrix m) const override {
      report_error(
          "matrix_transpose_premultiply_inplace only works for "
          "square matrices.");
    }

    SpdMatrix inner() const override {
      return SpdMatrix(1, 1.0);
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      return SpdMatrix(1, weights[0]);
    }

    void add_to_block(SubMatrix block) const override { block(0, 0) += 1.0; }

    Matrix dense() const override {
      Matrix ans(nrow(), ncol(), 0.0);
      ans(0, 0) = 1.0;
      return ans;
    }

   private:
    int nrow_;
  };

  //======================================================================
  // A rectangular matrix consisting of an Identity matrix with rows of zeros
  // appended on bottom.
  class ZeroPaddedIdentityMatrix : public SparseMatrixBlock {
   public:
    ZeroPaddedIdentityMatrix(int nrow, int ncol) : nrow_(nrow), ncol_(ncol) {
      if (nrow < ncol) {
        report_error(
            "A ZeroPaddedIdentityMatrix must have at least as many "
            "rows as columns.");
      }
    }
    ZeroPaddedIdentityMatrix *clone() const override {
      return new ZeroPaddedIdentityMatrix(*this);
    }
    int nrow() const override { return nrow_; }
    int ncol() const override { return ncol_; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_rows(lhs.size());
      conforms_to_cols(rhs.size());
      for (int i = 0; i < ncol_; ++i) {
        lhs[i] = rhs[i];
      }
      for (int i = ncol_; i < lhs.size(); ++i) {
        lhs[i] = 0.0;
      }
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      conforms_to_rows(lhs.size());
      conforms_to_cols(rhs.size());
      for (int i = 0; i < ncol_; ++i) {
        lhs[i] += rhs[i];
      }
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_cols(lhs.size());
      conforms_to_rows(rhs.size());
      for (int i = 0; i < ncol_; ++i) {
        lhs[i] = rhs[i];
      }
    }

    void multiply_inplace(VectorView x) const override {
      report_error("multiply_inplace only applies to square matrices.");
    }

    void matrix_multiply_inplace(SubMatrix x) const override {
      report_error("matrix_multiply_inplace only applies to square matrices.");
    }

    void matrix_transpose_premultiply_inplace(SubMatrix x) const override {
      report_error(
          "matrix_transpose_premultiply_inplace only applies "
          "to square matrices.");
    }

    SpdMatrix inner() const override {
      return SpdMatrix(ncol(), 1.0);
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      if (weights.size() != nrow()) {
        report_error("Wrong size weight vector.");
      }
      SpdMatrix ans(ncol(), 0.0);
      ans.diag() = ConstVectorView(weights, 0, ncol());
      return ans;
    }

    void add_to_block(SubMatrix m) const override {
      conforms_to_rows(m.nrow());
      conforms_to_cols(m.ncol());
      m.diag() += 1.0;
    }

    Matrix dense() const override {
      Matrix ans(nrow_, ncol_, 0.0);
      ans.diag() = 1.0;
      return ans;
    }

   private:
    int nrow_;
    int ncol_;
  };

  //============================================================================
  // A diagonal matrix that is zero in all but (at most) one element.
  class SingleSparseDiagonalElementMatrixBase : public SparseMatrixBlock {
   public:
    SingleSparseDiagonalElementMatrixBase(int dim, int which_element)
        : dim_(dim), which_element_(which_element) {}
    virtual double value() const = 0;

    void set_element(int which_element) { which_element_ = which_element; }

    int nrow() const override { return dim_; }
    int ncol() const override { return dim_; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_rows(lhs.size());
      conforms_to_cols(rhs.size());
      lhs = 0;
      lhs[which_element_] = value() * rhs[which_element_];
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      conforms_to_rows(lhs.size());
      conforms_to_cols(rhs.size());
      lhs[which_element_] += value() * rhs[which_element_];
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      // Symmetric
      multiply(lhs, rhs);
    }
    void multiply_inplace(VectorView x) const override {
      conforms_to_cols(x.size());
      double tmp_value = x[which_element_] * value();
      x = 0;
      x[which_element_] = tmp_value;
    }

    SpdMatrix inner() const override {
      SpdMatrix ans(ncol(), 0.0);
      ans(which_element_, which_element_) = square(value());
      return ans;
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      if (weights.size() != nrow()) {
        report_error("Wrong size weight vector.");
      }
      SpdMatrix ans(ncol(), 0.0);
      ans(which_element_, which_element_) =
          square(value()) * weights[which_element_];
      return ans;
    }

    void add_to_block(SubMatrix block) const override {
      check_can_add(block);
      block(which_element_, which_element_) += value();
    }

   private:
    int dim_;
    int which_element_;
  };

  class SingleSparseDiagonalElementMatrix
      : public SingleSparseDiagonalElementMatrixBase {
   public:
    SingleSparseDiagonalElementMatrix(int dim, double value, int which_element)
        : SingleSparseDiagonalElementMatrixBase(dim, which_element),
          value_(value) {}
    SingleSparseDiagonalElementMatrix *clone() const override {
      return new SingleSparseDiagonalElementMatrix(*this);
    }
    double value() const override { return value_; }
    void set_value(double value) { value_ = value; }

   private:
    double value_;
  };

  class SingleSparseDiagonalElementMatrixParamView
      : public SingleSparseDiagonalElementMatrixBase {
   public:
    SingleSparseDiagonalElementMatrixParamView(int dim,
                                               const Ptr<UnivParams> &value,
                                               int which_element)
        : SingleSparseDiagonalElementMatrixBase(dim, which_element),
          value_(value) {}
    SingleSparseDiagonalElementMatrixParamView *clone() const override {
      return new SingleSparseDiagonalElementMatrixParamView(*this);
    }
    double value() const override { return value_->value(); }

   private:
    Ptr<UnivParams> value_;
  };

  //===========================================================================
  // A possibly rectangular matrix with a single nonzero element in in the first
  // row, with all other elements equal to zero.  This matrix picks a single
  // element from a vector it multiplies, puts it in the first position, scales
  // it, and sets the remaining elements to zero.
  class SingleElementInFirstRow : public SparseMatrixBlock {
   public:
    SingleElementInFirstRow(int nrow, int ncol, int position,
                            double value = 1.0)
        : nrow_(nrow), ncol_(ncol), position_(position), value_(value) {}
    SingleElementInFirstRow *clone() const override {
      return new SingleElementInFirstRow(*this);
    }
    int nrow() const override { return nrow_; }
    int ncol() const override { return ncol_; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_inplace(VectorView x) const override;
    void matrix_multiply_inplace(SubMatrix m) const override;
    void matrix_transpose_premultiply_inplace(SubMatrix m) const override;

    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;
    void add_to_block(SubMatrix block) const override;

   private:
    int nrow_;
    int ncol_;
    int position_;
    double value_;
  };

  //===========================================================================
  // A diagonal matrix whose diagonal entries are zero beyond a certain point.
  // Diagonal entry i is the product of a BOOM::UnivParams and a constant scalar
  // factor.  Interesting special cases that can be handled include
  //  *) The entire diagonal is nonzero.
  //  *) All scale factors are 1.
  class UpperLeftDiagonalMatrix : public SparseMatrixBlock {
   public:
    UpperLeftDiagonalMatrix(const std::vector<Ptr<UnivParams>> &diagonal,
                            int dim)
        : diagonal_(diagonal),
          dim_(dim),
          constant_scale_factor_(diagonal.size(), 1.0) {
      check_diagonal_dimension(dim_, diagonal_);
      check_scale_factor_dimension(diagonal, constant_scale_factor_);
    }

    // Args:
    //   diagonal: The vector of parameter values forming the nonzero part of
    //     the matrix diagonal.
    //   dim:  The number of (notional) rows and colums in the matrix.
    //   scale_factor: A constant vector of numbers multiplying the diagonal
    //     elements to produce the matrix diagonal.
    UpperLeftDiagonalMatrix(const std::vector<Ptr<UnivParams>> &diagonal,
                            int dim, const Vector &scale_factor)
        : diagonal_(diagonal), dim_(dim), constant_scale_factor_(scale_factor) {
      check_diagonal_dimension(dim_, diagonal_);
      check_scale_factor_dimension(diagonal_, constant_scale_factor_);
    }

    UpperLeftDiagonalMatrix *clone() const override {
      return new UpperLeftDiagonalMatrix(*this);
    }
    int nrow() const override { return dim_; };
    int ncol() const override { return dim_; }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      for (int i = 0; i < diagonal_.size(); ++i) {
        lhs[i] = rhs[i] * diagonal_[i]->value() * constant_scale_factor_[i];
      }
      for (int i = diagonal_.size(); i < dim_; ++i) lhs[i] = 0;
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      for (int i = 0; i < diagonal_.size(); ++i) {
        lhs[i] += rhs[i] * diagonal_[i]->value() * constant_scale_factor_[i];
      }
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      multiply(lhs, rhs);
    }
    void multiply_inplace(VectorView x) const override {
      conforms_to_cols(x.size());
      for (int i = 0; i < diagonal_.size(); ++i) {
        x[i] *= diagonal_[i]->value() * constant_scale_factor_[i];
      }
      for (int i = diagonal_.size(); i < dim_; ++i) x[i] = 0;
    }

    SpdMatrix inner() const override {
      SpdMatrix ans(ncol(), 0.0);
      for (int i = 0; i < diagonal_.size(); ++i) {
        ans(i, i) = square(diagonal_[i]->value() * constant_scale_factor_[i]);
      }
      return ans;
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      if (weights.size() != nrow()) {
        report_error("Wrong size weight vector.");
      }
      SpdMatrix ans(ncol(), 0.0);
      for (int i = 0; i < diagonal_.size(); ++i) {
        ans(i, i) = square(diagonal_[i]->value() * constant_scale_factor_[i])
            * weights[i];
      }
      return ans;
    }

    void add_to_block(SubMatrix block) const override {
      conforms_to_rows(block.nrow());
      conforms_to_cols(block.ncol());
      for (int i = 0; i < diagonal_.size(); ++i) {
        block(i, i) += diagonal_[i]->value() * constant_scale_factor_[i];
      }
    }

   private:
    std::vector<Ptr<UnivParams>> diagonal_;
    int dim_;
    Vector constant_scale_factor_;

    void check_diagonal_dimension(
        int dim, const std::vector<Ptr<UnivParams>> &diagonal) {
      if (dim < diagonal.size()) {
        report_error(
            "dim must be at least as large as diagonal in "
            "constructor for UpperLeftDiagonalMatrix");
      }
    }

    void check_scale_factor_dimension(
        const std::vector<Ptr<UnivParams>> &diagonal,
        const Vector &scale_factor) {
      if (diagonal.size() != scale_factor.size()) {
        report_error(
            "diagonal and scale_factor must be the same size in "
            "constructor for UpperLeftDiagonalMatrix");
      }
    }
  };

  //======================================================================
  // A matrix with K identical rows, represented by a single SparseVector.
  class IdenticalRowsMatrix : public SparseMatrixBlock {
   public:
    IdenticalRowsMatrix(const SparseVector &row, int nrows)
        : row_(row), dense_row_(row_.dense()), nrow_(nrows) {}
    IdenticalRowsMatrix *clone() const override {
      return new IdenticalRowsMatrix(*this);
    }
    int nrow() const override { return nrow_; }
    int ncol() const override { return row_.size(); }
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      lhs = row_.dot(rhs);
    }
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      conforms_to_cols(rhs.size());
      conforms_to_rows(lhs.size());
      lhs += row_.dot(rhs);
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      conforms_to_cols(lhs.size());
      conforms_to_rows(rhs.size());
      lhs = dense_row_ * rhs.sum();
    }

    void multiply_inplace(VectorView x) const override {
      if (nrow() == ncol()) {
        conforms_to_cols(x.size());
        multiply(x, x);
      } else {
        report_error("multiply_inplace only works for square matrices.");
      }
    }

    SpdMatrix inner() const override {
      return nrow_ * outer(dense_row_);
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      if (weights.size() != nrow()) {
        report_error("Wrong size weight vector.");
      }
      return sum(weights) * outer(dense_row_);
    }

    void add_to_block(SubMatrix block) const override {
      conforms_to_cols(block.ncol());
      conforms_to_rows(block.nrow());
      for (int i = 0; i < nrow(); ++i) {
        block.row(i) += dense_row_;
      }
    }

   private:
    SparseVector row_;
    Vector dense_row_;
    int nrow_;
  };

  //===========================================================================
  // I - 11^T / dim.  This matrix removes the mean from a vector.  It is
  // symmetric and idempotent.
  class EffectConstraintMatrix : public SparseMatrixBlock {
   public:
    explicit EffectConstraintMatrix(int dim) : dim_(dim) {}
    EffectConstraintMatrix *clone() const override {
      return new EffectConstraintMatrix(*this);
    }
    int nrow() const override {return dim_;}
    int ncol() const override {return dim_;}
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      lhs = rhs;
      lhs -= mean(lhs);
    }
    void multiply_and_add(VectorView lhs, const ConstVectorView &rhs) const override {
      lhs += rhs - mean(rhs);
    }
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override {
      multiply(lhs, rhs);
    }
    void multiply_inplace(VectorView x) const override {
      x -= mean(x);
    }

    void add_to_block(SubMatrix block) const override {
      conforms_to_rows(block.nrow());
      conforms_to_cols(block.ncol());
      block.diag() += 1.0;
      block -= 1.0 / dim_;
    }

    // (I - 11t/s) * (I - 11t/s)
    // = I - 11t/s - 11t/s + 11t * 11t / s^2)
    // = I + (-1/s -1/s + s/s^2)11t
    // = I - 11t/s
    SpdMatrix inner() const override {
      return dense();
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      return dense().inner(weights);
    }

    Matrix dense() const override {
      Matrix ans(dim_, dim_, -1.0 / dim_);
      ans.diag() += 1.0;
      return ans;
    }

   private:
    int dim_;
  };

  //===========================================================================
  // The product of a SparseMatrixBlock and the matrix I - 11'/S, where I is the
  // identity matrix, 1 is a matrix of 1's, and everything is dimension S.  This
  // matrix takes a vector and subtracts the mean from each element.
  class EffectConstrainedMatrixBlock
      : public SparseMatrixBlock {
   public:
    explicit EffectConstrainedMatrixBlock(
        const Ptr<SparseMatrixBlock> &unconstrained)
        : unconstrained_(unconstrained) {}

    EffectConstrainedMatrixBlock(const EffectConstrainedMatrixBlock &rhs)
        : SparseMatrixBlock(rhs),
          unconstrained_(rhs.unconstrained_->clone()) {}
    EffectConstrainedMatrixBlock & operator=(const EffectConstrainedMatrixBlock &rhs) {
      if (&rhs != this) {
        unconstrained_ = rhs.unconstrained_->clone();
      }
      return *this;
    }

    EffectConstrainedMatrixBlock & operator=(
        EffectConstrainedMatrixBlock &&rhs) = default;

    EffectConstrainedMatrixBlock(EffectConstrainedMatrixBlock &&rhs) = default;

    EffectConstrainedMatrixBlock *clone() const override {
      return new EffectConstrainedMatrixBlock(*this);
    }

    int nrow() const override {return unconstrained_->nrow();}
    int ncol() const override {return unconstrained_->ncol();}
    void multiply(VectorView lhs, const ConstVectorView &rhs) const override {
      unconstrained_->multiply(lhs, rhs - mean(rhs));
    }

    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override {
      // lhs += this * rhs
      Vector old_lhs = lhs;
      multiply(lhs, rhs);
      lhs += old_lhs;
    }

    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override{
      unconstrained_->Tmult(lhs, rhs);
      lhs -= mean(lhs);
    }

    void multiply_inplace(VectorView x) const override {
      x -= mean(x);
      unconstrained_->multiply_inplace(x);
    }

    SpdMatrix inner() const override {
      SpdMatrix ans(unconstrained_->inner());
      for (int i = 0; i < ans.nrow(); ++i) {
        ans.row(i) -= mean(ans.row(i));
      }
      for (int i = 0; i < ans.ncol(); ++i) {
        ans.col(i) -= mean(ans.col(i));
      }
      return ans;
    }

    SpdMatrix inner(const ConstVectorView &weights) const override {
      return EffectConstraintMatrix(ncol()).sandwich_transpose(
          unconstrained_->inner(weights));
    }

    Matrix dense() const override {
      return (*this) * SpdMatrix(ncol(), 1.0);
    }

    void add_to_block(SubMatrix block) const override {
      block += dense();
    }

   private:
    Ptr<SparseMatrixBlock> unconstrained_;
  };

  //===========================================================================
  // A sparse matrix whose rows and columns are sparse vectors.  This matrix is
  // somewhat expensive to construct, because a mapping must be constructed
  // between its row and column representations.
  class GenericSparseMatrixBlock;
  class GenericSparseMatrixBlockElementProxy {
   public:
    GenericSparseMatrixBlockElementProxy(int row, int col, double value,
                                         GenericSparseMatrixBlock *matrix)
        : row_(row), col_(col), value_(value), matrix_(matrix) {}
    explicit operator double() const { return value_; }
    GenericSparseMatrixBlockElementProxy &operator=(double new_value);

   private:
    int row_;
    int col_;
    double value_;
    GenericSparseMatrixBlock *matrix_;
  };

  class GenericSparseMatrixBlock : public SparseMatrixBlock {
   public:
    explicit GenericSparseMatrixBlock(int nrow = 0, int ncol = 0);
    GenericSparseMatrixBlock *clone() const override {
      return new GenericSparseMatrixBlock(*this);
    }

    // Element access.
    GenericSparseMatrixBlockElementProxy operator()(int row, int col);
    double operator()(int row, int col) const;

    // Set a specific row or column of the sparse matrix.  The dimensions of the
    // inserted row (column) must match ncol() (nrow()).
    void set_row(const SparseVector &row, int row_number);
    void set_column(const SparseVector &column, int col_number);

    int nrow() const override { return nrow_; }
    int ncol() const override { return ncol_; }

    void multiply(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_and_add(VectorView lhs,
                          const ConstVectorView &rhs) const override;
    void Tmult(VectorView lhs, const ConstVectorView &rhs) const override;
    void multiply_inplace(VectorView x) const override;

    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;

    void add_to_block(SubMatrix block) const override;

    void insert_element(uint row, uint col, double value) {
      insert_element_in_rows(row, col, value);
      insert_element_in_columns(row, col, value);
    }

    const SparseVector &row(int row_number) const;
    const SparseVector &column(int col_number) const;

   private:
    void insert_element_in_rows(uint row, uint col, double value);
    void insert_element_in_columns(uint row, uint col, double value);

    // The notional dimensions of the sparse matrix.
    int nrow_;
    int ncol_;

    // The size of the rows_ map, which may be smaller than nrow_.  This is an
    // optimization, because the standard says that rows_.size() has linear
    // complexity.  A separate entry for the size of the columns_ map is not
    // needed.
    int nrow_compressed_;

    // Data stored in row-major format.
    std::map<uint, SparseVector> rows_;

    // Identical data stored in column major format (for transpose operations).
    std::map<uint, SparseVector> columns_;

    SparseVector empty_row_;
    SparseVector empty_column_;
  };

  //======================================================================
  // A matrix formed by stacking a set of GlmCoefs.
  class StackedRegressionCoefficients : public SparseKalmanMatrix {
   public:
    const GlmCoefs &coefficients(int i) const { return *coefficients_[i]; }
    void add_row(const Ptr<GlmCoefs> &beta);

    int nrow() const override {return coefficients_.size();}
    int ncol() const override {
      if (coefficients_.empty()) {
        return 0;
      } else {
        return coefficients_[0]->nvars_possible();
      }
    }

    Vector operator*(const Vector &v) const override;
    Vector operator*(const VectorView &v) const override;
    Vector operator*(const ConstVectorView &v) const override;

    // Expose the matrix-matrix multiplication operator from the base class.
    using SparseKalmanMatrix::operator*;

    using SparseKalmanMatrix::Tmult;
    Vector Tmult(const ConstVectorView &x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;

    Matrix &add_to(Matrix &P) const override;
    SubMatrix add_to_submatrix(SubMatrix P) const override;

   private:
    // Each coefficient vector is one row in the matrix.
    std::vector<Ptr<GlmCoefs>> coefficients_;
  };

  //======================================================================
  // The state transition equation for a dynamic linear model will
  // typically involve a block diagonal matrix.  The blocks will
  // typically be:  SeasonalStateSpaceMatrix, IdentityMatrix, etc.
  class BlockDiagonalMatrix : public SparseKalmanMatrix {
   public:
    // Start off with an empty matrix.  Use add_block() to add blocks
    // Adds a block to the block diagonal matrix
    BlockDiagonalMatrix();

    void add_block(const Ptr<SparseMatrixBlock> &m);
    void replace_block(int which_block, const Ptr<SparseMatrixBlock> &b);
    void clear();

    int nrow() const override;
    int ncol() const override;

    Vector operator*(const Vector &v) const override;
    Vector operator*(const VectorView &v) const override;
    Vector operator*(const ConstVectorView &v) const override;

    // The 'using' statement on the following line exposes the matrix-matrix
    // multiplication operator from the base class.
    using SparseKalmanMatrix::operator*;

    using SparseKalmanMatrix::Tmult;
    Vector Tmult(const ConstVectorView &x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;

    // P -> this * P * this.transpose()
    void sandwich_inplace(SpdMatrix &P) const override;
    void sandwich_inplace_submatrix(SubMatrix P) const override;

    // sandwich(P) = this * P * this.transpose()
    SpdMatrix sandwich(const SpdMatrix &P) const override;

    Matrix &add_to(Matrix &P) const override;
    SubMatrix add_to_submatrix(SubMatrix P) const override;

   private:
    // Replace middle with left * middle * right.transpose()
    void sandwich_inplace_block(const SparseMatrixBlock &left,
                                const SparseMatrixBlock &right,
                                SubMatrix middle) const;

    // Returns the (i,j) block of the matrix m, with block sizes
    // determined by the rows and columns of the entries in blocks_.
    SubMatrix get_block(Matrix &m, int i, int j) const;
    SubMatrix get_submatrix_block(SubMatrix m, int i, int j) const;
    std::vector<Ptr<SparseMatrixBlock>> blocks_;

    int nrow_;
    int ncol_;

    // row_boundaries_[i] contains the one-past-the-end position of the upper
    // row boundary of block i.
    std::vector<int> row_boundaries_;

    // col_boundaries_[i] contains the one-past-the-end position of the upper
    // column boundary of block i.
    std::vector<int> col_boundaries_;
  };
  //============================================================================
  // A SparseKalmanMatrix made of blocks that form vertical strips (analogous to
  // cbind in R):  [B1 B2 B3...].
  class SparseVerticalStripMatrix : public SparseKalmanMatrix {
   public:
    SparseVerticalStripMatrix() : ncol_(0) {}
    int nrow() const override {
      return blocks_.empty() ? 0 : blocks_[0]->nrow();
    }
    int ncol() const override { return ncol_; }

    void clear() {
      blocks_.clear();
      ncol_ = 0;
    }

    void add_block(const Ptr<SparseMatrixBlock> &block);

    Vector operator*(const Vector &v) const override;
    Vector operator*(const VectorView &v) const override;
    Vector operator*(const ConstVectorView &v) const override;

    Vector Tmult(const ConstVectorView &v) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;

    // P += *this
    Matrix &add_to(Matrix &P) const override;
    SubMatrix add_to_submatrix(SubMatrix P) const override;

   private:
    void check_can_multiply(int vector_size) const;
    void check_can_Tmult(int vector_size) const;
    void check_can_add(int rows, int cols) const;
    int ncol_;
    std::vector<Ptr<SparseMatrixBlock>> blocks_;
  };

  //===========================================================================
  // A ErrorExpanderMatrix is a BlockDiagonalMatrix that allows for blocks that
  // have ncol==0.  The effect of such blocks is to add block->nrow() rows of
  // 0's to the dense representation of the matrix.
  class ErrorExpanderMatrix : public SparseKalmanMatrix {
   public:
    ErrorExpanderMatrix();

    // Seasonal state models may change their number of columns over time,
    // because there will be one additional element of "noise" in time periods
    // when a new season begins.  For this reason we don't assume nrow and ncol
    // are fixed.  They get recomputed each time.
    int nrow() const override;
    int ncol() const override;

    void add_block(const Ptr<SparseMatrixBlock> &block);
    void replace_block(int block_index,
                       const Ptr<SparseMatrixBlock> &block);

    // Remove all the blocks, making the matrix empty.
    void clear();

    Vector operator*(const Vector &v) const override;
    Vector operator*(const VectorView &v) const override;
    Vector operator*(const ConstVectorView &v) const override;

    // Use the base class matrix-matrix multiply operations.
    using SparseKalmanMatrix::operator*;
    using SparseKalmanMatrix::Tmult;

    Vector Tmult(const ConstVectorView &x) const override;
    SpdMatrix inner() const override;
    SpdMatrix inner(const ConstVectorView &weights) const override;

    // These operations will raise errors if the ErrorExpanderMatrix is not
    // square.
    void sandwich_inplace(SpdMatrix &P) const override;
    void sandwich_inplace_submatrix(SubMatrix P) const override;

    // sandwich(P) = this * P * this.transpose()
    SpdMatrix sandwich(const SpdMatrix &P) const override;

    Matrix &add_to(Matrix &P) const override;
    SubMatrix add_to_submatrix(SubMatrix P) const override;

    // Solve the equation rhs = this * lhs.  It may be the case that some matrix
    // blocks are non-invertible, in which case calling this function will
    // report an error.
    //
    // The initial reason for this implementing this method is to evaluate the
    // transition density for state space models when the error distribution is
    // less than full rank.   In those settings we must be able to evaluate
    // error_expander->left_inverse( new_state - T * old_state).
    virtual Vector left_inverse(const ConstVectorView &rhs) const;

   private:
    std::vector<Ptr<SparseMatrixBlock>> blocks_;

    int nrow_;
    int ncol_;
    std::vector<int> row_boundaries_;
    std::vector<int> col_boundaries_;

    void recompute_sizes();
    void increment_sizes(const Ptr<SparseMatrixBlock> &block);
  };

  //===========================================================================

  Vector operator*(const SparseMatrixBlock &, const Vector &);

  // P += TPK * K.transpose * w
  void add_outer_product(SpdMatrix &P, const Vector &TPK, const Vector &K,
                         double w);

  // P += RQR
  void add_block_diagonal(SpdMatrix &P, const BlockDiagonalMatrix &RQR);

}  // namespace BOOM
#endif  // BOOM_SPARSE_MATRIX_HPP_
