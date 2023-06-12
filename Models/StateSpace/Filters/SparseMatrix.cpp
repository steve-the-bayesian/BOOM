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

#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include <utility>
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  void SparseKalmanMatrix::conforms_to_rows(int i) const {
    if (i == nrow()) return;
    std::ostringstream err;
    err << "object of length " << i
        << " does not conform with the number of rows (" << nrow() << ")";
    report_error(err.str());
  }

  void SparseKalmanMatrix::conforms_to_cols(int i) const {
    if (i == ncol()) return;
    std::ostringstream err;
    err << "object of length " << i
        << " does not conform with the number of columns (" << ncol() << ")";
    report_error(err.str());
  }

  void SparseKalmanMatrix::check_can_multiply(int vector_size) const {
    conforms_to_cols(vector_size);
  }

  void SparseKalmanMatrix::check_can_Tmult(int vector_size) const {
    conforms_to_rows(vector_size);
  }

  void SparseKalmanMatrix::check_can_add(int rows, int cols) const {
    conforms_to_rows(rows);
    conforms_to_cols(cols);
  }

  SparseMatrixProduct::SparseMatrixProduct() {}

  void SparseMatrixProduct::add_term(const Ptr<SparseKalmanMatrix> &term,
                                     bool transpose) {
    check_term(term, transpose);
    terms_.push_back(term);
    transposed_.push_back(transpose);
  }

  int SparseMatrixProduct::nrow() const {
    if (terms_.empty()) {
      return 0;
    }
    return transposed_[0] ? terms_[0]->ncol() : terms_[0]->nrow();
  }

  int SparseMatrixProduct::ncol() const {
    if (terms_.empty()) {
      return 0;
    }
    return transposed_.back() ? terms_.back()->nrow() : terms_.back()->ncol();
  }

  Vector SparseMatrixProduct::operator*(const ConstVectorView &v) const {
    Vector ans = v;
    for (int i = terms_.size() - 1; i >= 0; --i) {
      if (transposed_[i]) {
        ans = terms_[i]->Tmult(ans);
      } else {
        ans = (*terms_[i]) * ans;
      }
    }
    return ans;
  }

  Vector SparseMatrixProduct::operator*(const Vector &rhs) const {
    return (*this) * ConstVectorView(rhs);
  }

  Vector SparseMatrixProduct::operator*(const VectorView &rhs) const {
    return (*this) * ConstVectorView(rhs);
  }

  Matrix SparseMatrixProduct::operator*(const Matrix &rhs) const {
    Matrix ans(rhs);
    for (int i = terms_.size() - 1; i >= 0; --i) {
      if (transposed_[i]) {
        ans = terms_[i]->Tmult(ans);
      } else {
        ans = (*terms_[i]) * ans;
      }
    }
    return ans;
  }

  Vector SparseMatrixProduct::Tmult(const ConstVectorView &rhs) const {
    Vector ans(rhs);
    for (size_t i = 0; i < terms_.size(); ++i) {
      if (transposed_[i]) {
        ans = *terms_[i] * ans;
      } else {
        ans = terms_[i]->Tmult(ans);
      }
    }
    return ans;
  }

  Matrix SparseMatrixProduct::Tmult(const Matrix &rhs) const {
    Matrix ans(rhs);
    for (size_t i = 0; i < terms_.size(); ++i) {
      if (transposed_[i]) {
        ans = *terms_[i] * ans;
      } else {
        ans = terms_[i]->Tmult(ans);
      }
    }
    return ans;
  }

  Ptr<SparseMatrixProduct>
  SparseMatrixProduct::sparse_sandwich(const SpdMatrix &N) const {
    NEW(DenseSpd, SparseN)(N);
    NEW(SparseMatrixProduct, ans)();
    for (int i = terms_.size() - 1; i >= 0; --i) {
      ans->add_term(terms_[i], !transposed_[i]);
    }
    ans->add_term(SparseN);
    for (int i = 0; i < terms_.size(); ++i) {
      ans->add_term(terms_[i], transposed_[i]);
    }
    return ans;
  }

  Ptr<SparseMatrixProduct>
  SparseMatrixProduct::sparse_sandwich_transpose(const SpdMatrix &N) const {
    NEW(DenseSpd, SparseN)(N);
    NEW(SparseMatrixProduct, ans)();
    for (int i = 0; i < terms_.size(); ++i) {
      ans->add_term(terms_[i], transposed_[i]);
    }
    ans->add_term(SparseN);
    for(int i = terms_.size() - 1; i >= 0; --i) {
      ans->add_term(terms_[i], !transposed_[i]);
    }
    return ans;
  }

  Vector SparseMatrixProduct::diag() const {
    int m = std::min(nrow(), ncol());
    Vector ans(m);
    for (int i = 0; i < m; ++i) {
      Vector one(ncol(), 0.0);
      one[i] = 1.0;
      ans[i] = ((*this) * one)[i];
    }
    return ans;
  }

  SpdMatrix SparseMatrixProduct::inner() const {
    SpdMatrix ans(nrow(), 1.0);
    for (int i = 0; i < terms_.size(); ++i) {
      if (transposed_[i]) {
        ans = terms_[i]->sandwich(ans);
      } else {
        ans = terms_[i]->sandwich_transpose(ans);
      }
    }
    return ans;
  }

  SpdMatrix SparseMatrixProduct::inner(const ConstVectorView &weights) const {
    SpdMatrix ans(weights.size());
    ans.diag() = weights;

    for (int i = 0; i < terms_.size(); ++i) {
      if (transposed_[i]) {
        ans = terms_[i]->sandwich(ans);
      } else {
        ans = terms_[i]->sandwich_transpose(ans);
      }
    }
    return ans;
  }

  Matrix &SparseMatrixProduct::add_to(Matrix &P) const {
    P += this->dense();
    return P;
  }

  Matrix SparseMatrixProduct::dense() const {
    SpdMatrix Id(ncol(), 1.0);
    return (*this) * Id;
  }

  void SparseMatrixProduct::check_term(const Ptr<SparseKalmanMatrix> &term,
                                       bool transpose) {
    if (terms_.empty()) {
      return;
    }
    bool final_transpose = transposed_.back();
    const Ptr<SparseKalmanMatrix>& final_term(terms_.back());
    size_t final_dim =
        final_transpose ? final_term->nrow() : final_term->ncol();
    size_t leading_dim = transpose ? term->ncol() : term-> nrow();
    if (final_dim != leading_dim) {
      std::ostringstream err;
      err << "Incompatible matrix following term " << terms_.size()
          << ".  Final dimension of previous term: "
          << final_dim
          << ".  Leading dimension of new term: " << leading_dim
          << ".";
      report_error(err.str());
    }
  }

  //===========================================================================

  SparseMatrixSum::SparseMatrixSum() {}

  void SparseMatrixSum::add_term(
      const Ptr<SparseKalmanMatrix> &term, double coefficient) {
    if (!terms_.empty()) {
      if (term->nrow() != terms_.back()->nrow()
          || term->ncol() != terms_.back()->ncol()) {
        report_error("Incompatible sparse matrices in sum.");
      }
    }
    terms_.push_back(term);
    coefficients_.push_back(coefficient);
  }

  int SparseMatrixSum::nrow() const {
    if (terms_.empty()) {
      return 0;
    }
    return terms_.back()->nrow();
  }

  int SparseMatrixSum::ncol() const {
    if (terms_.empty()) {
      return 0;
    }
    return terms_.back()->ncol();
  }

  Vector SparseMatrixSum::operator*(const ConstVectorView &rhs) const {
    Vector ans(nrow(), 0.0);
    for (int i = 0; i < terms_.size(); ++i) {
      ans += coefficients_[i] * ((*terms_[i]) * rhs);
    }
    return ans;
  }

  Vector SparseMatrixSum::operator*(const Vector &rhs) const {
    return (*this) * ConstVectorView(rhs);
  }

  Vector SparseMatrixSum::operator*(const VectorView &rhs) const {
    return (*this) * ConstVectorView(rhs);
  }

  Matrix SparseMatrixSum::operator*(const Matrix &rhs) const {
    Matrix ans(nrow(), rhs.ncol(), 0.0);
    for (int i = 0; i < terms_.size(); ++i) {
      ans += coefficients_[i] * ((*terms_[i]) * rhs);
    }
    return ans;
  }

  Vector SparseMatrixSum::Tmult(const ConstVectorView &rhs) const {
    Vector ans(ncol());
    for (int i = 0; i < terms_.size(); ++i) {
      ans += coefficients_[i] * terms_[i]->Tmult(rhs);
    }
    return ans;
  }

  Matrix SparseMatrixSum::Tmult(const Matrix &rhs) const {
    Matrix ans(ncol(), rhs.ncol());
    for (int i = 0; i < terms_.size(); ++i) {
      ans += coefficients_[i] * terms_[i]->Tmult(rhs);
    }
    return ans;
  }

  SpdMatrix SparseMatrixSum::inner() const {
    return dense().inner();
  }

  SpdMatrix SparseMatrixSum::inner(const ConstVectorView &weights) const {
    return dense().inner(weights);
  }

  Matrix &SparseMatrixSum::add_to(Matrix &rhs) const {
    for (int i = 0; i < terms_.size(); ++i) {
      if (coefficients_[i] != 0.0) {
        rhs /= coefficients_[i];
        terms_[i]->add_to(rhs);
        rhs *= coefficients_[i];
      }
    }
    return rhs;
  }

  //===========================================================================
  SparseWoodburyInverse::SparseWoodburyInverse(
      const Ptr<SparseKalmanMatrix> &Ainv,
      double logdet_Ainv,
      const Ptr<SparseKalmanMatrix> &U,
      const SpdMatrix &Cinv)
      : Ainv_(Ainv),
        U_(U)
  {
    inner_matrix_ = U_->Tmult(*Ainv_ * U_->dense());
    if (Cinv.nrow() > 0) {
      inner_matrix_ += Cinv;
    } else {
      // If Cinv is empty then it is assumed to be the identity.
      inner_matrix_.diag() += 1.0;
    }
    inner_matrix_condition_number_ = inner_matrix_.condition_number();

    inner_matrix_ = inner_matrix_.inv();

    // From the Matrix Determinant lemma det(A + UCU') = det(Cinv + U'AinvU) *
    // det(C) * det(A).  The determinant of the inverse is the reciprocal of the
    // determinant.  Now take logs.
    logdet_ = inner_matrix_.logdet() + logdet_Ainv;
    if (Cinv.nrow() > 0) {
      logdet_ += Cinv.logdet();
    }
    // If Cinv is empty (and thus implicitly the identity) then its log
    // determinant is zero.
  }

  SparseWoodburyInverse::SparseWoodburyInverse(
      const Ptr<SparseKalmanMatrix> &Ainv,
      const Ptr<SparseKalmanMatrix> &U,
      const SpdMatrix &inner_matrix,
      double logdet,
      double condition_number)
      : Ainv_(Ainv),
        U_(U),
        inner_matrix_(inner_matrix),
        logdet_(logdet),
        inner_matrix_condition_number_(condition_number)
  {
    if (inner_matrix_.nrow() == 0 || inner_matrix_.ncol() == 0) {
      report_error("inner_matrix_ must have positive dimension.");
    }
  }

  Vector SparseWoodburyInverse::operator*(const ConstVectorView &rhs) const {
    Vector Ar = *Ainv_ * rhs;
    return Ar - *Ainv_ * (*U_ * (inner_matrix_ * (U_->Tmult(Ar))));
  }

  Vector SparseWoodburyInverse::operator*(const Vector &rhs) const {
    return (*this) * ConstVectorView(rhs);
  }

  Vector SparseWoodburyInverse::operator*(const VectorView &rhs) const {
    return (*this) * ConstVectorView(rhs);
  }

  Matrix SparseWoodburyInverse::operator*(const Matrix &rhs) const {
    Matrix Ar = *Ainv_ * rhs;
    return Ar - *Ainv_ * (*U_ * (inner_matrix_ * (U_->Tmult(Ar))));
  }

  Vector SparseWoodburyInverse::Tmult(const ConstVectorView &rhs) const {
    return (*this) * rhs;
  }

  Matrix SparseWoodburyInverse::Tmult(const Matrix &rhs) const {
    return (*this) * rhs;
  }

  Matrix &SparseWoodburyInverse::add_to(Matrix &rhs) const {
    rhs += this->dense();
    return rhs;
  }

  SpdMatrix SparseWoodburyInverse::inner() const {
    return this->dense().inner();
  }

  SpdMatrix SparseWoodburyInverse::inner(const ConstVectorView &weights) const {
    return this->dense().inner(weights);
  }

  double SparseWoodburyInverse::logdet() const {
    return logdet_;
  }

  Matrix SparseWoodburyInverse::dense() const {
    SpdMatrix I(ncol(), 1.0);
    return (*this) * I;
  }

  //===========================================================================

  SparseBinomialInverse::SparseBinomialInverse(
      const Ptr<SparseKalmanMatrix> &Ainv,
      const Ptr<SparseKalmanMatrix> &U,
      const SpdMatrix &B,
      double Ainv_logdet)
      : Ainv_(Ainv),
        U_(U),
        B_(B)
  {
    SparseMatrixProduct tmp;
    tmp.add_term(U, true);
    tmp.add_term(Ainv);
    tmp.add_term(U);

    inner_matrix_ = SpdMatrix(B.nrow(), 1.0);
    inner_matrix_ += B * tmp.dense();

    inner_matrix_condition_number_ =
        inner_matrix_.condition_number();

    if (okay()) {
      inner_matrix_ = inner_matrix_.inv();
      logdet_ = Ainv_logdet + inner_matrix_.logdet();
    } else {
      logdet_ = negative_infinity();
      inner_matrix_ = SpdMatrix();
    }
  }

  SparseBinomialInverse::SparseBinomialInverse(
      const Ptr<SparseKalmanMatrix> &Ainv,
      const Ptr<SparseKalmanMatrix> &U,
      const SpdMatrix &B,
      const Matrix &inner,
      double logdet,
      double inner_matrix_condition_number)
      : Ainv_(Ainv),
        U_(U),
        B_(B),
        inner_matrix_(inner),
        logdet_(logdet),
        inner_matrix_condition_number_(inner_matrix_condition_number)
  {}

  Vector SparseBinomialInverse::operator*(const ConstVectorView &rhs) const {
    throw_if_not_okay();
    Vector ans = (*Ainv_) * rhs;
    ans -= (*Ainv_) * (*U_ * (inner_matrix_ * (B_ * (U_->Tmult(*Ainv_ * rhs)))));
    return ans;
  }

  Vector SparseBinomialInverse::operator*(const Vector &rhs) const {
    throw_if_not_okay();
    return (*this) * ConstVectorView(rhs);
  }

  Vector SparseBinomialInverse::operator*(const VectorView &rhs) const {
    throw_if_not_okay();
    return (*this) * ConstVectorView(rhs);
  }

  Matrix SparseBinomialInverse::operator*(const Matrix &rhs) const {
    throw_if_not_okay();
    Matrix ans = *Ainv_ * rhs;
    ans -= *Ainv_ * (*U_ * (inner_matrix_ * (B_ * (U_->Tmult(*Ainv_ * rhs)))));
    return ans;
  }

  Vector SparseBinomialInverse::Tmult(const ConstVectorView &rhs) const {
    throw_if_not_okay();
    return (*this) * rhs;
  }

  Matrix SparseBinomialInverse::Tmult(const Matrix &rhs) const {
    throw_if_not_okay();
    return (*this) * rhs;
  }

  Matrix SparseBinomialInverse::dense() const {
    throw_if_not_okay();
    SpdMatrix I(ncol(), 1.0);
    return (*this) * I;
  }

  Matrix & SparseBinomialInverse::add_to(Matrix &rhs) const {
    throw_if_not_okay();
    rhs += this->dense();
    return rhs;
  }

  SpdMatrix SparseBinomialInverse::inner() const {
    throw_if_not_okay();
    return this->dense().inner();
  }

  SpdMatrix SparseBinomialInverse::inner(const ConstVectorView &weights) const {
    throw_if_not_okay();
    return this->dense().inner(weights);
  }

  double SparseBinomialInverse::logdet() const {
    throw_if_not_okay();
    return logdet_;
  }

  bool SparseBinomialInverse::okay() const {
    return inner_matrix_condition_number_ < 1e+8;
  }

  void SparseBinomialInverse::throw_if_not_okay() const {
    if (!okay()) {
      report_error("The condition number of the 'inner matrix' used by "
                   "SparseBinomialInverse was too large.  The caluclation is "
                   "likely invalid.  Please use another method.");
    }
  }
  //===========================================================================

  namespace {
    template <class VECTOR>
    Vector sparse_multiply_impl(const SparseMatrixBlock &m, const VECTOR &v) {
      m.conforms_to_cols(v.size());
      Vector ans(m.nrow(), 0.0);
      m.multiply(VectorView(ans), v);
      return ans;
    }
  }  // namespace

  void SparseMatrixBlock::check_can_multiply(
      const VectorView &lhs, const ConstVectorView &rhs) const {
    if (lhs.size() != nrow()) {
      report_error("Left hand side is the wrong dimension.");
    }
    if (rhs.size() != ncol()) {
      report_error("Right hand side is the wrong dimension.");
    }
  }

  Vector SparseMatrixBlock::operator*(const Vector &v) const {
    return sparse_multiply_impl(*this, v);
  }
  Vector SparseMatrixBlock::operator*(const VectorView &v) const {
    return sparse_multiply_impl(*this, v);
  }
  Vector SparseMatrixBlock::operator*(const ConstVectorView &v) const {
    return sparse_multiply_impl(*this, v);
  }
  Matrix SparseMatrixBlock::operator*(const Matrix &rhs) const {
    conforms_to_cols(rhs.nrow());
    Matrix ans(nrow(), rhs.ncol());
    for (int j = 0; j < rhs.ncol(); ++j) {
      multiply(ans.col(j), rhs.col(j));
    }
    return ans;
  }

  Vector SparseMatrixBlock::Tmult(const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    Vector ans(ncol());
    Tmult(VectorView(ans), rhs);
    return ans;
  }

  Matrix SparseMatrixBlock::Tmult(const Matrix &rhs) const {
    conforms_to_rows(rhs.nrow());
    Matrix ans(ncol(), rhs.ncol());
    for (int j = 0; j < ans.ncol(); ++j) {
      Tmult(ans.col(j), rhs.col(j));
    }
    return ans;
  }

  void SparseKalmanMatrix::check_can_add(const SubMatrix &block) const {
    if (block.nrow() != nrow() || block.ncol() != ncol()) {
      std::ostringstream err;
      err << "cant add SparseMatrix to SubMatrix: rows and columnns "
          << "are incompatible" << endl
          << "this->nrow() = " << nrow() << endl
          << "this->ncol() = " << ncol() << endl
          << "that.nrow()  = " << block.nrow() << endl
          << "that.ncol()  = " << block.ncol() << endl;
      report_error(err.str());
    }
  }

  void SparseMatrixBlock::matrix_multiply_inplace(SubMatrix m) const {
    for (int i = 0; i < m.ncol(); ++i) {
      multiply_inplace(m.col(i));
    }
  }

  Matrix SparseMatrixBlock::dense() const {
    if (nrow() == ncol()) {
      Matrix ans(nrow(), ncol(), 0.0);
      ans.diag() = 1.0;
      matrix_multiply_inplace(SubMatrix(ans));
      return ans;
    } else {
      return *this * SpdMatrix(ncol(), 1.0);
    }
  }

  void SparseMatrixBlock::matrix_transpose_premultiply_inplace(
      SubMatrix m) const {
    for (int i = 0; i < m.nrow(); ++i) {
      multiply_inplace(m.row(i));
    }
  }

  Matrix & SparseMatrixBlock::add_to(Matrix &P) const {
    add_to_block(SubMatrix(P));
    return P;
  }

  Matrix SparseKalmanMatrix::dense() const {
    Matrix ans(nrow(), ncol(), 0.0);
    add_to(ans);
    return ans;
  }

  Vector SparseMatrixBlock::left_inverse(const ConstVectorView &x) const {
    report_error("'left_inverse' called for a SparseMatrixBlock that didn't "
                 "define the operation.");
    return Vector(0);
  }

  //======================================================================
  BlockDiagonalMatrixBlock::BlockDiagonalMatrixBlock(
      const BlockDiagonalMatrixBlock &rhs)
      : dim_(0) {
    for (int i = 0; i < rhs.blocks_.size(); ++i) {
      add_block(rhs.blocks_[i]->clone());
    }
  }

  BlockDiagonalMatrixBlock *BlockDiagonalMatrixBlock::clone() const {
    return new BlockDiagonalMatrixBlock(*this);
  }

  BlockDiagonalMatrixBlock &BlockDiagonalMatrixBlock::operator=(
      const BlockDiagonalMatrixBlock &rhs) {
    if (this != &rhs) {
      blocks_.clear();
      for (int i = 0; i < rhs.blocks_.size(); ++i) {
        add_block(rhs.blocks_[i]->clone());
      }
    }
    return *this;
  }

  void BlockDiagonalMatrixBlock::add_block(
      const Ptr<SparseMatrixBlock> &block) {
    if (!block) {
      report_error("nullptr argument passed to BlockDiagonalMatrixBlock::"
                   "add_block");
    }
    if (block->nrow() != block->ncol()) {
      report_error("Sub-blocks of a BlockDiagonalMatrixBlock must be square.");
    }
    dim_ += block->nrow();
    blocks_.push_back(block);
  }

  void BlockDiagonalMatrixBlock::multiply(VectorView lhs,
                                          const ConstVectorView &rhs) const {
    check_can_multiply(lhs, rhs);
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      VectorView left(lhs, position, local_dim);
      ConstVectorView right(rhs, position, local_dim);
      blocks_[b]->multiply(left, right);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    check_can_multiply(lhs, rhs);
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      VectorView left(lhs, position, local_dim);
      ConstVectorView right(rhs, position, local_dim);
      blocks_[b]->multiply_and_add(left, right);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::Tmult(VectorView lhs,
                                       const ConstVectorView &rhs) const {
    check_can_multiply(lhs, rhs);
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      VectorView left(lhs, position, local_dim);
      ConstVectorView right(rhs, position, local_dim);
      blocks_[b]->Tmult(left, right);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::multiply_inplace(VectorView x) const {
    conforms_to_cols(x.size());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      VectorView local(x, position, local_dim);
      blocks_[b]->multiply_inplace(local);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::matrix_multiply_inplace(SubMatrix m) const {
    conforms_to_cols(m.nrow());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      SubMatrix rows_of_m(m, position, position + local_dim - 1, 0,
                          m.ncol() - 1);
      blocks_[b]->matrix_multiply_inplace(rows_of_m);
      position += local_dim;
    }
  }

  void BlockDiagonalMatrixBlock::matrix_transpose_premultiply_inplace(
      SubMatrix m) const {
    // The number of columns in m must match the number of rows in
    // this->transpose(), which is the same as the number of rows in this.
    conforms_to_cols(m.ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      // We really want the number of rows in blocks_[b].transpose(), but since
      // the matrices are square it does not matter.
      int local_dim = blocks_[b]->ncol();
      SubMatrix m_columns(m, 0, m.nrow() - 1, position,
                          position + local_dim - 1);
      blocks_[b]->matrix_transpose_premultiply_inplace(m_columns);
      position += local_dim;
    }
  }

  SpdMatrix BlockDiagonalMatrixBlock::inner() const {
    SpdMatrix ans(ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->ncol();
      SubMatrix inner_block(ans, position, position + local_dim - 1,
                            position, position + local_dim - 1);
      inner_block = blocks_[b]->inner();
      position += local_dim;
    }
    return ans;
  }

  SpdMatrix BlockDiagonalMatrixBlock::inner(const ConstVectorView &weights) const {
    SpdMatrix ans(ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->ncol();
      const ConstVectorView local_weights(weights, position, local_dim);
      SubMatrix inner_block(ans, position, position + local_dim - 1,
                            position, position + local_dim - 1);
      inner_block = blocks_[b]->inner(local_weights);
      position += local_dim;
    }
    return ans;
  }

  void BlockDiagonalMatrixBlock::add_to_block(SubMatrix block) const {
    conforms_to_rows(block.nrow());
    conforms_to_cols(block.ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      SubMatrix local(block, position, position + local_dim - 1, position,
                      position + local_dim - 1);
      blocks_[b]->add_to_block(local);
      position += local_dim;
    }
  }

  //======================================================================
  StackedMatrixBlock::StackedMatrixBlock(const StackedMatrixBlock &rhs)
      : nrow_(0), ncol_(0) {
    for (int b = 0; b < rhs.blocks_.size(); ++b) {
      add_block(rhs.blocks_[b]->clone());
    }
  }

  StackedMatrixBlock &StackedMatrixBlock::operator=(
      const StackedMatrixBlock &rhs) {
    if (&rhs != this) {
      nrow_ = 0;
      ncol_ = 0;
      blocks_.clear();
      for (int b = 0; b < rhs.blocks_.size(); ++b) {
        add_block(rhs.blocks_[b]->clone());
      }
    }
    return *this;
  }

  void StackedMatrixBlock::clear() {
    blocks_.clear();
    nrow_ = 0;
    ncol_ = 0;
  }

  void StackedMatrixBlock::add_block(const Ptr<SparseMatrixBlock> &block) {
    if (nrow_ == 0) {
      nrow_ = block->nrow();
      ncol_ = block->ncol();
    } else {
      if (block->ncol() != ncol_) {
        report_error(
            "Blocks in a stacked matrix must have the same "
            "number of columns.");
      }
      nrow_ += block->nrow();
    }
    blocks_.push_back(block);
  }

  void StackedMatrixBlock::multiply(VectorView lhs,
                                    const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int nr = blocks_[b]->nrow();
      VectorView view(lhs, position, nr);
      blocks_[b]->multiply(view, rhs);
      position += nr;
    }
  }

  void StackedMatrixBlock::multiply_and_add(VectorView lhs,
                                            const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int nr = blocks_[b]->nrow();
      VectorView view(lhs, position, nr);
      blocks_[b]->multiply_and_add(view, rhs);
      position += nr;
    }
  }

  void StackedMatrixBlock::Tmult(VectorView lhs,
                                 const ConstVectorView &rhs) const {
    conforms_to_cols(lhs.size());
    conforms_to_rows(rhs.size());
    int position = 0;
    lhs = 0;
    Vector workspace(ncol_, 0.0);
    for (int b = 0; b < blocks_.size(); ++b) {
      int stride = blocks_[b]->nrow();
      ConstVectorView view(rhs, position, stride);
      blocks_[b]->Tmult(VectorView(workspace), view);
      lhs += workspace;
      position += stride;
    }
  }

  void StackedMatrixBlock::multiply_inplace(VectorView x) const {
    report_error("multiply_inplace only works for square matrices.");
  }

  SpdMatrix StackedMatrixBlock::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    for (int b = 0; b < blocks_.size(); ++b) {
      ans += blocks_[b]->inner();
    }
    return ans;
  }

  SpdMatrix StackedMatrixBlock::inner(const ConstVectorView &weights) const {
    if (weights.size() != nrow()) {
      report_error("Weight vector was the wrong size.");
    }
    SpdMatrix ans(ncol(), 0.0);
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int local_dim = blocks_[b]->nrow();
      const ConstVectorView local_weights(weights, position, local_dim);
      ans += blocks_[b]->inner(local_weights);
      position += local_dim;
    }
    return ans;
  }

  void StackedMatrixBlock::add_to_block(SubMatrix block) const {
    conforms_to_rows(block.nrow());
    conforms_to_cols(block.ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix lhs_block(block, position, position + blocks_[b]->nrow() - 1,
                          0, ncol_ - 1);
      blocks_[b]->add_to_block(lhs_block);
      position += blocks_[b]->nrow();
    }
  }

  Matrix StackedMatrixBlock::dense() const {
    Matrix ans(nrow(), ncol());
    int position = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix ans_block(ans, position, position + blocks_[b]->nrow() - 1,
                          0, ncol_ - 1);
      ans_block = blocks_[b]->dense();
      position += blocks_[b]->nrow();
    }
    return ans;
  }

  Vector StackedMatrixBlock::left_inverse(const ConstVectorView &x) const {
    SpdMatrix xtx = this->inner();
    Vector xty(ncol(), 0.0);
    this->Tmult(VectorView(xty), x);
    return xtx.solve(xty);
  }

  //======================================================================
  LocalLinearTrendMatrix *LocalLinearTrendMatrix::clone() const {
    return new LocalLinearTrendMatrix(*this);
  }

  void LocalLinearTrendMatrix::multiply(VectorView lhs,
                                        const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] = rhs[0] + rhs[1];
    lhs[1] = rhs[1];
  }

  void LocalLinearTrendMatrix::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] += rhs[0] + rhs[1];
    lhs[1] += rhs[1];
  }

  void LocalLinearTrendMatrix::Tmult(VectorView lhs,
                                     const ConstVectorView &rhs) const {
    conforms_to_cols(lhs.size());
    conforms_to_rows(rhs.size());
    lhs[0] = rhs[0];
    lhs[1] = rhs[0] + rhs[1];
  }

  void LocalLinearTrendMatrix::multiply_inplace(VectorView v) const {
    conforms_to_cols(v.size());
    v[0] += v[1];
  }

  SpdMatrix LocalLinearTrendMatrix::inner() const {
    // 1 0 * 1 1  = 1 1
    // 1 1   0 1    1 2
    SpdMatrix ans(2);
    ans = 1.0;
    ans(1, 1) = 2.0;
    return ans;
  }

  SpdMatrix LocalLinearTrendMatrix::inner(const ConstVectorView &weights) const {
    // 1 0 * w1 0  * 1 1  =  w1  w1
    // 1 1   0  w2   0 1     w1  w1 + w2
    if (weights.size() != 2) {
      report_error("Wrong size weight vector");
    }
    SpdMatrix ans(2);
    ans(0, 0) = ans(0, 1) = ans(1, 0) = weights[0];
    ans(1, 1) = weights[0] + weights[1];
    return ans;
  }

  void LocalLinearTrendMatrix::add_to_block(SubMatrix block) const {
    check_can_add(block);
    block.row(0) += 1;
    block(1, 1) += 1;
  }

  Matrix LocalLinearTrendMatrix::dense() const {
    Matrix ans(2, 2, 1.0);
    ans(1, 0) = 0.0;
    return ans;
  }

  //======================================================================
  namespace {
    typedef DiagonalMatrixParamView DMPV;
  }  // namespace

  void DMPV::add_variance(const Ptr<UnivParams> &variance) {
    variances_.push_back(variance);
    set_observer(variance);
    current_ = false;
  }

  void DMPV::ensure_current() const {
    if (current_) return;
    diagonal_elements_.resize(variances_.size());
    for (int i = 0; i < diagonal_elements_.size(); ++i) {
      diagonal_elements_[i] = variances_[i]->value();
    }
    current_ = true;
  }

  void DMPV::set_observer(const Ptr<UnivParams> &variance) {
    variance->add_observer(this, [this]() { current_ = false; });
  }

  //======================================================================
  namespace {
    typedef SparseDiagonalMatrixBlockParamView SDMB;
  }  // namespace

  SDMB *SDMB::clone() const { return new SDMB(*this); }

  void SDMB::add_element(const Ptr<UnivParams> &element, int position) {
    if (position < 0) {
      report_error("Position must be non-negative.");
    }
    if (!positions_.empty() && position < positions_.back()) {
      report_error("Please add elements in position order.");
    }
    if (position >= dim_) {
      report_error("Position value exceeds matrix dimension.");
    }
    elements_.push_back(element);
    positions_.push_back(position);
  }

  void SDMB::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs = 0;
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      lhs[pos] = rhs[pos] * elements_[i]->value();
    }
  }

  void SDMB::multiply_and_add(VectorView lhs,
                              const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      lhs[pos] += rhs[pos] * elements_[i]->value();
    }
  }

  void SDMB::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    multiply(lhs, rhs);
  }

  void SDMB::multiply_inplace(VectorView x) const {
    conforms_to_cols(x.size());
    int x_index = 0;
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      while (x_index < pos) {
        x[x_index++] = 0;
      }
      x[x_index++] *= elements_[i]->value();
    }
    while (x_index < x.size()) {
      x[x_index++] = 0;
    }
  }

  SpdMatrix SDMB::inner() const {
    Matrix ans(nrow(), 0.0);
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      ans(pos, pos) = square(elements_[i]->value());
    }
    return ans;
  }

  SpdMatrix SDMB::inner(const ConstVectorView &weights) const {
    if (weights.size() != nrow()) {
      report_error("Wrong size weight vector.");
    }
    Matrix ans(nrow(), 0.0);
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      ans(pos, pos) = square(elements_[i]->value()) * weights[i];
    }
    return ans;
  }

  void SDMB::add_to_block(SubMatrix block) const {
    conforms_to_cols(block.ncol());
    conforms_to_rows(block.nrow());
    for (int i = 0; i < positions_.size(); ++i) {
      int pos = positions_[i];
      block(pos, pos) += elements_[i]->value();
    }
  }

  //======================================================================
  namespace {
    typedef SeasonalStateSpaceMatrix SSSM;
  }  // namespace

  SSSM::SeasonalStateSpaceMatrix(int number_of_seasons)
      : number_of_seasons_(number_of_seasons) {}

  SeasonalStateSpaceMatrix *SSSM::clone() const {
    return new SeasonalStateSpaceMatrix(*this);
  }

  int SSSM::nrow() const { return number_of_seasons_ - 1; }

  int SSSM::ncol() const { return number_of_seasons_ - 1; }

  void SSSM::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] = 0;
    for (int i = 0; i < ncol(); ++i) {
      lhs[0] -= rhs[i];
      if (i > 0) lhs[i] = rhs[i - 1];
    }
  }

  void SSSM::multiply_and_add(VectorView lhs,
                              const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    for (int i = 0; i < ncol(); ++i) {
      lhs[0] -= rhs[i];
      if (i > 0) lhs[i] += rhs[i - 1];
    }
  }

  void SSSM::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    conforms_to_cols(lhs.size());
    double first = rhs[0];
    for (int i = 0; i < rhs.size() - 1; ++i) {
      lhs[i] = rhs[i + 1] - first;
    }
    lhs[rhs.size() - 1] = -first;
  }

  void SSSM::multiply_inplace(VectorView x) const {
    conforms_to_rows(x.size());
    int stride = x.stride();
    int n = x.size();
    double *now = &x[n - 1];
    double total = -*now;
    for (int i = 0; i < n - 1; ++i) {
      double *prev = now - stride;
      total -= *prev;
      *now = *prev;
      now = prev;
    }
    *now = total;
  }

  SpdMatrix SSSM::inner() const {
    // -1  1  0  0 .... 0          -1 -1 -1 -1 ... -1
    // -1  0  1  0 .... 0           1  0  0  0 .... 0
    // -1  0  0  1 .... 0           0  1  0  0 .... 0
    // -1  0  0  0 .... 0           0  0  1  0 .... 0
    // -1  0  0  0 .... 0           0  0  0  1 .... 0
    SpdMatrix ans(nrow());
    ans = 1.0;
    ans.diag() = 2.0;
    ans.diag().back() = 1.0;
    return ans;
  }

  SpdMatrix SSSM::inner(const ConstVectorView &weights) const {
    // -1  1  0  0 .... 0   w1        -1 -1 -1 -1 ... -1
    // -1  0  1  0 .... 0     w2       1  0  0  0 .... 0
    // -1  0  0  1 .... 0       w3     0  1  0  0 .... 0
    // -1  0  0  0 .... 0         w4   0  0  1  0 .... 0
    // -1  0  0  0 .... 0           w5 0  0  0  1 .... 0
    //
    // = w1 + w2  w1    w1 ....
    //   w1     w1 + w3 w1 ....
    //   w1       w1    w1 + w4
    if (weights.size() != nrow()) {
      report_error("Wrong size weight vector.");
    }
    SpdMatrix ans(nrow(), 0.0);
    ans += weights[0];
    VectorView(ans.diag(), 0, nrow() - 1) +=
        ConstVectorView(weights, 1, nrow() - 1);
    return ans;
  }

  void SSSM::add_to_block(SubMatrix block) const {
    check_can_add(block);
    block.row(0) -= 1;
    VectorView d(block.subdiag(1));
    d += 1;
  }

  Matrix SSSM::dense() const {
    Matrix ans(nrow(), ncol(), 0.0);
    ans.row(0) = -1;
    ans.subdiag(1) = 1.0;
    return ans;
  }

  Vector SSSM::left_inverse(const ConstVectorView &x) const {
    Vector ans = ConstVectorView(x, 1);
    ans.push_back(-1 * x.sum());
    return(ans);
  }
  //======================================================================
  AutoRegressionTransitionMatrix::AutoRegressionTransitionMatrix(
      const Ptr<GlmCoefs> &rho)
      : autoregression_params_(rho) {}

  AutoRegressionTransitionMatrix::AutoRegressionTransitionMatrix(
      const AutoRegressionTransitionMatrix &rhs)
      : SparseMatrixBlock(rhs),
        autoregression_params_(rhs.autoregression_params_->clone()) {}

  AutoRegressionTransitionMatrix *AutoRegressionTransitionMatrix::clone()
      const {
    return new AutoRegressionTransitionMatrix(*this);
  }

  int AutoRegressionTransitionMatrix::nrow() const {
    return autoregression_params_->nvars_possible();
  }

  int AutoRegressionTransitionMatrix::ncol() const {
    return autoregression_params_->nvars_possible();
  }

  void AutoRegressionTransitionMatrix::multiply(
      VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] = 0;
    int p = nrow();
    const Vector &rho(autoregression_params_->value());
    for (int i = 0; i < p; ++i) {
      lhs[0] += rho[i] * rhs[i];
      if (i > 0) lhs[i] = rhs[i - 1];
    }
  }

  void AutoRegressionTransitionMatrix::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    int p = nrow();
    const Vector &rho(autoregression_params_->value());
    for (int i = 0; i < p; ++i) {
      lhs[0] += rho[i] * rhs[i];
      if (i > 0) lhs[i] += rhs[i - 1];
    }
  }

  void AutoRegressionTransitionMatrix::Tmult(VectorView lhs,
                                             const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    conforms_to_cols(lhs.size());
    int p = ncol();
    const Vector &rho(autoregression_params_->value());
    for (int i = 0; i < p; ++i) {
      lhs[i] = rho[i] * rhs[0] + (i + 1 < p ? rhs[i + 1] : 0);
    }
  }

  void AutoRegressionTransitionMatrix::multiply_inplace(VectorView x) const {
    conforms_to_cols(x.size());
    int p = x.size();
    double first_entry = 0;
    const Vector &rho(autoregression_params_->value());
    for (int i = p - 1; i >= 0; --i) {
      first_entry += rho[i] * x[i];
      if (i > 0) {
        x[i] = x[i - 1];
      } else {
        x[i] = first_entry;
      }
    }
  }

  SpdMatrix AutoRegressionTransitionMatrix::inner() const {
    SpdMatrix ans = outer(autoregression_params_->value());
    int dim = ans.nrow();
    VectorView(ans.diag(), 0, dim - 1) += 1.0;
    return ans;
  }

  SpdMatrix AutoRegressionTransitionMatrix::inner(
      const ConstVectorView &weights) const {
    SpdMatrix ans = outer(autoregression_params_->value());
    int dim = ans.nrow();
    if (weights.size() != dim) {
      report_error("Wrong size weight vector.");
    }
    ans *= weights[0];
    ConstVectorView shifted_weights(weights, 1);
    VectorView(ans.diag(), 0, dim - 1) += shifted_weights;
    return ans;
  }

  void AutoRegressionTransitionMatrix::add_to_block(SubMatrix block) const {
    check_can_add(block);
    block.row(0) += autoregression_params_->value();
    VectorView d(block.subdiag(1));
    d += 1;
  }

  Matrix AutoRegressionTransitionMatrix::dense() const {
    int p = nrow();
    Matrix ans(p, p, 0.0);
    ans.row(0) = autoregression_params_->value();
    ans.subdiag(1) = 1.0;
    return ans;
  }

  Vector AutoRegressionTransitionMatrix::left_inverse(
      const ConstVectorView &x) const {
      // The forward operation turns [x1, x2, x3] into [dot, x1, x2].  To
      // reverse this operation, we shift everything up by 1, then solve for the missing piece.
      Vector ans = ConstVectorView(x, 1);
      ans.push_back(0);
      double dot = autoregression_params_->predict(ans);
      int dim = autoregression_params_->nvars_possible();
      ans.back() = (x[0] - dot) / autoregression_params_->Beta(dim - 1);
      return ans;
  }

  //======================================================================
  namespace {
    typedef SingleElementInFirstRow SEIFR;
  }  // namespace

  void SEIFR::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs = 0;
    lhs[0] = rhs[position_] * value_;
  }

  void SEIFR::multiply_and_add(VectorView lhs,
                               const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] += rhs[position_] * value_;
  }

  void SEIFR::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_cols(lhs.size());
    conforms_to_rows(rhs.size());
    lhs = 0;
    lhs[position_] = value_ * rhs[0];
  }

  void SEIFR::multiply_inplace(VectorView x) const {
    conforms_to_cols(x.size());
    double tmp = x[position_] * value_;
    x = 0;
    x[0] = tmp;
  }

  void SEIFR::matrix_multiply_inplace(SubMatrix m) const {
    conforms_to_cols(m.nrow());
    m.row(0) = value_ * m.row(position_);
    if (m.nrow() > 1) {
      SubMatrix(m, 1, m.nrow() - 1, 0, m.ncol() - 1) = 0;
    }
  }

  void SEIFR::matrix_transpose_premultiply_inplace(SubMatrix m) const {
    conforms_to_rows(m.nrow());
    m.col(0) = m.col(position_) * value_;
    SubMatrix(m, 0, m.nrow() - 1, 1, m.ncol() - 1) = 0;
  }

  SpdMatrix SEIFR::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    ans(position_, position_) = square(value_);
    return ans;
  }

  SpdMatrix SEIFR::inner(const ConstVectorView &weights) const {
    SpdMatrix ans(ncol(), 0.0);
    ans(position_, position_) = square(value_) * weights[0];
    return ans;
  }

  void SEIFR::add_to_block(SubMatrix block) const {
    conforms_to_rows(block.nrow());
    conforms_to_cols(block.ncol());
    block(0, position_) += value_;
  }

  //======================================================================
  GenericSparseMatrixBlockElementProxy &GenericSparseMatrixBlockElementProxy::
  operator=(double new_value) {
    matrix_->insert_element(row_, col_, new_value);
    value_ = new_value;
    return *this;
  }

  GenericSparseMatrixBlock::GenericSparseMatrixBlock(int nrow, int ncol)
      : nrow_(nrow),
        ncol_(ncol),
        nrow_compressed_(0),
        empty_row_(ncol_),
        empty_column_(nrow_) {
    if (nrow < 0 || ncol < 0) {
      report_error("Negative matrix dimension.");
    }
  }

  GenericSparseMatrixBlockElementProxy GenericSparseMatrixBlock::operator()(
      int row, int col) {
    auto it = rows_.find(row);
    if (it == rows_.end()) {
      return GenericSparseMatrixBlockElementProxy(row, col, 0, this);
    } else {
      return GenericSparseMatrixBlockElementProxy(row, col, it->second[col],
                                                  this);
    }
  }

  double GenericSparseMatrixBlock::operator()(int row, int col) const {
    auto it = rows_.find(row);
    if (it == rows_.end()) {
      return 0;
    } else {
      return it->second[col];
    }
  }

  void GenericSparseMatrixBlock::set_row(const SparseVector &row,
                                         int row_number) {
    if (row.size() != ncol()) {
      report_error("Size of inserted row must match the number of columns.");
    }
    auto it = rows_.find(row_number);
    if (it == rows_.end()) {
      ++nrow_compressed_;
    }
    rows_[row_number] = row;
    for (const auto &el : row) {
      insert_element_in_columns(row_number, el.first, el.second);
    }
  }

  void GenericSparseMatrixBlock::set_column(const SparseVector &column,
                                            int col_number) {
    if (column.size() != nrow()) {
      report_error("Size of inserted column must match the number of rows.");
    }
    columns_[col_number] = column;
    for (const auto &el : column) {
      insert_element_in_rows(el.first, col_number, el.second);
    }
  }

  void GenericSparseMatrixBlock::insert_element_in_columns(uint row, uint col,
                                                           double value) {
    auto it = columns_.find(col);
    if (it == columns_.end()) {
      SparseVector column_vector(nrow_);
      column_vector[row] = value;
      columns_.insert(std::make_pair(col, column_vector));
    } else {
      it->second[row] = value;
    }
  }

  void GenericSparseMatrixBlock::insert_element_in_rows(uint row, uint col,
                                                        double value) {
    auto it = rows_.find(row);
    if (it == rows_.end()) {
      SparseVector row_vector(ncol_);
      row_vector[col] = value;
      rows_.insert(std::make_pair(row, row_vector));
      ++nrow_compressed_;
    } else {
      it->second[col] = value;
    }
  }

  void GenericSparseMatrixBlock::multiply(VectorView lhs,
                                          const ConstVectorView &rhs) const {
    lhs = 0.0;
    multiply_and_add(lhs, rhs);
  }

  void GenericSparseMatrixBlock::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_cols(rhs.size());
    conforms_to_rows(lhs.size());
    for (const auto &row : rows_) {
      lhs[row.first] += row.second.dot(rhs);
    }
  }

  void GenericSparseMatrixBlock::Tmult(VectorView lhs,
                                       const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    conforms_to_cols(lhs.size());
    lhs = 0;
    for (const auto &col : columns_) {
      lhs[col.first] = col.second.dot(rhs);
    }
  }

  void GenericSparseMatrixBlock::multiply_inplace(VectorView x) const {
    if (nrow() != ncol()) {
      report_error("multiply_inplace is only defined for square matrices.");
    }
    conforms_to_cols(x.size());
    Vector ans(nrow_compressed_);
    int counter = 0;
    for (const auto &row : rows_) {
      ans[counter++] = row.second.dot(x);
    }
    x = 0;
    counter = 0;
    for (const auto &row : rows_) {
      x[row.first] = ans[counter++];
    }
  }

  SpdMatrix GenericSparseMatrixBlock::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    for (const auto &el : rows_) {
      el.second.add_outer_product(ans);
    }
    return ans;
  }

  SpdMatrix GenericSparseMatrixBlock::inner(
      const ConstVectorView &weights) const {
    SpdMatrix ans(ncol(), 0.0);
    for (const auto &el : rows_) {
      uint position = el.first;
      el.second.add_outer_product(ans, weights[position]);
    }
    return ans;
  }

  void GenericSparseMatrixBlock::add_to_block(SubMatrix block) const {
    conforms_to_rows(block.nrow());
    conforms_to_cols(block.ncol());
    for (const auto &row : rows_) {
      row.second.add_this_to(block.row(row.first), 1.0);
    }
  }

  const SparseVector &GenericSparseMatrixBlock::row(int row_number) const {
    const auto it = rows_.find(row_number);
    if (it == rows_.end()) {
      return empty_row_;
    } else {
      return it->second;
    }
  }

  const SparseVector &GenericSparseMatrixBlock::column(int col_number) const {
    const auto it = columns_.find(col_number);
    if (it == columns_.end()) {
      return empty_column_;
    } else {
      return it->second;
    }
  }

  //======================================================================

  StackedRegressionCoefficients *StackedRegressionCoefficients::clone() const {
    return new StackedRegressionCoefficients(*this);
  }

  void StackedRegressionCoefficients::add_row(const Ptr<GlmCoefs> &beta) {
    if (!coefficients_.empty()) {
      if (beta->nvars_possible() != coefficients_[0]->nvars_possible()) {
        report_error("All coefficient vectors must be the same size.");
      }
    }
    coefficients_.push_back(beta);
  }

  namespace {
    template <class VECTOR>
    Vector stacked_regression_vector_mult(
        const VECTOR &v, const StackedRegressionCoefficients &coef) {
      Vector ans(coef.nrow());
      for (int i = 0; i < coef.nrow(); ++i) {
        ans[i] = coef.coefficients(i).predict(v);
      }
      return ans;
    }
  }  // namespace

  void StackedRegressionCoefficients::multiply(
      VectorView lhs, const ConstVectorView &rhs) const {
    check_can_multiply(lhs, rhs);
    for (int i = 0; i < lhs.size(); ++i) {
      lhs[i] = coefficients_[i]->predict(rhs);
    }
  }

  void StackedRegressionCoefficients::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    check_can_multiply(rhs.size());
    if (lhs.size() != nrow()) {
      report_error("lhs argument is the wrong size in "
                   "StackedRegressionCoefficients::multiply_and_add.");
    }
    for (int i = 0; i < lhs.size(); ++i) {
      lhs[i] += coefficients_[i]->predict(rhs);
    }
  }

  void StackedRegressionCoefficients::add_to_block(SubMatrix block) const {
    for (int i = 0; i < block.nrow(); ++i) {
      coefficients_[i]->add_to(block.row(i));
    }
  }

  void StackedRegressionCoefficients::Tmult(
      VectorView lhs, const ConstVectorView &rhs) const {
    check_can_Tmult(rhs.size());
    if (lhs.size() != ncol()) {
      report_error("lhs argument is the wrong size in "
                   "StackedRegressionCoefficients::Tmult.");
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      lhs[i] = 0;
      for (size_t j = 0; j < rhs.size(); ++j) {
        lhs[i] += coefficients_[j]->value()[i] * rhs[j];
      }
    }
  }

  void StackedRegressionCoefficients::multiply_inplace(
      VectorView x) const {
    check_can_multiply(x.size());
    if (nrow() != ncol()) {
      report_error("multiply_inplace only applies to square matrices.");
    }
    x = *this * x;
  }

  Vector StackedRegressionCoefficients::operator*(
      const Vector &v) const {
    return stacked_regression_vector_mult(v, *this);
  }
  Vector StackedRegressionCoefficients::operator*(
      const VectorView &v) const {
    return stacked_regression_vector_mult(v, *this);
  }
  Vector StackedRegressionCoefficients::operator*(
      const ConstVectorView &v) const {
    return stacked_regression_vector_mult(v, *this);
  }

  Vector StackedRegressionCoefficients::Tmult(
      const ConstVectorView &x) const {
    Vector ans(ncol());
    this->Tmult(VectorView(ans), x);
    return ans;
  }

  SpdMatrix StackedRegressionCoefficients::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    for (int i = 0; i < nrow(); ++i) {
      ans.add_outer(coefficients_[i]->value(), coefficients_[i]->inc());
    }
    return ans;
  }

  SpdMatrix StackedRegressionCoefficients::inner(
      const ConstVectorView &weights) const {
    SpdMatrix ans(ncol(), 0.0);
    for (int i = 0; i < nrow(); ++i) {
      ans.add_outer(coefficients_[i]->value(),
                    coefficients_[i]->inc(), weights[i]);
    }
    return ans;
  }

  Matrix &StackedRegressionCoefficients::add_to(Matrix &P) const {
    for (int i = 0; i < nrow(); ++i) {
      P.row(i) += coefficients_[i]->value();
    }
    return P;
  }

  SubMatrix StackedRegressionCoefficients::add_to_submatrix(SubMatrix P) const {
    for (int i = 0; i < nrow(); ++i) {
      P.row(i) += coefficients_[i]->value();
    }
    return P;
  }

  //======================================================================
  Matrix SparseKalmanMatrix::operator*(const Matrix &rhs) const {
    int nr = nrow();
    int nc = rhs.ncol();
    Matrix ans(nr, nc);
    for (int i = 0; i < nc; ++i) {
      ans.col(i) = (*this) * rhs.col(i);
    }
    return ans;
  }

  Matrix SparseKalmanMatrix::Tmult(const Matrix &rhs) const {
    Matrix ans(ncol(), rhs.ncol());
    for (int i = 0; i < rhs.ncol(); ++i) {
      ans.col(i) = this->Tmult(rhs.col(i));
    }
    return ans;
  }

  void SparseKalmanMatrix::sandwich_inplace(SpdMatrix &P) const {
    // First replace P with *this * P, which corresponds to *this
    // multiplying each column of P.
    for (int i = 0; i < P.ncol(); ++i) {
      P.col(i) = (*this) * P.col(i);
    }
    // Next, post-multiply P by this->transpose.  A * B is the same
    // thing as taking each row of A and transpose-multiplying it by
    // B.  (This follows because A * B = (B^T * A^T)^T ).  Because the
    // final factor of the product is this->transpose(), the
    // 'transpose-multiply' operation is really just a regular
    // multiplication.
    for (int i = 0; i < P.nrow(); ++i) {
      P.row(i) = (*this) * P.row(i);
    }
  }

  void SparseKalmanMatrix::sandwich_inplace_submatrix(SubMatrix P) const {
    SpdMatrix tmp(P.to_matrix());
    sandwich_inplace(tmp);
    P = tmp;
  }

  // Replaces P with this.transpose * P * this
  void SparseKalmanMatrix::sandwich_inplace_transpose(SpdMatrix &P) const {
    // First replace P with this->Tmult(P), which just
    // transpose-multiplies each column of P by *this.
    for (int i = 0; i < P.ncol(); ++i) {
      P.col(i) = this->Tmult(P.col(i));
    }
    // Next take the resulting matrix and post-multiply it by 'this',
    // which is just the transpose of this->transpose * that.
    for (int j = 0; j < P.nrow(); ++j) {
      P.row(j) = this->Tmult(P.row(j));
    }
  }

  // The logic of this function parallels "sandwich_inplace" above.
  // The implementation is different because for non-square matrices
  // we need a temporary variable to store intermediate results.
  SpdMatrix SparseKalmanMatrix::sandwich(const SpdMatrix &P) const {
    SpdMatrix ans(nrow());
    Matrix tmp(nrow(), ncol());
    for (int i = 0; i < ncol(); ++i) {
      tmp.col(i) = (*this) * P.col(i);
    }
    for (int i = 0; i < nrow(); ++i) {
      ans.row(i) = (*this) * tmp.row(i);
    }
    ans.fix_near_symmetry();
    return ans;
  }

  SpdMatrix SparseKalmanMatrix::sandwich_transpose(const SpdMatrix &P) const {
    SpdMatrix ans(ncol());
    Matrix tmp(ncol(), nrow());
    for (int i = 0; i < nrow(); ++i) {
      tmp.col(i) = this->Tmult(P.col(i));
    }
    for (int i = 0; i < ncol(); ++i) {
      ans.row(i) = this->Tmult(tmp.row(i));
    }
    return ans;
  }

  SubMatrix SparseKalmanMatrix::add_to_submatrix(SubMatrix P) const {
    Matrix tmp(P.to_matrix());
    this->add_to(tmp);
    P = tmp;
    return P;
  }

  // Returns this * rhs.transpose().
  Matrix SparseKalmanMatrix::multT(const Matrix &rhs) const {
    if (ncol() != rhs.ncol()) {
      report_error(
          "SparseKalmanMatrix::multT called with "
          "incompatible matrices.");
    }
    Matrix ans(nrow(), rhs.nrow());
    for (int i = 0; i < rhs.nrow(); ++i) {
      ans.col(i) = (*this) * rhs.row(i);
    }
    return ans;
  }

  Matrix operator*(const Matrix &lhs, const SparseKalmanMatrix &rhs) {
    int nr = lhs.nrow();
    int nc = rhs.ncol();
    Matrix ans(nr, nc);
    for (int i = 0; i < nr; ++i) {
      ans.row(i) = rhs.Tmult(lhs.row(i));
    }
    return ans;
  }

  // Returns lhs * rhs.transpose().  This is the same as the transpose of
  // rhs.Tmult(lhs.transpose()), but of course lhs is symmetric.  The answer can
  // be computed by filling the rows of the solution with
  // rhs.Tmult(columns_of_lhs).
  Matrix multT(const SpdMatrix &lhs, const SparseKalmanMatrix &rhs) {
    Matrix ans(lhs.nrow(), rhs.nrow());
    for (int i = 0; i < ans.nrow(); ++i) {
      ans.row(i) = rhs * lhs.col(i);
    }
    return ans;
  }

  //======================================================================
  BlockDiagonalMatrix::BlockDiagonalMatrix() : nrow_(0), ncol_(0) {}

  BlockDiagonalMatrix::BlockDiagonalMatrix(const BlockDiagonalMatrix &rhs)
      : nrow_(0),
        ncol_(0)
  {
    for (const auto &block : rhs.blocks_) {
      add_block(block->clone());
    }
  }

  BlockDiagonalMatrix & BlockDiagonalMatrix::operator=(const BlockDiagonalMatrix &rhs) {
    if (&rhs != this) {
      clear();
      for (const auto &block : rhs.blocks_) {
        add_block(block->clone());
      }
    }
    return *this;
  }

  BlockDiagonalMatrix * BlockDiagonalMatrix::clone() const {
    return new BlockDiagonalMatrix(*this);
  }

  void BlockDiagonalMatrix::add_block(const Ptr<SparseMatrixBlock> &m) {
    blocks_.push_back(m);
    nrow_ += m->nrow();
    ncol_ += m->ncol();
    row_boundaries_.push_back(nrow_);
    col_boundaries_.push_back(ncol_);
  }

  void BlockDiagonalMatrix::replace_block(int which_block,
                                          const Ptr<SparseMatrixBlock> &b) {
    if (b->nrow() != blocks_[which_block]->nrow() ||
        b->ncol() != blocks_[which_block]->ncol()) {
      report_error("");
    }
    blocks_[which_block] = b;
  }

  void BlockDiagonalMatrix::clear() {
    blocks_.clear();
    nrow_ = ncol_ = 0;
    row_boundaries_.clear();
    col_boundaries_.clear();
  }

  int BlockDiagonalMatrix::nrow() const { return nrow_; }
  int BlockDiagonalMatrix::ncol() const { return ncol_; }

  //---------------------------------------------------------------------------
  // TODO(steve): add a unit test for the case where diagonal
  // blocks are not square.
  void block_multiply_view(VectorView ans, const ConstVectorView &v, int nrow, int ncol,
                           const std::vector<Ptr<SparseMatrixBlock>> &blocks) {
    if (v.size() != ncol) {
      report_error("incompatible vector in BlockDiagonalMatrix::operator*");
    }
    int lhs_pos = 0;
    int rhs_pos = 0;
    for (int b = 0; b < blocks.size(); ++b) {
      int nr = blocks[b]->nrow();
      VectorView lhs(ans, lhs_pos, nr);
      lhs_pos += nr;

      int nc = blocks[b]->ncol();
      if (nc > 0) {
        ConstVectorView rhs(v, rhs_pos, nc);
        rhs_pos += nc;
        blocks[b]->multiply(lhs, rhs);
      } else {
        lhs = 0.0;
      }
    }
  }

  Vector block_multiply(const ConstVectorView &v, int nrow, int ncol,
                        const std::vector<Ptr<SparseMatrixBlock>> &blocks) {
    Vector ans(nrow);
    block_multiply_view(VectorView(ans), v, nrow, ncol, blocks);
    return ans;
  }

  void block_multiply_and_add(
      VectorView ans,
      const ConstVectorView &v,
      int nrow,
      int ncol,
      const std::vector<Ptr<SparseMatrixBlock>> &blocks) {
    if (v.size() != ncol) {
      report_error("incompatible vector in BlockDiagonalMatrix::operator*");
    }
    int lhs_pos = 0;
    int rhs_pos = 0;
    for (int b = 0; b < blocks.size(); ++b) {
      int nr = blocks[b]->nrow();
      VectorView lhs(ans, lhs_pos, nr);
      lhs_pos += nr;

      int nc = blocks[b]->ncol();
      if (nc > 0) {
        ConstVectorView rhs(v, rhs_pos, nc);
        rhs_pos += nc;
        blocks[b]->multiply_and_add(lhs, rhs);
      }
    }
  }

  void block_transpose_multiply_view(
      VectorView lhs,
      const ConstVectorView &v,
      int nrow,
      int ncol,
      const std::vector<Ptr<SparseMatrixBlock>> &blocks) {

    if (v.size() != nrow) {
      report_error("incompatible vector in Tmult");
    }
    if (lhs.size() != ncol) {
      report_error("Incompatible LHS in block_transpose_multiply.");
    }

    int lhs_pos = 0;
    int rhs_pos = 0;

    for (int b = 0; b < blocks.size(); ++b) {
      VectorView lhs_chunk(lhs, lhs_pos, blocks[b]->ncol());
      lhs_pos += blocks[b]->ncol();
      ConstVectorView rhs_chunk(v, rhs_pos, blocks[b]->nrow());
      rhs_pos += blocks[b]->nrow();
      blocks[b]->Tmult(lhs_chunk, rhs_chunk);
    }
  }

  Vector block_transpose_multiply(
      const ConstVectorView &v,
      int nrow,
      int ncol,
      const std::vector<Ptr<SparseMatrixBlock>> &blocks) {
    Vector ans(ncol, 0);
    block_transpose_multiply_view(VectorView(ans), v, nrow, ncol, blocks);
    return ans;
  }

  void block_multiply_inplace(VectorView x, int nrow, int ncol,
                              const std::vector<Ptr<SparseMatrixBlock>> & blocks) {
    if (nrow != ncol) {
      report_error("multiply_inplace only works for square matrices.");
    }
    int start = 0;
    for (const auto &block : blocks) {
      if (block->nrow() != block->ncol()) {
        report_error("All individual blocks must be square for multiply_inplace.");
      }
      VectorView chunk(x, start, block->ncol());
      block->multiply_inplace(chunk);
      start += block->nrow();
    }
  }

  //---------------------------------------------------------------------------
  void BlockDiagonalMatrix::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    block_multiply_view(lhs, rhs, nrow(), ncol(), blocks_);
  }

  void BlockDiagonalMatrix::multiply_and_add(VectorView lhs,
                                             const ConstVectorView &rhs) const {
    block_multiply_and_add(lhs, rhs, nrow(), ncol(), blocks_);
  }

  void BlockDiagonalMatrix::multiply_inplace(VectorView v) const {
    block_multiply_inplace(v, nrow(), ncol(), blocks_);
  }

  Vector BlockDiagonalMatrix::operator*(const Vector &v) const {
    return block_multiply(ConstVectorView(v), nrow(), ncol(), blocks_);
  }

  Vector BlockDiagonalMatrix::operator*(const VectorView &v) const {
    return block_multiply(ConstVectorView(v), nrow(), ncol(), blocks_);
  }
  Vector BlockDiagonalMatrix::operator*(const ConstVectorView &v) const {
    return block_multiply(v, nrow(), ncol(), blocks_);
  }

  void BlockDiagonalMatrix::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    block_transpose_multiply_view(lhs, rhs, nrow(), ncol(), blocks_);
  }

  Vector BlockDiagonalMatrix::Tmult(const ConstVectorView &x) const {
    return block_transpose_multiply(x, nrow(), ncol(), blocks_);
  }

  namespace {
    // Fills dest with left * source * right.transpose.
    void sandwich_block(const SparseMatrixBlock &left,
                        const SparseMatrixBlock &right,
                        const ConstSubMatrix &source,
                        SubMatrix &dest,
                        Matrix &workspace) {
      // Workspace will hold the reult of left * source.
      workspace.resize(left.nrow(), source.ncol());
      for (int i = 0; i < source.ncol(); ++i) {
        left.multiply(workspace.col(i), source.col(i));
      }
      // Now put the result of workspace * right^T into dest.  We can do this by
      // putting the result of right * workspace^T into dest^T.
      for (int i = 0; i < workspace.nrow(); ++i) {
        // We want workspace * right^T.  The transpose of this is right *
        // workspace^T.  Multiply right by each row of workspace, and place the
        // result in the rows of dest.
        right.multiply(dest.row(i), workspace.row(i));
      }
    }

    // The implementation for BlockDiagonalMatrix::sandwich.  It has been
    // refactored to a free function so that it can be shared with similar
    // block-structured matrices.
    SpdMatrix block_sandwich(const SpdMatrix &P, int nrow, int ncol,
                             const std::vector<Ptr<SparseMatrixBlock>> &blocks,
                             const std::vector<int> &col_boundaries,
                             const std::vector<int> &row_boundaries) {
      // If *this is rectangular then the result will not be the same dimension
      // as P.  P must be ncol() X ncol().
      if (ncol != P.nrow()) {
        report_error("'sandwich' called on a non-conforming matrix.");
      }
      SpdMatrix ans(nrow);
      Matrix workspace;
      for (int i = 0; i < blocks.size(); ++i) {
        const SparseMatrixBlock &left(*(blocks[i]));
        if (left.ncol() == 0) {
          continue;
        }
        for (int j = i; j < blocks.size(); ++j) {
          const SparseMatrixBlock &right(*(blocks[j]));
          if (right.ncol() == 0) {
            continue;
          }
          // The source matrix is determined by columns.  The number of columns
          // in the left block determines the number of rows in the source
          // block.
          int rlo = (i == 0) ? 0 : col_boundaries[i - 1];
          int rhi = col_boundaries[i] - 1;
          // The number of rows in the transpose of the right block (i.e. the
          // number of columns in the right block) determines the number of
          // columns in the source block.
          int clo = (j == 0) ? 0 : col_boundaries[j - 1];
          int chi = col_boundaries[j] - 1;
          ConstSubMatrix source(P, rlo, rhi, clo, chi);

          // The destination block is determined by row boundaries.  The number
          // of rows in the left block determines the number of rows in the
          // destination block.
          rlo = (i == 0) ? 0 : row_boundaries[i - 1];
          rhi = row_boundaries[i] - 1;
          // The number of columns in the destination block is the number of
          // columns in the transpose of the right block (i.e. the number of
          // rows in the right block).
          clo = (j == 0) ? 0 : row_boundaries[j - 1];
          chi = row_boundaries[j] - 1;
          SubMatrix dest(ans, rlo, rhi, clo, chi);
          workspace.resize(left.nrow(), right.nrow());
          sandwich_block(left, right, source, dest, workspace);
        }
      }
      ans.reflect();
      return ans;
    }

  }  // namespace

  SpdMatrix BlockDiagonalMatrix::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    int start = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int end = start + blocks_[b]->ncol();
      SubMatrix(ans, start, end - 1, start, end - 1)
          = blocks_[b]->inner();
      start = end;
    }
    return ans;
  }

  SpdMatrix BlockDiagonalMatrix::inner(const ConstVectorView &weights) const {
    if (weights.size() != nrow()) {
      report_error("Wrong size weight vector for BlockDiagonalMatrix.");
    }
    SpdMatrix ans(ncol(), 0.0);
    int ans_start = 0;
    int weight_start = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      const SparseMatrixBlock &block(*blocks_[b]);
      int ans_end = ans_start + block.ncol();
      const ConstVectorView local_weights(weights, weight_start, block.nrow());
      SubMatrix(ans, ans_start, ans_end - 1, ans_start, ans_end - 1)
          = block.inner(local_weights);
      ans_start += block.ncol();
      weight_start += block.nrow();
    }
    return ans;
  }

  // this * P * this.transpose.
  SpdMatrix BlockDiagonalMatrix::sandwich(const SpdMatrix &P) const {
    return block_sandwich(P, nrow(), ncol(), blocks_,
                          col_boundaries_, row_boundaries_);
  }

  void BlockDiagonalMatrix::sandwich_inplace(SpdMatrix &P) const {
    for (int i = 0; i < blocks_.size(); ++i) {
      int rlo = i == 0 ? 0 : row_boundaries_[i - 1];
      int rhi = row_boundaries_[i] - 1;
      blocks_[i]->matrix_multiply_inplace(
          SubMatrix(P, rlo, rhi, 0, P.ncol() - 1));
    }
    for (int i = 0; i < blocks_.size(); ++i) {
      int clo = i == 0 ? 0 : col_boundaries_[i - 1];
      int chi = col_boundaries_[i] - 1;
      blocks_[i]->matrix_transpose_premultiply_inplace(
          SubMatrix(P, 0, P.nrow() - 1, clo, chi));
    }
  }

  void BlockDiagonalMatrix::sandwich_inplace_submatrix(SubMatrix P) const {
    for (int i = 0; i < blocks_.size(); ++i) {
      for (int j = 0; j < blocks_.size(); ++j) {
        sandwich_inplace_block((*blocks_[i]), (*blocks_[j]),
                               get_submatrix_block(P, i, j));
      }
    }
  }

  void BlockDiagonalMatrix::sandwich_inplace_block(
      const SparseMatrixBlock &left, const SparseMatrixBlock &right,
      SubMatrix middle) const {
    for (int i = 0; i < middle.ncol(); ++i) {
      left.multiply_inplace(middle.col(i));
    }

    for (int i = 0; i < middle.nrow(); ++i) {
      right.multiply_inplace(middle.row(i));
    }
  }

  SubMatrix BlockDiagonalMatrix::get_block(Matrix &m, int i, int j) const {
    int rlo = (i == 0 ? 0 : row_boundaries_[i - 1]);
    int rhi = row_boundaries_[i] - 1;

    int clo = (j == 0 ? 0 : col_boundaries_[j - 1]);
    int chi = col_boundaries_[j] - 1;
    return SubMatrix(m, rlo, rhi, clo, chi);
  }

  SubMatrix BlockDiagonalMatrix::get_submatrix_block(SubMatrix m, int i,
                                                     int j) const {
    int rlo = (i == 0 ? 0 : row_boundaries_[i - 1]);
    int rhi = row_boundaries_[i] - 1;

    int clo = (j == 0 ? 0 : col_boundaries_[j - 1]);
    int chi = col_boundaries_[j] - 1;
    return SubMatrix(m, rlo, rhi, clo, chi);
  }

  Matrix &BlockDiagonalMatrix::add_to(Matrix &P) const {
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix block = get_block(P, b, b);
      blocks_[b]->add_to_block(block);
    }
    return P;
  }

  SubMatrix BlockDiagonalMatrix::add_to_submatrix(SubMatrix P) const {
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix block = get_submatrix_block(P, b, b);
      blocks_[b]->add_to_block(block);
    }
    return P;
  }

  //===========================================================================
  namespace {
    template <class VECTOR>
    Vector block_multiply_impl(
        const std::vector<Ptr<SparseMatrixBlock>> &blocks, const VECTOR &rhs) {
      Vector ans(blocks.back()->nrow(), 0.0);
      int start = 0;
      for (int i = 0; i < blocks.size(); ++i) {
        int ncol = blocks[i]->ncol();
        blocks[i]->multiply_and_add(VectorView(ans),
                                    ConstVectorView(rhs, start, ncol));
        start += ncol;
      }
      return ans;
    }

  }  // namespace

  void SparseVerticalStripMatrix::add_block(
      const Ptr<SparseMatrixBlock> &block) {
    if (!blocks_.empty() && block->nrow() != blocks_.back()->nrow()) {
      report_error("All blocks must have the same number of rows");
    }
    blocks_.push_back(block);
    ncol_ += block->ncol();
  }

  Vector SparseVerticalStripMatrix::operator*(const Vector &v) const {
    check_can_multiply(v.size());
    return block_multiply_impl(blocks_, v);
  }
  Vector SparseVerticalStripMatrix::operator*(const VectorView &v) const {
    check_can_multiply(v.size());
    return block_multiply_impl(blocks_, v);
  }
  Vector SparseVerticalStripMatrix::operator*(const ConstVectorView &v) const {
    check_can_multiply(v.size());
    return block_multiply_impl(blocks_, v);
  }

  Vector SparseVerticalStripMatrix::Tmult(const ConstVectorView &v) const {
    check_can_Tmult(v.size());
    Vector ans(ncol());
    int start = 0;
    for (int i = 0; i < blocks_.size(); ++i) {
      int dim = blocks_[i]->ncol();
      blocks_[i]->Tmult(VectorView(ans, start, dim), v);
      start += dim;
    }
    return ans;
  }

  SpdMatrix SparseVerticalStripMatrix::inner() const {
    SpdMatrix ans(ncol(), 0.0);
    std::vector<Matrix> dense_blocks;
    dense_blocks.reserve(blocks_.size());
    for (int b = 0; b < blocks_.size(); ++b) {
      dense_blocks.push_back(blocks_[b]->dense());
    }
    int row_start = 0;
    for (int b0 = 0; b0 < blocks_.size(); ++b0) {
      BlockDiagonalMatrix row_block;
      row_block.add_block(blocks_[b0]);
      int col_start = row_start;
      int row_end = row_start + blocks_[b0]->ncol();
      for (int b1 = b0; b1 < blocks_.size(); ++b1) {
        int col_end = col_start + blocks_[b1]->ncol();
        SubMatrix(ans, row_start, row_end - 1, col_start, col_end - 1)
            = row_block.Tmult(dense_blocks[b1]);
        col_start = col_end;
      }
      row_start = row_end;
    }
    ans.reflect();
    return ans;
  }

  SpdMatrix SparseVerticalStripMatrix::inner(
      const ConstVectorView &weights) const {
    SpdMatrix ans(ncol(), 0.0);
    std::vector<Matrix> dense_blocks;
    dense_blocks.reserve(blocks_.size());
    DiagonalMatrix weight_block(weights);
    for (int b = 0; b < blocks_.size(); ++b) {
      dense_blocks.push_back(weight_block * blocks_[b]->dense());
    }

    int row_start = 0;
    for (int b0 = 0; b0 < blocks_.size(); ++b0) {
      BlockDiagonalMatrix row_block;
      row_block.add_block(blocks_[b0]);
      int col_start = row_start;
      int row_end = row_start + blocks_[b0]->ncol();
      for (int b1 = b0; b1 < blocks_.size(); ++b1) {
        int col_end = col_start + blocks_[b1]->ncol();
        SubMatrix(ans, row_start, row_end - 1, col_start, col_end - 1)
            = row_block.Tmult(dense_blocks[b1]);
        col_start = col_end;
      }
      row_start = row_end;
    }
    ans.reflect();
    return ans;
  }

  Matrix &SparseVerticalStripMatrix::add_to(Matrix &P) const {
    check_can_add(P.nrow(), P.ncol());
    int start_column = 0;
    for (int i = 0; i < blocks_.size(); ++i) {
      int ncol = blocks_[i]->ncol();
      blocks_[i]->add_to_block(
          SubMatrix(P, 0, nrow() - 1, start_column, start_column + ncol - 1));
      start_column += ncol;
    }
    return P;
  }

  SubMatrix SparseVerticalStripMatrix::add_to_submatrix(SubMatrix P) const {
    check_can_add(P.nrow(), P.ncol());
    int start_column = 0;
    for (int i = 0; i < blocks_.size(); ++i) {
      int ncol = blocks_[i]->ncol();
      blocks_[i]->add_to_block(
          SubMatrix(P, 0, nrow() - 1, start_column, start_column + ncol - 1));
      start_column += ncol;
    }
    return P;
  }


  //===========================================================================
  // LHS = *this * RHS

  ErrorExpanderMatrix::ErrorExpanderMatrix()
      : nrow_(0),
        ncol_(0)
  {}

  ErrorExpanderMatrix::ErrorExpanderMatrix(const ErrorExpanderMatrix &rhs)
      : nrow_(0),
        ncol_(0)
  {
    for (const auto &b : rhs.blocks_) {
      add_block(b->clone());
    }
  }

  ErrorExpanderMatrix & ErrorExpanderMatrix::operator=(
      const ErrorExpanderMatrix &rhs) {
    if (&rhs != this) {
      clear();
      for (const auto &b : rhs.blocks_) {
        add_block(b->clone());
      }
    }
    return *this;
  }

  ErrorExpanderMatrix * ErrorExpanderMatrix::clone() const {
    return new ErrorExpanderMatrix(*this);
  }

  int ErrorExpanderMatrix::nrow() const {
    return nrow_;
  }

  int ErrorExpanderMatrix::ncol() const {
    return ncol_;
  }

  void ErrorExpanderMatrix::add_block(const Ptr<SparseMatrixBlock> &block) {
    blocks_.push_back(block);
    increment_sizes(block);
  }

  // void ErrorExpanderMatrix::add_block(const Ptr<ErrorExpanderMatrix> &blocks) {
  //   for (const auto &block : blocks->blocks_) {
  //     add_block(block);
  //   }
  // }

  void ErrorExpanderMatrix::replace_block(
      int block_index,
      const Ptr<SparseMatrixBlock> &block) {
    const Ptr<SparseMatrixBlock> &old_block(blocks_[block_index]);
    bool recompute_needed =
        block->nrow() != old_block->nrow()
        || block->ncol() != old_block->ncol();
    blocks_[block_index] = block;

    if (recompute_needed) {
      recompute_sizes();
    }
  }

  void ErrorExpanderMatrix::recompute_sizes() {
    nrow_ = 0;
    ncol_ = 0;
    row_boundaries_.clear();
    col_boundaries_.clear();
    for (const auto &block : blocks_) {
      increment_sizes(block);
    }
  }

  void ErrorExpanderMatrix::increment_sizes(
      const Ptr<SparseMatrixBlock> &block) {
    nrow_ += block->nrow();
    ncol_ += block->ncol();
    row_boundaries_.push_back(nrow_);
    col_boundaries_.push_back(ncol_);
  }

  void ErrorExpanderMatrix::clear() {
    blocks_.clear();
    recompute_sizes();
  }

  // This logic is almost the same as the 'block_multiply' function above.
  void ErrorExpanderMatrix::multiply(VectorView lhs,
                                     const ConstVectorView &rhs) const {
    int lhs_pos = 0;
    int rhs_pos = 0;
    for (const auto &block : blocks_) {
      int nr = block->nrow();
      VectorView left_block(lhs, lhs_pos, nr);

      int nc = block->ncol();
      if (nc > 0) {
        const ConstVectorView right_block(rhs, rhs_pos, nc);
        block->multiply(left_block, right_block);
      } else {
        left_block = 0;
      }
      lhs_pos += nr;
      rhs_pos += nc;
    }
  }

  void ErrorExpanderMatrix::multiply_and_add(
      VectorView lhs, const ConstVectorView &rhs) const {
    int lhs_pos = 0;
    int rhs_pos = 0;
    for (const auto &block : blocks_) {
      int nr = block->nrow();
      VectorView left_block(lhs, lhs_pos, nr);

      int nc = block->ncol();
      if (nc > 0) {
        const ConstVectorView right_block(rhs, rhs_pos, nc);
        block->multiply_and_add(left_block, right_block);
      }
      lhs_pos += nr;
      rhs_pos += nc;
    }
  }

  Vector ErrorExpanderMatrix::operator*(const Vector &v) const {
    return block_multiply(v, nrow(), ncol(), blocks_);
  }

  Vector ErrorExpanderMatrix::operator*(const VectorView &v) const {
    return block_multiply(v, nrow(), ncol(), blocks_);
  }

  Vector ErrorExpanderMatrix::operator*(const ConstVectorView &v) const {
    return block_multiply(v, nrow(), ncol(), blocks_);
  }

  Vector ErrorExpanderMatrix::Tmult(const ConstVectorView &x) const {
    return block_transpose_multiply(x, nrow(), ncol(), blocks_);
  }

  void ErrorExpanderMatrix::Tmult(VectorView lhs,
                                  const ConstVectorView &rhs) const {
    block_transpose_multiply_view(lhs, rhs, nrow(), ncol(), blocks_);
  }

  void ErrorExpanderMatrix::multiply_inplace(VectorView x) const {
    block_multiply_inplace(x, nrow(), ncol(), blocks_);
  }

  void ErrorExpanderMatrix::add_to_block(SubMatrix block) const {
    if (block.nrow() != nrow()) {
      report_error("Block must have the same number of rows as the "
                   "ErrorExpanderMatrix.");
    }
    if (block.ncol() != ncol()) {
      report_error("Block must have the same number of columns as the "
                   "ErrorExpanderMatrix.");
    }

    size_t row_start = 0;
    size_t col_start = 0;
    for (const auto &b : blocks_) {
      b->add_to_block(SubMatrix(
          block, row_start, row_start + b->nrow() - 1,
          col_start, col_start + b->ncol() - 1));
      row_start += b->nrow();
      col_start += b->ncol();
    }
  }

  SpdMatrix ErrorExpanderMatrix::inner() const {
    int dim = ncol();
    SpdMatrix ans(dim, 0.0);
    int start = 0;
    for (const auto &block : blocks_) {
      if (block->ncol() == 0) {
        continue;
      }
      int end = start + block->ncol();
      SubMatrix(ans, start, end - 1, start, end - 1) = block->inner();
      start = end;
    }
    return ans;
  }

  SpdMatrix ErrorExpanderMatrix::inner(
      const ConstVectorView &weights) const {
    if (weights.size() != nrow()) {
      report_error("Wrong size weight vector.");
    }
    SpdMatrix ans(ncol(), 0.0);
    int ans_start = 0;
    int weight_start = 0;
    for (const auto &block : blocks_) {
      if (block->ncol() == 0) {
        weight_start += block->nrow();
      } else {
        int ans_end = ans_start + block->ncol();
        const ConstVectorView local_weights(
            weights, weight_start, block->nrow());
        SubMatrix(ans, ans_start, ans_end - 1, ans_start, ans_end - 1)
            = block->inner(local_weights);
        ans_start += block->ncol();
        weight_start += block->nrow();
      }
    }
    return ans;
  }

  void ErrorExpanderMatrix::sandwich_inplace(SpdMatrix &P) const {
    report_error("ErrorExpanderMatrix cannot sandwich_inplace.");
  }
  void ErrorExpanderMatrix::sandwich_inplace_submatrix(SubMatrix P) const {
    report_error("ErrorExpanderMatrix cannot sandwich_inplace_submatrix.");
  }

  SpdMatrix ErrorExpanderMatrix::sandwich(const SpdMatrix &P) const {
    return block_sandwich(P, nrow(), ncol(), blocks_,
                          row_boundaries_, col_boundaries_);
  }

  Matrix &ErrorExpanderMatrix::add_to(Matrix &P) const {
    int row_start = 0;
    int col_start = 0;
    for (const auto &block : blocks_) {
      if (block->ncol() > 0) {
        block->add_to_block(
            SubMatrix(P, row_start, row_start + block->nrow() - 1,
                      col_start, col_start + block->ncol() - 1));
        row_start += block->nrow();
        col_start += block->ncol();
      } else {
        row_start += block->ncol();
      }
    }
    return P;
  }

  SubMatrix ErrorExpanderMatrix::add_to_submatrix(SubMatrix P) const {
    int row_start = 0;
    int col_start = 0;
    for (const auto &block : blocks_) {
      if (block->ncol() > 0) {
        block->add_to_block(
            SubMatrix(P, row_start, row_start + block->nrow() - 1,
                      col_start, col_start + block->ncol() - 1));
        row_start += block->nrow();
        col_start += block->ncol();
      } else {
        row_start += block->ncol();
      }
    }
    return P;
  }

  Vector ErrorExpanderMatrix::left_inverse(const ConstVectorView &rhs) const {
    if (rhs.size() != nrow()) {
      report_error("Wrong size argument passed to left_inverse().");
    }
    Vector ans(ncol());
    int lhs_pos = 0;
    int rhs_pos = 0;
    for (const auto &block : blocks_) {
      if (block->ncol() > 0) {
        ConstVectorView rhs_block(rhs, rhs_pos, block->nrow());
        VectorView lhs(ans, lhs_pos, block->ncol());
        lhs = block->left_inverse(rhs_block);
        rhs_pos += block->ncol();
      }
      lhs_pos += block->nrow();
    }
    return ans;
  }



}  // namespace BOOM
