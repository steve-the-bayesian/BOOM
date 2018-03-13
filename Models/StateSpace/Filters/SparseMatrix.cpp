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

#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <cpputil/report_error.hpp>
#include <LinAlg/SpdMatrix.hpp>
#include <iostream>

namespace BOOM{

  void SparseMatrixBlock::conforms_to_rows(int i) const {
    if (i==nrow()) return;
    std::ostringstream err;
    err << "object of length "
        << i
        << " does not conform with the number of rows ("
        << nrow() << ")";
    report_error(err.str());
  }

  void SparseMatrixBlock::conforms_to_cols(int i) const {
    if (i==ncol()) return;
    std::ostringstream err;
    err << "object of length "
        << i
        << " does not conform with the number of columns ("
        << ncol() << ")";
    report_error(err.str());
  }

  void SparseMatrixBlock::check_can_add(const SubMatrix &block) const {
    if (block.nrow() != nrow() || block.ncol()!=ncol()) {
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

  void SparseMatrixBlock::matrix_transpose_premultiply_inplace(
      SubMatrix m) const {
    for (int i = 0; i < m.nrow(); ++i) {
      multiply_inplace(m.row(i));
    }
  }

  Matrix SparseMatrixBlock::dense() const {
    Matrix ans(nrow(), ncol());
    ans.set_diag(1.0);
    for (int i = 0; i < ncol(); ++i) {
      this->multiply_inplace(ans.col(i));
    }
    return ans;
  }

  //======================================================================
  LocalLinearTrendMatrix * LocalLinearTrendMatrix::clone() const {
    return new LocalLinearTrendMatrix(*this);
  }

  void LocalLinearTrendMatrix::multiply(VectorView lhs,
                                        const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] = rhs[0] + rhs[1];
    lhs[1] = rhs[1];
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

  void LocalLinearTrendMatrix::add_to(SubMatrix m) const {
    check_can_add(m);
    m.row(0) += 1;
    m(1,1) += 1;
  }

  Matrix LocalLinearTrendMatrix::dense() const {
    Matrix ans(2,2, 1.0);
    ans(1,0) = 0.0;
    return ans;
  }

  //======================================================================
  typedef SeasonalStateSpaceMatrix SSSM;

  SSSM::SeasonalStateSpaceMatrix(int nseasons)
      : number_of_seasons_(nseasons)
  {}

  SeasonalStateSpaceMatrix * SSSM::clone() const {
    return new SeasonalStateSpaceMatrix(*this);
  }

  int SSSM::nrow() const {
    return number_of_seasons_ - 1;
  }

  int SSSM::ncol() const {
    return number_of_seasons_ - 1;
  }

  void SSSM::multiply(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(lhs.size());
    conforms_to_cols(rhs.size());
    lhs[0] = 0;
    for (int i = 0; i < ncol(); ++i) {
      lhs[0] -= rhs[i];
      if (i > 0) lhs[i] = rhs[i-1];
    }
  }

  void SSSM::Tmult(VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    conforms_to_cols(lhs.size());
    double first = rhs[0];
    for (int i = 0; i < rhs.size() - 1; ++i) {
      lhs[i] = rhs[i+1] - first;
    }
    lhs[rhs.size() - 1] = -first;
  }

  void SSSM::multiply_inplace(VectorView x) const {
    conforms_to_rows(x.size());
    int stride = x.stride();
    int n = x.size();
    double *now = &x[n-1];
    double total = -*now;
    for (int i = 0; i < n-1; ++i) {
      double *prev = now - stride;
      total -= *prev;
      *now = *prev;
      now = prev;
    }
    *now = total;
  }

  void SSSM::add_to(SubMatrix block) const {
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
  //======================================================================
  AutoRegressionTransitionMatrix::AutoRegressionTransitionMatrix(
      const Ptr<GlmCoefs> &rho) : autoregression_params_(rho)
  {}

  AutoRegressionTransitionMatrix::AutoRegressionTransitionMatrix(
      const AutoRegressionTransitionMatrix &rhs)
      : SparseMatrixBlock(rhs),
        autoregression_params_(rhs.autoregression_params_->clone())
  {}

  AutoRegressionTransitionMatrix *
  AutoRegressionTransitionMatrix::clone() const {
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
      if (i > 0) lhs[i] = rhs[i-1];
    }
  }

  void AutoRegressionTransitionMatrix::Tmult(
    VectorView lhs, const ConstVectorView &rhs) const {
    conforms_to_rows(rhs.size());
    conforms_to_cols(lhs.size());
    int p = ncol();
    const Vector &rho(autoregression_params_->value());
    for (int i = 0; i < p; ++i) {
      lhs[i] = rho[i]*rhs[0] + (i+1 < p ? rhs[i+1] : 0);
    }
  }

  void AutoRegressionTransitionMatrix::multiply_inplace(VectorView x) const {
    conforms_to_cols(x.size());
    int p = x.size();
    double first_entry = 0;
    const Vector &rho(autoregression_params_->value());
    for (int i = p-1; i >= 0; --i) {
      first_entry += rho[i] * x[i];
      if (i > 0) x[i] = x[i-1];
      else x[i] = first_entry;
    }
  }

  void AutoRegressionTransitionMatrix::add_to(SubMatrix block) const {
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
  //======================================================================
  void SparseKalmanMatrix::sandwich_inplace(SpdMatrix &P) const {
    // First replace P with *this * P, which corresponds to *this
    // multiplying each column of P.
    for (int i = 0; i < P.ncol(); ++i) {
      P.col(i) = (*this)*P.col(i);
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

  Matrix SparseKalmanMatrix::dense() const {
    Matrix ans(nrow(), ncol());
    Vector v(ncol(), 0.0);
    for (int i = 0; i < ncol(); ++i) {
      v[i] = 1.0;
      ans.col(i) = (*this) * v;
      v[i] = 0.0;
    }
    return ans;
  }
  //======================================================================
  BlockDiagonalMatrix::BlockDiagonalMatrix()
      : nrow_(0),
        ncol_(0)
  {}

  void BlockDiagonalMatrix::add_block(const Ptr<SparseMatrixBlock> &m) {
    blocks_.push_back(m);
    nrow_ += m->nrow();
    ncol_ += m->ncol();
    row_boundaries_.push_back(nrow_);
    col_boundaries_.push_back(ncol_);
  }

  void BlockDiagonalMatrix::replace_block(int which_block,
                                          const Ptr<SparseMatrixBlock> &b) {
    if (b->nrow() != blocks_[which_block]->nrow()
       || b->ncol() != blocks_[which_block]->ncol()) {
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

  int BlockDiagonalMatrix::nrow() const {return nrow_;}
  int BlockDiagonalMatrix::ncol() const {return ncol_;}

  // TODO(stevescott): add a unit test for the case where diagonal
  // blocks are not square.
  Vector block_multiply(const ConstVectorView &v, int nrow, int ncol,
                     const std::vector<Ptr<SparseMatrixBlock> > &blocks_) {
    if (v.size() != ncol) {
      report_error(
          "incompatible vector in "
          "BlockDiagonalMatrix::operator*");
    }
    Vector ans(nrow);

    int lhs_pos = 0;
    int rhs_pos = 0;
    for (int b = 0; b < blocks_.size(); ++b) {
      int nr = blocks_[b]->nrow();
      VectorView lhs(ans, lhs_pos, nr);
      lhs_pos += nr;

      int nc = blocks_[b]->ncol();
      ConstVectorView rhs(v, rhs_pos, nc);
      rhs_pos += nc;
      blocks_[b]->multiply(lhs, rhs);
    }
    return ans;
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

  Vector BlockDiagonalMatrix::Tmult(const Vector &x) const {
    if (x.size() != nrow()) {
      report_error(
          "incompatible vector in "
          "BlockDiagonalMatrix::Tmult");
    }
    int lhs_pos = 0;
    int rhs_pos = 0;
    Vector ans(ncol(), 0);

    for (int b = 0; b < blocks_.size(); ++b) {
      VectorView lhs(ans, lhs_pos, blocks_[b]->ncol());
      lhs_pos += blocks_[b]->ncol();
      ConstVectorView rhs(x, rhs_pos, blocks_[b]->nrow());
      rhs_pos += blocks_[b]->nrow();
      blocks_[b]->Tmult(lhs, rhs);
    }
    return ans;
  }

  // This assumes blocks_ are square
  SpdMatrix BlockDiagonalMatrix::sandwich(const SpdMatrix &P) const {
    SpdMatrix ans(P);
    for (int i = 0; i < blocks_.size(); ++i) {
      const SparseMatrixBlock &Ti(*(blocks_[i]));
      for (int j = i; j < blocks_.size(); ++j) {
        const SparseMatrixBlock &Tj(*(blocks_[j]));
        sandwich_inplace_block(Ti, Tj, get_block(ans, i, j));
      }
    }
    ans.reflect();
    return ans;
  }

  // void BlockDiagonalMatrix::sandwich_inplace(SpdMatrix &P) const {
  //   for (int i = 0; i < blocks_.size(); ++i) {
  //     for (int j = 0; j < blocks_.size(); ++j) {
  //       sandwich_inplace_block((*blocks_[i]),
  //                              (*blocks_[j]),
  //                              get_block(P, i, j));
  //     }
  //   }
  // }

  void BlockDiagonalMatrix::sandwich_inplace(SpdMatrix &P) const {
    for (int i = 0; i < blocks_.size(); ++i) {
      blocks_[i]->matrix_multiply_inplace(get_row_block(P, i));
    }
    for (int i = 0; i < blocks_.size(); ++i) {
      blocks_[i]->matrix_transpose_premultiply_inplace(get_col_block(P, i));
    }
  }

  void BlockDiagonalMatrix::sandwich_inplace_submatrix(SubMatrix P) const {
    for (int i = 0; i < blocks_.size(); ++i) {
      for (int j = 0; j < blocks_.size(); ++j) {
        sandwich_inplace_block((*blocks_[i]),
                               (*blocks_[j]),
                               get_submatrix_block(P, i, j));
      }
    }
  }

  void BlockDiagonalMatrix::sandwich_inplace_block(
      const SparseMatrixBlock &left,
      const SparseMatrixBlock &right,
      SubMatrix middle) const {
    for (int i = 0; i < middle.ncol(); ++i) {
      left.multiply_inplace(middle.col(i));
    }

    for (int i = 0; i < middle.nrow(); ++i) {
      right.multiply_inplace(middle.row(i));
    }
  }

  SubMatrix BlockDiagonalMatrix::get_block(Matrix &m, int i, int j) const {
    int rlo = i==0 ? 0 : row_boundaries_[i-1];
    int rhi = row_boundaries_[i] - 1;

    int clo = j==0 ? 0 : col_boundaries_[j-1];
    int chi = col_boundaries_[j] - 1;
    return SubMatrix(m, rlo, rhi, clo, chi);
  }

  SubMatrix BlockDiagonalMatrix::get_row_block(Matrix &m, int block) const {
    int rlo = block == 0 ? 0 : row_boundaries_[block - 1];
    int rhi = row_boundaries_[block] - 1;
    return SubMatrix(m, rlo, rhi, 0, m.ncol() - 1);
  }

  SubMatrix BlockDiagonalMatrix::get_col_block(Matrix &m, int block) const {
    int clo = block == 0 ? 0 : col_boundaries_[block - 1];
    int chi = col_boundaries_[block] - 1;
    return SubMatrix(m, 0, m.nrow()-1, clo, chi);
  }

  SubMatrix BlockDiagonalMatrix::get_submatrix_block(
      SubMatrix m, int i, int j) const {
    int rlo = i==0 ? 0 : row_boundaries_[i-1];
    int rhi = row_boundaries_[i] - 1;

    int clo = j==0 ? 0 : col_boundaries_[j-1];
    int chi = col_boundaries_[j] - 1;
    return SubMatrix(m, rlo, rhi, clo, chi);
  }

  Matrix & BlockDiagonalMatrix::add_to(Matrix &P) const {
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix block = get_block(P, b, b);
      blocks_[b]->add_to(block);
    }
    return P;
  }

  SubMatrix BlockDiagonalMatrix::add_to_submatrix(SubMatrix P) const {
    for (int b = 0; b < blocks_.size(); ++b) {
      SubMatrix block = get_submatrix_block(P, b, b);
      blocks_[b]->add_to(block);
    }
    return P;
  }

}
