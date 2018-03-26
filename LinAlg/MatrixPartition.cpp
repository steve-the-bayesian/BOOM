// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#include "LinAlg/MatrixPartition.hpp"

namespace BOOM {
  MatrixPartition::MatrixPartition(Matrix *m, const std::vector<int> &row_sizes,
                                   const std::vector<int> &col_sizes)
      : m_(m),
        row_start_(row_sizes.size()),
        col_start_(col_sizes.size()),
        row_max_(row_sizes.size() - 1),
        col_max_(col_sizes.size() - 1) {
    row_start_[0] = 0;
    for (int i = 1; i < row_sizes.size(); ++i) {
      row_start_[i] = row_start_[i - 1] + row_sizes[i - 1];
    }

    col_start_[0] = 0;
    for (int j = 1; j < col_sizes.size(); ++j) {
      col_start_[j] = col_start_[j - 1] + col_sizes[j - 1];
    }
  }

  SubMatrix MatrixPartition::operator()(int i, int j) {
    int rlo = row_start_[i];
    int rhi = (i < row_max_) ? row_start_[i + 1] - 1 : m_->nrow() - 1;
    int clo = col_start_[j];
    int chi = (j < col_max_) ? col_start_[j + 1] - 1 : m_->ncol() - 1;
    return SubMatrix(*m_, rlo, rhi, clo, chi);
  }

  const SubMatrix MatrixPartition::operator()(int i, int j) const {
    int rlo = row_start_[i];
    int rhi = (i < row_max_) ? row_start_[i + 1] - 1 : m_->nrow() - 1;
    int clo = col_start_[j];
    int chi = (j < col_max_) ? col_start_[j + 1] - 1 : m_->ncol() - 1;
    return SubMatrix(*m_, rlo, rhi, clo, chi);
  }

  VectorView MatrixPartition::view(Vector &v, int i, bool premultiply) const {
    const std::vector<int> &pos(premultiply ? col_start_ : row_start_);
    int max = premultiply ? col_max_ : row_max_;
    int start = pos[i];
    int stop = (i < max) ? pos[i + 1] - 1 : length(v) - 1;
    return VectorView(v, start, stop - start + 1);
  }

  VectorView MatrixPartition::view(VectorView v, int i,
                                   bool premultiply) const {
    const std::vector<int> &pos(premultiply ? col_start_ : row_start_);
    int max = premultiply ? col_max_ : row_max_;
    int start = pos[i];
    int stop = (i < max) ? pos[i + 1] - 1 : length(v) - 1;
    return VectorView(v, start, stop - start + 1);
  }

  void MatrixPartition::reset(Matrix *m) { m_ = m; }

}  // namespace BOOM
