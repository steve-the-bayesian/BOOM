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

#include "LinAlg/SWEEP.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  typedef SweptVarianceMatrix SVM;

  namespace {
    void do_sweep(Matrix &target, int sweep_index, int sign) {
      double x = target(sweep_index, sweep_index);
      if (!std::isfinite(1.0 / x)) {
        report_error(
            "Zero variance implied by SWEEP operation.  "
            "Matrix might be less than full rank.");
      }
      target(sweep_index, sweep_index) = -1.0 / x;
      uint dimension = target.nrow();
      for (uint i = 0; i < dimension; ++i) {
        if (i != sweep_index) {
          for (uint j = 0; j < dimension; ++j) {
            if (j != sweep_index) {
              target(i, j) -= target(i, sweep_index) * target(sweep_index, j) / x;
            }
          }
        }
      }
      x *= sign;
      for (uint i = 0; i < dimension; ++i) {
        if (i != sweep_index) {
          target(i, sweep_index) /= x;
          target(sweep_index, i) /= x;
        }
      }
    }

  }  // namespace

  SVM::SweptVarianceMatrix(const SpdMatrix &m, bool inverse)
      : S_(m), swept_(m.nrow(), inverse) {
    if (inverse) {
      S_ *= -1;
    }
  }

  void SVM::SWP(const Selector &to_sweep) {
    uint p = to_sweep.nvars_possible();
    assert(p == S_.nrow());
    for (uint i = 0; i < p; ++i) {
      if (to_sweep[i] && !swept_[i]) {
        SWP(i);
      } else if (swept_[i] && !to_sweep[i]) {
        RSW(i);
      }
    }
  }

  void SVM::SWP(uint sweep_index) {
    if (swept_[sweep_index]) return;
    swept_.add(sweep_index);
    do_sweep(S_, sweep_index, 1);
  }


  //------------------------------------------------------------
  void SVM::RSW(uint sweep_index) {
    if (!swept_[sweep_index]) return;
    swept_.drop(sweep_index);
    do_sweep(S_, sweep_index, -1);
  }

  uint SVM::xdim() const { return swept_.nvars(); }
  uint SVM::ydim() const { return swept_.nvars_excluded(); }

  //------------------------------------------------------------
  Matrix SVM::Beta() const {  // E(y|x) = Beta * x
    return swept_.complement().select_rows(swept_.select_cols(S_));
  }
  //------------------------------------------------------------
  Vector SVM::conditional_mean(const Vector &known_subset,
                               const Vector &unconditional_mean) const {
    return Beta() * (known_subset - swept_.select(unconditional_mean)) +
           swept_.complement().select(unconditional_mean);
  }
  //------------------------------------------------------------
  SpdMatrix SVM::residual_variance() const {
    return swept_.complement().select(S_);
  }
  //------------------------------------------------------------
  SpdMatrix SVM::precision_of_swept_elements() const {
    return -1 * swept_.select(S_);
  }

}  // namespace BOOM
