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

#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using LLSM = LocalLevelStateModel;
  }

  LLSM::LocalLevelStateModel(double sigma)
      : ZeroMeanGaussianModel(sigma),
        state_transition_matrix_(new IdentityMatrix(1)),
        state_variance_matrix_(new ConstantMatrixParamView(1, Sigsq_prm())),
        initial_state_mean_(1, 0.0),
        initial_state_variance_(1, 1.0) {}

  LLSM::LocalLevelStateModel(const LocalLevelStateModel &rhs)
      : Model(rhs),
        StateModel(rhs),
        ZeroMeanGaussianModel(rhs),
        state_transition_matrix_(new IdentityMatrix(1)),
        state_variance_matrix_(new ConstantMatrixParamView(1, Sigsq_prm())),
        initial_state_mean_(rhs.initial_state_mean_),
        initial_state_variance_(rhs.initial_state_variance_) {}

  LocalLevelStateModel *LLSM::clone() const {
    return new LocalLevelStateModel(*this);
  }

  void LLSM::observe_state(const ConstVectorView &then,
                           const ConstVectorView &now, int time_now) {
    double current_level = now[0];
    double previous_level = then[0];
    double diff = current_level - previous_level;
    suf()->update_raw(diff);
  }

  uint LLSM::state_dimension() const { return 1; }

  void LLSM::simulate_state_error(RNG &rng, VectorView eta, int) const {
    eta[0] = rnorm_mt(rng, 0, sigma());
  }

  void LLSM::simulate_initial_state(RNG &rng, VectorView eta) const {
    eta[0] = rnorm_mt(rng, initial_state_mean_[0],
                      sqrt(initial_state_variance_(0, 0)));
  }

  Ptr<SparseMatrixBlock> LLSM::state_transition_matrix(int) const {
    return state_transition_matrix_;
  }

  Ptr<SparseMatrixBlock> LLSM::state_variance_matrix(int) const {
    return state_variance_matrix_;
  }

  Ptr<SparseMatrixBlock> LLSM::state_error_expander(int t) const {
    return state_transition_matrix(t);
  }

  Ptr<SparseMatrixBlock> LLSM::state_error_variance(int t) const {
    return state_variance_matrix(t);
  }

  SparseVector LLSM::observation_matrix(int) const {
    SparseVector ans(1);
    ans[0] = 1;
    return ans;
  }

  Vector LLSM::initial_state_mean() const { return initial_state_mean_; }

  SpdMatrix LLSM::initial_state_variance() const {
    return initial_state_variance_;
  }

  void LLSM::set_initial_state_mean(const Vector &m) {
    initial_state_mean_ = m;
  }

  void LLSM::set_initial_state_mean(double m) { initial_state_mean_[0] = m; }

  void LLSM::set_initial_state_variance(const SpdMatrix &v) {
    initial_state_variance_ = v;
  }

  void LLSM::set_initial_state_variance(double v) {
    initial_state_variance_(0, 0) = v;
  }

  void LLSM::update_complete_data_sufficient_statistics(
      int, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    if (state_error_mean.size() != 1 || state_error_variance.nrow() != 1 ||
        state_error_variance.ncol() != 1) {
      report_error(
          "Wrong size arguments to LocalLevelStateModel::"
          "update_complete_data_sufficient_statistics.");
    }
    double mean = state_error_mean[0];
    double var = state_error_variance(0, 0);
    suf()->update_expected_value(1.0, mean, var + square(mean));
  }

  void LLSM::increment_expected_gradient(
      VectorView gradient, int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    if (gradient.size() != 1 || state_error_mean.size() != 1 ||
        state_error_variance.nrow() != 1 || state_error_variance.ncol() != 1) {
      report_error(
          "Wrong size arguments to LocalLevelStateModel::"
          "increment_expected_gradient.");
    }
    double mean = state_error_mean[0];
    double var = state_error_variance(0, 0);
    double sigsq = ZeroMeanGaussianModel::sigsq();
    gradient[0] += (-.5 / sigsq) + .5 * (var + mean * mean) / (sigsq * sigsq);
  }

  //===========================================================================


}  // namespace BOOM
