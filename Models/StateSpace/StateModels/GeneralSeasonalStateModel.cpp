/*
  Copyright (C) 2005-2021 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/StateSpace/StateModels/GeneralSeasonalStateModel.hpp"

namespace BOOM {

  namespace {
    using GSLLT = GeneralSeasonalLLT;
  }  // namespace

  GSLLT::GeneralSeasonalLLT(int nseasons, int season_duration)
      : nseasons_(nseasons),
        season_duration_(season_duration)
  {
    build_subordinate_models();
    build_state_matrices();
  }

  GSLLT::GeneralSeasonalLLT(const GSLLT &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        nseasons_(rhs.nseasons_),
        season_duration_(rhs.season_duration_)
  {
    for (int s = 0; s < subordinate_models_.size(); ++s) {
      subordinate_models_.push_back(rhs.subordinate_models_[s]->clone());
    }
    build_state_matrices();
  }

  GSLLT::GeneralSeasonalLLT(GSLLT &&rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        nseasons_(rhs.nseasons_),
        season_duration_(rhs.season_duration_),
        state_transition_matrix_(rhs.state_transition_matrix_),
        state_error_variance_(rhs.state_error_variance_),
        state_variance_matrix_(rhs.state_variance_matrix_),
        state_error_expander_(rhs.state_error_expander_),
        subordinate_models_(rhs.subordinate_models_)
  {}

  GSLLT & GSLLT::operator=(const GSLLT &rhs) {
    if (&rhs != this) {
      CompositeParamPolicy::operator=(rhs);
      DataPolicy::operator=(rhs);

      nseasons_ = rhs.nseasons_;
      season_duration_ = rhs.season_duration_;
      subordinate_models_.clear();
      for (int i = 0; i < subordinate_models_.size(); ++i) {
        subordinate_models_.push_back(rhs.subordinate_models_[i]);
      }
    }
    return *this;
  }

  GSLLT & GSLLT::operator=(GSLLT &&rhs) {
    if (&rhs != this) {
      CompositeParamPolicy::operator=(rhs);
      DataPolicy::operator=(rhs);

      nseasons_ = rhs.nseasons_;
      season_duration_ = rhs.season_duration_;

      subordinate_models_ = std::move(rhs.subordinate_models_);
      state_transition_matrix_ = rhs.state_transition_matrix_;
      state_variance_matrix_ = rhs.state_variance_matrix_;
      state_error_variance_ = rhs.state_error_variance_;
      state_error_expander_ = rhs.state_error_expander_;
    }
    return *this;
  }

  GSLLT * GSLLT::clone() const {
    return new GSLLT(*this);
  }

  void GSLLT::observe_state(const ConstVectorView &then, const ConstVectorView &now,
                            int time_now) {
    for (int s = 0; s < subordinate_models_.size(); ++s) {
      subordinate_models_[s]->observe_state(
          ConstVectorView(then, 2 * s, 2),
          ConstVectorView(now, 2 * s, 2),
          time_now);
    }
  }

  void GSLLT::observe_initial_state(const ConstVectorView &state) {
    for (int s = 0; s < subordinate_models_.size(); ++s) {
      subordinate_models_[s]->observe_initial_state(
          ConstVectorView(state, 2*s, 2));
    }
  }

  void GSLLT::update_complete_data_sufficient_statistics(
      int t,
      const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error("Not implemented");
  }

  void GSLLT::increment_expected_gradient(
      VectorView gradient, int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error("Not implemented");
  }

  void GSLLT::simulate_state_error(RNG &rng, VectorView eta, int t) const {
    for (int s = 0; s < subordinate_models_.size(); ++s) {
      VectorView chunk(eta, 2 * s, 2);
      subordinate_models_[s]->simulate_state_error(rng, chunk, t);
    }
  }

  SparseVector GSLLT::observation_matrix(int t) const {
    SparseVector ans(state_dimension());
    int which = 2 * t % state_dimension();
    ans[which] = 1;
    return ans;
  }

  void GSLLT::set_initial_state_mean(const Vector &initial_state_mean) {
    if (initial_state_mean.size() != state_dimension()) {
      std::ostringstream err;
      err << "initial_state_mean has size " << initial_state_mean.size()
          << ". Expected size " << state_dimension()
          << ".";
      report_error(err.str());
    }
    initial_state_mean_ = initial_state_mean;
  }

  Vector GSLLT::initial_state_mean() const {
    if (initial_state_mean_.empty()) {
      report_error("initial_state_mean has not been set.");
    }
    return initial_state_mean_;
  }

  void GSLLT::set_initial_state_variance(const SpdMatrix &initial_state_variance) {
    if (initial_state_variance.nrow() != state_dimension()) {
      std::ostringstream err;
      err << "initial_state_variance has row dimension "
          << initial_state_variance.nrow()
          << ". Expected size " << state_dimension()
          << ".";
      report_error(err.str());
    }
    initial_state_variance_ = initial_state_variance;
  }

  SpdMatrix GSLLT::initial_state_variance() const {
    if (initial_state_variance_.nrow() == 0) {
      report_error("initial_state_variance has not been set.");
    }
    return initial_state_variance_;
  }

  void GSLLT::build_subordinate_models() {
    subordinate_models_.clear();
    CompositeParamPolicy::clear();
    for (int i = 0; i < nseasons_; ++i) {
      NEW(LocalLinearTrendStateModel, trend)();
      subordinate_models_.push_back(trend);
      CompositeParamPolicy::add_model(trend);
    }
  }

  void GSLLT::build_state_matrices() {
    NEW(BlockDiagonalMatrixBlock, base_transition_matrix)();
    state_error_expander_.reset(
        new SubsetEffectConstraintMatrix(2 * nseasons_, 2, 0));
    state_error_variance_.reset(new BlockDiagonalMatrixBlock);

    int fake_time = 0;
    for (int i = 0; i < subordinate_models_.size(); ++i) {
      base_transition_matrix->add_block(
          subordinate_models_[i]->state_transition_matrix(fake_time));
      state_error_variance_->add_block(
          subordinate_models_[i]->state_error_variance(fake_time));
    }

    state_transition_matrix_.reset(new ProductSparseMatrixBlock(
        state_error_expander_,
        base_transition_matrix));

    state_variance_matrix_.reset(new DenseMatrix(SpdMatrix(2 * nseasons_)));
    state_variance_matrix_current_ = false;
    for (int i = 0; i < subordinate_models_.size(); ++i ) {
      subordinate_models_[i]->Sigma_prm()->add_observer(
          this,
          [this]() {
            this->state_variance_matrix_current_ = false;
          });
    }
  }

  void GSLLT::update_state_variance_matrix() const {
    if (!state_variance_matrix_current_) {
      Matrix M = state_error_expander_->dense();
      SpdMatrix V = state_error_variance_->dense();
      state_variance_matrix_->set(sandwich(M, V));
      state_variance_matrix_current_ = true;
    }
  }

}  // namespace BOOM
