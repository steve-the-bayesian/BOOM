/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/StateModels/ScalarStateModelAdapter.hpp"

namespace BOOM {

  namespace {
    using SSMMA = ScalarStateModelMultivariateAdapter;
  }

  SSMMA::ScalarStateModelMultivariateAdapter(
      MultivariateStateSpaceModelBase *host, int nseries)
      : host_(host),
        regression_coefficients_current_(false),
        regression_coefficients_(nseries),
        observation_coefficients_(
            new DenseSparseRankOneMatrixBlock(
                regression_coefficients_,
                SparseVector(1)))
  {
    int regression_dim = 1;
    for (int i = 0; i < nseries; ++i) {
      observation_models_.push_back(
          new RegressionModel(regression_dim));
      observation_models_.back()->coef_prm()->add_observer(
          this, [this]() {
                  this->regression_coefficients_current_ = false;
                });
    }
  }

  SSMMA::~ScalarStateModelMultivariateAdapter() {
    remove_observers();
  }

  SSMMA::ScalarStateModelMultivariateAdapter(const SSMMA &rhs)
      : SharedStateModel(rhs),
        ParamPolicy(rhs),
        NullDataPolicy(rhs),
        PriorPolicy(rhs),
        host_(rhs.host_),
        regression_coefficients_current_(false),
        regression_coefficients_(rhs.regression_coefficients_),
        observation_coefficients_(new DenseSparseRankOneMatrixBlock(
            regression_coefficients_,
            SparseVector(1)))

  {
    for (const auto &el : rhs.observation_models_) {
      observation_models_.push_back(el->clone());
      observation_models_.back()->coef_prm()->add_observer(
          this, [this]() {
                  this->regression_coefficients_current_ = false;
                });
    }
    for (const auto &el : rhs.component_models_) {
      add_state(el->clone());
    }
  }

  SSMMA & SSMMA::operator=(const SSMMA &rhs) {
    if (&rhs != this) {
      remove_observers();
      SharedStateModel::operator=(rhs);
      ParamPolicy::operator=(rhs);
      NullDataPolicy::operator=(rhs);
      PriorPolicy::operator=(rhs);
      host_ = rhs.host_;
      regression_coefficients_current_ = false;
      regression_coefficients_ = rhs.regression_coefficients_;
      observation_coefficients_ = new DenseSparseRankOneMatrixBlock(
          regression_coefficients_, SparseVector(1));

      observation_models_.clear();
      for (const auto &el : rhs.observation_models_) {
        observation_models_.push_back(el->clone());
        observation_models_.back()->coef_prm()->add_observer(
            this, [this]() {
                    this->regression_coefficients_current_ = false;
                  });
      }

      component_models_.clear();
      for (const auto &el : rhs.component_models_) {
        add_state(el->clone());
      }
    }
    return *this;
  }

  SSMMA * SSMMA::clone() const {return new SSMMA(*this);}

  void SSMMA::observe_state(const ConstVectorView &then,
                            const ConstVectorView &now,
                            int time_now) {
    int start = 0;
    for (size_t s = 0; s < component_models_.size(); ++s) {
      StateModelBase *state = component_models_.state_model(s);
      ConstVectorView then_view(then, start, state->state_dimension());
      ConstVectorView now_view(now, start, state->state_dimension());
      state->observe_state(then, now, time_now);
      start += state->state_dimension();
    }
  }

  void SSMMA::simulate_state_error(RNG &rng, VectorView eta, int t) const {
    int start = 0;
    for (size_t s = 0; s < component_models_.size(); ++s) {
      const StateModelBase *state = component_models_.state_model(s);
      VectorView error_component(eta, start, state->state_error_dimension());
      state->simulate_state_error(rng, error_component, t);
    }
  }

  void SSMMA::simulate_initial_state(RNG &rng, VectorView eta) const {
    int start = 0;
    for (size_t s = 0; s < component_models_.size(); ++s) {
      const StateModelBase *state = component_models_.state_model(s);
      VectorView component(eta, start, state->state_dimension());
      state->simulate_initial_state(rng, component);
    }
  }

  // The observation coefficients are Beta * Z[t], where Z[t] is the set of
  // observation coefficients from the component state models and Beta is the
  // set of regression coefficients.  If the dimension of y[t] is m x 1 and the
  // dimension of the managed state is p x 1, then Beta is m x 1, Z[t] is 1 x p,
  // and so Beta * Z[t] is m x p, but it is a product of two rank 1 matrices.
  Ptr<SparseMatrixBlock> SSMMA::observation_coefficients(
      int t, const Selector &observed) const {
    observation_coefficients_->update(
        regression_coefficients(),
        component_observation_coefficients(t));
    return observation_coefficients_;
  }

  Vector SSMMA::initial_state_mean() const {
    Vector ans;
    for (const auto &mod : component_models_) {
      ans.concat(mod->initial_state_mean());
    }
    return ans;
  }

  SpdMatrix SSMMA::initial_state_variance() const {
    std::vector<SpdMatrix> blocks;
    for (const auto &mod : component_models_) {
      blocks.push_back(mod->initial_state_variance());
    }
    return block_diagonal_spd(blocks);
  }

  void SSMMA:: update_complete_data_sufficient_statistics(
      int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error("not implemented");
  }

  void SSMMA::increment_expected_gradient(
      VectorView gradient, int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error("not implemented");
  }

  const Vector &SSMMA::regression_coefficients() const {
    if (!regression_coefficients_current_) {
      for (size_t i = 0; i < observation_models_.size(); ++i) {
        regression_coefficients_[i] = observation_models_[i]->Beta()[0];
      }
      regression_coefficients_current_ = true;
    }
    return regression_coefficients_;
  }

  SparseVector SSMMA::component_observation_coefficients(int t) const {
    SparseVector ans;
    for (const auto &mod : component_models_) {
      ans.concatenate(mod->observation_matrix(t));
    }
    return ans;
  }

  void SSMMA::remove_observers() {
    for (const auto &reg : observation_models_) {
      reg->coef_prm()->remove_observer(this);
    }
  }

} // namespace BOOM
