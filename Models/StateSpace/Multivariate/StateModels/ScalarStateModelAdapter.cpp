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

  SSMMA::ScalarStateModelMultivariateAdapter()
  {}

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
      VectorView error_component(eta, start, state->state_dimension());
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

  SparseVector SSMMA::component_observation_coefficients(int t) const {
    SparseVector ans;
    for (int s = 0; s < component_models_.size(); ++s) {
      ans.concatenate(component_models_.state_model(s)->observation_matrix(t));
    }
    return ans;
  }

  namespace {
    using Adapter =
        ConditionallyIndependentScalarStateModelMultivariateAdapter;

    using Host = ConditionallyIndependentMultivariateStateSpaceModelBase;
  }  // namespace

  Adapter::ConditionallyIndependentScalarStateModelMultivariateAdapter(
      Host *host,
      int nseries)
      : Base(),
        host_(host),
        sufficient_statistics_(nseries),
        observation_coefficient_slopes_(new ConstrainedVectorParams(
            Vector(nseries, 1.0),
            new ProportionalSumConstraint(nseries))),
        empty_(new EmptyMatrix)
  {
    ParamPolicy::add_params(observation_coefficient_slopes_);
  }

  Adapter *Adapter::clone() const {
    return new Adapter(*this);
  }

  void Adapter::clear_data() {
    for (auto &el : sufficient_statistics_) {
      el.clear();
    }
    Base::clear_data();
  }

  void Adapter::observe_state(const ConstVectorView &then,
                              const ConstVectorView &now,
                              int time_now) {
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->observe_state(
          state_component(then, s),
          state_component(now, s),
          time_now);
    }

    const Selector &observed(host_->observed_status(time_now));
    if (observed.nvars() > 0) {
      Vector residual_y =
          host_->adjusted_observation(time_now)
          - (*host_->observation_coefficients(time_now, observed)
             * host_->shared_state(time_now))
          + (*observation_coefficients(time_now, observed)) * now;

      double predictive_state = component_observation_coefficients(
          time_now).dot(now);

      for (int i = 0; i < observed.nvars(); ++i) {
        int I = observed.sparse_index(i);
        sufficient_statistics_[I].increment(predictive_state, residual_y[i]);
      }
    }
  }

  // The observation coefficients are Beta * Z[t], where Z[t] is the set of
  // observation coefficients from the component state models and Beta is the
  // set of regression coefficients.  If the dimension of y[t] is m x 1 and the
  // dimension of the managed state is p x 1, then Beta is m x 1, Z[t] is 1 x p,
  // and so Beta * Z[t] is m x p, but it is a product of two rank 1 matrices.
  Ptr<SparseMatrixBlock> Adapter::observation_coefficients(
      int t, const Selector &observed) const {
    if (observed.nvars() == 0) {
      return empty_;
    } else if (observed.nvars() == observed.nvars_possible()) {
      return new DenseSparseRankOneMatrixBlock(
          observation_coefficient_slopes_->value(),
          component_observation_coefficients(t));
    } else {
      return new DenseSparseRankOneMatrixBlock(
          observed.select(observation_coefficient_slopes_->value()),
          component_observation_coefficients(t));
    }
  }

  std::string ScalarRegressionSuf::print() const {
    std::ostringstream out;
    print(out);
    return out.str();
  }

  std::ostream & ScalarRegressionSuf::print(std::ostream &out) const {
    out << "n = " << count_ << "\n"
        << "xtx = " << xtx_ << "\n"
        << "xty = " << xty_ << "\n"
        << "yty = " << yty_ << "\n";
    return out;
  }

  std::ostream & operator<<(std::ostream &out, const ScalarRegressionSuf &suf) {
    return suf.print(out);
  }

} // namespace BOOM
