// Copyright 2018 Google LLC. All Rights Reserved.
#ifndef BOOM_STATE_SPACE_STATIC_INTERCEPT_STATE_MODEL_HPP
#define BOOM_STATE_SPACE_STATIC_INTERCEPT_STATE_MODEL_HPP
/*
  Copyright (C) 2017 Steven L. Scott

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

#include "Models/StateSpace/StateModels/StateModel.hpp"

namespace BOOM {

  // A state component consisting of a intercept that does not change over time.
  // This is not necessary if the model contains a nonstationary trend component
  // (e.g. LocalLevel), but it is useful for modeling a seasonal pattern around
  // a nonzero intercept, for example.  Also useful for modeling an AR(1) trend
  // around a nonzero intercept.
  //
  // The state of this model is entirely determined by its initial distribution.
  // The transition matrix T is the number 1, and the residual variance matrix
  // is the number 0.
  class StaticInterceptStateModel : virtual public StateModel {
   public:
    StaticInterceptStateModel();
    StaticInterceptStateModel(const StaticInterceptStateModel &rhs) = default;
    StaticInterceptStateModel *clone() const override {
      return new StaticInterceptStateModel(*this);
    }

    // There is nothing to do here.
    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override {}

    uint state_dimension() const override { return 1; }

    // The model is deterministic, so there is no state error.
    uint state_error_dimension() const override { return 1; }

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override {
      eta[0] = 0.0;
    }

    void simulate_initial_state(RNG &rng, VectorView eta) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override {
      return state_transition_matrix_;
    }

    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override {
      return state_variance_matrix_;
    }

    Ptr<SparseMatrixBlock> state_error_expander(int t) const override {
      return state_transition_matrix_;
    }

    Ptr<SparseMatrixBlock> state_error_variance(int t) const override {
      return state_variance_matrix_;
    }

    SparseVector observation_matrix(int t) const override {
      return observation_matrix_;
    }

    Vector initial_state_mean() const override { return initial_state_mean_; }

    SpdMatrix initial_state_variance() const override {
      return initial_state_variance_;
    }

    // The initial state mean and variance are the _prior_ mean and variance for
    // the static intercept.  Its value is drawn by the data augmentation
    // algorithm, so there is no need to assign a posterior sampler.
    void set_initial_state_mean(double mean);
    void set_initial_state_variance(double variance);

    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override {}

    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override {}

    // There are no parameters to learn for this model, it holds no data, and as
    // mentioned above there is no need for a separate posterior sampler.  The
    // following pure virtual functions declared in ModelTypes.hpp need to be
    // overridden with no-ops.

    std::vector<Ptr<Params>> parameter_vector() override {
      return std::vector<Ptr<Params>>();
    }
    const std::vector<Ptr<Params>> parameter_vector() const override {
      return std::vector<Ptr<Params>>();
    }
    void add_data(const Ptr<Data> &) override {}
    void clear_data() override {}
    void combine_data(const Model &other, bool just_suf = true) override {}
    void sample_posterior() override {}
    double logpri() const override { return 0; }
    int number_of_sampling_methods() const override { return 0; }
    PosteriorSampler *sampler(int i) override { return nullptr; }
    PosteriorSampler const *const sampler(int i) const override {
      return nullptr;
    }

   private:
    Ptr<IdentityMatrix> state_transition_matrix_;
    Ptr<ZeroMatrix> state_variance_matrix_;
    SparseVector observation_matrix_;

    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_STATIC_INTERCEPT_STATE_MODEL_HPP
