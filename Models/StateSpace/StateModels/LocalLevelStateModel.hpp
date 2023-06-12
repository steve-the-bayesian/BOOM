#ifndef BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
#define BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
// Copyright 2019 Steven L. Scott.
// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"

#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"

namespace BOOM {

  // The local level state model assumes
  //
  //      y[t] = mu[t] + epsilon[t]
  //   mu[t+1] = mu[t] + eta[t].
  //
  // That is, it is a random walk observed in noise.  It is the simplest useful
  // state model.
  class LocalLevelStateModel : virtual public StateModel, public ZeroMeanGaussianModel {
   public:
    explicit LocalLevelStateModel(double sigma = 1);
    LocalLevelStateModel(const LocalLevelStateModel &rhs);
    LocalLevelStateModel *clone() const override;
    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    uint state_dimension() const override;
    uint state_error_dimension() const override { return 1; }
    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    void simulate_initial_state(RNG &rng, VectorView eta) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    SparseVector observation_matrix(int t) const override;

    Vector initial_state_mean() const override;
    SpdMatrix initial_state_variance() const override;

    void set_initial_state_mean(double m);
    void set_initial_state_mean(const Vector &m);
    void set_initial_state_variance(const SpdMatrix &v);
    void set_initial_state_variance(double v);

    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

   private:
    Ptr<IdentityMatrix> state_transition_matrix_;
    Ptr<ConstantMatrixParamView> state_variance_matrix_;
    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

  //===========================================================================
  class DynamicInterceptLocalLevelStateModel
      : public LocalLevelStateModel,
        public DynamicInterceptStateModel {
   public:
    explicit DynamicInterceptLocalLevelStateModel(double sigma = 1.0)
        : LocalLevelStateModel(sigma)
    {}

    DynamicInterceptLocalLevelStateModel *clone() const override {
      return new DynamicInterceptLocalLevelStateModel(*this);
    }

    bool is_pure_function_of_time() const override {return true;}

    Ptr<SparseMatrixBlock> observation_coefficients(
        int t,
        const StateSpace::TimeSeriesRegressionData &data_point) const override {
      // In single threaded code we could optimize here by creating a single
      // IdenticalRowsMatrix and changing the number of rows each time.
      //
      // In multi-threaded code that would create a race condition.
      return new IdenticalRowsMatrix(observation_matrix(t),
                                     data_point.sample_size());
    }
  };

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_LOCAL_LEVEL_STATE_MODEL_HPP
