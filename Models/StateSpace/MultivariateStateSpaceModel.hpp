#ifndef BOOM_MULTIVARIATE_STATE_SPACE_MODEL_HPP_
#define BOOM_MULTIVARIATE_STATE_SPACE_MODEL_HPP_
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/IndependentMvnModel.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"

namespace BOOM {

  // A multivariate state space model that assumes conditionally independent
  // errors in the observation equation.  This model can be used to describe
  // time series that are quite highly correlated, but the correlations are
  // assumed to be fully captured by the state equation.
  //
  // The model is
  //       y[t] = Z[t] * alpha[t] + error[t]
  // alpha[t+1] = T[t] * alpha[t] + state_error[t]
  //
  // Unlike the scalar model, y[t] is a vector, which may be partially observed.
  // The Z[t] coefficients combine the Z[t] coefficients from the scalar model
  // with a set of regression coefficients for the different elements of y[t].
  
  class MultivariateStateSpaceModel
      : public ConditionallyIndependentMultivariateStateSpaceModelBase,
        public IID_DataPolicy<PartiallyObservedVectorData>,
        public PriorPolicy {
   public:
    // Args:
    //   dim: The dimension of the time series to be modeled (i.e. the number of
    //     parallel time series).
    MultivariateStateSpaceModel(int dim);
    MultivariateStateSpaceModel(const MultivariateStateSpaceModel &rhs);
    MultivariateStateSpaceModel(MultivariateStateSpaceModel &&rhs) = default;
    MultivariateStateSpaceModel * clone() const override;
    MultivariateStateSpaceModel & operator=(
        const MultivariateStateSpaceModel &rhs);
    MultivariateStateSpaceModel & operator=(
        MultivariateStateSpaceModel &&rhs) = default;
    
    void add_shared_state(const Ptr<MultivariateStateModel> &state_model);
    
    IndependentMvnModel *observation_model() override;
    const IndependentMvnModel *observation_model() const override;

    void observe_data_given_state(int t) override;
    int time_dimension() const override { return dat().size(); }

    // Simulate the next 'horizon' time points
    Matrix simulate_forecast(RNG &rng, int horizon,
                             const Vector &final_state) const;

    const SparseKalmanMatrix *observation_coefficients(
        int t, const Selector &observed) const override;
    DiagonalMatrix observation_variance(int t) const override;
    double single_observation_variance(int t, int dim) const override;

    const Vector &observation(int t) const override;
    const Selector &observed_status(int t) const override;

    Matrix state_contributions(int which_state_model) const override;
    
    using ConditionallyIndependentMultivariateStateSpaceModelBase::get_filter;

    bool is_missing_observation(int t) const override {
      return dat()[t]->missing() != Data::observed;
    }

   private:
    Ptr<IndependentMvnModel> observation_model_;
    std::vector<Ptr<MultivariateStateModel>> state_models_;

    Ptr<BlockDiagonalMatrix> observation_coefficients_;
  };

}  // namespace BOOM

#endif // BOOM_MULTIVARIATE_STATE_SPACE_MODEL_HPP_
