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

namespace BOOM {

  // A multivariate state space model that assumes conditionally independent
  // errors in the observation equation.  This model can be used to describe
  // time series that are quite highly correlated, but the correlations are
  // // assumed to be fully captured by the state equation.
  // class MultivariateStateSpaceModel
  //     : pubic ConditionallyIndependentMultivariateStateSpaceModelBase
  //       public IID_DataPolicy<VectorData>,
  //       public PriorPolicy {
  //  public:

  //   IndependentMvnModel *observation_model() override;
  //   const IndependentMvnModel *observation_model() const override;

  //   int time_dimension() const override;
  //   void observe_data_given_state(int t) override;

  //   // Simulate the next 'horizon' time points
  //   Matrix simulate_forecast(RNG &rng, int horizon, const Vector &final_state);

  //  private:
  //   Ptr<IndependentMvnModel> observation_model_;
  // };

}  // namespace BOOM

#endif // BOOM_MULTIVARIATE_STATE_SPACE_MODEL_HPP_
