#ifndef BOOM_GENERAL_SEASONAL_STATE_MODEL_HPP_
#define BOOM_GENERAL_SEASONAL_STATE_MODEL_HPP_
/*
  Copyright (C) 2018 Steven L. Scott

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

  // A state model composed of a superposition of other state models.
  class CompositeStateModel : virtual public StateModel {
   public:
   private:
    std::vector<Ptr<StateModel>> state_;
  };
  
  // A seasonal state model where each season is its own state model.  Each
  // season is observed every S time periods

  // The seasons are constrained to sum to zero at each time period.
  class GeneralSeasonalStateModel
      : virtual public StateModel {
   public:
    explicit GeneralSeasonalStateModel(int season_duration);
    ~GeneralSeasonalStateModel(){}
    
    GeneralSeasonalStateModel(const GeneralSeasonalStateModel &rhs);
    GeneralSeasonalStateModel(GeneralSeasonalStateModel &&rhs);
    GeneralSeasonalStateModel & operator=(const GeneralSeasonalStateModel &rhs);
    GeneralSeasonalStateModel & operator=(GeneralSeasonalStateModel &&rhs);
    
    GeneralSeasonalStateModel * clone ()const override;

    void add_seasonal_model(const Ptr<StateModel> &seasonal_model);

    void observe_time_dimension(int max_time) override;
    void observe_state(const ConstVectorView &then,
                       const ConstVectorView &now,
                       int time_now) override;

    void observe_initial_state(const ConstVectorView &state) override;
    uint state_dimension() const override;
    uint state_error_dimension() const override;

    void update_complete_data_sufficient_statistics(
        int t const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;
    
    // Simulates the state eror at time t, for moving to time t+1.
    // Args:
    //   rng:  The random number generator to use for the simulation.
    //   eta: A view into the error term to be simulated.  ***NOTE*** eta.size()
    //     matches state_dimension(), not state_error_dimension().  If the error
    //     distribution is not full rank then some components of eta will be
    //     deterministic functions of others (most likely just zero).
    //   t: The time index of the error.  The convention is that state[t+1] =
    //     T[t] * state[t] + error[t], so errors at time t are part of the state
    //     at time t+1.
    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;

    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;

    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;

    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    SparseVector observation_matrix(int t) const override;

    Ptr<SparseMatrixBlock>
    dynamic_intercept_regression_observation_coefficients(
        int t const StateSpace::TimeSeriesRegressionData &data_point) const override;

    Vector initial_state_mean() const override;
    SpdMatrix initial_state_variance() const override;
    
   private:
    std::vector<Ptr<StateModel>> seasonal_models_;
    int season_duration_;
  };

  
}  // namespace BOOM

#endif  // BOOM_GENERAL_SEASONAL_STATE_MODEL_HPP_
