#ifndef BOOM_STATE_SPACE_MULTIVARIATE_SHARED_SEASONAL_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_SHARED_SEASONAL_HPP_

/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/StateModels/SharedStateModel.hpp"

#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"
#include "Models/StateSpace/Multivariate/StateModels/ObservationParameterManager.hpp"

namespace BOOM {

  //===========================================================================
  // A seasonal state model for multivariate outcomes.  The latent state
  // consists of K seasonal factors, each with the same period (S) and season
  // duration.
  //
  // Each factor comprises a vector of size S-1, with the first element
  // containing the seasonal effect for the current period, and the remaining
  // S-2 elements containing the lag-1, lag-2, etc seasonal effects.
  //
  // Series i relates to factor k through a coefficient beta[i, k]
  //
  // The state vector for this model is an (S-1 x K) vector that can be thought
  // of as the stacked columns of an (S-1 x K) matrix with column k giving the
  // seasonal contribution for factor k.  The seasonal factor works as in the
  // SeasonalStateModel with gamma[t, 0, k] giving the contribution from factor
  // k at time t.  gamma[t, 1, k] is a time shift of last period's seasonal
  // factor (so gamma[t, 1, k] = gamma[t-1, 0, k]), as is each subsequent
  // element (so gamma[t, j, k] = gamma[t-1, j-1, k]).
  //
  // En tableau, the state at time t looks like the following, with columns
  // containing factors, and rows containing lags of factors:
  //
  // g[t, 0, 0]   g[t, 0, 1]   ...   g[t, 0, K]
  // g[t, 1, 0]   g[t, 1, 1]   ...   g[t, 1, K]
  // g[t, 2, 0]   g[t, 2, 1]   ...   g[t, 2, K]
  //                   ...
  // g[t, S-1, 0] g[t, S-1, 1] ...   g[t, S-1, K].
  //
  // The model matrices, T_t, Q_t, and R_t are block diagonal matrices, with
  // block k containing the corresponding matrices from a scalar
  // SeasonalStateModel.  Specifically:
  //   - The blocks of T_t are the transition matrices of a SeasonalStateModel.
  //   - The blocks of Q_t are the identity (each block is just the scalar 1), and
  //   - The blocks of R_t are the state error expanders from a seasonal state
  //     model.
  //
  // The state is linked to the observed data for series i by a set of
  // K-dimensional regression coefficients beta[i].
  // ===========================================================================
  class SharedSeasonalStateModel
      : public SharedStateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy
  {
   public:
    SharedSeasonalStateModel(
        ConditionallyIndependentMultivariateStateSpaceModelBase *host,
        int number_of_factors,
        int nseasons,
        int season_duration);

    SharedSeasonalStateModel(const SharedSeasonalStateModel &rhs);
    SharedSeasonalStateModel(SharedSeasonalStateModel &&rhs) = default;
    SharedSeasonalStateModel & operator=(const SharedSeasonalStateModel &rhs);
    SharedSeasonalStateModel & operator=(
        SharedSeasonalStateModel &&rhs) = default;

    SharedSeasonalStateModel * clone() const override;

    //---------------------------------------------------------------------------
    // Sizes of things.
    uint state_dimension() const override {
      return (nseasons() - 1) * (number_of_factors());
    }
    uint state_error_dimension() const override {return number_of_factors();}
    int nseries() const {return observation_parameter_manager_.nseries();}

    int number_of_factors() const {
      return observation_parameter_manager_.coefs(0)->size();
    }

    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;
    void simulate_initial_state(RNG &rng, VectorView eta) const override;

    //---------------------------------------------------------------------------
    // Do we REALLY need the host?
    // Right now it gets the observation status at time t and the number of series.
    ConditionallyIndependentMultivariateStateSpaceModelBase *host() {
      return host_;}
    const ConditionallyIndependentMultivariateStateSpaceModelBase *host() const {
      return host_;}

    //---------------------------------------------------------------------------
    // Member functions inherited from StateModel.
    void observe_initial_state(const ConstVectorView &state) override;
    void observe_state(const ConstVectorView &then,
                       const ConstVectorView &now,
                       int time_now) override;


    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    Vector initial_state_mean() const override {
      return initial_state_mean_;
    }
    void set_initial_state_mean(const Vector &initial_state_mean);

    SpdMatrix initial_state_variance() const override {
      return initial_state_variance_;
    }
    void set_initial_state_variance(const SpdMatrix &variance);

    //---------------------------------------------------------------------------
    // Member functions inherited from SharedStateModel.
    Ptr<SparseMatrixBlock> observation_coefficients(
        int t, const Selector &observed) const override;

    //---------------------------------------------------------------------------
    int nseasons() const {return nseasons_;}
    int season_duration() const {return season_duration_;}

    // Return true iff 'time' indexes the start of a new season.
    bool new_season(int time) const {
      time -= time_of_first_observation_;
      if (time < 0) {
        time -= season_duration_ * time;
      }
      return ((time % season_duration_) == 0);
    }

    const Ptr<WeightedRegSuf> & suf(int series) const {
      return observation_parameter_manager_.suf(series);
    }

    // The observation coefficients relating a particular observed series to the
    // current values of the different factors.  The dimension of these
    // coefficients is number_of_factors().
    Ptr<GlmCoefs> compressed_observation_coefficients(int series) {
      return observation_parameter_manager_.coefs(series);
    }

    const Ptr<GlmCoefs> compressed_observation_coefficients(int series) const {
      return observation_parameter_manager_.coefs(series);
    }

   private:
    //---------------------------------------------------------------------------
    // A set of utility functions to be called during construction.
    Selector create_current_factors(
        int nfactors, int nseasons, int season_duration) const;
    void create_transition_matrix(int nseasons, int number_of_factors);
    void create_variance_matrices(int nseasons, int number_of_factors);
    void create_error_expander(int nseasons, int number_of_factors);

    // Create the set of raw_observation_coefficients_ from current_factors_ and
    // the coefficients held in the observation_parameter_manager_.
    void create_raw_observation_coefficients();

    // Set an observer in each of the observation coefficients held by the
    // observation_parameter_manager_ so that if any of them change
    // observation_coefficients_current_ gets set to false.
    void set_observation_coefficient_observers();

    // Expand the coefficients held in observation_parameter_manager_ into
    // raw_observation_coefficients_, which in turn exposes them to
    // observation_coefficients_.
    void map_observation_coefficients() const;

    //---------------------------------------------------------------------------
    // Data section
    //---------------------------------------------------------------------------
    ConditionallyIndependentMultivariateStateSpaceModelBase *host_;

    int nseasons_;
    int season_duration_;
    int time_of_first_observation_;

    // There is one set of model matrices for observations occurring at the
    // start of a new season.  A second set covers the remaining observations

    Ptr<BlockDiagonalMatrixBlock> transition_;
    Ptr<IdentityMatrix> no_transition_;

    Ptr<IdentityMatrix> state_error_variance_;
    Ptr<ZeroMatrix> no_state_error_variance_;

    Ptr<BlockDiagonalMatrixBlock> state_variance_matrix_;
    Ptr<ZeroMatrix> no_state_variance_matrix_;

    Ptr<BlockDiagonalMatrix> state_error_expander_;

    // A Selector for picking the currently active values for each factor out
    // from among all the lags.  current_factors_.select(state) takes a state
    // vector of size state_dimension() and returns one of size
    // number_of_factors().
    Selector current_factors_;

    // The observation_parameter_manager_ keeps track of the observation
    // coefficients in a form that we can't directly use with the Kalman filter.
    // These coefficients link the observed series with the current set of
    // seasonal effects.  The Kalman filter needs them to be linked with the
    // lags as well.  To get around this problem we keep a second set of
    // observation coefficients with the larger (notional) dimension.  When the
    // observation coefficients are requested by the Kalman filter then we check
    // to see if the mapping is current, and refresh the map if it is out of
    // date.
    //
    // For this scheme to work, the raw_observation_coefficients_ must not be
    // exposed outside of this class.  They are intended to be reflections of
    // the coefficients held inside the observation_parameter_manager_.
    ObservationParameterManager observation_parameter_manager_;
    mutable bool observation_coefficients_current_;
    mutable std::vector<Ptr<GlmCoefs>> raw_observation_coefficients_;

    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

}



#endif  // BOOM_STATE_SPACE_MULTIVARIATE_SHARED_SEASONAL_HPP_
