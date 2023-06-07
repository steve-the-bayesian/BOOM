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

#include "Models/StateSpace/Multivariate/StateModels/SharedSeasonal.hpp"
#include "distributions.hpp"

namespace BOOM {

  SharedSeasonalStateModel::SharedSeasonalStateModel(
      ConditionallyIndependentMultivariateStateSpaceModelBase *host,
      int number_of_factors,
      int nseasons,
      int season_duration)
      : host_(host),
        nseasons_(nseasons),
        season_duration_(season_duration),
        time_of_first_observation_(0),
        current_factors_(create_current_factors(
            number_of_factors, nseasons, season_duration)),
        observation_parameter_manager_(
            host_->nseries(), number_of_factors),
        observation_coefficients_current_(false)
  {
    create_transition_matrix(nseasons, number_of_factors);
    create_variance_matrices(nseasons, number_of_factors);
    create_error_expander(nseasons, number_of_factors);
    create_raw_observation_coefficients();
    set_observation_coefficient_observers();
    map_observation_coefficients();
    register_params();
  }

  SharedSeasonalStateModel::SharedSeasonalStateModel(const SharedSeasonalStateModel &rhs)
      : nseasons_(rhs.nseasons_),
        season_duration_(rhs.season_duration_),
        time_of_first_observation_(rhs.time_of_first_observation_),
        transition_(rhs.transition_->clone()),
        no_transition_(rhs.no_transition_->clone()),
        state_error_variance_(rhs.state_error_variance_->clone()),
        no_state_error_variance_(rhs.no_state_error_variance_->clone()),
        state_variance_matrix_(rhs.state_variance_matrix_->clone()),
        no_state_variance_matrix_(rhs.no_state_variance_matrix_->clone()),
        state_error_expander_(rhs.state_error_expander_->clone()),
        current_factors_(rhs.current_factors_),
        observation_parameter_manager_(rhs.observation_parameter_manager_),
        observation_coefficients_current_(false)
  {
    create_raw_observation_coefficients();
    set_observation_coefficient_observers();
    map_observation_coefficients();
    register_params();
  }

  SharedSeasonalStateModel & SharedSeasonalStateModel::operator=(
      const SharedSeasonalStateModel &rhs) {
    if (&rhs != this) {
      nseasons_ = rhs.nseasons_;
      season_duration_ = rhs.season_duration_;
      time_of_first_observation_ = rhs.time_of_first_observation_;
      transition_.reset(rhs.transition_->clone());
      no_transition_.reset(rhs.no_transition_->clone());
      state_error_variance_.reset(rhs.state_error_variance_->clone());
      no_state_error_variance_.reset(rhs.no_state_error_variance_->clone());
      state_variance_matrix_.reset(rhs.state_variance_matrix_->clone());
      no_state_error_variance_.reset(rhs.no_state_error_variance_->clone());
      state_error_expander_.reset(rhs.state_error_expander_->clone());
      current_factors_ = rhs.current_factors_;
      observation_parameter_manager_ = rhs.observation_parameter_manager_;
    }
    register_params();
    return *this;
  }

  SharedSeasonalStateModel * SharedSeasonalStateModel::clone() const {
    return new SharedSeasonalStateModel(*this);
  }

  void SharedSeasonalStateModel::simulate_state_error(
      RNG &rng, VectorView eta, int t) const {
    for (int i = 0; i < eta.size(); ++i) {
      eta[i] = rnorm_mt(rng, 0, 1);
    }
  }

  void SharedSeasonalStateModel::simulate_initial_state(
      RNG &rng, VectorView eta) const {
    eta = rmvn_mt(rng, initial_state_mean(), initial_state_variance());
  }

  void SharedSeasonalStateModel::observe_initial_state(const ConstVectorView &state) {
    Vector dummy(0);
    observe_state(dummy, state, 0);
  }

  void SharedSeasonalStateModel::observe_state(
      const ConstVectorView &,
      const ConstVectorView &now,
      int time_now) {
    const Selector &observed(host()->observed_status(time_now));
    Vector residual_y = observation_parameter_manager_.compute_residual(
        now, time_now, host(), this);
    Vector seasonal_effects = current_factors_.select(now);
    for (int i = 0; i < observed.nvars(); ++i) {
      int series = observed.expanded_index(i);
      observation_parameter_manager_.suf(series)->add_data(
          seasonal_effects,
          residual_y[i],
          host()->weight(time_now, series));
    }
  }

  Ptr<SparseMatrixBlock>
  SharedSeasonalStateModel::state_transition_matrix(int t) const {
    if (new_season(t + 1)) {
      return transition_;
    } else {
      return no_transition_;
    }
  }

  Ptr<SparseMatrixBlock>
  SharedSeasonalStateModel::state_variance_matrix(int t) const {
    if (new_season(t + 1)) {
      return state_variance_matrix_;
    } else {
      return no_state_variance_matrix_;
    }
  }

  Ptr<SparseMatrixBlock>
  SharedSeasonalStateModel::state_error_expander(int t) const {
    return state_error_expander_;
  }

  Ptr<SparseMatrixBlock>
  SharedSeasonalStateModel::state_error_variance(int t) const {
    if (new_season(t + 1)) {
      return state_error_variance_;
    } else {
      return no_state_error_variance_;
    }
  }

  void SharedSeasonalStateModel::set_initial_state_mean(
      const Vector &initial_state_mean) {
    if (initial_state_mean.size() != state_dimension()) {
      std::ostringstream err;
      err << "Wrong size argument to set_initial_state_mean.  "
          << "State dimension is "
          << state_dimension()
          << " but argument was of size "
          << initial_state_mean.size()
          << ".";
      report_error(err.str());
    }
    initial_state_mean_ = initial_state_mean;
  }

  void SharedSeasonalStateModel::set_initial_state_variance(
      const SpdMatrix &initial_state_variance) {
    if (initial_state_variance.nrow() != state_dimension()) {
      std::ostringstream err;
      err << "Wrong size argument to set_initial_state_variance.  "
          << "State dimension is "
          << state_dimension()
          << " but argument was of row dimension "
          << initial_state_variance.nrow()
          << ".";
      report_error(err.str());
    }
    initial_state_variance_ = initial_state_variance;
  }

  // If there is a different observation pattern at each time period then each Z
  // matrix will need to have a different set of rows.  There's not really an
  // easy and efficient way to keep just one matrix and return it over and over.
  Ptr<SparseMatrixBlock> SharedSeasonalStateModel::observation_coefficients(
      int, const Selector &observed) const {
    map_observation_coefficients();
    return new StackedRegressionCoefficients(
        observed.select(raw_observation_coefficients_));
  }

  Selector SharedSeasonalStateModel::create_current_factors(
      int nfactors, int nseasons, int season_duration) const {
    Selector ans((nseasons - 1) * nfactors, false);
    int pos = 0;
    for (int i = 0; i < nfactors; ++i) {
      ans.add(pos);
      pos += nseasons -1;
    }
    return ans;
  }

  void SharedSeasonalStateModel::create_transition_matrix(
      int nseasons, int number_of_factors) {
    NEW(SeasonalStateSpaceMatrix, seasonal_transition)(nseasons);
    transition_.reset(new BlockDiagonalMatrixBlock(
        std::vector<Ptr<SparseMatrixBlock>>(
            number_of_factors, seasonal_transition)));
    no_transition_.reset(new IdentityMatrix(state_dimension()));
  }

  void SharedSeasonalStateModel::create_variance_matrices(
      int nseasons, int number_of_factors) {
    NEW(UpperLeftCornerMatrix, single_factor_variance)(nseasons - 1, 1.0);
    state_variance_matrix_.reset(new BlockDiagonalMatrixBlock(
        std::vector<Ptr<SparseMatrixBlock>>(
            number_of_factors, single_factor_variance)));
    no_state_variance_matrix_.reset(new ZeroMatrix(state_dimension()));

    state_error_variance_.reset(new IdentityMatrix(number_of_factors));
    no_state_error_variance_.reset(new ZeroMatrix(number_of_factors));
  }

  void SharedSeasonalStateModel::create_error_expander(
      int nseasons, int number_of_factors) {
    state_error_expander_.reset(new BlockDiagonalMatrix(
        std::vector<Ptr<SparseMatrixBlock>>(
            number_of_factors,
            new FirstElementSingleColumnMatrix(nseasons - 1))));
  }

  void SharedSeasonalStateModel::create_raw_observation_coefficients() {
    raw_observation_coefficients_.clear();
    for (int series = 0; series < nseries(); ++series) {
      NEW(GlmCoefs, row)(current_factors_.expand(
          observation_parameter_manager_.coefs(series)->Beta()),
          current_factors_);
      raw_observation_coefficients_.push_back(row);
    }
  }

  void SharedSeasonalStateModel::set_observation_coefficient_observers() {
    for (int series = 0; series < nseries(); ++series) {
      observation_parameter_manager_.coefs(series)->add_observer(
          this,
          [this]() {
            this->observation_coefficients_current_ = false;
          });
    }
  }

  void SharedSeasonalStateModel::map_observation_coefficients() const {
    if (!observation_coefficients_current_) {
      for (int series = 0; series < nseries(); ++series) {
        raw_observation_coefficients_[series]->set_Beta(
            current_factors_.expand(
                observation_parameter_manager_.coefs(series)->Beta()));
      }
    }
    observation_coefficients_current_ = true;
  }

  void SharedSeasonalStateModel::register_params() {
    ManyParamPolicy::clear();
    for (int series = 0; series < nseries(); ++series) {
      ManyParamPolicy::add_params(observation_parameter_manager_.coefs(series));
    }
  }

}  // namespace BOOM
