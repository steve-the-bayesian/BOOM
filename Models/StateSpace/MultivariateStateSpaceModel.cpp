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

#include "Models/StateSpace/MultivariateStateSpaceModel.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using MSSM = MultivariateStateSpaceModel;
  }  // namespace

  MSSM::MultivariateStateSpaceModel(int dim)
      : ConditionallyIndependentMultivariateStateSpaceModelBase(dim),
        observation_model_(new IndependentMvnModel(dim)),
        observation_coefficients_(new BlockDiagonalMatrix)
  {}
        
  MSSM::MultivariateStateSpaceModel(const MSSM &rhs)
      : ConditionallyIndependentMultivariateStateSpaceModelBase(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        observation_model_(rhs.observation_model_->clone()),
        observation_coefficients_(new BlockDiagonalMatrix)
  {
    clear_state_models();
    for (int i = 0; i < rhs.state_models_.size(); ++i) {
      add_shared_state(rhs.state_models_[i]->clone());
    }
  }

  MSSM & MSSM::operator=(const MSSM &rhs) {
    if (&rhs != this) {
      ConditionallyIndependentMultivariateStateSpaceModelBase::operator=(rhs);
      DataPolicy::operator=(rhs);
      PriorPolicy::operator=(rhs);
      observation_model_.reset(rhs.observation_model_->clone());
      observation_coefficients_.reset(new BlockDiagonalMatrix);
      clear_state_models();
      for (int i = 0; i < rhs.state_models_.size(); ++i) {
        add_state(rhs.state_models_[i]->clone());
      }
    }
    return *this;
  }
  
  MSSM *MSSM::clone() const {return new MSSM(*this);} 

  void MSSM::add_shared_state(const Ptr<MultivariateStateModel> &state_model) {
    state_models_.push_back(state_model);
    StateSpaceModelBase::add_state(state_model);
  }
  
  IndependentMvnModel *MSSM::observation_model() {
    return observation_model_.get();
  }

  const IndependentMvnModel *MSSM::observation_model() const {
    return observation_model_.get();
  }

  void MSSM::observe_data_given_state(int t) {
    if (!is_missing_observation(t)) {
      const Selector &observed(observed_status(t));
      Vector mu = *observation_coefficients(t, observed) * state(t);
      const Vector &obs(observation(t));
      for (int i = 0; i < observed.nvars(); ++i) {
        int I = observed.indx(i);
        observation_model()->suf()->update_single_dimension(
            obs[I] - mu[I], I);
      }
    }
  }

  Matrix MSSM::simulate_forecast(RNG &rng, int horizon,
                                 const Vector &final_state) const {
    Matrix ans(horizon, nseries());
    int t0 = time_dimension();
    Vector state = final_state;
    Selector observed(nseries(), true);
    for (int t = 0; t < horizon; ++t) {
      state = simulate_next_state(rng, state, t + t0);
      ans.row(t) = rmvn_mt(
          rng,
          *observation_coefficients(t + t0, observed) * state,
          observation_variance(t + t0));
    }
    return ans;
  }

  const SparseKalmanMatrix *MSSM::observation_coefficients(
      int t, const Selector &observed) const {
    observation_coefficients_->clear();
    for (int i = 0; i < number_of_state_models(); ++i) {
      observation_coefficients_->add_block(
          state_models_[i]->observation_coefficients(t, observed));
    }
    return observation_coefficients_.get();
  }

  DiagonalMatrix MSSM::observation_variance(int t) const  {
    return observation_model()->diagonal_variance();
  }
  double MSSM::single_observation_variance(int t, int dim) const {
    return observation_model()->sigsq(dim);
  }

  const Vector &MSSM::observation(int t) const {
    return dat()[t]->value();
  }
  const Selector &MSSM::observed_status(int t) const {
    return dat()[t]->observation_status();
  }

  Matrix MSSM::state_contributions(int which_state_model) const {
    const Matrix &state(this->state());
    if (ncol(state) != time_dimension() || nrow(state) != state_dimension()) {
      ostringstream err;
      err << "state is the wrong size in "
          << "ScalarStateSpaceModelBase::state_contribution" << endl
          << "State contribution matrix has " << ncol(state) << " columns.  "
          << "Time dimension is " << time_dimension() << "." << endl
          << "State contribution matrix has " << nrow(state) << " rows.  "
          << "State dimension is " << state_dimension() << "." << endl;
      report_error(err.str());
    }
    if (nseries() <= 0) {
      report_error("Error in state_contributions: the dimension of y is "
                   "time varying.");
    }
    Matrix ans(time_dimension(), nseries());

    Selector observed(nseries(), true);
    for (int t = 0; t < time_dimension(); ++t) {
      ConstVectorView local_state(state_component(
          state.col(t), which_state_model));
      ans.row(t) = *(state_models_[which_state_model]->observation_coefficients(
          t, observed)) * local_state;
    }
    return ans;
  }

}  // namespace BOOM
