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

#include "Models/HMM/GeneralHmmStateSpaceWrapper.hpp"
#include "distributions.hpp"
#include "LinAlg/Cholesky.hpp"

namespace BOOM {

  namespace {
    using GHSW = GeneralHmmStateSpaceWrapper;
  }  // namespace

  GHSW::GeneralHmmStateSpaceWrapper(const Ptr<ScalarStateSpaceModelBase> &model)
      : DeferredDataPolicy(model),
        model_(model)
  {
    CompositeParamPolicy::add_model(model_);
  }

  GHSW::GeneralHmmStateSpaceWrapper(const GHSW &rhs)
      : Model(rhs),
        GeneralContinuousStateHmm(rhs),
        CompositeParamPolicy(rhs),
        DeferredDataPolicy(rhs),
        NullPriorPolicy(rhs),
        model_(rhs.model_->clone())
  {
    CompositeParamPolicy::add_model(model_);
    DeferredDataPolicy::set_model(model_);
  }

  GHSW & GHSW::operator=(const GHSW &rhs) {
    if (&rhs != this) {
      GeneralContinuousStateHmm::operator=(rhs);
      CompositeParamPolicy::operator=(rhs);
      DeferredDataPolicy::operator=(rhs);
      NullPriorPolicy::operator=(rhs);
      model_ = rhs.model_->clone();
      CompositeParamPolicy::add_model(model_);
      DeferredDataPolicy::set_model(model_);
    }
    return *this;
  }

  GHSW * GHSW::clone() const {
    return new GHSW(*this);
  }
  
  double GHSW::log_observation_density(
      const Data &observed_data, const Vector &state, int time_index,
      const Vector &parameters) const {
    ParameterHolder params(model_.get(), parameters);
    double observation_mean = model_->observation_matrix(time_index).dot(state);
    double observation_variance = model_->observation_variance(time_index);
    double observation_sd = sqrt(observation_variance);
    double y = dynamic_cast<const DoubleData &>(observed_data).value();
    return dnorm(y, observation_mean, observation_sd, true);
  }

  double GHSW::log_transition_density(
      const Vector &new_state, const Vector &old_state, int old_time,
      const Vector &parameters) const {
    ParameterHolder params(model_.get(), parameters);
    Vector scaled_change =
        model_->state_error_expander(old_time)->left_inverse(
            new_state
            - (*model_->state_transition_matrix(old_time)) * old_state);
    // The distribution of 'change' is RQR, which may be less than full rank.
    // The distribution of scaled_change is Q.

    // Deal with variances through the Cholesky decomposition, so that you
    // don't need to decompose twice (for inverse and log determinant of the
    // inverse).  The log determinant of the inverse matrix is -1 times the
    // log determinant of the original.
    Cholesky variance_cholesky(model_->state_error_variance(old_time)->dense());
    return dmvn_zero_mean(scaled_change,
                          variance_cholesky.inv(),
                          -variance_cholesky.logdet(),
                          true);
  }


  
  
}  // namespace BOOM
