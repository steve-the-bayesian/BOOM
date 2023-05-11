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

#include "Models/StateSpace/Filters/KalmanFilterBase.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"

namespace BOOM {
  namespace Kalman {

    MarginalDistributionBase::MarginalDistributionBase(int dim, int time_index)
        : time_index_(time_index),
          state_mean_(dim),
          state_variance_(dim) {}

    void MarginalDistributionBase::set_state_variance(const SpdMatrix &var) {
      check_variance(var);
      state_variance_ = var;
    }

    void MarginalDistributionBase::increment_state_variance(
        const SpdMatrix &variance_increment) {
      state_variance_ += variance_increment;
      check_variance(state_variance_);
    }

    void MarginalDistributionBase::check_variance(const SpdMatrix &v) const {
      for (int i = 0; i < v.nrow(); ++i) {
        if (v(i, i) < 0.0) {
          // The check is for < 0 and not <= 0, because some models with
          // deterministic state can have zero variance.
          std::ostringstream err;
          err << "Variance can't be negative." << std::endl << v;
          report_error(err.str());
        }
      }
    }
  }  // namespace Kalman

  //===========================================================================
  KalmanFilterBase::KalmanFilterBase()
      : status_(NOT_CURRENT), log_likelihood_(negative_infinity()) {}

  std::ostream &KalmanFilterBase::print(std::ostream &out) const {
    for (int i = 0; i < size(); ++i) {
      out << (*this)[i].state_mean() << std::endl;
    }
    return out;
  }

  std::string KalmanFilterBase::to_string() const {
    std::ostringstream out;
    print(out);
    return out.str();
  }

  Matrix KalmanFilterBase::state_mean() const {
    Matrix ans;
    int time_dimension = size();
    if (time_dimension > 0) {
      Vector v0 = (*this)[0].state_mean();
      ans.resize(v0.size(), time_dimension);
      ans.col(0) = v0;
      for (int t = 1; t < time_dimension; ++t) {
        ans.col(t) = (*this)[t].state_mean();
      }
    }
    return ans;
  }

  double KalmanFilterBase::compute_log_likelihood() {
    if (status_ == NOT_CURRENT) {
      clear_loglikelihood();
      // Some optimizers need to compute log likelihood by trying various
      // combinations of parameter values.  If those parameter values result in
      // NaN's or Inf's (e.g. because of a negative variance) then code down the
      // stack will throw an exception.  This function is the right place to
      // catch that exception and just return negative infinity.
      try {
        // If the update() call succeeds it will set the status to CURRENT.
        update();
      } catch (...) {
        log_likelihood_ = negative_infinity();
        status_ = NOT_CURRENT;
      }
    }
    return log_likelihood_;
  }


  void KalmanFilterBase::clear_loglikelihood() {
    log_likelihood_ = 0;
    status_ = NOT_CURRENT;
  }

}  // namespace BOOM
