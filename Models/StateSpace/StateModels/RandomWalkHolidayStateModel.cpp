/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include <Models/StateSpace/StateModels/Holiday.hpp>
#include <Models/StateSpace/StateModels/RandomWalkHolidayStateModel.hpp>
#include <distributions.hpp>
#include <cpputil/report_error.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM {
  typedef RandomWalkHolidayStateModel RWHSM;
  RWHSM::RandomWalkHolidayStateModel(Holiday *holiday, const Date &time_zero)
      : holiday_(holiday),
        time_zero_(time_zero)
  {
    int dim = holiday->maximum_window_width();
    initial_state_mean_.resize(dim);
    initial_state_variance_.resize(dim);
    identity_transition_matrix_ = new IdentityMatrix(dim);
    zero_state_variance_matrix_ = new ZeroMatrix(dim);
    for(int i = 0; i < dim; ++i){
      NEW(SingleSparseDiagonalElementMatrixParamView, variance_matrix)(
          dim, Sigsq_prm(), i);
      active_state_variance_matrix_.push_back(variance_matrix);
    }
  }

  RandomWalkHolidayStateModel * RWHSM::clone()const{
    return new RandomWalkHolidayStateModel(*this);}

  void RWHSM::observe_state(const ConstVectorView then,
                            const ConstVectorView now,
                            int time_now){
    Date today = time_zero_ + time_now;
    if(holiday_->active(today)){
      Date holiday_date = holiday_->nearest(today);
      int position = today - holiday_->earliest_influence(holiday_date);
      double delta = now[position] - then[position];
      suf()->update_raw(delta);
    }
  }

  uint RWHSM::state_dimension()const{
    return holiday_->maximum_window_width();
  }

  void RWHSM::simulate_state_error(RNG &rng, VectorView eta, int t)const{
    Date now = time_zero_ + t;
    eta = 0;
    if(holiday_->active(now)){
      Date holiday_date(holiday_->nearest(now));
      int position = now - holiday_->earliest_influence(holiday_date);
      eta[position] = rnorm_mt(rng, 0, sigma());
    }
  }

  Ptr<SparseMatrixBlock>  RWHSM::state_transition_matrix(int t)const{
    return identity_transition_matrix_;
  }

  Ptr<SparseMatrixBlock> RWHSM::state_variance_matrix(int t)const{
    Date now = time_zero_ + t;
    if(holiday_->active(now)){
      Date holiday_date(holiday_->nearest(now));
      int position = now - holiday_->earliest_influence(holiday_date);
      return active_state_variance_matrix_[position];
    }
    return zero_state_variance_matrix_;
  }

  Ptr<SparseMatrixBlock> RWHSM::state_error_expander(int t) const {
    return state_transition_matrix(t);
  }

  Ptr<SparseMatrixBlock> RWHSM::state_error_variance(int t) const {
    return state_variance_matrix(t);
  }

  SparseVector RWHSM::observation_matrix(int t)const{
    Date now = time_zero_ + t;
    SparseVector ans(state_dimension());
    if(holiday_->active(now)){
      Date holiday_date(holiday_->nearest(now));
      int position = now - holiday_->earliest_influence(holiday_date);
      ans[position] = 1.0;
    }
    return ans;
  }

  Vector RWHSM::initial_state_mean()const{
    return initial_state_mean_;
  }

  SpdMatrix RWHSM::initial_state_variance()const{
    return initial_state_variance_;
  }

  void RWHSM::update_complete_data_sufficient_statistics(
      int t,
      const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    if (state_error_mean.size() != 1
        || state_error_variance.nrow() != 1
        || state_error_variance.ncol() != 1) {
      report_error("Wrong size argument to RandomWalkHolidayStateModel::"
                   "update_complete_data_sufficient_statistics");
    }
    double mean = state_error_mean[0];
    double var = state_error_variance(0, 0);
    suf()->update_expected_value(1.0, mean, var + square(mean));
  }


  void RWHSM::set_initial_state_mean(const Vector &v){
    initial_state_mean_ = v;
  }

  void RWHSM::set_initial_state_variance(const SpdMatrix &Sigma){
    initial_state_variance_ = Sigma;
  }

  void RWHSM::set_time_zero(const Date &time_zero){
    time_zero_ = time_zero;
  }

}  // namespace BOOM
