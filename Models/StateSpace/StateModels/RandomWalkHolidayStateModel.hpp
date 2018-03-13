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

#ifndef BOOM_RANDOM_WALK_HOLIDAY_STATE_MODEL_HPP_
#define BOOM_RANDOM_WALK_HOLIDAY_STATE_MODEL_HPP_

#include <memory>
#include <cpputil/Date.hpp>
#include <Models/StateSpace/StateModels/StateModel.hpp>
#include <Models/ZeroMeanGaussianModel.hpp>
#include <Models/StateSpace/StateModels/Holiday.hpp>

namespace BOOM {

  // TODO(stevescott): Need a new class of HolidayStateModel that
  // adjusts when floating holidays occur on weekends.

  // A RandomWalkHolidayStateModel assumes the holiday will produce an
  // effect of size delta[t], where the state model is
  //
  //    delta[t] = delta[t-1time_unit] + error,
  //
  // if t is in the holiday window, and delta[t] = 0 otherwise.  The
  // notation t-1time_unit indicates the same position in the time
  // window the last time the holiday occurred.  In other words,
  // holiday effects are modeled as a random walk (relative to the
  // last incidence of the holiday) inside the holiday influence
  // window.
  //
  // This model allows for arbitrarily shaped 'bumps' in both positive
  // and negative directions.  The state dimension is the holiday
  // window width (i.e. the dimension matches the number of days that
  // the holiday influences).  The transition matrix is always the
  // identity.  The error variance matrix is sigma^2 * outer(e[t]),
  // where e[t] is column t of the identity matrix.
  class RandomWalkHolidayStateModel :
      public StateModel,
      public ZeroMeanGaussianModel{
   public:
    // Args:
    //   holiday: A heap allocated pointer to a holiday that this
    //     model describes.  This object takes ownership of the
    //     pointer and deletes it upon destruction.
    //   time_zero: The date at t = 0, where t is an integer number of
    //     days.
    RandomWalkHolidayStateModel(Holiday *holiday, const Date &time_zero);
    RandomWalkHolidayStateModel * clone() const override;
    void observe_state(const ConstVectorView then,
                       const ConstVectorView now,
                       int time_now) override;

    uint state_dimension() const override;
    uint state_error_dimension() const override {
      return 1;
    }
    void simulate_state_error(RNG &rng, VectorView eta, int t) const override;

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override;
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override;
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override;

    SparseVector observation_matrix(int t) const override;
    Vector initial_state_mean() const override;
    SpdMatrix initial_state_variance() const override;
    void update_complete_data_sufficient_statistics(
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override;

    void set_initial_state_mean(const Vector &v);
    void set_initial_state_variance(const SpdMatrix &Sigma);
    void set_time_zero(const Date &time_zero);

   private:
    // TODO(stevescott): Make this a unique_ptr once available.
    std::shared_ptr<Holiday> holiday_;
    Date time_zero_;
    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
    Ptr<IdentityMatrix> identity_transition_matrix_;
    Ptr<ZeroMatrix> zero_state_variance_matrix_;

    std::vector<Ptr<SingleSparseDiagonalElementMatrixParamView> >
    active_state_variance_matrix_;
  };

}  // namespace BOOM

#endif //  BOOM_RANDOM_WALK_HOLIDAY_STATE_MODEL_HPP_
