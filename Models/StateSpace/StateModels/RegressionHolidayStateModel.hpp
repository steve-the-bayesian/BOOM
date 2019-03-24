#ifndef BOOM_STATE_SPACE_REGRESSION_HOLIDAY_STATE_MODEL_HPP_
#define BOOM_STATE_SPACE_REGRESSION_HOLIDAY_STATE_MODEL_HPP_
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Policies/ManyParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/NullPriorPolicy.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "Models/StateSpace/StateModels/Holiday.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "cpputil/Date.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  //===========================================================================
  // A base class for sharing implementation details among different regression
  // based holiday state models.
  //
  // All regression based state models have in common that the state is trivial.
  // It is the number 1, with all regression information packed into the
  // observation matrix Z(t).
  class RegressionHolidayBaseImpl {
   public:
    // Args:
    //   time_of_first_observation: The date of the observation associated with
    //     time index 0.
    //   residual_variance: The residual variance parameter from the observation
    //     equation.  This parameter is shared by the regression model or models
    //     in the state transition equation.  This abstraction can only be used
    //     with Gaussian observation models.  If we were to allow a bit of state
    //     error then we could introduce a free residual_variance parameter
    //     here.
    RegressionHolidayBaseImpl(const Date &time_of_first_observation,
                              const Ptr<UnivParams> &residual_variance);

    // Observing the time dimension builds the mapping between integer t and
    // which day of which holiday is active at time t.
    void observe_time_dimension(int max_time);

    // Add a holiday to the set of holidays represented by the model.
    void add_holiday(const Ptr<Holiday> &holiday);
    const Holiday *holiday(int t) const {
      return (t >= 0 && t < holidays_.size()) ? holidays_[t].get() : nullptr;
    }

    // The state of a regression model is just the number 1.  This state gets
    // multiplied by Z_t (observation_matrix) containing the results of the
    // linear predictor at time t.
    int state_dimension() const { return 1; }
    int state_error_dimension() const { return 1; }

    // The state is deterministic, so simulating state error means filling in a
    // zero for the first element.
    void simulate_state_error(VectorView eta) const {
      eta[0] = 0.0;
    }

    // The state value is always 1, so simulating the initial state just fills
    // eta with 1.
    void simulate_initial_state(VectorView eta) const { eta[0] = 1.0; }

    // The state transition matrix is the identity.
    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const {
      return state_transition_matrix_;
    }
    // The state variance matrix is zero.
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const {
      return state_variance_matrix_;
    }
    // The error expander is kind of ill defined because there is no error.
    Ptr<SparseMatrixBlock> state_error_expander(int t) const {
      return state_error_expander_;
    }
    // There is no error so the error variance is of size zero.
    Ptr<SparseMatrixBlock> state_error_variance(int t) const {
      return state_error_variance_;
    }

    // The index of the holiday model active at time t.
    //
    // This function assumes that holidays have been added using add_holiday(),
    // and that observe_time_dimension() has been called with a number larger
    // than t.
    int which_holiday(int t) const {
      return (t >= 0 && t < which_holiday_.size()) ? which_holiday_[t] : -1;
    }

    // The number of days into the influence window of the active holiday at
    // time t, or -1 if no holiday is active at time t.
    //
    // This function assumes that holidays have been added using add_holiday(),
    // and that observe_time_dimension() has been called with a number larger
    // than t.
    int which_day(int t) const {
      return (t >= 0 && t < which_day_.size()) ? which_day_[t] : -1;
    }

    const Vector &initial_state_mean() const { return initial_state_mean_; }
    const SpdMatrix &initial_state_variance() const {
      return initial_state_variance_;
    }

    // The residual variance parameter from the observation equation.
    Ptr<UnivParams> residual_variance() { return residual_variance_; }

    // The numerical value held by the residual variance parameter.
    double residual_variance_value() const {
      return residual_variance_->value();
    }

    void set_residual_variance_prm(const Ptr<UnivParams> &sigsq) {
      residual_variance_ = sigsq;
    }

    static Ptr<UnivParams> extract_residual_variance_parameter(
        ScalarStateSpaceModelBase &model);

   private:
    Date time_of_first_observation_;
    Ptr<UnivParams> residual_variance_;
    std::vector<Ptr<Holiday>> holidays_;

    // State space model matrices.  These are trivial.
    Ptr<IdentityMatrix> state_transition_matrix_;  // The 1x1 identity.
    Ptr<ZeroMatrix> state_variance_matrix_;        // The 1x1 zero matrix.
    Ptr<IdentityMatrix> state_error_expander_;     // The 1x1 identity.
    Ptr<ZeroMatrix> state_error_variance_;         // 1x1

    // A mapping from integer time t to which holiday is active at time t, and
    // which day in the holiday is active at time t.  These are filled when
    // observe_time_dimension is called.
    std::vector<int> which_holiday_;
    std::vector<int> which_day_;

    // The state is alwasy 1, so the mean is 1, and the variance is zero.
    Vector initial_state_mean_;
    SpdMatrix initial_state_variance_;
  };

  //===========================================================================
  // A RegressionHolidayStateModel describes a set of holidays using a
  // regression on dummy variables.  Each holiday is described by an influence
  // window with a dummy variable and associated regression coefficient for each
  // day in the window.  These dummy variables never co-occur, so X'X is a
  // diagonal matrix with diagonal elements containing occurrance counts for
  // each day.
  //
  // This is an abstract class because a concrete model class member is needed
  // to implement observe_state.
  class RegressionHolidayStateModel : virtual public StateModel,
                                      public ManyParamPolicy,
                                      public NullDataPolicy,
                                      public NullPriorPolicy {
   public:
    // Args:
    //   time_of_first_observation: The date of the observation associated with
    //     time index 0.
    //   residual_variance: The residual variance parameter from the observation
    //     equation.
    //   prior: A prior on the typical size of a daily effect.  The same prior
    //     is used for all effects.
    //   rng: A random number generator used to seed the random number generator
    //     used for posterior sampling.
    RegressionHolidayStateModel(const Date &time_of_first_observation,
                                const Ptr<UnivParams> &residual_variance,
                                const Ptr<GaussianModel> &prior,
                                RNG &seeding_rng = GlobalRng::rng);
    RegressionHolidayStateModel(const RegressionHolidayStateModel &rhs);
    RegressionHolidayStateModel &operator=(
        const RegressionHolidayStateModel &rhs);
    RegressionHolidayStateModel(RegressionHolidayStateModel &&rhs) = default;
    RegressionHolidayStateModel &operator=(RegressionHolidayStateModel &&rhs) =
        default;

    RegressionHolidayStateModel *clone() const override = 0;

    // Add a holiday to the set of holidays modeled by this object.
    void add_holiday(const Ptr<Holiday> &holiday);

    // Clear all sufficient statistics for the managed holidays.
    void clear_data() override {
      int number_of_holidays = daily_totals_.size();
      for (int i = 0; i < number_of_holidays; ++i) {
        daily_totals_[i] = 0;
        daily_counts_[i] = 0;
      }
    }

    // The vector of effects for the given holiday.  The size of the return
    // value matches the window width of the associated holiday, with each
    // element representing the expected increment to the response variable on
    // that day of the holiday influence window.
    //
    // Args:
    //   holiday:  The index of the desired holiday.
    const Vector &holiday_pattern(int holiday) const {
      return holiday_mean_contributions_[holiday]->value();
    }

    // Args:
    //   holiday:  The index of the desired holiday.
    //   pattern: A vector with size matching the window width of the holiday
    //     being assigned.
    void set_holiday_pattern(int holiday, const Vector &pattern) {
      holiday_mean_contributions_[holiday]->set(pattern);
    }

    void sample_posterior() override;

    // The numerical value of the residual variance parameter from the
    // observation equation.
    double residual_variance() const {
      return impl_.residual_variance_value();
    }

    void observe_time_dimension(int max_time) override;

    uint state_dimension() const override {
      return impl_.state_dimension();
    }

    uint state_error_dimension() const override {
      return impl_.state_error_dimension();
    }

    // Calling this throws an exception.
    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override {
      report_error("Not implemented.");
    }

    // Calling this throws an exception.
    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override {
      report_error("Not implemented.");
    }

    // The state is fully deterministic, so simulating state error means filling
    // a 1-dimensional zero.
    void simulate_state_error(RNG &rng, VectorView eta, int t) const override {
      impl_.simulate_state_error(eta);
    }
    void simulate_initial_state(RNG &rng, VectorView eta) const override {
      impl_.simulate_initial_state(eta);
    }
    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override {
      return impl_.state_transition_matrix(t);
    }
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override {
      return impl_.state_variance_matrix(t);
    }
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override {
      return impl_.state_error_expander(t);
    }
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override {
      return impl_.state_error_variance(t);
    }

    // The return value is size 1.  It contains the state contribution from the
    // holiday at time t, which is zero if no holiday is active.
    SparseVector observation_matrix(int t) const override;

    Vector initial_state_mean() const override {
      return impl_.initial_state_mean();
    }
    SpdMatrix initial_state_variance() const override {
      return impl_.initial_state_variance();
    }

    // Sufficient statistics for each holiay.  These are mainly here for
    // testing.
    const Vector &daily_counts(int holiday) const {
      return daily_counts_[holiday];
    }
    const Vector &daily_totals(int holiday) const {
      return daily_totals_[holiday];
    }

    Ptr<VectorParams> holiday_pattern_parameter(int i) {
      return holiday_mean_contributions_[i];
    }

   protected:
    void increment_daily_suf(int holiday, int day, double incremental_total,
                             double incremental_count) {
      daily_totals_[holiday][day] += incremental_total;
      daily_counts_[holiday][day] += incremental_count;
    }

    const RegressionHolidayBaseImpl &impl() const {return impl_;}
    
   private:
    RegressionHolidayBaseImpl impl_;
    std::vector<Ptr<VectorParams>> holiday_mean_contributions_;
    std::vector<Vector> daily_totals_;
    std::vector<Vector> daily_counts_;

    Ptr<GaussianModel> prior_;
    RNG rng_;
  };

  //===========================================================================
  class ScalarStateSpaceModelBase;
  class ScalarRegressionHolidayStateModel
      : public RegressionHolidayStateModel {
   public:
    ScalarRegressionHolidayStateModel(const Date &time_of_first_observation,
                                      ScalarStateSpaceModelBase *model,
                                      const Ptr<GaussianModel> &prior,
                                      RNG &seeding_rng = GlobalRng::rng);
    
    ScalarRegressionHolidayStateModel *clone()const override {
      return new ScalarRegressionHolidayStateModel(*this);
    }
    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;
   
   private:
    const ScalarStateSpaceModelBase *model_;
  };

  //===========================================================================
  class DynamicInterceptRegressionModel;
  class DynamicInterceptRegressionHolidayStateModel
      : public RegressionHolidayStateModel,
        public DynamicInterceptStateModel {
   public:
    DynamicInterceptRegressionHolidayStateModel(
        const Date &time_of_first_observation,
        DynamicInterceptRegressionModel *model,
        const Ptr<GaussianModel> &prior,
        RNG &seeding_rng = GlobalRng::rng);
    
    DynamicInterceptRegressionHolidayStateModel *clone() const override {
      return new DynamicInterceptRegressionHolidayStateModel(*this);
    }

    Ptr<SparseMatrixBlock> observation_coefficients(
        int t,
        const StateSpace::TimeSeriesRegressionData &data_point) const override;
    
    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;
    bool is_pure_function_of_time() const override {
      return true;
    }

   private:
    DynamicInterceptRegressionModel *model_;
  };
  
}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_REGRESSION_HOLIDAY_STATE_MODEL_HPP_
