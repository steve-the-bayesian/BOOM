#ifndef BOOM_HIERARCHICAL_REGRESSION_HOLIDAY_STATE_MODEL_HPP_
#define BOOM_HIERARCHICAL_REGRESSION_HOLIDAY_STATE_MODEL_HPP_
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

#include "Models/Hierarchical/HierarchicalGaussianRegressionModel.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "Models/StateSpace/StateModels/Holiday.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/Date.hpp"

namespace BOOM {

  // The state of the model is the number 1.  The model matrices are T = 1, R =
  // 0, Z = holiday coefficients.
  //
  // Usage idiom:
  //   HierarchicalRegressionHolidayStateModel model.(time0, residual_variance);
  //   // Add some holidays
  //   model.add_holiday(h1);
  //   model.add_holiday(h2);
  //   model.add_holiday(h3);
  //   // Set the sampler.
  //   NEW(HierarchicalGaussianRegressionAsisSampler, sampler)(
  //     model.model(), prior1, prior2, rng);
  //   model.model()->set_method(sampler);
  //   // Observe the time dimension, to build initial data structures.
  //   model.observe_time_dimension(83);
  class HierarchicalRegressionHolidayStateModel
      : public StateModel,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy
  {
   public:

    // Args:
    //   time_of_first_observation: The date on which the first observed data
    //     point took place.  Data are assumed to occur daily thereafter.
    //   residual_variance: The residual variance parameter from the observation
    //     equation.  NOTE: The fact that this parameter is constant restricts
    //     this state model from being used with observation models that assume
    //     time-dependent variance, including GLM's.
    // TODO:  Lift the residual variance restriction.
    HierarchicalRegressionHolidayStateModel(const Date &time_of_first_observation,
                                            const Ptr<UnivParams> &residual_variance);
    HierarchicalRegressionHolidayStateModel * clone() const override {
      return new HierarchicalRegressionHolidayStateModel(*this);
    }

    // Add a holiday to the set of holidays managed by this model.  The more,
    // similar holidays added the better, because there will be more
    // opportunities to borrow strength.  Each holiday must have the same sized
    // influence window.
    void add_holiday(const Ptr<Holiday> &holiday);
    
    // Tabulates the list of dates and which holidays are active when.
    void observe_time_dimension(int max_time) override;

    void observe_state(const ConstVectorView &then,
                       const ConstVectorView &now,
                       int time_now,
                       ScalarStateSpaceModelBase *model) override;

    void observe_dynamic_intercept_regression_state(
        const ConstVectorView &then,
        const ConstVectorView &now,
        int time_now,
        DynamicInterceptRegressionModel *model) override;

    uint state_dimension() const override {return 1;}

    uint state_error_dimension() const override {return 0;}

    // Calling this throws an exception.
    void update_complete_data_sufficient_statistics(
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override {
      report_error("Not implemented.");
    }

    void increment_expected_gradient(
        VectorView gradient,
        int t,
        const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override {
      report_error("Not implemented.");
    }

    //======================================================================
    // TODO: Iron out what to do in the case of a purely deterministic state
    // component.
    //
    // The state error is zero-dimensional, so calling this does nothing.
    void simulate_state_error(RNG &rng, VectorView eta, int t) const override {
    }

    void simulate_initial_state(RNG &rng, VectorView eta) const override {
      eta[0] = 1.0;
    }

    Ptr<SparseMatrixBlock> state_transition_matrix(int t) const override {
      return state_transition_matrix_;
    }
    Ptr<SparseMatrixBlock> state_variance_matrix(int t) const override {
      return state_variance_matrix_;
    }
    Ptr<SparseMatrixBlock> state_error_expander(int t) const override {
      return state_error_expander_;
    }
    Ptr<SparseMatrixBlock> state_error_variance(int t) const override {
      return state_error_variance_;
    }

    SparseVector observation_matrix(int t) const override;

    Ptr<SparseMatrixBlock>
    dynamic_intercept_regression_observation_coefficients(
        int t, const StateSpace::MultiplexedData &data_point) const override {
      return new IdenticalRowsMatrix(observation_matrix(t),
                                     data_point.total_sample_size());
    }
    
    Vector initial_state_mean() const override {return initial_state_mean_;}

    SpdMatrix initial_state_variance() const override {
      return initial_state_variance_;
    }

    void clear_data() override;

    HierarchicalGaussianRegressionModel *model() {return model_.get();}
    const HierarchicalGaussianRegressionModel *model() const {
      return model_.get();
    }

    const Vector &daily_dummies(int day) const {
      return daily_dummies_[day];
    }
    
   private:
    Date time_of_first_observation_;
    Ptr<UnivParams> residual_variance_;
    std::vector<Ptr<Holiday>> holidays_;

    // State space model matrices.  These are trivial.
    Ptr<IdentityMatrix> state_transition_matrix_;      // The 1x1 identity.
    Ptr<ZeroMatrix> state_variance_matrix_;
    Ptr<EmptyMatrix> state_error_expander_;
    Ptr<EmptyMatrix> state_error_variance_;

    // The model whose coefficients describe the holiday effects.  The posterior
    // sampler for this model must be set externally.
    Ptr<HierarchicalGaussianRegressionModel> model_;

    // The number 1.
    Vector initial_state_mean_;
    // The number 0.
    SpdMatrix initial_state_variance_;

    // A mapping from integer time t to which holiday is active at time t, and
    // which day in the holiday is active at time t.  These are filled when
    // observe_time_dimension is called.
    std::vector<int> which_holiday_;
    std::vector<int> which_day_;

    // Element d is a dummy variable with a 1 in position d and 0's elsewhere.
    std::vector<Vector> daily_dummies_;
  };
  
}  // namespace BOOM

#endif  //  BOOM_HIERARCHICAL_REGRESSION_HOLIDAY_STATE_MODEL_HPP_
