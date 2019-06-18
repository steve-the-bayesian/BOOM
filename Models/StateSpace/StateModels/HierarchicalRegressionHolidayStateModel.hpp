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

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "Models/Hierarchical/HierarchicalGaussianRegressionModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "Models/StateSpace/StateModels/Holiday.hpp"
#include "Models/StateSpace/StateModels/RegressionHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "cpputil/Date.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  // A model for describing groups of holidays with similar but non-identical
  // effects.  All holidays in this model must have the same window widths.
  //
  // If time t is on day d of the influence window associated with holiday h
  // then the contribution to the mean is beta(h, d).  The vectors beta[h]
  // follow a hierarchical model
  //
  //     beta[h] ~ N( b0,  V )
  //
  // where b0 is the typical influence pattern for a holiday, and V describes
  // how this pattern can vary from one holiday to the next.  This is a static
  // model in the sense that the effect of a specific holiday (e.g. Valentine's
  // day) is the same every year.  The model requires a hyperprior distribution
  // on (b0, V), which is to be handled by a PosteriorSampler designed for a
  // HierarchicalGaussianRegressionModel.
  //
  // This model borrows strength across holidays, which should be especialy
  // helpful when only a few years of daily data are observed, implying that
  // each holiday is poorly estimated.
  //
  // As with most regression state models, the state of the model is the number
  // 1.  The model matrices are T = 1, R = 0, Z = holiday coefficients.
  //
  // Usage idiom:
  //   HierarchicalRegressionHolidayStateModel model.(time0, residual_variance);
  //   model.add_holiday(h1);    
  //   model.add_holiday(h2);
  //   model.add_holiday(h3);
  //   // Set any sampler you like, but this is a good choice.
  //   NEW(HierarchicalGaussianRegressionAsisSampler, sampler)(
  //     model.model(), prior1, prior2, rng);
  //   model.model()->set_method(sampler);
  //   // Observe the time dimension, to build initial data structures.
  //   model.observe_time_dimension(83);
  class HierarchicalRegressionHolidayStateModel : virtual public StateModel,
                                                  public CompositeParamPolicy,
                                                  public NullDataPolicy,
                                                  public PriorPolicy {
   public:
    // Args:
    //   time_of_first_observation: The date on which the first observed data
    //     point took place.  Data are assumed to occur daily thereafter.
    //   residual_variance: The residual variance parameter from the observation
    //     equation.  NOTE: The fact that this parameter is constant restricts
    //     this state model from being used with observation models that assume
    //     time-dependent variance, including GLM's.
    HierarchicalRegressionHolidayStateModel(
        const Date &time_of_first_observation,
        const Ptr<UnivParams> &residual_variance);
    HierarchicalRegressionHolidayStateModel *clone() const override = 0;

    // Add a holiday to the set of holidays managed by this model.  The more,
    // similar holidays added the better, because there will be more
    // opportunities to borrow strength.  Each holiday must have the same sized
    // influence window.
    void add_holiday(const Ptr<Holiday> &holiday);

    void observe_time_dimension(int max_time) override;

    uint state_dimension() const override { return impl_.state_dimension(); }
    uint state_error_dimension() const override {
      return impl_.state_error_dimension();
    }

    // This is a required virtual function needed for MLE/MAP estimation, which
    // is not supported for this state model.  Calling this throws an exception.
    void update_complete_data_sufficient_statistics(
        int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override {
      report_error("Not implemented.");
    }

    // This is a required virtual function needed for MLE/MAP estimation, which
    // is not supported for this state model.  Calling this throws an exception.
    void increment_expected_gradient(
        VectorView gradient, int t, const ConstVectorView &state_error_mean,
        const ConstSubMatrix &state_error_variance) override {
      report_error("Not implemented.");
    }

    // The state error is zero-dimensional, so simulation is a no-op.
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

    // The observation matrix at time t is either 0 (if no holiday is active at
    // time t), or it contains the active holiday's contribution to the state
    // mean.
    SparseVector observation_matrix(int t) const override;

    Vector initial_state_mean() const override {
      return impl_.initial_state_mean();
    }

    SpdMatrix initial_state_variance() const override {
      return impl_.initial_state_variance();
    }

    // Clear all sufficient statistics for the managed holidays.
    void clear_data() override;

    // The hierarchical model describing the state effects.
    HierarchicalGaussianRegressionModel *model() { return model_.get(); }
    const HierarchicalGaussianRegressionModel *model() const {
      return model_.get();
    }

    // The regression model used here is based on daily dummy variables
    // indicating which day of the influence window is being observed.  None of
    // the dummy variables can co-occur, so this 'regression model' is really a
    // set of independent means.
    //
    // Returns the set of 'predictor variables' indicating the active day of the
    // influence window.  Element 'day' is 1, all other elements are 0.
    const Vector &daily_dummies(int day) const { return daily_dummies_[day]; }

   protected:
    const RegressionHolidayBaseImpl &impl() const {return impl_;}
    
   private:
    RegressionHolidayBaseImpl impl_;

    // The model whose coefficients describe the holiday effects.  The posterior
    // sampler for this model must be set externally.
    Ptr<HierarchicalGaussianRegressionModel> model_;

    // Element d is a dummy variable with a 1 in position d and 0's elsewhere.
    std::vector<Vector> daily_dummies_;
  };

  //===========================================================================
  class ScalarStateSpaceModelBase;
  class ScalarHierarchicalRegressionHolidayStateModel
      : public HierarchicalRegressionHolidayStateModel {
   public: 
    ScalarHierarchicalRegressionHolidayStateModel(
        const Date &time_of_first_observation,
        ScalarStateSpaceModelBase *model);

    ScalarHierarchicalRegressionHolidayStateModel *clone() const override {
      return new ScalarHierarchicalRegressionHolidayStateModel(*this);
    }

    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

   private:
    const ScalarStateSpaceModelBase *state_space_model_;
  };

  //===========================================================================
  class DynamicInterceptRegressionModel;
  class DynamicInterceptHierarchicalRegressionHolidayStateModel
      : public HierarchicalRegressionHolidayStateModel,
        public DynamicInterceptStateModel {
   public:
    DynamicInterceptHierarchicalRegressionHolidayStateModel(
        const Date &time_of_first_observation,
        DynamicInterceptRegressionModel *model);
          
    DynamicInterceptHierarchicalRegressionHolidayStateModel *
    clone() const override {
      return new DynamicInterceptHierarchicalRegressionHolidayStateModel(*this);
    }
    void observe_state(const ConstVectorView &then, const ConstVectorView &now,
                       int time_now) override;

    Ptr<SparseMatrixBlock> observation_coefficients(
        int t,
        const StateSpace::TimeSeriesRegressionData &data_point) const override;
    
    bool is_pure_function_of_time() const override {
      return true;
    }
    
   private:
    const DynamicInterceptRegressionModel *state_space_model_;
  };
  
}  // namespace BOOM

#endif  //  BOOM_HIERARCHICAL_REGRESSION_HOLIDAY_STATE_MODEL_HPP_
