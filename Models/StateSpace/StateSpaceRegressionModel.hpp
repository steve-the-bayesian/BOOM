// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#ifndef BOOM_STATE_SPACE_REGRESSION_HPP_
#define BOOM_STATE_SPACE_REGRESSION_HPP_

#include <vector>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"

namespace BOOM {

  namespace StateSpace {
    // A data type representing one or more RegressionData observations at the
    // same time point.
    class MultiplexedRegressionData : public MultiplexedData {
     public:
      MultiplexedRegressionData();
      MultiplexedRegressionData(double y, const Vector &x);
      explicit MultiplexedRegressionData(
          const std::vector<Ptr<RegressionData>> &data);
      MultiplexedRegressionData *clone() const override;
      std::ostream &display(std::ostream &out) const override;

      // Add a RegressionData observation to this data point.  If dp
      void add_data(const Ptr<RegressionData> &dp);

      double adjusted_observation(const GlmCoefs &coefficients) const;
      const RegressionData &regression_data(int i) const;
      Ptr<RegressionData> regression_data_ptr(int i);

      double state_model_offset() const { return state_model_offset_; }
      void set_state_model_offset(double offset) {
        state_model_offset_ = offset;
      }

      // The total number of data points, both missing and observed.
      int total_sample_size() const override { return regression_data_.size(); }

      const Matrix &predictors() const { return predictors_; }

     private:
      std::vector<Ptr<RegressionData>> regression_data_;
      double state_model_offset_;
      Matrix predictors_;
    };
  }  // namespace StateSpace

  // A contemporaneous regression model, where y[t] =
  // beta.dot(X.row(t)) + state space.
  class StateSpaceRegressionModel
      : public ScalarStateSpaceModelBase,
        public IID_DataPolicy<StateSpace::MultiplexedRegressionData>,
        public PriorPolicy {
   public:
    // xdim is the dimension of the x's in the regression part of the
    // model.
    explicit StateSpaceRegressionModel(int xdim);

    // y is the time series of observations.  X is the design matrix,
    // with rows contemporaneous to y.  If some of the y's are
    // missing, use 'observed' to indicate which are observed.  The X's
    // must be fully observed.
    StateSpaceRegressionModel(
        const Vector &y, const Matrix &X,
        const std::vector<bool> &observed = std::vector<bool>());

    StateSpaceRegressionModel(const StateSpaceRegressionModel &rhs);
    StateSpaceRegressionModel *clone() const override;
    StateSpaceRegressionModel *deepclone() const override {
      StateSpaceRegressionModel *ans = clone();
      ans->copy_samplers(*this);
      return ans;
    }

    // The number of time points in the data.
    int time_dimension() const override { return dat().size(); }
    int xdim() const  {
      return observation_model()->xdim();
    }

    // Variance of observed data y[t], given state alpha[t].  Durbin
    // and Koopman's H.
    double observation_variance(int t) const override;

    double adjusted_observation(int t) const override;
    bool is_missing_observation(int t) const override;
    RegressionModel *observation_model() override { return regression_.get(); }
    const RegressionModel *observation_model() const override {
      return regression_.get();
    }

    void observe_data_given_state(int t) override;

    // Forecast the next nrow(newX) time steps given the current data,
    // using the Kalman filter.  The first column of Matrix is the mean
    // of the forecast.  The second column is the standard errors.
    Matrix forecast(const Matrix &newX);

    // Simulate the next nrow(newX) time periods, given current parameters and
    // state, from the posterior predictive distribution.
    Vector simulate_forecast(RNG &rng, const Matrix &newX,
                             const Vector &final_state);
    Vector simulate_forecast(RNG &rng, const Matrix &newX);

    // Simulate the next nrow(newX) time periods from the posterior predictive
    // distribution.  The rows of the returned matrix are the contributions of
    // each state model to the prediction.  The final row contains the
    // individual errors.  The sum across rows is equivalent to a call to
    // simulate_forecast.
    Matrix simulate_forecast_components(RNG &rng, const Matrix &newX,
                                        const Vector &final_state);

    // Simulate a forecast based on multiplexed data, where multiple
    // observations can have the same timestamp.
    //
    // Args:
    //   rng:  The random number generator.
    //   newX:  The matrix of predictors where forecasts are needed.
    //   final_state: Contains the simulated state values for the model as of
    //     the time of the final observation in the training data.
    //   timestamps: Each entry corresponds to a row in forecast_predictors, and
    //     gives the number of time periods after time_dimension() at which to
    //     make the prediction.  A zero-value in timestamps corresponds to one
    //     period after the end of the training data.
    //
    // Returns:
    //   A vector of forecasts simulated from the posterior predictive
    //   distribution.  Each entry corresponds to a row of newX.
    Vector simulate_multiplex_forecast(RNG &rng, const Matrix &newX,
                                       const Vector &final_state,
                                       const std::vector<int> &timestamps);

    // Contribution of the regression model to the overall mean of y at each
    // time point.  In the case of multiplexed data, the average regression
    // contribution for each time point is computed (averaging across
    // observations with potentially different predictors).
    Vector regression_contribution() const override;
    bool has_regression() const override { return true; }

    // Return the vector of one-step-ahead predictions errors from a
    // holdout sample, following immediately after the training data.
    //
    // Args:
    //   newX: The predictor variables for the holdout sample.
    //   newY: The response variable for the holdout data.
    //   final_state:  The state vector as of the end of the training data.
    //   standardize: Should the prediction errors be divided by the square root
    //     of the one step ahead forecast variance?
    //
    // Returns:
    //   The vector of one step ahead prediction errors for the holdout data.
    //   This is the same length as holdout_y.
    Vector one_step_holdout_prediction_errors(const Matrix &newX,
                                              const Vector &newY,
                                              const Vector &final_state,
                                              bool standardize = false) const;

    Matrix simulate_holdout_prediction_errors(
        int niter, int cutpoint_number, bool standardize) override;

    Ptr<RegressionModel> regression_model() { return regression_; }
    const Ptr<RegressionModel> regression_model() const { return regression_; }

    // Need to override add_data so that x's can be shared with the
    // regression model.
    void add_data(const Ptr<Data> &dp) override;

    // Promotes dp to a MultiplexedRegressionData containing a single
    // observation, then calls add_multiplexed_data.
    void add_regression_data(const Ptr<RegressionData> &dp);

    // Adds dp to the vector of data, as the most recent observation, and adds
    // the regression data in 'dp' to the underlying regression model.
    void add_multiplexed_data(
        const Ptr<StateSpace::MultiplexedRegressionData> &dp);

   private:
    // The regression model holds the regression coefficients and the
    // observation error variance.
    Ptr<RegressionModel> regression_;

    // Initialization work common to several constructors
    void setup();
  };

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_REGRESSION_HPP_
