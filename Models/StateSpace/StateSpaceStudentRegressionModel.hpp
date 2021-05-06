// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#ifndef BOOM_STATE_SPACE_STUDENT_REGRESSION_MODEL_HPP_
#define BOOM_STATE_SPACE_STUDENT_REGRESSION_MODEL_HPP_

#include "Models/Glm/TRegression.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/StateSpaceNormalMixture.hpp"

namespace BOOM {
  namespace StateSpace {

    // If y ~ T(mu, sigma^2, nu) then
    //          y = mu + epsilon / sqrt(w)
    //    epsilon ~ N(0, sigma^2)
    //          w ~ Gamma(nu / 2, nu / 2)
    // with epsilon independent of w.
    //
    // This class denotes 'w' as the 'weight' of y, which is a latent variable
    // that can be imputed from its full conditional distribution.  w / sigma^2
    // is the precision of y.
    //
    // In the case of multiplexed data, the "value" of this data point is the
    // precision weighted average of (y - x * beta) across observations.  The
    // precision weighted average has precision = (sum of weights) / sigma^2.
    class AugmentedStudentRegressionData : public MultiplexedData {
     public:
      AugmentedStudentRegressionData();
      AugmentedStudentRegressionData(double y, const Vector &x);
      explicit AugmentedStudentRegressionData(
          const std::vector<Ptr<RegressionData>> &data);

      AugmentedStudentRegressionData *clone() const override;
      std::ostream &display(std::ostream &out) const override;

      void add_data(const Ptr<RegressionData> &observation);

      double weight(int observation) const { return weights_[observation]; }
      void set_weight(double weight, int observation);

      double adjusted_observation(const GlmCoefs &coefficients) const;
      double sum_of_weights() const;

      double state_model_offset() const { return state_model_offset_; }
      void set_state_model_offset(double offset);

      const RegressionData &regression_data(int observation) const {
        return *(regression_data_[observation]);
      }
      Ptr<RegressionData> regression_data_ptr(int observation) {
        return regression_data_[observation];
      }
      int total_sample_size() const override { return regression_data_.size(); }

     private:
      std::vector<Ptr<RegressionData>> regression_data_;
      Vector weights_;
      double state_model_offset_;
    };
  }  // namespace StateSpace

  class StateSpaceStudentRegressionModel
      : public StateSpaceNormalMixture,
        public IID_DataPolicy<StateSpace::AugmentedStudentRegressionData>,
        public PriorPolicy {
   public:
    explicit StateSpaceStudentRegressionModel(int xdim);
    StateSpaceStudentRegressionModel(
        const Vector &response, const Matrix &predictors,
        const std::vector<bool> &observed = std::vector<bool>());
    StateSpaceStudentRegressionModel(
        const StateSpaceStudentRegressionModel &rhs);
    StateSpaceStudentRegressionModel *clone() const override;
    StateSpaceStudentRegressionModel *deepclone() const override {
      StateSpaceStudentRegressionModel *ans = clone();
      ans->copy_samplers(*this);
      return ans;
    }

    int time_dimension() const override;
    int xdim() const {return observation_model()->xdim();}

    // The total number of observations across all time points.
    int total_sample_size() const;

    int total_sample_size(int time) const override {
      return dat()[time]->total_sample_size();
    }

    const RegressionData &data(int t, int observation) const override {
      return dat()[t]->regression_data(observation);
    }

    // Returns the imputed observation variance from the latent data
    // for observation t.  This is sigsq() / w[t] from the comment
    // above.
    double observation_variance(int t) const override;

    // Returns the value for observation t minus x[t]*beta.  Returns
    // infinity if observation t is missing.
    double adjusted_observation(int t) const override;

    // Returns true if observation t is missing, false otherwise.
    bool is_missing_observation(int t) const override;

    TRegressionModel *observation_model() override {
      return observation_model_.get();
    }
    const TRegressionModel *observation_model() const override {
      return observation_model_.get();
    }

    // Set the offset in the data to the state contribution.
    void observe_data_given_state(int t) override;

    Vector simulate_forecast(RNG &rng, const Matrix &predictors,
                             const Vector &final_state);
    Matrix simulate_forecast_components(
        RNG &rng, const Matrix &predictors, const Vector &final_state);

    // Simulate a forecast based on multiplexed data, where multiple
    // observations can have the same timestamp.
    //
    // Args:
    //   rng:  The random number generator.
    //   predictors:  The matrix of predictors where forecasts are needed.
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
    Vector simulate_multiplex_forecast(RNG &rng,
                                       const Matrix &predictors,
                                       const Vector &final_state,
                                       const std::vector<int> &timestamps);


    // Return the vector of one-step-ahead predictions errors from a
    // holdout sample, following immediately after the training data.
    //
    // Args:
    //   rng: The random number generator used to simulate the latent "weight"
    //     variable in the student T distribution.
    //   response: The response variable for the holdout data.
    //   predictors: The predictor variables for the holdout sample.
    //   final_state:  The state vector as of the end of the training data.
    //   standardize: Should the prediction errors be divided by the square root
    //     of the one step ahead forecast variance?
    //
    // Returns:
    //   The vector of one step ahead prediction errors for the holdout data.
    //   This is the same length as holdout_y.
    Vector one_step_holdout_prediction_errors(RNG &rng,
                                              const Vector &response,
                                              const Matrix &predictors,
                                              const Vector &final_state,
                                              bool standardize = false);

    Matrix simulate_holdout_prediction_errors(
        int niter, int cutpoint_number, bool standardize) override;

   private:
    // Returns the marginal variance of the student error distribution.  If the
    // 'nu' degrees of freedom parameter <= 2 this is technically infinity, but
    // a "large value" will be returned instead.
    double student_marginal_variance() const;

    // Sets up observers on model parameters, so that the Kalman
    // filter knows when it needs to be recomputed.
    void set_observers();
    Ptr<TRegressionModel> observation_model_;
  };

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_STUDENT_REGRESSION_MODEL_HPP_
