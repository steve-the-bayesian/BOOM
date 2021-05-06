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

#ifndef BOOM_STATE_SPACE_LOGIT_MODEL_HPP_
#define BOOM_STATE_SPACE_LOGIT_MODEL_HPP_

#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/BinomialRegressionData.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/StateSpaceNormalMixture.hpp"

namespace BOOM {

  class BinomialLogitDataImputer;

  namespace StateSpace {

    // Gaussian mixture representation of BinomialRegressionData.
    //
    // Let y_t denote the number of successes out of n_t trials, where
    // n_t is taken as a known constant.  The observation equation is
    //
    //         y_t ~ Binomial(n_t, p_t), where
    //  logit(p_t) = Z_t^T \alpha_t + \beta * x_t
    //             = mu_t.
    //
    // Observation y_t is the sum of n_t Bernoulli random variables, each of
    // which is associated with a pair of variables z_it, v_it where z_it \sim
    // N(mu_t, v_it), and y_t = sum_i I(z_it > 0).
    //
    // The latent continuous value of point t is the precision weighted average
    // of z_it.
    //
    //                sum_i z_{it} / v_{it}
    //      zbar_t =  ------------------    .
    //                 sum_i  1 / v_{it}
    //
    // It is a complete data sufficient statistic for observation t.  It is
    // zbar_t that gets imputed, along with its variance
    //
    //       V_t = 1.0 / sum_i(1.0 / v_{it}).
    //
    // In the case of multiplexed data each binomial observation y_jt gets
    // imputed as above, with a corresponding zbar_jt and V_jt.
    class AugmentedBinomialRegressionData : public MultiplexedData {
     public:
      // Constructs an empty data point.  Observations can be added later using
      // add_data().
      AugmentedBinomialRegressionData();

      // A constructor for the usual case where there is only one data point per
      // time period.
      AugmentedBinomialRegressionData(double y, double n, const Vector &x);

      // A constructor for the multiplexed case, where there are multiple
      // observations at each time point.
      explicit AugmentedBinomialRegressionData(
          const std::vector<Ptr<BinomialRegressionData>> &binomial_data);

      AugmentedBinomialRegressionData *clone() const override;
      std::ostream &display(std::ostream &out) const override;

      void add_data(const Ptr<BinomialRegressionData> &binomial_data);

      // Set the latent data and precision for the specific observation owned by
      // this data object.
      //
      // Args:
      //   value:  The value of zbar_t as described in the class comments.
      //   precision: The precision 1.0 / V_t, as described in the class
      //     comments.
      //   observation: The observation number.  This will be zero except in the
      //     case of multiplexed data.
      void set_latent_data(double value, double precision, int observation);

      double latent_data_variance(int observation) const;
      double latent_data_value(int observation) const;
      double adjusted_observation(const GlmCoefs &coefficients) const;
      double latent_data_overall_variance() const;

      void set_state_model_offset(double offset);
      double state_model_offset() const { return state_model_offset_; }

      const BinomialRegressionData &binomial_data(int observation) const {
        return *(binomial_data_[observation]);
      }

      Ptr<BinomialRegressionData> binomial_data_ptr(int observation) {
        return binomial_data_[observation];
      }

      double total_trials() const;
      double total_successes() const;
      int total_sample_size() const override { return binomial_data_.size(); }

     private:
      std::vector<Ptr<BinomialRegressionData>> binomial_data_;

      // The precision weighted mean of the underlying Gaussian observations
      // associated with each binomial observation.
      Vector latent_continuous_values_;

      // The sum of the precisions of the underlying latent Gaussians associated
      // with each binomial observation.
      Vector precisions_;

      // The state contribution (minus the static regression effect) to the mean
      // of latent_continuous_values_.
      double state_model_offset_;
    };

  }  // namespace StateSpace

  class StateSpaceLogitModel
      : public StateSpaceNormalMixture,
        public IID_DataPolicy<StateSpace::AugmentedBinomialRegressionData>,
        public PriorPolicy {
   public:
    explicit StateSpaceLogitModel(int xdim);
    StateSpaceLogitModel(
        const Vector &successes, const Vector &trials,
        const Matrix &design_matrix,
        const std::vector<bool> &observed = std::vector<bool>());

    StateSpaceLogitModel(const StateSpaceLogitModel &rhs);
    StateSpaceLogitModel *clone() const override;
    StateSpaceLogitModel *deepclone() const override {
      StateSpaceLogitModel *ans = clone();
      ans->copy_samplers(*this);
      return ans;
    }
    int total_sample_size(int time) const override {
      return dat()[time]->total_sample_size();
    }

    int xdim() const { return observation_model()->xdim(); }

    const BinomialRegressionData &data(int t, int observation) const override {
      return dat()[t]->binomial_data(observation);
    }

    int time_dimension() const override;

    // Returns the imputed observation variance from the latent
    // data for observation t.  This is V_t from the class comment
    // above.
    double observation_variance(int t) const override;

    // Returns the imputed value for observation t (zbar_t in the
    // the class comment, above), minus x[t]*beta.  Returns
    // infinity if observation t is missing.
    double adjusted_observation(int t) const override;

    // Returns true if observation t is missing, false otherwise.
    bool is_missing_observation(int t) const override;

    BinomialLogitModel *observation_model() override {
      return observation_model_.get();
    }
    const BinomialLogitModel *observation_model() const override {
      return observation_model_.get();
    }

    // Set the offset in the data to the state contribution.
    void observe_data_given_state(int t) override;

    // Returns a vector of draws from the posterior predictive
    // distribution of the next nrow(forecast_predictors) time
    // periods.  The draws are on the same (binomial) scale as the
    // original data (as opposed to the logit scale).
    //
    // Args:
    //   rng:  Random number generator to use for the simulation.
    //   forecast_predictors: A matrix of predictors to use for the
    //     forecast period.  If no regression component is desired,
    //     then a single column matrix of 1's (an intercept) should be
    //     supplied so that the length of the forecast period can be
    //     determined.
    //   trials: A vector of non-negative integers giving the number
    //     of trials that will take place at each point in the
    //     forecast period.
    //   final_state: A draw of the value of the state vector at the
    //     final time period in the training data.
    Vector simulate_forecast(RNG &rng,
                             const Matrix &forecast_predictors,
                             const Vector &trials,
                             const Vector &final_state);

    // Return a draw from the posterior predictive distribution of the
    // contribution of each state model to the predictive distribuiton, on the
    // logit scale.
    //
    // Args:
    //   rng:  Random number generator to use for the simulation.
    //   forecast_predictors: A matrix of predictors to use for the
    //     forecast period.  If no regression component is desired,
    //     then a single column matrix of 1's (an intercept) should be
    //     supplied so that the length of the forecast period can be
    //     determined.
    //   trials: A vector of non-negative integers giving the number
    //     of trials that will take place at each point in the
    //     forecast period.
    //   final_state: A draw of the value of the state vector at the
    //     final time period in the training data.
    //
    // Returns:
    //   A matrix, with rows corresponding to state components, and columns to
    //   time points, containing the contribution of each state model to the
    //   forecast distribution, on the logit scale.  The last row of the matrix
    //   contains a draw from the posterior predictive distribution, equivalent
    //   to simulate_forecast().
    Matrix simulate_forecast_components(
        RNG &rng,
        const Matrix &forecast_predictors,
        const Vector &trials,
        const Vector &final_state);

    // Returns a vector of draws from the posterior predictive distribution for
    // a multiplexed prediction problem.  That is, a prediction problem where
    // some time periods to be predicted have more than one observation with
    // different covariates.
    //
    // Args:
    //   forecast_predictors: A matrix of predictors to use for the
    //     forecast period.  If no regression component is desired,
    //     then a single column matrix of 1's (an intercept) should be
    //     supplied so that the length of the forecast period can be
    //     determined.
    //   trials: A vector of non-negative integers giving the number
    //     of trials that will take place at each point in the
    //     forecast period.
    //   final_state: A draw of the value of the state vector at the
    //     final time period in the training data.
    //   timestamps: Each entry corresponds to a row in forecast_predictors, and
    //     gives the number of time periods after time_dimension() at which to
    //     make the prediction.  A zero-value in timestamps corresponds to one
    //     period after the end of the training data.
    //
    // Returns:
    //   A vector of draws with length equal to nrow(forecast_predictors), from
    //   the posterior distribution of the conditional state at time t.
    Vector simulate_multiplex_forecast(RNG &rng,
                                       const Matrix &forecast_predictors,
                                       const Vector &trials,
                                       const Vector &final_state,
                                       const std::vector<int> &timestamps);

    // Args:
    //   rng:  A U(0,1) random number generator.
    //   data_imputer: A data imputer that can be used to unmix the
    //     binomial observations into a latent logistic, and then to a
    //     mixture of normals.
    //   successes: The vector of success counts during the holdout
    //     period.
    //   trials: The vector of trial counts during the holdout period.
    //   predictors: The matrix of predictors for the holdout period.
    //     If the model contains no regression component then a single
    //     column matrix of 1's should be supplied.
    //   final_state: A draw of the value of the state vector at the
    //     final time period in the training data.
    //
    // Returns:
    //   A draw from the posterior distribution of the one-state
    //   holdout errors.  The draw is on the scale of the original
    //   data, so it will consist of integers, but it is an error, so
    //   it may be positive or negative.
    //
    //  TODO: consider whether this would make more sense
    //  on the logit scale.
    Vector one_step_holdout_prediction_errors(
        RNG &rng,
        BinomialLogitDataImputer &data_imputer,
        const Vector &successes,
        const Vector &trials,
        const Matrix &predictors,
        const Vector &final_state);

    Matrix simulate_holdout_prediction_errors(
        int niter, int cutpoint_number, bool standardize) override;

   private:
    Ptr<BinomialLogitModel> observation_model_;
  };

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_LOGIT_MODEL_HPP_
