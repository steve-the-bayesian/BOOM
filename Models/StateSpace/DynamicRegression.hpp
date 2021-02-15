#ifndef BOOM_DYNAMIC_COEFFICIENT_REGRESSION_MODEL_HPP_
#define BOOM_DYNAMIC_COEFFICIENT_REGRESSION_MODEL_HPP_
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

#include "LinAlg/Vector.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "Models/Policies/ManyParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/SpdParams.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"

namespace BOOM {
  class DynamicRegressionModel;

  namespace StateSpace {

    // A regression data set at a point in time.  If fewer than 'xdim' data
    // points are observed then the data set is stored as raw data.  When xdim
    // or more data points are present then the raw data are discarded, and
    // sufficient statistics are used instead.
    class RegressionDataTimePoint : public Data {
     public:
      // Args:
      //   xdim: The dimension of the predictor variable.  The default value of
      //   -1 is a signal that the dimension is unknown.  It will be set on the
      //   first call to add_data().
      RegressionDataTimePoint(int xdim = -1):
          xdim_(xdim), yty_(0.0), suf_(nullptr) {}

      // Args:
      //   X:  Matrix of predictors.
      //   y:  Vector of repsonses.
      RegressionDataTimePoint(const Matrix &X, const Vector &y);

      RegressionDataTimePoint(const RegressionDataTimePoint &rhs);
      RegressionDataTimePoint(RegressionDataTimePoint &&rhs) = default;

      // -------- Mandatory overrides.
      RegressionDataTimePoint * clone() const override {
        return new RegressionDataTimePoint(*this);
      }

      std::ostream &display(std::ostream &out) const override;

      // Dimension of the predictor variable.
      int xdim() const {return xdim_;}

      //-------- Accumulate data about this time point.

      // Add the data point to the managed collection of data.  If the number of
      // data points exceeds the predictor dimension, calling add_data will
      // switch from storing raw data to storing sufficient statistics.
      void add_data(const Ptr<RegressionData> &dp);

      //-------- Sufficient statisitcs for this time point ------

      // The number of data points observed at time t.
      int sample_size() const;

      // The conditional sufficient statistics xtx and xty given the inclusion
      // vector.
      std::pair<SpdMatrix, Vector> xtx_xty(const Selector &inc) const;

      // The sum of squares of the y variable.
      double yty() const;

      // The sum of squared errors relative to a given set of regression
      // coefficients.
      double SSE(const GlmCoefs &coefs) const;

     private:
      // xdim_ is set to -1 by the default constructor, and updated when the
      // first data point is added.  Adding data that conflicts with xdim raises
      // an error.
      int xdim_;
      double yty_;

      std::vector<Ptr<RegressionData>> raw_data_;

      // On construction suf_ is a nullptr.  When switch_to_suf is called
      // raw_data_ is cleared and suf_ is populated.
      Ptr<NeRegSuf> suf_;
    };

    //=========================================================================
    // A SelectorMatrix is formed by taking the columns of the identity
    // matrix chosen by a Selector.
    //
    // If G1 and G2 are selector matrices, the ProductSelectorMatrix is G2' *
    // G1.
    class ProductSelectorMatrix {
     public:
      ProductSelectorMatrix(const Selector &inc1, const Selector &inc2)
          : inc1_(inc1), inc2_(inc2) {}

      int nrow() const {return inc2_.nvars();}
      int ncol() const {return inc1_.nvars();}

      // A dense matrix representation of the ProductSelectorMatrix.
      Matrix dense() const;

      Vector operator*(const Vector &v) const;
      Vector operator*(const ConstVectorView &v) const;

      // Returns this * P * this'
      SpdMatrix sandwich(const SpdMatrix &P) const;

      DiagonalMatrix sandwich(const DiagonalMatrix &D) const;

      ProductSelectorMatrix transpose() const {
        return ProductSelectorMatrix(inc2_, inc1_);
      }

     private:
      // inc1_ and inc2_ are the selectors for the first and second time
      // periods.
      Selector inc1_;
      Selector inc2_;
    };

    //==========================================================================
    // A node for a contemporaneous Kalman filter describing the
    // DynamicRegressionModel defined below.  A node represents the conditional
    // distribution of the included coefficients at time t, given the inclusion
    // indicators, model parameters, and all data up to and including t.
    //
    // The distribution is defined up to the residual variance parameter sigsq.
    // Each node gives a mean and an unscaled variance.  The actual variance is
    // the unscaled variance times sigsq.
    class DynamicRegressionKalmanFilterNode {
     public:
      using Node = DynamicRegressionKalmanFilterNode;

      DynamicRegressionKalmanFilterNode()
          : state_mean_(1, 0.0),
            state_variance_(new SpdParams(SpdMatrix(1, 1.0))) {}

      // To be called by the Node managing the first data point in a data set.
      // Set the distribution conditional on the data and a pre-specified prior.
      //
      // Args:
      //   inc:  Inclusion indicators for the initial time point.
      //   initial_mean: Prior state mean for initial time point, given
      //     that all variables are included.
      //   unscaled_initial_precision: Prior state precision for initial time
      //     point, given that all variables are included.  The precision is
      //     "unscaled" because it must be divided by the residual variance to
      //     get the actual precision.
      //   data:  Initial data point.
      //   sigsq: The residual variance parameter.
      //
      // Returns:
      //   The log likelihood of the initial data point.
      double initialize(const Selector &inc,
                        const Vector &initial_mean,
                        const SpdMatrix &unscaled_initial_precision,
                        const RegressionDataTimePoint &data,
                        double sigsq);

      // Compute the distribution of today's state given today's data and the
      // previous node.
      //
      // Args:
      //   previous: The node for the previous period, giving p(beta[t-1] |
      //     Y[t-1]).
      //   data:  Data for the current time period t.
      //   model: The DynamicRegressionModel that will supply all the state space
      //     matrices and other filter inputs not specified as arguments.
      //   time_index:  The current time t.
      //
      // Returns:
      //   The conditional log likelihood of data given previous.
      //
      // Effects:
      //   state_mean_ and state_variance_ are computed.  Their dimension may
      //   change if the inclusion indicators for time t have changed since the
      //   last update.
      double update(const Node &previous, const RegressionDataTimePoint &data,
                    const DynamicRegressionModel &model, int t);

      // Simulate the coefficients for the given time index, conditional on
      // inclusion indicators, model parameters and the simulated value at time
      // t+1.
      Vector simulate_coefficients(
          const DynamicRegressionModel &model, int time_index, RNG &rng);

      // The conditional mean of the included regression coefficients at time t,
      // given inclusion indicators, model parameters, and data up to and
      // including time t.
      const Vector &state_mean() const {return state_mean_;}

      // The unscaled conditional variance and precision of the included
      // regression coefficients at time t, given inclusion indicators, model
      // parameters, and data up to and including time t.  The actual
      // conditional variance is the unscaled variance times the scalar residual
      // variance parameter sigsq.
      const SpdMatrix &unscaled_state_variance() const {
        return state_variance_->var();
      }
      const SpdMatrix &unscaled_state_precision() const {
        return state_variance_->ivar();
      }

     private:
      Vector state_mean_;

      // Storing the variance as SpdData means that we can easily switch back
      // and forth between precision and variance.
      Ptr<SpdParams> state_variance_;
    };

    //==========================================================================
    // A Kalman filter for dynamic regression models.
    class DynamicRegressionKalmanFilter {
     public:

      // Args:
      //   model:  The model whose coefficients are to be imputed.
      //   rng:  Random number generator to use for the imputation.
      //
      // Returns:
      //   The log likelihood of the observed data in the model.  This is a
      //   byproduct of Kalman filtering.
      //
      // Effects:
      //   Filter nodes are updated to the conditional mean and variance given
      //   data up to the time point they represent.  Model coefficients are set
      //   to imputed values.  The imputation is done conditional on inclusion
      //   vectors, observed data, and model parameters.
      double impute_state(DynamicRegressionModel &model, RNG &rng);

      // Args:
      //   model: The model containing the filter parameters and the data to be
      //     filtered.
      //
      // Returns:
      //   The log likelihood of the filtered data.
      //
      // Effects:
      //   The filter nodes are updated with the conditional mean and variance
      //   of the regression coefficients, given other model parameters,
      //   inclusion indicators, and observed data up to the time point they
      //   represent.
      double filter(const DynamicRegressionModel &model);

      // Run the "backward simulator" to simulate the set of included regression
      // coefficients for the model.  The draw is made conditional on inclusion
      // indicators.  This function assumes that the filtering step has just
      // completed.
      void simulate_coefficients(DynamicRegressionModel &model, RNG &rng);

     private:
      void ensure_storage(int number_of_time_points);

      Vector initial_mean_;
      SpdMatrix initial_variance_;
      std::vector<DynamicRegressionKalmanFilterNode> nodes_;
    };

  }  // namespace StateSpace

  //===========================================================================
  // Data policy for managing time series regression data.
  class TimeSeriesRegressionDataPolicy
      : public DefaultDataInfoPolicy<StateSpace::RegressionDataTimePoint> {
   public:

    TimeSeriesRegressionDataPolicy(int xdim);

    int xdim() const {return xdim_;}

    // Mandatory overrides.
    void add_data(const Ptr<Data> &dp) override;
    void clear_data() override;
    void combine_data(const Model &other_model, bool just_suf = true) override;

    //------------------------------------------------------------
    // Different ways of adding data.
    //
    // Add dp to the last time point.
    void add_data(const Ptr<RegressionData> &dp);

    // Add dp to a specific time point.  If that point does not yet exist,
    // create it and all intervening time points.
    void add_data(const Ptr<RegressionData> &dp, int time_index);

    // Add a new time point.
    void add_data(const Ptr<StateSpace::RegressionDataTimePoint> &dp);

    //------------------------------------------------------------
    // Accessing data and sufficient statistics.
    std::vector<Ptr<StateSpace::RegressionDataTimePoint>> &dat() override {
      return data_;
    }

    const std::vector<Ptr<StateSpace::RegressionDataTimePoint>>
    &dat() const override {
      return data_;
    }

    const StateSpace::RegressionDataTimePoint *data(int time) const {
      return data_[time].get();
    }

    // The number of observed or implied time points.
    int time_dimension() const {
      return data_.size();
    }

    // The total number of observations.
    int sample_size() const {
      int ans = 0;
      for (int i = 0; i < data_.size(); ++i) {
        ans += data_[i]->sample_size();
      }
      return ans;
    }

   protected:
    // To be overloaded by the main model class.
    //
    // Ensure that any data structures that depend on time are populated with at
    // least time_dimension() elements.
    virtual void ensure_time_dimension() = 0;

   private:
    // Number of predictor variables.
    int xdim_;

    // Storage for data at a time point.  Once the number of observations
    // exceeds xdim data are stored as sufficient statistics rather than raw
    // observations.

    std::vector<Ptr<StateSpace::RegressionDataTimePoint>> data_;
  };

  //==========================================================================
  // A DynamicRegressionModel is a time series regression model where the
  // coefficients obey a classic state space model.  Note that the number of
  // observations at each time point might differ.  The model is implemented as
  // a multivariate state space model.  Through data augmentation one can extend
  // this model to most GLM's.
  //
  // Define the set of responses at time t as Y'_t = [y_1t, y_2t, ... y_n_tt],
  // where Y_t = X[t] * beta[t] + error[t], with temporally IID error term
  // error[t] ~ N(0, Diagonal(sigma^2)).
  //
  // Each coefficient beta[j, t] is zero with probability determined by a Markov
  // chain Pr(beta[j, t] = s | beta[j, t-1] = r) = q[r, s], for r, s \in {0, 1}.
  //
  // The conditional distribution of beta[j, t] given beta[j, t-1], and given
  // that both are nonzero, is normal with mean b_jt = T_ij b_jt-1, and variance
  // tau^2.
  class DynamicRegressionModel
      : public ManyParamPolicy,
        public TimeSeriesRegressionDataPolicy,
        public PriorPolicy {
   public:
    friend class DynamicRegressionDirectGibbsSampler;

    explicit DynamicRegressionModel(int xdim);
    DynamicRegressionModel(const DynamicRegressionModel &rhs);
    DynamicRegressionModel *clone() const override {
      return new DynamicRegressionModel(*this);
    }
    DynamicRegressionModel(DynamicRegressionModel &&rhs) = default;

    // The prior distribution for the initial set of regression coefficients.
    void set_initial_state_mean(const Vector &mean);
    const Vector &initial_state_mean() const {
      return initial_state_mean_;
    }

    // The initial state variance is unscaled.
    void set_unscaled_initial_state_variance(const SpdMatrix &variance);
    const SpdMatrix &unscaled_initial_state_precision() const {
      return unscaled_initial_state_variance_->ivar();
    }

    // ----------------------------------------------------------------------
    // Static (non-time-varying) parameters.

    // The log of the transition probability for an inclusion indicator.  One of
    // the big simplifying assumptions for this model is that this is the same
    // for all indicators and across all times.
    double log_transition_probability(
        bool from, bool to, int predictor_index) const {
      return inclusion_transition_models_[predictor_index]->
          log_transition_probability(from, to);
    }

    // The "sigma squared" parameter describing the variance of the residuals.
    double residual_variance() const {return residual_variance_->value();}
    double residual_sd() const {return std::sqrt(residual_variance());}
    void set_residual_variance(double sigsq) {residual_variance_->set(sigsq);}

    // The "unscaled" variance describing a one-time-period change in the
    // specified coefficient.  The actual variance is the unscaled variance
    // times the residual variance.
    double unscaled_innovation_variance(int predictor_index) const {
      return innovation_error_models_[predictor_index]->sigsq();
    }

    // The vector of unscaled innovation variance parameters across all
    // coefficients.
    const Vector &unscaled_innovation_variances() const;

    // ----------------------------------------------------------------------
    // Regression coefficients and inclusion indicators.
    // Args:
    //   time_index:  The index of a time point.
    //   predictor_index:  The index of a predictor variable.

    // Returns:
    //   Whether the specified predictor is included at the specified time.
    bool inclusion_indicator(int time_index, int predictor_index) const {
      return coefficients_[time_index]->inc(predictor_index);
    }

    // The vector of inclusion indicators at time t.
    const Selector &inclusion_indicators(int time_index) const {
      return coefficients_[time_index]->inc();
    }

    void set_inclusion_indicators(int time_index, const Selector &inc) {
      coefficients_[time_index]->set_inc(inc);
    }

    // The regression coefficient of the requested variable at the requested
    // time.  This will be zero if the coefficient is excluded.
    double coefficient(int time_index, int predictor_index) const {
      return coefficients_[time_index]->Beta(predictor_index);
    }

    Vector included_coefficients(int time_index) const {
      return coefficients_[time_index]->included_coefficients();
    }

    void set_included_coefficients(int time_index, const Vector &beta) {
      coefficients_[time_index]->set_included_coefficients(beta);
    }

    GlmCoefs &coef(int time_index) {
      return *coefficients_[time_index];
    }

    const GlmCoefs &coef(int time_index) const {
      return *coefficients_[time_index];
    }

    double draw_coefficients_given_inclusion(RNG &rng) {
      return filter_.impute_state(*this, rng);
    }

    Ptr<MarkovModel> transition_model(int predictor_index) {
      return inclusion_transition_models_[predictor_index];
    }

    Ptr<ZeroMeanGaussianModel> innovation_error_model(int predictor_index) {
      return innovation_error_models_[predictor_index];
    }

   protected:
    void ensure_time_dimension() override;

   private:
    std::vector<Ptr<GlmCoefs>> coefficients_;
    Ptr<UnivParams> residual_variance_;

    // Prior distribution of the initial state vector, conditional on all
    // variables being included.
    Vector initial_state_mean_;
    // The variance is unscaled.  sigsq * unscaled is the actual variance.
    Ptr<SpdParams> unscaled_initial_state_variance_;

    // The variance that describes the rate at which each ACTIVE coefficient
    // changes over time:
    //
    //    beta[j, t+1] = beta[j, t] + N(0, sigsq * innovation_variance(j))
    //
    // Note that these variances include information about the scale of the
    // predictor variables multiplying the coefficients, but the residual
    // variance is factored out.
    std::vector<Ptr<ZeroMeanGaussianModel>> innovation_error_models_;

    // The next few items are a bit of structure that allows the variances from
    // innovation_error_models_ to be returned as a Vector.  That is an
    // operation which occurs in an inner loop, so it needs to be fast.  We set
    // up an observer pattern where the vector of variances can watch the model
    // parameters.
    mutable bool innovation_variances_current_;
    mutable Vector innovation_variances_;
    void observe_innovation_variances() {
      innovation_variances_current_ = false;
    }
    void refresh_innovation_variances() const;

    // A 2-state Markov model describes the frequency of jumps in and out of the
    // model.  This is the temporal equivalent of the "prior inclusion
    // probabilities" in a static model.  There is one Markov model per
    // coefficient.
    std::vector<Ptr<MarkovModel>> inclusion_transition_models_;

    StateSpace::DynamicRegressionKalmanFilter filter_;
  };

}  // namespace BOOM

#endif  // BOOM_DYNAMIC_COEFFICIENT_REGRESSION_MODEL_HPP_
