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

#include "Models/Policies/ManyParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/MarkovModel.hpp"

namespace BOOM {
  namespace StateSpace {
    class RegressionDataTimePoint : public Data {
     public:
      // Default constructor sets xdim to -1.
      RegressionDataTimePoint(int xdim = -1): xdim_(xdim), suf_(nullptr) {}

      RegressionDataTimePoint(const RegressionDataTimePoint &rhs);
      RegressionDataTimePoint(RegressionDataTimePoint &&rhs) = default;

      // Mandatory overrides.
      RegressionDataTimePoint * clone() const override {
        return new RegressionDataTimePoint(*this);
      }

      std::ostream &display(std::ostream &out) const override;

      // Add the data point to the managed collection of data.  If the number of
      // data points exceeds the predictor dimension, switch from storing raw
      // data to storing sufficient statistics.
      void add_data(const Ptr<RegressionData> &dp);

      bool using_suf() const {return !!suf_;}

      const NeRegSuf &suf() const {
        if (!suf_) {
          report_error("Not enough data to use sufficient statistics.");
        }
        return *suf_;
      }

      int sample_size() const {
        if (!suf_) {
          return raw_data_.size();
        } else {
          return lround(suf_->n());
        }
      }

     private:
      // xdim_ is set to -1 by the default constructor, and updated when the
      // first data point is added.  Adding data that conflicts with xdim raises
      // an error.
      int xdim_;

      std::vector<Ptr<RegressionData>> raw_data_;

      // On construction suf_ is a nullptr.  When switch_to_suf is called
      // raw_data_ is cleared and suf_ is populated.
      Ptr<NeRegSuf> suf_;
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

    //
    void add_data(const Ptr<Data> &dp) override;

    // Add dp to the last time point.
    void add_data(const Ptr<RegressionData> &dp);

    // Add dp to a specific time point.  If that point does not yet exist,
    // create it and all intervening time points.
    void add_data(const Ptr<RegressionData> &dp, int time_index);

    // Add a new time point.
    void add_data(const Ptr<StateSpace::RegressionDataTimePoint> &dp);

    std::vector<Ptr<StateSpace::RegressionDataTimePoint>> &dat() {return data_;}
    const std::vector<Ptr<StateSpace::RegressionDataTimePoint>> &dat() const {
      return data_;
    }

    void clear_data() override;
    void combine_data(const Model &other_model, bool just_suf = true) override;

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

   private:
    // Number of predictor variables.
    int xdim_;

    // Storage for data at a time point.  Once the number of observations
    // exceeds xdim data are stored as sufficient statistics rather than raw
    // observations.

    std::vector<Ptr<StateSpace::RegressionDataTimePoint>> data_;
  };


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
    explicit DynamicRegressionModel(int xdim);
    DynamicRegressionModel(const DynamicRegressionModel &rhs);
    DynamicRegressionModel *clone() const override {
      return new DynamicRegressionModel(*this);
    }
    DynamicRegressionModel(DynamicRegressionModel &&rhs) = default;


    // The vector of inclusion indicators at time t.
    const Selector &inclusion_indicators(int time_index) const {
      return coefficients_[time_index]->inc();
    }
    void set_inclusion_indicators(const Selector &inc, int time_index) const {
      coefficients_[time_index]->set_inc(inc);
    }

    // Args:
    //   time_index:  The index of a time point.
    //   predictor_index:  The index of a predictor variable.
    //
    // Returns:
    //   Whether the specified variable is included at the specified time.
    bool inclusion_indicator(int time_index, int predictor_index) const {
      return coefficients_[time_index]->inc(predictor_index);
    }

    // The log of the transition probability for an inclusion indicator.  One of
    // the big simplifying assumptions for this model is that this is the same
    // for all indicators and across all times.
    double log_transition_probability(bool from, bool to) const {
      return inclusion_transition_model_->log_transition_probability(from, to);
    }

   private:
    std::vector<Ptr<GlmCoefs>> coefficients_;
    Ptr<UnivParams> residual_variance_;

    // The variance that describes the rate at which each ACTIVE coefficient
    // changes over time:
    //
    //    beta[j, t+1] = beta[j, t] + N(0, innovation_variances_[j])
    //
    // Note that these variances include information about the scale of the
    // predictor variables multiplying the coefficients.
    std::vector<Ptr<UnivParams>> innovation_variances_;

    // A 2-state Markov model describes the frequency of jumps in and out of the
    // model.  This is the temporal equivalent of the "prior inclusion
    // probabilities" in a static model.  In a later edition of this model we
    // might want to have more of these.
    Ptr<MarkovModel> inclusion_transition_model_;
  };

}  // namespace BOOM

#endif  // BOOM_DYNAMIC_COEFFICIENT_REGRESSION_MODEL_HPP_
