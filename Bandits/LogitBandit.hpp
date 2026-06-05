#ifndef BOOM_BANDITS_LOGIT_BANDIT_HPP_
#define BOOM_BANDITS_LOGIT_BANDIT_HPP_
/*
  Copyright (C) 2005-2026 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Bandits/GenericBanditBase.hpp"
#include "Models/Glm/BinomialLogitModel.hpp"
#include "stats/Encoders.hpp"

#include "Bandits/LinearBanditEncoder.hpp"

namespace BOOM {

  // A logit bandit is a particular form of multivariate contextual bandit for
  // success/failure outcomes.  The bandit models the success probability for
  // each trial using a logistic regression model.  The model formula for the
  // logistic regression is determined by a pair of DatasetEncoders.
  class LogitBandit : public GenericBanditBase {
   public:
    LogitBandit(const Ptr<BinomialLogitModel> &model,
                const Ptr<LinearBanditEncoder> &encoder); 

    // The number of arms the bandit is tracking.  This is determined by the
    // ArmMap component of the encoder.
    int number_of_arms() const override {return encoder_->number_of_arms();}

    // The number of draws representing the posterior distribution.  This is
    // determined by the most recent call to update_posterior().
    int ndraws() const {return coefficient_draws_.nrow();}
    
    // Record the outcomes of a particular arm being used with a particular set
    // of context.
    //
    // Args:
    //   arm:  Which 
    void observe_data(int arm,
                      int num_successes,
                      int num_trials,
                      const MixedMultivariateData &context);
    
    double value(int arm, const MixedMultivariateData &context) const;
    
    void update_posterior(int ndraws);

    // Args:
    //   context:  The context data describing an individual subject.
    //
    // Returns:
    //   A matrix, where row i contains the predictor vector for arm i under the
    //   supplied context.
    Matrix arm_predictors(const MixedMultivariateData &context) const;
    
    // This can probably be optimized if it uses too much memory.  It also
    // encodes the same things over and over again, so serilization might help.
    Vector optimal_arm_probabilities(const MixedMultivariateData &context,
                                     RNG &rng = GlobalRng::rng) const;
    
    Vector value_remaining_distribution(const MixedMultivariateData &context,
                                        RNG &rng = GlobalRng::rng) const;
    
   private:
    Ptr<BinomialLogitModel> model_;
    Ptr<LinearBanditEncoder> encoder_;

    Matrix coefficient_draws_;
  };
  
}  // namespace BOOM

#endif  // BOOM_BANDITS_LOGIT_BANDIT_HPP_
