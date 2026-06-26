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
    
    virtual double value(int arm, const MixedMultivariateData &context) const;
    
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
    virtual Vector optimal_arm_probabilities(
        const MixedMultivariateData &context,
        RNG &rng = GlobalRng::rng) const;

    // Return one draw of Thompson sampling for the bandit.  This does not
    // update the posterior distribution.  It samples one set of model
    // parameters from the set of posterior draws, calls
    // 'optimal_arm_probabilities' assuming that draw is the true set of
    // parameters, and returns the values of the chosen arm.
    //
    // Args:
    //   context: THe context data describing an individual subject.
    //   rng:  The random number generator to use for sampling.
    // 
    // Returns:
    //   Vector of strings describing the levels of the action/experiment
    //   variables for the chosen arm.
    virtual std::vector<std::string> thompson(
        const MixedMultivariateData &context,
        RNG &rng = GlobalRng::rng) const;
    
    // Return the index of the MCMC draw selected in the most recent call to
    // thompson().
    int last_thompson_row() const {return last_thompson_row_;}
    
    // Return the index of the arm selected in the most recent call to thompson().
    int last_thompson_arm() const {return last_thompson_arm_;}
    
    virtual Vector value_remaining_distribution(
        const MixedMultivariateData &context,
        RNG &rng = GlobalRng::rng) const;

    const Ptr<LinearBanditEncoder> &encoder() const {
      return encoder_;
    }

    const Ptr<BinomialLogitModel> &model() const {
      return model_;
    }

    const Matrix &draws() const {
      return coefficient_draws_;
    }

    const Vector &log_likelihood() const {
      return log_likelihood_;
    }

    void set_draws(const Matrix &draws);

   protected:
    void set_thompson_row(int row) const {last_thompson_row_ = row;}
    void set_thompson_arm(int arm) const {last_thompson_arm_ = arm;}
    
   private:
    Ptr<BinomialLogitModel> model_;
    Ptr<LinearBanditEncoder> encoder_;

    Matrix coefficient_draws_;

    Vector log_likelihood_;

    // These are for temporary record keeping during Thompson sampling.
    mutable int last_thompson_row_;
    mutable int last_thompson_arm_;
  };
  
}  // namespace BOOM

#endif  // BOOM_BANDITS_LOGIT_BANDIT_HPP_
