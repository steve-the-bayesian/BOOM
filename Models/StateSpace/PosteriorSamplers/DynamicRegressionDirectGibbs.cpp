/*
  Copyright (C) 2005-2020 Steven L. Scott

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

#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionDirectGibbs.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using DRDGS = DynamicRegressionDirectGibbsSampler;
  }

  void DRDGS::draw() {
    draw_inclusion_indicators();
    draw_coefficients_given_inclusion();
    draw_residual_variance();
    draw_state_innovation_variance();
    draw_transition_probabilities();
  }

  double DRDGS::logpri() const {
    report_error("Not implemented.");
    return -1;
  }

  void DRDGS::mcmc_one_flip(Selector &inc, int time_index, int predictor_index) {
    double logp_old = log_model_prob(inc, time_index, predictor_index);
    inc.flip(predictor_index);
    double logp_new = log_model_prob(inc, time_index, predictor_index);
    double u = runif_mt(rng(), 0, 1);
    if (log(u) > logp_new - logp_old) {
      inc.flip(predictor_index); // reject the draw
    }
    // Default value is to keep the flip.
  }

  // The unnormalized log posterior proportional to p(gamma[t] | *), where * is
  // everything but the model coefficients, which are assumed integrated out.
  //
  // The prior distribution on beta is simpler than the general g-prior, because
  // betas, conditional in inclusion, are independent across predictors.  The
  // only aspect of the prior that needs to be evaluated is element j.
  double DRDGS::log_model_prob(const Selector &inc, int t, int j) const {
    //    const StateSpace::RegressionDataTimePoint &data = *model_->dat()[t];

    double ans = log_inclusion_prior(inc, t, j);


    return ans;
  }

  double DRDGS::log_inclusion_prior(const Selector &inc, int t, int j) const {
    double ans = 0;
    bool inc_now = inc[j];
    if (t != 0) {
      bool inc_prev = model_->inclusion_indicator(t - 1, j);
      ans += model_->log_transition_probability(inc_prev, inc_now);
    }
    if (t+1 < model_->time_dimension()) {
      bool inc_next = model_->inclusion_indicator(t + 1, j);
      ans += model_->log_transition_probability(inc_now, inc_next);
    }
    return ans;
  }

  void DRDGS::draw_inclusion_indicators() {
    for (int t = 0; t < model_->time_dimension(); ++t) {
      Selector inc = model_->inclusion_indicators(t);
      for (int j = 0; j < model_->xdim(); ++j) {
        // Can we do forward-backward here?
        // Do Direct Gibbs first and see where the trouble spots are.
        mcmc_one_flip(inc, t, j);
      }
      model_->set_inclusion_indicators(inc, t);
    }
  }


  void DRDGS::draw_coefficients_given_inclusion() {
    for (int t = 0; i < model_->time_dimension(); ++t) {

    }
  }

  void DRDGS::draw_residual_variance() {
  }

  void DRDGS::draw_state_innovation_variance() {
  }

  void DRDGS::draw_transition_probabilities() {
    report_error("Not implemented.");
  }

}  // namespace BOOM
