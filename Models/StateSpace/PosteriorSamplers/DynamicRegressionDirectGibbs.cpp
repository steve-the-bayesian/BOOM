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
    Vector prior_variance = this->compute_prior_variance(inc, t);
    std::pair<SpdMatrix, Vector> suf = model_->data(t)->xtx_xty(inc);

    const SpdMatrix &xtx(suf.first);
    const Vector &xty(suf.second);

    // suf.first is xtx.  suf.second is xty.x
    SpdMatrix posterior_precision = xtx;
    posterior_precision.diag() += 1.0 / prior_variance;
    Vector posterior_mean = posterior_precision.solve(xty);

    double ans = log_inclusion_prior(inc, t, j);
    int mapped_index = inc.INDX(j);
    ans += -0.5 * log(prior_variance[mapped_index]);
    ans -= -0.5 * posterior_precision.logdet();

    double sigsq = model_->residual_variance();
    double SSE = model_->data(t)->yty() + xtx.Mdist(posterior_mean)
        - 2 * posterior_mean.dot(xty);
    double prior_sum_of_squares =
        square(posterior_mean[mapped_index]) / prior_variance[mapped_index];

    double sum_of_squares = SSE + prior_sum_of_squares;
    ans -=  0.5 * sum_of_squares / sigsq;

    return ans;
  }

  Vector DRDGS::compute_prior_variance(const Selector &inc, int t) const {
    // For the set of included variables, look left and right until you find the
    // first exclusion.

    // TODO: This algorithm is TERRIBLE!  Improve it!!  But for now don't let
    // the best be the enemy of the good.

    Vector ans(inc.nvars());
    for (int i = 0; i < inc.nvars(); ++i) {
      int I = inc.indx(i);
      // left and right are the number of spaces to the left or right of t that
      // must be moved to finde the first excluded value.
      double left_steps = -1;
      for (int left = 1; left <= t; ++left) {
        if (!model_->inclusion_indicator(t - left, I)) {
          left_steps = left;
          break;
        }
      }
      if (left_steps < 0) {
        left_steps = t;
      }

      double right_steps = -1;
      for (int right = 1; t + right < model_->time_dimension(); ++right) {
        if (!model_->inclusion_indicator(t + right, I)) {
          right_steps = right;
        }
      }
      if (right_steps < 0) {
        right_steps = infinity();
      }
      double interval_length = left_steps + right_steps;
      double condition_factor = 1.0 - left_steps / interval_length;
      ans[i] = model_->innovation_variance(I) * condition_factor;
    }
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
    for (int t = 0; t < model_->time_dimension(); ++t) {
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
