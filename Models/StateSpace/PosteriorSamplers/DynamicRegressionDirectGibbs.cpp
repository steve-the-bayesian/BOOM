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
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

namespace BOOM {

  namespace {
    using DRDGS = DynamicRegressionDirectGibbsSampler;
  }  // namespace

  DRDGS::DynamicRegressionDirectGibbsSampler(
      DynamicRegressionModel *model,
      double residual_sd_prior_guess,
      double residual_sd_prior_sample_size,
      const Vector &innovation_sd_prior_guess,
      const Vector &innovation_sd_prior_sample_size,
      const Vector &prior_inclusion_probabilities,
      const Vector &expected_time_between_transitions,
      const Vector &transition_probability_prior_sample_size,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        residual_precision_prior_(new ChisqModel(
            residual_sd_prior_sample_size,
            residual_sd_prior_guess)),
        residual_variance_sampler_(residual_precision_prior_)
  {
    for (int j = 0; j < model_->xdim(); ++j) {
      // Set the posterior sampler for the innovation variance associated with
      // variable j.
      NEW(ChisqModel, innovation_precision_prior)(
          innovation_sd_prior_sample_size[j],
          innovation_sd_prior_guess[j]);
      NEW(ZeroMeanGaussianConjSampler, innovation_variance_sampler)(
          model_->innovation_error_model(j).get(),
          innovation_precision_prior,
          rng());
      model_->innovation_error_model(j)->set_method(
          innovation_variance_sampler);

      // Set the initial distribution of the inclusion indicators for variable
      // j.  It is static and will not be updated.
      Vector initial_inclusion_probabilities{
        1 - prior_inclusion_probabilities[j],
            prior_inclusion_probabilities[j]};
      model_->transition_model(j)->set_pi0(initial_inclusion_probabilities);
      // The prior distribution for the transition probability matrix is implied
      // by the steady state prior inclusion probability, the expected length of
      // an inclusion period (given that one has occurred), and the number of
      // observations worth of weight given to the prior estimates.
      Matrix prior_counts = infer_Markov_prior(
          prior_inclusion_probabilities[j],
          expected_time_between_transitions[j],
          transition_probability_prior_sample_size[j]);
      NEW(MarkovConjSampler, transition_sampler)(
          model_->transition_model(j).get(), prior_counts, rng());
      model_->transition_model(j)->set_method(transition_sampler);
    }
  }

  void DRDGS::draw() {
    draw_inclusion_indicators();
    model_->draw_coefficients_given_inclusion(rng());
    draw_residual_variance();
    draw_unscaled_state_innovation_variance();
    draw_transition_probabilities();
  }

  double DRDGS::logpri() const {
    double ans = residual_variance_sampler_.log_prior(model_->residual_variance());
    for (int j = 0; j < model_->xdim(); ++j) {
      ans += model_->innovation_error_model(j)->logpri();
      ans += model_->transition_model(j)->logpri();
    }
    return ans;
  }

  void DRDGS::mcmc_one_flip(Selector &inc, int time_index, int predictor_index) {
    double logp_old = log_model_prob(inc, time_index, predictor_index);
    inc.flip(predictor_index);
    double logp_new = log_model_prob(inc, time_index, predictor_index);
    double u = runif_mt(rng(), 0, 1);
    if (log(u) > logp_new - logp_old) {
      inc.flip(predictor_index); // reject the draw
    }
    // Otherwise keep the draw, in which case no further action is needed.
  }

  // The unnormalized log posterior proportional to p(gamma[t] | *), where * is
  // everything but the model coefficients, which are assumed integrated out.
  //
  // The prior distribution on beta is simpler than the general g-prior, because
  // betas, conditional in inclusion, are independent across predictors.  The
  // only aspect of the prior that needs to be evaluated is element j.
  double DRDGS::log_model_prob(const Selector &inc,
                               int time_index,
                               int predictor_index) const {
    double ans = log_inclusion_prior(inc, time_index, predictor_index);
    if (inc.nvars() == 0) {
      // double SSE = model_->data(time_index)->yty();

      // TODO: Take the -inf line out.  it is only present for debugging.
      return negative_infinity();
      // return ans - 0.5 * SSE / model_->residual_variance();
    }
    Vector unscaled_prior_variance =
        compute_unscaled_prior_variance(inc, time_index);
    std::pair<SpdMatrix, Vector> suf = model_->data(time_index)->xtx_xty(inc);

    const SpdMatrix &xtx(suf.first);
    const Vector &xty(suf.second);

    // suf.first is xtx.  suf.second is xty.x
    SpdMatrix posterior_precision = xtx;
    posterior_precision.diag() += 1.0 / unscaled_prior_variance;
    Vector posterior_mean = posterior_precision.solve(xty);

    int mapped_index = inc.INDX(predictor_index);
    ans += 0.5 * log(unscaled_prior_variance[mapped_index]);
    ans -= 0.5 * posterior_precision.logdet();

    double SSE = model_->data(time_index)->yty() + xtx.Mdist(posterior_mean)
        - 2 * posterior_mean.dot(xty);
    double prior_sum_of_squares = square(posterior_mean[mapped_index])
        / unscaled_prior_variance[mapped_index];

    double sum_of_squares = SSE + prior_sum_of_squares;
    ans -=  0.5 * sum_of_squares / model_->residual_variance();

    return ans;
  }

  Vector DRDGS::compute_unscaled_prior_variance(
      const Selector &inc, int time_index) const {
    // For the set of included variables, look left and right until you find the
    // first exclusion.

    // TODO: This algorithm is TERRIBLE!  Improve it!!  But for now don't let
    // the best be the enemy of the good.

    Vector ans(inc.nvars());
    for (int i = 0; i < inc.nvars(); ++i) {
      int I = inc.indx(i);
      // 'left' and 'right' are the number of spaces to the left or right of t
      // that must be moved to find the first excluded value.
      double left_steps = -1;
      for (int left = 1; left <= time_index; ++left) {
        if (!model_->inclusion_indicator(time_index - left, I)) {
          left_steps = left;
          break;
        }
      }
      if (left_steps < 0) {
        left_steps = time_index;
      }

      double right_steps = -1;
      for (int right = 1; time_index + right < model_->time_dimension();
           ++right) {
        if (!model_->inclusion_indicator(time_index + right, I)) {
          right_steps = right;
        }
      }
      if (right_steps < 0) {
        right_steps = infinity();
      }
      double interval_length = left_steps + right_steps;
      double condition_factor = 1.0 - left_steps / interval_length;
      ans[i] = model_->unscaled_innovation_variance(I) * condition_factor;
    }
    return ans;
  }

  double DRDGS::log_inclusion_prior(
      const Selector &inc, int time_index, int predictor_index) const {
    double ans = 0;
    bool inc_now = inc[predictor_index];
    if (time_index != 0) {
      bool inc_prev = model_->inclusion_indicator(
          time_index - 1, predictor_index);
      ans += model_->log_transition_probability(
          inc_prev, inc_now, predictor_index);
    }
    if (time_index + 1 < model_->time_dimension()) {
      bool inc_next = model_->inclusion_indicator(
          time_index + 1, predictor_index);
      ans += model_->log_transition_probability(
          inc_now, inc_next, predictor_index);
    }
    return ans;
  }

  void DRDGS::draw_inclusion_indicators() {
    for (int t = 0; t < model_->time_dimension(); ++t) {
      Selector inc = model_->inclusion_indicators(t);
      for (int j = 0; j < model_->xdim(); ++j) {
        // TODO(steve): Explore FB recursions here.  Easy to do if we can get
        // the prior right.
        mcmc_one_flip(inc, t, j);
      }
      model_->set_inclusion_indicators(t, inc);
    }
  }

  void DRDGS::draw_residual_variance() {
    double sse = 0;
    double sample_size = 0;
    for (int t = 0; t < model_->time_dimension(); ++t) {
      sample_size += model_->data(t)->sample_size();
      sse += model_->data(t)->SSE(model_->coef(t));
    }
    double sigsq = residual_variance_sampler_.draw(rng(), sample_size, sse);
    model_->set_residual_variance(sigsq);
  }

  void DRDGS::draw_unscaled_state_innovation_variance() {
    // The innovation variance is the variance of the innovation model times the
    // residual variance.  To preserve this definition, we must divide dbeta[t]
    // by sigma.
    double sigma = model_->residual_sd();

    for (int j = 0; j < model_->xdim(); ++j) {
      Ptr<GaussianSuf> suf = model_->innovation_error_model(j)->suf();
      suf->clear();
      // The for loop starts from 1 to allow differencing.
      for (int t = 1; t < model_->time_dimension(); ++t) {
        if (model_->inclusion_indicator(t, j)) {
          double dbeta = model_->coefficient(t, j)
              - model_->coefficient(t - 1, j);
          suf->update_raw(dbeta / sigma);
        }
      }
      model_->innovation_error_model(j)->sample_posterior();
    }
  }

  void DRDGS::draw_transition_probabilities() {
    for (int j = 0; j < model_->xdim(); ++j) {
      model_->transition_model(j)->suf()->clear();
      bool then = model_->inclusion_indicator(0, j);
      for (int t = 1; t < model_->time_dimension(); ++t) {
        bool now = model_->inclusion_indicator(t, j);
        model_->transition_model(j)->suf()->add_transition(then, now);
        then = now;
      }
      model_->transition_model(j)->sample_posterior();
    }
  }

  Matrix DRDGS::infer_Markov_prior(double prior_success_prob,
                                   double expected_time,
                                   double sample_size) {
    double pi = prior_success_prob;
    if (pi <= 0 || pi >= 1) {
      report_error("prior_success_prob must be between 0 and 1.");
    }

    double p11 = 1.0 - 1.0 / expected_time;
    if (p11 <= 0.0 || p11 >= 1.0) {
      report_error("expected_time must be greater than 1.");
    }

    if (sample_size <= 0.0) {
      report_error("sample_size must be positive.");
    }

    // The stationary distribution of a 2-state Markov chain is proportional
    // to (p10, p01).  Given pi1 and p11 we can solve for p00.

    double p01 = (1 - p11) * pi / (1 - pi);
    p01 = std::min(p01, .9999);
    p01 = std::max(p01, .0001);
    double p00 = 1 - p01;

    // Given p00, build the Matrix and multiply by prior sample size.
    Matrix ans(2, 2);
    ans(0, 0) = p00;
    ans(0, 1) = 1.0 - p00;
    ans(1, 1) = p11;
    ans(1, 0) = 1.0 - p11;
    return sample_size * ans;
  }

}  // namespace BOOM
