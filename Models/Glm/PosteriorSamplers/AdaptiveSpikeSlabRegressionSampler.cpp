/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/AdaptiveSpikeSlabRegressionSampler.hpp"
#include "distributions.hpp"
#include <algorithm>

namespace BOOM {

  AdaptiveSpikeSlabRegressionSampler::AdaptiveSpikeSlabRegressionSampler(
      RegressionModel *model,
      const Ptr<MvnGivenScalarSigmaBase> &slab,
      const Ptr<GammaModelBase> &residual_precision_prior,
      const Ptr<VariableSelectionPrior> &spike,
      RNG &rng)
      :PosteriorSampler(rng),
       model_(model),
       slab_(slab),
       residual_precision_prior_(residual_precision_prior),
       spike_(spike),
       sigsq_sampler_(residual_precision_prior_),
       allow_model_selection_(true),
       max_flips_(100),
       iteration_count_(0),
       step_size_(.001),
       target_acceptance_rate_(.345),
       birth_rates_(model_->xdim(), 1.0),
       death_rates_(model_->xdim(), 1.0),
       current_log_model_prob_(negative_infinity()),
       store_model_probs_(true)
  {}

  double AdaptiveSpikeSlabRegressionSampler::logpri() const {
    const Selector &included_coefficients(model_->coef().inc());
    double ans = spike_->logp(included_coefficients);  // p(gamma)
    if (ans <= BOOM::negative_infinity()) return ans;

    ans += sigsq_sampler_.log_prior(model_->sigsq());
    if (included_coefficients.nvars() > 0) {
      ans += dmvn(included_coefficients.select(model_->Beta()),
                  included_coefficients.select(slab_->mu()),
                  included_coefficients.select(slab_->siginv()), true);
    }
    return ans;
  }

  void AdaptiveSpikeSlabRegressionSampler::draw() {
    Selector included_coefficients = model_->coef().inc();
    if (allow_model_selection_) {
      int flips = std::min<int>(max_flips_, included_coefficients.nvars_possible());
      current_log_model_prob_ = log_model_prob(included_coefficients);
      for (int i = 0; i < flips; ++i) {
        double u = runif_mt(rng());
        if (u < .5) {
          birth_move(included_coefficients);
        } else {
          death_move(included_coefficients);
        }
      }
      model_->coef().set_inc(included_coefficients);
    }
    set_posterior_moments(included_coefficients);
    draw_residual_variance();
    draw_coefficients();
    ++iteration_count_;
  }

  double AdaptiveSpikeSlabRegressionSampler::log_model_prob(
      const Selector &included_coefficients) {
    if (store_model_probs_) {
      auto lookup = log_model_probabilities_.find(included_coefficients);
      if (lookup != log_model_probabilities_.end()) {
        return lookup->second;
      }
    }

    if (included_coefficients.nvars() == 0) {
      // Integrate out sigma.  The empty model is handled as a special
      // case because information matrices cancel, and do not appear
      // in the sum of squares.  It is easier to handle them here than
      // to impose a global requirement about what logdet() should
      // mean for an empty matrix.
      double ss = model_->suf()->yty() + prior_ss();
      double df = model_->suf()->n() + prior_df();
      return spike_->logp(included_coefficients) - (.5 * df - 1) * log(ss);
    }
    double ans = spike_->logp(included_coefficients);
    if (ans == negative_infinity()) {
      return ans;
    }
    set_posterior_moments(included_coefficients);
    if (logdet_omega_inverse_ <= negative_infinity()) {
      return negative_infinity();
    }
    ans += .5 * (logdet_omega_inverse_ - unscaled_posterior_precision_.logdet())
        - (.5 * posterior_df_ - 1) * log(posterior_sum_of_squares_);
    if (store_model_probs_) {
      log_model_probabilities_[included_coefficients] = ans;
    }
    return ans;
  }

  //---------------------------------------------------------------------------
  void AdaptiveSpikeSlabRegressionSampler::set_posterior_moments(
      const Selector &inclusion_indicators) {
    SpdMatrix unscaled_prior_precision =
        inclusion_indicators.select(slab_->unscaled_precision());
    logdet_omega_inverse_ = unscaled_prior_precision.logdet();
    Vector prior_mean = inclusion_indicators.select(slab_->mu());

    unscaled_posterior_precision_ =
        unscaled_prior_precision
        + model_->suf()->xtx(inclusion_indicators);
    bool positive_definite = true;
    posterior_mean_ = unscaled_posterior_precision_.solve(
        model_->suf()->xty(inclusion_indicators) +
        unscaled_prior_precision * inclusion_indicators.select(slab_->mu()),
        positive_definite);

    posterior_df_ = prior_df() + model_->suf()->n();
    posterior_sum_of_squares_ =
        prior_ss()
        + model_->suf()->relative_sse(
            GlmCoefs(posterior_mean_, inclusion_indicators))
        + unscaled_prior_precision.Mdist(posterior_mean_, prior_mean);
  }

  //---------------------------------------------------------------------------
  // set_posterior_moments must have been called prior to calling draw_coefficients.
  void AdaptiveSpikeSlabRegressionSampler::draw_coefficients() {
    Vector coefficients = rmvn_ivar_mt(
        rng(), posterior_mean_,
        unscaled_posterior_precision_ / model_->sigsq());
    model_->set_included_coefficients(coefficients);
  }

  void AdaptiveSpikeSlabRegressionSampler::draw_residual_variance() {
    double data_df = posterior_df_ - prior_df();
    double data_sumsq = posterior_sum_of_squares_ - prior_ss();
    double sigsq = sigsq_sampler_.draw(rng(), data_df, data_sumsq);
    model_->set_sigsq(sigsq);
  }

  void AdaptiveSpikeSlabRegressionSampler::birth_move(
      Selector &included_coefficients) {
    const Selector excluded = included_coefficients.complement();
    if (excluded.nvars() == 0) {
      return;
    }
    Vector weights = excluded.select(birth_rates_);
    int which = rmulti_mt(rng(), weights);

    uint candidate_index = excluded.indx(which);
    included_coefficients.add(candidate_index);

    double candidate_log_model_prob = log_model_prob(included_coefficients);
    double log_MH_ratio_numerator =
        candidate_log_model_prob - log(weights[which] / sum(weights));
    double log_MH_ratio_denominator =
        current_log_model_prob_
        - log(death_rates_[candidate_index] /
              included_coefficients.sparse_sum(death_rates_));

    double log_MH_ratio = log_MH_ratio_numerator - log_MH_ratio_denominator;
    double logu = log(runif_mt(rng()));
    if (logu < log_MH_ratio) {
      current_log_model_prob_ = candidate_log_model_prob;
      adjust_birth_rate(candidate_index, exp(log_MH_ratio));
    } else {
      included_coefficients.drop(candidate_index);
    }
  }

  void AdaptiveSpikeSlabRegressionSampler::adjust_birth_rate(
      int which_variable, double MH_alpha) {
    MH_alpha = std::min<double>(1.0, MH_alpha);
    double adjustment = step_size_ / ((1.0 + iteration_count_) / model_->xdim());
    adjustment *= (MH_alpha - target_acceptance_rate_);
    birth_rates_[which_variable] *= exp(adjustment);
  }

  void AdaptiveSpikeSlabRegressionSampler::death_move(
      Selector &included_coefficients) {
    if (included_coefficients.nvars() == 0) {
      return;
    }
    Vector weights = included_coefficients.select(death_rates_);
    int which = rmulti_mt(rng(), weights);
    uint candidate_index = included_coefficients.indx(which);
    included_coefficients.drop(candidate_index);
    double candidate_log_model_prob = log_model_prob(included_coefficients);
    double log_MH_ratio_numerator =
        candidate_log_model_prob - log(weights[which] / sum(weights));
    Selector excluded(included_coefficients.complement());
    double log_MH_ratio_denominator =
        current_log_model_prob_
        - log(birth_rates_[candidate_index] / excluded.sparse_sum(birth_rates_));

    double log_MH_ratio = log_MH_ratio_numerator - log_MH_ratio_denominator;
    double logu = log(runif_mt(rng()));
    if (logu < log_MH_ratio) {
      current_log_model_prob_ = candidate_log_model_prob;
      adjust_death_rate(candidate_index, exp(log_MH_ratio));
    } else {
      included_coefficients.add(candidate_index);
    }
  }

  void AdaptiveSpikeSlabRegressionSampler::adjust_death_rate(
      int which_variable, double MH_alpha) {
    MH_alpha = std::min<double>(1.0, MH_alpha);
    double adjustment = step_size_ / ((1.0 + iteration_count_) / model_->xdim());
    adjustment *= (MH_alpha - target_acceptance_rate_);
    death_rates_[which_variable] *= exp(adjustment);
  }

  void AdaptiveSpikeSlabRegressionSampler::set_step_size(double step_size) {
    if (step_size <= 0) {
      report_error("Step size must be positive.");
    }
    step_size_ = step_size;
  }

  void AdaptiveSpikeSlabRegressionSampler::set_target_acceptance_rate(double rate) {
    if (rate <= 0 || rate >= 1) {
      report_error("Target acceptance rate must be strictly between 0 and 1.");
    }
    target_acceptance_rate_ = rate;
  }

}  // namespace BOOM
