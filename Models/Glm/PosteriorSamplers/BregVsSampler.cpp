// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "Models/ChisqModel.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/seq.hpp"
#include "cpputil/shuffle.hpp"
#include "distributions.hpp"
#include "distributions/trun_gamma.hpp"

namespace BOOM {

  namespace {
    typedef BregVsSampler BVS;

    Ptr<GammaModelBase> create_siginv_prior(RegressionModel *model,
                                            double prior_nobs,
                                            double expected_rsq) {
      double sample_variance = model->suf()->SST() / (model->suf()->n() - 1);
      assert(expected_rsq > 0 && expected_rsq < 1);
      double sigma_guess = sqrt(sample_variance * (1 - expected_rsq));
      return new ChisqModel(prior_nobs, sigma_guess);
    }
  }  // namespace

  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *model, double prior_nobs,
                     double expected_rsq, double expected_model_size,
                     bool first_term_is_intercept, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        residual_precision_prior_(
            create_siginv_prior(model, prior_nobs, expected_rsq)),
        indx(seq<uint>(0, model_->nvars_possible() - 1)),
        max_nflips_(indx.size()),
        draw_beta_(true),
        draw_sigma_(true),
        // Initialize mutable workspace variables to illegal values.
        posterior_mean_(1, negative_infinity()),
        unscaled_posterior_precision_(1, negative_infinity()),
        DF_(negative_infinity()),
        SS_(negative_infinity()),
        sigsq_sampler_(residual_precision_prior_),
        failure_count_(0) {
    uint p = model_->nvars_possible();
    Vector prior_mean = Vector(p, 0.0);
    if (first_term_is_intercept) {
      prior_mean[0] = model_->suf()->ybar();
    }
    SpdMatrix ominv(model_->suf()->xtx());
    double n = model_->suf()->n();
    ominv *= prior_nobs / n;
    slab_ = check_slab_dimension(
        new MvnGivenScalarSigma(prior_mean, ominv, model_->Sigsq_prm()));

    double prob = expected_model_size / p;
    if (prob > 1) prob = 1.0;
    Vector pi(p, prob);
    if (first_term_is_intercept) {
      pi[0] = 1.0;
    }

    spike_ = check_spike_dimension(new VariableSelectionPrior(pi));
  }
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *model, double prior_sigma_nobs,
                     double prior_sigma_guess, double prior_beta_nobs,
                     double diagonal_shrinkage,
                     double prior_inclusion_probability, bool force_intercept,
                     RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        residual_precision_prior_(
            new ChisqModel(prior_sigma_nobs, prior_sigma_guess)),
        indx(seq<uint>(0, model_->nvars_possible() - 1)),
        max_nflips_(indx.size()),
        draw_beta_(true),
        draw_sigma_(true),
        sigsq_sampler_(residual_precision_prior_),
        failure_count_(0) {
    uint p = model_->nvars_possible();
    Vector b = Vector(p, 0.0);
    double ybar = model_->suf()->ybar();
    b[0] = ybar;
    SpdMatrix ominv(model_->suf()->xtx());
    double n = model_->suf()->n();

    if (prior_sigma_guess <= 0) {
      ostringstream msg;
      msg << "illegal value of prior_sigma_guess in constructor "
          << "to BregVsSampler" << endl
          << "supplied value:  " << prior_sigma_guess << endl
          << "legal values are strictly > 0";
      report_error(msg.str());
    }
    ominv *= prior_beta_nobs / n;

    // handle diagonal shrinkage:  ominv =alpha*diag(ominv) + (1-alpha)*ominv
    // This prevents a perfectly singular ominv.
    double alpha = diagonal_shrinkage;
    if (alpha > 1.0 || alpha < 0.0) {
      ostringstream msg;
      msg << "illegal value of 'diagonal_shrinkage' in "
          << "BregVsSampler constructor.  Supplied value = " << alpha
          << ".  Legal values are [0, 1].";
      report_error(msg.str());
    }

    if (alpha < 1.0) {
      diag(ominv).axpy(diag(ominv), alpha / (1 - alpha));
      ominv *= (1 - alpha);
    } else {
      ominv.set_diag(diag(ominv));
    }
    slab_ = check_slab_dimension(
        new MvnGivenScalarSigma(b, ominv, model_->Sigsq_prm()));

    Vector pi(p, prior_inclusion_probability);
    if (force_intercept) pi[0] = 1.0;
    spike_ = check_spike_dimension(new VariableSelectionPrior(pi));
  }
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *model, const Vector &prior_mean,
                     const SpdMatrix &unscaled_prior_precision,
                     double sigma_guess, double df,
                     const Vector &prior_inclusion_probs, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slab_(check_slab_dimension(new MvnGivenScalarSigma(
            prior_mean, unscaled_prior_precision, model_->Sigsq_prm()))),
        residual_precision_prior_(new ChisqModel(df, sigma_guess)),
        spike_(check_spike_dimension(
            new VariableSelectionPrior(prior_inclusion_probs))),
        indx(seq<uint>(0, model_->nvars_possible() - 1)),
        max_nflips_(indx.size()),
        draw_beta_(true),
        draw_sigma_(true),
        sigsq_sampler_(residual_precision_prior_),
        failure_count_(0) {}
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *model,
                     const ZellnerPriorParameters &prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slab_(check_slab_dimension(new MvnGivenScalarSigma(
            prior.prior_beta_guess, prior.prior_beta_information,
            model_->Sigsq_prm()))),
        residual_precision_prior_(new ChisqModel(prior.prior_sigma_guess_weight,
                                                 prior.prior_sigma_guess)),
        spike_(check_spike_dimension(
            new VariableSelectionPrior(prior.prior_inclusion_probabilities))),
        indx(seq<uint>(0, model_->nvars_possible() - 1)),
        max_nflips_(indx.size()),
        draw_beta_(true),
        draw_sigma_(true),
        sigsq_sampler_(residual_precision_prior_),
        failure_count_(0) {}
  //----------------------------------------------------------------------
  BVS::BregVsSampler(RegressionModel *model,
                     const Ptr<MvnGivenScalarSigmaBase> &slab,
                     const Ptr<GammaModelBase> &residual_precision_prior,
                     const Ptr<VariableSelectionPrior> &spike, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slab_(check_slab_dimension(slab)),
        residual_precision_prior_(residual_precision_prior),
        spike_(check_spike_dimension(spike)),
        indx(seq<uint>(0, model_->nvars_possible() - 1)),
        max_nflips_(indx.size()),
        draw_beta_(true),
        draw_sigma_(true),
        sigsq_sampler_(residual_precision_prior_),
        failure_count_(0) {}
  //----------------------------------------------------------------------
  void BVS::limit_model_selection(uint n) { max_nflips_ = n; }
  void BVS::allow_model_selection(bool allow) {
    if (allow) {
      max_nflips_ = indx.size();
    } else {
      suppress_model_selection();
    }
  }
  void BVS::suppress_model_selection() { max_nflips_ = 0; }
  void BVS::suppress_beta_draw() { draw_beta_ = false; }
  void BVS::allow_beta_draw() { draw_beta_ = false; }
  void BVS::suppress_sigma_draw() { draw_sigma_ = false; }
  void BVS::allow_sigma_draw() { draw_sigma_ = false; }

  //  since alpha = df/2 df is 2 * alpha, likewise for beta
  double BVS::prior_df() const {
    return 2 * residual_precision_prior_->alpha();
  }
  double BVS::prior_ss() const { return 2 * residual_precision_prior_->beta(); }

  double BVS::log_model_prob(const Selector &g) const {
    if (g.nvars() == 0) {
      // Integrate out sigma.  The empty model is handled as a special
      // case because information matrices cancel, and do not appear
      // in the sum of squares.  It is easier to handle them here than
      // to impose a global requirement about what logdet() should
      // mean for an empty matrix.
      double ss = model_->suf()->yty() + prior_ss();
      double df = model_->suf()->n() + prior_df();
      double ans = spike_->logp(g) - (.5 * df - 1) * log(ss);
      return ans;
    }
    double ans = spike_->logp(g);
    if (ans == negative_infinity()) {
      return ans;
    }
    double ldoi = set_reg_post_params(g, true);
    if (ldoi <= negative_infinity()) {
      return negative_infinity();
    }
    ans += .5 * (ldoi - unscaled_posterior_precision_.logdet());
    ans -= (.5 * DF_ - 1) * log(SS_);
    return ans;
  }
  //----------------------------------------------------------------------
  double BVS::mcmc_one_flip(Selector &model, uint which_var, double logp_old) {
    model.flip(which_var);
    double logp_new = log_model_prob(model);
    double u = runif_mt(rng(), 0, 1);
    if (log(u) > logp_new - logp_old) {
      model.flip(which_var);  // reject draw
      return logp_old;
    }
    return logp_new;
  }
  //----------------------------------------------------------------------
  void BVS::draw() {
    if (max_nflips_ > 0) {
      draw_model_indicators();
    }
    if (draw_beta_ || draw_sigma_) {
      set_reg_post_params(model_->coef().inc(), false);
    }
    if (draw_sigma_) draw_sigma();
    if (draw_beta_) draw_beta();
  }
  //----------------------------------------------------------------------
  bool BVS::model_is_empty() const { return model_->coef().inc().nvars() == 0; }
  //----------------------------------------------------------------------
  void BVS::set_sigma_upper_limit(double sigma_upper_limit) {
    sigsq_sampler_.set_sigma_max(sigma_upper_limit);
  }

  void BVS::find_posterior_mode(double) {
    set_reg_post_params(model_->coef().inc(), true);
    model_->set_included_coefficients(posterior_mean_);
    model_->set_sigsq(SS_ / DF_);
  }

  void BVS::attempt_swap() {
    if (correlation_map_.threshold() >= 1.0) {
      return;
    }
    if (!correlation_map_.filled()) {
      correlation_map_.fill(*model_->suf());
    }
    Selector included = model_->coef().inc();
    if (included.nvars() == 0 ||
        included.nvars() == included.nvars_possible()) {
      return;
    }
    int index = included.random_included_position(rng());
    double forward_proposal_weight;
    int candidate = correlation_map_.propose_swap(
        rng(), included, index, &forward_proposal_weight);
    if (candidate < 0) return;

    double original_model_log_probability = log_model_prob(included);
    included.drop(index);
    included.add(candidate);
    double reverse_proposal_weight = correlation_map_.proposal_weight(
        included, candidate, index);
    double log_MH_numerator =
        log_model_prob(included) - log(forward_proposal_weight);
    double log_MH_denominator =
        original_model_log_probability - log(reverse_proposal_weight);
    double logu = log(runif_mt(rng()));
    if (logu < log_MH_numerator - log_MH_denominator) {
      model_->coef().set_inc(included);
    } else {
      // reject the proposal by doing nothing.
    }
  }

  //----------------------------------------------------------------------
  void BVS::draw_sigma() {
    double df, ss;
    if (model_is_empty()) {
      ss = model_->suf()->yty();
      df = model_->suf()->n();
    } else {
      df = DF_ - prior_df();
      ss = SS_ - prior_ss();
    }
    double sigsq = draw_sigsq_given_sufficient_statistics(df, ss);
    model_->set_sigsq(sigsq);
  }
  //----------------------------------------------------------------------
  void BVS::draw_beta() {
    if (model_is_empty()) return;
    SpdMatrix posterior_precision =
        unscaled_posterior_precision_ / model_->sigsq();
    // The posterior precision might be nearly rank deficient.
    bool ok = false;
    Matrix posterior_precision_lower_cholesky = posterior_precision.chol(ok);
    if (ok) {
      posterior_mean_ = rmvn_precision_upper_cholesky_mt(
          rng(), posterior_mean_,
          posterior_precision_lower_cholesky.transpose());
      model_->set_included_coefficients(posterior_mean_);
      failure_count_ = 0;
    } else {
      // Handle the case where the information matrix is degenerate.  This
      // should not happen mathematically, but it might happen for numerical
      // reasons.  If we're here it is because the variable selection component
      // messed up, so just bail on this draw and try again.
      if (++failure_count_ > 10) {
        report_error("The posterior information matrix is not positive "
                     "definite.  Check your data or consider adjusting "
                     "your prior.");
      }
      draw();
    }
  }
  //----------------------------------------------------------------------
  void BVS::draw_model_indicators() {
    Selector g = model_->coef().inc();
    shuffle(indx, rng());
    double logp = log_model_prob(g);

    if (!std::isfinite(logp)) {
      // If the model starts from an illegal configuration, legalize it (move it
      // to a place of positive prior probability) before we begin.
      spike_->make_valid(g);
      logp = log_model_prob(g);
    }
    if (!std::isfinite(logp)) {
      ostringstream err;
      err << "BregVsSampler did not start with a legal configuration." << endl
          << "Selector vector:  " << g << endl
          << "beta: " << model_->included_coefficients() << endl;
      report_error(err.str());
    }

    uint n = std::min<uint>(max_nflips_, g.nvars_possible());
    for (uint i = 0; i < n; ++i) {
      logp = mcmc_one_flip(g, indx[i], logp);
    }
    model_->coef().set_inc(g);
    attempt_swap();
  }
  //----------------------------------------------------------------------
  double BVS::logpri() const {
    const Selector &g(model_->coef().inc());
    double ans = spike_->logp(g);  // p(gamma)
    if (ans <= BOOM::negative_infinity()) return ans;

    double sigsq = model_->sigsq();
    ans += sigsq_sampler_.log_prior(sigsq);

    if (g.nvars() > 0) {
      ans += dmvn(g.select(model_->Beta()), g.select(slab_->mu()),
                  g.select(slab_->siginv()), true);
    }
    return ans;
  }
  //----------------------------------------------------------------------
  double BVS::set_reg_post_params(const Selector &inclusion_indicators,
                                  bool do_ldoi) const {
    if (inclusion_indicators.nvars() == 0) {
      return 0;
    }
    Vector prior_mean = inclusion_indicators.select(slab_->mu());
    // Sigma = sigsq * Omega, so
    // siginv = ominv / sigsq, so
    // ominv = siginv * sigsq.
    SpdMatrix unscaled_prior_precision =
        inclusion_indicators.select(slab_->unscaled_precision());
    double ldoi = do_ldoi ? unscaled_prior_precision.logdet() : 0.0;

    Ptr<RegSuf> s = model_->suf();

    SpdMatrix xtx = s->xtx(inclusion_indicators);
    Vector xty = s->xty(inclusion_indicators);

    // unscaled_posterior_precision_ / sigsq is the conditional posterior
    // precision matrix, given inclusion_indicators.
    unscaled_posterior_precision_ = unscaled_prior_precision + xtx;
    // posterior_mean_ is the posterior mean, given inclusion_indicators.
    posterior_mean_ = unscaled_prior_precision * prior_mean + xty;
    bool positive_definite = true;
    posterior_mean_ =
        unscaled_posterior_precision_.solve(posterior_mean_, positive_definite);
    if (!positive_definite) {
      posterior_mean_ = Vector(unscaled_posterior_precision_.nrow());
      return negative_infinity();
    }
    DF_ = s->n() + prior_df();
    // SS_ starts off with the prior sum of squares from the prior on sigma^2.
    SS_ = prior_ss();
    if (!std::isfinite(SS_)) {
      report_error("Prior sum of squares is wrong.");
    }

    // Add in the sum of squared errors around posterior_mean_
    double likelihood_ss =
        s->yty() - 2 * posterior_mean_.dot(xty) + xtx.Mdist(posterior_mean_);
    SS_ += likelihood_ss;
    if (!std::isfinite(SS_)) {
      report_error("Quadratic form caused infinite SS.");
    }

    // Add in the sum of squares component arising from the discrepancy between
    // the prior and posterior means.
    SS_ += unscaled_prior_precision.Mdist(posterior_mean_, prior_mean);
    if (SS_ < 0) {
      report_error(
          "Illegal data caused negative sum of squares "
          "in Breg::set_reg_post_params.");
    } else if (!std::isfinite(SS_)) {
      report_error("Prior to Posterior Mahalanobis distance caused "
                   "infinite SS.");
    }
    return ldoi;
  }

  const Ptr<MvnGivenScalarSigmaBase> &BVS::check_slab_dimension(
      const Ptr<MvnGivenScalarSigmaBase> &slab) {
    if (slab->dim() != model_->xdim()) {
      report_error("Slab dimension did not match model dimension.");
    }
    return slab;
  }

  const Ptr<VariableSelectionPrior> &BVS::check_spike_dimension(
      const Ptr<VariableSelectionPrior> &spike) {
    if (spike->potential_nvars() != model_->xdim()) {
      report_error("Spike dimension did not match model dimension.");
    }
    return spike;
  }

}  // namespace BOOM
