// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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
#include "Models/Glm/PosteriorSamplers/SpikeSlabDaRegressionSampler.hpp"
#include <functional>
#include "LinAlg/SWEEP.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef SpikeSlabDaRegressionSampler SSDRS;
  }

  SSDRS::SpikeSlabDaRegressionSampler(
      RegressionModel *model,
      const Ptr<IndependentMvnModelGivenScalarSigma> &beta_prior,
      const Ptr<GammaModelBase> &siginv_prior,
      const Vector &prior_inclusion_probabilities,
      double complete_data_information_matrix_fudge_factor,
      double fallback_probability, RNG &seeding_rng)
      : BregVsSampler(model, beta_prior, siginv_prior,
                      new VariableSelectionPrior(prior_inclusion_probabilities),
                      seeding_rng),
        model_(model),
        beta_prior_(beta_prior),
        siginv_prior_(siginv_prior),
        log_prior_inclusion_probabilities_(
            prior_inclusion_probabilities.size()),
        log_prior_exclusion_probabilities_(
            prior_inclusion_probabilities.size()),
        missing_design_matrix_(model_->xdim(), model_->xdim()),
        missing_y_(model_->xdim()),
        complete_data_xtx_diagonal_(model_->xdim()),
        complete_data_xty_(model_->xdim()),
        prior_is_current_(false),
        fallback_probability_(fallback_probability) {
    for (int i = 0; i < log_prior_inclusion_probabilities_.size(); ++i) {
      double p = prior_inclusion_probabilities[i];
      log_prior_inclusion_probabilities_[i] =
          p > 0 ? log(p) : negative_infinity();
      p = 1.0 - p;
      log_prior_exclusion_probabilities_[i] =
          p > 0 ? log(p) : negative_infinity();
    }
    determine_missing_design_matrix(
        complete_data_information_matrix_fudge_factor);
    compute_leverage_of_missing_design_points();
    beta_prior_->prm1()->add_observer(
        this,
        [this]() { this->observe_changes_in_prior(); });
    beta_prior_->prm2()->add_observer(
        this,
        [this]() { this->observe_changes_in_prior(); });
    check_prior();
  }

  //----------------------------------------------------------------------
  double SSDRS::logpri() const {
    check_prior();
    // Prior is evaluated on the scale of the variance, not the
    // precision, so we must include the jacobian of the reciprocal
    // transformation.
    double ans =
        siginv_prior_->logp(1.0 / model_->sigsq()) - 2 * log(model_->sigsq());
    const Vector &beta(model_->Beta());
    const Selector &inclusion_indicators(model_->coef().inc());
    for (int i = 0; i < log_prior_inclusion_probabilities_.size(); ++i) {
      if (inclusion_indicators[i]) {
        ans += log_prior_inclusion_probabilities_[i] +
               dnorm(beta[i], beta_prior_->mu()[i],
                     beta_prior_->sd_for_element(i), true);
      } else {
        ans += log_prior_exclusion_probabilities_[i];
      }
      if (ans <= BOOM::negative_infinity()) return ans;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  void SSDRS::draw() {
    if (fallback_probability_ > 0 && runif_mt(rng()) < fallback_probability_) {
      BregVsSampler::draw();
    } else {
      impute_latent_data();
      draw_model_indicators_given_complete_data();
      draw_beta_given_observed_data();
      draw_sigma_given_observed_data();
    }
  }

  //----------------------------------------------------------------------
  void SSDRS::impute_latent_data() {
    // complete_data_xty_ is
    // (X - Xbar) * (y - ybar_hat * one)
    //
    // where X is the observed design matrix, Xbar is a matrix where
    // all rows are equal to xbar, the column means of X, y is the
    // observed response vector, ybar_hat is the scalar xbar * beta,
    // and one is a column vector of 1's.  The computation is
    // simplified by (X - Xbar)^T * one = 0, leaving us with (X -
    // Xbar) * y.  We build complete_data_xty_ in a 3 step process.
    // Step 1: Initialize with X * y
    complete_data_xty_ = model_->suf()->xty();

    // Step 2: Subtract off the effect of centering (i.e -= Xbar * y)
    int n = model_->suf()->n();
    double ybar = model_->suf()->ybar();
    complete_data_xty_.axpy(model_->suf()->xbar(), -n * ybar);

    // Step 3: Draw the missing data, and accumulate their
    // contributions.
    //
    // missing_design_matrix_ already has xbar subtracted out, so
    // missing_y_ is (x - xbar) * beta.  The error term gets added in
    // the loop below.
    missing_y_ = model_->coef().predict(missing_design_matrix_);
    double sigma = model_->sigma();
    for (int i = 0; i < missing_y_.size(); ++i) {
      missing_y_[i] += rnorm_mt(rng(), 0, sigma);
      complete_data_xty_.axpy(missing_design_matrix_.row(i), missing_y_[i]);
    }
  }

  //----------------------------------------------------------------------
  void SSDRS::draw_model_indicators_given_complete_data() {
    Selector inclusion_indicators = model_->coef().inc();
    int N = inclusion_indicators.nvars_possible();
    // Special care is needed here because the intercept is not
    // identified in the centered model.  This is why the loop counter
    // starts from 1 instead of 0 in the following loop.  The
    // following will lead to the correct posterior distribution even
    // if there is no intercept term.
    for (int i = 1; i < N; ++i) {
      double inclusion_probability = compute_inclusion_probability(i);
      double u = runif_mt(rng());
      if (u < inclusion_probability) {
        inclusion_indicators.add(i);
      } else {
        inclusion_indicators.drop(i);
      }
    }
    model_->coef().set_inc(inclusion_indicators);
    draw_intercept_indicator();
  }

  //----------------------------------------------------------------------
  void SSDRS::draw_intercept_indicator() {
    // If prior_inclusion_probabilities >= 1 then you know for sure
    // the intercept is included.  These are stored on the log scale.
    if (log_prior_inclusion_probabilities_[0] >= 0.0) {
      model_->coef().add(0);
      return;
    } else if (log_prior_exclusion_probabilities_[0] >= 0.0) {
      // If the probability of _excluding_ the intercept is >= 1 then
      // we know for sure we we can drop it.
      model_->coef().drop(0);
      return;
    }
    Selector inclusion_indicators = model_->coef().inc();
    bool intercept_is_included = inclusion_indicators[0];
    double original_logp = log_model_prob(inclusion_indicators);
    mcmc_one_flip(inclusion_indicators, 0, original_logp);
    if (inclusion_indicators[0] != intercept_is_included) {
      // We only need to update the inclusion indicators if something
      // changed.
      model_->coef().set_inc(inclusion_indicators);
    }
  }

  //----------------------------------------------------------------------
  // Returns the probability that variable j is included in the model,
  // given complete data and sigma, but integrating out beta.  The
  // formula for this is
  //
  //  pi[j] = unknown_constant * prior_probability[j]
  //    * (prior_precision[j] / posterior_precision[j])^.5
  //    * exp(-.5 SSE[j] + SSB[j])
  //
  // where
  // SSE[j] = -2*\tilde\beta[j] * xty[j] + \tilde\beta[j]^2 * xtx[j]
  // SSB[j] = (\tilde\beta[j] - b[j])^2 / prior_variance[j]
  //
  // \tilde\beta = xty[j] / (xtx[j] + prior_information[j])
  //
  // For derivation, compile the latex code at the end of this file.
  double SSDRS::compute_inclusion_probability(int j) const {
    check_prior();
    double prior_mean = beta_prior_->mu()[j];
    double unscaled_posterior_information =
        complete_data_xtx_diagonal_[j] + unscaled_prior_information(j);
    double posterior_mean = posterior_mean_beta_given_complete_data(j);

    double SSE = square(posterior_mean) * complete_data_xtx_diagonal_[j] -
                 2 * posterior_mean * complete_data_xty_[j];
    double SSB =
        square(posterior_mean - prior_mean) * unscaled_prior_information(j);

    double logp_in = log_prior_inclusion_probabilities_[j] +
                     .5 * (log(unscaled_prior_information(j)) -
                           log(unscaled_posterior_information) -
                           (SSE + SSB) / model_->sigsq());

    double logp_out = log_prior_exclusion_probabilities_[j];

    double M = std::max<double>(logp_in, logp_out);
    double prob_in = exp(logp_in - M);
    double prob_out = exp(logp_out - M);
    double total = prob_in + prob_out;
    return prob_in / total;
  }

  //----------------------------------------------------------------------
  double SSDRS::unscaled_prior_information(int j) const {
    check_prior();
    return unscaled_prior_precision_[j];
  }

  //----------------------------------------------------------------------
  void SSDRS::draw_sigma_given_observed_data() {
    const RegSuf &suf(*(model_->suf()));
    double sigsq = draw_sigsq_given_sufficient_statistics(
        suf.n(), suf.relative_sse(model_->coef()));
    model_->set_sigsq(sigsq);
  }

  //----------------------------------------------------------------------
  // Draws regression coefficients given observed data and sigsq.
  void SSDRS::draw_beta_given_observed_data() {
    const Selector &inclusion_indicators(model_->coef().inc());
    if (inclusion_indicators.nvars() == 0) {
      return;
    }
    const RegSuf &suf(*(model_->suf()));

    SpdMatrix posterior_information = suf.xtx(inclusion_indicators);
    Vector unscaled_prior_information =
        1.0 /
        inclusion_indicators.select(beta_prior_->unscaled_variance_diagonal());
    posterior_information.diag() += unscaled_prior_information;

    Vector prior_mean = inclusion_indicators.select(beta_prior_->mu());
    Vector posterior_mean =
        suf.xty(inclusion_indicators) + unscaled_prior_information * prior_mean;
    posterior_mean = posterior_information.solve(posterior_mean);

    posterior_information /= model_->sigsq();
    Vector included_coefficients =
        rmvn_ivar_mt(rng(), posterior_mean, posterior_information);
    model_->set_included_coefficients(included_coefficients);
  }

  //----------------------------------------------------------------------
  void SSDRS::observe_changes_in_prior() const { prior_is_current_ = false; }

  //----------------------------------------------------------------------
  double SSDRS::posterior_mean_beta_given_complete_data(int j) const {
    double posterior_information =
        complete_data_xtx_diagonal_[j] + unscaled_prior_precision_[j];
    return (complete_data_xty_[j] + information_weighted_prior_mean(j)) /
           posterior_information;
  }

  //----------------------------------------------------------------------
  void SSDRS::check_prior() const {
    if (!prior_is_current_) {
      unscaled_prior_precision_ =
          1.0 / beta_prior_->unscaled_variance_diagonal();
      information_weighted_prior_mean_ =
          beta_prior_->mu() * unscaled_prior_precision_;
    }
    prior_is_current_ = true;
  }

  //----------------------------------------------------------------------
  double SSDRS::information_weighted_prior_mean(int j) const {
    check_prior();
    return information_weighted_prior_mean_[j];
  }

  //----------------------------------------------------------------------
  namespace {
    // Find elements of m that are likely to be true mathematical
    // zeros, but which are nonzero because of numerical error.
    // Replace them with actual zeros.
    void detect_numerical_zeros(Matrix &m) {
      const double root_epsilon = sqrt(std::numeric_limits<double>::epsilon());
      for (int i = 0; i < m.nrow(); ++i) {
        for (int j = 0; j < m.ncol(); ++j) {
          if (fabs(m(i, j)) < root_epsilon) {
            m(i, j) = 0.0;
          }
        }
      }
    }
  }  // namespace
  //----------------------------------------------------------------------
  void SSDRS::determine_missing_design_matrix(
      double complete_data_information_matrix_fudge_factor) {
    // Scale xtx to have constant diagonal.  Assuming X has an
    // intercept, centering xtx sets the first element to zero.  The
    // other diagonal elements will be 1.
    SpdMatrix centered_xtx = model_->suf()->centered_xtx();
    Vector scale_factor = sqrt(centered_xtx.diag());
    int number_of_variables = ncol(centered_xtx);

    // If the model has an intercept term then the upper left element
    // of xtx is the sample size, but after centering (which makes xtx
    // proportional to the variance) the element is zero.
    double root_machine_epsilon = sqrt(std::numeric_limits<double>::epsilon());
    bool has_intercept = fabs(model_->suf()->n() - model_->suf()->xtx()(0, 0)) <
                             root_machine_epsilon &&
                         fabs(centered_xtx(0, 0)) < root_machine_epsilon;

    for (int i = has_intercept; i < number_of_variables; ++i) {
      for (int j = has_intercept; j < number_of_variables; ++j) {
        double scale = scale_factor[i] * scale_factor[j];
        if (isnan(scale) || scale == 0.0) {
          scale = 1.0;
        }
        centered_xtx(i, j) /= scale;
      }
    }
    double max_eigenvalue = largest_eigenvalue(centered_xtx);
    complete_data_xtx_diagonal_ =
        max_eigenvalue * (1 + complete_data_information_matrix_fudge_factor);
    if (has_intercept) {
      complete_data_xtx_diagonal_[0] = 0.0;
    }

    // Set xtx_missing to the scaled version of D - centered_xtx.
    SpdMatrix xtx_missing = -centered_xtx;
    xtx_missing.diag() += complete_data_xtx_diagonal_;
    detect_numerical_zeros(xtx_missing);

    // And finally our missing design matrix is the square root of
    // xtx_missing.
    missing_design_matrix_ = eigen_root(xtx_missing);
    if (has_intercept) {
      missing_design_matrix_.col(0) = 0.0;
    }

    for (int i = 0; i < number_of_variables; ++i) {
      missing_design_matrix_.col(i) *= scale_factor[i];
      complete_data_xtx_diagonal_[i] *= square(scale_factor[i]);
    }
  }

  //----------------------------------------------------------------------
  void SSDRS::compute_leverage_of_missing_design_points() {
    missing_leverage_.resize(nrow(missing_design_matrix_));
    for (int i = 0; i < missing_leverage_.size(); ++i) {
      missing_leverage_[i] =
          complete_data_leverage(missing_design_matrix_.row(i));
    }
  }

  //----------------------------------------------------------------------
  double SSDRS::complete_data_leverage(const ConstVectorView &x) const {
    return (ConstVectorView(x, 1) /
            ConstVectorView(complete_data_xtx_diagonal_, 1))
        .dot(ConstVectorView(x, 1));
  }

}  // namespace BOOM

/*
 * Documentation for the posterior distribution of model indicators.
 *
\documentclass{article}
\usepackage{amsmath}
\usepackage{fullpage}
\newcommand{\nc}{\newcommand}
\nc{\bx}{{\bf x}}
\nc{\bX}{{\bf X}}
\nc{\by}{{\bf y}}
\nc{\ominv}{\Omega^{-1}}
\nc{\xtx}{\bX^T\bX}
\nc{\xty}{\bX^T\by}

\title{Notes on orthogonal data augmentation}
\author{Steven L. Scott}
\begin{document}
\maketitle

Begin with the model
\begin{align*}
 y &\sim N(X \beta, \sigma^2 I) \\
 \beta |\gamma &\sim N(b_\gamma, \sigma^2 D_\gamma^{-1}) \\
 1/\sigma^2 &\sim Ga(df/2, ss/2)\\
 \gamma &\sim p(\gamma) = \prod_j \pi_j^{\gamma_j}(1 - \pi_j)^{1 - \gamma_j}
\end{align*}
where $D$ is a diagonal matrix.  We can factor the joint distribution
of all knowns and unknowns in two ways
\begin{equation*}
  p(\by|\beta, \gamma, \sigma) p(\beta | \gamma, \sigma)p(\gamma) p(1/\sigma^2)
= p(\beta|\gamma, \sigma, \by) p(\gamma | \sigma, \by) p(\sigma|\by) p(\by).
\end{equation*}

Subsuming distributions which do not depend on $\gamma$ into the
proportionality constant gives
\begin{equation*}
p(\gamma |\sigma, \by) \propto
\frac{
  p(\gamma) p(\by |\beta, \gamma, \sigma) p(\beta | \gamma, \sigma)
}{
  p(\beta |\gamma, \sigma, \by)
}
\end{equation*}
Denote the posterior variance of $\beta$ in the complete model (with
all $\gamma_j = 1$) by $V^{-1} = (\xtx + D) / \sigma^2$, and the
posterior mean by $\tilde\beta = (\xtx + D)^{-1}(\xty + Db)$.
Because $\xtx$ and $D$ are both diagonal, notice that $V^{-1}_\gamma$
and $\tilde\beta_\gamma$ are just the indicated subsets of $V^{-1}$
and $\tilde \beta$.

Now, because $p(\gamma|\sigma, \by)$ does not depend on $\beta$ we can
evaluate the previous expression at any beta we like.  Evaluate at
$\tilde \beta_\gamma$ to obtain
\begin{equation*}
p(\gamma|\by, \sigma) \propto
p(\gamma)
\frac{
 |D_\gamma/\sigma^2|^{1/2}
}{
  |V_\gamma^{-1}|^{1/2}
}
\exp\left( -\frac{1}{2} \left[
    SSE_\gamma(\tilde \beta) +
    (\tilde\beta - b)_\gamma^TD_\gamma(\tilde\beta - b)_\gamma
  \right] / \sigma^2 \right).
\end{equation*}

Write
\begin{equation*}
 SSE_\gamma(\tilde \beta) = \by^T\by - 2 \tilde\beta^T_\gamma(\xty)_\gamma +
\tilde\beta^T\xtx\tilde\beta. \end{equation*} Discard the $\by^T\by$ term, which
contains no $\gamma$'s, and write the other terms as \begin{equation*}
  \tilde\beta_\gamma^T (\xty)_\gamma = \sum_j [\tilde\beta_j(\xty)_j]^{\gamma_j}
  \qquad
  \text{and}
  \qquad
  \tilde\beta_\gamma^T\xtx_\gamma\tilde\beta\gamma = \sum_j (\tilde\beta_j^2
\xtx_{jj})^{\gamma_j}. \end{equation*}

Then we may write
\begin{equation*}
  p(\gamma|\by, \sigma) \propto
  \prod_{j=1}^p
  \left[\pi_j \sqrt{d_j / v_j} \exp\left(
    -\frac{1}{2} Q_j / \sigma^2
    \right)
    \right]^{\gamma_j} (1 - \pi_j)^{1- \gamma_j}
\end{equation*}
where
$$
Q_j = \tilde\beta_j^2 \xtx_{jj} - 2 \tilde\beta_j\xty_j + D_j(\tilde\beta_j -
b_j)^2.
$$
\end{document}
 */
