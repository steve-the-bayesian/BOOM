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

#ifndef BOOM_SPIKE_SLAB_DA_REGRESSION_SAMPLER_HPP_
#define BOOM_SPIKE_SLAB_DA_REGRESSION_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/IndependentMvnModelGivenScalarSigma.hpp"
#include "Models/MvnGivenSigma.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  // A posterior sampler for linear models under a spike and slab prior
  // based on the data augmentation algorithm from Clyde and Ghosh
  // (2011).
  //
  // This sampler is slightly different than the Clyde and Ghosh method.
  // Let y = X * beta + error.  The prior distribution is
  //   *  p(beta|sigma) ~ind N(b[i], sigma^2 v[i] gamma[i])
  //   *  gamma[i] ~ind Bernoulli(pi[i])
  //   *  1 / sigma^2 ~ ChiSquare(df, sigma_guess)
  //
  // The Clyde and Ghosh observation was that if X'X was diagonal then
  // p(gamma | Y, sigma^2) would be the product of independent
  // Bernoulli's.  So introduce a [p x p] matrix Xa (a for
  // 'agumented').  Let Xc' = [X'Xa'], with Xa chosen so that Xc'Xc is
  // diagonal.  Then all you need to do is impute the Ya's that go
  // along with Xa, and given that complete data you can compute
  // p(gamma | Yc).
  //
  // Ghosh and Clyde explain that the diagonal matrix should be a
  // constant with elements equal to the largest diagonal of X'X
  // (after scaling X'X, which introduces a minor bit of accounting).
  // The problem is that when you take any square root of D - X'X you
  // can end up with an Xa containing some very high leverage points.
  // High leverage in the missing data means the missing points can
  // essentially determine the mean function, leading to high
  // correlation between the missing Y's and the model parameters, and
  // poor mixing.
  //
  // One thing that helps with the high leverage problem is to center
  // the X's before scaling and diagonalizing.  Thus we work with
  // Xstar = X - Xbar, where Xbar is a matrix with n identical rows
  // containing the column means of X.  To use this trick we must
  // reparameterize the model a bit, with Y = Xbar * beta + Xstar *
  // beta + error.  We have a new latent variable Ystar = Y - Xbar *
  // beta (which is latent becase beta is unknown).  Written another
  // way, Ystar = Xstar * beta + error.
  //
  // The algorithm implemented by this sampler is:
  // At construction time, find
  // Xa = sqrt(D * (1 + fudge_factor) - Xstar'Xstar).
  // Then repeatedly iterate through the following steps:
  //
  // (1) Given values of beta and sigma, simulate Yc = (Ystar, Ya_star).
  //   (1a) Ystar = Y - Xbar * beta  (so Ystar = Xstar * beta + error)
  //   (1b) Ya_star = Xstar * beta + error.
  //     At the end of step 1 we have
  //     Yc = (Ystar, Ya_star) ~ (Xstar, Xa) * beta + errror.
  //     The intercept term is not identified
  //     because centering X replaced the initial column of 1's with a
  //     column of 0's.
  // (2) Draw gamma given Yc, Xstar, and sigma, independently for each
  //     element of gamma.  Run one step of SSVS to draw gamma[0]
  //     given (only) observed data.
  // (3) Draw beta given sigma and (observed) Y.
  // (4) Draw sigma given beta and (observed) Y.
  class SpikeSlabDaRegressionSampler : public BregVsSampler {
   public:
    // Args:
    //   model: The model for which posterior draws are desired.  The
    //     data should have already been assigned to the model before
    //     being passed here.  The sampler will use information about
    //     the design matrix that cannot be changed after the
    //     constructor is called.  The first column of the design
    //     matrix for the model is assumed to contain an intercept
    //     (all 1's).
    //   beta_prior: Prior distribution for regression coefficients,
    //     given inclusion.
    //   siginv_prior:  Prior distribution for the residual variance.
    //   prior_inclusion_probabilities: Prior probability that each
    //     variable is "in" the model.
    //   complete_data_information_matrix_fudge_factor: This argument
    //     should be a small positive number.  Its purpose is to
    //     prevent numerical underflow.  The complete data cross
    //     product matrix (XTX) will be set to a constant diagonal
    //     matrix with elements d * (1 + cdimff), where d is the
    //     largest eigenvalue of the (centered) observed cross product
    //     matrix.
    //   fallback_probability: When there is a high degree of
    //     dependence on the latent data this sampler can get confused
    //     and give you slow mixing.  To counter this, you can mix in
    //     some fraction of stochastic search variable selection
    //     (SSVS) draws, which are slower but often more robust.  On
    //     any given iteration, this sampler will fall back to SSVS
    //     with probability 'fallback_probability.'
    SpikeSlabDaRegressionSampler(
        RegressionModel *model,
        const Ptr<IndependentMvnModelGivenScalarSigma> &beta_prior,
        const Ptr<GammaModelBase> &siginv_prior,
        const Vector &prior_inclusion_probabilities,
        double complete_data_information_matrix_fudge_factor = .01,
        double fallback_probability = 0.0, RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

    // Compute the inclusion probability of coefficient i given complete
    // data.  The complete data makes all the coefficients independent.
    double compute_inclusion_probability(int i) const;

    void impute_latent_data();

    // The prior information for variable j.
    double unscaled_prior_information(int j) const;

    // Returns the leverage of the data point 'x' with respect to the
    // centered complete data design matrix.  This really only a
    // proper 'leverage' if x is a row of the training data (either
    // observed or latent).
    double complete_data_leverage(const ConstVectorView &x) const;

    //----------------------------------------------------------------------
    // Accessors for private objects, exposed for testing.
    const Vector &log_prior_inclusion_probabilities() const {
      return log_prior_inclusion_probabilities_;
    }
    const Vector &log_prior_exclusion_probabilities() const {
      return log_prior_exclusion_probabilities_;
    }
    const Matrix &missing_design_matrix() const {
      return missing_design_matrix_;
    }
    const Vector &missing_leverage() const { return missing_leverage_; }
    const Vector &complete_data_xtx_diagonal() const {
      return complete_data_xtx_diagonal_;
    }
    const Vector &missing_y() const { return missing_y_; }
    const Vector &complete_data_xty() const { return complete_data_xty_; }

   private:
    // NOTE: This function assumes that the first column of the
    // original X matrix is all 1's.
    void determine_missing_design_matrix(
        double complete_data_information_matrix_fudge_factor);

    // After calling determine_missing_design_matrix(), it can be
    // useful to determine if the missing points are high leverage
    // points.
    void compute_leverage_of_missing_design_points();

    // The draw of the model indicators is given sigma, but with beta
    // integrated out.  Otherwise the relevant sums of squares do not
    // factor as a product of per-variable contributions.
    // draw_model_indicators_given_complete_data() calls
    // draw_intercept_indicator().
    void draw_model_indicators_given_complete_data();
    void draw_intercept_indicator();

    // Note, the draw of beta is given sigma.  Integrating over sigma
    // (i.e. not conditioning on it) would make the marginal
    // distribution of the model indicators polynomial (e.g. it would
    // look like the T distribution) instead of exponential
    // (e.g. looking like the Gaussian distribution).  We need it to
    // be exponential so it factors variable-by-variable.
    void draw_beta_given_observed_data();

    // This is a slightly different function than draw_sigsq() found
    // in the BregVsSampler base class.  That one draws sigsq() given
    // model indicators, integrating out the nonzero coefficients.
    // This one conditions on the values of the coefficients.
    void draw_sigma_given_observed_data();

    // This is an 'observer' to be attached to the parameters of the
    // prior distribution, so we can be notified when they change.  It
    // sets prior_is_current_ to false.
    void observe_changes_in_prior() const;

    // If the prior is not current, then reset the
    // unscaled_prior_precision_ and information_weighted_prior_mean_,
    // and then mark the prior as current.
    void check_prior() const;

    double information_weighted_prior_mean(int j) const;
    double posterior_mean_beta_given_complete_data(int j) const;

    RegressionModel *model_;
    Ptr<IndependentMvnModelGivenScalarSigma> beta_prior_;
    Ptr<GammaModelBase> siginv_prior_;

    // The prior probability that each varaiable is included in the
    // model.  The intercept can be excluded just as any other
    // variable.
    Vector log_prior_inclusion_probabilities_;

    // exp(log_prior_exclusion_probabilities_) =
    // 1-exp(log_prior_inclusion_probabilities_)
    Vector log_prior_exclusion_probabilities_;

    // The missing design matrix is the upper cholesky triangle of the
    // sum of squares matrix required to diagonalize the cross product
    // matrix of the observed data.
    Matrix missing_design_matrix_;
    Vector missing_leverage_;

    // The elements of the response vector corresponding to the rows
    // in missing_design_matrix_.
    Vector missing_y_;

    // The diagonal elements of the posterior information matrix xtx +
    // xtx_missing + prior_information.  The off-diagonal elements are
    // zero by design.  This matrix will be constant once determined
    // by determine_missing_design_matrix().
    //
    // This matrix must be divided by model_->sigsq() to get the
    // posterior Fisher information.
    Vector complete_data_xtx_diagonal_;

    // The elements of the un-normalized posterior mean: xty +
    // xty_missing + prior_information * prior_mean.  The xty_missing
    // portion of this sum will change with each MCMC iteration.
    Vector complete_data_xty_;

    // Transformations of parameters of the prior distribution.

    // The prior precision, unscaled by sigma.  This is the prior
    // precision that would be obtained if sigma == 1.
    mutable Vector unscaled_prior_precision_;

    // unscaled_prior_precision_ * prior_mean
    mutable Vector information_weighted_prior_mean_;

    // A flag indicating that the values of the prior distribution for
    // beta are current.  This flag can only be set to false by a call
    // to observe_changes_in_prior(), which is to be set as an
    // observer in the parameters of the prior distribution.
    mutable bool prior_is_current_;

    // With this probability, ignore everything in this sampler and
    // implement draw() using the more robust SSVS method instead.
    double fallback_probability_;
  };
}  // namespace BOOM

#endif  //  BOOM_SPIKE_SLAB_DA_REGRESSION_SAMPLER_HPP_
