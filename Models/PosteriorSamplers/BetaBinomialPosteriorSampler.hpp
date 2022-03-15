// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_BETA_BINOMIAL_POSTERIOR_SAMPLER_HPP_
#define BOOM_BETA_BINOMIAL_POSTERIOR_SAMPLER_HPP_

#include "Models/BetaBinomialModel.hpp"
#include "Models/BetaModel.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Samplers/MH_Proposals.hpp"
#include "Samplers/MetropolisHastings.hpp"
#include "Samplers/ScalarSliceSampler.hpp"
#include "TargetFun/Transformation.hpp"

namespace BOOM {

  //======================================================================
  // The log posterior of the beta binomial model on the (prob,
  // sample_size) scale.
  class BetaBinomialLogPosterior {
   public:
    BetaBinomialLogPosterior(const BetaBinomialModel *model,
                             const Ptr<BetaModel> &probability_prior,
                             const Ptr<DiffDoubleModel> &sample_size_prior);
    double operator()(const Vector &prob_samplesize, Vector &gradient,
                      Matrix &Hessian, uint nderiv) const;

   private:
    const BetaBinomialModel *model_;
    const Ptr<BetaModel> probability_prior_;
    const Ptr<DiffDoubleModel> sample_size_prior_;
  };

  // A Jacobian class for the transformation (a, b) -> (prob,
  // sample_size) where prob = a/(a+b) and sample_size = a+b.
  //
  //  a = prob * sample_size,  b = (1-prob) * sample_size
  //  J = |  da / dprob         db / dprob        |
  //      |  da / dsample_size  db / dsample_size |
  //
  //     = | sample_size   -sample_size|
  //       | prob          1-prob      |
  //
  //  |J| = sample_size * (1 - prob) + sample_size * prob
  //      = sample_size
  class ProbSamplesizeJacobian : public Jacobian {
   public:
    // Default constructor says to prefer the new (prob, sample_size)
    // parameterization.
    ProbSamplesizeJacobian();

    // Args:
    //   ab: A two-element vector containing the parameters of the
    //     beta distribution, with the prior success count 'a' in the
    //     first position and the prior failure count 'b' in the
    //     second.
    void evaluate_original_parameterization(const Vector &ab) override;

    // Args:
    //   prob_size: A two-element vector with prob in the first
    //     position and sample_size in the second.
    void evaluate_new_parameterization(const Vector &prob_size) override;

    double logdet() override { return log(sample_size_); }

    Matrix &matrix() override;

    // second_order_element(r,s,t) is the derivative with respect to
    // new_parameterization[r] of the (s,t) element of the Jacobian
    // matrix.
    //
    // If r==s then the derivative is a second derivative with respect
    // to the same variable, and thus zero.  If r != s the derivative
    // is a cross derivative.  If it is a derivative of a (t==0) then
    // the cross derivative is 1, otherwise it is a cross derivative
    // of b, and thus -1.
    double second_order_element(int r, int s, int t) override {
      if (r == s) return 0;
      return t == 0 ? 1 : -1;
    }

    // This override takes advantage of the sparsity in
    // second_order_element.  It omits calls where the elements are
    // known to be zero.
    void transform_second_order_gradient(
        SpdMatrix &working_hessian, const Vector &original_gradient) override;

    // Derivatives of logdet() with respect to prob are all zero, so
    // the gradient and Hessian are easy.
    void add_logdet_gradient(Vector &gradient) override;
    void add_logdet_Hessian(Matrix &Hessian) override;

   private:
    double prob_;
    double sample_size_;
    Matrix matrix_;
    bool matrix_is_current_;
  };

  //======================================================================
  // The Jacobian for the transformation of (prob, size) to
  // (logit(prob), log(sample_size)), which is more likely to be
  // approximately Gaussian.
  //
  // Let eta = logit(prob), and nu = log(sample_size).  The inverse
  // transsformation is
  // prob = exp(eta) / (1 + exp(eta))
  // sample_size = exp(nu)
  //
  // The Jacobian matrix is
  //
  //    J = | d_prob / d_eta   d_size / d_eta |
  //        | d_prob / d_nu    d_size / d_nu  |
  //
  //      = | prob * (1-prob)    0  |
  //        |   0              size |
  //
  //
  class LogitLogJacobian : public Jacobian {
   public:
    LogitLogJacobian();
    void evaluate_original_parameterization(const Vector &prob_size) override;
    void evaluate_new_parameterization(const Vector &eta_nu) override;
    double logdet() override;

    Matrix &matrix() override;

    double second_order_element(int r, int s, int t) override {
      // Becaue the Jacobian matrix is diagonal, if s !=t the Jacobian
      // matrix element is zero, so its derivative is zero too.  If r
      // != s then the derivative is a cross derivative, and so the
      // answer is zero.
      if (r == 0 && s == 0 && t == 0) {
        // Derivative of prob * (1 - prob) with respect to eta.  Using
        // the chain rule to get the derivative, where d_prob / d_eta
        // is the 0,0 element of the Jacobian matrix.
        return (1 - 2 * prob_) * prob_ * (1 - prob_);
      } else if (r == 1 && s == 1 && t == 1) {
        // Derivative of size = exp(nu) with respect to nu.
        return sample_size_;
      } else
        return 0;
    }

    // Overrides the default implementation because only two calls to
    // second_order_element are needed.
    void transform_second_order_gradient(
        SpdMatrix &working_hessian, const Vector &original_gradient) override;

    void add_logdet_gradient(Vector &gradient) override;
    void add_logdet_Hessian(Matrix &hessian) override;

   private:
    double prob_;
    double sample_size_;
    bool matrix_is_current_;
    Matrix matrix_;
  };

  //======================================================================
  // This is a posterior sampler for the BetaBinomialModel.  It
  // differs from the BetaBinomialSampler, which is a sampler for the
  // binomial model based on a beta prior.
  class BetaBinomialPosteriorSampler : public PosteriorSampler {
   public:
    enum SamplingMethod { SLICE, DATA_AUGMENTATION, TIM };
    BetaBinomialPosteriorSampler(BetaBinomialModel *model,
                                 const Ptr<BetaModel> &probability_prior,
                                 const Ptr<DiffDoubleModel> &sample_size_prior,
                                 RNG &seeding_rng = GlobalRng::rng);

    BetaBinomialPosteriorSampler * clone_to_new_host(
        Model *new_host) const override;

    void draw() override;
    double logpri() const override;

    // Set model parameters to the posterior mode.  The mode will be
    // found on the scale (logit(a/(a+b)), log(a+b)).  These results
    // will be stored in the MetropolisHastings proposal distribution.
    void find_posterior_mode(double epsilon = 1e-5) override;

    bool can_find_posterior_mode() const override { return true; }

    void draw_slice();
    void draw_data_augmentation();
    void draw_tim();

    // Full conditional distributions of the probability and sample
    // size parameters.
    double logp(double prob, double sample_size) const;
    double logp_prob(double prob) const;
    double logp_sample_size(double sample_size) const;

    // Determines which sampling method will be used when draw() is
    // called.
    void set_sampling_method(SamplingMethod method) {
      sampling_method_ = method;
    }

    // Set the prior distribution on the "sample_size" model parameter
    // (a + b).
    void set_prior_on_sample_size(
        const Ptr<DiffDoubleModel> &sample_size_prior);

    // Returns a functor for evaluating logp and derivatives on
    // the probability/sample_size scale.
    BetaBinomialLogPosterior prob_sample_size_log_posterior();

    // Returns a functor for evaluating logp and derivatives on the
    // scale of logit(prob) and log(sample_size).
    Transformation approximately_gaussian_log_posterior();

    void observe_new_data() { trouble_locating_mode_ = false; }

   private:
    BetaBinomialModel *model_;
    Ptr<BetaModel> probability_prior_;
    Ptr<DiffDoubleModel> sample_size_prior_;

    ScalarSliceSampler probability_sampler_;
    ScalarSliceSampler sample_size_sampler_;

    SamplingMethod sampling_method_;
    BetaSuf complete_data_suf_;

    // These start off as nullptr. They're set the first time there's
    // a call to draw_tim().
    void allocate_tim_sampler();
    Ptr<MvtIndepProposal> tim_proposal_distribution_;
    Ptr<MetropolisHastings> tim_sampler_;

    // If the sampler has trouble locating the mode, this flag will be
    // set to true, so that calls to draw_tim() do not repeatedly
    // waste time finding the mode and rediscovering the same error.
    bool trouble_locating_mode_;
  };

}  // namespace BOOM
#endif  //  BOOM_BETA_BINOMIAL_POSTERIOR_SAMPLER_HPP_
