// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#ifndef BOOM_DIRICHLET_POSTERIOR_SAMPLER_HPP
#define BOOM_DIRICHLET_POSTERIOR_SAMPLER_HPP

#include <memory>
#include "Models/DirichletModel.hpp"
#include "Models/DoubleModel.hpp"
#include "Models/VectorModel.hpp"
#include "Samplers/ScalarLangevinSampler.hpp"
#include "Samplers/ScalarSliceSampler.hpp"
#include "Samplers/TIM.hpp"
#include "Samplers/UnivariateLangevinSampler.hpp"
#include "Samplers/UnivariateSliceSampler.hpp"

#include "TargetFun/TargetFun.hpp"
#include "TargetFun/MultinomialLogitTransform.hpp"

namespace BOOM {

  namespace DirichletSampler {
    // The DirichletPosteriorSampler defers its draws to one or more
    // instances of DirichletSamplerImpl, which codify different
    // sampling strategies.
    class DirichletSamplerImpl {
     public:
      // Args:
      //   model:  The model being managed.
      //   phi_prior: A prior on the unconstrained elements of phi,
      //     the mean of the Dirichlet distribution.  The first
      //     element of phi is written as a function of the others.
      //   alpha_prior: A prior on the "sample_size" parameter of the
      //     Dirichlet distribution.
      DirichletSamplerImpl(DirichletModel *model,
                           const Ptr<VectorModel> &phi_prior,
                           const Ptr<DoubleModel> &alpha_prior,
                           RNG *rng);
      virtual ~DirichletSamplerImpl() {}

      virtual void draw() = 0;

      const DirichletModel *model() const { return model_; }
      const VectorModel *phi_prior() const { return phi_prior_.get(); }
      const DoubleModel *alpha_prior() const { return alpha_prior_.get(); }
      RNG *rng() { return rng_; }

     protected:
      DirichletModel *model() { return model_; }

     private:
      DirichletModel *model_;
      Ptr<VectorModel> phi_prior_;
      Ptr<DoubleModel> alpha_prior_;
      RNG *rng_;
    };
  }  // namespace DirichletSampler.

  //======================================================================
  // A posterior sampler for the Dirichlet distribution.  For the
  // purpose of this sampler, the model is parameterized as alpha *
  // phi, where alpha is a positive scalar, and phi is a discrete
  // probability distribution (i.e. a Vector of non-negative numbers
  // summing to 1).
  class DirichletPosteriorSampler : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model being managed.
    //   phi_prior: A prior for the discrete probability distribution
    //     parameter phi.  This should assign logp = negative_infinity
    //     if phi is not a discrete probability distribution.
    //   alpha_prior: A prior for the "sample size" parameter alpha.
    //     This should assign logp = negative_infinity if alpha <= 0.
    //   seeding_rng:  The random number generator used to seed the sampler.
    DirichletPosteriorSampler(DirichletModel *model,
                              const Ptr<VectorModel> &phi_prior,
                              const Ptr<DoubleModel> &alpha_prior,
                              RNG &seeding_rng = GlobalRng::rng);
    DirichletPosteriorSampler *clone_to_new_host(
        Model *new_host) const override;
    void draw() override;
    double logpri() const override;
    uint dim() const;  // Dimension of model_->nu().

    // Replaces the existing set of methods with this method.
    void set_method(
        const std::shared_ptr<DirichletSampler::DirichletSamplerImpl> &method,
        double weight);

    // Add 'method' to the set of primary sampling methods.  At each
    // iteration methods will be chosen in proportion to their weight.
    void add_method(
        const std::shared_ptr<DirichletSampler::DirichletSamplerImpl> &method,
        double weight);

   private:
    DirichletModel *model_;
    // For the purposes of this sampler, the model is parameterized as
    // nu = alpha * phi, where alpha = sum(nu), and alpha and phi are
    // given independent priors.
    Ptr<VectorModel> phi_prior_;
    Ptr<DoubleModel> alpha_prior_;

    // The draw method will pick an implementation at random, in
    // proportion to weights.
    std::vector<std::shared_ptr<DirichletSampler::DirichletSamplerImpl>>
        sampler_implementations_;
    Vector weights_;

    void draw_impl(
        const std::vector<
            std::shared_ptr<DirichletSampler::DirichletSamplerImpl>> &impl,
        const Vector &weights);
  };

  // Now for some concrete instances of sampling methods that the
  // sampler can mix among.  All of these assume the prior in the
  // DirichletSampler class above, although several assume the priors
  // can also provide derivatives.
  namespace DirichletSampler {

    //----------------------------------------------------------------------
    // A functor representing the log posterior of a single element of
    // the 'nu' vector, conditional on the other elements.
    class DirichletLogp : public ScalarTargetFun {
     public:
      DirichletLogp(uint pos, const Vector &nu, const Vector &sumlogpi,
                    double nobs, const VectorModel *phi_prior,
                    const DoubleModel *alpha_prior, double min_nu = 0);
      double operator()(double nu) const override;

     private:
      double logp() const;
      const Vector &sumlogpi_;
      const double nobs_;
      const uint pos_;
      mutable Vector nu_;
      const double min_nu_;

      const DoubleModel *alpha_prior_;
      const VectorModel *phi_prior_;
    };

    //----------------------------------------------------------------------
    // Computes the un-normalized log posterior of phi given alpha,
    // along with derivatives (if desired), with respect to phi.
    class DirichletPhiLogPosterior : public d2TargetFun {
     public:
      DirichletPhiLogPosterior(DirichletModel *model,
                               const Ptr<DiffVectorModel> &phi_prior);

      // Args:
      //   truncated_phi: The mean parameter for the Dirichlet model,
      //     with the first element removed.
      //   gradient:  Will be filled with the gradient, if desired.
      //   Hessian:  Will be filled with the Hessian if desired.
      //   number_of_desired_derivatives: The number of derivatives to
      //     take.  Either 0, 1, or 2.
      // Returns:
      //   The log density with respect to phi, along with derivatives
      //   in the output arguments, if desired.
      double operator()(const Vector &truncated_phi, Vector &gradient,
                        Matrix &Hessian,
                        uint number_of_desired_derivatives) const override;
      using d2TargetFun::operator();

     private:
      DirichletModel *model_;
      Ptr<DiffVectorModel> phi_prior_;
    };

    //----------------------------------------------------------------------
    // Computes the log posterior of eta (multinomial logit transform
    // of phi) given alpha.  A Jacobian term is needed, which gets
    // kind of complicated.
    class MultinomialLogitLogPosterior : public d2TargetFun,
                                         public dScalarEnabledTargetFun {
     public:
      MultinomialLogitLogPosterior(DirichletModel *model,
                                   const Ptr<DiffVectorModel> &phi_prior);

      // Transformations between the S-vector phi, which is a discrete
      // probability distribution, and its S-1 dimensional multinomial
      // logit tranformation eta.  Eta is log(phi / phi[0]) (with the
      // first element omitted because it is always zero).  The
      // inverse is phi = exp(eta) / sum(exp(eta)), with the leading
      // zero-element added back in.
      Vector to_full_phi(const Vector &eta) const;
      Vector to_eta(const Vector &full_phi) const;

      // This returns log p(eta|.) and associated derivatives.
      // p(eta|dot) is p(to_phi(eta)|.)|Jacobian(phi)|
      double operator()(const Vector &eta, Vector &gradient, Matrix &Hessian,
                        uint number_of_desired_derivatives) const override;
      double operator()(const Vector &x) const override {
        return d2TargetFun::operator()(x);
      }
      double operator()(const Vector &x, Vector &g) const override {
        return d2TargetFun::operator()(x, g);
      }
      double operator()(const Vector &x, Vector &g, Matrix &h) const override {
        return d2TargetFun::operator()(x, g, h);
      }

      double scalar_derivative(const Vector &eta, double &derivative,
                               int position) const override;

     private:
      DirichletModel *model_;
      Ptr<DiffVectorModel> phi_prior_;
    };  // class MultinomialLogitLogPosterior

    // The log posterior distribution of log_alpha (where alpha =
    // sum(model_->nu())), and its derivatives (with respect to log
    // alpha).
    class LogAlphaLogPosterior : public d2ScalarTargetFun {
     public:
      LogAlphaLogPosterior(DirichletModel *model,
                           const Ptr<DiffDoubleModel> &alpha_prior);
      double operator()(double log_alpha, double &d1, double &d2,
                        uint nderiv) const override;
      using d2ScalarTargetFun::operator();

     private:
      DirichletModel *model_;
      Ptr<DiffDoubleModel> alpha_prior_;
    };

    //=======================================================================
    // Concrete classes implementing DirichletSamplerImpl.
    //----------------------------------------------------------------------
    // Draws model parameters using one-variable at a time slice
    // sampling on the untransformed 'nu' scale.
    class NuSliceImpl : public DirichletSamplerImpl {
     public:
      NuSliceImpl(DirichletModel *model, const Ptr<VectorModel> &phi_prior,
                  const Ptr<DoubleModel> &alpha_prior, RNG *rng);
      void draw() override;
    };

    //----------------------------------------------------------------------
    // A sampler for drawing eta = MultinomialLogit(phi) and
    // log(alpha) from their scalar full conditional distributions
    // using slice sampling.
    class MlogitSliceImpl : public DirichletSamplerImpl {
     public:
      MlogitSliceImpl(DirichletModel *model,
                      const Ptr<DiffVectorModel> &phi_prior,
                      const Ptr<DiffDoubleModel> &alpha_prior, RNG *rng);
      void draw() override;

     private:
      MultinomialLogitLogPosterior phi_logpost_;
      UnivariateSliceSampler phi_sampler_;
      LogAlphaLogPosterior alpha_logpost_;
      ScalarSliceSampler log_alpha_sampler_;
    };

    //----------------------------------------------------------------------
    // A sampler that draws eta = MultinomialLogit(phi) using a
    // tailored independence Metropolis algorithm, and log(alpha)
    // using slice sampling.
    class TimImpl : public DirichletSamplerImpl {
     public:
      TimImpl(DirichletModel *model, const Ptr<DiffVectorModel> &phi_prior,
              const Ptr<DiffDoubleModel> &alpha_prior, RNG *rng);
      void draw() override;
      void draw_alpha_given_phi();
      void draw_phi_given_alpha();

     private:
      MultinomialLogitLogPosterior phi_logpost_;
      TIM phi_sampler_;
      LogAlphaLogPosterior alpha_logpost_;
      ScalarSliceSampler log_alpha_sampler_;
    };

    //----------------------------------------------------------------------
    // A Langevin sampler for drawing eta = MultinomialLogit(phi) and
    // log(alpha) from their scalar full conditional distributions.
    class LangevinImpl : public DirichletSamplerImpl {
     public:
      LangevinImpl(DirichletModel *model, const Ptr<DiffVectorModel> &phi_prior,
                   const Ptr<DiffDoubleModel> &alpha_prior, RNG *rng);
      void draw() override;
      void draw_alpha_given_phi();
      void draw_phi_given_alpha();

     private:
      Ptr<MultinomialLogitLogPosterior> phi_logpost_;
      UnivariateLangevinSampler phi_sampler_;
      Ptr<LogAlphaLogPosterior> alpha_logpost_;
      ScalarLangevinSampler log_alpha_sampler_;
    };
  }  // namespace DirichletSampler
}  // namespace BOOM

#endif  // BOOM_DIRICHLET_POSTERIOR_SAMPLER_HPP
