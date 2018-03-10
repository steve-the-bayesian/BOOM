// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_HIERARCHICAL_POISSON_REGRESSION_POSTERIOR_SAMPLER_HPP_
#define BOOM_HIERARCHICAL_POISSON_REGRESSION_POSTERIOR_SAMPLER_HPP_

#include "Models/Glm/HierarchicalPoissonRegression.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonRegressionAuxMixSampler.hpp"
#include "Models/MvnModel.hpp"
#include "Models/PosteriorSamplers/MvnMeanSampler.hpp"
#include "Models/PosteriorSamplers/MvnVarSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnConjSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"
#include "Models/ZeroMeanMvnModel.hpp"

#include "Models/PosteriorSamplers/MvnIndependentVarianceSampler.hpp"
#include "Models/SpdModel.hpp"

namespace BOOM {

  // The model is
  //
  // y[i, j] ~ Poisson(exposure[i, j] * exp(beta[i] * x[i, j]))
  //
  // with beta[i] ~ N(mu, Sigma).  The parameters of the model are mu
  // and Sigma, so a prior distribution on mu and Sigma is needed.
  //
  // The simulation strategy is to sample latent variables u[i,j]
  // given y[i, j] and beta[i], such that the full conditional
  // posterior of beta[i] is that of a regression model.  Then update
  // beta[i] and (mu, Sigma) according to the interweaving sampler of
  // Yu and Meng (JCGS 2011).  That is, draw beta[i], draw (mu,Sigma)
  // given beta, then compute alpha[i] = beta[i] - mu, and re-draw
  // (mu, Sigma) given alpha and the collection of complete data
  // sufficient statistics for each beta[i].
  //
  // Let xtx[i] and xtu[i] denote the complete data sufficient
  // statistics for beta[i].  That is xtx[i] = sum_j
  // x[i,j]x[i,j]^T/v[i,j], and xtu[i] = sum_j
  // x[i,j](u[i,j]-m[i,j])/v[i,j], where m[i,j] and v[i,j] are the
  // normal mean and variance from the mixture approximation to the
  // negative log gamma distribution described by Fhuhwirth-Schnatter,
  // Fruhwirth, Held, and Rue (submitted to JCGS in 2007).
  //
  // The complete data likelihood of mu given alpha and u is normal
  // with sufficient statistics sum_i xtx[i] and
  // sum_i(xtu[i]-xtx[i]*alpha[i]).  The complete data likelihood for
  // Sigma^{-1} is Wishart with sufficient statistic sum_i
  // alpha[i]alpha[i]^T.
  //
  // This base leaves the particular form of the prior on Sigma up to
  // its child classes.
  class HierarchicalPoissonRegressionPosteriorSampler
      : public PosteriorSampler {
   public:
    HierarchicalPoissonRegressionPosteriorSampler(
        HierarchicalPoissonRegressionModel *model, const Ptr<MvnBase> &mu_prior,
        int nthreads = 1, RNG &seeding_rng = GlobalRng::rng);

    void draw() override;

    // impute_latent_data draws complete data sufficient statistics
    // and regression coefficients for each data_model.
    void impute_latent_data();
    void compute_zero_mean_sufficient_statistics();
    void draw_mu_given_zero_mean_sufficient_statistics();
    virtual void draw_mu_and_sigma_given_beta() = 0;
    virtual void draw_sigma_given_zero_mean_sufficient_statistics() = 0;

    // Ensures that each data model in model_ is paired with a sampler
    // in data_model_samplers_.
    void check_data_model_samplers();

   protected:
    ZeroMeanMvnModel *zero_mean_random_effect_model();
    const ZeroMeanMvnModel *zero_mean_random_effect_model() const;
    MvnModel *data_parent_model() { return model_->data_parent_model(); }
    const MvnModel *data_parent_model() const {
      return model_->data_parent_model();
    }
    const MvnBase *mu_prior() const { return mu_prior_.get(); }

   private:
    HierarchicalPoissonRegressionModel *model_;
    std::vector<Ptr<PoissonRegressionAuxMixSampler> > data_model_samplers_;

    Ptr<MvnBase> mu_prior_;
    Ptr<ZeroMeanMvnModel> zero_mean_random_effect_model_;

    int nthreads_;

    // Sufficient statistics for mu given alpha
    SpdMatrix
        xtx_;     // sum of the xtx sufficient statistics for each data model.
    Vector xtu_;  // sum of xtu() for each data model - xtx[i]*alpha[i]
  };

  //----------------------------------------------------------------------
  // This sampler assumes a Wishart prior on the variance of the
  // data_parent_model.
  class HierarchicalPoissonRegressionConjugatePosteriorSampler
      : public HierarchicalPoissonRegressionPosteriorSampler {
   public:
    HierarchicalPoissonRegressionConjugatePosteriorSampler(
        HierarchicalPoissonRegressionModel *model, const Ptr<MvnBase> &mu_prior,
        const Ptr<WishartModel> &siginv_prior, int nthreads = 1);
    double logpri() const override;
    void draw_mu_and_sigma_given_beta() override;
    void draw_sigma_given_zero_mean_sufficient_statistics() override;

   private:
    Ptr<WishartModel> siginv_prior_;

    // Sampler for Sigma in the zero-mean parameterization.
    Ptr<ZeroMeanMvnConjSampler> zero_mean_sigma_sampler_;
    // Samplers for data_parent_model given group-level draws of beta.
    Ptr<MvnVarSampler> sigma_given_beta_sampler_;
    Ptr<MvnMeanSampler> mu_given_beta_sampler_;
  };

  //----------------------------------------------------------------------
  // This sampler assumes that the elements of Sigma^-1 are diagonal,
  // to be modeled independently by a set of truncated gamma
  // distributions.
  //
  // TODO: For wide hierarchies, it might not be prudent
  // to use a full variance matrix for the prior or "data_parent"
  // model.  It may be better to replace the MvnModel in the
  // HierarchicalPoissonRegressionModel with an IndependentMvnModel.
  // That would mean a different HierarchicalPoissonRegressionModel
  // class and a corresponding sampler.
  class HierarchicalPoissonRegressionIndependencePosteriorSampler
      : public HierarchicalPoissonRegressionPosteriorSampler {
   public:
    HierarchicalPoissonRegressionIndependencePosteriorSampler(
        HierarchicalPoissonRegressionModel *model, const Ptr<MvnBase> &mu_prior,
        const std::vector<Ptr<GammaModelBase> > &siginv_priors,
        const Vector &upper_sigma_truncation_point, int nthreads = 1);

    double logpri() const override;

    void draw_mu_and_sigma_given_beta() override;
    void draw_sigma_given_zero_mean_sufficient_statistics() override;

    void set_sigma_upper_limits(const Vector &sigma_upper_limits);

   private:
    std::vector<Ptr<GammaModelBase> > siginv_priors_;

    Ptr<MvnMeanSampler> mu_given_beta_sampler_;
    Ptr<MvnIndependentVarianceSampler> sigma_given_beta_sampler_;

    Ptr<ZeroMeanMvnCompositeIndependenceSampler> zero_mean_sigma_sampler_;
  };

}  // namespace BOOM

#endif  // BOOM_HIERARCHICAL_POISSON_REGRESSION_POSTERIOR_SAMPLER_HPP_
