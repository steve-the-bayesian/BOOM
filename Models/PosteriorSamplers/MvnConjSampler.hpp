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

#ifndef BOOM_MVN_CONJ_SAMPLER_HPP
#define BOOM_MVN_CONJ_SAMPLER_HPP

#include "Models/MvnGivenSigma.hpp"
#include "Models/MvnModel.hpp"
#include "Models/PosteriorSamplers/HierarchicalPosteriorSampler.hpp"
#include "Models/WishartModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"

namespace BOOM {

  namespace NormalInverseWishart {

    // Parameters of the normal inverse Wishart model for (mu, Siginv), where
    // Siginv is the matrix inverse of the variance matrix Sigma.  The model is
    //
    //   (mu | Sigma) ~ N(mu0, Sigma / kappa)
    //         Sigma  ~ W(nu, sum_of_squares)
    //
    // Here mean refers to mu0, mean_sample_size refers to kappa, and
    // variance_sample_size refers to nu.
    class NormalInverseWishartParameters {
     public:
      // Construct the object by storing pointers to the prior models.
      NormalInverseWishartParameters(const MvnGivenSigma *mean_prior,
                                     const WishartModel *precision_prior);

      // Updates the parameters of the Normal inverse Wishart model given data
      // summarized in suf.
      //
      // Args:
      //   suf: Sufficient statistics from observed multivariate normal data.
      //
      // Effects:
      //   Sets the 4 model parameters to their values in the posterior
      //   distribution given the data in suf.
      void compute_mvn_posterior(const MvnSuf &suf);

      const SpdMatrix &sum_of_squares() const { return sum_of_squares_; }
      double variance_sample_size() const { return variance_sample_size_; }
      double mean_sample_size() const { return mean_sample_size_; }
      const Vector &mean() const { return mean_; }

      // Reset model parameters to prior values.
      void reset_to_prior();

     private:
      const MvnGivenSigma *mean_model_;
      const WishartModel *precision_model_;

      SpdMatrix sum_of_squares_;
      double variance_sample_size_;
      double mean_sample_size_;
      Vector mean_;
      Vector workspace_;
    };
  }  // namespace NormalInverseWishart

  class MvnConjSampler : public ConjugateHierarchicalPosteriorSampler {
   public:
    // Args:
    //   model: The model to be sampled. Must be non-NULL.
    //   mean: The prior mean of the mean parameter 'mu' in *model.
    //   kappa: The prior sample size for the 'mu' parameter.  This is the
    //     number of observations worth of weight to be placed on 'mean'
    //     relative to the mean of the data.
    //   Sigma_guess:  The prior mean of the variance parameter Sigma in *model.
    //   prior_df: The prior sample size for the 'Sigma' parameter.  This si the
    //     number of observations worth of weight to be placed on 'Sigma_guess'
    //     relative to the sample variance of the data.
    //   seeding_rng: The random number generator used to set the seed for this
    //     object.
    MvnConjSampler(MvnModel *model,
                   const Vector &mean,
                   double kappa,
                   const SpdMatrix &Sigma_guess,
                   double prior_df,
                   RNG &seeding_rng = GlobalRng::rng);

    // Args:
    //   model: The model to be sampled.  Unlike the other constructor, the
    //     'model' argument here CAN be nullptr if the sampler is going to be
    //     used as part of a hierarchical model.
    //   mu: Prior distribution for the 'mu' parameter in the model being
    //     sampled.
    //   Siginv: Prior distribution for the 'siginv' precision parameter in the
    //     model being sampled.
    //   seeding_rng: The random number generator used to set the seed for this
    //     object.
    MvnConjSampler(MvnModel *model,
                   const Ptr<MvnGivenSigma> &mu,
                   const Ptr<WishartModel> &Siginv,
                   RNG &seeding_rng = GlobalRng::rng);

    MvnConjSampler *clone_to_new_host(Model *new_host) const override;

    void draw() override;
    double logpri() const override;

    void draw_model_parameters(Model &model) override;
    void draw_model_parameters(MvnModel &model);

    double log_prior_density(const ConstVectorView &parameters) const override;
    double log_prior_density(const Model &model) const override;
    double log_prior_density(const MvnModel &model) const;

    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override { return true; }
    double kappa() const;
    double prior_df() const;
    const Vector &mu0() const;
    const SpdMatrix &prior_SS() const;

    double log_marginal_density(const Ptr<Data> &dp,
                                const ConjugateModel *model) const override;

    double log_marginal_density(const VectorData &data_point,
                                const MvnModel *model) const;

   private:
    MvnModel *model_;
    Ptr<MvnGivenSigma> mu_;
    Ptr<WishartModel> siginv_;

    mutable NormalInverseWishart::NormalInverseWishartParameters prior_;
    mutable NormalInverseWishart::NormalInverseWishartParameters posterior_;
  };

}  // namespace BOOM
#endif  // BOOM_MVN_CONJ_SAMPLER_HPP
