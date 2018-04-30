// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#ifndef BOOM_POSTERIOR_SAMPLERS_HIERARCHICAL_GAUSSIAN_REGRESSION_ASIS_SAMPLER_HPP_
#define BOOM_POSTERIOR_SAMPLERS_HIERARCHICAL_GAUSSIAN_REGRESSION_ASIS_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/Hierarchical/HierarchicalGaussianRegressionModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/MvnVarSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/WishartModel.hpp"

namespace BOOM {

  // An ASIS version of the HierarchicalGaussianRegressionSampler.  (See Yu and
  // Meng 2011
  // http://www.stat.harvard.edu/Faculty_Content/meng/jcgs.2011-article.pdf).
  // The ASIS sampler has better theoretical convergence properties than the
  // classic sampler used by HierarchicalGaussianRegressionSampler.
  class HierGaussianRegressionAsisSampler : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model to be posterior-sampled.
    //   coefficient_mean_hyperprior: The prior distribution of the prior mean
    //     vector (describing the central tendency among regression coefficient
    //     vectors across groups) in *model.
    //   coefficient_precision_hyperprior: The prior distribution of the prior
    //     precision matrix (describing the variation between regression
    //     coefficients vectors across groups) in *model.
    //   residual_precision_prior: Prior distribution on the reciprocal of the
    //     residual variance parameter in *model.  This argument can also be
    //     nullptr, in which case the sampler assumes that the residual variance
    //     parameter will be managed elsewhere.
    //   seeding_rng: The random number generator used to set the seed in this
    //     sampler's RNG.
    HierGaussianRegressionAsisSampler(
        HierarchicalGaussianRegressionModel *model,
        const Ptr<MvnModel> &coefficient_mean_hyperprior,
        const Ptr<WishartModel> &coefficient_precision_hyperprior,
        const Ptr<GammaModelBase> &residual_precision_prior,
        RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

    // Reset the hyperprior models used in the sampler.  These have the same
    // meaning as in the constructor.  The residual_precision_prior can be
    // nullptr if the residual variance is to be either held fixed or managed by
    // another class.
    void set_hyperprior(
        const Ptr<MvnModel> &coefficient_mean_hyperprior,
        const Ptr<WishartModel> &coefficient_precision_hyperprior,
        const Ptr<GammaModelBase> &residual_precision_prior);

   private:
    HierarchicalGaussianRegressionModel *model_;
    Ptr<MvnModel> coefficient_mean_hyperprior_;
    Ptr<WishartModel> coefficient_precision_hyperprior_;
    Ptr<GammaModelBase> residual_variance_prior_;
    GenericGaussianVarianceSampler residual_variance_sampler_;

    // The matrix and vector below are workspace for the draw algorithm.
    // Calling refresh_working_suf() will resize them, set xty_ to zero, and
    // fill xtx_ with the sum of the cross product matrices in the group-level
    // sufficient statistics held by the regression models in *model_.
    void refresh_working_suf();
    SpdMatrix xtx_;
    Vector xty_;
  };

}  // namespace BOOM
#endif  // BOOM_POSTERIOR_SAMPLERS_HIERARCHICAL_GAUSSIAN_REGRESSION_SAMPLER_HPP_
