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
#ifndef BOOM_TREGRESSION_SAMPLER_HPP_
#define BOOM_TREGRESSION_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/Glm/PosteriorSamplers/TDataImputer.hpp"
#include "Models/Glm/TRegression.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/ScaledChisqModel.hpp"
#include "Samplers/ScalarSliceSampler.hpp"

namespace BOOM {
  // A posterior sampler for T regression models.  Uses data
  // augmentation to turn the T regression into a Gaussian regression.
  // The DF parameters can be sampled either conditional on the latent
  // data (computationally fast, but slow mixing) or after
  // marginalizing out the latent data.
  class TRegressionSampler : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model whose parameters are to be sampled.
    //   coefficient_prior: Prior distribution on the regression
    //     coefficients.
    //   siginv_prior: Prior distribution for the reciprocal of the
    //     residual variance parameter.
    //   nu_prior: Prior for the tail thickness (or "degrees of
    //     freedom") parameter.  The onus is on the caller to ensure
    //     this distribution supports only the positive real line.
    TRegressionSampler(TRegressionModel *model,
                       const Ptr<MvnBase> &coefficient_prior,
                       const Ptr<GammaModelBase> &siginv_prior,
                       const Ptr<DoubleModel> &nu_prior,
                       RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void impute_latent_data();
    void draw_beta_full_conditional();
    void draw_sigsq_full_conditional();
    void draw_nu_given_complete_data();
    void draw_nu_given_observed_data();

    void set_sigma_upper_limit(double max_sigma);

    const WeightedRegSuf &complete_data_sufficient_statistics() const {
      return complete_data_sufficient_statistics_;
    }

    // Normally the sampler manages the draw of the complete data
    // sufficient statistics in the draw() method by a call to
    // impute_latent_data().  Calling fix_latent_data(true) turns
    // impute_latent_data() into a no-op, so either the
    // complete_data_sufficient_statistics_ stay constant from
    // call-to-call (e.g. for debugging), or else control of them
    // passes to an outside object.
    void fix_latent_data(bool fixed = true);

    // Clears the complete data_sufficient statistics for beta and
    // sigma, and the weight_model for nu.  It is not normally
    // necessary to call this function unless an outside object wants
    // to assume control of the complete_data_sufficient_statistics_.
    void clear_complete_data_sufficient_statistics();

    // Updates the complete_data_sufficient_statistics_ and the
    // weight_model for nu given the specified arguments.  It is not
    // normally necessary to call this function unless an outside
    // object wants to assume control of the
    // complete_data_sufficient_statistics_.
    void update_complete_data_sufficient_statistics(double y, const Vector &x,
                                                    double weight);

   private:
    TRegressionModel *model_;
    Ptr<MvnBase> coefficient_prior_;
    Ptr<GammaModelBase> siginv_prior_;
    Ptr<DoubleModel> nu_prior_;

    // nu_model_ holds a PosteriorSampler that can draw nu conditional
    // on the latent data.  The draw is computationally cheap, because
    // it does not need to loop over the data, but it can mix poorly
    // relative to nu_sampler_, which integrates out the latent data.
    Ptr<ScaledChisqModel> weight_model_;

    WeightedRegSuf complete_data_sufficient_statistics_;
    GenericGaussianVarianceSampler sigsq_sampler_;
    TDataImputer data_imputer_;

    // Draws a value of nu given beta and sigma, but not given the latent data.
    ScalarSliceSampler nu_observed_data_sampler_;

    // Draws a value of nu given beta, sigma, and complete data.  This is mainly
    // kept for education purposes, to show how slow this approach can be.
    ScalarSliceSampler nu_complete_data_sampler_;

    bool latent_data_is_fixed_;
  };


  //===========================================================================

  class CompleteDataStudentRegressionPosteriorSampler
      : public PosteriorSampler {
   public:
    CompleteDataStudentRegressionPosteriorSampler(
        CompleteDataStudentRegressionModel *model,
        const Ptr<MvnBase> &coefficient_prior,
        const Ptr<GammaModelBase> &residual_precision_prior,
        const Ptr<DoubleModel> &tail_thickness_prior,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void draw_beta_full_conditional();
    void draw_sigsq_full_conditional();
    void draw_nu_given_observed_data();

    void set_sigma_upper_limit(double max_sigma) {
      sigsq_sampler_.set_sigma_max(max_sigma);
    }

   private:
    CompleteDataStudentRegressionModel *model_;
    Ptr<MvnBase> coefficient_prior_;
    Ptr<GammaModelBase> residual_precision_prior_;
    Ptr<DoubleModel> tail_thickness_prior_;

    GenericGaussianVarianceSampler sigsq_sampler_;
    ScalarSliceSampler nu_observed_data_sampler_;
  };

}  // namespace BOOM

#endif  //  BOOM_TREGRESSION_SAMPLER_HPP_
