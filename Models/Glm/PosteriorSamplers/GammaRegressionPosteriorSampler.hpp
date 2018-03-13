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

#ifndef BOOM_GLM_GAMMA_REGRESSION_POSTERIOR_SAMPLER_HPP_
#define BOOM_GLM_GAMMA_REGRESSION_POSTERIOR_SAMPLER_HPP_

#include "Models/DoubleModel.hpp"
#include "Models/Glm/GammaRegressionModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Samplers/MetropolisHastings.hpp"

namespace BOOM {

  // The GammaRegressionPosteriorSampler approximates the conditional
  // distribution of the regression coefficients and the log of the
  // shape parameter as a normal.  It locates the mode of the
  // posterior and stores the mode so that it can make TIM proposals.
  class GammaRegressionPosteriorSampler : public PosteriorSampler {
   public:
    GammaRegressionPosteriorSampler(
        GammaRegressionModelBase *model, const Ptr<MvnBase> &coefficient_prior,
        const Ptr<DiffDoubleModel> &shape_parameter_prior,
        RNG &seeding_rng = GlobalRng::rng);

    void set_epsilon(double epsilon) { epsilon_ = epsilon; }

    // Set the prior on the shape parameter to the distribution
    // pointed to by the argument.  Calling this function resets the
    // MH sampler to NULL, because it changes the value of the
    // posterior mode.  The sampler will be reconstituted the next
    // time a call is made to draw() or find_posterior_mode().
    void reset_shape_parameter_prior(
        const Ptr<DiffDoubleModel> &shape_parameter_prior);

    void draw() override;
    double logpri() const override;

    bool can_find_posterior_mode() const override { return true; }

    // Calling find_posterior_mode also sets the mh_sampler_.
    void find_posterior_mode(double convergence_epsilon = 1e-5) override;

    // If find_posterior_mode was called successfully, then
    // mh_sampler_ will be defined. Otherwise it will be nullptr.
    bool posterior_mode_found() const { return !!mh_sampler_; }

    // Returns the log posterior and its derivatives with respect to
    // (log alpha, beta).
    double log_posterior(const Vector &log_alpha_beta, Vector &gradient,
                         Matrix &Hessian, uint nderiv) const;

   private:
    GammaRegressionModelBase *model_;
    Ptr<MvnBase> coefficient_prior_;
    Ptr<DiffDoubleModel> shape_parameter_prior_;

    // Convergence threshold for posterior mode finding.
    double epsilon_;

    // Value is nullptr until set by find_posterior_mode.
    Ptr<MetropolisHastings> mh_sampler_;
  };

}  // namespace BOOM

#endif  //  BOOM_GLM_GAMMA_REGRESSION_POSTERIOR_SAMPLER_HPP_
