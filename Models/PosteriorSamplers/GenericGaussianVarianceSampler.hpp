// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#ifndef BOOM_GENERIC_GAUSSIAN_VARIANCE_SAMPLER_HPP_
#define BOOM_GENERIC_GAUSSIAN_VARIANCE_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // A class for sampling the scalar "residual variance" parameter.
  // This is a utility class, and not a proper posterior sampler,
  // because it is not specific to a particular type of model.
  class GenericGaussianVarianceSampler {
   public:
    // Args:
    //   prior:  A prior distribution for 1 / sigsq.
    explicit GenericGaussianVarianceSampler(const Ptr<GammaModelBase> &prior);

    // Args:
    //   prior:  A prior distribution for 1 / sigsq.
    //   sigma_max: The largest acceptable value for the standard
    //     deviation sigma.
    GenericGaussianVarianceSampler(const Ptr<GammaModelBase> &prior,
                                   double sigma_max);

    // Sets the largest acceptable value for sigma (the standard deviation, not
    // the variance).  This can be infinity() if you want to remove a previously
    // set value.  It can also be 0, if you want to constrain all draws of sigsq
    // to be zero.  If sigma_max is negative an error is reported.
    void set_sigma_max(double sigma_max);
    double sigma_max() const { return sigma_max_; }

    // Draws a value for the residual variance (not standard
    // deviation).  The variance scale is used because most models are
    // parameterized in terms of variances.
    // Args:
    //   rng:  The U(0, 1) random number generator.
    //   data_df: The "Degrees of freedom" supplied by the data.  Do
    //     not include prior degrees of freedom, as these will be
    //     added by this function.
    //   data_ss: The sum of squares supplied by the data.  Do not
    //     include the prior sum of squares, as this will be supplied
    //     by this function.
    //   prior_sigma_guess_scale_factor: A number by which to scale the prior
    //     guess at the standard deviation.  This will usually be 1.0.  However,
    //     this argument allows the sampler to be applied to multiple models
    //     that differ only by a scale factor.  Some hierarchical models and
    //     regression models can take advantage of this argument.
    // Returns:
    //   A draw of the residual variance, which will be <= sigma_max_^2.
    double draw(RNG &rng, double data_df, double data_ss,
                double prior_sigma_guess_scale_factor = 1.0) const;

    // Returns the posterior mode of the residual variance based on
    // the inverse Gamma distribution.  If theta ~ Gamma(a, b), then
    // the mode of 1/theta is b / (a + 1).
    double posterior_mode(double data_df, double data_ss) const;

    // Returns the log of the prior on the scale of sigma^2.
    double log_prior(double sigsq) const;

    void set_prior(const Ptr<GammaModelBase> &new_prior) { prior_ = new_prior; }

    double sigma_prior_guess() const;
    double sigma_prior_sample_size() const;

   private:
    Ptr<GammaModelBase> prior_;
    double sigma_max_;
  };

}  // namespace BOOM

#endif  //  BOOM_GENERIC_GAUSSIAN_VARIANCE_SAMPLER_HPP_
