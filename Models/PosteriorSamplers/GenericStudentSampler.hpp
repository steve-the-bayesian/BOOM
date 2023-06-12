/*
  Copyright (C) 2022 Steven L. Scott

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

#ifndef BOOM_GENERIC_STUDENT_SAMPLER_HPP_
#define BOOM_GENERIC_STUDENT_SAMPLER_HPP_

#include "Models/DoubleModel.hpp"
#include "LinAlg/Vector.hpp"
#include "distributions/rng.hpp"
#include "TargetFun/TargetFun.hpp"

namespace BOOM {

  // A class for sampling from the joint distribution of the scale parameter
  // (sigma) and the tail thickness parameter (nu) from a zero-mean scalar
  // student T distribution.

  class ZeroMeanStudentLogLikelihood {
   public:
    explicit ZeroMeanStudentLogLikelihood(const Vector &residuals)
        : residuals_(residuals)
    {}

    double operator()(const Vector &x,
                      Vector *g = nullptr,
                      Matrix *h = nullptr,
                      bool reset_derivatives = true) const;

   private:
    const Vector &residuals_;
  };

  class GenericStudentSampler {
   public:
    // Args:
    //   sigma_prior: Prior distribution on the scale parameter 'sigma'.  Be
    //     sure to include any Jacobian terms if the prior is expressed on
    //     another scale, such as sigma^2 or 1/sigma^2.
    //   nu_prior: Prior distribution on the tail thickness parameter 'nu'.
    explicit GenericStudentSampler(const DoubleModel *sigma_prior,
                                   const DoubleModel *nu_prior)

        : sigma_prior_(sigma_prior),
          nu_prior_(nu_prior)
    {}

    // Draws a value for the residual variance (not standard
    // deviation).  The variance scale is used because most models are
    // parameterized in terms of variances.
    // Args:
    //   rng:  The U(0, 1) random number generator.
    //   sigma:  Current value of the scale parameter.
    //   nu: Current value of the tail-thickness parameter.
    //   residuals:  Vector of zero-mean residuals on which to base the draw.
    // Returns:
    //   A pair.  The first element is the updated draw of sigma.  The second is
    //   the updated draw of nu.
    std::pair<double, double> draw(RNG &rng,
                                   double sigma,
                                   double nu,
                                   const Vector &residuals) const;

   private:
    const DoubleModel *sigma_prior_;
    const DoubleModel *nu_prior_;
  };

}  // namespace BOOM

#endif  //  BOOM_GENERIC_GAUSSIAN_VARIANCE_SAMPLER_HPP_
