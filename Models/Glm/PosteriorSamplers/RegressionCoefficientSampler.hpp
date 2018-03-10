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

#ifndef BOOM_REGRESSION_COEFFICIENT_SAMPLER_HPP_
#define BOOM_REGRESSION_COEFFICIENT_SAMPLER_HPP_

#include "Models/Glm/RegressionModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // Simulate regression coefficients from their full conditional distribution
  // given the data and the residual variance parameter.  For a sampler that
  // simulates both parameters, see RegressionConjSampler.
  class RegressionCoefficientSampler : public PosteriorSampler {
   public:
    RegressionCoefficientSampler(RegressionModel *model,
                                 const Ptr<MvnBase> &prior,
                                 RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

    // Simulate the vector of regression coefficients from their posterior
    // distribution given the sufficient statistics in 'model,' the residual
    // variance parameter in 'model', and the specified prior distribution.
    // Store the simulated value in 'model'.
    static void sample_regression_coefficients(RNG &rng, RegressionModel *model,
                                               const MvnBase &prior);

    // Simulate the vector of regression coefficients from their posterior
    // distribution given the directly supplied sufficient statistics, the
    // residual variance, and the specified prior distribution.
    //
    // Args:
    //   rng:  A U[0, 1) random number generator.
    //   xtx:  The cross product matrix from the regression.
    //   xty:  The cross product of the predictor matrix with the response
    //   vector. sigsq:  The residual variance from the regression model. prior:
    //   The prior distribution for the regression coefficients.
    //
    // Returns:
    //   A draw of the regression coefficients from their posterior
    //   distribution.
    static Vector sample_regression_coefficients(RNG &rng, const SpdMatrix &xtx,
                                                 const Vector &xty,
                                                 double sigsq,
                                                 const MvnBase &prior);

   private:
    RegressionModel *model_;
    Ptr<MvnBase> prior_;
  };

}  // namespace BOOM

#endif  //  BOOM_REGRESSION_COEFFICIENT_SAMPLER_HPP_
