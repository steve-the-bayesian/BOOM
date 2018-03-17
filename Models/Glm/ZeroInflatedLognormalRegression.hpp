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

#ifndef BOOM_ZERO_INFLATED_LOGNORMAL_REGRESSION_MODEL_HPP_
#define BOOM_ZERO_INFLATED_LOGNORMAL_REGRESSION_MODEL_HPP_

#include "Models/Glm/Glm.hpp"
#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Hierarchical/HierarchicalZeroInflatedGammaModel.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // A model for semicontinuous non-negative data.  A mixture of a
  // point mass at 0 and a regression model where log(y) follows a
  // Gaussian distribution with mean beta.dot(x) and constant variance
  // sigma.  The probability of being nonzero is given by a logistic
  // regression with parameter vector alpha.
  //
  // Let p(x) = 1 / (1 + exp(alpha.dot(-x))) be the probability that y
  // is nonzero, and let mu(x) = beta.dot(x).  The distribution
  // is a mixture:
  //
  //   f(y) = (1-p(x)) * I{0} + p(x) * Lognormal(mu, sigma^2).
  //
  // The mean of this distribution is p(x) * exp(mu(x) + 0.5 *
  // sigsq).
  class ZeroInflatedLognormalRegressionModel
      : public ParamPolicy_3<GlmCoefs, UnivParams, GlmCoefs>,
        public SufstatDataPolicy<RegressionData, RegSuf>,
        public PriorPolicy {
   public:
    // Args:
    //   dimension:  The number of predictor variables.
    //   zero_threshold: A positive number below which observations will be
    //     counted as zero.
    explicit ZeroInflatedLognormalRegressionModel(int dimension,
                                                  double zero_threshold = 1e-5);
    ZeroInflatedLognormalRegressionModel *clone() const override;

    double expected_value(const Vector &x) const;
    double variance(const Vector &x) const;
    double standard_deviation(const Vector &x) const;
    double probability_nonzero(const Vector &x) const;
    double probability_zero(const Vector &x) const;

    // NOTE:
    // The regression portion of the model can use sufficient statistics from
    // the subset of the data containing nonzero responses.  These statistics
    // are not sufficient for the logistic regression portion of the model.
    void add_data(const Ptr<Data> &dp) override;
    void add_data(const Ptr<RegressionData> &dp) override;

    Ptr<GlmCoefs> regression_coefficient_ptr();
    const GlmCoefs &regression_coefficients() const;
    Ptr<UnivParams> sigsq_prm();
    double sigsq() const;
    double sigma() const;
    void set_sigsq(double sigsq);
    Ptr<GlmCoefs> logit_coefficient_ptr();
    const GlmCoefs &logit_coefficients() const;

    // Observations smaller than this number will be treated as zero.
    double zero_threshold() const { return zero_threshold_; }

    double log_likelihood(const Vector &logit_coefficients,
                          const Vector &regression_coefficients,
                          double sigsq) const;
    double sim(const Vector &x, RNG &rng = BOOM::GlobalRng::rng) const;

    HierarchicalZeroInflatedGammaData simulate_sufficient_statistics(
        const Vector &x, int64_t n, RNG &rng = BOOM::GlobalRng::rng) const;

   private:
    double zero_threshold_;
  };

}  // namespace BOOM

#endif  //  BOOM_ZERO_INFLATED_LOGNORMAL_REGRESSION_MODEL_HPP_
