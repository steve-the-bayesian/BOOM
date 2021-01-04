// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_T_REGRESSION_HPP
#define BOOM_T_REGRESSION_HPP

#include "Models/Glm/Glm.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_3.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  class WeightedRegSuf;

  class TRegressionModel
      : public GlmModel,
        public ParamPolicy_3<GlmCoefs, UnivParams, UnivParams>,
        public IID_DataPolicy<RegressionData>,
        public PriorPolicy,
        public NumOptModel {
   public:
    explicit TRegressionModel(uint p);  // dimension of beta
    TRegressionModel(const Vector &b, double Sigma, double nu = 30);
    TRegressionModel(const Matrix &X, const Vector &y);
    TRegressionModel *clone() const override;

    GlmCoefs &coef() override;
    const GlmCoefs &coef() const override;
    Ptr<GlmCoefs> coef_prm() override;
    const Ptr<GlmCoefs> coef_prm() const override;
    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm() const;
    Ptr<UnivParams> Nu_prm();
    const Ptr<UnivParams> Nu_prm() const;

    // beta() and Beta() inherited from GlmModel;
    const double &sigsq() const;
    double sigma() const;
    void set_sigsq(double s2);

    const double &nu() const;
    void set_nu(double Nu);

    // The argument to Loglike is a vector containing the included
    // regression coefficients, followed by the residual 'dispersion'
    // parameter sigsq, followed by the tail thickness parameter nu.
    double Loglike(const Vector &beta_sigsq_nu, Vector &g, Matrix &h,
                   uint nd) const override;

    // Args:
    //   full_beta: The full set of regression coefficients, including
    //     any that are set to zero.
    //   sigma:  The "residual standard deviation" parameter.
    //   nu:  The tail thickness parameter.
    double log_likelihood(const Vector &full_beta, double sigma,
                          double nu) const;

    double log_likelihood() const override {
      return log_likelihood(Beta(), sigma(), nu());
    }

    // The MLE is computed using an EM algorithm.
    void mle() override;

    double pdf(const Ptr<Data> &dp, bool) const;
    double pdf(const Ptr<DataType> &dp, bool) const;

    Ptr<RegressionData> sim(RNG &rng = GlobalRng::rng) const;
    Ptr<RegressionData> sim(const Vector &X,
                            RNG &rng = GlobalRng::rng) const;

   private:
    // Clear 'suf' and fill it with the expected complete data
    // sufficient statistics.
    void EStep(WeightedRegSuf &suf) const;

    // Take the contents of suf and use it to set model parameters to
    // their MLE's.  Estimate of nu is based on the observed data.
    // Return the observed data log likelihood given the new
    // parameters.
    double MStep(const WeightedRegSuf &suf);
  };

}  // namespace BOOM

#endif  // BOOM_T_REGRESSION_HPP
