// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#ifndef BOOM_GAUSSIAN_MODEL_GIVEN_SIGMA_HPP
#define BOOM_GAUSSIAN_MODEL_GIVEN_SIGMA_HPP

#include "Models/ModelTypes.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

#include "Models/GaussianModelBase.hpp"

namespace BOOM {

  // A Gaussian model parameterized as N(mu0, sigma^2/kappa), where sigma^2 is
  // owned by another model.  This model is the conjugate prior for the mean of
  // a normal distribution, conditional on its variance (sigma^2).
  class GaussianModelGivenSigma : public GaussianModelBase,
                                  public ParamPolicy_2<UnivParams, UnivParams>,
                                  public PriorPolicy {
   public:
    // Args:
    //   scaling_variance: The 'sigma^2' parameter that scales the variance of
    //     this distribution.  If left as nullptr then 'set_scaling_variance'
    //     will have to be called before this model is used.
    //   mean: The mean of the distribution.  This is 'mu0' in the class
    //     comment.
    //   sample_size: The denominator of the variance of this distribution.
    //     This is 'kappa' in the class comment.
    explicit GaussianModelGivenSigma(
        const Ptr<UnivParams> &scaling_variance = nullptr,
        double mean = 0,
        double sample_size = 1);
    GaussianModelGivenSigma *clone() const override;

    // Sets the parameter in the numerator of the variance of the normal
    // distribution.
    void set_scaling_variance(const Ptr<UnivParams> &scaling_variance);

    void set_params(double mu0, double kappa);
    void set_mu(double mu0);
    void set_kappa(double kappa);

    double mu() const override;

    // Note that sigsq() is the variance of the distribution, as promised by
    // GaussianModelBase.  It is not the scaling variance.
    double sigsq() const override;

    double scaling_variance() const;

    // The sample_size parameter in the denominator of the variance.
    double kappa() const;

    Ptr<UnivParams> Mu_prm();
    Ptr<UnivParams> Kappa_prm();
    const Ptr<UnivParams> Mu_prm() const;
    const Ptr<UnivParams> Kappa_prm() const;

    void mle() override;
    double Loglike(const Vector &mu_kappa, Vector &g, Matrix &h,
                   uint nderiv) const override;
    double log_likelihood() const override {
      return LoglikeModel::log_likelihood();
    }

   private:
    Ptr<UnivParams> scaling_variance_;
  };

}  // namespace BOOM
#endif  // BOOM_GAUSSIAN_MODEL_GIVEN_SIGMA_HPP
