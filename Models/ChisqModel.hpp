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
#ifndef BOOM_CHI_SQUARE_MODEL_HPP
#define BOOM_CHI_SQUARE_MODEL_HPP
#include "Models/GammaModel.hpp"

namespace BOOM {

  // Models the scaled chi-square distribution.  Mainly used as a
  // prior distribution for scalar variance parameters.  This version
  // of the chi-square is parameterized by df and sigma.
  // ChisqModel(df, sigma) is the same as GammaModel(df/2, df*sigma^2/2).
  // Its mean is 1.0 / sigma^2.
  class ChisqModel : public GammaModelBase,
                     public ParamPolicy_2<UnivParams, UnivParams>,
                     public PriorPolicy {
   public:
    // Args:
    //   df: The 'degrees of freedom' parameter for the chi-square
    //     distribution.
    //   sigma_est: An estimate of the standard deviation being
    //     modeled.  Note that the argument is sigma, not sigma^2.
    explicit ChisqModel(double df = 1.0, double sigma_estimate = 1.0);
    ChisqModel *clone() const override;

    // Df_prm holds "sample size".
    Ptr<UnivParams> Df_prm();
    // Sigsq_prm holds the reciprocal expected value.
    Ptr<UnivParams> Sigsq_prm();

    double df() const;
    void set_df(double df);

    double sigma() const;
    void set_sigma_estimate(double sigma_estimate);

    double sigsq() const;
    void set_sigsq_estimate(double sigsq_estimate);

    double sum_of_squares() const;

    double alpha() const override;
    double beta() const override;
    double Loglike(const Vector &nu_sigsq, Vector &g, Matrix &h,
                   uint nd) const override;
    void mle() override;
  };
}  // namespace BOOM
#endif  // BOOM_CHI_SQUARE_MODEL_HPP
