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

#ifndef BOOM_SCALED_CHISQ_MODEL_HPP
#define BOOM_SCALED_CHISQ_MODEL_HPP

#include "Models/GammaModel.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"

namespace BOOM {

  //======================================================================
  class ScaledChisqModel : public GammaModelBase,
                           public ParamPolicy_1<UnivParams>,
                           public PriorPolicy {
    // w ~ Gamma(nu/2, nu/2) the factor of 2 needs to be accounted for
    // in log likelihood derivatives

   public:
    explicit ScaledChisqModel(double nu = 30.0);
    ScaledChisqModel(const ScaledChisqModel &rhs);
    ScaledChisqModel *clone() const override;

    Ptr<UnivParams> Nu_prm();
    const Ptr<UnivParams> Nu_prm() const;

    const double &nu() const;
    void set_nu(double);

    double alpha() const override { return nu() / 2; }
    double beta() const override { return nu() / 2; }

    // probability calculations
    double Loglike(const Vector &nu, Vector &g, Matrix &h,
                   uint nd) const override;
    double log_likelihood(double nu) const;
    double log_likelihood() const override { return log_likelihood(nu()); }
    void mle() override { d2LoglikeModel::mle(); }

    int number_of_observations() const override { return dat().size(); }
  };

}  // namespace BOOM
#endif  // BOOM_SCALED_CHISQ_MODEL_HPP
