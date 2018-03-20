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

#ifndef BOOM_LOGNORMAL_MODEL_HPP_
#define BOOM_LOGNORMAL_MODEL_HPP_

#include "Models/DoubleModel.hpp"
#include "Models/GaussianModelBase.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"

namespace BOOM {

  class LognormalModel : public ParamPolicy_2<UnivParams, UnivParams>,
                         public SufstatDataPolicy<DoubleData, GaussianSuf>,
                         public PriorPolicy,
                         public DiffDoubleModel,
                         public LocationScaleDoubleModel {
   public:
    // Args:
    //   mu:  Mean of the log of the data begin modeled.
    //   sigma:  Standard deviation of the log of the data being modeled.
    explicit LognormalModel(double mu = 0.0, double sigma = 1.0);

    // Args:
    //   mu:  Mean of the log of the data begin modeled.
    //   sigsq: Variance of the log of the data being modeled.  Notice
    //     that this differs from the constructor taking doubles as
    //     arguments.
    LognormalModel(const Ptr<UnivParams> &mu, const Ptr<UnivParams> &sigsq);

    LognormalModel *clone() const override;

    Ptr<UnivParams> Mu_prm();
    Ptr<UnivParams> Sigsq_prm();
    double mu() const;
    double sigsq() const;
    double sigma() const { return sqrt(sigsq()); }

    void set_mu(double mu);
    void set_sigsq(double sigsq);
    void set_sigma(double sigma) { set_sigsq(sigma * sigma); }

    double mean() const override;
    double variance() const override;
    double sd() const { return sqrt(variance()); }

    double Logp(double x, double &d1, double &d2, uint nderiv) const override;
    double sim(RNG &rng = GlobalRng::rng) const override;
    int number_of_observations() const override { return dat().size(); }
  };

}  // namespace BOOM
#endif  //  BOOM_LOGNORMAL_MODEL_HPP_
