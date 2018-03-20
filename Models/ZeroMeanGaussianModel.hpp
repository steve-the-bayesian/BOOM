// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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

#ifndef BOOM_ZERO_MEAN_GAUSSIAN_MODEL_HPP
#define BOOM_ZERO_MEAN_GAUSSIAN_MODEL_HPP
#include "Models/GammaModel.hpp"
#include "Models/GaussianModelBase.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  class ZeroMeanGaussianModel : public GaussianModelBase,
                                public ParamPolicy_1<UnivParams>,
                                public PriorPolicy {
   public:
    explicit ZeroMeanGaussianModel(double sigma = 1.0);
    explicit ZeroMeanGaussianModel(const std::vector<double> &);
    ZeroMeanGaussianModel *clone() const override;

    void set_sigsq(double sigsq);

    Ptr<UnivParams> Sigsq_prm();
    const Ptr<UnivParams> Sigsq_prm() const;

    double mu() const override { return 0; }
    double sigsq() const override;
    double sigma() const override;

    void mle() override;

    double Loglike(const Vector &sigsq_vec, Vector &g, Matrix &h,
                   uint nd) const override;
    double log_likelihood(double sigsq, double *g, double *h) const;
    double log_likelihood() const override {
      return log_likelihood(sigsq(), nullptr, nullptr);
    }
  };
}  // namespace BOOM
#endif  // BOOM_ZERO_MEAN_GAUSSIAN_MODEL_HPP
