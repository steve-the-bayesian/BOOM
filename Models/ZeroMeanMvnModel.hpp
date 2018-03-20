// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_ZERO_MEAN_MVN_MODEL_HPP_
#define BOOM_ZERO_MEAN_MVN_MODEL_HPP_
#include "Models/MvnBase.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"

namespace BOOM {

  class ZeroMeanMvnConjSampler;

  class ZeroMeanMvnModel : public MvnBase,
                           public LoglikeModel,
                           public ParamPolicy_1<SpdParams>,
                           public SufstatDataPolicy<VectorData, MvnSuf>,
                           public PriorPolicy {
   public:
    explicit ZeroMeanMvnModel(int dim);
    ZeroMeanMvnModel *clone() const override;
    const Vector &mu() const override;
    const SpdMatrix &Sigma() const override;
    virtual void set_Sigma(const SpdMatrix &);
    const SpdMatrix &siginv() const override;
    virtual void set_siginv(const SpdMatrix &);
    double ldsi() const override;

    void mle() override;
    double loglike(const Vector &siginv_triangle) const override;
    virtual double pdf(const Ptr<Data> &dp, bool logscale) const;

    Ptr<SpdParams> Sigma_prm();
    const Ptr<SpdParams> Sigma_prm() const;

   private:
    Vector mu_;
  };
}  // namespace BOOM
#endif  // BOOM_ZERO_MEAN_MVN_MODEL_HPP_
