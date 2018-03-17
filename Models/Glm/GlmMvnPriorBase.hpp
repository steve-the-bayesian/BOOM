// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "Models/Glm/GlmCoefs.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/MvnBase.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/Sufstat.hpp"

/*
  Base class that can serve as a prior model for GlmCoefs.  Might
  depend on a scalar sigma, or not.  Might depend on XTX, which might
  or might not change from one iteration to the next.  Probably
  depends on a scalar 'prior sample size' kappa as well.
 */

namespace BOOM {

  class GlmMvnSuf;

  class GlmMvnPriorBase : public SufstatDataPolicy<GlmCoefs, GlmMvnSuf> {
    // model:  beta | b, V, sigsq, gamma ~ N(b_g, sigsq * V_g/kappa)

   public:
    explicit GlmMvnPriorBase(uint dim);
    virtual double pdf(const Ptr<Data> &, bool logscale) const = 0;
    virtual double pdf(const const Ptr<GlmCoefs> &beta,
                       bool logscale) const = 0;

    virtual double sigsq() const;  // default return is 1.0;
    virtual double kappa() const;  // default return is 1.0;

    virtual Vector mu() const = 0;         // not conditional on gamma
    virtual SpdMatrix siginv() const = 0;  // not conditional on gamma
    // conceptually, siginv is the inverse of sigsq * XTX / kappa
  };

  class GlmMvnSuf : public SufstatDetails<GlmCoefs> {
   public:
    explicit GlmMvnSuf(uint p = 0);
    explicit GlmMvnSuf(const std::vector<Ptr<GlmCoefs> > &d);
    GlmMvnSuf *clone() const override;

    void clear() override;
    void Update(const GlmCoefs &beta) override;

    SpdMatrix center_sumsq(const Vector &b) const;
    const Vector &vnobs() const;   // sum of gamma
    const SpdMatrix &GTG() const;  // sum of gamma gamma^T
    const Matrix &BTG() const;     // sum of beta gamma^T
    uint nobs() const;

   private:
    mutable SpdMatrix bbt_;  // sum of beta beta.transpose()
    mutable SpdMatrix ggt_;  // sum of gamma * gamma^T
    Matrix bgt_;             // sum of beta gamma.transpose()
    Vector vnobs_;           // sum of gamma
    uint nobs_;              // number of observations

    Vector b, gam;
    mutable bool sym_;
    void make_symmetric() const;
  };

}  // namespace BOOM
