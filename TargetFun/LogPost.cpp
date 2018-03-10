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

#include "TargetFun/LogPost.hpp"
#include "Models/VectorModel.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  LogPostTF::LogPostTF(const Target &loglike, const Ptr<VectorModel> &prior)
      : loglike_(loglike), prior_(prior) {}

  //--------------------------------------------------
  dLogPostTF::dLogPostTF(const dLoglikeTF &loglike,
                         const Ptr<dVectorModel> &prior)
      : LogPostTF(loglike, prior), dloglike_(loglike), dprior_(prior) {}

  dLogPostTF::dLogPostTF(const Target &loglike, const dTarget &dloglike,
                         const Ptr<dVectorModel> &prior)
      : LogPostTF(loglike, prior), dloglike_(dloglike), dprior_(prior) {}

  //--------------------------------------------------
  d2LogPostTF::d2LogPostTF(const d2LoglikeTF &loglike,
                           const Ptr<d2VectorModel> &prior)
      : dLogPostTF(loglike, prior), d2loglike_(loglike), d2prior_(prior) {}

  d2LogPostTF::d2LogPostTF(const Target &loglike, const dTarget &dloglike,
                           const d2Target &d2loglike,
                           const Ptr<d2VectorModel> &prior)
      : dLogPostTF(loglike, dloglike, prior),
        d2loglike_(d2loglike),
        d2prior_(prior) {}

  //--------------------------------------------------

  double LogPostTF::operator()(const Vector &z) const {
    double ans = prior_->logp(z);
    if (ans == BOOM::negative_infinity()) {
      return ans;
    }
    ans += loglike_(z);
    return ans;
  }

  //----------------------------------------------------------------------

  double dLogPostTF::operator()(const Vector &x, Vector &g) const {
    g = 0.0;
    Vector g1 = g;
    double ans = dloglike_(x, g);
    ans += dprior_->dlogp(x, g1);
    g += g1;
    return ans;
  }

  //----------------------------------------------------------------------
  double d2LogPostTF::operator()(const Vector &x, Vector &g, Matrix &h) const {
    g = 0.0;
    Vector g1 = g;
    h = 0.0;
    Matrix h1 = h;
    double ans = d2loglike_(x, g, h);    // derivatives wrt x
    ans += d2prior_->d2logp(x, g1, h1);  // derivatives wrt x
    g += g1;
    h += h1;
    return ans;
  }

}  // namespace BOOM
