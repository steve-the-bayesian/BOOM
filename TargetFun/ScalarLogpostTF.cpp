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

#include "TargetFun/ScalarLogpostTF.hpp"
#include "Models/DoubleModel.hpp"
#include "TargetFun/Loglike.hpp"

namespace BOOM {
  typedef ScalarLogpostTF SLT;

  SLT::ScalarLogpostTF(LoglikeModel *loglike, const Ptr<DoubleModel> &prior)
      : loglike_(LoglikeTF(loglike)), prior_(prior) {}

  double SLT::operator()(const Vector &x) const {
    double ans = loglike_(x);
    ans += prior_->logp(x[0]);
    return ans;
  }

  double SLT::operator()(double x) const {
    Vector v(1, x);
    double ans = loglike_(v);
    ans += prior_->logp(x);
    return ans;
  }

}  // namespace BOOM
