// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_TRUNCATED_GAMMA_MODEL_HPP
#define BOOM_TRUNCATED_GAMMA_MODEL_HPP

#include "Models/GammaModel.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  // This is not a fully fledged model, because there is no mechanism
  // for inference.
  class TruncatedGammaModel : public GammaModel {
   public:
    TruncatedGammaModel(double a, double b, double lower = 0,
                        double upper = infinity());
    double logp(double x) const override;
    double dlogp(double x, double &derivative) const override;
    double sim(RNG &rng = GlobalRng::rng) const override;

   private:
    double lower_truncation_point_;
    double upper_truncation_point_;
    double plo_;
    double phi_;
    double lognc_;
  };

}  // namespace BOOM
#endif  // BOOM_TRUNCATED_GAMMA_MODEL_HPP
