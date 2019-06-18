// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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
#ifndef BOOM_POISSON_REGRESSION_DATA_HPP_
#define BOOM_POISSON_REGRESSION_DATA_HPP_

#include "Models/Glm/Glm.hpp"

namespace BOOM {
  class PoissonRegressionData : public GlmData<IntData> {
   public:
    // Args:
    //   y:  The number of successes (or number of events).
    //   x:  The vector of predictors.
    //   exposure: The opportunity to generate events.  In some
    //     applications this is an interval of time.  In others it is
    //     a number of trials.
    PoissonRegressionData(int64_t y, const Vector &x, double exposure = 1.0);

    // Args:
    //   y:  The number of successes / events.
    //   x: The vector of predictors.  This constructor allows the x's
    //     to be shared with other objects.
    //   exposure: The opportunity to generate events.  In some
    //     applications this is an interval of time.  In others it is
    //     a number of trials.
    PoissonRegressionData(int64_t y, const Ptr<VectorData> &x, double exposure);

    PoissonRegressionData *clone() const override;
    std::ostream &display(std::ostream &out) const override;
    double exposure() const;
    double log_exposure() const;

    // Sets the value of this observation's exposure.
    // Args:
    //   exposure:  The new exposure value.
    //   signal: If 'true' then any observers who are watching this
    //     data object will be notified of the change.  If false then
    //     observers will not be notified.
    void set_exposure(double exposure, bool signal = true);

   private:
    double exposure_;
    double log_exposure_;
    // saving both exposure and log_exposure keeps us from computing
    // the log of the same thing over and over again in the sampler.
  };
}  // namespace BOOM

#endif  //  BOOM_POISSON_REGRESSION_DATA_HPP_
