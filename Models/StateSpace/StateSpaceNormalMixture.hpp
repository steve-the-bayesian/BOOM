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

#ifndef BOOM_STATE_SPACE_NORMAL_MIXTURE_HPP_
#define BOOM_STATE_SPACE_NORMAL_MIXTURE_HPP_

#include "Models/Glm/Glm.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"

namespace BOOM {

  // A base class for code common to all normal mixture state space
  // models.

  class StateSpaceNormalMixture : public ScalarStateSpaceModelBase {
   public:
    // Args:
    //   has_regression: derived classes should set to 'true' of the
    //     model has a regression component.
    explicit StateSpaceNormalMixture(bool has_regression);

    bool has_regression() const override { return has_regression_; }

    void set_regression_flag(bool has_regression) {
      has_regression_ = has_regression;
    }

    const GlmModel *observation_model() const override = 0;
    GlmModel *observation_model() override = 0;

    // The number of observed and missing observations at the specified time
    // point.
    virtual int total_sample_size(int time) const = 0;

    virtual const GlmBaseData &data(int time, int observation) const = 0;

    // If the model has a regression contribution, then the return
    // vector gives the contribution of the regression component at
    // each time point, on the "linear predictor" scale.  I.e. the
    // return value is x * beta at each time point.
    //
    // In the case of multiplexed data, the value at each time point is the
    // average of the regression contributions to each of the sub-observations
    // at that time point.
    Vector regression_contribution() const override;

   private:
    bool has_regression_;
  };

}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_NORMAL_MIXTURE_HPP_
