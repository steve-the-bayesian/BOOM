// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#ifndef BOOM_BINOMIAL_REGRESSION_DATA_HPP_
#define BOOM_BINOMIAL_REGRESSION_DATA_HPP_

#include "Models/Glm/Glm.hpp"

namespace BOOM {
  class BinomialRegressionData : public GlmData<DoubleData> {
   public:
    typedef GlmData<DoubleData> Base;

    // Args:
    //   y:  The number of successes, where y >= 0.
    //   n:  The number of trials, where n >= y.
    //   x:  The vector of predictor variables.
    BinomialRegressionData(double y, double n, const Vector &x);

    // Args:
    //   y:  The number of successes, where y >= 0.
    //   n:  The number of trials, where n >= y.
    //   x: The vector of predictor variables.  This constructor
    //     allows the x's to be shared with other objects.
    BinomialRegressionData(double y, double n, const Ptr<VectorData> &x);

    BinomialRegressionData *clone() const override;
    void set_n(double n, bool check = true);
    void set_y(double y, bool check = true);
    void increment(double incremental_y, double incremental_n);
    
    double n() const;
    void check() const;  // throws if n < y
    std::ostream &display(std::ostream &out) const override;

   private:
    // Number of binomial trials.
    double n_;
    // y() is the number of successes.
  };

}  // namespace BOOM
#endif  // BOOM_BINOMIAL_REGRESSION_DATA_HPP_
