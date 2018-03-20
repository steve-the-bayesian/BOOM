// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_BRENT_MINIMIZER_HPP_
#define BOOM_BRENT_MINIMIZER_HPP_

#include "numopt.hpp"

namespace BOOM {

  // One dimensional function minimization using Brent's method.
  class BrentMinimizer {
   public:
    explicit BrentMinimizer(const ScalarTarget &target);

    void minimize(double starting_value, double second_candidate);
    void minimize(double starting_value);

    double minimizing_x() const;
    double minimum_value() const;

    void set_tolerance(double tol);

   private:
    ScalarTarget target_;
    double minimizing_x_;
    double minimum_value_;
    double tolerance_;
  };

  class BrentMaximizer {
   public:
    explicit BrentMaximizer(const ScalarTarget &f);
    void maximize(double starting_value);
    void maximize(double starting_value, double second_candidate);

    double maximizing_x() const;
    double maximum_value() const;

    void set_tolerance(double tol);

   private:
    ScalarNegation f_;
    BrentMinimizer minimizer_;
  };

}  // namespace BOOM

#endif  // BOOM_BRENT_MINIMIZER_HPP_
