// Copyright 2018 Google LLC. All Rights Reserved.
#ifndef BOOM_DENSITY_ESTIMATE_HPP_
#define BOOM_DENSITY_ESTIMATE_HPP_

/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "stats/Mspline.hpp"

namespace BOOM {

  // An empirical density function similar to a kernel density estimate.  The
  // estimator works by modeling the empirical CDF of its data using monotonic
  // splines.  The density is then returned as the derivative of the CDF.
  //
  // NOTE: These density estimates can be kind of rough, and should be viewed as
  // inferior to kernel density estimates.  But they're easy and cheap to
  // compute.
  //
  // TODO(user): KDE's can be done quickly using the FFT, or slowly from
  // the definition.  The current approach could be improved using Lagrange
  // multipliers to prevent the coefficients from being negative.
  class EmpiricalDensity {
   public:
    // Args:
    //   data: The data set from which to build an empirical density function.
    //   knots: The vector of knots to use in the spline model of the CDF.
    EmpiricalDensity(const ConstVectorView &data, const Vector &knots);

    // Args:
    //   data: The data set from which to build an empirical density function.
    //   num_knots: The number of equally spaced knots to use in the spline
    //     model of the CDF.
    explicit EmpiricalDensity(const ConstVectorView &data, int num_knots = 10);

    // The value of the estimated density function at x.
    double operator()(double x) const;

    // Args:
    //   values:  A vector of values where density estimates are desired.
    // Returns:
    //   A vector of the same size as the input argument.  Each spot in the
    //   returned vector contains the estimated density value for its
    //   corresponding input.
    Vector operator()(const Vector &values) const;

   private:
    // Create a vector of equally spaced knots ranging from the smallest to the
    // largest values of the input data.
    Vector create_knots(const ConstVectorView &data, int num_knots) const;

    // The spline modeling the CDF.  The Ispline basis is monotonic.  Its
    // derivative is the Mspline basis.  The approximation fits coefficients to
    // model the ECDF.  The coefficients can then be applied to the Mspline
    // basis (derivative) of the Ispline to give an estimate of the density.
    Ispline spline_;
    Vector coefficients_;
  };

}  // namespace BOOM

#endif  // BOOM_DENSITY_ESTIMATE_HPP_
