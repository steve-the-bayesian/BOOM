// Copyright 2018 Google LLC. All Rights Reserved.
#ifndef BOOM_MODELS_INITIALIZE_DERIVATIVES_HPP_
#define BOOM_MODELS_INITIALIZE_DERIVATIVES_HPP_
/*
  Copyright (C) 2005-2014 Steven L. Scott

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
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {
  // Either resize and initialize derivatives to zero, or else check
  // that they are the proper size, reporting an error if they are not.
  // Args:
  //   gradient:  Either NULL, or the gradient vector.
  //   Hessian:  Either NULL, or the Hessian matrix.
  //   size:  The dimension of that the derivatives should match.
  //   reset: If reset is 'true', then non-NULL derivatives are
  //     resized and set to zero.  If reset is 'false' then the size
  //     of non-NULL derivatives is checked, and an error is reported
  //     (using report_error, which throws an exception by default) if
  //     their dimension fails to match 'dimension'.  In either case,
  //     Hessian is only examined if gradient is non-NULL.
  void initialize_derivatives(Vector *gradient, Matrix *Hessian, int dimension,
                              bool reset);
}  // namespace BOOM
#endif  //  BOOM_MODELS_INITIALIZE_DERIVATIVES_HPP_
