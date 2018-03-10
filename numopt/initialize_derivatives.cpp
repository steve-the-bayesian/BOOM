// Copyright 2018 Google LLC. All Rights Reserved.
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

#include "numopt/initialize_derivatives.hpp"
#include <sstream>
#include "cpputil/report_error.hpp"

namespace BOOM {

  void initialize_derivatives(Vector *gradient, Matrix *Hessian, int dimension,
                              bool reset) {
    if (reset) {
      if (gradient) {
        gradient->resize(dimension);
        *gradient = 0;
        if (Hessian) {
          Hessian->resize(dimension, dimension);
          *Hessian = 0;
        }
      }
    } else {
      if (gradient && gradient->size() != dimension) {
        std::ostringstream err;
        err << "Error:  gradient->size() == " << gradient->size()
            << " but there are " << dimension << " variables." << std::endl;
        report_error(err.str());
      }
      if (gradient && Hessian &&
          (Hessian->nrow() != dimension || Hessian->ncol() != dimension)) {
        std::ostringstream err;
        err << "Hessian dimensions are [" << Hessian->nrow() << " x "
            << Hessian->ncol() << "] but there are " << dimension
            << " variables." << std::endl;
        report_error(err.str());
      }
    }
  }

}  // namespace BOOM
