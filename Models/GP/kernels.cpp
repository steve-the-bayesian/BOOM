/*
  Copyright (C) 2005-2023 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/GP/kernels.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  ZeroFunction * ZeroFunction::clone() const {
    return new ZeroFunction(*this);
  }

  //===========================================================================

  RadialBasisFunction::RadialBasisFunction(double scale)
  {
    set_scale(scale);
  }

  RadialBasisFunction * RadialBasisFunction::clone() const {
    return new RadialBasisFunction(*this);
  }

  void RadialBasisFunction::set_scale(double scale) {
    if (scale <= 0) {
      std::ostringstream err;
      err << "Scale parameter for RadialBasisFunction must be positive.  Got "
          << scale
          << ".";
      report_error(err.str());
    }
    scale_ = scale;
  }

  double RadialBasisFunction::operator()(
      const ConstVectorView &x, const ConstVectorView &y) const {
    double length = (x - y).normsq();
    return exp( -2 * length / (scale_ * scale_));
  }

  std::ostream &RadialBasisFunction::display(std::ostream &out) const {
    out << "Radial Basis Function with scale " << scale_;
    return out;
  }

  Vector RadialBasisFunction::vectorize(bool) const {
    return Vector(1, scale_);
  }

  Vector::const_iterator RadialBasisFunction::unvectorize(Vector::const_iterator &v, bool) {
    scale_ = *v;
    return ++v;
  }

  Vector::const_iterator RadialBasisFunction::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.cbegin();
    return unvectorize(it, minimal);
  }

}  // namespace BOOM
