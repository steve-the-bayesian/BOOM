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

  SpdMatrix KernelParams::operator()(const Matrix &X) const {
    size_t sample_size = X.nrow();
    SpdMatrix ans(sample_size);
    for (int i = 0; i < sample_size; ++i) {
      for (int j = 0; j <= i; ++j) {
        ans(i, j) = (*this)(X.row(i), X.row(j));
        if (j < i) {
          ans(j, i) = ans(i, j);
        }
      }
    }
    return ans;
  }

  //===========================================================================

  RadialBasisFunction::RadialBasisFunction(double scale)
      : scale_(1, scale)  { }

  RadialBasisFunction::RadialBasisFunction(const Vector &scale)
  {
    set_scale(scale);
  }

  RadialBasisFunction * RadialBasisFunction::clone() const {
    return new RadialBasisFunction(*this);
  }

  void RadialBasisFunction::set_scale(const Vector &scale) {
    for (int i = 0; i < scale.size(); ++i) {
      if (scale[i] <= 0) {
        std::ostringstream err;
        err << "Scale parameter for RadialBasisFunction must be positive.  "
            << "Got scale[" << i << "] = " << scale[i]
            << ".";
        report_error(err.str());
      }
    }
    signal();
    scale_ = scale;
  }

  double RadialBasisFunction::operator()(
      const ConstVectorView &x, const ConstVectorView &y) const {
    if (scale_.size() == 1  && x.size() > 1) {
      double scalar = scale_[0];
      scale_.resize(x.size());
      scale_ = scalar;
    }
    Vector delta = (x - y) / scale_;
    double distance = delta.normsq();
    return exp( -2 * distance);
  }

  std::ostream &RadialBasisFunction::display(std::ostream &out) const {
    out << "Radial Basis Function with scale " << scale_;
    return out;
  }

  Vector RadialBasisFunction::vectorize(bool) const {
    return scale_;
  }

  Vector::const_iterator RadialBasisFunction::unvectorize(Vector::const_iterator &v, bool) {
    for (size_t i = 0; i < scale_.size(); ++i) {
      scale_[i] = *v;
      ++v;
    }
    return v;
  }

  Vector::const_iterator RadialBasisFunction::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.cbegin();
    return unvectorize(it, minimal);
  }

  //===========================================================================

  MahalanobisKernel::MahalanobisKernel(int dim, double scale)
      : scale_(scale),
        sample_size_(1.0),
        scaled_shrunk_xtx_inv_(dim, 1.0)
  {}

  MahalanobisKernel::MahalanobisKernel(const Matrix &X, double scale, double diagonal_shrinkage)
      : scale_(1.0),
        sample_size_(X.nrow()),
        scaled_shrunk_xtx_inv_(X.inner())
  {
    if (scaled_shrunk_xtx_inv_.diag().min() <= 0.0) {
      report_error("An all-zero column was passed as part of X.");
    }
    if (!scaled_shrunk_xtx_inv_.all_finite()) {
      report_error("The matrix X contains non-finite values.");
    }
    scaled_shrunk_xtx_inv_ *= scale / sample_size_;
    self_diagonal_average_inplace(scaled_shrunk_xtx_inv_, diagonal_shrinkage);
    scaled_shrunk_xtx_inv_ = scaled_shrunk_xtx_inv_.inv();
  }

  MahalanobisKernel * MahalanobisKernel::clone() const {
    return new MahalanobisKernel(*this);
  }

  uint MahalanobisKernel::size(bool) const {
      return 1;
  }

  void MahalanobisKernel::set_scale(double scale) {
    if (scale <= 0.0) {
      report_error("scale must be positive.");
    }
    if (!std::isfinite(scale)) {
      report_error("scale must be finite.");
    }
    scaled_shrunk_xtx_inv_ *= scale_;  // undo the old scale
    scaled_shrunk_xtx_inv_ /= scale;   // apply the new scale
    scale_ = scale;                    // store the new scale
    signal();
  }

  double MahalanobisKernel::operator()(const ConstVectorView &x1,
                                       const ConstVectorView &x2) const {
    return exp(-.5 * scaled_shrunk_xtx_inv_.Mdist(x1, x2));
  }

  std::ostream & MahalanobisKernel::display(std::ostream &out) const {
    out << "MahalanobisKernel with respect to the matrix: \n"
        << scaled_shrunk_xtx_inv_;
    return out;
  }

  Vector MahalanobisKernel::vectorize(bool) const {
    return Vector(1, scale_);
  }

  Vector::const_iterator MahalanobisKernel::unvectorize(
      Vector::const_iterator &v, bool) {
    scale_ = *v;
    return ++v;
  }

  Vector::const_iterator MahalanobisKernel::unvectorize(
      const Vector &v, bool minimal) {
    Vector::const_iterator b = v.begin();
    return unvectorize(b, minimal);
  }

}  // namespace BOOM
