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

#include "Models/GP/GpMeanFunction.hpp"
#include "Models/GP/GaussianProcessRegressionModel.hpp"

namespace BOOM {

    Vector FunctionParams::operator()(const Matrix &X) const {
    size_t sample_size = X.nrow();
    Vector ans(sample_size);
    for (size_t i = 0; i < sample_size; ++i) {
      ans[i] = (*this)(X.row(i));
    }
    return ans;
  }

  //===========================================================================

  ZeroFunction * ZeroFunction::clone() const {
    return new ZeroFunction(*this);
  }

  //===========================================================================

  LinearMeanFunction::LinearMeanFunction(const Ptr<GlmCoefs> &coef)
      : coefficients_(coef)
  {}

  LinearMeanFunction::LinearMeanFunction(const LinearMeanFunction &rhs)
      : coefficients_(rhs.coefficients_->clone())
  {}

  LinearMeanFunction & LinearMeanFunction::operator=(const LinearMeanFunction &rhs) {
    if (&rhs != this) {
      coefficients_.reset(rhs.coefficients_->clone());
    }
    return *this;
  }

  LinearMeanFunction * LinearMeanFunction::clone() const {
    return new LinearMeanFunction(*this);
  }

  uint LinearMeanFunction::size(bool minimal) const {
    return coefficients_->size(minimal);
  }

  double LinearMeanFunction::operator()(const ConstVectorView &x) const {
    return coefficients_->predict(x);
  }

  std::ostream &LinearMeanFunction::display(std::ostream &out) const {
    out << "LinearMeanFunction with coefficients: " << *coefficients_;
    return out;
  }

  Vector LinearMeanFunction::vectorize(bool minimal) const {
    return coefficients_->vectorize(minimal);
  }

  Vector::const_iterator LinearMeanFunction::unvectorize(
      Vector::const_iterator &v, bool minimal) {
    return coefficients_->unvectorize(v, minimal);
  }

  //===========================================================================
  GpMeanFunction::GpMeanFunction(const Ptr<GaussianProcessRegressionModel> &gp)
      : gp_(gp)
  {}

  GpMeanFunction::GpMeanFunction(const GpMeanFunction &rhs)
      : gp_(rhs.gp_->clone())
  {}

  GpMeanFunction & GpMeanFunction::operator=(const GpMeanFunction &rhs) {
    if (&rhs != this) {
      gp_.reset(rhs.gp_->clone());
    }
    return *this;
  }

  GpMeanFunction * GpMeanFunction::clone() const {
    return new GpMeanFunction(*this);
  }

  double GpMeanFunction::operator()(const ConstVectorView &x) const {
    return gp_->predict(x);
  }

  uint GpMeanFunction::size(bool minimal) const {
    return gp_->mean_param()->size(minimal) + gp_->kernel_param()->size() + 1;
  }

  std::ostream &GpMeanFunction::display(std::ostream &out) const {
    out << "GpMeanFunction with prior mean function: \n"
        << *gp_->mean_param() << "\n"
        << "kernel: \n"
        << *gp_->kernel_param() << "\n"
        << "and residual SD: "
        << gp_->residual_sd() << "\n";
    return out;
  }

  Vector GpMeanFunction::vectorize(bool minimal) const {
    return gp_->vectorize_params(minimal);
  }

  Vector::const_iterator GpMeanFunction::unvectorize(
      Vector::const_iterator &v, bool minimal) {
    v = gp_->mean_param()->unvectorize(v, minimal);
    v = gp_->kernel_param()->unvectorize(v, minimal);
    v = gp_->sigsq_param()->unvectorize(v, minimal);
    return v;
  }

}  // namespace BOOM
