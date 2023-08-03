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

#include "Models/GP/GaussianProcessRegressionModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  GaussianProcessRegressionModel::GaussianProcessRegressionModel(
      const Ptr<FunctionParams> &mean_function,
      const Ptr<KernelParams> &kernel)
      : ParamPolicy(mean_function, kernel),
        kernel_matrix_current_(false)
  {
    add_observers();
  }

  GaussianProcessRegressionModel::GaussianProcessRegressionModel(
      const GaussianProcessRegressionModel &rhs)
      : Model(rhs),
        ParamPolicy(Ptr<FunctionParams>(mean_param()->clone()),
                    Ptr<KernelParams>(kernel_param()->clone())),
        DataPolicy(rhs),
        PriorPolicy(rhs)
  {
    add_observers();
  }


  GaussianProcessRegressionModel * GaussianProcessRegressionModel::clone() const {
    return new GaussianProcessRegressionModel(*this);
  }

  double GaussianProcessRegressionModel::predict(const Vector &x) const {
    refresh_kernel_matrix();

    // The vector of kernel values of the new x against the training data.
    const std::vector<Ptr<RegressionData>> & data(dat());
    int nobs = data.size();
    Vector Kstrip(nobs);
    for (size_t i = 0; i < nobs; ++i) {
      Kstrip(i) = kernel(x, data[i]->x());
    }

    return mean_function(x) + Kstrip.dot(Kinv_ * residuals_);
  }

  Ptr<MvnModel> GaussianProcessRegressionModel::predict_distribution(
      const Matrix &X) const {
    refresh_kernel_matrix();

    const std::vector<Ptr<RegressionData>> & data(dat());
    int nobs = data.size();
    int nx = X.nrow();

    Matrix Kstrip(nobs, nx);
    Vector yhat(nx);

    SpdMatrix base_variance(nx);
    for (int i = 0; i < nx; ++i) {
      yhat[i] = mean_function(X.row(i));
      for (int j = 0; j <= i; ++i) {
        base_variance(i, j) = kernel(X.row(i), X.row(j));
        if (j < i) {
          base_variance(j, i) = base_variance(i, j);
        }
      }
    }

    for (int j = 0; j < nx; ++j) {
      for (int i = 0; i < nobs; ++i) {
        Kstrip(i, j) = kernel(data[i]->x(), X.row(j));
      }
    }

    Vector mean = yhat + Kstrip.Tmult(Kinv_ * residuals_);
    SpdMatrix variance = base_variance + Kstrip.Tmult(Kinv_ * Kstrip);
    return new MvnModel(mean, variance);
  }

  void GaussianProcessRegressionModel::add_observers() {
    auto obs = [this]() {this->kernel_matrix_current_ = false;};
    kernel_param()->add_observer(this, obs);
    mean_param()->add_observer(this, obs);
  }


  void GaussianProcessRegressionModel::refresh_kernel_matrix() const {
    if (kernel_matrix_current_) {
      return;
    }

    const std::vector<Ptr<RegressionData>> & data(dat());
    int nobs = data.size();

    residuals_.resize(nobs);
    SpdMatrix K(nobs);

    for (size_t i = 0; i < nobs; ++i) {
      residuals_[i] = data[i]->y() - mean_function(data[i]->x());
      for (size_t j = 0; j <= i; ++j) {
        K(i, j) = kernel(data[i]->x(), data[j]->x());
        if (j < i) {
          K(j, i) = K(i, j);
        }
      }
    }

    Kinv_ = K.inv();
    kernel_matrix_current_ = true;
  }



}  // namespace BOOM
