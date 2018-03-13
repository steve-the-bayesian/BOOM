// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/Bart/GaussianLinearBartModel.hpp"

namespace BOOM {

  GaussianLinearBartModel::GaussianLinearBartModel(int number_of_trees,
                                                   int xdim)
      : regression_(new RegressionModel(xdim)),
        bart_(new GaussianBartModel(number_of_trees, 0.0)) {
    Init();
  }

  GaussianLinearBartModel::GaussianLinearBartModel(int number_of_trees,
                                                   const Vector &y,
                                                   const Matrix &x)
      : regression_(new RegressionModel(ncol(x))),
        bart_(new GaussianBartModel(number_of_trees, 0.0)) {
    if (y.size() != x.nrow()) {
      ostringstream err;
      err << "Error in GaussianLinearBartModel constructor.  "
          << "The number of rows in the predictor matrix (" << x.nrow()
          << ") did not match the number of elements in the "
          << "response vector (" << y.size() << ")." << endl;
      report_error(err.str());
    }
    regression_->only_keep_sufstats(true);
    regression_->use_normal_equations();
    for (int i = 0; i < y.size(); ++i) {
      Ptr<RegressionData> dp(new RegressionData(y[i], x.row(i)));
      this->add_data(dp);
    }
    Init();
  }

  GaussianLinearBartModel::GaussianLinearBartModel(
      const GaussianLinearBartModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        regression_(rhs.regression_->clone()),
        bart_(rhs.bart_->clone()) {
    Init();
  }

  GaussianLinearBartModel *GaussianLinearBartModel::clone() const {
    return new GaussianLinearBartModel(*this);
  }

  void GaussianLinearBartModel::add_data(const Ptr<Data> &dp) {
    this->add_data(dp.dcast<RegressionData>());
  }

  void GaussianLinearBartModel::add_data(const Ptr<RegressionData> &dp) {
    DataPolicy::add_data(dp);
    regression_->add_data(dp);
    bart_->add_data(dp);
  }

  double GaussianLinearBartModel::predict(const Vector &x) const {
    return predict(ConstVectorView(x));
  }

  double GaussianLinearBartModel::predict(const VectorView &x) const {
    return predict(ConstVectorView(x));
  }

  double GaussianLinearBartModel::predict(const ConstVectorView &x) const {
    return regression_->predict(x) + bart_->predict(x);
  }

  RegressionModel *GaussianLinearBartModel::regression() {
    return regression_.get();
  }

  const RegressionModel *GaussianLinearBartModel::regression() const {
    return regression_.get();
  }

  GaussianBartModel *GaussianLinearBartModel::bart() { return bart_.get(); }

  const GaussianBartModel *GaussianLinearBartModel::bart() const {
    return bart_.get();
  }

  void GaussianLinearBartModel::Init() {
    ParamPolicy::add_model(regression_);
    ParamPolicy::add_model(bart_);
  }

}  // namespace BOOM
