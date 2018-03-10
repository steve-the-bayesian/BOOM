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

#include "Models/Bart/GaussianBartModel.hpp"
#include "stats/moments.hpp"

namespace BOOM {
  GaussianBartModel::GaussianBartModel(int number_of_trees, double mean)
      : ParamPolicy(new UnivParams(1.0)),
        BartModelBase(number_of_trees, mean) {}

  GaussianBartModel::GaussianBartModel(int number_of_trees, const Vector &y,
                                       const Matrix &x)
      : ParamPolicy(new UnivParams(sd(y))),
        BartModelBase(number_of_trees, mean(y)) {
    for (int i = 0; i < y.size(); ++i) {
      NEW(RegressionData, dp)(y[i], x.row(i));
      add_data(dp);
    }
  }

  GaussianBartModel::GaussianBartModel(const GaussianBartModel &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        BartModelBase(rhs) {}

  GaussianBartModel *GaussianBartModel::clone() const {
    return new GaussianBartModel(*this);
  }

  int GaussianBartModel::sample_size() const { return dat().size(); }

  void GaussianBartModel::add_data(const Ptr<Data> &dp) { add_data(DAT(dp)); }

  void GaussianBartModel::add_data(const Ptr<RegressionData> &data) {
    DataPolicy::add_data(data);
    BartModelBase::observe_data(data->x());
  }

  double GaussianBartModel::sigsq() const { return prm_ref().value(); }

  void GaussianBartModel::set_sigsq(double sigsq) { prm_ref().set(sigsq); }

  Ptr<UnivParams> GaussianBartModel::Sigsq_prm() { return ParamPolicy::prm(); }

}  // namespace BOOM
