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

#include "Models/Hierarchical/HierarchicalPoissonModel.hpp"

namespace BOOM {

  HierarchicalPoissonData::HierarchicalPoissonData(double event_count,
                                                   double exposure)
      : event_count_(event_count), exposure_(exposure) {}

  HierarchicalPoissonData *HierarchicalPoissonData::clone() const {
    return new HierarchicalPoissonData(*this);
  }

  std::ostream &HierarchicalPoissonData::display(std::ostream &out) const {
    out << event_count_ << " " << exposure_;
    return out;
  }

  HierarchicalPoissonModel::HierarchicalPoissonModel(
      double lambda_prior_guess, double lambda_prior_sample_size)
      : HierarchicalBase(
            new GammaModel(lambda_prior_sample_size, lambda_prior_guess, 0)) {}

  HierarchicalPoissonModel::HierarchicalPoissonModel(
      const Ptr<GammaModel> &prior)
      : HierarchicalBase(prior) {}

  HierarchicalPoissonModel *HierarchicalPoissonModel::clone() const {
    return new HierarchicalPoissonModel(*this);
  }

  void HierarchicalPoissonModel::add_data(const Ptr<Data> &dp) {
    Ptr<HierarchicalPoissonData> data_point =
        dp.dcast<HierarchicalPoissonData>();
    double events = data_point->event_count();
    double exposure = data_point->exposure();
    double lambda_hat = 1;
    if (exposure > 0 && events > 0) {
      if (events > 0) {
        lambda_hat = events / exposure;
      } else {
        lambda_hat = 1.0 / exposure;
      }
    }
    NEW(PoissonModel, model)(lambda_hat);
    model->suf()->set(events, exposure);
    add_data_level_model(model);
  }

  double HierarchicalPoissonModel::prior_mean() const {
    return prior_model()->mean();
  }

  double HierarchicalPoissonModel::prior_sample_size() const {
    return prior_model()->alpha();
  }

}  // namespace BOOM
