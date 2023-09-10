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

#include "Models/GP/PosteriorSamplers/HierarchicalGpPosteriorSampler.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  HierarchicalGpPosteriorSampler::HierarchicalGpPosteriorSampler(
      HierarchicalGpRegressionModel *model, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model)
  {}

  double HierarchicalGpPosteriorSampler::logpri() const {
    return negative_infinity();
  }

  void HierarchicalGpPosteriorSampler::draw() {
    clear_data_adjustments();
    for (const std::string &group_name : model_->group_names()) {
      GaussianProcessRegressionModel *data_model
          = model_->data_model(group_name);
      data_model->sample_posterior();
      adjust_function_values(data_model);
    }

    model_->prior()->sample_posterior();
    clear_data_adjustments();
  }

  void HierarchicalGpPosteriorSampler::adjust_function_values(
      GaussianProcessRegressionModel *specific_model) {

    std::vector<Ptr<HierarchicalRegressionData>> &data(
        model_->data_set(specific_model));

    Matrix predictors(data.size(), specific_model->xdim());
    for (size_t i = 0; i < data.size(); ++i) {
      predictors.row(i) = data[i]->x();
    }

    Ptr<MvnBase> function_distribution = specific_model->predict_distribution(
        predictors, false);
    Vector function_values = function_distribution->sim(rng());
    for (size_t i = 0; i < data.size(); ++i) {
      data[i]->adjust_y(function_values[i]);
    }
  }

  void HierarchicalGpPosteriorSampler::clear_data_adjustments() {
    for (const std::string &group_name : model_->group_names()) {
      // This check runs every iteration, which is wasteful, but it only runs in
      // in debug mode.
      #ifndef NDEBUG
      GaussianProcessRegressionModel *data_model = model_->data_model(group_name);
      #endif
      assert(data_model->dat().size() == model_->data_set(data_model).size());

      std::vector<Ptr<HierarchicalRegressionData>> &data(
          model_->data_set(model_->data_model(group_name)));
      for (Ptr<HierarchicalRegressionData> &data_point : data) {
        data_point->adjust_y(0.0);
      }
    }
  }

}  // namespace BOOM
