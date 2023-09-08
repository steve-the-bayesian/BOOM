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

#include "Models/GP/HierarchicalGpRegressionModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  HierarchicalRegressionData::HierarchicalRegressionData(
      double y, const Vector &x, const std::string &group)
      : RegressionData(y, x),
        original_y_(y),
        group_(group)
  {}

  void HierarchicalRegressionData::adjust_y(double value_to_subtract) {
    set_y(original_y_ - value_to_subtract);
  }

  std::ostream &HierarchicalRegressionData::display(
      std::ostream &out) const {
    out << "Original response: " << original_y_ << "\n"
        << "Adjusted response: " << y() << "\n"
        << "Predictors       : " << x() << "\n"
        << "Group            : " << group_ << "\n";
    return out;
  }

  HierarchicalGpRegressionModel::HierarchicalGpRegressionModel(
      const Ptr<GaussianProcessRegressionModel> &mean_function_model)
      : shared_mean_function_model_(mean_function_model),
        shared_mean_function_param_(
            new GpMeanFunction(shared_mean_function_model_))
  {}

  HierarchicalGpRegressionModel::HierarchicalGpRegressionModel(
      const HierarchicalGpRegressionModel &rhs)
  {
    report_error("copy constructor not implemented.");
  }

  HierarchicalGpRegressionModel::HierarchicalGpRegressionModel(
      HierarchicalGpRegressionModel &&rhs)
  {
    report_error("move constructor not implemented.");
  }

  HierarchicalGpRegressionModel &HierarchicalGpRegressionModel::operator=(
      const HierarchicalGpRegressionModel &rhs) {
    if (&rhs != this) {
      report_error("Assignment operator not implemented.");
    }
    return *this;
  }

  HierarchicalGpRegressionModel &HierarchicalGpRegressionModel::operator=(
      HierarchicalGpRegressionModel &&rhs) {
    if (&rhs != this) {
      report_error("Move assignment operator not implemented.");
    }
    return *this;
  }

  HierarchicalGpRegressionModel * HierarchicalGpRegressionModel::clone() const {
    return new HierarchicalGpRegressionModel(*this);
  }

  void HierarchicalGpRegressionModel::add_model(
      const Ptr<GaussianProcessRegressionModel> &model,
      const std::string &index) {
    std::string idx = index;
    if (index.empty()) {
      std::ostringstream index_maker;
      index_maker << models_.size();
      idx = index_maker.str();
    }
    models_[idx] = model;
    group_names_.push_back(index);
    model->set_params(shared_mean_function_param_,
                      model->kernel_param(),
                      model->sigsq_param());
  }

  void HierarchicalGpRegressionModel::add_data(
      const Ptr<HierarchicalRegressionData> &data_point) {
    auto it = models_.find(data_point->group());
    if (it == models_.end()) {
      std::ostringstream err;
      err << "There is no model associated with the index "
          << data_point->group()
          << " available to receive the supplied data point.\n";
      report_error(err.str());
    }
    Ptr<RegressionData> regression_data(data_point);
    it->second->add_data(regression_data);
    prior()->add_data(regression_data);

    data_store_[it->second.get()].push_back(data_point);
  }

  void HierarchicalGpRegressionModel::add_data(const Ptr<Data> &dp) {
    add_data(dp.dcast<HierarchicalRegressionData>());
  }

  void HierarchicalGpRegressionModel::clear_data() {
    for (auto &it : models_) {
      it.second->clear_data();
    }
    prior()->clear_data();
  }

  void HierarchicalGpRegressionModel::combine_data(const Model &, bool) {
    report_error("combine data is not yet implemented.");
  }

  GaussianProcessRegressionModel *
  HierarchicalGpRegressionModel::prior() {
    return shared_mean_function_model_.get();
  }

  const GaussianProcessRegressionModel *
  HierarchicalGpRegressionModel::prior() const{
    return shared_mean_function_model_.get();
  }

  GaussianProcessRegressionModel *
  HierarchicalGpRegressionModel::data_model(const std::string &index) {
    auto it = models_.find(index);
    if (it == models_.end()) {
      std::ostringstream err;
      err << "There is no model indexed by " << index << ".\n";
      report_error(err.str());
    }
    return it->second.get();
  }

  const GaussianProcessRegressionModel *
  HierarchicalGpRegressionModel::data_model(const std::string &index) const {
    auto it = models_.find(index);
    if (it == models_.end()) {
      std::ostringstream err;
      err << "There is no model indexed by " << index << ".\n";
      report_error(err.str());
    }
    return it->second.get();
  }

  std::vector<Ptr<HierarchicalRegressionData>> &
  HierarchicalGpRegressionModel::data_set(GaussianProcessRegressionModel *model) {
    return data_store_[model];
  }

}  // namespace BOOM
