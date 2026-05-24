/*
  Copyright (C) 2005-2026 Steven L. Scott

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

#include "Bandits/LinearBanditEncoder.hpp"
#include "cpputil/report_error.hpp"
#include <algorithm>

namespace BOOM {

  ArmMap::ArmMap(const ExperimentStructure &xp)
      : xp_(xp)
  {
    FillArmValues_(xp);
  }

  void ArmMap::FillArmValues_(const ExperimentStructure &xp) {
    Configuration arm(xp.nlevels());
    arm_values_.clear();
    while(!arm.done()) {
      arm_values_.push_back(arm.levels());
      arm.next();
    }
  }

  std::vector<std::string> ArmMap::factor_level_names(int arm) const {
    std::vector<std::string> ans;
    for (int i = 0 ; i < xp_.nfactors(); ++i) {
      int level = arm_values_[arm][i];
      ans.push_back(xp_.full_level_name(i, level, ":"));
    }
    return ans;
  }

  //===========================================================================

  ExperimentArmEncoder::ExperimentArmEncoder(
      const std::string &variable_name,
      const Ptr<ArmMap> &arm_map,
      const std::string &baseline_level)
      : EffectsEncoderBase(variable_name),
        arm_map_(arm_map),
        current_level_(-1),
        baseline_level_(baseline_level),
        baseline_level_index_(-1)
  {
    int factor_index = -1;
    for (int i = 0; i < arm_map_->factor_names().size(); ++i) {
      if (arm_map_->factor_names()[i] == variable_name) {
        factor_index = i;
        break;
      }
    }

    if (factor_index == -1) {
      std::ostringstream err;
      err << "Variable named " << variable_name << " not found in arm map.";
      report_error(err.str());
    }

    key_.reset(new CatKey(arm_map_->factor_level_names(factor_index)));

    if (baseline_level_.empty()) {
      baseline_level_ = key_->labels().back();
    }

    // Find the baseline level
    baseline_level_index_ = key_->findstr_or_neg(baseline_level_);
    if (baseline_level_index_ <= 0) {
      key_ = key_->clone();
      key_->add_label(baseline_level);
      baseline_level_index_ = key_->max_levels() - 1;
    }

    
  }

  ExperimentArmEncoder::ExperimentArmEncoder(
      const ExperimentArmEncoder &rhs)
      : EffectsEncoderBase(rhs),
        arm_map_(rhs.arm_map_),
        current_level_(rhs.current_level_)
  {}

  ExperimentArmEncoder & ExperimentArmEncoder::operator=(
      const ExperimentArmEncoder &rhs) {
    if (&rhs != this) {
      EffectsEncoderBase::operator=(rhs);
      arm_map_ = rhs.arm_map_;
      current_level_ = rhs.current_level_;
    }
    return *this;
  }

  ExperimentArmEncoder::ExperimentArmEncoder(
      ExperimentArmEncoder &&rhs)
      : EffectsEncoderBase(std::move(rhs)),
        arm_map_(std::move(rhs.arm_map_)),
        current_level_(rhs.current_level_)
  {}

  ExperimentArmEncoder & ExperimentArmEncoder::operator=(
      ExperimentArmEncoder &&rhs) {
    if (&rhs != this) {
      EffectsEncoderBase::operator=(std::move(rhs));
      arm_map_ = std::move(rhs.arm_map_);
      current_level_ = rhs.current_level_;
    }
    return *this;
  }

  ExperimentArmEncoder * ExperimentArmEncoder::clone() const {
    return new ExperimentArmEncoder(*this);
  }

  int ExperimentArmEncoder::dim() const {
    return key_->max_levels() - 1;
  }

  std::vector<std::string>
  ExperimentArmEncoder::encoded_variable_names() const {
    std::vector<std::string> ans;
    for (int i = 0; i < dim(); ++i) {
      ans.push_back(variable_name() + ":" + key_->label(i));
    }
    return ans;
  }
  
  namespace {
    // Return true iff 'vec' contains 'value'.
    bool contains(const std::vector<std::string> &vec,
                  const std::string &value) {
      return std::find(vec.begin(), vec.end(), value) != vec.end();
    }
  }  // namespace
  
  Matrix ExperimentArmEncoder::encode_dataset(
      const DataTable &table) const {
    if (contains(table.variable_names(), variable_name())) {
      return EffectsEncoderBase::encode_dataset(table);
    } else {
      if (current_level_ < 0) {
        report_error("set_current_experiment_level must be called first.");
      }
      Vector v = EffectsEncoderBase::encode_level(current_level_);
      Matrix ans(table.nrow(), v.size());
      for (int i = 0; i < ans.nrow(); ++i) {
        ans.row(i) = v;
      }
      return ans;
    }
  }

  Vector ExperimentArmEncoder::encode_row(
      const MixedMultivariateData &row) const {
    Vector ans(dim());
    encode_row(row, VectorView(ans));
    return ans;
  }

  void ExperimentArmEncoder::encode_row(
      const MixedMultivariateData &row,
      VectorView view) const {
    if (contains(row.variable_names(), variable_name())) {
      // If the variable is present in the data table, then use the code from
      // EffectsEncoder::encode_row.
      const LabeledCategoricalData &data_point(row.categorical(variable_name()));
      encode_level(key_->findstr(data_point.label()), view);
    } else {
      // If the variable is not present in the data table, encode using
      // current_level_.
      if (current_level_ < 0) {
        report_error("set_current_experiment_level must be called first.");
      }
      EffectsEncoderBase::encode_level(current_level_, view);
    }
  }

  Matrix ExperimentArmEncoder::encode(
      const CategoricalVariable &variable) const {
    Matrix ans(variable.size(), dim());
    for (size_t i = 0; i < variable.size(); ++i) {
      encode_level(variable[i]->value(), ans.row(i));
    }
    return ans;
  }

  int ExperimentArmEncoder::baseline_level() const {
    return baseline_level_index_;
  }

  //===========================================================================

  LinearBanditEncoder::LinearBanditEncoder(
      const Ptr<ArmMap> &arm_map,
      const Ptr<DatasetEncoder> &dataset_encoder)
      : arm_map_(arm_map),
        dataset_encoder_(dataset_encoder)
  {
    ensure_arm_coverage();
  }

  Vector LinearBanditEncoder::encode_row(
      int arm, const MixedMultivariateData &context) {

    const std::vector<int> &levels(arm_map_->integer_factor_levels(arm));

    for (int i = 0; i < levels.size(); ++i) {
      experiment_encoders_[
          arm_map_->factor_names()[i]]->set_current_experiment_level(
              levels[i]);
    }

    return dataset_encoder_->encode_row(context);
  }

  void LinearBanditEncoder::ensure_arm_coverage() {

    std::set<std::string> experiment_variable_names(
        arm_map_->factor_names().begin(),
        arm_map_->factor_names().end());
    
    for (auto &enc : dataset_encoder_->encoders()) {
      Ptr<ExperimentArmEncoder> xp_enc = enc.dcast<ExperimentArmEncoder>();
      if (!!xp_enc) {
        // Found an experiment arm encoder.
        const std::string &vname(xp_enc->variable_name());
        if (experiment_variable_names.find(vname) !=
            experiment_variable_names.end()) {
          experiment_encoders_[vname] = xp_enc;
        } else {
          // The variable name in the experiment arm encoder does not match any
          // of the factor names in arm_map.  This might happen if an encoder
          // appears twice in the encoder list, but that should be an error
          // anyway.
          std::ostringstream err;
          err << "An encoder for a variable named "
              << vname
              << " was found, but that name did not match any of the "
              << "names in the arm map: ";
          for (const std::string &arm_map_name : arm_map_->factor_names()) {
            err << std::endl
                << arm_map_name;
          }
          report_error(err.str());
        } // closes the else branch for variable name not found.
      } 
    }
  }
  
  
}  // namespace BOOM
