/*
  Copyright (C) 2005-2020 Steven L. Scott

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

#include <vector>
#include <algorithm>
#include <cstdint>
#include "cpputil/ToString.hpp"
#include "Models/Glm/LoglinearModel.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using MCD = MultivariateCategoricalData;
  }  // namespace

  MCD::MultivariateCategoricalData(const DataTable &table, int row_number,
                                   double frequency)
      : frequency_(frequency) {
    if (row_number >= table.nrow()) {
      report_error("The requested row does not exist in the data table.");
    }
    for (int i = 0; i < table.nvars(); ++i) {
      if (table.variable_type(i) == VariableType::categorical) {
        data_.push_back(table.get_nominal(row_number, i));
      }
    }
  }

  MCD::MultivariateCategoricalData(const MCD &rhs) {
    operator=(rhs);
  }

  MCD & MCD::operator=(const MCD &rhs) {
    if (&rhs != this) {
      data_ = rhs.data_;
      for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] = data_[i]->clone();
      }
    }
    return *this;
  }

  std::ostream &MCD::display(std::ostream &out) const {
    for (size_t i = 0; i < data_.size(); ++i) {
      out << *(data_[i]);
      if (i + 1 < data_.size()) {
        out << ' ';
      }
    }
    if (frequency_ != 1.0) {
      out << ' ' << frequency_;
    }
    return out;
  }

  std::vector<int> MCD::to_vector() const {
    std::vector<int> ans(data_.size());
    for (int i = 0; i < data_.size(); ++i) {
      ans[i] = data_[i]->value();
    }
    return ans;
  }
  //===========================================================================
  CategoricalMainEffect::CategoricalMainEffect(
      int which_variable, const Ptr<CatKeyBase> &key)
      : encoder_(which_variable, key),
        which_variables_(1, which_variable),
        nlevels_(1, encoder_.dim() + 1)
  {}

  Vector CategoricalMainEffect::encode(const MCD &data) const {
    return encoder_.encode(data[encoder_.which_variable()]);
  }
  Vector CategoricalMainEffect::encode(const std::vector<int> &data) const {
    return encoder_.encode(data[encoder_.which_variable()]);
  }

  //===========================================================================
  CategoricalInteraction::CategoricalInteraction(
      const Ptr<CategoricalDataEncoder> &enc1,
      const Ptr<CategoricalDataEncoder> &enc2)
      : enc1_(enc1),
        enc2_(enc2)
  {
    // Check to make sure that the interaction has no common factors.
    std::vector<int> intersection;
    std::set_intersection(enc1_->which_variables().begin(),
                          enc1_->which_variables().end(),
                          enc2_->which_variables().begin(),
                          enc2_->which_variables().end(),
                          std::back_inserter(intersection));
    if (!intersection.empty()) {
      report_error("The terms in an interaction should not have sub-terms "
                   "in common.");
    }

    // Dump all the 'nlevels' information into a map so that we can look it up
    // after merging the 'which_variables' information.
    std::map<int, int> level_map;
    for (size_t i = 0; i < enc1_->which_variables().size(); ++i) {
      level_map[enc1_->which_variables()[i]] = enc1_->nlevels()[i];
    }
    for (size_t i = 0; i < enc2_->which_variables().size(); ++i) {
      level_map[enc2_->which_variables()[i]] = enc2_->nlevels()[i];
    }

    // Build "which_variables".
    std::merge(enc1_->which_variables().begin(),
               enc1_->which_variables().end(),
               enc2_->which_variables().begin(),
               enc2_->which_variables().end(),
               back_inserter(which_variables_));

    // Build "nlevels".
    for (const auto &el : which_variables_) {
      nlevels_.push_back(level_map[el]);
    }

  }

  namespace {
    template <class DATA>
    Vector encode_interaction(
        const DATA &data,
        int dim,
        const CategoricalDataEncoder &enc1,
        const CategoricalDataEncoder &enc2) {
      Vector v1 = enc1.encode(data);
      Vector v2 = enc2.encode(data);
      Vector ans(dim);
      int index = 0;
      // The order here must match the order in which we iterate through an
      // array.  The 'i' index must change most rapidly.
      for (size_t j = 0; j < v2.size(); ++j) {
        for (size_t i = 0; i < v1.size(); ++i) {
          ans[index++] = v1[i] * v2[j];
        }
      }
      return ans;
    }
  } //

  Vector CategoricalInteraction::encode(
      const std::vector<int> &data) const {
    return encode_interaction(data, dim(), *enc1_, *enc2_);
  }

  Vector CategoricalInteraction::encode(const MCD &data) const {
    return encode_interaction(data, dim(), *enc1_, *enc2_);
  }

  //===========================================================================
  void MultivariateCategoricalEncoder::add_effect(
      const Ptr<CategoricalDataEncoder> &effect) {
    encoders_.push_back(effect);
    encoders_by_index_[effect->which_variables()] = effect;
    effect_position_[effect->which_variables()] = dim_;
    dim_ += effect->dim();
  }

  int MultivariateCategoricalEncoder::effect_position(
      const std::vector<int> &index) const {
    auto it = effect_position_.find(index);
    if (it == effect_position_.end()) {
      std::ostringstream err;
      err << "The requested effect: [" << ToString(index)
          << "] was not found in the encoder.";
      report_error(err.str());
    }
    return it->second;
  }

  namespace {
    template <class DATA>
    Vector encode_variable(
        const DATA &data,
        const std::vector<Ptr<CategoricalDataEncoder>> &encoders,
        int dim,
        bool add_intercept) {
      uint start = 0;
      Vector ans(dim, 0.0);
      if (add_intercept) {
        ans[0] = 1;
        ++start;
      }
      for (int i = 0; i < encoders.size(); ++i) {
        uint encoder_dim = encoders[i]->dim();
        VectorView view(ans, start, encoder_dim);
        view = encoders[i]->encode(data);
        start += encoder_dim;
      }
      return ans;
    }
  }  // namespace

  Vector MultivariateCategoricalEncoder::encode(
      const MCD &data) const {
    return encode_variable(data, encoders_, dim(), add_intercept_);
  }
  Vector MultivariateCategoricalEncoder::encode(
      const std::vector<int> &data) const {
    return encode_variable(data, encoders_, dim(), add_intercept_);
  }

  const CategoricalDataEncoder &MultivariateCategoricalEncoder::encoder(
      int i) const {
    return *encoders_[i];
  }

  const CategoricalDataEncoder &MultivariateCategoricalEncoder::encoder(
      const std::vector<int> &index) const {
    auto it = encoders_by_index_.find(index);
    if (it == encoders_by_index_.end()) {
      std::ostringstream err;
      err << "The requested effect: [" << ToString(index) << "] was not "
          << "found in the MultivariateCategoricalEncoder.";
      report_error(err.str());
    }
    return *it->second;
  }

  //===========================================================================
  std::ostream &LoglinearModelSuf::print(std::ostream &out) const {
    out << "sufficient statistics for a log linear model\n";
    return out;
  }

  Vector LoglinearModelSuf::vectorize(bool minimal) const {
    Vector ans;
    for (const auto &el : cross_tabulations_) {
      ans.concat(Vector(el.second.begin(), el.second.end()));
    }
    return ans;
  }

  Vector::const_iterator LoglinearModelSuf::unvectorize(
      Vector::const_iterator &v, bool) {
    for (auto &el : cross_tabulations_) {
      std::copy(v, v + el.second.size(), el.second.begin());
      v += el.second.size();
    }
    return v;
  }

  Vector::const_iterator LoglinearModelSuf::unvectorize(
      const Vector &v, bool minimal) {
    auto vit = v.cbegin();
    return unvectorize(vit, minimal);
  }

  void LoglinearModelSuf::clear() {
    for (auto &el : cross_tabulations_) {
      cross_tabulations_[el.first] = 0.0;
    }
    sample_size_ = 0;
    valid_ = true;
  }

  void LoglinearModelSuf::clear_data_and_structure() {
    clear();
    effects_.clear();
  }

  void LoglinearModelSuf::refresh(const std::vector<Ptr<MCD>> &data) {
    clear();
    for (const auto &el : data) {
      Update(*el);
    }
  }

  void LoglinearModelSuf::Update(const MCD &data) {
    // Each element in the for loop is a "margin" of the table.  el.first
    // indicates which variables are involved in the margin.  el.second is the
    // cross tabulation.
    if (!valid_) {
      report_error("LoglinearModelSuf::Update called from an invalid state.");
    }
    sample_size_ += data.frequency();
    for (auto &el : cross_tabulations_) {
      std::vector<int> index = el.first;
      // index starts off containing the indices of the variables involved in an
      // effect.  Replace each element of 'index' with the value of the variable
      // at that index.
      for (int j = 0; j < index.size(); ++j) {
        const CategoricalData &variable(data[index[j]]);
        index[j] = variable.value();
      }
      el.second[index] += data.frequency();
    }
  }

  void LoglinearModelSuf::add_effect(
      const Ptr<CategoricalDataEncoder> &effect) {
    effects_.push_back(effect);
    cross_tabulations_[effect->which_variables()] = Array(
        effect->nlevels(), 0.0);
    if (sample_size_ > 0) {
      valid_ = false;
    }
  }

  void LoglinearModelSuf::combine(const LoglinearModelSuf &rhs) {
    for (const auto &el : rhs.cross_tabulations_) {
      cross_tabulations_[el.first] += el.second;
    }
  }

  void LoglinearModelSuf::combine(const Ptr<LoglinearModelSuf> &rhs) {
    combine(*rhs);
  }

  LoglinearModelSuf *LoglinearModelSuf::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  const Array &LoglinearModelSuf::margin(const std::vector<int> &index) const {
    const auto it = cross_tabulations_.find(index);
    if (it == cross_tabulations_.end()) {
      std::ostringstream err;
      err << "Index " << ToString(index) << " not found.";
      report_error(err.str());
    }
    return it->second;
  }

  //===========================================================================

  LoglinearModel::LoglinearModel()
      : ParamPolicy(new GlmCoefs(0)),
        DataPolicy(new LoglinearModelSuf)
  {}

  LoglinearModel::LoglinearModel(const DataTable &table)
      : ParamPolicy(nullptr),
        DataPolicy(new LoglinearModelSuf)
  {
    std::vector<int> categorical_variables;
    for (int j = 0; j < table.nvars(); ++j) {
      if (table.variable_type(j) == VariableType::categorical) {
        categorical_variables.push_back(j);
      }
    }

    if (categorical_variables.empty()) {
      report_error("There were no categorical variables in the data table.");
    }
    for (size_t i = 0; i < table.nrow(); ++i) {
      NEW(MultivariateCategoricalData, data_point)();
      for (size_t j = 0; j < categorical_variables.size(); ++j) {
        data_point->push_back(table.get_nominal(
            categorical_variables[j])[i]);
      }
      add_data(data_point);
    }
  }

  LoglinearModel * LoglinearModel::clone() const {
    return new LoglinearModel(*this);
  }

  void LoglinearModel::add_data(const Ptr<MCD> &data) {
    if (main_effects_.empty()) {
      for (int i = 0; i < data->nvars(); ++i) {
        NEW(CategoricalMainEffect, main_effect)(i, (*data)[i].key());
        add_effect(main_effect);
        main_effects_.push_back(main_effect);
      }
    }
    DataPolicy::add_data(data);
  }

  int LoglinearModel::nvars() const {
    if (dat().empty()) {
      return 0;
    } else {
      return dat()[0]->nvars();
    }
  }

  void LoglinearModel::add_interaction(
      const std::vector<int> &which_variables) {
    std::vector<int> vars(which_variables);
    std::sort(vars.begin(), vars.end());
    if (vars.size() < 2) {
      report_error("An interaction requires at least 2 variables.");
    }
    for (int i = 0; i < vars.size(); ++i) {
      if (vars[i] < 0 || vars[i] >= main_effects_.size()) {
        report_error("Index out of bounds in 'add_interaction'.");
      }
      if (i > 0 && vars[i] == vars[i-1]) {
        report_error("Duplicate index found in 'add_interaction'.");
      }
    }

    NEW(CategoricalInteraction, interaction)(
        main_effects_[vars[0]], main_effects_[vars[1]]);
    for (int i = 2; i < vars.size(); ++i) {
      Ptr<CategoricalDataEncoder> first = interaction;
      interaction = new CategoricalInteraction(
          first, main_effects_[vars[i]]);
    }

    add_effect(interaction);
  }

  void LoglinearModel::refresh_suf() {
    suf()->refresh(dat());
  }

  void LoglinearModel::set_effect_coefficients(
      const Vector &coefficients,
      int effect_index) {
    const CategoricalDataEncoder &enc(encoder_.encoder(effect_index));
    int start = encoder_.effect_position(enc.which_variables());
    if (coefficients.size() != enc.dim()) {
      report_error("Dimension mismatch when setting effect coefficients.");
    }
    prm()->set_subset(coefficients, start);
  }

  void LoglinearModel::initialize_missing_data(RNG &rng) {
    for (auto &observation : dat()) {
      int nvars = observation->nvars();
      for (int i = 0; i < nvars; ++i) {
        Ptr<CategoricalData> variable = observation->mutable_element(i);
        if (variable->missing() != Data::missing_status::observed) {
          int level = rmulti_mt(rng, 0, variable->nlevels() - 1);
          variable->set(level);
        }
      }
    }
  }

  void LoglinearModel::impute_missing_data(RNG &rng) {
    std::vector<int> workspace(nvars());
    for (auto &observation : dat()) {
      for (int i = 0; i < nvars(); ++i) {
        Ptr<CategoricalData> variable = observation->mutable_element(i);
        if (variable->missing() != Data::missing_status::observed) {
          impute_single_variable(*observation, i, rng, workspace);
        }
      }
    }
  }

  void LoglinearModel::impute_single_variable(
      MCD &observation, int position, RNG &rng, std::vector<int> &workspace) {
    Ptr<CategoricalData> variable = observation.mutable_element(position);
    for (int i = 0; i < nvars(); ++i) {
      workspace[i] = observation[i].value();
    }
    int nlevels = variable->nlevels();
    Vector prob(nlevels);
    for (int i = 0; i < nlevels; ++i) {
      workspace[position] = i;
      prob = logp(workspace);
    }
    prob.normalize_logprob();
    int level = rmulti_mt(rng, prob);
    variable->set(level);
  }

  double LoglinearModel::logp(const MultivariateCategoricalData &data) const {
    return coef().predict(encoder_.encode(data));
  }

  double LoglinearModel::logp(const std::vector<int> &data) const {
    return coef().predict(encoder_.encode(data));
  }

  // Private method used to
  void LoglinearModel::add_effect(const Ptr<CategoricalDataEncoder> &effect) {
    encoder_.add_effect(effect);
    suf()->add_effect(effect);
    set_prm(new GlmCoefs(encoder_.dim()));
  }

}
