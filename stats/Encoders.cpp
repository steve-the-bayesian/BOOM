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

#include "stats/Encoders.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  IntEffectsEncoder::IntEffectsEncoder(const IntEffectsEncoder &rhs)
      : EffectsEncoderBase(rhs),
        key_(rhs.key_->clone())
  {}

  IntEffectsEncoder & IntEffectsEncoder::operator=(
      const IntEffectsEncoder &rhs) {
    if (&rhs != this) {
      EffectsEncoderBase::operator=(rhs);
      key_.reset(rhs.key_->clone());
    }
    return *this;
  }

  IntEffectsEncoder * IntEffectsEncoder::clone() const {
    return new IntEffectsEncoder(*this);
  }

  std::vector<std::string> IntEffectsEncoder::encoded_variable_names() const {
    std::vector<std::string> ans;
    for (int i = 0; i < dim(); ++i) {
      ans.push_back(variable_name() + ":" + std::to_string(i));
    }
    return ans;
  }

  int IntEffectsEncoder::baseline_level() const {
    return key_->max_levels() - 1;
  }

  EffectsEncoder::EffectsEncoder(const std::string &variable_name,
                                 const Ptr<CatKey> &key,
                                 const std::string &baseline_level)
      : EffectsEncoderBase(variable_name),
        key_(key),
        baseline_level_(baseline_level)
  {
    if (key_->max_levels() <= 0) {
      report_error("A categorical data key used to create an EffectsEncoder "
                   "must have a defined maximum number of levels. ");
    }

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

  EffectsEncoder::EffectsEncoder(const EffectsEncoder &rhs)
      : EffectsEncoderBase(rhs),
        key_(rhs.key_->clone())
  {}

  EffectsEncoder & EffectsEncoder::operator=(const EffectsEncoder &rhs) {
    if (&rhs != this) {
      EffectsEncoderBase::operator=(rhs);
      key_.reset(rhs.key_->clone());
    }
    return *this;
  }

  EffectsEncoder *EffectsEncoder::clone() const {
    return new EffectsEncoder(*this);
  }

  int EffectsEncoderBase::dim() const {
    return key().max_levels() - 1;
  }

  Vector EffectsEncoderBase::encode_level(int level) const {
    Vector ans(dim());
    encode_level(level, VectorView(ans));
    return ans;
  }

  void EffectsEncoderBase::encode_level(int level, VectorView view) const {
    if (level == baseline_level()) {
      view = -1;
    } else {
        view = 0.0;
      if (level < baseline_level()) {
        view[level] = 1.0;
      } else {
        view[level - 1] = 1.0;
      }
    }
  }


  Matrix EffectsEncoderBase::encode_dataset(const DataTable &table) const {
    return encode(table.get_nominal(variable_name()));
  }

  //---------------------------------------------------------------------------
  Matrix IntEffectsEncoder::encode(const CategoricalVariable &variable) const {
    Matrix ans(variable.size(), dim());
    for (size_t i = 0; i < variable.size(); ++i) {
      encode_level(variable[i]->value(), ans.row(i));
    }
    return ans;
  }

  Vector IntEffectsEncoder::encode_row(const MixedMultivariateData &row) const {
    return encode_level(row.categorical(variable_name()).value());
  }

  void IntEffectsEncoder::encode_row(
      const MixedMultivariateData &row, VectorView view) const {
    encode_level(row.categorical(variable_name()).value(), view);
  }

  Vector IntEffectsEncoder::encode(const CategoricalData &data) const {
    return encode_level(data.value());
  }

  void IntEffectsEncoder::encode(const CategoricalData &data,
                                 VectorView view) const {
    return encode_level(data.value(), view);
  }

  Vector EffectsEncoder::encode(const LabeledCategoricalData &data) const {
    return encode_level(key_->findstr(data.label()));
  }

  void EffectsEncoder::encode(const LabeledCategoricalData &data,
                              VectorView view) const {
    return encode_level(key_->findstr(data.label()), view);
  }

  //---------------------------------------------------------------------------
  Matrix EffectsEncoder::encode(const CategoricalVariable &variable) const {
    Matrix ans(variable.size(), dim());
    for (size_t i = 0; i < variable.size(); ++i) {
      const std::string &label(variable[i]->label());
      encode_level(key_->findstr(label), ans.row(i));
    }
    return ans;
  }

  Vector EffectsEncoder::encode_row(const MixedMultivariateData &row) const {
    const LabeledCategoricalData &data_point(row.categorical(variable_name()));
    return encode_level(key_->findstr(data_point.label()));
  }

  void EffectsEncoder::encode_row(
      const MixedMultivariateData &row, VectorView view) const {
    const LabeledCategoricalData &data_point(row.categorical(variable_name()));
    encode_level(key_->findstr(data_point.label()), view);
  }

  std::vector<std::string> EffectsEncoder::encoded_variable_names() const {
    std::vector<std::string> ans;
    for (int i = 0; i < dim(); ++i) {
      ans.push_back(variable_name() + ":" + key_->label(i));
    }
    return ans;
  }

  //===========================================================================

  IdentityEncoder::IdentityEncoder(const std::string &variable_name)
      : MainEffectEncoder(variable_name)
  {}

  IdentityEncoder * IdentityEncoder::clone() const {
    return new IdentityEncoder(*this);
  }

  Matrix IdentityEncoder::encode_dataset(const DataTable &data) const {
    return Matrix(data.nrow(), 1, data.get_numeric(variable_name()));
  }

  Vector IdentityEncoder::encode_row(const MixedMultivariateData &data) const {
    double value = data.numeric(variable_name()).value();
    return Vector(1, value);
  }

  void IdentityEncoder::encode_row(const MixedMultivariateData &data,
                                   VectorView view) const {
    view[0] = data.numeric(variable_name()).value();
  }

  std::vector<std::string> IdentityEncoder::encoded_variable_names() const {
    return std::vector<std::string>(1, variable_name());
  }

  //===========================================================================
  InteractionEncoder::InteractionEncoder(
      const Ptr<DataEncoder> &encoder1, const Ptr<DataEncoder> &encoder2)
      : encoder1_(encoder1),
        encoder2_(encoder2),
        wsp1_(encoder1->dim()),
        wsp2_(encoder2->dim())
  {}

  std::vector<std::string> InteractionEncoder::encoded_variable_names() const {
    std::vector<std::string> names1 = encoder1_->encoded_variable_names();
    std::vector<std::string> names2 = encoder2_->encoded_variable_names();
    std::vector<std::string> ans;

    for (const auto &name1 : names1) {
      for (const auto &name2 : names2) {
        ans.push_back(name1 + ":" + name2);
      }
    }
    return ans;
  }

  //===========================================================================
  Matrix DatasetEncoder::encode_dataset(const DataTable &table) const {
    int nrow = table.nrow();
    Matrix ans(nrow, dim());
    if (add_intercept_) {
      ans.col(0) = 1.0;
    }
    int start = add_intercept_;
    for (size_t i = 0; i < encoders_.size(); ++i) {
      int end = start + encoders_[i]->dim();
      SubMatrix(ans, 0, nrow-1, start, end-1) = encoders_[i]->encode_dataset(table);
      start = end;
    }
    return ans;
  }

  void DatasetEncoder::encode_row(const MixedMultivariateData &data,
                                  VectorView ans) const {
    if (add_intercept_) {
      ans[0] = 1.0;
    }
    int start = add_intercept_;
    for (size_t i = 0; i < encoders_.size(); ++i) {
      VectorView view(ans, start, encoders_[i]->dim());
      encoders_[i]->encode_row(data, view);
      start += encoders_[i]->dim();
    }
  }

  Vector DatasetEncoder::encode_row(const MixedMultivariateData &data) const {
    Vector ans(dim());
    encode_row(data, VectorView(ans));
    return ans;
  }

  std::vector<std::string> DatasetEncoder::encoded_variable_names() const {
    std::vector<std::string> ans;
    if (add_intercept_) {
      ans.push_back("(Intercept)");
    }

    for (const auto &enc : encoders_) {
      std::vector<std::string> names = enc->encoded_variable_names();
      ans.insert(ans.end(), names.begin(), names.end());
    }
    return ans;
  }

}  // namespace BOOM
