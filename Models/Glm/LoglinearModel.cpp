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

#include "Models/Glm/LoglinearModel.hpp"

namespace BOOM {

  namespace {
    using MCD = MultivariateCategoricalData;
  }  // namespace

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
      out << *data_[i];
      if (i + 1 < data_.size()) {
        out << ' ';
      }
    }
    return out;
  }
  //===========================================================================
  CategoricalMainEffect::CategoricalMainEffect(
      int which_variable, const Ptr<CatKey> &key):
      EffectsEncoder(which_variable, key)
  {}

  Vector CategoricalMainEffect::encode_categorical_data(const MCD &data) const {
    return encode(data[which_variable()]);
  }

  //===========================================================================
  CategoricalInteractionEncoder::CategoricalInteractionEncoder(
      const Ptr<CategoricalMainEffect> &main1,
      const Ptr<CategoricalMainEffect> &main2)
      : InteractionEncoder(main1, main2),
        enc1_(main1.get()),
        enc2_(main2.get())
  {}

  CategoricalInteractionEncoder::CategoricalInteractionEncoder(
      const Ptr<CategoricalMainEffect> &main_effect,
      const Ptr<CategoricalInteractionEncoder> &interaction)
      : InteractionEncoder(main_effect, interaction),
        enc1_(main_effect.get()),
        enc2_(interaction.get())
  {}

  CategoricalInteractionEncoder::CategoricalInteractionEncoder(
      const Ptr<CategoricalInteractionEncoder> &interaction1,
      const Ptr<CategoricalInteractionEncoder> &interaction2)
      : InteractionEncoder(interaction1, interaction2),
        enc1_(interaction1.get()),
        enc2_(interaction2.get())
  {}

  Vector CategoricalInteractionEncoder::encode_categorical_data(
      const MCD &data) const {
    Vector v1 = enc1_->encode_categorical_data(data);
    Vector v2 = enc2_->encode_categorical_data(data);
    Vector ans(dim());
    int index = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
      for (size_t j = 0; j < v2.size(); ++j) {
        ans[index++] = v1[i] * v2[j];
      }
    }
    return ans;
  }

  //===========================================================================
  void CategoricalDatasetEncoder::add_main_effect(
      const Ptr<CategoricalMainEffect> &main_effect) {
    DatasetEncoder::add_encoder(main_effect);
    categorical_encoders_.push_back(main_effect.get());
  }

  void CategoricalDatasetEncoder::add_interaction(
      const Ptr<CategoricalInteractionEncoder> &interaction) {
    DatasetEncoder::add_encoder(interaction);
    categorical_encoders_.push_back(interaction.get());
  }

  Vector CategoricalDatasetEncoder::encode_categorical_data(
      const MultivariateCategoricalData &data) const {
    uint start = 0;
    Vector ans(dim(), 0.0);
    for (int i = 0; i < categorical_encoders_.size(); ++i) {
      uint dim = categorical_encoders_[i]->dim();
      VectorView view(ans, start, dim);
      view = categorical_encoders_[i]->encode_categorical_data(data);
      start += dim;
    }
    return ans;
  }

  //===========================================================================

  double LoglinearModel::logp(const MultivariateCategoricalData &data) const {
    return coef().predict(encoder_.encode_categorical_data(data));
  }

}
