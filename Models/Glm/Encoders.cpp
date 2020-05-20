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

#include "Models/Glm/Encoders.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  EffectsEncoder::EffectsEncoder(int which_variable, const Ptr<CatKeyBase> &key)
      : MainEffectsEncoder(which_variable),
        key_(key)
  {
    if (key_->max_levels() <= 0) {
      report_error("A categorical data key used to create an EffectsEncoder "
                   "must have a defined maximum number of levels. ");
    }
  }

  int EffectsEncoder::dim() const {
    return key_->max_levels() - 1;
  }

  Vector EffectsEncoder::encode(const CategoricalData &data) const {
    return encode(data.value());
  }

  Vector EffectsEncoder::encode(int level) const {
    if (level == key_->max_levels() - 1) {
      return Vector(dim(), -1);
    } else {
      Vector ans(dim(), 0.0);
      ans[level] = 1.0;
      return ans;
    }
  }

  Matrix EffectsEncoder::encode(const CategoricalVariable &variable) const {
    Matrix ans(variable.size(), dim());
    for (size_t i = 0; i < variable.size(); ++i) {
      ans.row(i) = encode(*variable[i]);
    }
    return ans;
  }

  Matrix EffectsEncoder::encode_dataset(const DataTable &table) const {
    return encode(table.get_nominal(which_variable()));
  }

  //===========================================================================
  InteractionEncoder::InteractionEncoder(
      const Ptr<DataEncoder> &encoder1, const Ptr<DataEncoder> &encoder2)
      : encoder1_(encoder1),
        encoder2_(encoder2)
  {}


  //===========================================================================
  Matrix DatasetEncoder::encode_dataset(const DataTable &table) const {
    int nrow = table.nrow();
    Matrix ans(nrow, dim());
    int start = 0;
    for (size_t i = 0; i < encoders_.size(); ++i) {
      int end = start + encoders_[i]->dim();
      SubMatrix(ans, 0, nrow-1, start, end-1) = encoders_[i]->encode_dataset(table);
      start = end;
    }
    return ans;
  }

}  // namespace BOOM
