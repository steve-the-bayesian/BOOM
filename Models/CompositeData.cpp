// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "Models/CompositeData.hpp"

#include <utility>

namespace BOOM {

  CompositeData::CompositeData() = default;

  CompositeData::CompositeData(const std::vector<Ptr<Data>> &d) : dat_(d) {}

  CompositeData *CompositeData::clone() const {
    return new CompositeData(*this);
  }

  std::ostream &CompositeData::display(std::ostream &out) const {
    uint n = dat_.size();
    for (uint i = 0; i < n; ++i) {
      dat_[i]->display(out) << " ";
    }
    return out;
  }

  void CompositeData::add(const Ptr<Data> &dp) { dat_.push_back(dp); }

  uint CompositeData::dim() const { return dat_.size(); }

  Ptr<Data> CompositeData::get_ptr(uint i) { return dat_[i]; }

  const Data *CompositeData::get(uint i) const { return dat_[i].get(); }
}  // namespace BOOM
