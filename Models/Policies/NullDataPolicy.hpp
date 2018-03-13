// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#ifndef BOOM_NULL_DATA_POLICY_HPP
#define BOOM_NULL_DATA_POLICY_HPP
#include <fstream>
#include <list>
#include "Models/ModelTypes.hpp"
#include "Models/Policies/DataInfoPolicy.hpp"

namespace BOOM {
  // A class for models that don't have data (for whatever reason)
  class NullDataPolicy : virtual public Model {
   public:
    typedef NullDataPolicy DataPolicy;

   public:
    NullDataPolicy() {}
    void add_data(const Ptr<Data> &) override {}
    void clear_data() override {}
    void combine_data(const Model &, bool = true) override {}

   private:
  };
}  // namespace BOOM
#endif  // BOOM_NULL_DATA_POLICY_HPP
