// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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
#ifndef BOOM_NULL_PARAM_POLICY_HPP
#define BOOM_NULL_PARAM_POLICY_HPP

#include "Models/ModelTypes.hpp"
#include "cpputil/Ptr.hpp"

namespace BOOM {

  class NullParamPolicy : virtual public Model {
    // for use with models that have no parameters: e.g. uniform
    // distributions.
   public:
    typedef NullParamPolicy ParamPolicy;

    NullParamPolicy();
    NullParamPolicy(const NullParamPolicy &rhs);
    NullParamPolicy *clone() const override = 0;
    NullParamPolicy &operator=(const NullParamPolicy &);

    // over-rides for abstract base Model
    std::vector<Ptr<Params>> parameter_vector() override;
    const std::vector<Ptr<Params>> parameter_vector() const override;
  };

}  // namespace BOOM
#endif  // BOOM_NULL_PARAM_POLICY_HPP
