// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_NONPARAMETRIC_PARAM_POLICY_HPP_
#define BOOM_NONPARAMETRIC_PARAM_POLICY_HPP_
#include <Models/ModelTypes.hpp>

namespace BOOM {
  // This is a dummy mix-in policy to supply no-ops for the virtual
  // functions required by the Model base class for handling
  // parameters.  Nonparametric models don't have parameters, so this
  // class just returns empty parameter vectors.

  class NonparametricParamPolicy : virtual public Model{
   public:
    typedef NonparametricParamPolicy ParamPolicy;
    // Return an empty vector of parameters.
    ParamVector parameter_vector() override {
      return ParamVector();
    }
    const ParamVector parameter_vector() const override {
      return ParamVector();
    }
  };
}
#endif //  BOOM_NONPARAMETRIC_PARAM_POLICY_HPP_
