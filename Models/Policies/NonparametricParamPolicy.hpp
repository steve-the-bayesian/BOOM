#ifndef BOOM_NONPARAMETRIC_PARAM_POLICY_HPP_
#define BOOM_NONPARAMETRIC_PARAM_POLICY_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/ModelTypes.hpp"

namespace BOOM {

  // Nonparametric models don't have parameters, so parameter_vector() and
  // vectorize_params() return empty objects, and their inverse operations are
  // no-ops.
  class NonparametricParamPolicy : virtual public Model {
   public:
    typedef NonparametricParamPolicy ParamPolicy;
    std::vector<Ptr<Params>> parameter_vector() override {
      return std::vector<Ptr<Params>>();
    }
    const std::vector<Ptr<Params>> parameter_vector() const override {
      return std::vector<Ptr<Params>>();
    }

    Vector vectorize_params(bool minimal = true) const override {
      return Vector(0);
    }
    void unvectorize_params(const Vector &v, bool minimal = true) override {}
  };

}  // namespace BOOM

#endif  //  BOOM_NONPARAMETRIC_PARAM_POLICY_HPP_
