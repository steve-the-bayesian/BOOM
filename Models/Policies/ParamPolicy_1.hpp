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
#ifndef BOOM_PARAM_POLICY_1_HPP
#define BOOM_PARAM_POLICY_1_HPP

#include "Models/ModelTypes.hpp"
#include "cpputil/Ptr.hpp"

namespace BOOM {

  template <class P>
  class ParamPolicy_1 : virtual public Model {
   public:
    typedef P ParamType;
    typedef ParamPolicy_1<P> ParamPolicy;

    ParamPolicy_1();
    explicit ParamPolicy_1(const Ptr<P> &pPrm);
    ParamPolicy_1(const ParamPolicy_1 &rhs);
    ParamPolicy_1<P> &operator=(const ParamPolicy_1 &rhs);

    void set_prm(const Ptr<P> &p) {
      prm_ = p;
      set_parameter_vector();
    }
    Ptr<P> prm() { return prm_; }
    const Ptr<P> prm() const { return prm_; }
    P &prm_ref() { return *prm_; }
    const P &prm_ref() const { return *prm_; }

    // over-rides for abstract base Model
    std::vector<Ptr<Params>> parameter_vector() override;
    const std::vector<Ptr<Params>> parameter_vector() const override;

   private:
    Ptr<P> prm_;
    std::vector<Ptr<Params>> parameter_vector_;
    void set_parameter_vector();
  };
  //------------------------------------------------------------

  template <class P>
  void ParamPolicy_1<P>::set_parameter_vector() {
    parameter_vector_ = std::vector<Ptr<Params>>(1, prm_);
  }

  template <class P>
  ParamPolicy_1<P>::ParamPolicy_1() : prm_() {
    set_parameter_vector();
  }
  template <class P>
  ParamPolicy_1<P>::ParamPolicy_1(const Ptr<P> &pPrm) : prm_(pPrm) {
    set_parameter_vector();
  }

  template <class P>
  ParamPolicy_1<P>::ParamPolicy_1(const ParamPolicy_1 &rhs)
      : Model(rhs), prm_(rhs.prm_->clone()) {
    set_parameter_vector();
  }

  template <class P>
  ParamPolicy_1<P> &ParamPolicy_1<P>::operator=(const ParamPolicy_1 &rhs) {
    if (&rhs != this) {
      prm_ = rhs.prm_->clone();
      set_parameter_vector();
    }
    return *this;
  }

  template <class P>
  std::vector<Ptr<Params>> ParamPolicy_1<P>::parameter_vector() {
    return parameter_vector_;
  }

  template <class P>
  const std::vector<Ptr<Params>> ParamPolicy_1<P>::parameter_vector() const {
    return parameter_vector_;
  }

}  // namespace BOOM
#endif  // BOOM_PARAM_POLICY_1_HPP
