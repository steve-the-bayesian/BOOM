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
#ifndef BOOM_PARAM_POLICY_3_HPP
#define BOOM_PARAM_POLICY_3_HPP
#include "Models/ModelTypes.hpp"
#include "cpputil/Ptr.hpp"
namespace BOOM {

  template <class P1, class P2, class P3>
  class ParamPolicy_3 : virtual public Model {
   public:
    typedef P1 param_type_1;
    typedef P2 param_type_3;
    typedef ParamPolicy_3<P1, P2, P3> ParamPolicy;

    ParamPolicy_3();
    ParamPolicy_3(const Ptr<P1> &p1, Ptr<P2> p2, Ptr<P3> p3);
    ParamPolicy_3(const ParamPolicy_3 &rhs);
    ParamPolicy_3<P1, P2, P3> &operator=(const ParamPolicy_3 &rhs);

    Ptr<P1> prm1() { return prm1_; }
    const Ptr<P1> prm1() const { return prm1_; }
    P1 &prm1_ref() { return *prm1_; }
    const P1 &prm1_ref() const { return *prm1_; }

    Ptr<P2> prm2() { return prm2_; }
    const Ptr<P2> prm2() const { return prm2_; }
    P2 &prm2_ref() { return *prm2_; }
    const P2 &prm2_ref() const { return *prm2_; }

    Ptr<P3> prm3() { return prm3_; }
    const Ptr<P3> prm3() const { return prm3_; }
    P3 &prm3_ref() { return *prm3_; }
    const P3 &prm3_ref() const { return *prm3_; }

    void set_params(const Ptr<P1> &p1, Ptr<P2> p2, Ptr<P3> p3) {
      prm1_ = p1;
      prm2_ = p2, prm3_ = p3;
      set_t();
    }

    // overrides for abstract base Model
    std::vector<Ptr<Params>> parameter_vector() override { return t_; }

    const std::vector<Ptr<Params>> parameter_vector() const override {
      return t_;
    }

   private:
    Ptr<P1> prm1_;
    Ptr<P2> prm2_;
    Ptr<P3> prm3_;
    std::vector<Ptr<Params>> t_;
    void set_t();
  };
  //------------------------------------------------------------

  template <class P1, class P2, class P3>
  void ParamPolicy_3<P1, P2, P3>::set_t() {
    t_ = std::vector<Ptr<Params>>(3);
    t_[0] = prm1_;
    t_[1] = prm2_;
    t_[2] = prm3_;
  }

  template <class P1, class P2, class P3>
  ParamPolicy_3<P1, P2, P3>::ParamPolicy_3() : prm1_(), prm2_(), prm3_() {
    set_t();
  }
  template <class P1, class P2, class P3>
  ParamPolicy_3<P1, P2, P3>::ParamPolicy_3(const Ptr<P1> &p1, Ptr<P2> p2,
                                           Ptr<P3> p3)
      : prm1_(p1), prm2_(p2), prm3_(p3) {
    set_t();
  }

  template <class P1, class P2, class P3>
  ParamPolicy_3<P1, P2, P3>::ParamPolicy_3(const ParamPolicy_3 &rhs)
      : Model(rhs),
        prm1_(rhs.prm1_->clone()),
        prm2_(rhs.prm2_->clone()),
        prm3_(rhs.prm3_->clone()) {
    set_t();
  }

  template <class P1, class P2, class P3>
  ParamPolicy_3<P1, P2, P3> &ParamPolicy_3<P1, P2, P3>::operator=(
      const ParamPolicy_3 &rhs) {
    if (&rhs != this) {
      prm1_ = rhs.prm1_->clone();
      prm2_ = rhs.prm2_->clone();
      prm3_ = rhs.prm3_->clone();
      set_t();
    }
    return *this;
  }

}  // namespace BOOM
#endif  // BOOM_PARAM_POLICY_3_HPP
