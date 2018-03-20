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

#ifndef BOOM_SCALAR_SLICE_SAMPLER_HPP
#define BOOM_SCALAR_SLICE_SAMPLER_HPP
#include <functional>
#include <iosfwd>
#include "Samplers/Sampler.hpp"
#include "TargetFun/TargetFun.hpp"
namespace BOOM {

  class ScalarSliceSampler : public ScalarSampler {
   public:
    typedef std::function<double(double)> Fun;

    explicit ScalarSliceSampler(const Fun &F, bool Unimodal = false,
                       double suggested_dx = 1.0, RNG *rng = 0);
    void set_limits(double lo, double hi);
    void set_lower_limit(double lo);
    void set_upper_limit(double hi);
    void unset_limits();
    void set_suggested_dx(double dx);
    void estimate_dx(bool should_dx_be_estimated);
    void set_min_dx(double dx);
    double draw(double x) override;
    virtual double logp(double x) const;

   private:
    //    const ScalarTargetFun &logf_;
    Fun logf_;
    double lo_;
    double hi_;
    double suggested_dx_;
    double min_dx_;
    //    double x_;
    double logplo_;
    double logphi_;
    double logp_slice_;

    double lower_bound_;
    double upper_bound_;
    bool lo_set_manually_;
    bool hi_set_manually_;
    bool unimodal_;
    bool estimate_dx_;

    void find_limits(double x);
    bool find_lower_limit(double x);
    bool find_upper_limit(double x);
    bool find_limits_unbounded(double x);
    void find_limits_unbounded_unimodal(double x);

    void contract(double x, double xstar, double logp);
    bool done_doubling() const;
    void double_lo(double x);
    void double_hi(double x);

    // quality assurance and error handling
    std::ostream &print_state(std::ostream &) const;
    //    void ensure_slice(double x);
    void check_probs(double x);  // ensure probabilities are legal
    void check_slice(double x);  // ensure lo <= x <= hi
    void check_lower_limit(double x);
    void check_upper_limit(double x);
    void check_finite(double x, double logpstar);

    bool doubly_bounded() const;  // bounded on both sides
    bool lower_bounded() const;
    bool upper_bounded() const;
    bool unbounded() const;  // on either side
    void handle_error(const std::string &msg, double x) const;
    std::string error_message(double lo, double hi, double x, double logplo,
                              double logphi, double logp_slice) const;
  };
}  // namespace BOOM
#endif  // BOOM_SCALAR_SLICE_SAMPLER_HPP
