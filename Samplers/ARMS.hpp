// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#include "Samplers/Sampler.hpp"
#include "cpputil/Ptr.hpp"

#include "LinAlg/Vector.hpp"
#include "numopt.hpp"

namespace BOOM {

  class ArmsSampler : virtual public Sampler {
    /*======================================================================

      Adaptive rejection sampling from Gilks, Best, and Tan 1995,
      applied statistics.  Calls code from Wally Gilks' web site.

      ======================================================================*/
   public:
    ArmsSampler(const Target &target, const Vector &initial_value,
                bool log_convex = false);

    void find_limits();
    Vector draw(const Vector &old) override;
    virtual double logp(const Vector &x) const;
    void set(double);
    double eval() const;
    void set_limits(const Vector &lo, const Vector &hi);
    void set_lower_limits(const Vector &lo);
    void set_upper_limits(const Vector &hi);

   private:
    Target target_;
    Vector x_;
    Vector lower_limits_;
    Vector upper_limits_;
    uint which_;
    uint ninit_;
    bool log_convex_;  // set false if not sure;
  };
}  // namespace BOOM
