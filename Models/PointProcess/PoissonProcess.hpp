/*
  Copyright (C) 2005-2012 Steven L. Scott

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
#ifndef BOOM_POISSON_PROCESS_HPP_
#define BOOM_POISSON_PROCESS_HPP_

#include <functional>
#include "Models/ModelTypes.hpp"
#include "Models/PointProcess/PointProcess.hpp"
#include "cpputil/DateTime.hpp"

namespace BOOM {

  struct NullDataGenerator {
    Data *operator()() { return 0; }
  };

  // A base class for the Poisson process.  The HomogeneousPoissonProcess and
  // variaous flavors of inhomogeneous poisson processes can generalize.
  class PoissonProcess : virtual public Model {
   public:
    PoissonProcess *clone() const override = 0;

    // lambda(t)
    virtual double event_rate(const DateTime &t) const = 0;

    // Integral of lambda(t) from t0 to t1.
    virtual double expected_number_of_events(const DateTime &t0,
                                             const DateTime &t1) const = 0;

    // Adding data.
    virtual void add_exposure_window(const DateTime &t0,
                                     const DateTime &t1) = 0;
    virtual void add_event(const DateTime &t) = 0;

    // Simulate a PointProcess between t0 and t1.  If a function-like object
    // that returns a Data * is passed as the third object then the process will
    // include marks for those events where the returned value is non-NULL.
    virtual PointProcess simulate(
        RNG &rng, const DateTime &t0, const DateTime &t1,
        std::function<Data *()> mark_generator = NullDataGenerator()) const = 0;
  };

}  // namespace BOOM

#endif  // BOOM_POISSON_PROCESS_HPP_
