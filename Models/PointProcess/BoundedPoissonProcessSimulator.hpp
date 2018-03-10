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

#ifndef BOOM_BOUNDED_POISSON_PROCESS_SIMULATOR_HPP_
#define BOOM_BOUNDED_POISSON_PROCESS_SIMULATOR_HPP_

#include <functional>
#include "Models/PointProcess/PoissonProcess.hpp"

namespace BOOM {
  // A class to help concrete PoissonProcess models implement the
  // simulate() method using thinning.  To use it, you must know the
  // maximum value of lambda(t) for t in the interior of the
  // observation window.
  class BoundedPoissonProcessSimulator {
   public:
    BoundedPoissonProcessSimulator(const PoissonProcess *process_to_simulate,
                                   double max_event_rate);
    PointProcess simulate(RNG &rng, const DateTime &t0, const DateTime &t1,
                          const std::function<Data *()> &mark_generator =
                              NullDataGenerator()) const;

   private:
    const PoissonProcess *process_;
    double max_event_rate_;
    mutable RNG rng_;
  };
}  // namespace BOOM

#endif  //  BOOM_BOUNDED_POISSON_PROCESS_SIMULATOR_HPP_
