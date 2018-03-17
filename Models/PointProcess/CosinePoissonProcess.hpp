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

#ifndef BOOM_COSINE_POISSON_PROCESS_HPP_
#define BOOM_COSINE_POISSON_PROCESS_HPP_

#include <functional>
#include "Models/PointProcess/PoissonProcess.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {
  // The CosinePoissonProcess is an inhomogeneous Poisson process with
  // rate function lambda * (1 + cos(frequency * t)), where t is the
  // time in days since Jan 1 1970..  It is mainly useful for testing
  // code involving inhomogeneous processes.
  class CosinePoissonProcess : public PoissonProcess,
                               public ParamPolicy_2<UnivParams, UnivParams>,
                               public IID_DataPolicy<PointProcess>,
                               public PriorPolicy {
   public:
    explicit CosinePoissonProcess(double lambda = 1.0, double frequency = 1.0);
    CosinePoissonProcess *clone() const override;

    double event_rate(const DateTime &t) const override;
    double expected_number_of_events(const DateTime &t0,
                                     const DateTime &t1) const override;

    // Adding data is a no-op since this
    void add_exposure_window(const DateTime &t0, const DateTime &t1) override {}
    void add_event(const DateTime &t) override {}

    double lambda() const;
    double frequency() const;

    PointProcess simulate(RNG &rng, const DateTime &t0, const DateTime &t1,
                          std::function<Data *()> mark_generator =
                              NullDataGenerator()) const override;

   private:
    DateTime origin_;
  };

}  // namespace BOOM
#endif  // BOOM_COSINE_POISSON_PROCESS_HPP_
