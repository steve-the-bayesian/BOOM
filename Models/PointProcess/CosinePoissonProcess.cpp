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

#include "Models/PointProcess/CosinePoissonProcess.hpp"
#include <cmath>
#include "Models/PointProcess/BoundedPoissonProcessSimulator.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  CosinePoissonProcess::CosinePoissonProcess(double lambda, double frequency)
      : ParamPolicy(new UnivParams(lambda), new UnivParams(frequency)),
        origin_(Date(Jan, 1, 1970), 0.0) {
    if ((lambda < 0) || (frequency <= 0)) {
      report_error("Invalid arguments to CosinePoissonProcess.");
    }
  }

  CosinePoissonProcess *CosinePoissonProcess::clone() const {
    return new CosinePoissonProcess(*this);
  }

  double CosinePoissonProcess::event_rate(const DateTime &t) const {
    double dt = t - origin_;
    return lambda() * (1 + std::cos(frequency() * dt));
  }

  double CosinePoissonProcess::expected_number_of_events(
      const DateTime &t0, const DateTime &t1) const {
    double real_t0 = t0 - origin_;
    double real_t1 = t1 - origin_;
    double ans = lambda() * (real_t1 - real_t0);
    ans += (lambda() / frequency()) *
           (sin(frequency() * real_t1) - sin(frequency() * real_t0));
    return ans;
  }

  double CosinePoissonProcess::lambda() const { return prm1_ref().value(); }

  double CosinePoissonProcess::frequency() const { return prm2_ref().value(); }

  PointProcess CosinePoissonProcess::simulate(
      RNG &rng, const DateTime &t0, const DateTime &t1,
      std::function<Data *()> mark_generator) const {
    BoundedPoissonProcessSimulator simulator(this, 2 * lambda());
    return simulator.simulate(rng, t0, t1, mark_generator);
  }

}  // namespace BOOM
