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
#include "Models/PointProcess/BoundedPoissonProcessSimulator.hpp"
#include "distributions.hpp"

namespace BOOM {

  BoundedPoissonProcessSimulator::BoundedPoissonProcessSimulator(
      const PoissonProcess *process, double max_event_rate)
      : process_(process), max_event_rate_(max_event_rate) {}

  PointProcess BoundedPoissonProcessSimulator::simulate(
      RNG &rng, const DateTime &t0, const DateTime &t1,
      const std::function<Data *()> &mark_generator) const {
    PointProcess ans(t0, t1);
    double duration = t1 - t0;
    int number_of_candidate_events = rpois(max_event_rate_ * duration);
    Vector times(number_of_candidate_events);
    for (int i = 0; i < number_of_candidate_events; ++i) {
      times[i] = runif_mt(rng, 0, duration);
    }
    times.sort();

    for (int i = 0; i < times.size(); ++i) {
      DateTime cand = t0 + times[i];
      double prob = process_->event_rate(cand) / max_event_rate_;
      if (runif_mt(rng, 0, 1) < prob) {
        Data *mark = mark_generator();
        if (mark) {
          ans.add_event(cand, Ptr<Data>(mark));
        } else {
          ans.add_event(cand);
        }
      }
    }
    return ans;
  }

}  // namespace BOOM
