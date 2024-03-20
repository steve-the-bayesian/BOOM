/*
  Copyright (C) 2005-2024 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "numopt/SimulatedAnnealingOptimizer.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    const double big = 1.0e+35;
    const double E1 = 1.7182818; /* exp(1.0)-1.0 */
  }  // namespace

  SimulatedAnnealingOptimizer::SimulatedAnnealingOptimizer(
      const Target &target)
      : target_(target),
        fun_count_(0),
        max_fun_count_(10000),
        max_fun_count_per_temp_(1000),
        initial_temp_(0.1)
  {}


  // This code was adapted from 'sann' in R.  Changes were made to
  // *) Make the code more readable by replacing variable names with English
  //    language equivalents.  E.g. t-> temperature.
  // *) Replace function parameters with object data members, where appropriate.
  // *) Vectorize code using C++ objects instead of C for loops E.g. for vector
  //    addition.
  // *) Make the code thread safe by requiring a RNG to be passed in as an
  //    argument instead of modifying global state.
  double SimulatedAnnealingOptimizer::minimize(Vector &best_point, RNG &rng) {
    size_t n = best_point.size();
    Vector point(n);
    Vector perturbation(n);
    Vector candidate(n);
    double best_y = target_(best_point);
    if (!std::isfinite(best_y)) {
      best_y = big;
    }

    point = best_point;
    double y = best_y; /* init system state p, y */
    double scale = 1.0 / initial_temp_;
    int local_fun_count = 1;
    while (local_fun_count < max_fun_count_) {             
      // cool down system 
      // Temperature annealing schedule.
      double temperature = initial_temp_ / log((double)local_fun_count + E1); 
      Int within_temp_function_count = 1;
      while ((within_temp_function_count <= max_fun_count_per_temp_)
             && (local_fun_count < max_fun_count_)) {
        /* Iterate at constant temperature. */

        // Generate a random perturbation in perturbation.
        perturbation.randomize_gaussian(0, scale * temperature, rng);
        
        // Add the perturbation to the current value to get the new candidate
        // point.
        candidate = point + perturbation;  
        double ytry = target_(candidate);      
        if (!std::isfinite(ytry)) {
          ytry = big;
        }
        double dy = ytry - y;
        if ((dy <= 0.0) || (runif_mt(rng, 0, 1) < exp(-dy / temperature))) { 
          // Accept new point?
          point = candidate;
          y = ytry;    /* update system state p, y */
          if (y <= best_y) {
            /* if system state is best, then update best system state
             * best_point, best_y */
            best_point = point;
            best_y = y;
          }
        }
        ++local_fun_count;
        ++within_temp_function_count;
      }
    }
    fun_count_ += local_fun_count;
    return best_y;
  }

}  // namespace BOOM
