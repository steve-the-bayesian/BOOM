#ifndef BOOM_NUMOPT_SIMULATED_ANNEALING_OPTIMIZER_HPP_
#define BOOM_NUMOPT_SIMULATED_ANNEALING_OPTIMIZER_HPP_

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

#include "LinAlg/Vector.hpp"
#include "numopt.hpp"
#include "uint.hpp"

namespace BOOM {

  class SimulatedAnnealingOptimizer {
   public:
    SimulatedAnnealingOptimizer(const Target &target);

    // Args:
    //   inputs: On input this is the starting value of x in the search for the
    //     global minimum.  On output it the value of x producing the min value.
    //   rng:  A random number generator.
    //
    // Returns:
    //   The minimum value located in the search.
    double minimize(Vector &x, RNG &rng);

    void set_initial_temp(double t0) {initial_temp_ = t0;}
    void set_max_fun_count(Int max_count) {max_fun_count_ = max_count;} 
    void set_max_fun_count_per_temperatue(Int max_count) {
      max_fun_count_per_temp_ = max_count;
    }
    
    Int function_count() const {return fun_count_;}
    
   private:
    Target target_;
    Int fun_count_;

    Int max_fun_count_;
    Int max_fun_count_per_temp_;
    double initial_temp_;
  };
  
}  // namespace BOOM


#endif  //  BOOM_NUMOPT_SIMULATED_ANNEALING_OPTIMIZER_HPP_
