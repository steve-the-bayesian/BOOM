#ifndef BOOM_CPPUTIL_TIMER_HPP_
#define BOOM_CPPUTIL_TIMER_HPP_
/*
  Copyright (C) 2019 Steven L. Scott

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

#include <chrono>

namespace BOOM {

  // A simple timer for timing code snippets.
  class Timer {
   public:
    Timer()
        : elapsed_time_(0.0)
    {
      start();
    }

    void start() {
      start_time_ = clock_.now();
    }
    
    void stop() {
      stop_time_ = clock_.now();
      std::chrono::duration<double> dt = stop_time_ - start_time_;
      elapsed_time_ += dt.count();
    }

    void clear() {
      elapsed_time_ = 0.0;
    }
    
    double time_in_seconds() const {
      return elapsed_time_;
    }

   private:
    std::chrono::high_resolution_clock clock_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_time_;
    double elapsed_time_;
  };
}  // namespace BOOM

#endif  //  BOOM_CPPUTIL_TIMER_HPP_
