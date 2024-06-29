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
#include <ostream>
#include <iomanip>
#include "LinAlg/Vector.hpp"
#include "math/Permutation.hpp"

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
      running_ = true;
    }

    void stop() {
      if (running_) {
        stop_time_ = clock_.now();
        std::chrono::duration<double> dt = stop_time_ - start_time_;
        elapsed_time_ += dt.count();
        running_ = false;
      }
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
    bool running_;
  };

  //===========================================================================
  class TimeRecorder {
   public:
    void record_elapsed_time(const std::string &category, double seconds) {
      seconds_per_category_[category] += seconds;
    }

    std::ostream & print(std::ostream &out) const {
      std::vector<std::string> categories;
      Vector times;
      for (const auto &el : seconds_per_category_) {
        categories.push_back(el.first);
        times.push_back(el.second);
      }

      Permutation<Int> order = Permutation<Int>::order(times);
      for (Int i = 0; i < order.size(); ++i) {
        out << std::setprecision(4) << std::setw(12) << times[order[i]]
        << "   " << categories[order[i]] << "\n";
      }
      out << std::flush;
      return out;
    }

    double operator[](const std::string &label) const {
      auto it = seconds_per_category_.find(label);
      if (it == seconds_per_category_.end()) {
        return negative_infinity();
      } else {
        return it->second;
      }
    }

   private:
    std::map<std::string, double> seconds_per_category_;
  };

  std::ostream & operator<<(std::ostream &out, const TimeRecorder &recorder) {
    return recorder.print(out);
  }

  //===========================================================================
  // A Timer that can be started at the beginning of a function or code block.
  // When the timer goes out of scope it records in 'recorder' the amount of
  // time since its creation.
  class ScopedTimer : public Timer {
   public:

    // Args:
    //   recorder: The TimeRecorder that will record the timer's value when it
    //     goes out of scope.
    //   category:  The label that recorder will use to record the elapsed time.
    ScopedTimer(TimeRecorder *recorder, const std::string &category)
        : recorder_(recorder),
          category_(category)
    {}

    ~ScopedTimer() {
      if (recorder_) {
        stop();
        recorder_->record_elapsed_time(category_, time_in_seconds());
      }
    }

   private:
    TimeRecorder *recorder_;
    std::string category_;
  };
}  // namespace BOOM

#endif  //  BOOM_CPPUTIL_TIMER_HPP_
