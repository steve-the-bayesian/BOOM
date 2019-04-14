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

#ifndef BOOM_POINT_PROCESS_DATA_HPP_
#define BOOM_POINT_PROCESS_DATA_HPP_

#include "Models/DataTypes.hpp"
#include "cpputil/DateTime.hpp"

namespace BOOM {

  class PointProcessEvent : public Data {
   public:
    // Implicit conversion from DateTime is intentional.
    PointProcessEvent(const DateTime &time);  // NOLINT
    PointProcessEvent(const DateTime &time, const Ptr<Data> &mark);
    PointProcessEvent *clone() const override;
    std::ostream &display(std::ostream &) const override;

    const DateTime &timestamp() const;
    const Data *mark() const;
    Ptr<Data> mark_ptr() const;
    bool has_mark() const;

    bool operator<(const PointProcessEvent &rhs) const;

    // Comparisons to DateTime.
    bool operator<(const DateTime &rhs) const;
    bool operator==(const DateTime &rhs) const;
    bool operator!=(const DateTime &rhs) const {return !(*this == rhs);}
    bool operator<=(const DateTime &rhs) const {
      return *this < rhs || *this == rhs;
    }
    bool operator>(const DateTime &rhs) const {return !(*this <= rhs);}
    bool operator>=(const DateTime &rhs) const {return !(*this < rhs);}
    
   private:
    DateTime timestamp_;
    Ptr<Data> mark_;
  };

  // A point process is a set of events inside a time window.  Note
  // that if you want to use real time (instead of calendar time, you
  // can wrap each event time in DateTime(real_number).
  class PointProcess : public Data {
   public:
    // If you create an empty point process the beginning and end of
    // the observation window will be the default value for DateTime.
    // You must adjust the window before you can use the object.
    PointProcess() {}
    PointProcess(const DateTime &begin, const DateTime &end);
    PointProcess(const DateTime &begin, const DateTime &end,
                 const std::vector<DateTime> &events);
    PointProcess(const DateTime &begin, const DateTime &end,
                 const std::vector<DateTime> &events,
                 const std::vector<Ptr<Data> > &marks);

    // Use this constructor when the observation window is unknown and
    // all you have is the vector of event times.  It will set the
    // observation window to coincide with the first and last event
    // time.
    explicit PointProcess(const std::vector<DateTime> &events);
    PointProcess(const std::vector<DateTime> &events,
                 const std::vector<Ptr<Data> > &marks);

    PointProcess(const PointProcess &rhs);

    PointProcess *clone() const override;
    std::ostream &display(std::ostream &out) const override;

    uint number_of_events() const;
    double window_duration() const;  // Time is measured in days.

    const PointProcessEvent &event(int i) const;

    // The interarrival time between events i and i-1.  If i==0 the
    // arrival time is measured from the start of the observation
    // window.  If i == number_of_events() then the arrival time is
    // the time from the last event to the end of the observation
    // window.  Time is measured in days.
    double arrival_time(int i) const;

    // If *this and rhs share an endpoint then they will be joined and
    // their events will be combined.  It is an error to append a
    // point process that does not adjoin *this.
    PointProcess &append(const PointProcess &rhs);

    // Predicates indicating whether *this and rhs share an endpoint
    // in their observation windows.
    bool immediately_follows(const PointProcess &rhs) const;
    bool immediately_precedes(const PointProcess &rhs) const;
    bool adjoins(const PointProcess &rhs) const;

    // It is an error to add a timestamp that is outside the window
    // covered by the process.
    PointProcess &add_event(const DateTime &timestamp);
    PointProcess &add_event(const DateTime &timestamp, const Ptr<Data> &mark);
    PointProcess &add_event(const PointProcessEvent &event);

    // Accessors and setters for either end of the observation window.
    const DateTime &window_begin() const;
    const DateTime &window_end() const;
    void set_window(const DateTime &begin, const DateTime &end);
    void set_window_end(const DateTime &end);
    void set_window_begin(const DateTime &start);

    void set_resolution(double time_in_days);

   private:
    // Functions to check state
    void check_endpoints(const DateTime &begin, const DateTime &end) const;
    void check_events_inside_window(const DateTime &begin,
                                    const DateTime &end) const;
    void check_event_inside_window(const DateTime &) const;
    void check_legal_event_number(int i) const;

    DateTime begin_;
    DateTime end_;
    std::vector<PointProcessEvent> events_;

    // Events with less than resolution_ time scale differences are
    // equivalent.  Defaults to 1 microsecond.
    double resolution_;
  };

}  // namespace BOOM
#endif  //  BOOM_POINT_PROCESS_DATA_HPP_
