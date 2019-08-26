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

#include <algorithm>

#include "Models/PointProcess/PointProcess.hpp"
#include "cpputil/DateTime.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  PointProcessEvent::PointProcessEvent(const DateTime &time)
      : timestamp_(time), mark_(0) {}

  PointProcessEvent::PointProcessEvent(const DateTime &time,
                                       const Ptr<Data> &mark)
      : timestamp_(time), mark_(mark) {}

  PointProcessEvent *PointProcessEvent::clone() const {
    return new PointProcessEvent(*this);
  }

  std::ostream &PointProcessEvent::display(std::ostream &out) const {
    out << timestamp_;
    if (!!mark_) out << *mark_;
    return out;
  }

  const DateTime &PointProcessEvent::timestamp() const { return timestamp_; }
  const Data *PointProcessEvent::mark() const { return mark_.get(); }
  Ptr<Data> PointProcessEvent::mark_ptr() const { return mark_; }
  bool PointProcessEvent::has_mark() const {
    return mark_.get() && !mark_->missing();
  }

  bool PointProcessEvent::operator<(const PointProcessEvent &rhs) const {
    return timestamp_ < rhs.timestamp_;
  }

  bool PointProcessEvent::operator==(const DateTime &rhs) const {
    return timestamp_ == rhs;
  }
  
  bool PointProcessEvent::operator<(const DateTime &rhs) const {
    return timestamp_ < rhs;
  }

  //======================================================================

  PointProcess::PointProcess(const DateTime &begin, const DateTime &end)
      : begin_(begin),
        end_(end),
        resolution_(DateTime::microseconds_to_days(1.0)) {
    check_endpoints(begin_, end_);
  }

  PointProcess::PointProcess(const DateTime &begin, const DateTime &end,
                             const std::vector<DateTime> &events)
      : begin_(begin),
        end_(end),
        resolution_(DateTime::microseconds_to_days(1.0)) {
    events_.assign(events.begin(), events.end());
    check_endpoints(begin_, end_);
    std::sort(events_.begin(), events_.end());
    check_events_inside_window(begin_, end_);
  }

  PointProcess::PointProcess(const DateTime &begin, const DateTime &end,
                             const std::vector<DateTime> &events,
                             const std::vector<Ptr<Data> > &marks)
      : begin_(begin),
        end_(end),
        resolution_(DateTime::microseconds_to_days(1.0)) {
    int n = events.size();
    if (events.size() != marks.size()) {
      ostringstream err;
      err << "events and marks must have the same length in "
          << "PointProcess constructor." << endl
          << "size of 'events' = " << events.size() << endl
          << "size of 'marks'  = " << marks.size() << endl;
      report_error(err.str());
    }
    check_endpoints(begin_, end_);
    events_.reserve(n);
    for (int i = 0; i < n; ++i) {
      PointProcessEvent event(events[i], marks[i]);
      events_.push_back(event);
    }
    std::sort(events_.begin(), events_.end());
    check_events_inside_window(begin_, end_);
  }

  PointProcess::PointProcess(const std::vector<DateTime> &events)
      : resolution_(DateTime::microseconds_to_days(1.0)) {
    events_.assign(events.begin(), events.end());
    std::sort(events_.begin(), events_.end());
    begin_ = events_[0].timestamp();
    end_ = events_.back().timestamp();
  }

  PointProcess::PointProcess(const std::vector<DateTime> &events,
                             const std::vector<Ptr<Data> > &marks)
      : resolution_(DateTime::microseconds_to_days(1.0)) {
    int n = events.size();
    if (n == 0) {
      report_error("Attempt to create an empty PointProcess");
    }

    if (marks.size() != n) {
      ostringstream err;
      err << "events and marks must have the same length "
          << "in PointProcess constructor." << endl
          << "size of 'events' = " << events.size() << endl
          << "size of 'marks'  = " << marks.size() << endl;
      report_error(err.str());
    }
    for (int i = 0; i < n; ++i) {
      PointProcessEvent event(events[i], marks[i]);
      events_.push_back(event);
    }
    std::sort(events_.begin(), events_.end());
    begin_ = events_[0].timestamp();
    end_ = events_.back().timestamp();
  }

  PointProcess::PointProcess(const PointProcess &rhs)
      : Data(rhs),
        begin_(rhs.begin_),
        end_(rhs.end_),
        events_(rhs.events_),
        resolution_(rhs.resolution_) {}

  void PointProcess::check_endpoints(const DateTime &begin,
                                     const DateTime &end) const {
    if (begin > end) {
      ostringstream err;
      err << "The end of a PointProcess must not be before the beginning:"
          << endl
          << "begin = " << begin << endl
          << "end   = " << end << endl;
      report_error(err.str());
    }
  }

  void PointProcess::check_legal_event_number(int i) const {
    if (events_.empty() || i < 0 || i >= events_.size()) {
      ostringstream err;
      err << "An illegal event number " << i
          << " was passed to a PointProcess containing " << events_.size()
          << " events." << endl;
      report_error(err.str());
    }
  }

  void PointProcess::check_event_inside_window(const DateTime &event) const {
    if (event < begin_ || event > end_) {
      ostringstream err;
      err << "The event at time " << event << " is not inside the observation "
          << "window for the process." << endl
          << "[" << begin_ << ", " << end_ << "]" << endl;
      report_error(err.str());
    }
  }

  void PointProcess::check_events_inside_window(const DateTime &begin,
                                                const DateTime &end) const {
    if (events_.empty()) return;
    if (events_[0] < begin) {
      ostringstream err;
      err << "The first event in a point process occurred before "
          << "the beginning of the observation window." << endl
          << "Beginning of observation window:  " << begin << endl
          << "Time of first event            :  " << events_[0] << endl;
      report_error(err.str());
    }

    if (events_.back() > end) {
      ostringstream err;
      err << "The final event in a point process occurred after the "
          << "end of the observation window." << endl
          << "Time of last event       :  " << events_.back() << endl
          << "End of observation window:  " << end << endl;
      report_error(err.str());
    }
  }

  PointProcess *PointProcess::clone() const { return new PointProcess(*this); }

  std::ostream &PointProcess::display(std::ostream &out) const {
    out << begin_ << "--- beginning of observation window" << endl;
    for (int i = 0; i < number_of_events(); ++i) {
      out << events_[i] << endl;
    }
    out << end_ << "--- end of observation window" << endl;
    return out;
  }

  uint PointProcess::number_of_events() const { return events_.size(); }

  double PointProcess::window_duration() const { return end_ - begin_; }

  const PointProcessEvent &PointProcess::event(int i) const {
    check_legal_event_number(i);
    return events_[i];
  }

  double PointProcess::arrival_time(int i) const {
    check_legal_event_number(i);
    if (i == 0) return events_[0].timestamp() - begin_;
    if (i == events_.size()) return end_ - events_.back().timestamp();
    return events_[i].timestamp() - events_[i - 1].timestamp();
  }

  PointProcess &PointProcess::append(const PointProcess &rhs) {
    events_.reserve(events_.size() + rhs.number_of_events());
    if (immediately_follows(rhs)) {
      begin_ = rhs.begin_;
      events_.insert(events_.begin(), rhs.events_.begin(), rhs.events_.end());
    } else if (immediately_precedes(rhs)) {
      end_ = rhs.end_;
      events_.insert(events_.end(), rhs.events_.begin(), rhs.events_.end());
    } else {
      ostringstream err;
      err << "You can only append point processes that adjoin "
          << "the current process." << endl
          << "The current process covers " << begin_ << " to " << end_ << endl
          << "The process you're trying to append covers from " << rhs.begin_
          << " to " << rhs.end_ << endl;
      report_error(err.str());
    }
    return *this;
  }

  bool PointProcess::immediately_follows(const PointProcess &rhs) const {
    return fabs(begin_ - rhs.end_) < resolution_;
  }

  bool PointProcess::immediately_precedes(const PointProcess &rhs) const {
    return rhs.immediately_follows(*this);
  }

  bool PointProcess::adjoins(const PointProcess &rhs) const {
    return immediately_follows(rhs) || immediately_precedes(rhs);
  }

  PointProcess &PointProcess::add_event(const PointProcessEvent &event) {
    check_event_inside_window(event.timestamp());
    events_.insert(std::lower_bound(events_.begin(), events_.end(), event),
                   event);
    return *this;
  }
  PointProcess &PointProcess::add_event(const DateTime &event) {
    PointProcessEvent e(event);
    return add_event(e);
  }

  PointProcess &PointProcess::add_event(const DateTime &event,
                                        const Ptr<Data> &mark) {
    PointProcessEvent e(event, mark);
    return add_event(e);
  }

  const DateTime &PointProcess::window_begin() const { return begin_; }

  const DateTime &PointProcess::window_end() const { return end_; }

  void PointProcess::set_window(const DateTime &begin, const DateTime &end) {
    check_endpoints(begin, end);
    check_events_inside_window(begin, end);
    begin_ = begin;
    end_ = end;
  }

  void PointProcess::set_window_end(const DateTime &end) {
    check_endpoints(begin_, end);
    check_events_inside_window(begin_, end);
    end_ = end;
  }

  void PointProcess::set_window_begin(const DateTime &begin) {
    check_endpoints(begin, end_);
    check_events_inside_window(begin, end_);
    begin_ = begin;
  }

  void PointProcess::set_resolution(double resolution) {
    if (resolution <= 0) {
      report_error("resolution must be greater than zero\n");
    }
  }

}  // namespace BOOM
