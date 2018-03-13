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

#include <r_interface/create_point_process.hpp>

#include <cpputil/DateTime.hpp>
#include <vector>

namespace BOOM {
  namespace RInterface {

  namespace {
    void extract_date_time(SEXP r_process, DateTime &start, DateTime &stop,
                           std::vector<DateTime> &events) {
      SEXP r_events = getListElement(r_process, "events");
      int n = Rf_length(r_events);
      events.reserve(n);
      double * pevents = REAL(r_events);

      BOOM::DateTime::TimeScale timescale = BOOM::DateTime::second_scale;
      for (int i = 0; i < n; ++i) {
        DateTime event(pevents[i], timescale);
        events.push_back(event);
      }

      double start_time = Rf_asReal(getListElement(r_process, "start"));
      start = DateTime(start_time, timescale);
      double end_time = Rf_asReal(getListElement(r_process, "end"));
      stop = DateTime(end_time, timescale);
    }
  }  // namespace

    Ptr<PointProcess> CreatePointProcess(SEXP r_process) {
      DateTime start, end;
      std::vector<DateTime> events;
      extract_date_time(r_process, start, end, events);
      return new PointProcess(start, end, events);
    }

    Ptr<PointProcess> CreatePointProcess(
        SEXP r_process, const std::vector<Ptr<Data> > &marks) {
      DateTime start, end;
      std::vector<DateTime> events;
      extract_date_time(r_process, start, end, events);
      return new PointProcess(start, end, events, marks);
    }
  }  // namespace RInterface
}  // namespace BOOM
