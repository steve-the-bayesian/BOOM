// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#ifndef CLICKSTREAM_SESSION_HPP
#define CLICKSTREAM_SESSION_HPP

#include "Models/HMM/Clickstream/Event.hpp"
#include "Models/TimeSeries/TimeSeries.hpp"

namespace BOOM {
  namespace Clickstream {
    class Session : public TimeSeries<Event> {
     public:
      // Args:
      //   events:  A sequence of events comprising the session.
      //   add_eos_if_missing: If true, and if the last element of
      //     events is not the EOS indicator (assumed to be the final
      //     level), then an EOS event will be added to events.
      Session(const std::vector<Ptr<Event> > &events, bool add_eos_if_missing);

      Session(const Session &rhs);
      Session *clone() const override;
      int number_of_events_including_eos() const;
      Ptr<Event> event(int i);
      const Ptr<Event> event(int i) const;

     private:
      // Checks for two things:
      // 1) If any element but the last is equal to eos, or
      // 2) the last element is NOT equal to EOS
      // then an error is reported using report_error().
      void check_eos();

      void set_links();
    };

  }  // namespace Clickstream
}  // namespace BOOM

#endif  // CLICKSTREAM_SESSION_HPP
