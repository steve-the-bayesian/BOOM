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

#ifndef BOOM_CLICKSTREAM_EVENT_HPP
#define BOOM_CLICKSTREAM_EVENT_HPP

#include "Models/MarkovModel.hpp"

namespace BOOM {
  namespace Clickstream {

    // Having a dedicated "event" class is kind of overkill at the
    // mooment.  In the future I may add support for modeling
    // inter-event times, which would distinguish an Event from a
    // MarkovData.  Presently, however, an Event is a glorified
    // typedef.
    class Event : public MarkovData {
     public:
      // Constructor for first event in a session.
      // Args:
      //   event_type:  An integer indicating the type of event.
      //   key:  A CatKey with labels describing the event types.
      Event(int event_type, const Ptr<CatKeyBase> &key);

      // Constructor for subsequent events.
      // Args:
      //   event_type:  An integer indicating the type of event.
      //   prev:  The previous event in the session.
      Event(int event_type, const Ptr<Event> &prev);

      Event *clone() const;
    };

  }  // namespace Clickstream
}  // namespace BOOM
#endif  // BOOM_CLICKSTREAM_EVENT_HPP
