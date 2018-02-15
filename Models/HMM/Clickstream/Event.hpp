#ifndef BOOM_CLICKSTREAM_EVENT_HPP
#define BOOM_CLICKSTREAM_EVENT_HPP

#include "Models/MarkovModel.hpp"

namespace BOOM {
  namespace Clickstream{

    // Having a dedicated "event" class is kind of overkill at the
    // mooment.  In the future I may add support for modeling
    // inter-event times, which would distinguish an Event from a
    // MarkovData.  Presently, however, an Event is a glorified
    // typedef.
    class Event : public MarkovData{
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

      Event * clone()const;

    };

  }  // namespace Clickstream
}  // namespace BOOM
#endif// BOOM_CLICKSTREAM_EVENT_HPP
