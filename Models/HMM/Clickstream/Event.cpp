#include <Models/HMM/Clickstream/Event.hpp>
#include <cpputil/math_utils.hpp>

namespace BOOM {
  namespace Clickstream{

    Event::Event(int event_type, const Ptr<CatKeyBase> &key)
        : MarkovData(event_type, key)
    {}

    Event::Event(int event_type, const Ptr<Event> &prev)
        : MarkovData(event_type, Ptr<MarkovData>(prev))
    {}

    Event * Event::clone()const{return new Event(*this);}

  }  // namespace Clickstream
}  // namespace BOOM
