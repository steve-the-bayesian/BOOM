#ifndef CLICKSTREAM_SESSION_HPP
#define CLICKSTREAM_SESSION_HPP

#include <Models/HMM/Clickstream/Event.hpp>
#include <Models/TimeSeries/TimeSeries.hpp>

namespace BOOM {
  namespace Clickstream{
    class Session : public TimeSeries<Event>{
     public:
      // Args:
      //   events:  A sequence of events comprising the session.
      //   add_eos_if_missing: If true, and if the last element of
      //     events is not the EOS indicator (assumed to be the final
      //     level), then an EOS event will be added to events.
      Session(const std::vector<Ptr<Event> > &events, bool add_eos_if_missing);

      Session(const Session & rhs);
      Session * clone() const override;
      int number_of_events_including_eos()const;
      Ptr<Event> event(int i);
      const Ptr<Event> event(int i)const;

     private:
      // Checks for two things:
      // 1) If any element but the last is equal to eos, or
      // 2) the last element is NOT equal to EOS
      // then an error is reported using report_error().
      void check_eos();
    };

  }  // namespace Clickstream
}  // namespace BOOM

#endif// CLICKSTREAM_SESSION_HPP
