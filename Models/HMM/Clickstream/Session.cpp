#include <Models/HMM/Clickstream/Session.hpp>

#include <cpputil/string_utils.hpp>
#include <distributions.hpp>

namespace BOOM {
  namespace Clickstream{
    namespace {
      typedef TimeSeries<Event> TS;
    }

    Session::Session(const std::vector<Ptr<Event> > &v, bool add_eos_if_missing)
        : TS(v, true)
    {
      Ptr<Event> last = v.back();
      if (last->value() != last->nlevels() - 1
          && add_eos_if_missing) {
        NEW(Event, eos)(last->nlevels() - 1, last);
        add_1(eos);
      }
      check_eos();
    }

    Session::Session(const Session & rhs)
        : Data(rhs),
          TS(rhs)
    {
      set_links();
    }

    Session * Session::clone()const{return new Session(*this);}
    int Session::number_of_events_including_eos()const{return this->length();}
    Ptr<Event> Session::event(int i){ return (*this)[i]; }
    const Ptr<Event> Session::event(int i)const{ return (*this)[i]; }

    void Session::check_eos() {
      if (empty()) {
        return;
      } else {
        int eos = back()->nlevels() - 1;
        if (back()->value() != eos) {
          report_error("Final element of Session was not the EOS indicator.");
        }
        if (size() > 1) {
          for (int i = 0; i < size() - 1; ++i) {
            if ((*this)[i]->value() == eos) {
              ostringstream err;
              err << "Non-terminal Session element " << i << " is EOS.";
              report_error(err.str());
            }
          }
        }
      }
    }

  }  // namespace Clickstream
}  // namespace BOOM
