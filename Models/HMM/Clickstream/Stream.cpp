#include <Models/HMM/Clickstream/Stream.hpp>

#include <distributions.hpp>
#include <cpputil/string_utils.hpp>
#include <algorithm>
#include <iomanip>

namespace BOOM {
  namespace Clickstream{

    Stream::Stream(const std::vector<Ptr<Session> > &sessions)
        : sessions_(sessions)
    {
      if (sessions.empty()) {
        report_error("Cannot construct a Stream with an empty "
                     "vector of Sessions.");
      }
    }

    Stream * Stream::clone()const{return new Stream(*this);}

    //----------------------------------------------------------------------
    int Stream::number_of_page_categories_including_eos()const{
      if (sessions_.empty()) {
        return -1;
      }
      if (session(0)->empty()) {
        return -1;
      }
      return session(0)->event(0)->nlevels();
    }
    //----------------------------------------------------------------------
    int Stream::nsessions()const{
      return sessions_.size();
    }
    //----------------------------------------------------------------------
    std::vector<int> Stream::session_sizes()const{
      int number_of_sessions = nsessions();
      std::vector<int> ans(number_of_sessions);
      for(int i=0; i < number_of_sessions; ++i) {
        ans[i] = sessions_[i]->number_of_events_including_eos();
      }
      return ans;
    }
    //----------------------------------------------------------------------
    int Stream::number_of_events_including_eos()const{
      int ns = nsessions();
      int ans=0;
      for(int i=0; i<ns; ++i) {
        ans += sessions_[i]->number_of_events_including_eos();
      }
      return ans;
    }

    //----------------------------------------------------------------------
    const std::vector<Ptr<Session> > & Stream::sessions()const{return sessions_;}
    Ptr<Session> Stream::session(int i)const{ return sessions_[i]; }

  }  // namespace Clickstream
}  // namespace BOOM
