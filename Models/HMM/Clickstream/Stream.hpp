#ifndef CLICKSTREAM_STREAM_HPP
#define CLICKSTREAM_STREAM_HPP

#include  <string>

#include <Models/HMM/Clickstream/Session.hpp>

namespace BOOM {
  namespace Clickstream{
    // A Stream is the data type for a NestedHmm.  It contains a
    // sequence of Sessions.
    class Stream : public BOOM::Data{
     public:
      Stream(const std::vector<Ptr<Session> > &sessions);
      Stream * clone()const override;

      int nsessions()const;
      std::vector<int> session_sizes()const;
      int number_of_events_including_eos()const;  // sum(session_sizes());
      const std::vector<Ptr<Session> > & sessions()const;
      Ptr<Session> session(int i)const;

      int number_of_page_categories_including_eos()const;

     private:
      ostream & display(ostream &out)const override{return out;}
      int size(bool = true)const{return 0;}
      std::vector<Ptr<Session> > sessions_;
    };

  }  // namespace Clickstream
}  // namespace BOOM

#endif  // CLICKSTREAM_PERSON_HPP
