// Copyright 2018 Google LLC. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

#ifndef CLICKSTREAM_STREAM_HPP
#define CLICKSTREAM_STREAM_HPP

#include <string>

#include "Models/HMM/Clickstream/Session.hpp"

namespace BOOM {
  namespace Clickstream {
    // A Stream is the data type for a NestedHmm.  It contains a
    // sequence of Sessions.
    class Stream : public BOOM::Data {
     public:
      explicit Stream(const std::vector<Ptr<Session> > &sessions);
      Stream *clone() const override;

      int nsessions() const;
      std::vector<int> session_sizes() const;
      int number_of_events_including_eos() const;  // sum(session_sizes());
      const std::vector<Ptr<Session> > &sessions() const;
      Ptr<Session> session(int i) const;

      int number_of_page_categories_including_eos() const;

     private:
      std::ostream &display(std::ostream &out) const override { return out; }
      int size(bool = true) const { return 0; }
      std::vector<Ptr<Session> > sessions_;
    };

  }  // namespace Clickstream
}  // namespace BOOM

#endif  // CLICKSTREAM_PERSON_HPP
