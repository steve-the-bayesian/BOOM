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

#include "Models/HMM/Clickstream/Session.hpp"
#include "cpputil/string_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace Clickstream {
    namespace {
      typedef TimeSeries<Event> TS;
    }

    Session::Session(const std::vector<Ptr<Event>> &v, bool add_eos_if_missing)
        : TS(v) {
      Ptr<Event> last = v.back();
      if (last->value() != last->nlevels() - 1 && add_eos_if_missing) {
        NEW(Event, eos)(last->nlevels() - 1, last);
        push_back(eos);
      }
      check_eos();
      set_links();
    }

    Session::Session(const Session &rhs) : Data(rhs), TS(rhs) { set_links(); }

    Session *Session::clone() const { return new Session(*this); }
    int Session::number_of_events_including_eos() const {
      return this->length();
    }
    Ptr<Event> Session::event(int i) { return (*this)[i]; }
    const Ptr<Event> Session::event(int i) const { return (*this)[i]; }

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

    void Session::set_links() {
      for (int i = 1; i < size(); ++i) {
        (*this)[i]->set_prev((*this)[i - 1].get());
      }
    }

  }  // namespace Clickstream
}  // namespace BOOM
