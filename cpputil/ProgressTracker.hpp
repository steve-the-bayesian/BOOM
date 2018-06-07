/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_PROGRESS_TRACKER_CLASS_HPP
#define BOOM_PROGRESS_TRACKER_CLASS_HPP

#include "uint.hpp"
#include "cpputil/Ptr.hpp"
#include "cpputil/RefCounted.hpp"

namespace BOOM {
  class ProgressTracker : private RefCounted {
    string fname;
    ostream *msg_;
    uint nskip;
    uint n;
    string sep;
    bool owns_msg;
    void start(const string &prog_name);
    ProgressTracker(const ProgressTracker &) : RefCounted() {}

   public:
    // Write progress messages to a file named "msg" in directory dname.
    explicit ProgressTracker(const string &dname, uint nskip = 100, bool restart = false,
                    const string &prog_name = "",
                    bool keep_existing_msg = false);

    // Write progress messages to std::cout
    explicit ProgressTracker(uint nskip = 100, const string &prog_name = "");

    // Write progress to an arbitrary stream
    explicit ProgressTracker(ostream &out, uint nskip = 100,
                    const string &prog_name = "");
    ~ProgressTracker() override;
    ProgressTracker &operator++() {
      update();
      return *this;
    }
    ProgressTracker &operator++(int) {
      update();
      return *this;
    }
    void update();
    uint restart();
    void set_niter(uint n);

    ostream &msg();

    friend void intrusive_ptr_add_ref(ProgressTracker *m);
    friend void intrusive_ptr_release(ProgressTracker *m);
  };
  void intrusive_ptr_add_ref(ProgressTracker *m);
  void intrusive_ptr_release(ProgressTracker *m);
}  // namespace BOOM
#endif  // BOOM_PROGRESS_TRACKER_CLASS_HPP
