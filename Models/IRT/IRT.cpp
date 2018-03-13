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
#include "Models/IRT/IRT.hpp"
#include "Models/IRT/Item.hpp"
#include "Models/IRT/Subject.hpp"

namespace BOOM {
  namespace IRT {
    bool SubjectLess::operator()(const Ptr<Subject> &s1,
                                 const Ptr<Subject> &s2) const {
      return s1->id() < s2->id();
    }

    bool ItemLess::operator()(const Ptr<Item> &i1, const Ptr<Item> &i2) const {
      return i1->id() < i2->id();
    }

    void add_subject(SubjectSet &Sub, const Ptr<Subject> &s) {
      SubjectLess sl;
      SubjectSet::iterator it = std::lower_bound(Sub.begin(), Sub.end(), s, sl);
      if (it == Sub.end())
        Sub.push_back(s);
      else {
        Ptr<Subject> s2(*it);
        if (s2 != s) Sub.insert(it, s);
      }
    }

  }  // namespace IRT
}  // namespace BOOM
