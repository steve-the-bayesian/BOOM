// Copyright 2018 Google LLC. All Rights Reserved.
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
#include "Models/Sufstat.hpp"
#include <sstream>

namespace BOOM {
  Vector vectorize(const std::vector<Ptr<Sufstat> > &v, bool minimal) {
    Vector ans;
    for (uint i = 0; i < v.size(); ++i) {
      Vector tmp = v[i]->vectorize(minimal);
      ans.concat(tmp);
    }
    return ans;
  }

  void unvectorize(std::vector<Ptr<Sufstat> > &svec, const Vector &v,
                   bool minimal) {
    Vector::const_iterator it = v.begin();
    for (uint i = 0; i < svec.size(); ++i) {
      it = svec[i]->unvectorize(it, minimal);
    }
  }

  std::string Sufstat::print_to_string() const {
    std::ostringstream out;
    out << *this;
    return out.str();
  }
}  // namespace BOOM
