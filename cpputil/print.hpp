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
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
   USA
 */

#ifndef PRINT_BOOM_H
#define PRINT_BOOM_H

#include <iostream>
#include <list>

namespace BOOM {

  template <class T>
  std::ostream &print_list(std::ostream &out, std::list<T> &l) {
    out << "list.size(): " << l.size() << std::endl;
    if (l.empty()) {
      out << "empty list" << std::endl;
    } else
      for (typename std::list<T>::iterator it = l.begin(); it != l.end();
           ++it) {
        out << *it << std::endl;
      }
    return out;
  }

  template <class T>
  std::ostream &operator<<(std::ostream &out, std::list<T> &l) {
    print_list(out, l);
    return out;
  }
}  // namespace BOOM
#endif  // PRINT_BOOM_H
