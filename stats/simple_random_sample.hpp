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

#ifndef BOOM_SIMPLE_RANDOM_SAMPLE_HPP
#define BOOM_SIMPLE_RANDOM_SAMPLE_HPP

#include <algorithm>
#include <map>
#include <vector>
#include "uint.hpp"
namespace BOOM {

  std::vector<bool> SRS_indx(unsigned int N, unsigned int n);

  template <class T, template <class> class Cont>
  std::vector<T> simple_random_sample(const Cont<T> &c, unsigned int n) {
    unsigned int N = c.size();
    std::vector<bool> in = SRS_indx(N, n);
    std::vector<T> ans(n);
    unsigned int i(0), I(0);
    typedef typename Cont<T>::const_iterator It;
    for (It it = c.begin(); it != c.end(); ++it) {
      if (in[i++]) ans[I++] = *it;
      if (I == n) break;
    }
    return ans;
  }

  template <class T, class Indx>
  std::vector<T> simple_random_sample(const std::map<Indx, T> &c,
                                      unsigned int n) {
    unsigned int N = c.size();
    std::vector<bool> in = SRS_indx(N, n);
    std::vector<T> ans(n);
    typedef typename std::map<Indx, T>::const_iterator mapit;
    unsigned int i(0), I(0);
    for (mapit it = c.begin(); it != c.end(); ++it) {
      if (in[i++]) ans[I++] = it->second;
      if (I == n) break;
    }
    return ans;
  }

}  // namespace BOOM
#endif  // BOOM_SIMPLE_RANDOM_SAMPLE_HPP
