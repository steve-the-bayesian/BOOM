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

#include<vector>
#include<cmath>

namespace BOOM{

  namespace SeqTmp{
    template<class T>
    bool inrange(const T & tmp, const T& from, const T& to){
      if(from < to) return (tmp <= to && tmp >=from);
      else if(from > to) return (tmp >=to && tmp <=from);
      return tmp==from && tmp==to;
    }

    template <class T>
    unsigned int dist(const T &from, const T &to, const T &by){
      bool inc(from < to);
      T tmp = inc ? (to-from)/by : (from-to)/by;
      return static_cast<unsigned int>(1u+ floor(tmp));
    }

  }
  template<class T>
  std::vector<T> seq(const T &from, const T& to, const T & by){
    using namespace SeqTmp;
    if(from==to) return std::vector<T>(1, from);
    bool incr(from < to);
    unsigned int len = dist(from, to, by);
    std::vector<T> ans;
    ans.reserve(len);
    T tmp= from;
    ans.push_back(tmp);
    bool ok=true;
    while(ok){
      if(incr) tmp+=by;
      else tmp-= by;
      ok = inrange(tmp, from, to);
      if(ok) ans.push_back(tmp);
    }
    return ans;
  }

  template<class T>
  std::vector<T> seq(const T &from, const T & to){
    T one(1);
    return seq(from, to, one);
  }

}
