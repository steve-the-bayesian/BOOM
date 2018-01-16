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

#ifndef BOOM_PERMUTE_HPP
#define BOOM_PERMUTE_HPP
namespace BOOM{
  template<class T>
  void permute_std_vector(std::vector<T> &v, const std::vector<uint> &perm);

  template <class T>
  void permute_std_vector(std::vector<T> &v,
                          const std::vector<uint> &perm);{
    std::vector<T> w(v.size());
    for(uint i=0; i<v.size(); ++i) w[i] = v[perm[i]];
    std::swap(v,w); }

}
#endif //  BOOM_PERMUTE_HPP
