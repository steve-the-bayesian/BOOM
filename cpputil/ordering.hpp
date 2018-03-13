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

#ifndef ORDERING_HPP
#define ORDERING_HPP

#include <BOOM.hpp>
#include <vector>
#include <map>

namespace BOOM{

   template <class T=string>
   class ordering{
     std::map<T,uint> ord;
   public:
     ordering(T[] arr);
     ordering(const std::vector<T> &);

     bool operator()(const T &arg1, const T &arg2)const;
     // return true if arg1 "<" arg2
   };


   //-----------------------------
   template<class T>
   ordering<T>::ordering(T[] arr){
     std::vector<T> v(arr, arr+sizeof(arr)/sizeof(arr[0]));
     for(uint i=0; i<v.size(); ++i) ord[v[i]] = i; }

   template<class T>
   ordering<T>::ordering(const std::vector<T>&v){
     for(uint i=0; i<v.size(); ++i) ord[v[i]] = i; }

   template <class T>
   bool ordering<T>::ordering
   (const T &arg1, const T & arg2)const{
     return ord[arg1] < ord[arg2]; }

}


#endif //ORDERING_HPP
