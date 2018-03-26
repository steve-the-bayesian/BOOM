/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_RINTERFACE_DETERMINE_NTHREADS_HPP_
#define BOOM_RINTERFACE_DETERMINE_NTHREADS_HPP_

#include <r_interface/boom_r_tools.hpp>
namespace BOOM{
  namespace RInterface{
    // If r_nthreads is an integer then that integer will be returned.
    // If it is an R NULL value then a call will be made to
    // 'hardware_concurrency()' to determine the number cores the
    // hardware can support.  If neither case is true an exception
    // will be thrown.
    int determine_nthreads(SEXP r_nthreads);
  }
}
#endif // BOOM_RINTERFACE_DETERMINE_NTHREADS_HPP_
