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
#include <r_interface/determine_nthreads.hpp>
#include <cpputil/report_error.hpp>

#ifdef NO_BOOST_THREADS
namespace BOOM {
  namespace RInterface{
    int determine_nthreads(SEXP r_nthreads) {
      return 1;
    }
  }
}

#else

#include <boost/thread/thread.hpp>  // TODO(stevescott) use std library equivalent when possible

namespace BOOM {
  namespace RInterface{
    int determine_nthreads(SEXP r_nthreads) {
      if (Rf_isInteger(r_nthreads)) {
        return Rf_asInteger(r_nthreads);
      } else if (Rf_isNull(r_nthreads)) {
        return boost::thread::hardware_concurrency();
      } else {
        report_error("r_nthreads must be an integer or NULL.");
      }
      return -1;
    }
  }  // namespace RInterface
}  // namespace BOOM

#endif  // NO_BOOST_THREADS
