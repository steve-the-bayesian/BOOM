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

#ifndef BOOM_RINTERFACE_CREATE_POINT_PROCESS_HPP_
#define BOOM_RINTERFACE_CREATE_POINT_PROCESS_HPP_

#include <Models/PointProcess/PointProcess.hpp>
#include <r_interface/boom_r_tools.hpp>

namespace BOOM{
  namespace RInterface{

    // Factory functions to create a PointProcess data structure based
    // on inputs from R.  The rpoint_process object is created by the
    // PointProcess constructor in the BoomEvents package.  It
    // contains POSIXt elements denoting the beginning and end of the
    // observation window for the point process, and a sorted vector
    // of POSIXt elements for the event time stamps.
    Ptr<PointProcess> CreatePointProcess(SEXP r_point_process);

    // As above, but associates a mark with each event.  The size of
    // marks must be the same as the number of events in
    // rpoint_process.
    Ptr<PointProcess> CreatePointProcess(SEXP r_point_process,
                                         const std::vector<Ptr<Data> > &marks);
  }
}
#endif // BOOM_RINTERFACE_CREATE_POINT_PROCESS_HPP_
