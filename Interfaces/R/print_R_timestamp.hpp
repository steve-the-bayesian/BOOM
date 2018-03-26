/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_R_INTERFACE_PRINT_TIMESTAMP_HPP_
#define BOOM_R_INTERFACE_PRINT_TIMESTAMP_HPP_
namespace BOOM {
  // print_R_timestamp(1000) prints message like the following on the
  // R terminal window
  // =-=-=-=-= Iteration 1000 Wed Aug 10 10:32:01 2011 =-=-=-=-=
  // Args:
  //   iteration_number: The iteration number of the algorithm whose
  //     progress is being tracked.
  //   ping: The desired frequency of status update messages.  If ping
  //     <= 0 then this function is a no-op.  Otherwise the update
  //     message is printed if 'iteration_number' is a multiple of
  //     'ping'.
  void print_R_timestamp(int iteration_number, int ping = 1);
}
#endif //  BOOM_R_INTERFACE_PRINT_TIMESTAMP_HPP_
