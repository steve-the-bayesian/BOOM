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

#ifndef BOOM_RINTERFACE_CREATE_POISSON_PROCESS_HPP_
#define BOOM_RINTERFACE_CREATE_POISSON_PROCESS_HPP_

#include <r_interface/list_io.hpp>
#include <Models/PointProcess/PoissonProcess.hpp>

namespace BOOM{
  namespace RInterface{
  // Factory method for creating a PoissonProcess from an associated R
  // object inheriting from class "PoissonProcessSpecification".
  // Args:
  //   r_poisson_process: An object inheriting from "PoissonProcess".
  //     The specific class of the object will determine which type of
  //     PoissonProcess gets instantiated.  The object has initial
  //     parameter values, a prior distribution, and the name to use
  //     in the list managed by io_manager.
  //   io_manager: A pointer to the object managing the R list that
  //     will record (or has already recorded) the MCMC output.
  //   process_name: The name of this process, to be used in the list
  //     managed by io_manager.
  // Returns:
  //   A PoissonProcess model of the appropriate type.  The model has
  //   a posterior sampling method assigned, but no data is assigned.
  Ptr<PoissonProcess> CreatePoissonProcess(SEXP r_poisson_process,
                                           RListIoManager *io_manager,
                                           const std::string &process_name);
  }
}
#endif // BOOM_RINTERFACE_CREATE_POISSON_PROCESS_HPP_
