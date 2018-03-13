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

#ifndef BOOM_RINTERFACE_CREATE_POISSON_CLUSTER_COMPONENTS_HPP_
#define BOOM_RINTERFACE_CREATE_POISSON_CLUSTER_COMPONENTS_HPP_

#include <Models/PointProcess/PoissonClusterProcess.hpp>
#include <r_interface/list_io.hpp>

namespace BOOM{
  namespace RInterface{
    // Extracts the set of PoissonProcess'es required to build a
    // PoissonClusterProcess.
    // Args:
    //   rpoisson_process_list: A list of R objects inheriting from
    //     "PoissonProcess".  Each must also inherit from a concrete
    //     type recognized by CreatePoissonProcess.  The full list of
    //     such types can be found in the file
    //     create_poisson_process.cpp, but the intent is to have an R
    //     class corresponding to each BOOM class that inherits from
    //     BOOM::PoissonProcess.  The objects in this list must have
    //     one of the following names: * background, primary.birth,
    //     primary.traffic, primary.death, secondary.traffic, secondary.death
    //   io_manager:  The RListIoManager in charge of storing MCMC draws.
    // Returns:
    //   A vector of PoissonProcess pointers suitable for use in the
    //   PoissonClusterProcess constructor.
    BOOM::PoissonClusterComponentProcesses CreatePoissonClusterComponents(
        SEXP rpoisson_process_list,
        RListIoManager *io_manager);
  }
}
#endif //  BOOM_RINTERFACE_CREATE_POISSON_CLUSTER_COMPONENTS_HPP_
