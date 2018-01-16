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
#include <r_interface/create_poisson_cluster_components.hpp>
#include <r_interface/create_poisson_process.hpp>

namespace BOOM{
  namespace RInterface{
    PoissonClusterComponentProcesses CreatePoissonClusterComponents(
        SEXP rpoisson_process_list,
        RListIoManager *io_manager) {
      Ptr<PoissonProcess> background = CreatePoissonProcess(
          getListElement(rpoisson_process_list, "background"),
          io_manager,
          "background");
      Ptr<PoissonProcess> primary_birth = CreatePoissonProcess(
          getListElement(rpoisson_process_list, "primary.birth"),
          io_manager,
          "primary.birth");
      Ptr<PoissonProcess> primary_traffic = CreatePoissonProcess(
          getListElement(rpoisson_process_list, "primary.traffic"),
          io_manager,
          "primary.traffic");
      Ptr<PoissonProcess> primary_death = CreatePoissonProcess(
          getListElement(rpoisson_process_list, "primary.death"),
          io_manager,
          "primary.death");
      Ptr<PoissonProcess> secondary_traffic = CreatePoissonProcess(
          getListElement(rpoisson_process_list, "secondary.traffic"),
          io_manager,
          "secondary.traffic");
      Ptr<PoissonProcess> secondary_death = CreatePoissonProcess(
          getListElement(rpoisson_process_list, "secondary.death"),
          io_manager,
          "secondary.death");

      PoissonClusterComponentProcesses components;
      components.background = background;
      components.primary_birth = primary_birth;
      components.primary_traffic = primary_traffic;
      components.primary_death = primary_death;
      components.secondary_traffic = secondary_traffic;
      components.secondary_death = secondary_death;
      return components;
    }
  }
}
