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

#ifndef BOOM_RINTERFACE_CREATE_MIXTURE_COMPONENT_HPP_
#define BOOM_RINTERFACE_CREATE_MIXTURE_COMPONENT_HPP_

#include <r_interface/boom_r_tools.hpp>
#include <r_interface/list_io.hpp>

#include <Models/ModelTypes.hpp>
#include <vector>

//======================================================================
// Note that the functions listed here throw exceptions.  Code that
// uses them should be wrapped in a try-block where the catch
// statement catches the exception and calls Rf_error() with an
// appropriate error message.  The functions handle_exception(), and
// handle_unknown_exception (in handle_exception.hpp), are suitable
// defaults.  These try-blocks should be present in any code called
// directly from R by .Call.
//======================================================================

namespace BOOM{
  namespace RInterface{
    // Creates a vector of 'MixtureComponent's based on R data
    // structures.  Each component is replicated 'state_space_size'
    // times, so that the result is suitable for passing to (e.g.) a
    // HiddenMarkovModel constructor.  Each MixtureComponent already
    // has a prior set.  The rmixture_components argument contains
    // data, which should be extracted separately using ExtractMixtureData.
    //
    // Args:
    //   rmixture_components:  An R list of class "CompositeMixtureComponent".
    //   state_space_size:  Size of the state space for the latent Markov chain.
    //   io_manager: A BOOM RListIoManager responsible for recording the
    //     MCMC output.
    //
    // Returns:
    //   A vector of MixtureComponents of length state_space_size,
    //   suitable for passing to the HiddenMarkovModel constructor.  Each
    //   element will have a PosteriorSampler assigned, but no data.  Each
    //   MixtureComponent has an actual type of BOOM::CompositeModel,
    //   which inherits from BOOM::MixtureComponent.
    std::vector<Ptr<MixtureComponent> > UnpackCompositeMixtureComponents(
        SEXP rmixture_components,
        int state_space_size,
        RListIoManager *io_manager);

    // Creates an associative container of named mixture components,
    // indexed by the component names.  Each component is replicated a
    // number of times equal to the length of
    // 'mixture_component_names'.  Each MixtureComponent will have its
    // prior set, but it will contain no data..  The
    // rmixture_components argument contains data, which should be
    // extracted separately using ExtractMixtureData.
    //
    // Args:
    //   rmixture_components: An R list of class
    //     "CompositeMixtureComponent" containing the specification
    //     for the mixture component to be created.
    //   mixture_component_names: A vector of strings naming each
    //     mixture component.
    //   io_manager: A BOOM RListIoManager responsible for recording the
    //     MCMC output.
    //
    // Returns:
    //   A map, keyed on mixture_component_names, containing BOOM
    //   pointers to the newly created mixture components.  each
    //   component will have a corresponding set of entries in
    //   io_manager.
    std::map<std::string, Ptr<MixtureComponent> >
    UnpackNamedCompositeMixtureComponents(
        SEXP rmixture_components,
        const std::vector<std::string> & mixture_component_names,
        RListIoManager *io_manager);

    // Creates a single mixture component that can be used as a part
    // of a composite model.  The prior distribution and posteriorior
    // sampler are set, but no data is assigned.  The names for
    // different components are determined by appending a suffix to
    // the base component name.
    // Args:
    //   mixture_component: An R list containing all the necessary
    //     information to create the MixtureComponent, including a string
    //     named 'type'.  This object should have been created using one
    //     of the R constructors found in ../R/create.mixture.components.R.
    //   state_number: An integer to be used in naming the parameters in
    //     the io_manager.
    //   io_manager:  An RListIoManager responsible for recording MCMC output.
    // Returns:
    //   A BOOM smart pointer to a MixtureComponent of the appropriate
    //   type.  This is a factory function that determines the derived
    //   type of the output from a string named 'type' contained in the
    //   'mixture_component' argument.
    BOOM::Ptr<BOOM::MixtureComponent> CreateMixtureComponent(
        SEXP mixture_component,
        int state_number,
        BOOM::RListIoManager *io_manager);

    // Creates a single mixture component that can be used as a part
    // of a composite model.  The prior distribution and posteriorior
    // sampler are set, but no data is assigned.  The names for
    // different components are determined by prefixing a string to
    // the base component name.
    // Args:
    //   mixture_component: An R list containing all the necessary
    //     information to create the MixtureComponent, including a string
    //     named 'type'.  This object should have been created using one
    //     of the R constructors found in ../R/create.mixture.components.R.
    //   component_name_prefix: A string to be prepended to the name of each
    //     mixture component's parameters.
    //   io_manager:  An RListIoManager responsible for recording MCMC output.
    // Returns:
    //   A BOOM smart pointer to a MixtureComponent of the appropriate
    //   type.  This is a factory function that determines the derived
    //   type of the output from a string named 'type' contained in the
    //   'mixture_component' argument.
    BOOM::Ptr<BOOM::MixtureComponent> CreateNamedMixtureComponent(
        SEXP mixture_component,
        const std::string& component_name_prefix,
        BOOM::RListIoManager *io_manager);

    // Extracts all the data from all the subjects in the given
    // rmixture_component.  Note that this function gets data from a
    // SINGLE component.  It is used to implement
    // ExtractDataFromMixtureComponentList.
    // Args:
    //   rmixture_component: An R object with named elements 'data' and
    //     'data.type'.  'data' is a list containing a time series of data
    //     structured in a way appropriate to the type described by the R
    //     text string 'data.type.'
    // Returns:
    //   A vector of time series, each time series is represented as a
    //   vector of Ptr's to the abstract Data type.  Each time series
    //   corresponds to one entry in 'data'.
    std::vector<std::vector<BOOM::Ptr<BOOM::Data> > >
    ExtractMixtureComponentData(SEXP rmixture_component);

    // Extracts data from a list of mixture components supplied by R.
    // Args:
    //   rmixture_component_list: A list of MixtureComponent objects
    //     passed by R.
    // Returns:
    //   A vector of vectors of Data pointers.  Each Data pointer is
    //   to a CompositeData object suitable for passing to a
    //   CompositeModel used as the basis for a FiniteMixtureModel or
    //   HMM.  The indexing is such that answer[i][j] is observation j
    //   for subject i.
    std::vector<std::vector<BOOM::Ptr<BOOM::Data> > >
    ExtractCompositeDataFromMixtureComponentList(SEXP rmixutre_component_list);

    // Args:
    //   rknown_source: An R vector indicating which mixture component
    //     each individual observation belongs to, or NA if this
    //     information is unavailable.
    // Returns:
    //   A vector of integers containing the data in rknown_source,
    //   with NA's replaced by -1's.
    std::vector<int> UnpackKnownDataSource(SEXP rknown_source);

  }  // namespace RInterface
}  // namespace BOOM

#endif // BOOM_RINTERFACE_CREATE_MIXTURE_COMPONENT_HPP_
