#ifndef BOOM_R_INTERFACE_CREATE_SHARED_STATE_MODEL_HPP_
#define BOOM_R_INTERFACE_CREATE_SHARED_STATE_MODEL_HPP_
/*
  Copyright (C) 2005-2019 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "create_state_model.h"
#include "r_interface/list_io.hpp"
#include <Models/StateSpace/StateSpaceModelBase.hpp>
#include <Models/StateSpace/MultivariateStateSpaceRegressionModel.hpp>

//==============================================================================
// The functions declared here throw exceptions.  Code that uses them should be
// wrapped in a try-block where the catch statement catches the exception and
// calls Rf_error() with an appropriate error message.  The functions
// handle_exception(), and handle_unknown_exception (in handle_exception.hpp),
// are suitable defaults.  These try-blocks should be present in any code called
// directly from R by .Call.
// ==============================================================================

namespace BOOM {

  // Forward declarations.  

  // Host model.
  class MultivariateStateSpaceModelBase;
  
  // State models.  This list will grow over time as more models are added.
  class SharedLocalLevelStateModel;

  namespace bsts {

    // A factory for creating state components that are shared across multiple
    // time series.
    class SharedStateModelFactory : public StateModelFactoryBase {
     public:

      // Args:
      //   nseries:  The number of time series begin modeled.
      //   io_manager: A pointer to the object manaaging the R list that will
      //     record (or has already recorded) the MCMC output.  If a nullptr is
      //     passed then states will be created without IoManager support.
      explicit SharedStateModelFactory(int nseries,
                                       RListIoManager *io_manager)
          : StateModelFactoryBase(io_manager),
            nseries_(nseries)
      {}

      using SharedStateModelVector =
          StateSpaceUtils::StateModelVector<SharedStateModel>;
      
      // Adds all the state components listed in
      // r_state_specification_list to the model.
      // Args:
      //   model: The model to which the state will be added.  
      //   state_models: The state model vector holding the shared state models.
      //     This is typically owned by 'model'.
      //   r_state_specification_list: An R list of state components to be added
      //     to the model.  This function intended to handle the state
      //     specification argument in bsts.
      //   prefix: An optional prefix added to the name of each state component.
      void AddState(SharedStateModelVector &state_models,
                    MultivariateStateSpaceModelBase *model,
                    SEXP r_shared_state_specification,
                    const std::string &prefix = "");

      // Save the final state (i.e. at time T) of the model for use with
      // prediction.  Do not call this function until after all components of
      // state have been added.
      // Args:
      //   model:  A pointer to the model that owns the state.
      //   final_state: A pointer to a Vector to hold the state.  This can be
      //     nullptr if the state is only going to be recorded.  If state is
      //     going to be read, then final_state must be non-NULL.  A non-NULL
      //     vector will be re-sized if it is the wrong size.
      //   list_element_name: The name of the final state vector in the R list
      //     holding the MCMC output.
      void SaveFinalState(
          MultivariateStateSpaceModelBase *model,
          BOOM::Vector *final_state = nullptr,
          const std::string &list_element_name = "final.shared.state");
      
     private:
      // The number of time series being modeled.
      int nseries_;
      
      // A factory function that unpacks information from an R object created by
      // AddXXX (where XXX is the name of a type of state model), and use it to
      // build the appropriate BOOM StateModel.  The specific R function
      // associated with each method is noted in the comments to the worker
      // functions that implement each specific type.
      //
      // Args:
      //   r_state_component:  The R object created by AddXXX.
      //   prefix: An optional prefix to be prepended to the name of the state
      //     component in the io_manager.
      //
      // Returns:
      //   A BOOM smart pointer to the appropriately typed MultivariateStateModel.
      Ptr<SharedStateModel> CreateSharedStateModel(
          MultivariateStateSpaceModelBase *model,
          SEXP r_state_component,
          const std::string &prefix);


      // Specific functions to create specific state models.
      Ptr<SharedStateModel> CreateSharedLocalLevel(
          SEXP r_state_component,
          MultivariateStateSpaceModelBase *model, 
          const std::string &prefix);
    };
    
  }  // namespace bsts
  
}  // namespace BOOM

#endif  // BOOM_R_INTERFACE_CREATE_SHARED_STATE_MODEL_HPP_

