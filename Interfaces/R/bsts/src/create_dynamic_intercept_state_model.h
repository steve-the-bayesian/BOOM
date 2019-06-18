#ifndef BSTS_CREATE_DYNAMIC_INTERCEPT_STATE_MODEL_H_
#define BSTS_CREATE_DYNAMIC_INTERCEPT_STATE_MODEL_H_
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
#include "Models/StateSpace/StateModels/StateModel.hpp"

namespace BOOM {

  // Forward declarations.

  // Host model
  class DynamicInterceptRegressionModel;

  // State models
  class DynamicInterceptLocalLevelStateModel;
  
  namespace bsts {

    class DynamicInterceptStateModelFactory : public StateModelFactoryBase {
     public:
      // Args:
      //   io_manager: A pointer to the object manaaging the R list that will
      //     record (or has already recorded) the MCMC output.  If a nullptr is
      //     passed then states will be created without IoManager support.
      explicit DynamicInterceptStateModelFactory(RListIoManager *io_manager)
          : StateModelFactoryBase(io_manager) {}

      void AddState(DynamicInterceptRegressionModel *model,
                    SEXP r_state_specification,
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
      void SaveFinalState(DynamicInterceptRegressionModel *model,
                          BOOM::Vector *final_state = nullptr,
                          const std::string &list_element_name = "final.state");
      
     private:
      Ptr<DynamicInterceptStateModel> CreateStateModel(
          DynamicInterceptRegressionModel *model,
          SEXP r_state_component,
          const std::string &prefix);

      DynamicInterceptLocalLevelStateModel *CreateDynamicLocalLevel(
          SEXP r_state_component,
          const std::string &prefix);

    };
    
  }  // namespace bsts
}  // namespace BOOM


#endif  // BSTS_CREATE_DYNAMIC_INTERCEPT_STATE_MODEL_H_

