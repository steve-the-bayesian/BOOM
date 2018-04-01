// Copyright 2018 Google LLC. All Rights Reserved.
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

#ifndef BOOM_R_INTERFACE_CREATE_STATE_MODEL_HPP_
#define BOOM_R_INTERFACE_CREATE_STATE_MODEL_HPP_

#include <r_interface/list_io.hpp>
#include <Models/StateSpace/StateSpaceModelBase.hpp>
#include <functional>
#include <list>

//======================================================================
// Note that the functions listed here throw exceptions.  Code that
// uses them should be wrapped in a try-block where the catch
// statement catches the exception and calls Rf_error() with an
// appropriate error message.  The functions handle_exception(), and
// handle_unknown_exception (in handle_exception.hpp), are suitable
// defaults.  These try-blocks should be present in any code called
// directly from R by .Call.
//======================================================================

namespace BOOM {

  // Trend models
  class LocalLevelStateModel;
  class LocalLinearTrendStateModel;
  class SemilocalLinearTrendStateModel;
  class StudentLocalLinearTrendStateModel;
  class StaticInterceptStateModel;
  class ArStateModel;

  // Regression models
  class DynamicRegressionStateModel;
  class DynamicRegressionArStateModel;

  // Seasonal Models
  class MonthlyAnnualCycle;
  class SeasonalStateModel;
  class TrigStateModel;
  class HarmonicTrigStateModel;

  // Holiday models
  class Holiday;
  class RandomWalkHolidayStateModel;
  class RegressionHolidayStateModel;
  class HierarchicalRegressionHolidayStateModel;


  namespace RInterface {
    // A factory for creating state components for use with state
    // space models.
    class StateModelFactory {
     public:
      // Args:
      //   io_manager: A pointer to the object manaaging the R list that will
      //     record (or has already recorded) the MCMC output.  If a nullptr is
      //     passed then states will be created without IoManager support.
      //   model: The model that will receive the state to be added.  If a
      //     nullptr is passed then this factory can still call
      //     CreateStateModel(), but AddState and SaveFinalState() will do
      //     nothing.
      StateModelFactory(RListIoManager *io_manager,
                        ScalarStateSpaceModelBase *model);

      // Adds all the state components listed in
      // r_state_specification_list to the model.
      // Args:
      //   r_state_specification_list: An R list of state components
      //     to be added to the model.  This function intended to
      //     handle the state.specification argument in bsts.
      //   prefix: An optional prefix added to the name of each state
      //     component.
      void AddState(SEXP r_state_specification_list,
                    const std::string &prefix = "");

      // Save the final state (i.e. at time T) of the model for use
      // with prediction.  Do not call this function until after all
      // components of state have been added.
      // Args:
      //   final_state: A pointer to a Vector to hold the state.  This
      //     can be NULL if the state is only going to be recorded.
      //     If state is going to be read, then final_state must be
      //     non-NULL.  A non-NULL vector will be re-sized if it is
      //     the wrong size.
      //   list_element_name: The name of the final state vector in
      //     the R list holding the MCMC output.
      void SaveFinalState(
          BOOM::Vector *final_state = NULL,
          const std::string & list_element_name = "final.state");

      // Factory method for creating a StateModel based on inputs
      // supplied to R.  Returns a smart pointer to the StateModel that
      // gets created.
      // Args:
      //   r_state_component: The portion of the state.specification list (that
      //     was supplied to R by the user), corresponding to the state model
      //     that needs to be created.
      //   prefix: A prefix to be added to the name field of the
      //     r_state_component in the io_manager.
      // Returns:
      //   A Ptr to a StateModel that can be added as a component of
      //   state to a state space model.
      Ptr<StateModel> CreateStateModel(SEXP r_state_component,
                                       const std::string &prefix);

      // Create a BOOM::Holiday from the supplied R object.
      // Args:
      //   holiday_spec:  An R object inheriting from "Holiday".
      // Returns:
      //   A BOOM::Holiday corresponding to 'holiday_spec'.
      static Ptr<Holiday> CreateHoliday(SEXP holiday_spec);

     private:
      // Concrete implementations of CreateStateModel.

      LocalLevelStateModel *CreateLocalLevel(
          SEXP r_state_component, const std::string &prefix);
      LocalLinearTrendStateModel *CreateLocalLinearTrend(
          SEXP r_state_component, const std::string &prefix);
      SemilocalLinearTrendStateModel *CreateSemilocalLinearTrend(
          SEXP r_state_component, const std::string &prefix);
      StudentLocalLinearTrendStateModel *CreateStudentLocalLinearTrend(
          SEXP r_state_component, const std::string &prefix);
      StaticInterceptStateModel *CreateStaticIntercept(
          SEXP r_state_component, const std::string &prefix);
      ArStateModel *CreateArStateModel(
          SEXP r_state_component, const std::string &prefix);
      ArStateModel *CreateAutoArStateModel(
          SEXP r_state_component, const std::string &prefix);

      DynamicRegressionStateModel *CreateDynamicRegressionStateModel(
          SEXP r_state_component, const std::string &prefix);
      DynamicRegressionArStateModel *CreateDynamicRegressionArStateModel(
          SEXP r_state_component, const std::string &prefix);

      RandomWalkHolidayStateModel *CreateRandomWalkHolidayStateModel(
          SEXP r_state_component, const std::string &prefix);
      RegressionHolidayStateModel *CreateRegressionHolidayStateModel(
          SEXP r_state_component, const std::string &prefix);
      HierarchicalRegressionHolidayStateModel *
      CreateHierarchicalRegressionHolidayStateModel(
          SEXP r_state_component, const std::string &prefix);
      
      SeasonalStateModel *CreateSeasonal(
          SEXP r_state_component, const std::string &prefix);
      TrigStateModel *CreateTrigStateModel(
          SEXP r_state_component, const std::string &prefix);
      HarmonicTrigStateModel *CreateHarmonicTrigStateModel(
          SEXP r_state_component, const std::string &prefix);
      MonthlyAnnualCycle *CreateMonthlyAnnualCycle(
          SEXP r_state_component, const std::string &prefix);

      Ptr<UnivParams> GetResidualVarianceParameter();

      // A pointer to the object manaaging the R list that will record (or has
      // already recorded) the MCMC output.  This can be a nullptr if IoManager
      // support is not desired.
      RListIoManager *io_manager_;

      // The model that needs state added.  This can be a nullptr if the state
      // space model is managed externally.
      ScalarStateSpaceModelBase *model_;

      // Some state models (notably dynamic regression) introduce special
      // elements of state (like the paths of regression coefficients) that
      // should be placed in the io_manager after the state model parameters.
      // If any such state elements exist, they will be placed in storage by
      // CreateStateModel.  This function should be called after the last call
      // to CreateStateModel.
      void InstallPostStateListElements() {
        if (io_manager_) {
          for (int i = 0; i < post_state_list_elements_.size(); ++i) {
            io_manager_->add_list_element(post_state_list_elements_[i]);
          }
        }
        // The post state list elements will be empty after a call to this
        // function, whether or not io_manager_ is defined.
        post_state_list_elements_.clear();
      }

      std::vector<RListIoElement *> post_state_list_elements_;
    };

  }  // namespace RInterface
}  // namespace BOOM
#endif  // BOOM_R_INTERFACE_CREATE_STATE_MODEL_HPP_
