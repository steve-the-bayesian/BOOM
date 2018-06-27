#ifndef BOOM_STATE_SPACE_STATE_TEST_MODULE_HPP_
#define BOOM_STATE_SPACE_STATE_TEST_MODULE_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    // A collection of objects used to test a specific StateModel.  Concrete
    // instances should included priors, samplers
    class StateModelTestModule {
     public:

      // Simulate the data for this state component.  After calling this method,
      // the data will be available by calling StateContribution().
      virtual void SimulateData(int time_dimension) = 0;
      virtual const Vector &StateContribution() const = 0;
      
      // Place the fully formed state model into model.
      virtual void ImbueState(StateSpaceModelBase &model) = 0;
      virtual void ImbueState(DynamicInterceptRegressionModel &model) = 0;

      // Before starting the MCMC algorithm, call CreateObservationSpace with
      // the desired number of MCMC iterations.  This will create a set of
      // Vector and Matrix objects to store the MCMC draws so they can be
      // checked after the run ends.
      virtual void CreateObservationSpace(int niter) = 0;
      
      // Record the current values of the state in the space created by
      // CreateObservationSpace.
      virtual void ObserveDraws(const StateSpaceModelBase &model) = 0;

      // Check the MCMC draws vs the true values used to create the simulated
      // data.
      virtual void Check() = 0;
    };

    //===========================================================================
    class StateModuleManager {
     public:
      void AddModule(StateModelTestModule *module) {
        modules_.emplace_back(module);
      }

      bool empty() const {return modules_.empty();}
      
      void SimulateData(int time_dimension) {
        for (auto &module : modules_) module->SimulateData(time_dimension); 
      }

      Vector StateContribution() const {
        Vector ans = modules_[0]->StateContribution();
        for (int i = 1; i < modules_.size(); ++i) {
          ans += modules_[i]->StateContribution();
        }
        return ans;
      }
      
      void ImbueState(StateSpaceModelBase &model) {
        for (auto &module : modules_) {
          module->ImbueState(model);
        }
      }

      void ImbueState(DynamicInterceptRegressionModel &model) {
        for (auto &module : modules_) {
          module->ImbueState(model);
        }
      }

      void CreateObservationSpace(int niter) {
        for (auto &module : modules_) {
          module->CreateObservationSpace(niter);
        }
      }

      void ObserveDraws(const StateSpaceModelBase &model) {
        for (auto &module : modules_) {
          module->ObserveDraws(model);
        }
      }

      void Check() {
        for (auto &module : modules_) {
          module->Check();
        }
      }
      
     private:
      std::vector<std::unique_ptr<StateModelTestModule>> modules_;
    };
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM 

#endif //  BOOM_STATE_SPACE_STATE_TEST_MODULE_HPP_



