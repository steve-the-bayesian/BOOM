#ifndef BOOM_STATE_SPACE_TEST_FRAMEWORK_BASE_HPP_
#define BOOM_STATE_SPACE_TEST_FRAMEWORK_BASE_HPP_

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

#include "Models/StateSpace/StateModels/test_utils/StateTestModule.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    // The abstract base class 
    class TestFrameworkBase {
     public:
      virtual ~TestFrameworkBase() {}
      void AddState(const StateModuleManager &state) {
        state_modules_ = state;
      }

      // Run mcmc for 'burn' iterations.
      virtual void Burn(int burn) = 0;
      
      void Test(int niter, int time_dimension, int burn = 0) {
        SimulateData(time_dimension);
        BuildModel();
        CreateObservationSpace(niter);
        if (burn > 0) {
          Burn(burn);
        }
        RunMcmc(niter);
        Check();
      }

     protected:
      StateModuleManager & state_modules() { return state_modules_; }
      const StateModuleManager & state_modules() const {return state_modules_;}
      
     private:
      virtual void SimulateData(int time_dimension) = 0;
      virtual void BuildModel() = 0;
      virtual void CreateObservationSpace(int niter) = 0;
      virtual void RunMcmc(int niter) = 0;
      virtual void Check() = 0;

      StateModuleManager state_modules_;
    };
        
  } // namespace StateSpaceTesting
} // namespace BOOM 

#endif  // BOOM_STATE_SPACE_TEST_FRAMEWORK_BASE_HPP_
