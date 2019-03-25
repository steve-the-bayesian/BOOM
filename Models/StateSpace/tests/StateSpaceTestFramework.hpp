#ifndef BOOM_STATE_SPACE_MODEL_TEST_FRAMEWORK_HPP_
#define BOOM_STATE_SPACE_MODEL_TEST_FRAMEWORK_HPP_

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

#include "Models/StateSpace/tests/TestFrameworkBase.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "LinAlg/Vector.hpp"


namespace BOOM {
  namespace StateSpaceTesting {

    class StateSpaceTestFramework
        : public TestFrameworkBase<StateModel, ScalarStateSpaceModelBase> {
     public:
      explicit StateSpaceTestFramework(double observation_sd);
      
      void SimulateData(int time_dimension) override;
      void BuildModel() override;
      void CreateObservationSpace(int niter) override;
      void Burn(int burn) override {
        for (int i = 0; i < burn; ++i) model_->sample_posterior();
      }
      void RunMcmc(int niter) override;
      void Check() override;

      const Vector &data() const {return data_;}
      
     private:
      double observation_sd_;
      Ptr<StateSpaceModel> model_;
      Ptr<ChisqModel> residual_precision_prior_;
      Ptr<ZeroMeanGaussianConjSampler> residual_precision_sampler_;
      
      Vector data_;
      Vector sigma_obs_draws_;
    };
        
  } // namespace StateSpaceTesting
} // namespace BOOM 

#endif  // BOOM_STATE_SPACE_MODEL_TEST_FRAMEWORK_HPP_
