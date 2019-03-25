#ifndef BOOM_STATE_SPACE_TESTING_DYNAMIC_INTERCEPT_TEST_FRAMEWORK_HPP_
#define BOOM_STATE_SPACE_TESTING_DYNAMIC_INTERCEPT_TEST_FRAMEWORK_HPP_

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
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "Models/ChisqModel.hpp"
#include "LinAlg/SpdMatrix.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    class DynamicInterceptTestFramework
        : public TestFrameworkBase<DynamicInterceptStateModel,
                                   DynamicInterceptRegressionModel> {
     public:
      DynamicInterceptTestFramework(const Vector &coefficients,
                                    double observation_sd,
                                    double poisson_observation_rate);
      
      void SimulateData(int time_dimension) override;
      void BuildModel() override;
      void CreateObservationSpace(int niter) override;
      void Burn(int burn) override {
        for (int i = 0; i < burn; ++i) model_->sample_posterior();
      }
      void RunMcmc(int niter) override;
      void Check() override;
      
     private:
      Vector true_beta_;
      double observation_sd_;
      double poisson_rate_;
      
      Ptr<DynamicInterceptRegressionModel> model_;
      Ptr<ChisqModel> residual_precision_prior_;
      std::vector<Ptr<StateSpace::TimeSeriesRegressionData>> data_;

      SpdMatrix xtx_;
      double total_nobs_;
      
      Matrix coefficient_draws_;
      Vector sigma_obs_draws_;
      
    };
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM

#endif //  BOOM_STATE_SPACE_TESTING_DYNAMIC_INTERCEPT_TEST_FRAMEWORK_HPP_
