// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2013 Steven L. Scott

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

#ifndef BOOM_POISSON_BART_HPP_
#define BOOM_POISSON_BART_HPP_

// This code is untested.  TODO:  test it.

#include "Models/Bart/Bart.hpp"
#include "Models/Glm/PoissonRegressionData.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/NonparametricParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  class PoissonBartModel : public BartModelBase,
                           public NonparametricParamPolicy,
                           public IID_DataPolicy<PoissonRegressionData>,
                           public PriorPolicy {
   public:
    // Args:
    //   number_of_trees:  The number of trees to use in the model.
    //   initial_prediction: The model is initialized to predict this
    //     constant (as the log of the mean) for all x.
    explicit PoissonBartModel(int number_of_trees,
                              double initial_prediction = 0.0);

    // Args:
    //   number_of_trees:  The number of trees to use in the model.
    //   responses: The response vector to be modeled.  If the average
    //     response is greater than zero, then the log of the mean
    //     response is set as the initial value predicted by the
    //     model.  Otherwise the initial prediction is set to zero.
    //   predictors: A matrix of predictors used to predict
    //     'responses'.  The number of rows in 'predictors' must match
    //     the length of 'responses'.
    PoissonBartModel(int number_of_trees, const std::vector<int> &responses,
                     const Matrix &predictors);

    // Args:
    //   number_of_trees:  The number of trees to use in the model.
    //   responses:  The response vector to be modeled.  If the average
    //     response is greater than zero, then the log of the mean
    //     response is set as the initial value predicted by the
    //     model.  Otherwise the initial prediction is set to zero.
    //   exposures: A vector of positive real values indicating the
    //     exposure time (or exposure count) for each observation.
    //   predictors: A matrix of predictors used to predict
    //     'responses'.  The number of rows in 'predictors' must match
    //     the length of 'responses'.
    PoissonBartModel(int number_of_trees, const std::vector<int> &responses,
                     const std::vector<double> &exposures,
                     const Matrix &predictors);
    PoissonBartModel(const PoissonBartModel &rhs);
    PoissonBartModel *clone() const override;
    int sample_size() const override;
    void add_data(const Ptr<Data> &) override;
    void add_data(const Ptr<PoissonRegressionData> &) override;

   private:
    // Sets the model to predict log_lambda for all predictor values.
    void set_constant_prediction(double log_lambda);
  };

}  // namespace BOOM
#endif  //  BOOM_POISSON_BART_HPP_
