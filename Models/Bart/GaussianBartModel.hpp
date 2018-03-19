// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_GAUSSIAN_BART_MODEL_HPP_
#define BOOM_GAUSSIAN_BART_MODEL_HPP_
#include "Models/Bart/Bart.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {
  //======================================================================
  // The BART model is that a response variable y is the sum of the
  // contributions from many trees, plus a
  class GaussianBartModel : public ParamPolicy_1<UnivParams>,
                            public IID_DataPolicy<RegressionData>,
                            public PriorPolicy,
                            public BartModelBase {
   public:
    // An empty model with no data assigned.
    // Args:
    //   number_of_trees:  The number of trees to use.
    //   mean:  The model is initialized to predict this constant.
    explicit GaussianBartModel(int number_of_trees, double mean = 0.0);

    // A model with the specifed data assigned.
    // Args:
    //   number_of_trees:  The number of trees to use.
    //   y:  A vector of responses to model.
    //   x: A vector of predictors used to fit the model.  The number
    //      of rows in x must match the length of y.
    //
    // This constructor assigns the proper data types created from y
    // and x.  The model is initialized to a constant equal to the
    // mean of y.
    GaussianBartModel(int number_of_trees, const Vector &y, const Matrix &x);

    GaussianBartModel(const GaussianBartModel &rhs);
    GaussianBartModel *clone() const override;

    // The number of observations in the training data.
    int sample_size() const override;

    // An override for add_data is needed so that variable_summaries_
    // can be adjusted when new data is observed.
    void add_data(const Ptr<Data> &) override;
    void add_data(const Ptr<RegressionData> &) override;

    virtual double sigsq() const;
    void set_sigsq(double sigsq);
    Ptr<UnivParams> Sigsq_prm();
  };
}  // namespace BOOM

#endif  // BOOM_GAUSSIAN_BART_MODEL_HPP_
