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

#ifndef BOOM_GAUSSIAN_LINEAR_BART_MODEL_HPP
#define BOOM_GAUSSIAN_LINEAR_BART_MODEL_HPP

#include "Models/Bart/GaussianBartModel.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"

namespace BOOM {
  // A GaussianLinearBartModel is a model relating y to x, where y is a
  // scalar and x is a vector, through the equation
  //
  //  y = beta %*% x + f(x) + epsilon
  //
  // where epsilon ~ N(0, sigma^2), and f(x) is the sum of BART trees
  // The usual prior on this model combines a spike and slab prior on
  // beta with a prior penalizing the number and complexity of trees
  // in f(x).
  class GaussianLinearBartModel : public CompositeParamPolicy,
                                  public IID_DataPolicy<RegressionData>,
                                  public PriorPolicy {
   public:
    GaussianLinearBartModel(int number_of_trees, int xdim);
    GaussianLinearBartModel(int number_of_trees, const Vector &y,
                            const Matrix &x);
    GaussianLinearBartModel(const GaussianLinearBartModel &rhs);
    GaussianLinearBartModel *clone() const override;

    void add_data(const Ptr<Data> &) override;
    void add_data(const Ptr<RegressionData> &) override;

    double predict(const Vector &x) const;
    double predict(const VectorView &x) const;
    double predict(const ConstVectorView &x) const;

    RegressionModel *regression();
    const RegressionModel *regression() const;

    GaussianBartModel *bart();
    const GaussianBartModel *bart() const;

   private:
    void Init();

    Ptr<RegressionModel> regression_;
    Ptr<GaussianBartModel> bart_;
  };

}  // namespace BOOM

#endif  //  BOOM_GAUSSIAN_LINEAR_BART_MODEL_HPP
