// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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
#ifndef BOOM_HIERARCHICAL_POISSON_REGRESSION_HPP_
#define BOOM_HIERARCHICAL_POISSON_REGRESSION_HPP_

#include "Models/Glm/PoissonRegressionModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {
  // A HierarchicalPoissonRegressionModel is a single level hierarchy
  // describing.  The idiom for working with a
  // HierarchicalPoissonRegressionModel is to create the model with
  // the top level MVN model for the Poisson regression coefficients.
  // Then you repeatedly add PoissonRegressionModel's that each
  // contain the data for individual groups.
  //
  // The model is y[i, j] ~ Poisson(exp(beta[i] * x[i, j])) with
  // beta[i] ~ Normal(mu, Sigma)
  //
  // TODO: Consider a parallel class with an
  // IndependentMvnModel instead of an MvnModel for the
  // data_parent_model.
  class HierarchicalPoissonRegressionModel : public CompositeParamPolicy,
                                             public PriorPolicy {
   public:
    explicit HierarchicalPoissonRegressionModel(
        const Ptr<MvnModel> &data_parent_model);
    HierarchicalPoissonRegressionModel(
        const HierarchicalPoissonRegressionModel &rhs);
    HierarchicalPoissonRegressionModel *clone() const override;

    void add_data_level_model(const Ptr<PoissonRegressionModel> &);

    // Required data policy virtual functions
    void clear_data() override;
    void combine_data(const Model &rhs, bool just_suf = true) override;
    void add_data(const Ptr<Data> &dp) override;

    int xdim() const;
    int number_of_groups() const;
    PoissonRegressionModel *data_model(int which_group);
    MvnModel *data_parent_model();

   private:
    // The group_model_ describes the variation in
    Ptr<MvnModel> data_parent_model_;
    std::vector<Ptr<PoissonRegressionModel> > data_level_models_;
  };

}  // namespace BOOM
#endif  //  BOOM_HIERARCHICAL_POISSON_REGRESSION_HPP_
