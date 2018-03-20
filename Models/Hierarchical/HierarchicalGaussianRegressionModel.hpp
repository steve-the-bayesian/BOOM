// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#ifndef BOOM_HIERARCHICAL_GAUSSIAN_REGRESSION_MODEL_HPP_
#define BOOM_HIERARCHICAL_GAUSSIAN_REGRESSION_MODEL_HPP_

#include "Models/Glm/RegressionModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  //  A model for nested regression data.  Each group of data has its own
  //  regression coefficients, which are viewed as draws from a common
  //  multivariate normal prior.  All groups share a common residual variance
  //  parameter.  In equations...
  //
  //    y[i, g] ~ N(beta[g] * (x[i, g]), sigma^2)
  //    beta[g] ~ Mvn(mu, V)
  class HierarchicalGaussianRegressionModel : public CompositeParamPolicy,
                                              public PriorPolicy {
   public:
    // Args:
    //   prior: The distribution describing how regression coefficients differ
    //     across groups.  The posterior sampler for the prior should be set
    //     before passing it to this constructor (or it can be set by an
    //     externally held pointer).
    //   residual_variance: The common residual variance parameter for the
    //     group-level regression models.
    explicit HierarchicalGaussianRegressionModel(
        const Ptr<MvnModel> &prior,
        const Ptr<UnivParams> &residual_variance = new UnivParams(1.0));

    HierarchicalGaussianRegressionModel(
        const HierarchicalGaussianRegressionModel &rhs);
    HierarchicalGaussianRegressionModel *clone() const override;

    // Data policy functions.  Data is stored by models.  In this setup,
    // add_model and add_data are pretty similar.  Calling add_data causes a new
    // regression model to be created to store the data.
    void add_model(const Ptr<RegressionModel> &model);
    void add_data(const Ptr<Data> &dp) override;
    void add_data(const Ptr<RegSuf> &suf);

    // Add new regression data to a particular model.
    void add_regression_data(const Ptr<RegressionData> &data, int group);

    // Clears the subordinate regression models holding the data, and removes
    // their parameters from the global list of model parameters.
    void clear_data() override;

    // Clears the data from the prior and the group level models.  Does not
    // delete the group level models themselves.
    void clear_data_keep_models();

    // Copies the sufficient statistics from other_model into this model.
    void combine_data(const Model &other_model, bool just_suf = true) override;

    int number_of_groups() const { return groups_.size(); }
    int xdim() const { return prior_->dim(); }

    RegressionModel *data_model(int which_group) {
      return groups_[which_group].get();
    }
    const RegressionModel *data_model(int which_group) const {
      return groups_[which_group].get();
    }

    MvnModel *prior() { return prior_.get(); }
    const MvnModel *prior() const { return prior_.get(); }

    double residual_variance() const { return residual_variance_->value(); }
    double residual_sd() const { return sqrt(residual_variance()); }
    void set_residual_variance(double sigsq) { residual_variance_->set(sigsq); }

   private:
    // Reset the list of model parameters managed by the ParamPolicy to those
    // from the prior, and the residual variance.
    void initialize_param_policy();

    std::vector<Ptr<RegressionModel>> groups_;
    Ptr<MvnModel> prior_;
    Ptr<UnivParams> residual_variance_;
  };

}  // namespace BOOM

#endif  //  BOOM_HIERARCHICAL_GAUSSIAN_REGRESSION_MODEL_HPP_
