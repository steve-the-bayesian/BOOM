#ifndef BOOM_GP_HIERARCHICAL_GP_REGRESSION_MODEL_HPP_
#define BOOM_GP_HIERARCHICAL_GP_REGRESSION_MODEL_HPP_

/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "Models/Glm/Glm.hpp"
#include "Models/GP/GaussianProcessRegressionModel.hpp"
#include "Models/GP/kernels.hpp"
#include "Models/GP/GpMeanFunction.hpp"

#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

// A hierarchical Gaussian process regression model.  Data are part of "K"
// groups (perhaps from K different experiments).

namespace BOOM {

  // Regression data for a hierarchical regression model.  The 'y' variable may
  // be adjusted for other effects in the model (e.g. by subtracting off the
  // prior mean).
  class HierarchicalRegressionData : public RegressionData {
   public:

    // Args:
    //   y:  The observed response for this observation.
    //   x:  The Vector of predictors for this observation.
    //   group: An int indicating the element of the hierarchy to which the
    //     observation belongs.
    HierarchicalRegressionData(double y, const Vector &x, const std::string &group);

    // The original unadjusted response value.
    double original_y() const {return original_y_;}

    // Adjust the 'y' value stored by this data point by subtracting the
    // function argument from "original_y".
    void adjust_y(double value_to_subtract);

    // The group/experiement/stratum/etc to which the data point belongs.
    const std::string &group() const {return group_;}

    std::ostream &display(std::ostream &out) const override;

   private:
    double original_y_;
    std::string group_;
  };

  // A collection of Gaussian process regression models, each of which have a
  // shared GpRegression model mean function.
  //
  // Let (Xj, yj) be the matrix of predictors and the vector of responses for
  // group j.  Let Kj be the kernel function for model j, and let sigma^2j be
  // that model's residual variance parameter.
  //
  // Let X be the union of all the Xj variables.  The HGP model states that the
  // "prior" or "shared" mean f0 obeys
  //
  //          f0(X) ~ N(m0(X), K0(X))
  //
  // where K0 is a kernel function for the prior mean function.  The
  // group-specific mean fj obeys
  //
  //         fj(Xj) ~ N(f0(Xj), Kj(Xj))
  //
  // with the observed data from group j obeying
  //
  //         yj | fj(Xj) ~ N(fj(Xj), sigma^2 I).
  class HierarchicalGpRegressionModel
      : public CompositeParamPolicy,
        public PriorPolicy {
   public:
    explicit HierarchicalGpRegressionModel(
        const Ptr<GaussianProcessRegressionModel> &mean_function_model);
    HierarchicalGpRegressionModel(const HierarchicalGpRegressionModel &rhs);
    HierarchicalGpRegressionModel & operator=(const HierarchicalGpRegressionModel &rhs);
    HierarchicalGpRegressionModel(HierarchicalGpRegressionModel &&rhs);
    HierarchicalGpRegressionModel & operator=(HierarchicalGpRegressionModel &&rhs);
    HierarchicalGpRegressionModel * clone() const override;

    // Data Policy overrides

    // The same data point is added to both the relevant model and the prior.
    void add_data(const Ptr<HierarchicalRegressionData> &data_point);
    void add_data(const Ptr<Data> &dp) override;
    void clear_data() override;
    void combine_data(const Model &other_model, bool just_suf = true) override;

    // Adding and working with sub-models.

    // Add a model and a corresponding hierarchy node to the model hierarchy.
    // Args:
    //   model:  The model to add to the hierarchy.
    //   index: The name of the data group for which the model is responsible.
    //     This name must match the "group" method
    void add_model(const Ptr<GaussianProcessRegressionModel> &model,
                   const std::string &index="");
    size_t number_of_groups() const {return group_names_.size();}
    const std::vector<std::string> &group_names() const { return group_names_; }


    GaussianProcessRegressionModel *prior();
    const GaussianProcessRegressionModel *prior() const;

    GaussianProcessRegressionModel *data_model(const std::string &index);
    const GaussianProcessRegressionModel *data_model(const std::string &index) const;

    // The data set owned by the given model.  The data are stored in the same
    // order as the model.
    std::vector<Ptr<HierarchicalRegressionData>> &data_set(
        GaussianProcessRegressionModel *model);

   private:
    std::map<std::string, Ptr<GaussianProcessRegressionModel>> models_;
    std::vector<std::string> group_names_;

    Ptr<GaussianProcessRegressionModel> shared_mean_function_model_;
    Ptr<GpMeanFunction> shared_mean_function_param_;

    // As far as the component models are concerned, they hold RegressionData.
    // But in reality they hold HierarchicalRegressionData.
    std::map<GaussianProcessRegressionModel *,
             std::vector<Ptr<HierarchicalRegressionData>>> data_store_;

  };

}  // namespace BOOM

#endif  // BOOM_GP_HIERARCHICAL_GP_REGRESSION_MODEL_HPP_
