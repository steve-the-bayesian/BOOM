// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_AGGREGATED_REGRESSION_MODEL_HPP_
#define BOOM_AGGREGATED_REGRESSION_MODEL_HPP_

#include "Models/DataTypes.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

#include <memory>

namespace BOOM {
  namespace Agreg {
    // A Transformation is intended to move the response variable onto the
    // Gaussian (normal) scale.  In an ordinary (non-aggregated)
    // regression model one would transform the data before modeling.  In
    // the aggregated regression setting we model z = f(y) as N(mu,
    // sigma), but we need to keep track of both y and z because the sum
    // of the y's is constrained to a particular value.  We also need to
    // keep track of the Jacobian of the transformation because it is
    // required by the MCMC algorithm.
    class Transformation {
     public:
      virtual ~Transformation() {}

      // The tranformation to normality: z = f(y) ~ N(mu, sigma)
      virtual double operator()(double y) const = 0;

      // The inverse transformation: y = f.inverse(z)
      virtual double inverse(double z) const = 0;

      // Log of the derivative of the inverse transformation: log (dz/dy).
      // Note that the argument is on the original un-transformed scale: y
      // = f(z)
      virtual double log_jacobian(double y) const = 0;

      virtual std::string name() const = 0;
    };

    class LogTransformation : public Transformation {
     public:
      double operator()(double y) const override { return log(y); }
      double inverse(double z) const override { return exp(z); }
      // Jacobian is 1/y, so log is -log(y)
      double log_jacobian(double y) const override { return -log(y); }
      std::string name() const override { return "log"; }
    };

    class SquareRootTransformation : public Transformation {
     public:
      double operator()(double y) const override { return sqrt(y); }
      double inverse(double z) const override { return z * z; }
      // Jacobian is .5/sqrt(y), so log is log(.5) - .5*log(y)
      double log_jacobian(double y) const override {
        return -0.693147180559945 - .5 * log(y);
      }
      std::string name() const override { return "sqrt"; }
    };

    class IdentityTransformation : public Transformation {
     public:
      double operator()(double x) const override { return x; }
      double inverse(double y) const override { return y; }
      double log_jacobian(double y) const override { return 0; }
      std::string name() const override { return ""; }
    };

    //======================================================================
    // A Group a collection of individual units, with a group-level
    // valuation (price) but individual unit-level predictors.  In
    // addition to storing the data, the Group knows how to randomly
    // apportion the total value amongs its units given regression
    // coefficients and standard deviation.
    class Group : public Data {
     public:
      Group(const std::string &name, double total_value,
            const Transformation &f);
      Group(const Group &rhs);

      // Virtual functions required by Data
      Group *clone() const override;
      std::ostream &display(std::ostream &out) const override;
      virtual uint size(bool minimal = true) const;

      // Add a new unit to an existing Group.  The RegressionData has two
      // data elements: x() and y().  The vector of predictors in x() is
      // immutable.  The scalar response variable y() will change with
      // each MCMC iteration.  The initial value of y is not relevant.
      void add_unit(const Ptr<RegressionData> &dp);

      // Distribute total value among units given current unit valuations,
      // as well as the transformed-regression model:
      // f(unit_price) = beta.dot(x) + N(0, sigma).
      //
      // The MCMC strategy is to choose a random match for each unit then
      // use slice sampling to allocate the sum of the two unit values
      // between them.
      //
      // The full conditional distribution is proportional to f_finv(normal(y1,
      // mu1, sigma)) * finv(normal(sum - y1, mu2, sigma)), where muj =
      // beta.dot(x[j])
      void distribute_total(const Vector &beta, double sigma);

      // Call this method once all the information about a group has
      // been read in, but before calling distribute_total.  Ensures
      // that implementation data is correctly sized, and sets each
      // unit to the average unit value.
      void initialize_unit_values();

     private:
      // Reapportion the values of units [i] and [j] so that the total
      // value of the group is maintained.  This method is used to
      // implement distribute_total.
      void modify_unit_value(int which_unit_1, int which_unit_2);

      // Name of the group in the input data
      std::string name_;

      // Total value of all the units in the group
      double total_value_;

      // Predictor variables and unit values for each unit in the group.
      // Predictors are immutable, unit values change with each call to
      // distribute_total.
      std::vector<Ptr<RegressionData> > unit_data_;

      // Workspace used for unit value calculation.  The invariant is
      // unit_values_[i] == exp(unit_data_[i]->y());
      Vector unit_values_;

      // local storage simplifies the interface to ModifyUnitValue
      const Vector *beta_;
      double sigma_;

      // Transformation to normality.
      const Transformation &f;
    };
  }  // namespace Agreg

  //======================================================================
  // The model is that individual units are indpendently transformed
  // normal with f(y[i])~ N(beta^Tx_i, sigma).
  class AggregatedRegressionModel : public CompositeParamPolicy,
                                    public IID_DataPolicy<Agreg::Group>,
                                    public PriorPolicy {
   public:
    typedef Agreg::Group Group;

    AggregatedRegressionModel(const Matrix &design_matrix_,
                              const std::vector<std::string> &group_names,
                              const Vector &group_values,
                              const std::string &transformation);

    AggregatedRegressionModel(const AggregatedRegressionModel &rhs);
    AggregatedRegressionModel *clone() const override;

    const Vector &beta() const { return model_->Beta(); }
    double sigma() const { return model_->sigma(); }
    void set_beta(const Vector &beta);
    void set_sigma(double sigma);

    // Gibbs sampler step that redistributes each group's total amount
    // among the units in the group, according to the current value of
    // the model parameters.
    void distribute_group_totals();

    RegressionModel *regression_model() const { return model_.get(); }

   private:
    // Method used to implement the constructor.
    // Args:
    //   unit_level_predictors: a matrix of predictor variables,
    //     with one row per unit.
    //   group_names: a vector of strings giving name of the group to
    //     which each unit belongs.  Size must equal the number of
    //     rows in unit_level_predictors.
    //   group_values: a numeric vector giving the value of each group.
    //     There must be a one-to-one correspondence between the entries
    //     in group_names and group_values.  That is, if group_names[i] ==
    //     group_names[j] then group_values[i] should equal group_values[j].
    void initialize_groups(const Matrix &unit_level_predictors,
                           const std::vector<std::string> &group_names,
                           const Vector &group_values);

    // Create an appropriate transformation for use in the constructor.
    static Agreg::Transformation *create_transformation(
        const std::string &transformation);

    // Returns the index (in groups_) of the group with the given name and
    // value.  If not present, a new group with these values is added.
    int find_group(const std::string &group_name, double group_value);

    // Let the model's sufficient statistics know about changes that
    // have been made to individual unit responses.
    void refresh_suf();

    // The model assumes that f(y) ~ Normal(X * beta, sigma^2)
    std::unique_ptr<Agreg::Transformation> f_;
    const Agreg::Transformation &f;

    // The model is implemented in terms of a RegressionModel
    Ptr<RegressionModel> model_;

    // A map to assist with reverse-lookup of group names.
    std::map<std::string, int> group_positions_;
  };

}  // namespace BOOM
#endif  //  BOOM_AGGREGATED_REGRESSION_MODEL_HPP_
