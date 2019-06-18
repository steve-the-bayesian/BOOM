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

#include "Models/Glm/AggregatedRegressionModel.hpp"

#include <iomanip>
#include <map>

#include "LinAlg/SpdMatrix.hpp"
#include "Samplers/ScalarSliceSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace Agreg {
    Group::Group(const std::string &name, double value, const Transformation &F)
        : name_(name), total_value_(value), f(F) {}
    //----------------------------------------------------------------------
    Group::Group(const Group &rhs)
        : name_(rhs.name_), total_value_(rhs.total_value_), f(rhs.f) {}
    //----------------------------------------------------------------------
    Group *Group::clone() const { return new Group(*this); }
    //----------------------------------------------------------------------
    std::ostream &Group::display(std::ostream &out) const {
      out << "name        = " << name_ << endl
          << "total_value = " << total_value_ << endl;
      int n = unit_data_.size();
      if (n == 0) {
        out << "(no predictors)" << endl;
        return out;
      }

      int p = unit_data_[0]->x().size();
      Matrix X(n, p);
      for (int i = 0; i < n; ++i) {
        const Vector &x(unit_data_[i]->x());
        if (x.size() != p) {
          ostringstream err;
          err << "Error in BOOM::Agreg::Group::display().  Row " << i
              << " in Group " << name_
              << " had a different number of predictors (" << x.size()
              << ") than the first row, which had " << p << ".";
          report_error(err.str());
        }
        X.row(i) = x;
      }

      out << X;
      return out;
    }
    //----------------------------------------------------------------------
    uint Group::size(bool minimal) const {
      uint ans = unit_data_[0]->x().size() * unit_data_.size();
      return minimal ? ans : ans + 2;
    }
    //----------------------------------------------------------------------
    void Group::add_unit(const Ptr<RegressionData> &dp) {
      unit_data_.push_back(dp);
    }
    //----------------------------------------------------------------------
    void Group::distribute_total(const Vector &beta, double sigma) {
      if (unit_data_.size() <= 1) {
        unit_data_[0]->set_y(f(total_value_));
        return;
      }
      if (fabs(sum(unit_values_) - total_value_) > .01) {
        report_error("TODO:  need descriptive error here");
      }

      beta_ = &beta;
      sigma_ = sigma;
      for (int i = 0; i < unit_data_.size(); ++i) {
        // Draw j uniformly from the remaining indicies not equal to i.
        int j = random_int(0, unit_data_.size() - 2);
        if (j >= i) ++j;

        modify_unit_value(i, j);
        unit_data_[i]->set_y(f(unit_values_[i]));
        unit_data_[j]->set_y(f(unit_values_[j]));
      }
    }

    //----------------------------------------------------------------------
    // A class to be used as the target distribution for the
    // ScalarSliceSampler used to implement distribute_total.
    class UnitValueDistribution {
     public:
      UnitValueDistribution(double mu_i, double mu_j, double sigma, double sum,
                            const Transformation &F)
          : mui_(mu_i), muj_(mu_j), sigma_(sigma), sum_(sum), f(F) {}

      double logp(double y, double mu) const {
        return dnorm(f(y), mu, sigma_, true) + f.log_jacobian(y);
      }

      double operator()(double y) const {
        if (y >= sum_ || y <= 0.0) {
          return (BOOM::negative_infinity());
        }
        return logp(y, mui_) + logp(sum_ - y, muj_);
      }

     private:
      double mui_, muj_, sigma_, sum_;
      const Transformation &f;
    };

    //----------------------------------------------------------------------
    void Group::modify_unit_value(int i, int j) {
      //  if (i == 0) return;
      if (fabs(total_value_ - unit_values_.sum()) > .01) {
        ostringstream err;
        err << "In BOOM::Agreg::Group::modify_unit_value: total_value and "
            << "unit_values_ have gotten out of sync.";
        report_error(err.str());
      }

      double total = unit_values_[i] + unit_values_[j];
      // Total is the total amount of value to be split between the two
      // assets.  If total is really small then both assets are zero, and
      // won't change.
      //  if (total < .01) return;

      if (unit_values_[i] > total) {
        ostringstream err;
        err << "unit_values_[" << i << "] on group " << name_
            << " is greater than the maximum possible value of " << total
            << " sum(unit values_) = " << sum(unit_values_)
            << " total group value = " << total_value_ << endl
            << "inidividual unit_values: " << unit_values_;
        report_error(err.str());
      }

      if (unit_values_[i] < 0 || unit_values_[j] < 0) {
        ostringstream err;
        err << "unit_values_ must be positive:" << endl
            << "unit_values_[" << i << "] = " << unit_values_[i] << endl
            << "unit_values_[" << j << "] = " << unit_values_[j] << endl;
        report_error(err.str());
      }

      double mu_i = beta_->dot(unit_data_[i]->x());
      double mu_j = beta_->dot(unit_data_[j]->x());
      UnitValueDistribution logf(mu_i, mu_j, sigma_, total, f);
      ScalarSliceSampler sam(logf);
      sam.set_limits(0, total);

      // Keep values at least slightly away from the boundary.
      //     if (fabs(unit_values_[i] - total) <= .01) {
      //       unit_values_[i] = total - .01;
      //     }
      for (int k = 0; k < 3; ++k) {
        // The slice sampler had trouble moving off of bad starting
        // values.  Iterating the slice sampler a few times gives us
        // close-to-direct draws from the target distribution
        unit_values_[i] = sam.draw(unit_values_[i]);
        unit_values_[j] = total - unit_values_[i];
      }
      if (unit_values_[i] < 0 || unit_values_[i] > total ||
          unit_values_[j] < 0 || unit_values_[j] > total) {
        ostringstream err;
        err << "unit values must be non-negative, but less than their sum: "
            << total << endl
            << "unit_values_[" << i << "] = " << unit_values_[i] << endl
            << "unit_values_[" << j << "] = " << unit_values_[j] << endl;
        report_error(err.str());
      }
    }
    //----------------------------------------------------------------------
    void Group::initialize_unit_values() {
      unit_values_.resize(unit_data_.size());
      unit_values_ = total_value_ / unit_values_.size();
      if (fabs(unit_values_.sum() - total_value_) > .01) {
        report_error(
            "Agreg::Group::initialize_unit_values:  unit_values_ and "
            "total_value_ have gotten out of sync");
      }
      for (int i = 0; i < unit_data_.size(); ++i) {
        unit_data_[i]->set_y(f(unit_values_[i]));
      }
    }

  }  // namespace Agreg

  //======================================================================
  typedef AggregatedRegressionModel ARM;
  ARM::AggregatedRegressionModel(const Matrix &design_matrix,
                                 const std::vector<std::string> &group_names,
                                 const Vector &group_values,
                                 const std::string &transformation)
      : f_(create_transformation(transformation)),
        f(*(f_.get())),
        model_(new RegressionModel(ncol(design_matrix))) {
    // Build groups and compute f(ybar) in each group (needed to build
    // the prior distribution).
    initialize_groups(design_matrix, group_names, group_values);
    ParamPolicy::add_model(model_);
  }

  ARM::AggregatedRegressionModel(const ARM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        f_(create_transformation(rhs.f_->name())),
        f(*f_.get()),
        model_(rhs.model_->clone()) {
    ParamPolicy::add_model(model_);
  }

  //----------------------------------------------------------------------
  ARM *ARM::clone() const { return new ARM(*this); }

  //----------------------------------------------------------------------

  // Take the design matrix and a vector of group names and group
  // values.  Picks out the information for each group, and then creates
  // and stores a Group.
  void AggregatedRegressionModel::initialize_groups(
      const Matrix &X, const std::vector<std::string> &group_names,
      const Vector &group_values) {
    if (nrow(X) != group_names.size() || nrow(X) != group_values.size()) {
      ostringstream err;
      err << "The number of rows in the design matrix (" << nrow(X)
          << ") should match the length of the group_names vector ("
          << group_names.size()
          << ") and the length of the group_values vector ("
          << group_values.size() << ")." << endl;
      report_error(err.str());
    }

    for (int i = 0; i < group_names.size(); ++i) {
      std::string group_name = group_names[i];
      int pos = find_group(group_name, group_values[i]);
      const Ptr<RegressionData> &dp(new RegressionData(0, X.row(i)));
      dat()[pos]->add_unit(dp);
      model_->add_data(dp);
    }

    for (int i = 0; i < dat().size(); ++i) {
      dat()[i]->initialize_unit_values();
    }

    refresh_suf();
  }
  //----------------------------------------------------------------------
  // Returns the index of a group with the specified name.  If not
  // found, then a new Group with the specified value is added, and the
  // index of the new Group is returned.
  int AggregatedRegressionModel::find_group(const std::string &group_name,
                                            double group_value) {
    std::map<std::string, int>::iterator it = group_positions_.find(group_name);
    if (it != group_positions_.end()) {
      // found the group
      return it->second;
    }
    const Ptr<Group> &group(new Group(group_name, group_value, f));
    add_data(group);
    int pos = dat().size() - 1;
    group_positions_[group_name] = pos;
    return pos;
  }
  //----------------------------------------------------------------------
  void AggregatedRegressionModel::distribute_group_totals() {
    for (int i = 0; i < dat().size(); ++i) {
      dat()[i]->distribute_total(model_->Beta(), model_->sigma());
    }
    refresh_suf();
  }
  //----------------------------------------------------------------------
  // Communicate changes in raw data to sufficient statistics.
  void AggregatedRegressionModel::refresh_suf() {
    const std::vector<Ptr<RegressionData> > &data(model_->dat());
    model_->suf()->clear();
    for (int i = 0; i < data.size(); ++i) {
      model_->suf()->update(data[i]);
    }
  }
  //----------------------------------------------------------------------
  Agreg::Transformation *AggregatedRegressionModel::create_transformation(
      const std::string &name) {
    if (name == "log") {
      return new Agreg::LogTransformation;
    } else if (name == "sqrt") {
      return new Agreg::SquareRootTransformation;
    } else if (name.empty()) {
      return new Agreg::IdentityTransformation;
    } else {
      ostringstream err;
      err << "unknown transformation string supplied to constructor "
          << "for AggregatedRegressionModel: " << name << endl
          << "Legal values are \"\" (empty string), \"log\", and \"sqrt\"";
      report_error(err.str());
    }
    return NULL;
  }
  //----------------------------------------------------------------------
  void AggregatedRegressionModel::set_beta(const Vector &beta) {
    model_->set_Beta(beta);
  }
  //----------------------------------------------------------------------
  void AggregatedRegressionModel::set_sigma(double sigma) {
    model_->set_sigsq(sigma * sigma);
  }
}  // namespace BOOM
