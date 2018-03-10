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

#include "Models/Bart/PoissonBartModel.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "stats/moments.hpp"

namespace BOOM {

  PoissonBartModel::PoissonBartModel(int number_of_trees, double mean)
      : BartModelBase(number_of_trees, mean) {}

  //----------------------------------------------------------------------
  PoissonBartModel::PoissonBartModel(int number_of_trees,
                                     const std::vector<int> &responses,
                                     const Matrix &predictors)
      : BartModelBase(number_of_trees, 0.0) {
    double ybar = mean(Vector(responses.begin(), responses.end()));
    if (ybar > 0) {
      set_constant_prediction(log(ybar));
    }

    if (responses.size() != nrow(predictors)) {
      ostringstream err;
      err << "Error in PoissonBartModel constructor.  The response vector had "
          << responses.size() << " elements, but the predictor matrix had "
          << nrow(predictors) << " rows.  They should match." << std::endl;
      report_error(err.str());
    }
    for (int i = 0; i < responses.size(); ++i) {
      Ptr<PoissonRegressionData> dp(
          new PoissonRegressionData(responses[i], predictors.row(i)));
      add_data(dp);
    }
  }

  //----------------------------------------------------------------------
  PoissonBartModel::PoissonBartModel(int number_of_trees,
                                     const std::vector<int> &responses,
                                     const std::vector<double> &exposures,
                                     const Matrix &predictors)
      : BartModelBase(number_of_trees, 0.0) {
    if (responses.size() != nrow(predictors)) {
      ostringstream err;
      err << "Error in PoissonBartModel constructor.  The response vector had "
          << responses.size() << " elements, but the predictor matrix had "
          << nrow(predictors) << " rows.  They should match." << std::endl;
      report_error(err.str());
    }
    if (exposures.size() != responses.size()) {
      ostringstream err;
      err << "Error in PoissonBartModel constructor.  The response vector had "
          << responses.size() << " elements, but the vector of exposures had "
          << exposures.size() << " elements.  They should match." << std::endl;
      report_error(err.str());
    }

    double total_responses = 0;
    double total_exposures = 0;
    for (int i = 0; i < responses.size(); ++i) {
      total_exposures += exposures[i];
      total_responses += responses[i];
    }
    if (total_exposures > 0) {
      double ybar = total_responses / total_exposures;
      if (ybar > 0) {
        set_constant_prediction(log(ybar));
      }
    }

    for (int i = 0; i < responses.size(); ++i) {
      Ptr<PoissonRegressionData> dp(new PoissonRegressionData(
          responses[i], predictors.row(i), exposures[i]));
      add_data(dp);
    }
  }

  //----------------------------------------------------------------------
  PoissonBartModel::PoissonBartModel(const PoissonBartModel &rhs)
      : Model(rhs),
        BartModelBase(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs) {}

  //----------------------------------------------------------------------
  PoissonBartModel *PoissonBartModel::clone() const {
    return new PoissonBartModel(*this);
  }

  //----------------------------------------------------------------------
  int PoissonBartModel::sample_size() const { return dat().size(); }

  //----------------------------------------------------------------------
  void PoissonBartModel::add_data(const Ptr<Data> &dp) { add_data(DAT(dp)); }

  //----------------------------------------------------------------------
  void PoissonBartModel::add_data(const Ptr<PoissonRegressionData> &dp) {
    DataPolicy::add_data(dp);
    BartModelBase::observe_data(dp->x());
  }

  //----------------------------------------------------------------------
  // Sets the value of each tree to log_lambda / number_of_trees, so
  // that the predicted value will be log_lambda for any x.
  void PoissonBartModel::set_constant_prediction(double log_lambda) {
    for (int which_tree = 0; which_tree < number_of_trees(); ++which_tree) {
      Bart::Tree *this_tree = tree(which_tree);
      for (Bart::Tree::NodeSetIterator it = this_tree->leaf_begin();
           it != this_tree->leaf_end(); ++it) {
        (*it)->set_mean(log_lambda / number_of_trees());
      }
    }
  }

}  // namespace BOOM
