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

#include "Models/Bart/LogitBartModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  LogitBartModel::LogitBartModel(int number_of_trees, double mean)
      : BartModelBase(number_of_trees, mean) {}

  LogitBartModel::LogitBartModel(int number_of_trees,
                                 const std::vector<int> &responses,
                                 const std::vector<int> &trials,
                                 const Matrix &predictors)
      : BartModelBase(number_of_trees, 0.0) {
    int n = responses.size();
    if (n != trials.size()) {
      std::ostringstream err;
      err << "There were " << n << " elements in the responses vector, but "
          << trials.size() << " in the trials vector.  "
          << "The two sizes must match." << endl;
      report_error(err.str());
    }
    check_predictor_dimension(n, predictors);
    for (int i = 0; i < n; ++i) {
      NEW(BinomialRegressionData, dp)
      (responses[i], trials[i], predictors.row(i));
      add_data(dp);
    }
  }

  LogitBartModel::LogitBartModel(int number_of_trees,
                                 const std::vector<bool> &responses,
                                 const Matrix &predictors)
      : BartModelBase(number_of_trees, 0.0) {
    int n = responses.size();
    check_predictor_dimension(n, predictors);
    for (int i = 0; i < n; ++i) {
      NEW(BinomialRegressionData, dp)(responses[i], 1, predictors.row(i));
      add_data(dp);
    }
  }

  LogitBartModel::LogitBartModel(const LogitBartModel &rhs)
      : Model(rhs),
        BartModelBase(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs) {}

  LogitBartModel *LogitBartModel::clone() const {
    return new LogitBartModel(*this);
  }

  int LogitBartModel::sample_size() const { return dat().size(); }

  void LogitBartModel::add_data(const Ptr<Data> &dp) { add_data(DAT(dp)); }

  void LogitBartModel::add_data(const Ptr<BinomialRegressionData> &dp) {
    DataPolicy::add_data(dp);
    BartModelBase::observe_data(dp->x());
  }

  void LogitBartModel::check_predictor_dimension(
      int number_of_observations, const Matrix &predictors) const {
    if (number_of_observations != nrow(predictors)) {
      ostringstream err;
      err << "There were " << nrow(predictors)
          << " rows in the predictor matrix, but " << number_of_observations
          << " elements in the response vector.  The two sizes must match."
          << endl;
      report_error(err.str());
    }
  }

}  // namespace BOOM
