/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/StudentMvssRegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/TDataImputer.hpp"

namespace BOOM {
  namespace {
    using StudentData = StudentMultivariateTimeSeriesRegressionData;
  }

  StudentData::StudentMultivariateTimeSeriesRegressionData(
      double y, const Vector &x, int series, int timestamp)
      : MultivariateTimeSeriesRegressionData(y, x, series, timestamp),
        weight_(1.0)
  {}

  StudentData::StudentMultivariateTimeSeriesRegressionData(
      const Ptr<DoubleData> & y,
      const Ptr<VectorData> &x,
      int series,
      int timestamp)
      : MultivariateTimeSeriesRegressionData(y, x, series, timestamp),
        weight_(1.0)
  {}

  StudentMvssRegressionModel::StudentMvssRegressionModel(int xdim, int nseries)
      : data_policy_(nseries)
  {
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    // TBD
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
  }

  StudentMvssRegressionModel * StudentMvssRegressionModel::clone() const {
    report_error("Model is not clonable");
    return const_cast<StudentMvssRegressionModel *>(this);
  }

  StudentMvssRegressionModel * StudentMvssRegressionModel::deepclone() const {
    report_error("Model is not deepclonable");
    return const_cast<StudentMvssRegressionModel *>(this);
  }

  Matrix simulate_forecast(
      RNG &rng,
      const Matrix &forecast_predictors,
      const Vector &final_shared_state,
      const std::vector<Vector> &series_specific_final_state) {

    /////////////////////////
    /////////////////////////
    /////////////////////////
    /////////////////////////
    /////////////////////////
    /////////////////////////
    /////////////////////////
    return Matrix(0, 0);
  }

  DiagonalMatrix StudentMvssRegressionModel::observation_variance(
      int t, const Selector &observed) const {
    Vector diagonal_elements(observed.nvars());
    for (int i = 0; i < observed.nvars(); ++i) {
      int I = observed.expanded_index(i);
      diagonal_elements[i] = observation_model()->model(I)->sigsq() /
          data_policy_.data_point(I, t)->weight();
    }
    return DiagonalMatrix(diagonal_elements);
  }

  void StudentMvssRegressionModel::impute_student_weights(RNG &rng) {
    TDataImputer data_imputer;
    for (size_t time = 0; time < time_dimension(); ++time) {
      const Selector &observed(observed_status(time));

      // state_contribution contains the contribution from the shared state to
      // each observed data value.  Its index runs from 0 to observed.nvars().
      Vector shared_state_contribution =
          *observation_coefficients(time, observed) * shared_state(time);

      for (size_t s = 0; s < observed.nvars(); ++s) {
        int series = observed.sparse_index(s);
        StudentData *data_point = data_policy_.data_point(series, time).get();
        double time_series_residual =
            data_point->y() - shared_state_contribution[s];
        if (has_series_specific_state()) {
          time_series_residual -= series_specific_state_contribution(series, time);
        }
        const TRegressionModel *obs_model = observation_model()->model(series);

        double residual = time_series_residual
            - obs_model->predict(data_point->x());
        double weight = data_imputer.impute(
            rng, residual, obs_model->sigma(), obs_model->nu());
        data_point->set_weight(weight);
      }
    }

    for (size_t i = 0; i < data_policy_.total_sample_size(); ++i) {
      // StudentData *data_point = data_policy_.data_point(i).get();
    }

  }

}  // namespace BOOM
