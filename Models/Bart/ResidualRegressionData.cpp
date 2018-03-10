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

#include "Models/Bart/ResidualRegressionData.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace Bart {

    ResidualRegressionData::ResidualRegressionData(const VectorData *x)
        : predictor_(x) {}

    //----------------------------------------------------------------------
    const Vector &ResidualRegressionData::x() const {
      return predictor_->value();
    }

    //----------------------------------------------------------------------
    void ResidualRegressionData::subtract_from_residual(double value) {
      add_to_residual(-value);
    }

    //----------------------------------------------------------------------
    void ResidualRegressionData::add_to_gaussian_suf(
        GaussianBartSufficientStatistics &) const {
      report_error(
          "Illegal combination of ResidualRegressionData with "
          "GaussianBartSufficientStatistics.");
    }

    //----------------------------------------------------------------------
    void ResidualRegressionData::add_to_poisson_suf(
        PoissonSufficientStatistics &) const {
      report_error(
          "Illegal combination of ResidualRegressionData with "
          "PoissonSufficientStatistics.");
    }

    //----------------------------------------------------------------------
    void ResidualRegressionData::add_to_probit_suf(
        ProbitSufficientStatistics &) const {
      report_error(
          "Illegal combination of ResidualRegressionData with "
          "ProbitSufficientStatistics.");
    }

    //----------------------------------------------------------------------
    void ResidualRegressionData::add_to_logit_suf(
        LogitSufficientStatistics &) const {
      report_error(
          "Illegal combination of ResidualRegressionData with "
          "LogitSufficientStatistics.");
    }

  }  // namespace Bart
}  // namespace BOOM
