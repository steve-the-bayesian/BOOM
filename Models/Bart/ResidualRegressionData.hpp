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

#ifndef BOOM_BART_RESIDUAL_REGRESSION_DATA_HPP_
#define BOOM_BART_RESIDUAL_REGRESSION_DATA_HPP_

#include "Models/DataTypes.hpp"

namespace BOOM {
  namespace Bart {
    class SufficientStatisticsBase;
    class GaussianBartSufficientStatistics;
    class ProbitSufficientStatistics;
    class LogitSufficientStatistics;
    class PoissonSufficientStatistics;

    // ResidualRegressionData is used by Bart posterior samplers to
    // associate one or more residuals with RegressionData,
    // PoissonRegressionData, and BinomialRegressionData.  Each data
    // point owned by the Bart model class is associated with a
    // corresponding data point owned by the posterior sampler.
    // During the fitting algorithm, the posterior sampler associates
    // these "residual" data points with each of the trees in the Bart
    // model.  As the sampler progresses, each tree can simply look at
    // its own copy of the data to see the "residual" that it must fit
    // (i.e. the part not fit by the other trees).  There is a
    // different notion of "residual" appropriate for each concrete
    // class of Bart models, so we encode the common parts here and
    // leave specifics to concrete classes.
    class ResidualRegressionData {
     public:
      explicit ResidualRegressionData(const VectorData *x);
      virtual ~ResidualRegressionData() {}
      // The vector of predictors associated with this observation.
      const Vector &x() const;

      // Adjust the residual at this data point by the specified
      // value.  The notion is
      //
      //     observed_value = predictor_from_tree + residual.
      //
      // Thus you call add_to_residual(3) if the prediction at a tree
      // node goes down by 3.  The notion of 'residual' will be
      // different for different exponential families.  E.g. for
      // Poisson models there are two sets of residuals, and logit
      // models may have several.  Thus the implementation of adding a
      // value to a residual is left to the concrete classes.
      virtual void add_to_residual(double value) = 0;
      void subtract_from_residual(double value);

      // The default implementation of each of the following methods
      // is to throw an exception.  Each data type must override the
      // add_to_XXX_suf() method for the concrete class of
      // SufficientStatisticsBase to which it corresponds.
      virtual void add_to_gaussian_suf(
          GaussianBartSufficientStatistics &suf) const;
      virtual void add_to_poisson_suf(PoissonSufficientStatistics &suf) const;
      virtual void add_to_probit_suf(ProbitSufficientStatistics &suf) const;
      virtual void add_to_logit_suf(LogitSufficientStatistics &suf) const;

     private:
      const VectorData *predictor_;
    };

  }  // namespace Bart
}  // namespace BOOM
#endif  //  BOOM_BART_RESIDUAL_REGRESSION_DATA_HPP_
