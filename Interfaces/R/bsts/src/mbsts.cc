// Copyright 2019 Steven L. Scott.  All rights reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

#include <ctime>
#include <iostream>

#include "model_manager.h"
#include "utils.h"

#include "create_state_model.h"
#include "create_shared_state_model.h"
#include "multivariate_gaussian_model_manager.h"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/seed_rng_from_R.hpp"

#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "LinAlg/SubMatrix.hpp"

#include "cpputil/report_error.hpp"
#include "cpputil/ThreadTools.hpp"

extern "C" {
  using BOOM::Vector;
  using BOOM::Ptr;
  using BOOM::bsts::ModelManager;
  using BOOM::RCheckInterrupt;
  using std::endl;
  using BOOM::bsts::MultivariateGaussianModelManager;
  using BOOM::getListElement;
  using BOOM::ConstSubMatrix;

  // Args:
  //   r_data_list: A list containing the following entries:
  //     - predictors:  A matrix of predictor variables.
  //     - response: A vector of responses
  //     - timestamp: A set of timestamps indicating the time period to which
  //          each observation belongs.
  //     - series: A vector of integers taking values in 0 .. nseries - 1
  //         indicating which time series each column is part of.
  //   r_state_specification: A list of shared state specification objects used
  //     to define the state for the shared-state portion of the model.
  //   r_series_state_specification: A list of lists.  Each list element
  //     describes the state specification for a single, scalar time series
  //     (column of the response matrix).  Each element's elements are scalar
  //     state specification objects like one would pass to the scalar version
  //     of bsts.
  //   r_prior: The prior distribution for the observation model.  If the model
  //     includes a regression component this is just a SpikeSlabPrior, which
  //     will be repeated for each series.  In the future hierarchical priors
  //     will be allowed.  If no regression is present then either an SdPrior or
  //     a list of SdPrior objects will be used: each giving the prior
  //     distribution on the residual variance for a given series.  
  //   r_options: Model options.  This is currently unused, but that will change
  //     in the future as more multivariate models are considered.
  //   r_niter: An R integer giving the number of desired MCMC iterations.
  //   r_ping: An R integer giving the desired frequency of status updates.  If
  //     ping <= 0 then no status updates are printed.
  //   r_seed:  An integer to use for the C++ random seed, or NULL.
  //
  // Returns:
  //   The multivariate bsts model to be returned to the user from the mbsts R
  //   function.
  SEXP analysis_common_r_fit_multivariate_bsts_model_(
      SEXP r_data_list,
      SEXP r_state_specification,
      SEXP r_series_state_specification,
      SEXP r_prior,
      SEXP r_options,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_seed) {
    BOOM::RErrorReporter error_reporter;
    BOOM::RMemoryProtector protector;
    try {
      BOOM::RInterface::seed_rng_from_R(r_seed);
      BOOM::RListIoManager io_manager;
      int xdim = 0;
      SEXP r_predictors = BOOM::getListElement(r_data_list, "predictors");
      if (!Rf_isNull(r_predictors)) {
        xdim = Rf_ncols(r_predictors);
      }
      BOOM::Factor series(BOOM::getListElement(r_data_list, "series.id", true));
      int ydim = series.number_of_levels();

      // TODO(steve): generalize this to handle other model families.  Each
      // family will have its own model manager, and the base class will have a
      // Create method taking a "family" string as well as xdim, ydim.
      std::unique_ptr<MultivariateGaussianModelManager> model_manager(
          new MultivariateGaussianModelManager(ydim, xdim));

      Ptr<BOOM::MultivariateStateSpaceModelBase> model(
          model_manager->CreateModel(
              r_data_list,
              r_state_specification,
              r_series_state_specification,
              r_prior,
              r_options,
              &io_manager));

      // Do one posterior sampling step before getting ready to write.  This
      // will ensure that any dynamically allocated objects have the correct
      // size before any R memory gets allocated in the call to
      // prepare_to_write().
      model->sample_posterior();
      int niter = lround(Rf_asReal(r_niter));
      int ping = lround(Rf_asReal(r_ping));

      SEXP ans = protector.protect(io_manager.prepare_to_write(niter));
      for (int i = 0; i < niter; ++i) {
        if (RCheckInterrupt()) {
          error_reporter.SetError("Canceled by user.");
          return R_NilValue;
        }
        BOOM::print_R_timestamp(i, ping);
        try {
          model->sample_posterior();
          io_manager.write();
        } catch(std::exception &e) {
          std::ostringstream err;
          err << "Caught an exception with the following error message in "
              << "MCMC iteration " << i << ".  Aborting." << std::endl
              << e.what() << std::endl;
          error_reporter.SetError(err.str());
          return BOOM::appendListElement(
              ans, BOOM::ToRVector(Vector(1, i)), "ngood");
        }
      }
      return ans;
    } catch (std::exception &e) {
      BOOM::RInterface::handle_exception(e);
    } catch (...) {
      BOOM::RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }

  // Args:
  //   r_mbsts_object:  A model object produced by 'mbsts'.
  //   r_prediction_data: A list containing the predictors and any supplemental
  //     information needed to carry out the prediction.
  //   r_burn:  The number of iterations in r_mbsts_object to discard as burn-in.
  //   r_seed: An integer (or NULL) to use as the seed for the C++ random number
  //     generator.
  //
  // Returns:
  //   A 3-way array with dimensions [draws, series, time] containing draws from
  //   the posterior predictive distribution.
  SEXP analysis_common_r_predict_multivariate_bsts_model_(
      SEXP r_mbsts_object,
      SEXP r_prediction_data,
      SEXP r_burn,
      SEXP r_seed) {
    try {
      BOOM::RInterface::seed_rng_from_R(r_seed);

      BOOM::Factor series(BOOM::getListElement(r_mbsts_object, "series.id", true));
      int ydim = series.number_of_levels();
      int xdim = BOOM::ToBoomMatrixView(BOOM::getListElement(
          r_mbsts_object, "predictors", true)).ncol();
      
      std::unique_ptr<MultivariateGaussianModelManager> model_manager(
          new MultivariateGaussianModelManager(ydim, xdim));
      return BOOM::ToRArray(model_manager->Forecast(
          r_mbsts_object,
          r_prediction_data,
          r_burn));
    } catch (std::exception &e) {
      BOOM::RInterface::handle_exception(e);
    } catch (...) {
      BOOM::RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }
  
}  // extern "C"
