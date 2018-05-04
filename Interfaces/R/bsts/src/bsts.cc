// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/create_state_model.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/seed_rng_from_R.hpp"

#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/ThreadTools.hpp"

extern "C" {
  using BOOM::Vector;
  using BOOM::Ptr;
  using BOOM::bsts::ModelManager;
  using BOOM::RCheckInterrupt;
  using std::endl;

  SEXP analysis_common_r_fit_bsts_model_(
      SEXP r_data_list,
      SEXP r_state_specification,
      SEXP r_prior,
      SEXP r_options,
      SEXP r_family,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_timeout_in_seconds,
      SEXP r_seed) {
    BOOM::RErrorReporter error_reporter;
    BOOM::RMemoryProtector protector;
    try {
      BOOM::RInterface::seed_rng_from_R(r_seed);
      BOOM::RListIoManager io_manager;
      std::string family = BOOM::ToString(r_family);
      int xdim = 0;
      SEXP r_predictors = BOOM::getListElement(r_data_list, "predictors");
      if (!Rf_isNull(r_predictors)) {
        xdim = Rf_ncols(r_predictors);
      }
      std::unique_ptr<ModelManager> model_manager(ModelManager::Create(
          family, xdim));
      Ptr<BOOM::ScalarStateSpaceModelBase> model(model_manager->CreateModel(
          r_data_list,
          r_state_specification,
          r_prior,
          r_options,
          nullptr,
          &io_manager));

      // Do one posterior sampling step before getting ready to write.  This
      // will ensure that any dynamically allocated objects have the correct
      // size before any R memory gets allocated in the call to
      // prepare_to_write().
      model->sample_posterior();
      int niter = lround(Rf_asReal(r_niter));
      int ping = lround(Rf_asReal(r_ping));
      double timeout_threshold_seconds = Rf_asReal(r_timeout_in_seconds);

      SEXP ans = protector.protect(io_manager.prepare_to_write(niter));
      clock_t start_time = clock();
      double time_threshold = CLOCKS_PER_SEC * timeout_threshold_seconds;
      for (int i = 0; i < niter; ++i) {
        if (RCheckInterrupt()) {
          error_reporter.SetError("Canceled by user.");
          return R_NilValue;
        }
        BOOM::print_R_timestamp(i, ping);
        try {
          model->sample_posterior();
          io_manager.write();
          clock_t current_time = clock();
          if (current_time - start_time > time_threshold) {
            std::ostringstream warning;
            warning << "Timeout threshold "
                    << time_threshold
                    << " exceeded in iteration " << i << "."
                    << std::endl
                    << "Time used was "
                    << double(current_time - start_time) / CLOCKS_PER_SEC
                    << " seconds.";
            Rf_warning(warning.str().c_str());
            return BOOM::appendListElement(
                ans,
                ToRVector(BOOM::Vector(1, i + 1)),
                "ngood");
          }
        } catch(std::exception &e) {
          std::ostringstream err;
          err << "Caught an exception with the following "
              << "error message in MCMC "
              << "iteration " << i << ".  Aborting." << std::endl
              << e.what() << std::endl;
          error_reporter.SetError(err.str());
          return BOOM::appendListElement(ans,
                                         ToRVector(Vector(1, i)),
                                         "ngood");
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

  // Returns the posterior predictive distribution of a model forecast
  // over a specified forecast period.
  // Args:
  //   r_bsts_object: The object on which the predictions are to be
  //     based, which was returned by the original call to bsts.
  //   r_prediction_data: An R list containing any additional data
  //     needed to make the prediction.  For simple state space models
  //     this is just an integer giving the time horizon over which to
  //     predict.  For models containing a regression component it
  //     contains the future values of the X's.  For binomial (or
  //     Poisson) models it contains a sequence of future trial counts
  //     (or exposures).
  //   r_burn: An integer giving the number of burn-in iterations to
  //     discard.  Negative numbers will be treated as zero.  Numbers
  //     greater than the number of MCMC iterations will raise an error.
  //   r_observed_data: An R list containing the observed data on which
  //     to base the prediction.  In typical cases this should be
  //     R_NilValue (R's NULL) signaling that the observed data should
  //     be taken from r_bsts_object.  However, if additional data have
  //     been observed since the model was trained, or if the model is
  //     being used to predict data that were part of the training set,
  //     or some other application other than making predictions
  //     starting from one time period after the training data ends,
  //     then one can use this argument to pass the "training data" on
  //     which the predictions should be based.  If this argument is
  //     used, then the Kalman filter will be run over the supplied
  //     data, which is expensive.  If this argument is left as
  //     R_NilValue (NULL) then the "final.state" element of
  //     r_bsts_object will be used as the basis for future predictions,
  //     which is a computational savings over filtering from scratch.
  //
  // Returns:
  //   An R matrix containing draws from the posterior predictive
  //   distribution.  Rows of the matrix correspond to MCMC iterations,
  //   and columns to time points.  The matrix will have 'burn' fewer
  //   rows than the number of MCMC iterations in r_bsts_object.
  SEXP analysis_common_r_predict_bsts_model_(
      SEXP r_bsts_object,
      SEXP r_prediction_data,
      SEXP r_burn,
      SEXP r_observed_data,
      SEXP r_seed) {
    try {
      BOOM::RInterface::seed_rng_from_R(r_seed);
      std::unique_ptr<ModelManager> model_manager(
          ModelManager::Create(r_bsts_object));
      return BOOM::ToRMatrix(model_manager->Forecast(
          r_bsts_object,
          r_prediction_data,
          r_burn,
          r_observed_data));
    } catch (std::exception &e) {
      BOOM::RInterface::handle_exception(e);
    } catch (...) {
      BOOM::RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }

  // Compute the distribution of one-step prediction errors for the
  // training data or a set of holdout data.
  //
  // Args:
  //   r_bsts_object: The object on which the predictions are to be
  //     based, which was returned by the original call to bsts.
  //   r_cutpoints: A set of integers ranging from 1 to
  //     bsts.object$number.of.time.points.  One bsts model run is needed for each
  //     cutpoint, using data up to that cutpoint.
  //
  // Returns:
  //    A list of R matrices with rows corresponding to MCMC draws and columns
  //    corresponding to time.
  SEXP analysis_common_r_bsts_one_step_prediction_errors_(
      SEXP r_bsts_object,
      SEXP r_cutpoints) {
    try {
      std::vector<int> cutpoints = BOOM::ToIntVector(r_cutpoints, true);
      std::vector<BOOM::Matrix> prediction_errors(cutpoints.size());

      std::vector<std::future<void>> futures;
      int desired_threads = std::min<int>(
          cutpoints.size(), std::thread::hardware_concurrency() - 1);
      BOOM::ThreadWorkerPool pool;
      pool.add_threads(desired_threads);
      for (int i = 0; i < cutpoints.size(); ++i) {
        std::unique_ptr<ModelManager> model_manager(
            ModelManager::Create(r_bsts_object));
        futures.emplace_back(pool.submit(model_manager->CreateHoldoutSampler(
            r_bsts_object, cutpoints[i], &prediction_errors[i])));
      }
      for (int i = 0; i < futures.size(); ++i) {
        futures[i].get();
      }

      BOOM::RMemoryProtector protector;
      SEXP ans = protector.protect(Rf_allocVector(VECSXP, cutpoints.size()));
      for (int i = 0; i < prediction_errors.size(); ++i) {
        SET_VECTOR_ELT(ans, i, ToRMatrix(prediction_errors[i]));
      }
      return ans;
    } catch (std::exception &e) {
      BOOM::RInterface::handle_exception(e);
    } catch (...) {
      BOOM::RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }

}  // extern "C"
