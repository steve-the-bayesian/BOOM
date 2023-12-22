// Copyright 2018 Steven L. Scott. All Rights Reserved.
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
#include "dynamic_intercept_model_manager.h"
#include "utils.h"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/seed_rng_from_R.hpp"

#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/ThreadTools.hpp"

extern "C" {
  using namespace BOOM;
  using namespace BOOM::RInterface;
  using namespace BOOM::bsts;

  SEXP analysis_common_r_fit_dirm_(
      SEXP r_data_list,
      SEXP r_state_specification,
      SEXP r_prior,
      SEXP r_options,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_seed) {
    BOOM::RErrorReporter error_reporter;
    BOOM::RMemoryProtector protector;
    try {
      seed_rng_from_R(r_seed);
      BOOM::RListIoManager io_manager;
      std::unique_ptr<DynamicInterceptModelManager> model_manager(
          DynamicInterceptModelManager::Create(r_data_list));
      Ptr<BOOM::Model> model(model_manager->CreateModel(
          r_data_list,
          r_state_specification,
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
      double timeout_threshold_seconds = Rf_asReal(getListElement(
          r_options, "timeout.seconds", true));

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
            warning << "Timeout threshold " << time_threshold << " exceeded "
                    << "in iteration " << i << "." << std::endl
                    << "Time used was "
                    << double(current_time - start_time) / CLOCKS_PER_SEC
                    << " seconds.";
            Rf_warning("%s", warning.str().c_str());
            return BOOM::appendListElement(
                ans,
                ToRVector(BOOM::Vector(1, i + 1)),
                "ngood");
          }
        } catch(std::exception &e) {
          std::ostringstream err;
          err << "Caught an exception with the following error message in MCMC "
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
      handle_exception(e);
    } catch (...) {
      handle_unknown_exception();
    }
    return R_NilValue;
  }

}  // extern "C"
