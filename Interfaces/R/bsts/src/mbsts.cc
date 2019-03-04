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

#include "r_interface/boom_r_tools.hpp"
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
      std::unique_ptr<MultivariateModelManager> model_manager(
          MultivariateModelManager::Create(xdim));

      Ptr<BOOM::MultivariateStateSpaceModelBase> model(model_manager->CreateModel(
          r_data_list,
          r_state_specification,
          r_prior,
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

}  // extern "C"
