// Copyright 2016 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#include <exception>
#include <string>

#include "cpputil/Ptr.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/seed_rng_from_R.hpp"
#include "r_interface/prior_specification.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/boom_r_tools.hpp"
#include "utils.h"

#include "Models/Glm/BinomialProbitModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialProbitCompositeSpikeSlabSampler.hpp"

#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/MvnModel.hpp"

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#include "Rinternals.h"
#include "R.h"

extern "C" {
  using namespace BOOM;  // NOLINT

  // This function is a wrapper for spike and slab regression.  It
  // takes input from R and formats it for the appropriate BOOM
  // objects that handle the computations.
  SEXP probit_spike_slab_wrapper(
      SEXP r_x,              // design matrix
      SEXP r_y,              // vector of success counts
      SEXP r_ny,             // vector of trial counts
      SEXP r_prior,          // SpikeSlabPrior
      SEXP r_niter,          // number of MCMC iterations
      SEXP r_ping,           // frequency of desired status updates
      SEXP r_beta0,          // initial value in the MCMC simulation
      SEXP r_clt_threshold,  // see comments in ../R/probit.spike.R
      SEXP r_proposal_df,
      SEXP r_sampling_weights,
      SEXP r_seed)  {
    RErrorReporter error_reporter;
    RMemoryProtector protector;
    try {
      BOOM::RInterface::seed_rng_from_R(r_seed);
      Matrix design_matrix(BOOM::ToBoomMatrix(r_x));
      std::vector<int> successes(BOOM::ToIntVector(r_y));
      std::vector<int> trials(BOOM::ToIntVector(r_ny));
      Ptr<BOOM::BinomialProbitModel> model(new BOOM::BinomialProbitModel(
          design_matrix.ncol()));
      int number_of_observations = successes.size();
      for (int i = 0; i < number_of_observations; ++i) {
        Ptr<BOOM::BinomialRegressionData>
            dp(new BOOM::BinomialRegressionData(
                successes[i],
                trials[i],
                design_matrix.row(i)));
        model->add_data(dp);
      }

      BOOM::RInterface::SpikeSlabGlmPrior prior(r_prior);

      Ptr<BOOM::BinomialProbitCompositeSpikeSlabSampler> sampler(
          new BOOM::BinomialProbitCompositeSpikeSlabSampler(
              model.get(),
              prior.slab(),
              prior.spike(),
              Rf_asInteger(r_clt_threshold),
              Rf_asReal(r_proposal_df)));
      sampler->set_sampling_weights(ToBoomVector(r_sampling_weights));

      if (prior.max_flips() > 0) {
        sampler->limit_model_selection(prior.max_flips());
      }

      BOOM::spikeslab::InitializeCoefficients(
          BOOM::ToBoomVector(r_beta0),
          prior.prior_inclusion_probabilities(),
          model,
          sampler);

      int niter = Rf_asInteger(r_niter);
      BOOM::RListIoManager io_manager;
      io_manager.add_list_element(
          new BOOM::GlmCoefsListElement(
              model->coef_prm(),
              "beta"));
      SEXP ans = protector.protect(io_manager.prepare_to_write(niter));
      int ping = Rf_asInteger(r_ping);

      for (int i = 0; i < niter; ++i) {
        if (RCheckInterrupt()) {
          error_reporter.SetError("Canceled by user.");
          return R_NilValue;
        }
        BOOM::print_R_timestamp(i, ping);
        sampler->draw();
        io_manager.write();
      }
      return ans;
    } catch(std::exception &e) {
      BOOM::RInterface::handle_exception(e);
    } catch(...) {
      BOOM::RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }
}
