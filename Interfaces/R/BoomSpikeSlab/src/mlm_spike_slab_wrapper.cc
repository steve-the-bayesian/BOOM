#include <Models/Glm/MultinomialLogitModel.hpp>
#include <Models/Glm/PosteriorSamplers/MultinomialLogitCompositeSpikeSlabSampler.hpp>
#include <Models/Glm/VariableSelectionPrior.hpp>
#include <Models/MvnModel.hpp>
#include <Models/IndependentMvnModel.hpp>

#include <r_interface/boom_r_tools.hpp>
#include <r_interface/create_mixture_component.hpp>
#include <r_interface/handle_exception.hpp>
#include <r_interface/list_io.hpp>
#include <r_interface/print_R_timestamp.hpp>
#include <r_interface/prior_specification.hpp>
#include <r_interface/seed_rng_from_R.hpp>

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#include <Rinternals.h>
#include <R.h>

namespace {
  using BOOM::Ptr;
  using namespace BOOM;

  std::pair<Ptr<MultinomialLogitModel>,
            Ptr<MultinomialLogitCompositeSpikeSlabSampler> >
      SpecifyMultinomialLogitModel(
      SEXP r_response_factor,
      SEXP r_subject_predictor_matrix,
      SEXP r_choice_predictor_matrix,
      SEXP r_choice_predictor_subject_id,
      SEXP r_choice_predictor_choice_id,
      SEXP r_multinomial_logit_spike_slab_prior,
      SEXP r_proposal_df,
      SEXP r_rwm_scale_factor,
      SEXP r_nthreads,
      SEXP r_mh_chunk_size,
      SEXP r_proposal_weights,
      RListIoManager *io_manager) {
    Factor response(r_response_factor);
    Matrix subject_predictor_matrix(
        ToBoomMatrix(r_subject_predictor_matrix));
    bool have_choice_data = !Rf_isNull(r_choice_predictor_matrix);
    int number_of_choices = response.number_of_levels();
    int subject_xdim = subject_predictor_matrix.ncol();
    int choice_xdim =
        have_choice_data ? Rf_ncols(r_choice_predictor_matrix) : 0;
    NEW(MultinomialLogitModel, model)(number_of_choices,
                                      subject_xdim,
                                      choice_xdim);
    // --------- Create data and add it to the model
    if (have_choice_data) {
      // dimensions are observations, choices, and choice_characteristics
      std::vector<int> dimensions(3);
      dimensions[0] = response.length();
      dimensions[1] = number_of_choices;
      dimensions[2] = choice_xdim;
      ConstArrayView choice_predictors(REAL(r_choice_predictor_matrix),
                                       dimensions);
      for (int i = 0; i < response.length(); ++i) {
        std::vector<Ptr<VectorData> > choice_predictor_data;
        choice_predictor_data.reserve(number_of_choices);
        for (int m = 0; m < number_of_choices; ++m) {
          choice_predictor_data.push_back(new VectorData(
              choice_predictors.vector_slice(i, m, -1)));
        }
        NEW(ChoiceData, dp)(response.to_categorical_data(i),
                            new VectorData(subject_predictor_matrix.row(i)),
                            choice_predictor_data);
        model->add_data(dp);
      }
    } else {
      std::vector<Ptr<VectorData> > empty;
      for (int i = 0; i < response.length(); ++i) {
        NEW(ChoiceData, dp)(
            response.to_categorical_data(i),
            new VectorData(subject_predictor_matrix.row(i)),
            empty);
        model->add_data(dp);
      }
    }

    // ------------- Create and set the prior and the posterior sampler. -----
    BOOM::RInterface::SpikeSlabGlmPrior prior(
        r_multinomial_logit_spike_slab_prior);
    double proposal_degrees_of_freedom = Rf_asReal(r_proposal_df);
    double rwm_variance_scale_factor = Rf_asReal(r_rwm_scale_factor);
    int nthreads = Rf_asInteger(r_nthreads);
    int mh_chunk_size = Rf_asInteger(r_mh_chunk_size);

    NEW(MultinomialLogitCompositeSpikeSlabSampler, sampler)(
        model.get(),
        prior.slab(),
        prior.spike(),
        proposal_degrees_of_freedom,
        rwm_variance_scale_factor,
        nthreads,
        mh_chunk_size);
    Vector proposal_weights(ToBoomVector(r_proposal_weights));
    sampler->set_move_probabilities(proposal_weights[0],
                                    proposal_weights[1],
                                    proposal_weights[2]);
    int max_flips = prior.max_flips();
    if (max_flips > 0) {
      sampler->limit_model_selection(max_flips);
    }

    model->set_method(sampler);

    // ------------- Add elements to the io_manager
    io_manager->add_list_element(
        new GlmCoefsListElement(model->coef_prm(), "beta"));

    return std::make_pair(model, sampler);
  }

}  // namespace

extern "C" {
  SEXP analysis_common_r_multinomial_logit_spike_slab(
      SEXP r_response_factor,
      SEXP r_subject_predictor_matrix,
      SEXP r_choice_predictor_matrix,
      SEXP r_choice_predictor_subject_id,
      SEXP r_choice_predictor_choice_id,
      SEXP r_multinomial_logit_spike_slab_prior,
      SEXP r_niter,
      SEXP r_ping,
      SEXP r_proposal_df,
      SEXP r_rwm_scale_factor,
      SEXP r_nthreads,
      SEXP r_mh_chunk_size,
      SEXP r_proposal_weights,
      SEXP r_seed) {
    RErrorReporter error_reporter;
    RMemoryProtector protector;
    try {
      RInterface::seed_rng_from_R(r_seed);
      RListIoManager io_manager;
      std::pair<Ptr<MultinomialLogitModel> ,
                Ptr<MultinomialLogitCompositeSpikeSlabSampler> >
          specification = SpecifyMultinomialLogitModel(
              r_response_factor,
              r_subject_predictor_matrix,
              r_choice_predictor_matrix,
              r_choice_predictor_subject_id,
              r_choice_predictor_choice_id,
              r_multinomial_logit_spike_slab_prior,
              r_proposal_df,
              r_rwm_scale_factor,
              r_nthreads,
              r_mh_chunk_size,
              r_proposal_weights,
              &io_manager);
      Ptr<MultinomialLogitModel> model = specification.first;
      Ptr<MultinomialLogitCompositeSpikeSlabSampler> sampler =
          specification.second;
      int niter = Rf_asInteger(r_niter);
      int ping = Rf_asInteger(r_ping);
      SEXP ans = protector.protect(io_manager.prepare_to_write(niter));
      for (int i = 0; i < niter; ++i) {
        if (RCheckInterrupt()) {
          error_reporter.SetError("Canceled by user.");
          return R_NilValue;
        }
        print_R_timestamp(i, ping);
        model->sample_posterior();
        io_manager.write();
      }
      ans = appendListElement(ans,
                              ToRMatrix(sampler->timing_report()),
                              "MH.accounting");
      return ans;
    } catch (std::exception &e) {
      RInterface::handle_exception(e);
    } catch(...) {
      RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }

}  // extern "C"
