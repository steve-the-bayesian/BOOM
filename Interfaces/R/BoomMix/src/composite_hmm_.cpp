#include <vector>
#include <r_interface/create_mixture_component.hpp>

#include <Models/CompositeModel.hpp>
#include <Models/CompositeData.hpp>
#include <Models/PosteriorSamplers/CompositeModelSampler.hpp>
#include <Models/HMM/HMM2.hpp>
#include <Models/HMM/PosteriorSamplers/HmmPosteriorSampler.hpp>
#include <Models/MarkovModel.hpp>

#include <r_interface/boom_r_tools.hpp>  // defines SEXP
#include <r_interface/prior_specification.hpp>
#include <r_interface/list_io.hpp>
#include <r_interface/print_R_timestamp.hpp>
#include <r_interface/seed_rng_from_R.hpp>

using BOOM::Ptr;
using BOOM::Data;
using BOOM::CompositeData;
using BOOM::CompositeModel;
using BOOM::CompositeModelSampler;
using BOOM::HiddenMarkovModel;
using BOOM::HmmPosteriorSampler;
using BOOM::MarkovModel;
using BOOM::MixtureComponent;
using BOOM::RListIoManager;
using BOOM::TimeSeries;
using BOOM::Matrix;

namespace {
//----------------------------------------------------------------------
// Extracts training data from the R mixture components.  Formats the
// data for BOOM, and assigns it to the hmm.
// Args:
//   rmixture_components: An R list of class
//     "CompositeMixtureComponent".  Each element in the list
//     contains the data to be modeled.  The data in a mixture
//     component is a list where each element is a time series
//     corresponding to an observational unit.  The mixture component
//     also contains a string explaining the type of data.
//   hmm:  A BOOM HiddenMarkovModel to which the data will be assigned.
// Returns:
//   On exit, the hmm will have its training data assigned, and will
//   take ownership of the data.
void AssignCompositeHmmData(SEXP rmixture_components,
                            Ptr<HiddenMarkovModel> hmm) {
  // The elements of raw_data must be converted to TimeSeries before
  // they can be added to hmm.
  std::vector<std::vector<Ptr<Data> > > raw_data =
      BOOM::RInterface::ExtractCompositeDataFromMixtureComponentList(
          rmixture_components);
  int number_of_series = raw_data.size();
  for (int series = 0; series < number_of_series; ++series) {
    NEW(TimeSeries<Data>, ts)(raw_data[series]);
    hmm->add_data_series(ts);
  }
}

class HmmLikelihoodGetter : public BOOM::ScalarIoCallback {
 public:
  explicit HmmLikelihoodGetter(Ptr<HiddenMarkovModel> m)
      :hmm_(m) {}
  virtual double get_value() const {
    return hmm_->saved_loglike();
  }
 private:
  Ptr<HiddenMarkovModel> hmm_;
};

class HmmLogpriorGetter : public BOOM::ScalarIoCallback {
 public:
  explicit HmmLogpriorGetter(Ptr<HiddenMarkovModel> m)
      :hmm_(m) {}
  virtual double get_value() const {
    return hmm_->logpri();
  }
 private:
  Ptr<HiddenMarkovModel> hmm_;
};

//----------------------------------------------------------------------
// Creates a BOOM::MarkovModel for use as the hidden Markov chain in
// a HiddenMarkovModel, based on R inputs.
// Args:
//   rlatent_markov_chain:  An R list of class "LatentMarkovChain".
//   io_manager:  An RListIoManager responsible for recording MCMC output.
// Returns:
//   A BOOM::Markov model.  No data is assigned, but a posterior
//   sampler has been assigned.  The model is ready to be used in the
//   constructor of a BOOM::HiddenMarkovModel.
Ptr<BOOM::MarkovModel> CreateLatentMarkovChain(
    SEXP rmarkov_prior,
    BOOM::RListIoManager *io_manager) {

  // The MarkovPrior object in prior_specification.hpp can be
  // initialized by the proper R object and create a BOOM MarkovModel
  // with the prior distribution already set.
  BOOM::RInterface::MarkovPrior markov_prior_spec(rmarkov_prior);
  Ptr<MarkovModel> latent_markov_chain(
      markov_prior_spec.create_markov_model());

  // Register model parameters with the io_manager.
  io_manager->add_list_element(new BOOM::MatrixListElement(
      latent_markov_chain->Q_prm(),
      "transition.probabilities"));
  io_manager->add_list_element(new BOOM::VectorListElement(
      latent_markov_chain->Pi0_prm(),
      "initial.state.distribution"));
  return(latent_markov_chain);
}

SEXP StoreStateProbabilities(Ptr<HiddenMarkovModel> hmm, SEXP output_list) {
  int number_of_subjects = hmm->nseries();
  if (number_of_subjects == 1) {
    return BOOM::appendListElement(
        output_list,
        ToRMatrix(hmm->report_state_probs(hmm->dat(0))),
        "state.probabilities");
  }

  SEXP state_probs;
  PROTECT(state_probs = Rf_allocVector(VECSXP, number_of_subjects));
  for(int i = 0; i < number_of_subjects; ++i) {
    SET_VECTOR_ELT(
        state_probs,
        i,
        ToRMatrix(hmm->report_state_probs(hmm->dat(i))));
  }
  SEXP ans;
  PROTECT(ans = BOOM::appendListElement(output_list,
                                        state_probs,
                                        "state.probabilities"));
  UNPROTECT(2);
  return ans;
}

//----------------------------------------------------------------------
// Creates the HiddenMarkovModel object based on R's inputs.  This is
// a driver function that calls many other Create functions.
// Args:
//   rmarkov_model_prior: An R list of class "MarkovModelPrior"
//     specifying the prior distribution for the parameters of the
//     hidden Markov chain.
//   rmixture_components:  An R list of class "CompositeMixtureComponent"
//   io_manager: A pointer to an RListIoManager that will be
//     responsible for recording the MCMC output.
// Returns:
//   A BOOM hidden Markov model, with posterior samplers and data
//   assigned.  Ready to call sample_posterior().
Ptr<HiddenMarkovModel> CreateCompositeHmm(SEXP rmarkov_model_prior,
                                          SEXP rmixture_components,
                                          RListIoManager *io_manager) {
  // Create the hidden Markov chain, including its prior
  // distribution and posterior sampler.
  Ptr<MarkovModel> latent_markov_chain = CreateLatentMarkovChain(
      rmarkov_model_prior, io_manager);

  // Create the mixture components, and create and assign their
  // posterior samplers.
  std::vector<Ptr<MixtureComponent> > composite_mixture_components =
      BOOM::RInterface::UnpackCompositeMixtureComponents(
          rmixture_components,
          latent_markov_chain->state_space_size(),
          io_manager);

  Ptr<HiddenMarkovModel> composite_hmm(new HiddenMarkovModel(
      composite_mixture_components, latent_markov_chain));
  Ptr<HmmPosteriorSampler> sampler(
      new HmmPosteriorSampler(composite_hmm.get()));
  composite_hmm->set_method(sampler);

  // Store the draws of log likelihood.
  io_manager->add_list_element(new BOOM::NativeUnivariateListElement(
      new HmmLikelihoodGetter(composite_hmm),
      "log.likelihood",
      NULL));

  // Store the draws of log prior.
  io_manager->add_list_element(new BOOM::NativeUnivariateListElement(
      new HmmLogpriorGetter(composite_hmm),
      "log.prior",
      NULL));

  AssignCompositeHmmData(rmixture_components, composite_hmm);
  return composite_hmm;
}

} // unnamed namespace

//======================================================================
extern "C" {
  SEXP composite_hmm_wrapper_(SEXP rmixture_components,
                              SEXP rmarkov_model_prior,
                              SEXP rniter,
                              SEXP rping,
                              SEXP rseed) {
    // Function for fitting a Bayesian HMM by MCMC.  The idiom is that
    // y[1] ... y[T] is a sequence of multivariate data to be modeled
    // with a hidden Markov chain.  y[i] = y[i, 1], ... y[i, k], where
    // y[i, j] given h[i] == s, is a vector modeled by model family
    // rmixture_components[[j]] with parameter vector theta[[s]].
    // Each of the mixtrue components in the input lists carries its
    // own training data.
    // Args:
    //   rmixture_components: An R list of mixture components.  The
    //     list is of class "CompositeMixtureComponent".  Each element
    //     in the list contains an aspect of the data to be modeled.
    //     Each element is created by one of the functions in
    //     create.mixture.components.R.
    //   rmarkov_model_prior: An R list of class MarkovModelPrior
    //     specifying the prior for the parameters of the hidden
    //     Markov chain.
    //   rniter:  The desired number of MCMC iterations.
    //   rping:  The desired frequency of status updates.
    //   rseed: An integer to use as the random seed for the C++
    //     random number generator.  Or R's NULL value, in which case
    //     the rng will be set by the clock.
    // Returns:
    //   An R list containing the MCMC draws for the hidden Markov model.
    RListIoManager io_manager;
    Ptr<HiddenMarkovModel> hmm;
    try {
      BOOM::RInterface::seed_rng_from_R(rseed);
      hmm = CreateCompositeHmm(
        rmarkov_model_prior,
        rmixture_components,
        &io_manager);
    } catch (std::exception &e) {
      std::ostringstream err;
      err << "Caught an exception with the following error message when "
          << "allocating space for the hidden Markov model:" << std::endl
          << e.what();
      Rf_error(err.str().c_str());
    }

    try {
      int niter = Rf_asInteger(rniter);
      int ping = Rf_asInteger(rping);
      SEXP result;
      PROTECT(result = io_manager.prepare_to_write(niter));
      hmm->save_state_probs();
      hmm->impute_latent_data();
      for (int i = 0; i < niter; ++i) {
        // TODO(stevescott): There is a potentially large resource
        // leak here, because the BOOM objects will not be freed if
        // Rf_error is called.
        R_CheckUserInterrupt();
        BOOM::print_R_timestamp(i, ping);
        hmm->sample_posterior();
        io_manager.write();
      }
      result = StoreStateProbabilities(hmm, result);
      // Result is not no longer PROTECTed because it points to new
      // memory, but we need to UNPROTECT the old memory before we
      // return result.  If more code is added between here and the
      // function return then the new result will need to be
      // PROTECTED.
      UNPROTECT(1);
      return result;
    } catch(std::exception &e) {
      std::ostringstream err;
      err << "Caught an exception with the following error message "
          << "during the MCMC algorithm:" << std::endl
          << e.what();
      hmm.reset(0);  // free the data from HMM before calling Rf_error.
      Rf_error(err.str().c_str());
    }
    return R_NilValue;
  }
}
