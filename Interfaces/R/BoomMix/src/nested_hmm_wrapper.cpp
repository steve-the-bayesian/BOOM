#include <vector>
#include <Models/MarkovModel.hpp>
#include <Models/PosteriorSamplers/MarkovConjSampler.hpp>
#include <Models/HMM/Clickstream/NestedHmm.hpp>
#include <Models/HMM/Clickstream/PosteriorSamplers/NestedHmmPosteriorSampler.hpp>

#include <r_interface/boom_r_tools.hpp>
#include <r_interface/determine_nthreads.hpp>
#include <r_interface/seed_rng_from_R.hpp>
#include <r_interface/handle_exception.hpp>
#include <r_interface/list_io.hpp>
#include <r_interface/print_R_timestamp.hpp>

using namespace BOOM;

namespace {

  // The model has the following parameters, with sizes measured by S0
  // (the number of observed event types, including the notional end
  // of session indicator), S1 (the number of latent event types), and
  // S2 (the number of session types).
  //
  // pi0:  Observed data level initial value.
  //       (niter x S2 x S1 x S0)
  // P0:  Observed data level transition probabilities
  //      (niter x S2 x S1 x S0 x S0)
  // pi1:  Latent event-level initial value
  //      (niter x S2 x S1)
  // P1:  Latent event-level transition probabilities
  //      (niter x S2 x S1 x S1)
  // pi2:  Latent session-level initial value
  //      (niter x S2)
  // P2:  Latent session-level transition probabilities
  //      (niter x S2 x S2)
  //
  // pi2 and P2 are single parameters, and can be handled with
  // established list_io elements.  The other 4 (pi1, P1, pi0, P0) all
  // need custom callbacks.
  class pi0Callback : public ArrayIoCallback {
   public:
    explicit pi0Callback(Ptr<NestedHmm> model)
        : model_(model)
    {}

    virtual std::vector<int> dim() const {
      std::vector<int> ans(3);
      ans[0] = model_->S2();
      ans[1] = model_->S1();
      ans[2] = model_->S0();
      return ans;
    }

    virtual void write_to_array(ArrayView &view) const {
      for (int H = 0; H < model_->S2(); ++H) {
        for (int h = 0; h < model_->S1(); ++h) {
          view.slice(H, h, -1) = model_->mix(H, h)->pi0();}}}

    virtual void read_from_array(const ArrayView &view) {
      for (int H = 0; H < model_->S2(); ++H) {
        for (int h = 0; h < model_->S1(); ++h) {
          Vector pi0 = model_->mix(H, h)->pi0();
          for (int i = 0; i < model_->S0(); ++i) {
            pi0[i] = view(H, h, i);
          }
          model_->mix(H, h)->set_pi0(pi0);}}}
   private:
    Ptr<NestedHmm> model_;
  };

  //----------------------------------------------------------------------
  class P0Callback : public ArrayIoCallback {
   public:
    explicit P0Callback(Ptr<NestedHmm> model) : model_(model) {}

    virtual std::vector<int> dim() const {
      std::vector<int> ans(4);
      ans[0] = model_->S2();
      ans[1] = model_->S1();
      ans[2] = model_->S0();
      ans[3] = model_->S0();
      return ans;
    }
    virtual void write_to_array(ArrayView &view) const {
      for (int H = 0; H < model_->S2(); ++H) {
        for (int h = 0; h < model_->S1(); ++h) {
          view.slice(H, h, -1, -1) = model_->mix(H, h)->Q();}}}
    virtual void read_from_array(const ArrayView &view) {
      for (int H = 0; H < model_->S2(); ++H) {
        for (int h = 0; h < model_->S1(); ++h) {
          Matrix P0 = model_->mix(H, h)->Q();
          for (int i = 0; i < P0.nrow(); ++i) {
            for (int j = 0; j < P0.nrow(); ++j) {
              P0(i, j) = view(H, h, i, j);}}
          model_->mix(H, h)->set_Q(P0);}}}
   private:
    Ptr<NestedHmm> model_;
  };
  //----------------------------------------------------------------------
  class pi1Callback : public ArrayIoCallback {
   public:
    explicit pi1Callback(Ptr<NestedHmm> model) : model_(model) {}
    virtual std::vector<int> dim() const {
      std::vector<int> ans(2);
      ans[0] = model_->S2();
      ans[1] = model_->S1();
      return ans;
    }
    virtual void write_to_array(ArrayView &view) const {
      for (int H = 0; H < model_->S2(); ++H) {
        view.slice(H, -1) = model_->event_model(H)->pi0();}}
    virtual void read_from_array(const ArrayView &view) {
      for (int H = 0; H < model_->S2(); ++H) {
        Vector pi1 = model_->event_model(H)->pi0();
        for (int h = 0; h < pi1.size(); ++h) {
          pi1[h] = view(H, h);
        }
        model_->event_model(H)->set_pi0(pi1);}}
   private:
    Ptr<NestedHmm> model_;
  };

  //----------------------------------------------------------------------
  class P1Callback : public ArrayIoCallback {
   public:
    explicit P1Callback(Ptr<NestedHmm> model) : model_(model) {}
    virtual std::vector<int> dim() const {
      std::vector<int> ans(3);
      ans[0] = model_->S2();
      ans[1] = model_->S1();
      ans[2] = model_->S1();
      return ans;
    }
    virtual void write_to_array(ArrayView &view) const {
      for (int H = 0; H < model_->S2(); ++H) {
        view.slice(H, -1, -1) = model_->event_model(H)->Q();
      }
    }
    virtual void read_from_array(const ArrayView &view) {
      for (int H = 0; H < model_->S2(); ++H) {
        Matrix P1 = model_->event_model(H)->Q();
        for (int r = 0; r < model_->S1(); ++r) {
          for (int s = 0; s < model_->S1(); ++s) {
            P1(r, s) = view(H, r, s);}}
        model_->event_model(H)->set_Q(P1);}}
   private:
    Ptr<NestedHmm> model_;
  };
  //----------------------------------------------------------------------
  class LogLikelihoodCallback : public ScalarIoCallback {
   public:
    explicit LogLikelihoodCallback(Ptr<NestedHmm> model) : model_(model) {}
    virtual double get_value() const {
      return model_->last_loglike();
    }
   private:
    Ptr<NestedHmm> model_;
  };

  //----------------------------------------------------------------------
  class LogPosteriorCallback : public ScalarIoCallback {
   public:
    explicit LogPosteriorCallback(Ptr<NestedHmm> model) : model_(model) {}
    virtual double get_value() const {
      return model_->last_logpost();
    }
   private:
    Ptr<NestedHmm> model_;
  };

  Ptr<NestedHmm> build_nested_hmm(SEXP r_streams,
                                  SEXP r_eos_label,
                                  SEXP r_nested_hmm_prior,
                                  SEXP r_threads,
                                  RListIoManager *io_manager) {
    // Extract the matrices needed for the prior.  The sizes of the
    // latent state spaces, are determined by the sizes of the
    // matrices in the prior.
    Matrix data_level_prior_transitions(
        ToBoomMatrix(getListElement(r_nested_hmm_prior, "N0")));
    Vector data_level_prior_initial_observations(
        ToBoomVector(getListElement(r_nested_hmm_prior, "n0")));

    Matrix event_level_prior_latent_transitions(
        ToBoomMatrix(getListElement(r_nested_hmm_prior, "N1")));
    Vector event_level_prior_initial_observations(
        ToBoomVector(getListElement(r_nested_hmm_prior, "n1")));
    Matrix session_level_prior_latent_transitions(
        ToBoomMatrix(getListElement(r_nested_hmm_prior, "N2")));
    Vector session_level_prior_initial_observations(
        ToBoomVector(getListElement(r_nested_hmm_prior, "n2")));

    int session_level_state_space_size =
        session_level_prior_latent_transitions.nrow();
    int event_level_state_space_size =
        event_level_prior_latent_transitions.nrow();

    //------------------------------------------------------------
    // Extract the data.  The only tricky part here is making sure
    // that the EOS label is present and is the last label in the list
    // of levels.
    int number_of_streams = Rf_length(r_streams);
    std::vector<Ptr<Clickstream::Stream> > streams;
    streams.reserve(number_of_streams);
    SEXP first_session = VECTOR_ELT(VECTOR_ELT(r_streams, 0), 0);
    std::vector<std::string> factor_levels = GetFactorLevels(first_session);
    // Check that the eos_label is either not included in the set of
    // factors, or if it is included it is the last one.
    std::string eos_label = CHAR(STRING_ELT(r_eos_label, 0));
    if (std::find(factor_levels.begin(), factor_levels.end(), eos_label)
        != factor_levels.end()) {
      // This block executes if eos_label is contained in factor_levels.
      if (factor_levels.back() != eos_label) {
        ostringstream err;
        err << "Found the end of session label at an illegal position." << endl
            << "The end-of-session indicator must either be implicit, "
            << "or it must be the last level of the factor." << endl;
        for (int i = 0; i < factor_levels.size(); ++i) {
          err << factor_levels[i] << endl;
        }
        report_error(err.str());
      }
    }
    NEW(CatKey, level_key)(factor_levels);
    for (int m = 0; m < number_of_streams; ++m) {
      SEXP r_stream = VECTOR_ELT(r_streams, m);
      // r_stream is a list of sessions
      int number_of_sessions = Rf_length(r_stream);
      std::vector<Ptr<Clickstream::Session> > sessions;
      sessions.reserve(number_of_sessions);
      for (int s = 0; s < number_of_sessions; ++s) {
        SEXP r_session = VECTOR_ELT(r_stream, s);
        // r_session is a factor.
        int number_of_events = Rf_length(r_session);
        int *values = INTEGER(r_session);
        std::vector<Ptr<Clickstream::Event> > events;
        events.reserve(number_of_events);
        NEW(Clickstream::Event, first_event)(
            values[0] - 1, Ptr<CatKeyBase>(level_key));
        events.push_back(first_event);
        for (int i = 1; i < number_of_events; ++i) {
          NEW(Clickstream::Event, next_event)(values[i] - 1, events.back());
          events.push_back(next_event);
        }
        NEW(Clickstream::Session, session)(events, true);
        sessions.push_back(session);
      }
      NEW(Clickstream::Stream, stream)(sessions);
      streams.push_back(stream);
    }

    NEW(NestedHmm, model)(streams,
                          session_level_state_space_size,
                          event_level_state_space_size);

    NEW(MarkovConjSampler, session_prior)(
        model->session_model().get(),
        session_level_prior_latent_transitions,
        session_level_prior_initial_observations);
    model->session_model()->set_method(session_prior);
    for (int H = 0; H < session_level_state_space_size; ++H) {
      NEW(MarkovConjSampler, event_transition_prior)(
          model->event_model(H).get(),
          event_level_prior_latent_transitions,
          event_level_prior_initial_observations);
      model->event_model(H)->set_method(event_transition_prior);
      for (int h = 0; h < event_level_state_space_size; ++h) {
        NEW(MarkovConjSampler, data_transition_prior)(
            model->mix(H, h).get(),
            data_level_prior_transitions,
            data_level_prior_initial_observations);
        model->mix(H, h)->set_method(data_transition_prior);
      }
    }
    NEW(NestedHmmPosteriorSampler, sampler)(model.get());
    model->set_method(sampler);

    // set up io_manager
    io_manager->add_list_element(new VectorListElement(
        model->session_model()->Pi0_prm(),
        "latent.session.initial.distribution"));
    io_manager->add_list_element(new MatrixListElement(
        model->session_model()->Q_prm(),
        "latent.session.transition.probabilities"));
    io_manager->add_list_element(new NativeArrayListElement(
        new pi1Callback(model),
        "latent.event.initial.distributions"));
    io_manager->add_list_element(new NativeArrayListElement(
        new P1Callback(model),
        "latent.event.transition.probabilities"));
    io_manager->add_list_element(new NativeArrayListElement(
        new pi0Callback(model),
        "observed.data.initial.distributions"));
    io_manager->add_list_element(new NativeArrayListElement(
        new P0Callback(model),
        "observed.data.transition.probabilities"));
    io_manager->add_list_element(new NativeUnivariateListElement(
        new LogLikelihoodCallback(model),
        "log.likelihood",
        NULL));
    io_manager->add_list_element(new NativeUnivariateListElement(
        new LogPosteriorCallback(model),
        "log.posterior",
        NULL));

    int threads = RInterface::determine_nthreads(r_threads);
    if (threads > 1)
      model->set_threads(threads);
    return model;
  }

  //  For debugging:
  void print_sufficient_statistics(Ptr<NestedHmm> model,
                                   int lowest_level) {
    std::ostringstream out;
    if (lowest_level <= 2) {
      out << "Level 2 sufficient statistics:" << endl
          << "Initial values: "
          << model->session_model()->suf()->init() << endl
          << "Transitions between sessions: " << endl
          << model->session_model()->suf()->trans() << endl;
    }

    if (lowest_level <= 1) {
      out << "Level 1 sufficient statistics: " << endl;
      for (int H = 0; H < model->S2(); ++H) {
        out << "Initial values for session type " << H << endl
            << model->event_model(H)->suf()->init() << endl
            << "Transitions between event types:" << endl
            << model->event_model(H)->suf()->trans() << endl;
      }
    }

    if (lowest_level <= 0) {
      out << "Data level sufficient statistics: " << endl;
      for (int H = 0; H < model->S2(); ++H) {
        for (int h = 0; h < model->S1(); ++h) {
          out << "Initial values for session ("
              << H << ", " << h << "):" << endl
              << model->mix(H, h)->suf()->init() << endl
              << "Transitions between page categories:" << endl
              << model->mix(H, h)->suf()->trans() << endl;
        }
      }
    }
    Rprintf("%s\n", out.str().c_str());
  }

  SEXP append_session_type_distribution(SEXP r_ans, Ptr<NestedHmm> model) {
    int number_of_streams = model->Nstreams();
    SEXP r_session_type_distribution;
    PROTECT(r_session_type_distribution = Rf_allocVector(VECSXP, number_of_streams));
    for (int i = 0; i < number_of_streams; ++i) {
      SET_VECTOR_ELT(r_session_type_distribution,
                     i,
                     ToRMatrix(
                         model->report_session_type_distribution(
                             model->stream(i))));

    }
    r_ans = appendListElement(
        r_ans, r_session_type_distribution, "session.type.distribution");
    UNPROTECT(1);
    return r_ans;
  }


}  // namespace

extern "C" {
  SEXP nested_hmm_wrapper_(SEXP r_streams,
                           SEXP r_eos_label,
                           SEXP r_nested_hmm_prior,
                           SEXP r_niter,
                           SEXP r_burn,
                           SEXP r_ping,
                           SEXP r_threads,
                           SEXP r_seed,
                           SEXP r_print_suf_level) {
    RListIoManager io_manager;
    try {
      Ptr<NestedHmm> model = build_nested_hmm(
          r_streams, r_eos_label,r_nested_hmm_prior, r_threads,
          &io_manager);
      int niter = Rf_asInteger(r_niter);
      int burn = Rf_asInteger(r_burn);
      int ping = Rf_asInteger(r_ping);
      int print_suf_level = Rf_asInteger(r_print_suf_level);
      RInterface::seed_rng_from_R(r_seed);
      SEXP ans;
      for (int i = 0; i < burn; ++i) {
        R_CheckUserInterrupt();
        model->sample_posterior();
      }
      model->clear_session_type_distribution();
      PROTECT(ans = io_manager.prepare_to_write(niter));
      for (int i = 0; i < niter; ++i) {
        R_CheckUserInterrupt();
        print_R_timestamp(i, ping);
        model->sample_posterior();
        io_manager.write();
        print_sufficient_statistics(model, print_suf_level);
      }
      ans = append_session_type_distribution(ans, model);
      UNPROTECT(1);
      return ans;
    } catch (std::exception &e) {
      BOOM::RInterface::handle_exception(e);
    } catch(...) {
      BOOM::RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }

}
