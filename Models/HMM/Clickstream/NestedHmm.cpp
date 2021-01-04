// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2018 Steven L. Scott

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
#include "Models/HMM/Clickstream/NestedHmm.hpp"
#include <thread>

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Selector.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/VectorView.hpp"
#include "Models/HMM/hmm_tools.hpp"
#include "cpputil/ThreadTools.hpp"
#include "distributions.hpp"
#include "distributions/Markov.hpp"

namespace BOOM {

  void NestedHmm::setup() {
    session_model_ = new MarkovModel(S2_);
    //    session_model_->free_pi0();
    ParamPolicy::add_model(session_model_);
    for (int H = 0; H < S2_; ++H) {
      NEW(MarkovModel, tmp)(S1_);
      // tmp->free_pi0();
      event_model_.push_back(tmp);
      ParamPolicy::add_model(tmp);
      for (int h = 0; h < S1_; ++h) {
        NEW(MarkovModel, tmp)(S0_);
        // tmp->free_pi0();
        mix_[H].push_back(tmp);
        ParamPolicy::add_model(tmp);
      }
    }
  }

  NestedHmm::NestedHmm(const std::vector<Ptr<Clickstream::Stream> > &streams,
                       int S2, int S1)
      : DataPolicy(streams),
        S0_(streams[0]->number_of_page_categories_including_eos()),
        S1_(S1),
        S2_(S2),
        mix_(S2),
        loglike_(new UnivParams(0.0)),
        logpost_(new UnivParams(0.0)),
        pi_(S1 * S2),
        logpi0_(S1 * S2),
        logd_(S1 * S2),
        one_(S1 * S2, 1.0),
        logQ1_(S1 * S2, S1 * S2),
        logQ2_(S1 * S2, S1 * S2) {
    setup();
  }

  NestedHmm::NestedHmm(int S2, int S1, int S0)
      : DataPolicy(),
        S0_(S0),
        S1_(S1),
        S2_(S2),
        mix_(S2),
        loglike_(new UnivParams(0.0)),
        logpost_(new UnivParams(0.0)),
        pi_(S1 * S2),
        logpi0_(S1 * S2),
        logd_(S1 * S2),
        one_(S1 * S2, 1.0),
        logQ1_(S1 * S2, S1 * S2),
        logQ2_(S1 * S2, S1 * S2) {
    setup();
  }

  //----------------------------------------------------------------------
  NestedHmm *NestedHmm::clone() const { return new NestedHmm(*this); }
  //----------------------------------------------------------------------
  int NestedHmm::S2() const { return S2_; }
  int NestedHmm::S1() const { return S1_; }
  int NestedHmm::S0() const { return S0_; }
  //----------------------------------------------------------------------
  const Ptr<MarkovModel> &NestedHmm::mix(int H, int h) { return mix_[H][h]; }

  const Ptr<MarkovModel> &NestedHmm::mix(int H, int h) const {
    return mix_[H][h];
  }
  //----------------------------------------------------------------------
  Ptr<MarkovModel> NestedHmm::event_model(int H) { return event_model_[H]; }
  const Ptr<MarkovModel> NestedHmm::event_model(int H) const {
    return event_model_[H];
  }
  //----------------------------------------------------------------------
  Ptr<MarkovModel> NestedHmm::session_model() { return session_model_; }
  const Ptr<MarkovModel> NestedHmm::session_model() const {
    return session_model_;
  }
  //----------------------------------------------------------------------
  int NestedHmm::Nstreams() const { return dat().size(); }
  //----------------------------------------------------------------------
  double NestedHmm::initialize(const Ptr<Event> &event) const {
    fill_logd(event);
    pi_ = logpi0_ + logd_;
    double M = max(pi_);
    pi_ -= M;
    pi_ = exp(pi_);
    double nc = sum(pi_);
    pi_ /= nc;
    return M + log(nc);
  }
  //----------------------------------------------------------------------
  double NestedHmm::loglike() {
    double ans = 0;
    for (int i = 0; i < Nstreams(); ++i) ans += fwd(stream(i));
    loglike_->set(ans);
    return ans;
  }
  //----------------------------------------------------------------------
  void NestedHmm::set_loglike(double x) { loglike_->set(x); }
  //----------------------------------------------------------------------
  void NestedHmm::set_logpost(double x) { logpost_->set(x); }
  //----------------------------------------------------------------------
  double NestedHmm::last_loglike() const { return loglike_->value(); }
  //----------------------------------------------------------------------
  double NestedHmm::last_logpost() const { return logpost_->value(); }
  //----------------------------------------------------------------------
  void NestedHmm::print_params(std::ostream &out) const {
    out << "phi2 = " << session_model()->pi0() << endl
        << "Phi2 = " << endl
        << session_model()->Q();
    for (int H = 0; H < S2(); ++H) {
      out << endl
          << "phi1(" << H << ") = " << event_model(H)->pi0() << endl
          << "Phi1(" << H << ") = " << endl
          << event_model(H)->Q() << endl;
    }

    for (int H = 0; H < S2(); ++H) {
      for (int h = 0; h < S1(); ++h) {
        out << endl
            << "phi0(" << H << "," << h << ") = " << mix(H, h)->pi0() << endl
            << "Phi0(" << H << "," << h << ") = " << endl
            << mix(H, h)->Q() << endl;
      }
    }
  }

  //----------------------------------------------------------------------
  void NestedHmm::print_event(std::ostream &out, const char *msg,
                              const Ptr<Stream> &u, const Ptr<Session> &session,
                              const Ptr<Event> &event, int j) const {
    out << msg << " for stream "
        << "The numerical value of this event is " << event->value() << endl
        << "It is event number " << j << " (counting from 0) in "
        << "the following session" << endl
        << *session << endl
        << "pi_ = " << pi_ << endl
        << "logd_ = " << logd_ << endl;

    print_params(out);
    print_filter(out, j);
  }
  //----------------------------------------------------------------------
  void NestedHmm::print_filter(std::ostream &out, int j) const {
    for (int i = 0; i <= j; ++i) {
      out << "filter for transition " << i << endl << P[i] << endl;
    }
  }

  //----------------------------------------------------------------------
  double NestedHmm::fwd(const Ptr<Stream> &u) const {
    double ans = 0;
    int Nsessions = u->nsessions();
    int stream_nevents = u->number_of_events_including_eos();
    check_filter_size(stream_nevents);
    int event_num = 0;
    for (int i = 0; i < Nsessions; ++i) {
      Ptr<Session> session = u->session(i);
      int nevents = session->number_of_events_including_eos();
      for (int j = 0; j < nevents; ++j) {
        Ptr<Event> event(session->event(j));
        if (i == 0 && j == 0) {
          ans += initialize(event);
          if (!std::isfinite(ans)) {
            ostringstream err;
            print_event(err,
                        "found an infinte value while initializing "
                        "the fb filter",
                        u, session, event, j);
            report_error(err.str());
          }
        } else {
          fill_logd(event);
          // use logQ1_ if the first event in a session.
          // use logQ2_ otherwise
          const Matrix &logQ(j == 0 ? logQ1_ : logQ2_);
          ans += fwd_1(pi_, P[event_num], logQ, logd_, one_);
          if (!std::isfinite(ans) || !std::isfinite(pi_[0])) {
            ostringstream err;
            print_event(err, "found an infinity in NestedHmm::fwd", u, session,
                        event, j);
            report_error(err.str());
          }
        }
        ++event_num;
      }
    }
    assert(event_num == stream_nevents);
    return ans;
  }
  //----------------------------------------------------------------------
  void NestedHmm::update_mixture(int H, int h, const Ptr<Event> &event,
                                 double p) {
    mix(H, h)->suf()->add_mixture_data(event, p);
  }
  //----------------------------------------------------------------------
  void NestedHmm::bkwd_smoothing(const Ptr<Stream> &u) {
    // check this.  make sure now-then correctly updated, and do
    // boundary cases.

    int Nsessions = u->nsessions();
    int event_num = u->number_of_events_including_eos();

    Vector hinit, Hinit;
    Matrix htrans, Htrans;
    Vector wsp(S2_ * S1_);

    for (int i = Nsessions; i != 0; --i) {
      Ptr<Session> session(u->session(i - 1));
      int nevents = session->number_of_events_including_eos();
      for (int j = nevents; j != 0; --j) {  // j - 1 is the current event
        --event_num;

        Ptr<Event> event(session->event(j - 1));
        // pi_ is the distribution of the hidden Markov chain
        // corresponding to event

        // P[event_num] is the joint distribution of the hidden Markov
        // chain for event and its predecessor

        // P[0] is undefined

        for (int H = 0; H < S2_; ++H) {
          for (int h = 0; h < S1_; ++h) {
            double p = pi_[encode_state(H, h)];
            update_mixture(H, h, event, p);
          }
        }

        if (j == 1) {  // first event in a session, record initial h
          for (int H = 0; H < S2_; ++H) {
            hinit = get_hinit(pi_, H);
            event_model(H)->suf()->add_initial_distribution(hinit);
          }

          if (i == 1) {  // first event in any session, record initial H
            Hinit = get_Hinit(pi_);
            session_model()->suf()->add_initial_distribution(Hinit);
          } else {  // first event in a later session, record H transition
            Htrans = get_Htrans(P[event_num]);
            session_model()->suf()->add_transition_distribution(Htrans);
          }
        } else {  // normal case.. not a first event.  record h transition
          for (int H = 0; H < S2_; ++H) {
            htrans = get_htrans(P[event_num], H);
            event_model(H)->suf()->add_transition_distribution(htrans);
          }
        }

        if (i > 1 || j > 1)
          bkwd_1(pi_, P[event_num], wsp, one_);  // sets pi_ for next iteration
      }  // ends loop over events in a session
    }    // ends loop over sessions
  }      // closes function

  //----------------------------------------------------------------------
  ConstVectorView NestedHmm::get_hinit(const Vector &pi, int H) const {
    ConstVectorView ans(pi.data() + H * S1_, S1_, pi.stride());
    return ans;
  }
  //----------------------------------------------------------------------
  Vector NestedHmm::get_Hinit(const Vector &pi) const {
    Vector ans(S2_);
    for (int H = 0; H < S2_; ++H) ans[H] = get_hinit(pi, H).sum();
    return ans;
  }
  //----------------------------------------------------------------------
  ConstSubMatrix NestedHmm::get_block(const Matrix &P, int H1, int H2) const {
    ConstSubMatrix ans(P, H1 * S1_, (H1 + 1) * S1_ - 1, H2 * S1_,
                       (H2 + 1) * S1_ - 1);
    return ans;
  }
  //----------------------------------------------------------------------
  ConstSubMatrix NestedHmm::get_htrans(const Matrix &P, int H) const {
    return get_block(P, H, H);
  }
  //----------------------------------------------------------------------
  Matrix NestedHmm::get_Htrans(const Matrix &P) const {
    Matrix ans(S2_, S2_);
    for (int H1 = 0; H1 < S2_; ++H1) {
      for (int H2 = 0; H2 < S2_; ++H2) {
        ans(H1, H2) = get_block(P, H1, H2).sum();
      }
    }
    return ans;
  }
  //----------------------------------------------------------------------
  double NestedHmm::fwd_bkwd_with_threads(bool bayes, bool find_mode) {
    clear_client_data();
    pass_params_to_workers();
    start_thread_em();
    double loglike = collect_threads();
    loglike_->set(loglike);
    if (bayes) {
      loglike += logpri();
      logpost_->set(loglike);
    }
    if (find_mode) complete_data_mode(bayes);
    return loglike;
  }

  //----------------------------------------------------------------------
  // One step of an EM algorithm for finding point estimates of model
  // parameters
  double NestedHmm::fwd_bkwd(bool bayes, bool find_mode) {
    if (!workers_.empty()) {
      return fwd_bkwd_with_threads(bayes, find_mode);
    } else {
      clear_client_data();
      int N = Nstreams();
      fill_big_Q();
      double loglike = 0;

      for (int i = 0; i < N; ++i) {
        loglike += fwd(stream(i));
        bkwd_smoothing(stream(i));
      }
      if (find_mode) complete_data_mode(bayes);

      loglike_->set(loglike);

      if (bayes) {
        loglike += logpri();
        logpost_->set(loglike);
      }
      return loglike;
    }
  }

  //----------------------------------------------------------------------
  // does a single M-step of an EM algorithm for finding either the
  // posterior mode (if(bayes)) or the MLE
  void NestedHmm::complete_data_mode(bool bayes) {
    if (bayes)
      session_model()->find_posterior_mode();
    else
      session_model()->mle();

    for (int H = 0; H < S2_; ++H) {
      if (bayes)
        event_model(H)->find_posterior_mode();
      else
        event_model(H)->mle();
      for (int h = 0; h < S1_; ++h) {
        if (bayes)
          mix(H, h)->find_posterior_mode();
        else
          mix(H, h)->mle();
      }
    }
  }
  //----------------------------------------------------------------------
  double NestedHmm::logpri() const {
    double ans = session_model()->logpri();
    for (int H = 0; H < S2_; ++H) {
      ans += event_model(H)->logpri();
      for (int h = 0; h < S1_; ++h) {
        ans += mix(H, h)->logpri();
      }
    }
    return ans;
  }

  //----------------------------------------------------------------------
  // A full EM algorithm for finding point estimates of model parameters
  double NestedHmm::EM(double eps, bool bayes) {
    double crit = 1 + eps;
    randomize_starting_values();
    double oldloglike = fwd_bkwd(bayes);
    while (crit > eps) {
      double loglike = fwd_bkwd(bayes);
      crit = loglike - oldloglike;
      oldloglike = loglike;
    }
    return oldloglike;
  }

  //----------------------------------------------------------------------

  std::ostream &NestedHmm::write_suf(std::ostream &out) const {
    out << "  Session Model:" << endl
        << "      init  = " << session_model()->suf()->init() << endl
        << "      trans = " << endl
        << session_model()->suf()->trans() << endl
        << endl;

    for (int H = 0; H < S2_; ++H) {
      out << "  Event Model " << H << endl
          << "     init  = " << event_model(H)->suf()->init() << endl
          << "     trans = " << endl
          << event_model(H)->suf()->trans() << endl
          << endl;
    }

    for (int H = 0; H < S2_; ++H) {
      for (int h = 0; h < S1_; ++h) {
        out << "  Obs Model " << H << "," << h << endl
            << "      init  = " << mix(H, h)->suf()->init() << endl
            << "      trans  = " << endl
            << mix(H, h)->suf()->trans() << endl
            << endl;
      }
    }
    return out;
  }

  //----------------------------------------------------------------------
  void NestedHmm::update(int H, int h, const Ptr<Event> &event) {
    mix(H, h)->add_data(event);
  }

  void NestedHmm::clear_session_type_distribution() {
    session_type_distribution_.clear();
  }

  Matrix NestedHmm::report_session_type_distribution(
      const Ptr<Stream> &stream) const {
    std::map<Ptr<Stream>, Matrix>::const_iterator it =
        session_type_distribution_.find(stream);
    if (it == session_type_distribution_.end()) {
      report_error(
          "Invalid stream passed to NestedHmm::"
          "report_session_type_distribution.");
    }
    Matrix ans = it->second;
    double total = sum(ans.row(0));
    if (total == 0.0) {
      report_error(
          "The stream passed to NestedHmm::"
          "report_session_type_distribution has never been assigned "
          "a session type");
    }
    return ans / total;
  }

  //----------------------------------------------------------------------
  void NestedHmm::bkwd_sampling(const Ptr<Stream> &u) {
    int Nsessions = u->nsessions();
    int event_num = u->number_of_events_including_eos();
    // be sure to grab the terminal state before you start the loop,
    // for singleton observations

    int Hnow, hnow;
    // This works for the final event because pi_ was set by fwd().
    int s = rmulti_mt(rng(), pi_);
    decode_state(s, Hnow, hnow);

    Matrix &session_type_distribution(session_type_distribution_[u]);
    if (session_type_distribution.nrow() != Nsessions ||
        session_type_distribution.ncol() != S2_) {
      session_type_distribution.resize(Nsessions, S2_);
      session_type_distribution = 0.0;
    }

    for (int i = Nsessions; i != 0; --i) {  // i - 1 is the current session
      Ptr<Session> session(u->session(i - 1));
      int nevents = session->number_of_events_including_eos();
      ++session_type_distribution(i - 1, Hnow);
      for (int j = nevents; j != 0; --j) {  // j - 1 is the current event

        // We start the recursion knowing the state of the current
        // event the purpose of the recursion is to draw the state of
        // the previous event.

        // The recursion stops when we've seen the first event from
        // the stream

        Ptr<Event> event(session->event(j - 1));
        update(Hnow, hnow, event);
        int Hthen = 0;  // these won't be used in first event of fist session
        int hthen = 0;

        --event_num;
        if (event_num > 0) {
          assert(i > 1 || j > 1);
          pi_ = P[event_num].col(s);      // P = joint dist. of yesterday,today
          int r = rmulti_mt(rng(), pi_);  // pi_  = dist. of yesterday's event
          decode_state(r, Hthen, hthen);
        }

        if (j == 1) {  // start of a new session
          event_model(Hnow)->suf()->add_initial_value(hnow);

          if (i == 1) {  // first event in first session
            session_model()->suf()->add_initial_value(Hnow);
          } else {  // first event in a later session
            session_model()->suf()->add_transition(Hthen, Hnow);
          }
        } else {  // typical situation... interior of a session
          event_model(Hnow)->suf()->add_transition(hthen, hnow);
        }

        Hnow = Hthen;
        hnow = hthen;
      }  // ends loop over events in a session
    }    // ends loop over sessions
  }
  //----------------------------------------------------------------------
  int NestedHmm::encode_state(int H, int h) const { return S1_ * H + h; }
  //----------------------------------------------------------------------
  void NestedHmm::decode_state(int state, int &H, int &h) const {
    h = state % S1_;
    H = state / S1_;
  }
  //----------------------------------------------------------------------
  Ptr<Clickstream::Stream> NestedHmm::stream(int i) { return this->dat()[i]; }
  //----------------------------------------------------------------------
  double NestedHmm::impute_latent_data() {
    if (!workers_.empty()) {
      return impute_latent_data_with_threads();
    }
    clear_client_data();
    double ans = 0;
    fill_big_Q();
    for (int i = 0; i < Nstreams(); ++i) {
      Ptr<Stream> u(stream(i));
      ans += fwd(u);
      bkwd_sampling(u);
    }
    loglike_->set(ans);
    logpost_->set(ans + logpri());
    return ans;
  }
  //----------------------------------------------------------------------
  double NestedHmm::impute_latent_data_with_threads() {
    clear_client_data();
    pass_params_to_workers();
    start_thread_imputation();
    double loglike = collect_threads();
    loglike_->set(loglike);
    logpost_->set(loglike + logpri());
    return loglike;
  }
  //----------------------------------------------------------------------
  struct ClickstreamSamplingImputer {
    Ptr<NestedHmm> mod;
    explicit ClickstreamSamplingImputer(const Ptr<NestedHmm> &m) : mod(m) {}
    void operator()() { mod->impute_latent_data(); }
  };
  //----------------------------------------------------------------------
  struct ClickstreamEmImputer {
    explicit ClickstreamEmImputer(const Ptr<NestedHmm> &m) : mod(m) {}
    Ptr<NestedHmm> mod;
    void operator()() { mod->fwd_bkwd(false, false); }
  };
  //----------------------------------------------------------------------
  void NestedHmm::set_threads(int n) {
    clear_workers();
    for (int i = 0; i < n; ++i) {
      NEW(NestedHmm, worker)(S2_, S1_, S0_);
      add_worker(worker);
    }
    allocate_data_to_workers();
  }
  //----------------------------------------------------------------------
  void NestedHmm::pass_params_to_workers() {
    Vector v = this->vectorize_params();
    for (int i = 0; i < workers_.size(); ++i)
      workers_[i]->unvectorize_params(v);
  }
  //----------------------------------------------------------------------
  void NestedHmm::start_thread_imputation() {
    ThreadWorkerPool pool;
    pool.add_threads(workers_.size());
    std::vector<std::future<void>> futures;
    for (int i = 0; i < workers_.size(); ++i){
      ClickstreamSamplingImputer imputer(workers_[i]);
      futures.emplace_back(pool.submit(imputer));
    }
    for (int i = 0; i < workers_.size(); ++i) {
      futures[i].get();
    }
  }
  //----------------------------------------------------------------------
  void NestedHmm::add_worker(const Ptr<NestedHmm> &w) { workers_.push_back(w); }
  //----------------------------------------------------------------------
  void NestedHmm::clear_workers() { workers_.clear(); }
  //----------------------------------------------------------------------
  void NestedHmm::allocate_data_to_workers() {
    int n = workers_.size();
    for (int i = 0; i < Nstreams(); ++i) {
      int id = i % n;
      workers_[id]->add_data(stream(i));
    }
  }
  //----------------------------------------------------------------------
  void NestedHmm::start_thread_em() {
    ThreadWorkerPool pool;
    pool.add_threads(workers_.size());
    std::vector<std::future<void>> futures;
    for (int i = 0; i < workers_.size(); ++i) {
      ClickstreamEmImputer imputer(workers_[i]);
      futures.emplace_back(pool.submit(imputer));
    }
    for (int i = 0; i < workers_.size(); ++i) {
      futures[i].get();
    }
  }
  //----------------------------------------------------------------------
  double NestedHmm::collect_threads() {
    double loglike = 0;
    for (int i = 0; i < workers_.size(); ++i) {
      session_model()->suf()->combine(workers_[i]->session_model()->suf());
      for (int H = 0; H < S2_; ++H) {
        event_model(H)->suf()->combine(workers_[i]->event_model(H)->suf());
        for (int h = 0; h < S1_; ++h) {
          mix(H, h)->suf()->combine(workers_[i]->mix(H, h)->suf());
        }
      }
      loglike += workers_[i]->last_loglike();
    }
    return loglike;
  }
  //----------------------------------------------------------------------
  void NestedHmm::clear_client_data() {
    session_model()->clear_data();
    for (int H = 0; H < S2_; ++H) {
      event_model(H)->clear_data();
      for (int h = 0; h < S1_; ++h) mix(H, h)->clear_data();
    }
  }
  //----------------------------------------------------------------------
  void NestedHmm::randomize_starting_values() {
    clear_client_data();

    Vector nu2(S2_, 1.0);
    Matrix Q2(S2_, S2_);
    for (int H = 0; H < S2_; ++H) {
      Q2.row(H) = rdirichlet(nu2);
    }
    session_model()->set_Q(Q2);
    session_model()->set_pi0(rdirichlet(nu2));

    Vector nu1(S1_, 1.0);
    for (int H = 0; H < S2_; ++H) {
      Matrix Q(S1_, S1_);
      for (int h = 0; h < S1_; ++h) {
        Q.row(h) = rdirichlet(nu1);
      }
      event_model(H)->set_Q(Q);
      event_model(H)->set_pi0(rdirichlet(nu1));
    }

    Vector nu0(S0_, 1.0);
    for (int H = 0; H < S2_; ++H) {
      for (int h = 0; h < S1_; ++h) {
        Matrix Q(S0_, S0_);
        for (int r = 0; r < S0_; ++r) {
          Q.row(r) = rdirichlet(nu0);
        }
        Q.last_row() = 0.0;
        Q.last_row().back() = 1.0;
        mix(H, h)->set_Q(Q);
        Vector pi0 = rdirichlet(nu0);
        pi0.back() = 0;
        pi0 /= pi0.sum();
        mix(H, h)->set_pi0(pi0);
      }
    }
  }
  //----------------------------------------------------------------------
  double NestedHmm::pdf(const Ptr<Data> &dp, bool logscale) const {
    double ans = fwd(DAT(dp));
    return logscale ? ans : exp(ans);
  }
  //----------------------------------------------------------------------
  void NestedHmm::check_filter_size(int nevents) const {
    if (P.size() < nevents) P.resize(nevents);
  }
  //----------------------------------------------------------------------
  void NestedHmm::fill_logd(const Ptr<Event> &dp) const {
    int i = 0;
    for (int H = 0; H < S2_; ++H) {
      for (int h = 0; h < S1_; ++h) {
        logd_[i++] = logp(dp, H, h);
      }
    }
  }
  //----------------------------------------------------------------------
  double NestedHmm::logp(const Ptr<Event> &event, int H, int h) const {
    return mix(H, h)->pdf(*event, true);
  }
  //----------------------------------------------------------------------
  void NestedHmm::fill_big_Q() const {
    // fills logpi0_ (for first event ever)
    //       logQ1_  (for transitions to the first event in a new session)
    //       logQ2_  (for transitions within a session)

    // pi0_ looks like:     phi2[1] * vector( phi1[H=1] )
    //                      phi2[2] * vector( phi1[H=2] )
    //                                ...

    // Q1 and Q2 have block structure.  Each block refers to an H transition,
    // Let stack(vector) be a square matrix where each row is a copy of 'vector'

    // Q1 looks like
    //        Phi2[1,1] * stack(phi1[H=1]), Phi2[1,2] * stack(phi1[H=2]), ...
    //        Phi2[2,1] * stack(phi1[H=1]), Phi2[2,2] * stack(phi1[H=2]), ...
    //                           ...                      ...

    // Q2 is a block diagonal matrix where block s = Phi1[H=s]

    int S = S1_ * S2_;

    if (logpi0_.size() != S) logpi0_.resize(S);
    if (logd_.size() != S) logd_.resize(S);

    if (logQ1_.nrow() != S || logQ1_.ncol() != S) logQ1_.resize(S, S);
    logQ1_ = 0;

    if (logQ2_.nrow() != S || logQ2_.ncol() != S) logQ2_.resize(S, S);
    logQ2_ = 0;

    int i = 0;
    const Vector &phi2(session_model()->pi0());
    std::vector<Matrix> stacked_phi1(S2_);
    for (int H = 0; H < S2_; ++H) {
      const Vector &pi0(event_model(H)->pi0());
      Matrix Pi0(S1_, S1_);
      for (int h = 0; h < S1_; ++h) Pi0.row(h) = pi0;
      stacked_phi1[H] = Pi0;
    }

    const Matrix &Phi2(session_model_->Q());

    for (int H = 0; H < S2_; ++H) {
      VectorView pi_H(logpi0_, i, S1_);
      pi_H = phi2[H] * event_model(H)->pi0();

      SubMatrix Q2(logQ2_, i, i + S1_ - 1, i, i + S1_ - 1);
      Q2 = event_model(H)->Q();

      int j = 0;
      for (int HH = 0; HH < S2_; ++HH) {
        SubMatrix Q1(logQ1_, i, i + S1_ - 1, j, j + S1_ - 1);
        Q1 = Phi2(H, HH) * stacked_phi1[HH];  // scalar times matrix
        j += S1_;
      }
      i += S1_;
    }

    logQ1_ = log(logQ1_);
    logQ2_ = log(logQ2_);
    logpi0_ = log(logpi0_);
  }
  //----------------------------------------------------------------------
  std::vector<Ptr<Sufstat> > NestedHmm::suf_vec() const {
    std::vector<Ptr<Sufstat> > ans;
    ans.push_back(session_model()->suf());
    int S2 = this->S2();
    int S1 = this->S1();
    for (int H = 0; H < S2; ++H) {
      ans.push_back(event_model(H)->suf());
      for (int h = 0; h < S1; ++h) {
        ans.push_back(mix(H, h)->suf());
      }
    }
    return ans;
  }

  // transition probability matrix in (h,y) space
  Matrix NestedHmm::augmented_Q(int H) const {
    int S0 = this->S0();
    int S1 = this->S1();
    int S = S0 * S1;
    Matrix ans(S, S);
    const Matrix &Phi1 = event_model(H)->Q();

    for (int h0 = 0; h0 < S1; ++h0) {
      for (int h1 = 0; h1 < S1; ++h1) {
        SubMatrix Q(ans, h0 * S0, (h0 + 1) * S0 - 1, h1 * S0,
                    (h1 + 1) * S0 - 1);
        Q = Phi1(h0, h1) * mix(H, h1)->Q();
      }
    }
    return ans;
  }

  Vector NestedHmm::augmented_pi0(int H) const {
    int S0 = this->S0();
    int S1 = this->S1();
    int S = S0 * S1;
    Vector ans(S);
    const Vector &event_pi0(event_model(H)->pi0());
    for (int h = 0; h < S1; ++h) {
      BOOM::VectorView tmp(ans, h * S0, S0);
      tmp = mix(H, h)->pi0() * event_pi0[h];
    }
    return ans;
  }

}  // namespace BOOM
