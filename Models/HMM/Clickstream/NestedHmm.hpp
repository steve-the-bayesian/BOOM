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

#ifndef CLICKSTREAM_MODEL_HPP
#define CLICKSTREAM_MODEL_HPP

#include "LinAlg/Selector.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "Models/MarkovModel.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/PosteriorSamplers/MarkovConjSampler.hpp"
#include "Models/PosteriorSamplers/MarkovConjShrinkageSampler.hpp"
#include "distributions/rng.hpp"

#include "Models/HMM/Clickstream/Stream.hpp"

namespace BOOM {

  class NestedHmm : public CompositeParamPolicy,
                    public IID_DataPolicy<Clickstream::Stream>,
                    public PriorPolicy {
   public:
    typedef Clickstream::Event Event;
    typedef Clickstream::Session Session;
    typedef Clickstream::Stream Stream;

    NestedHmm(const std::vector<Ptr<Stream> > &streams, int S2, int S1);
    NestedHmm(int S2, int S1, int S0);
    NestedHmm *clone() const override;

    // The mixture component for state H, h.
    const Ptr<MarkovModel> &mix(int H, int h);
    const Ptr<MarkovModel> &mix(int H, int h) const;

    // The model for latent state transitions between events.
    Ptr<MarkovModel> event_model(int H);
    const Ptr<MarkovModel> event_model(int H) const;

    // The model for latent state transitions between sessions.
    Ptr<MarkovModel> session_model();
    const Ptr<MarkovModel> session_model() const;

    // Session level latent state dimension.
    int S2() const;

    // Event level latent state dimension.
    int S1() const;

    // Number of levels in the observed sequence of events, inluding
    // the end of session indicator
    int S0() const;

    int Nstreams() const;
    Ptr<Stream> stream(int i);

    // Encode_state and decode state map between the state spaces H* =
    // (H,h) and S* = (0, ... , S1xS2-1).  encode_state moves from H* to
    // S*, and decode_state moves back.
    int encode_state(int H, int h) const;
    void decode_state(int state, int &H, int &h) const;

    double pdf(const Ptr<Data> &dp, bool logscale) const;

    double loglike();
    double last_loglike() const;
    double last_logpost() const;
    double logpri() const override;

    void set_loglike(double);
    void set_logpost(double);

    // Fit the model (find the MLE) using an EM algorithm.
    double EM(double epsilon, bool bayes = true);

    std::ostream &write_suf(std::ostream &) const;

    // Sets the number of threads to use for data imputation.
    void set_threads(int n);

    double impute_latent_data();

    virtual std::vector<Ptr<Sufstat> > suf_vec() const;

    double fwd_bkwd(bool bayes = false, bool find_mode = true);
    double fwd(const Ptr<Stream> &u) const;
    void bkwd_sampling(const Ptr<Stream> &u);
    void bkwd_smoothing(const Ptr<Stream> &u);

    virtual void complete_data_mode(bool bayes);
    virtual double logp(const Ptr<Event> &event, int H, int h) const;
    virtual void update(int H, int h, const Ptr<Event> &event);
    virtual void update_mixture(int H, int h, const Ptr<Event> &event,
                                double prob);
    virtual void randomize_starting_values();

    // computes the probability that a conversion occurs before the end
    // of a session.  conv_state[0..S0-1] is the observed state defining
    // a conversion.  return value is an S2 vector of S1*(S0-2) X S1*2
    // matrices.  Element [H][(h0,y0)][(h1,y1)] is the conditional
    // probability of being absorbed into state (h1,y1) given session
    // type H and initial state (h0,y0).
    //  std::vector<Matrix> conditional_conversion_probs(
    //      const BOOM::include &abs)const;

    // initial distribution and transition matrix in (h,y) space
    Matrix augmented_Q(int H) const;
    Vector augmented_pi0(int H) const;

    void print_params(std::ostream &out) const;  // for debugging
    void print_event(std::ostream &out, const char *msg, const Ptr<Stream> &u,
                     const Ptr<Session> &session, const Ptr<Event> &event,
                     int event_number) const;
    void print_filter(std::ostream &out, int j) const;
    //------------------------------------------------------------

    // Returns a Matrix, with rows corresponding to sessions, and
    // columns corresponding to session types, giving the posterior
    // probability that each session belongs to each session type.
    // The elements in a each row of the matrix will sum to 1.
    //
    // Args:
    //   stream:  The stream for which the distribution is desired.
    Matrix report_session_type_distribution(const Ptr<Stream> &stream) const;

    // This function is to facilitate burn-in.  It removes all
    // elements from session_type_distribution_.  They will be
    // automatically regenerated the next time bkwd_sampling is
    // called.
    void clear_session_type_distribution();

   private:
    const int S0_;  // observed data size, including the EOS marker
    const int S1_;  // number of event types
    const int S2_;  // number of session types

    // Each entry in the map is a matrix, with rows correspoding to
    // the session in the matrix, and columns to session types.  Each
    // time the MCMC algorithm assigns session type H to session s,
    // the (s, H) matrix element will be incremented.  When
    // report_session_distribution() is called, each element will be
    // divided by the number of MCMC iterations to get a Monte Carlo
    // estimate of the session distribution.
    //
    // Note: This estimate is vulnerable to label switching among the
    // session type latent variables (the H's).  It is invariant to
    // label switching among the event-level latent variables (the
    // h's).
    std::map<Ptr<Stream>, Matrix> session_type_distribution_;

    Ptr<MarkovModel> session_model_;
    std::vector<Ptr<MarkovModel> > event_model_;
    std::vector<std::vector<Ptr<MarkovModel> > > mix_;

    Ptr<UnivParams> loglike_;
    Ptr<UnivParams> logpost_;

    // stuff for the filter
    mutable std::vector<Matrix> P;
    mutable Vector pi_;
    mutable Vector logpi0_;
    mutable Vector logd_;
    const Vector one_;      // A vector of 1's
    mutable Matrix logQ1_;  // for the first obs in a session
    mutable Matrix logQ2_;  // for the subsequent observations

    RNG rng_;

    std::vector<Ptr<NestedHmm> > workers_;
    void setup();
    void pass_params_to_workers();
    void fill_logd(const Ptr<Event> &event) const;
    void fill_big_Q() const;
    void start_thread_imputation();
    void start_thread_em();
    double initialize(const Ptr<Event> &event) const;
    void check_filter_size(int n) const;
    ConstVectorView get_hinit(const Vector &pi, int H) const;
    Vector get_Hinit(const Vector &pi) const;
    ConstSubMatrix get_htrans(const Matrix &P, int H) const;
    ConstSubMatrix get_block(const Matrix &P, int H1, int H2) const;
    Matrix get_Htrans(const Matrix &P) const;
    double fwd_bkwd_with_threads(bool bayes = false, bool find_mode = true);
    double impute_latent_data_with_threads();

    double collect_threads();
    void clear_client_data();
    void allocate_data_to_workers();
    void add_worker(const Ptr<NestedHmm> &w);
    void clear_workers();
    RNG &rng() { return rng_; }
  };

  class NestedHmmDataImputer : public BOOM::PosteriorSampler {
   public:
    explicit NestedHmmDataImputer(NestedHmm *mod, RNG &seeding_rng = GlobalRng::rng)
        : PosteriorSampler(seeding_rng), m(mod) {}
    void draw() override { m->impute_latent_data(); }
    double logpri() const override { return 0; }

   private:
    NestedHmm *m;
  };

}  // namespace BOOM

#endif  // CLICKSTREAM_MODEL_HPP
