/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include "Models/PointProcess/MarkovModulatedPoissonProcess.hpp"

#include <algorithm>
#include <iterator>  // for back_inserter
#include <vector>

#include "cpputil/lse.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    // Some pure functions just used for local implementation.

    // Returns true if 'vector' fails to include 'thing'.
    template <class T>
    bool omits(const std::vector<T> &vector, const T &thing) {
      return std::find(vector.begin(), vector.end(), thing) == vector.end();
    }

    // Returns true if 'vector' includes 'thing'.
    template <class T>
    bool contains(const std::vector<T> &vector, const T &thing) {
      return !omits(vector, thing);
    }

    // TODO(stevescott): Check out omits() and contains() in a profiler.
    // If they are a bottleneck then consider using sorted ranges instead.

    // Args:
    //   P:  A reference to a Matrix full of un-normalized log probabilities.
    // Returns:
    //   Safely exponentiates everything in P, normalizes P so that
    //   the sum over all its elements is 1, and returns the log of
    //   the normalizing constant (which is the contribution of the
    //   most recent data point to log likelihood).
    double normalize_filter(Matrix &P) {
      double max_log = max(P);
      P -= max_log;
      P.exp();
      double total = sum(P);
      P /= total;
      return max_log + log(total);
    }

    // Return a vector of naked 'dumb' pointers from a vector of BOOM
    // Ptr's.
    std::vector<PoissonProcess *> dumb(
        const std::vector<Ptr<PoissonProcess> > &arg) {
      std::vector<PoissonProcess *> ans;
      ans.reserve(arg.size());
      for (int i = 0; i < arg.size(); ++i) {
        ans.push_back(arg[i].get());
      }
      return ans;
    }
  }  // namespace
  //======================================================================
  namespace MmppHelper {
    // The HmmState does not own the PoissonProcess * pointers that
    // it is passed.  Each should be owned by a Ptr external to this
    // class.
    HmmState::HmmState(const std::vector<PoissonProcess *> &processes)
        : active_processes_(processes), id_number_(-1) {
      if (processes.empty()) {
        report_error("Empty vector passed to HmmState constructor.");
      }
      std::sort(active_processes_.begin(), active_processes_.end());
    }

    // The id_number is this state's position in the vector of
    // hmm_states_ in the managing MMPP.
    int HmmState::id_number() const { return id_number_; }
    void HmmState::set_id_number(int id) { id_number_ = id; }

    // Two states compare equal if they have the same set of active
    // processes.  The comparison is fast because active_processes_
    // is kept sorted.
    bool HmmState::operator==(const HmmState &rhs) const {
      return active_processes_ == rhs.active_processes_;
    }

    // The number of processes constituting the state.
    int HmmState::number_of_active_processes() const {
      return active_processes_.size();
    }

    // Accessor for the set of active processes.  Note that the
    // constructor sorts them, so the order here may be different
    // than the order the processes were in when they were passed to
    // the constructor.
    const std::vector<PoissonProcess *> &HmmState::active_processes() const {
      return active_processes_;
    }

    // Add or remove processes to the set of active processes
    // defining the state.  Adding processes that are already part
    // of the state will not result in duplicates.  Removing
    // processes that are not part of the state is safe and results
    // in no action.
    void HmmState::add_processes(
        std::vector<PoissonProcess *> additional_processes) {
      std::sort(additional_processes.begin(), additional_processes.end());
      std::vector<PoissonProcess *> result;
      std::set_union(active_processes_.begin(), active_processes_.end(),
                     additional_processes.begin(), additional_processes.end(),
                     std::back_inserter(result));
      active_processes_ = result;
    }

    void HmmState::remove_processes(std::vector<PoissonProcess *> to_remove) {
      std::sort(to_remove.begin(), to_remove.end());
      std::vector<PoissonProcess *> result;
      std::set_difference(active_processes_.begin(), active_processes_.end(),
                          to_remove.begin(), to_remove.end(),
                          std::back_inserter(result));
      active_processes_ = result;
      if (active_processes_.empty()) {
        report_error("Empty HmmState after call to remove_processes.");
      }
    }

    const std::vector<HmmState *> &HmmState::potential_outgoing_transitions()
        const {
      return potential_outgoing_transitions_;
    }

    const std::vector<HmmState *> &HmmState::potential_incoming_transitions()
        const {
      return potential_incoming_transitions_;
    }

    // The set of processes that could be responsible for a transition
    // from *this to *destination.  The result could be a singleton,
    // or it could be empty.
    const std::vector<PoissonProcess *> &HmmState::processes_transitioning_to(
        const HmmState *destination) const {
      TransitionResponsibilityMap::const_iterator it =
          responsible_for_transition_to_.find(destination);
      if (it == responsible_for_transition_to_.end()) {
        report_error("Incomplete transition responsibility map");
      }
      return it->second;
    }

    // Inform the object that it is possible to transition to *this
    // from *previous_state.
    void HmmState::add_transition_from(HmmState *previous_state) {
      std::vector<HmmState *>::iterator it = std::lower_bound(
          potential_incoming_transitions_.begin(),
          potential_incoming_transitions_.end(), previous_state);
      if (it == potential_incoming_transitions_.end() ||
          *it != previous_state) {
        potential_incoming_transitions_.insert(it, previous_state);
      }
    }

    // Inform the object that it is possible to transition from *this
    // to *next_state.
    // Args:
    //   next_state: the state that *this transitions to.
    //   source_of_transition: The PoissonProcess whose event causes
    //     the transition to next_state.
    void HmmState::add_transition_to(HmmState *next_state,
                                     PoissonProcess *responsible_process) {
      std::vector<HmmState *>::iterator it =
          std::lower_bound(potential_outgoing_transitions_.begin(),
                           potential_outgoing_transitions_.end(), next_state);
      if (it == potential_outgoing_transitions_.end() || *it != next_state) {
        potential_outgoing_transitions_.insert(it, next_state);
      }
      responsible_for_transition_to_[next_state].push_back(responsible_process);
    }

    //======================================================================
    // Args:
    //   processes: The Poisson processes to be managed, including any
    //     birth or death processes.
    //   mixture_components: The mixture components responsible for
    //     the marks attached to the data.  If there is no
    //     mixture_component part of the model then this vector can be
    //     empty.  If it is not empty it should be the same length as
    //     the processes argument.  Some elements of the vector can be
    //     repeated if the same mixture component is shared by
    //     multiple processes.
    ProcessInfo::ProcessInfo(
        const std::vector<PoissonProcess *> &processes,
        const std::vector<MixtureComponent *> &mixture_components)
        : neginf_(negative_infinity()), processes_(processes) {
      std::sort(processes_.begin(), processes_.end());
      for (int i = 0; i < processes_.size(); ++i) {
        process_id_[processes_[i]] = i;
      }
      // Note: at this point the internal storage processes_ and the
      // argument 'processes' are no longer in the same order.  Use the
      // constructor argument to maintain the mapping between processes
      // and the associated mixture components.

      if (!mixture_components.empty()) {
        if (mixture_components.size() != processes.size()) {
          report_error(
              "Arguments are of different sizes in "
              "ProcessInfo constructor");
        }

        std::vector<MixtureComponent *> mixture_storage(mixture_components);
        std::sort(mixture_storage.begin(), mixture_storage.end());
        std::unique_copy(mixture_storage.begin(), mixture_storage.end(),
                         back_inserter(minimal_mixture_components_));

        // Now minimal_mixture_components_ is sorted.  We still
        // need to build the map between process_id
        // mixture_component_id.

        std::map<PoissonProcess *, MixtureComponent *> mixture_component_map;
        for (int i = 0; i < processes.size(); ++i) {
          mixture_component_map[processes[i]] = mixture_components[i];
        }
        mixture_component_id_.resize(processes.size());
        for (int i = 0; i < processes_.size(); ++i) {
          PoissonProcess *process = processes_[i];
          MixtureComponent *mix = mixture_component_map[process];
          std::vector<MixtureComponent *>::iterator it =
              std::lower_bound(minimal_mixture_components_.begin(),
                               minimal_mixture_components_.end(), mix);
          if (*it != mix) {
            report_error(
                "Error finding mixture components in "
                "ProcessInfo constructor.");
          }
          int position = it - minimal_mixture_components_.begin();
          mixture_component_id_[i] = position;
        }
      }
    }

    //----------------------------------------------------------------------
    // Evaluate the cumulative_hazard for the interval [t-1, t], the
    // instantaneous event rate at time t, and mixture component log
    // density for any marks at time t.
    // Args:
    //   data: The point process to evaluate.
    //   source: A vector of potential processes that could have
    //     been responsible for the event at time t.  In typical
    //     applications this will be unknown, in which case an empty
    //     vector can be supplied.  If a vector is supplied then the
    //     instantaneous event rates for the processes not listed in
    //     'source' will be set to zero.
    void ProcessInfo::evaluate(const PointProcess &data,
                               const SourceVector &source) {
      cumulative_hazard_.resize(processes_.size(), data.number_of_events());
      log_event_rate_.resize(processes_.size(), data.number_of_events());
      if (!(minimal_mixture_components_.empty())) {
        logp_.resize(minimal_mixture_components_.size(),
                     data.number_of_events());
      }

      bool no_source = source.empty();
      for (int t = 0; t < data.number_of_events(); ++t) {
        const DateTime &t0(t == 0 ? data.window_begin()
                                  : data.event(t - 1).timestamp());
        const DateTime &t1(data.event(t).timestamp());
        for (int i = 0; i < processes_.size(); ++i) {
          PoissonProcess *process = processes_[i];
          cumulative_hazard_(i, t) = process->expected_number_of_events(t0, t1);
          if (no_source || source[t].empty() || contains(source[t], process)) {
            log_event_rate_(i, t) = log(process->event_rate(t1));
          } else {
            log_event_rate_(i, t) = neginf_;
          }
        }

        if (data.event(t).has_mark() &&
            !(minimal_mixture_components_.empty())) {
          const Data *y = data.event(t).mark();
          for (int i = 0; i < minimal_mixture_components_.size(); ++i) {
            logp_(i, t) = minimal_mixture_components_[i]->pdf(y, true);
          }
        }
      }
    }

    // If the call to 'evaluate' indicated that 'process' was not a
    // possible source of the event at time 't' then this function
    // returns -infinity.  Otherwise it returns the log of the event
    // rate for 'process' at the timestamp of event 't'.
    double ProcessInfo::log_event_rate(const PoissonProcess *process,
                                       int t) const {
      return log_event_rate_(process_id(process), t);
    }

    // Returns the log density for the mixture component associated
    // with 'process' evalutated on event t.  If that event contains
    // no marks or if no mixture components are supplied then 0 is
    // returned.
    double ProcessInfo::mixture_log_likelihood(const PoissonProcess *process,
                                               int t) const {
      if (minimal_mixture_components_.empty()) {
        return 0.0;
      }
      return logp_(mixture_component_id_[process_id(process)], t);
    }

    // The sum of the cumulative hazards of the active processes in
    // state from time t-1 to t.  This return values is not affected
    // by the presence or absence of the 'source' argument to
    // 'evaluate'.
    double ProcessInfo::conditional_cumulative_hazard(const HmmState *state,
                                                      int t) const {
      double ans = 0;
      const std::vector<PoissonProcess *> &processes(state->active_processes());
      for (int i = 0; i < processes.size(); ++i) {
        int pid = process_id(processes[i]);
        ans += cumulative_hazard_(pid, t);
      }
      return ans;
    }

    int ProcessInfo::process_id(const PoissonProcess *process) const {
      std::unordered_map<const PoissonProcess *, int>::const_iterator it =
          process_id_.find(process);
      if (it == process_id_.end()) {
        report_error("Unknown process passed to ProcessInfo::process_id().");
      }
      return it->second;
    }
  }  // namespace MmppHelper
  //======================================================================
  typedef MarkovModulatedPoissonProcess MMPP;

  MMPP::MarkovModulatedPoissonProcess() {}

  MMPP::MarkovModulatedPoissonProcess(const MMPP &rhs)
      : Model(rhs), ParamPolicy(rhs), DataPolicy(rhs) {
    // What to do with component processes?  Clone them?  Copy them?
    // Probably clone them.
    report_error("MMPP copy constructor not yet implemented.");
  }

  MMPP *MMPP::clone() const { return new MMPP(*this); }

  // Adds 'process' as a component process to the MMPP.  Every
  // component process must be registered with the MMPP using this
  // function.  That includes processes mentioned in 'spawns' or
  // 'kills'.  Each process must appear as the 'process' argument
  // exactly once, but can appear in the 'spawns' or 'kills' list
  // for multiple other processes.
  //
  // NOTE: after adding your last component process, call
  //       make_hmm_states() before doing anything else with this
  //       model.
  //
  // Args:
  //   process:  The component process to add.
  //   spawns: A vector containing zero or more processes that are
  //     activated when 'process' generates an event.
  //   kills: A vector containing zero or more process that are
  //     deactivated when 'process' generates an event.  For birth or
  //     death processes, 'processes' can be included in this vector.
  //   emits: A model describing the marks that 'process' emits.
  void MMPP::add_component_process(const Ptr<PoissonProcess> &process,
                                   std::vector<Ptr<PoissonProcess> > spawns,
                                   std::vector<Ptr<PoissonProcess> > kills,
                                   const Ptr<MixtureComponent> &emits) {
    if (component_processes_.empty()) {
      have_mixture_components_ = !!emits;
    }

    // Report a problem if you have mixture components and are passed
    // a NULL, or if you don't have mixture components and are passed
    // a non-NULL.
    if (!!emits != have_mixture_components_) {
      report_error(
          "Error in MarkovModulatedPoissonProcess::add_component_process\n"
          "Some components have an associated mixture component, and "
          "some do not.");
    }

    check_first_entry(process);
    int pid = process_id_.size();
    process_id_[process.get()] = pid;

    check_for_new_process(process);
    for (int i = 0; i < spawns.size(); ++i) {
      check_for_new_process(spawns[i]);
    }
    for (int i = 0; i < kills.size(); ++i) {
      check_for_new_process(kills[i]);
    }

    spawns_[process.get()] = dumb(spawns);
    kills_[process.get()] = dumb(kills);

    if (!!emits) {
      check_for_new_mixture_component(emits);
      emits_[process.get()] = emits.get();
    }
  }

  // After all component processes have been added using
  // add_component_process() the user should call make_hmm_states()
  // to finalize construction.
  // Args:
  //   initial_state_elements: The collection of processes defining
  //      one of the states in the HMM.  This is not necessarily the
  //      'initial_state' in the sense of the state at time 0.  It
  //      is just a seed to use for determining which states of
  //      nature need to be considered.
  void MMPP::make_hmm_states(
      const std::vector<Ptr<PoissonProcess> > &initial_state_elements) {
    check_that_all_processes_have_been_registered();

    NEW(HmmState, initial_state)(dumb(initial_state_elements));
    hmm_states_.push_back(initial_state);
    bool done = false;
    std::vector<Ptr<HmmState> > completed_states;
    while (!done) {
      done = true;
      for (int i = 0; i < hmm_states_.size(); ++i) {
        Ptr<HmmState> state = hmm_states_[i];
        if (contains(completed_states, state)) {
          break;
        }
        done = false;
        generate_new_states(state);
        completed_states.push_back(state);
      }
    }

    // Set the id number for the hmm states
    for (int i = 0; i < hmm_states_.size(); ++i) {
      hmm_states_[i]->set_id_number(i);
    }

    create_process_info();
  }

  //----------------------------------------------------------------------
  // Add data to the model.  This had to be over-ridden because the
  // matrices that keep track of the probability of activity and the
  // probability of responsibility get modified when a new process
  // is added.
  void MMPP::add_data(const Ptr<Data> &dp) {
    Ptr<PointProcess> d = DAT(dp);
    add_data(d);
  }
  void MMPP::add_data(const Ptr<PointProcess> &dp) {
    int n = dp->number_of_events();
    int nproc = component_processes_.size();
    Matrix activity(nproc, n + 1, 0.0);
    Matrix responsibility(nproc, n, 0.0);
    probability_of_activity_.push_back(activity);
    probability_of_responsibility_.push_back(responsibility);
    DataPolicy::add_data(dp);
  }

  //----------------------------------------------------------------------
  // If a subset (possibly all) of the data are from a training set
  // where the original source is known then add the data using
  // add_supervised_data instead of add_data.  Each event
  // corresponds to a vector of PoissonProcess * pointers indicating
  // the set of possible models that could have generated the data.
  // If the source information is missing for a particular event
  // then you can use an empty vector to signal "don't know" for
  // that event.  The size of 'source' must match the number of
  // events in 'process'.  It cannot be empty.
  void MMPP::add_supervised_data(const Ptr<PointProcess> &dp,
                                 const SourceVector &source) {
    add_data(dp);
    if (dp->number_of_events() != source.size()) {
      ostringstream err;
      err << "Error in MarkovModulatedPoissonProcess::add_supervised_data."
          << endl
          << "The size of source (" << source.size() << ") does not match the"
          << " number of events in the corresponding point process ("
          << dp->number_of_events() << ")";
      report_error(err.str());
    }
    known_source_store_[dp.get()] = source;
  }

  //----------------------------------------------------------------------
  // Clears the data managed by this model.  The over-ride is
  // necessary because of internal state not managed by the
  // DataPolicy which must be cleared as well.
  void MMPP::clear_data() {
    probability_of_activity_.clear();
    probability_of_responsibility_.clear();
    DataPolicy::clear_data();
  }

  //----------------------------------------------------------------------
  // Calls clear_data() on all component processes and mixture
  // components.  Does not clear the data managed by this class.
  void MMPP::clear_client_data() {
    for (int i = 0; i < component_processes_.size(); ++i) {
      component_processes_[i]->clear_data();
    }
    for (int i = 0; i < mixture_components_.size(); ++i) {
      mixture_components_[i]->clear_data();
    }
  }

  //----------------------------------------------------------------------
  // Impute values for the latent processes using the forward
  // backward simulation algorithm.  Returns the observed-data log
  // likelihood of the current set of model parameters.
  double MMPP::impute_latent_data(RNG &rng) {
    const std::vector<Ptr<PointProcess> > &data(dat());
    double loglike = 0;
    clear_client_data();
    for (int i = 0; i < data.size(); ++i) {
      Ptr<PointProcess> process(data[i]);
      const SourceVector &source(known_source_store_[process.get()]);
      loglike += filter(*process, source);
      backward_sampling(rng, *process, probability_of_activity_[i],
                        probability_of_responsibility_[i]);
    }
    last_loglike_ = loglike;
    return loglike;
  }

  void MMPP::burn() {
    for (int i = 0; i < probability_of_responsibility_.size(); ++i) {
      probability_of_responsibility_[i] = 0;
    }
    for (int i = 0; i < probability_of_activity_.size(); ++i) {
      probability_of_activity_[i] = 0;
    }
  }

  void MMPP::sample_complete_data_posterior() {
    for (int i = 0; i < component_processes_.size(); ++i) {
      component_processes_[i]->sample_posterior();
    }
    if (have_mixture_components_) {
      for (int i = 0; i < mixture_components_.size(); ++i) {
        mixture_components_[i]->sample_posterior();
      }
    }
  }

  // Compute the log of the prior distribution at the current values
  // of the model parameters.  logpri is implemented here so the
  // model does not have to expose the component models to the
  // posterior sampler.
  double MMPP::logpri() const {
    double ans = 0;
    for (int i = 0; i < component_processes_.size(); ++i) {
      ans += component_processes_[i]->logpri();
    }
    if (have_mixture_components_) {
      for (int i = 0; i < mixture_components_.size(); ++i) {
        ans += mixture_components_[i]->logpri();
      }
    }
    return ans;
  }

  //----------------------------------------------------------------------
  // Forward filter the supplied process.
  // Args:
  //   process:  The point process to be filtered.
  //   source: Either an empty vector, or a vector-of-vectors with
  //     size process.number_of_events().  Each element indicates
  //     the set of component processes that might have produced the
  //     corresponding event in 'process'.  An empty elment
  //     indicates that any process is possible.  Otherwise the
  //     element is a vector of PoissonProcess pointers.
  //
  // Returns:
  //   The log likelihood of the process, given current model parameters.
  //
  // Details:
  //   On exit, pi0_ contains the marginal distribution of the final
  //   HmmState corresponding to the last event in process, and
  //   filter_[t] contains the joint distribution of HMM states t-1
  //   (rows) and t (columns).
  double MMPP::filter(const PointProcess &process, const SourceVector &source) {
    if (process.number_of_events() == 0) return 0;
    bool have_source = !source.empty();
    if (have_source && source.size() != process.number_of_events()) {
      ostringstream err;
      err << "Vector of known sources is not the same size as the PointProcess"
          << " in MMPP::filter." << endl;
      report_error(err.str());
    }
    process_info_->evaluate(process, source);
    double loglike = initialize_filter(process);
    for (int i = 0; i < process.number_of_events(); ++i) {
      loglike += fwd_1(i, *process_info_);
    }
    return loglike;
  }

  //----------------------------------------------------------------------
  // Updates the state of the filter at time t to give the conditional
  // distribution of state[t-1] and state[t] given observed data to
  // time t.
  // Args:
  //   t:  The index of the event to use for updating.
  //   process_info: An ProcessInfo object containting all the
  //     relevant likelihood evalutations for interval t.
  // Returns:
  //   log p(events[t] | events[0, ..., t-1])
  double MMPP::fwd_1(int t, const ProcessInfo &process_info) {
    Matrix &P(filter_[t]);  // Do we need a sparse matrix here?
    P = negative_infinity();
    int S = hmm_state_space_size();
    for (int r = 0; r < S; ++r) {
      const HmmState *first_state = hmm_states_[r].get();
      double log_prior_hazard =
          log(pi0_[r]) -
          process_info.conditional_cumulative_hazard(first_state, t);
      typedef std::vector<HmmState *> StateVector;
      const StateVector &potential_states(
          first_state->potential_outgoing_transitions());
      for (StateVector::const_iterator it = potential_states.begin();
           it != potential_states.end(); ++it) {
        const HmmState *second_state = *it;
        int s = second_state->id_number();
        P(r, s) =
            log_prior_hazard + conditional_event_loglikelihood(
                                   t, first_state, second_state, process_info);
      }
    }
    double loglike = normalize_filter(P);
    pi0_ = one_ * P;
    return loglike;
  }

  //----------------------------------------------------------------------
  // Simulates the hidden Markov chain corresponding to 'process',
  // assuming it has just been filtered.
  // Args:
  //   rng:  A random number generator.
  //   process: The observed data corresponding to the hidden Markov
  //     process to be imputed.
  //   source: The source vector passed to 'filter'.  This is only
  //     used for error checking.
  //   probability_of_activity: A matrix, with rows corresponding to
  //     the component processes, and columns corresponding to time,
  //     indicating the probability that each process was active
  //     between times t-1 and t.
  //   probability_of_responsibility: A matrix with the same size as
  //     probability_of_activity, with element (i, t) indicating the
  //     probability that process i was active at time t.
  //
  // Details:
  //   On exit the component processes and mixture components will
  //   have the data from 'process' attributed to them according to
  //   the value of the imputed Markov chain.
  //
  //   The two probability_of_ matrices compute probabilities by
  //   counting the number of times their event occurred in the
  //   MCMC.  Thus they are maintained as integers counting the
  //   number of events seen thus far.  They are turned into
  //   probabilities by dividing by the number of MCMC iterations.
  void MMPP::backward_sampling(RNG &rng, const PointProcess &process,
                               Matrix &probability_of_activity,
                               Matrix &probability_of_responsibility) {
    int n = process.number_of_events();
    if (n >= 1) {
      int current_state = rmulti_mt(rng, pi0_);
      // Record the probability of each process being active between
      // the time of the final event and the end of the observation
      // window.
      record_activity(probability_of_activity.col(n), current_state);
      update_exposure_time(process, n, current_state);

      for (int t = n - 1; t >= 0; --t) {
        int previous_state = draw_previous_state(rng, t, current_state);
        PoissonProcess *responsible_process = sample_responsible_process(
            rng, previous_state, current_state, *process_info_, t);
        update_exposure_time(process, t, previous_state);
        const PointProcessEvent &event(process.event(t));
        responsible_process->add_event(event.timestamp());
        if (event.has_mark() && have_mixture_components_) {
          MixtureComponent *mix = emits_[responsible_process];
          mix->add_data(event.mark_ptr());
        }

        // Record activity and responsibility.
        record_activity(probability_of_activity.col(t), previous_state);
        ++probability_of_responsibility(process_id(responsible_process), t);
        current_state = previous_state;
      }
    }
  }

  //----------------------------------------------------------------------
  int MMPP::hmm_state_space_size() const { return hmm_states_.size(); }

  //----------------------------------------------------------------------
  int MMPP::number_of_processes() const { return component_processes_.size(); }

  //----------------------------------------------------------------------
  // Return the probability that each state was active at time the
  // time of event t.
  // Args:
  //   data_series_number: The index of the data series managed by
  //     the model.  Series 0 is the first one added using
  //     add_data().  Series 1 is the second, and so on.
  // Returns:
  //   A matrix containing the probability that each state was
  //   active at time the time of event t.  Rows are component
  //   processes, columns are times.
  Matrix MMPP::probability_of_activity(int data_series_number) const {
    Matrix ans(probability_of_activity_[data_series_number]);
    // We need to divide ans by the number of MCMC iterations.  Each
    // probability_of_responsibility_ matrix has been incremented by 1
    // in exactly one column for each MCMC iteration, so we can get
    // the the number of MCMC iterations by summing any column in the
    // matrix.  It always has a column 0.
    double mcmc_iteration_count =
        sum(probability_of_responsibility_[data_series_number].col(0));
    if (mcmc_iteration_count > 0) ans /= mcmc_iteration_count;
    return ans;
  }

  //----------------------------------------------------------------------
  // Return the probability that each component process was
  // responsible for the event at time t.
  // Args:
  //   data_series_number: The index of the data series managed by
  //     the model.  Series 0 is the first one added using
  //     add_data().  Series 1 is the second, and so on.
  // Returns:
  //   A matrix containing the probability that each component
  //   process was responsible for the event at time t.  Rows are
  //   component processes, columns are times.
  Matrix MMPP::probability_of_responsibility(int data_series_number) const {
    Matrix ans(probability_of_responsibility_[data_series_number]);
    double mcmc_iteration_count = sum(ans.col(0));
    if (mcmc_iteration_count > 0) ans /= mcmc_iteration_count;
    return ans;
  }

  //----------------------------------------------------------------------
  // Return log(lambda_1(t) * p_1(y) + lambda_2(t) * p_2(y) + ...),
  // where
  //  * lambda_j(t) is the instantaneous event rate at the time
  //    of event t,
  //  * p_j(y) is the likelihood of the mark value y under mixture
  //    component j., and
  //  * the sum is over all possible processes that could have been
  //    responsible for the transition from first_state to
  //    second_state.
  double MMPP::conditional_event_loglikelihood(
      int t, const HmmState *first_state, const HmmState *second_state,
      const ProcessInfo &process_info) const {
    // Step 1: get list of potential processes that could have
    // produced the transition from first_state to second_state.
    const std::vector<PoissonProcess *> &potential_culprits(
        first_state->processes_transitioning_to(second_state));
    int nproc = potential_culprits.size();
    // Step 2: evaluate conditional event rates and mixture densities.
    double ans = 0;
    if (nproc == 1) {
      const PoissonProcess *process = potential_culprits[0];
      ans = process_info.log_event_rate(process, t) +
            process_info.mixture_log_likelihood(process, t);
    } else if (nproc > 1) {
      mutable_workspace_.resize(nproc);
      for (int i = 0; i < nproc; ++i) {
        const PoissonProcess *process = potential_culprits[i];
        mutable_workspace_[i] = process_info.log_event_rate(process, t) +
                                process_info.mixture_log_likelihood(process, t);
      }
      ans = lse(mutable_workspace_);
    } else if (nproc < 1) {
      report_error(
          "potential_culprits was empty in "
          "MMPP::conditional_event_loglikelihood.");
    }
    return ans;
  }

  //----------------------------------------------------------------------
  // Add the exposure time between events t-1 and t from 'process'
  // to the active processes from the hmm_state indexed by
  // 'previous_state'.
  void MMPP::update_exposure_time(const PointProcess &process, int t,
                                  int previous_state) {
    const DateTime &then(t > 0 ? process.event(t - 1).timestamp()
                               : process.window_begin());
    const DateTime &now(t < process.number_of_events()
                            ? process.event(t).timestamp()
                            : process.window_end());
    std::vector<PoissonProcess *> active_processes(
        hmm_states_[previous_state]->active_processes());
    for (int i = 0; i < active_processes.size(); ++i) {
      active_processes[i]->add_exposure_window(then, now);
    }
  }

  //----------------------------------------------------------------------
  // Draw the previous HMM state given the index of the current
  // state.
  // Args:
  //   rng:  A uniform random number generator.
  //   t:  The time index corresponding to 'current_state'.
  //   current_state:  The index of the HMM state at time t.
  int MMPP::draw_previous_state(RNG &rng, int t, int current_state_id) {
    const HmmState *current_state = hmm_states_[current_state_id].get();
    const std::vector<HmmState *> &potential_values(
        current_state->potential_incoming_transitions());
    if (potential_values.size() == 1) {
      return potential_values.front()->id_number();
    }
    mutable_workspace_.resize(potential_values.size());
    VectorView probs(filter_[t].col(current_state_id));
    for (int i = 0; i < potential_values.size(); ++i) {
      mutable_workspace_[i] = probs[potential_values[i]->id_number()];
    }
    mutable_workspace_.normalize_prob();
    int which_potential_value = rmulti_mt(rng, mutable_workspace_);
    return potential_values[which_potential_value]->id_number();
  }

  // Return the PoissonProcess responsible for the transition from
  // 'previous_state' to 'current_state.'
  // Args:
  //   rng: A uniform random number generator.
  //   previous_state:  The index of the HMM state at time t-1.
  //   current_state:  The index of the HMM state at time t.
  //   process_info: The ProcessInfo object that has calculated all
  //     the likelihood contributions from the component processes
  //     and mixture components.
  //   t:  The index of the current event in question.
  // Returns:
  //   The component process responsible for the transition from
  //   previous_state to current_state, as sampled from its full
  //   conditional distribution.
  PoissonProcess *MMPP::sample_responsible_process(
      RNG &rng, int previous_state_id, int current_state_id,
      const ProcessInfo &process_info, int t) {
    const HmmState *previous_state(hmm_states_[previous_state_id].get());
    const HmmState *current_state(hmm_states_[current_state_id].get());
    const std::vector<PoissonProcess *> &potential_culprits(
        previous_state->processes_transitioning_to(current_state));

    if (potential_culprits.size() == 1) {
      return potential_culprits[0];
    }
    mutable_workspace_.resize(potential_culprits.size());
    for (int i = 0; i < potential_culprits.size(); ++i) {
      mutable_workspace_[i] =
          process_info.log_event_rate(potential_culprits[i], t) +
          process_info.mixture_log_likelihood(potential_culprits[i], t);
    }
    mutable_workspace_.normalize_logprob();
    int index = rmulti_mt(rng, mutable_workspace_);
    return potential_culprits[index];
  }

  //----------------------------------------------------------------------
  // Records the current probability that each process is active.
  // Args:
  //   probability_of_activity: A vector with indices corresponding
  //     to the component processes in the MMPP.
  //   hmm_state_id:  An integer corresponding to particular hmm_state.
  // Details:
  //   This function increments the probability_of_activity vector
  //   by 1 in each element that corresponds to an active process in
  //   hmm_state_id.  The idea is that at the end of an MCMC run you
  //   can divide probability_of_activity by the number of MCMC
  //   iterations to get the marginal probability that each state is
  //   active at each time point, averaging over the parameters.
  void MMPP::record_activity(VectorView probs, int state) {
    const HmmState *hmm_state = hmm_states_[state].get();
    const std::vector<PoissonProcess *> &active_processes(
        hmm_state->active_processes());
    for (int s = 0; s < active_processes.size(); ++s) {
      ++probs[process_id(active_processes[s])];
    }
  }

  //======================================================================
  // Private member functions below this line.

  //----------------------------------------------------------------------
  // Checks to make sure that all the processes listed in
  // component_processes_ have corresponding entries in spawns_.
  // Because the only way to get an entry in spawns_ is through a
  // call to add_component_process(), this is a check that each of
  // the processes mentioned in a spawns or kills argument to
  // add_component_process was itself registered using
  // add_component_process.
  void MMPP::check_that_all_processes_have_been_registered() {
    for (int i = 0; i < component_processes_.size(); ++i) {
      PoissonProcess *process = component_processes_[i].get();
      if (spawns_.find(process) == spawns_.end()) {
        report_error(
            "At least one process listed in 'spawns' or "
            "'kills' was not added using add_component_process().");
      }
    }
  }
  //----------------------------------------------------------------------
  // Check to see if 'process' has already been added in a call to
  // add_component_process().
  void MMPP::check_first_entry(const Ptr<PoissonProcess> &process_ptr) {
    PoissonProcess *process = process_ptr.get();
    if (spawns_.find(process) != spawns_.end()) {
      report_error(
          "A process was submitted twice to "
          "MarkovModulatedPoissonProcess::add_component_process");
    }
  }
  //----------------------------------------------------------------------
  // Check to see if 'process' is already contained in
  // component_processes_.  If it is not, then check it in with
  // ParamPolicy, and addi it to component_processes_.
  void MMPP::check_for_new_process(const Ptr<PoissonProcess> &process) {
    for (int i = 0; i < component_processes_.size(); ++i) {
      if (component_processes_[i] == process) return;
    }
    ParamPolicy::add_model(process);
    component_processes_.push_back(process);
  }

  //----------------------------------------------------------------------
  // Check to see if 'component' is already contained in
  // mixture_components_.  If it is not, then check it in with
  // ParamPolicy and add it to mixture_components_.
  void MMPP::check_for_new_mixture_component(
      const Ptr<MixtureComponent> &component) {
    if (!component) {
      return;
    }
    for (int i = 0; i < mixture_components_.size(); ++i) {
      if (mixture_components_[i] == component) return;
    }
    ParamPolicy::add_model(component);
    mixture_components_.push_back(component);
  }

  //----------------------------------------------------------------------
  // If potential_state points to an object equivalent to one that
  // exists in the vector of hmm states then return the object from
  // hmm_states.  If not, then insert potential_state in hmm_states,
  // and return a Ptr to the (common) object.
  Ptr<MmppHelper::HmmState> MMPP::check_for_new_hmm_state(
      const Ptr<HmmState> &potential_state) {
    for (int i = 0; i < hmm_states_.size(); ++i) {
      if (*(hmm_states_[i]) == *(potential_state)) {
        return hmm_states_[i];
      }
    }
    // If you get here then 'potential_state' was not found in
    // hmm_states_.  Add it, and return a pointer.
    hmm_states_.push_back(potential_state);
    return potential_state;
  }

  //----------------------------------------------------------------------
  // Generate all the states that can be transitioned to from
  // 'state'.
  // Discover the state that is produced when each process in 'state'
  // produces an event.  Record any new hmm_states that are discovered
  // in hmm_states_, and add the potential from state to the new
  // state.
  void MMPP::generate_new_states(const Ptr<HmmState> &state) {
    for (int i = 0; i < state->number_of_active_processes(); ++i) {
      NEW(HmmState, potential_state)(state->active_processes());
      PoissonProcess *process = state->active_processes()[i];
      potential_state->add_processes(spawns_[process]);
      potential_state->remove_processes(kills_[process]);
      Ptr<HmmState> new_state = check_for_new_hmm_state(potential_state);

      state->add_transition_to(new_state.get(), process);
      new_state->add_transition_from(state.get());
    }
  }

  int MMPP::process_id(const PoissonProcess *process) const {
    std::unordered_map<const PoissonProcess *, int>::const_iterator it =
        process_id_.find(process);
    if (it == process_id_.end()) {
      return -1;
    }
    return it->second;
  }

  //----------------------------------------------------------------------
  // Determine the a priori state of the filter at the beginning of
  // the observation window.  Make sure everything is sized
  // correctly.
  double MMPP::initialize_filter(const PointProcess &data) {
    int S = hmm_state_space_size();
    int n = data.number_of_events();
    if (n == 0) return 0;
    double loglike = 0;
    pi0_.resize(S);
    pi0_ = 1.0 / S;

    if (one_.size() != S) {
      one_.resize(S);
      one_ = 1.0;
    }

    while (filter_.size() < data.number_of_events()) {
      Matrix P(S, S);
      filter_.push_back(P);
    }

    if (nrow(filter_[0]) < S) {
      for (int i = 0; i < filter_.size(); ++i) {
        filter_[i].resize(S, S);
      }
    }
    return loglike;
  }

  //----------------------------------------------------------------------
  // To be called at the end of make_hmm_states().  Allocates the
  // pointer to process_info_.
  void MMPP::create_process_info() {
    std::vector<PoissonProcess *> processes(dumb(component_processes_));
    std::vector<MixtureComponent *> mixture_components;
    if (!(mixture_components_.empty())) {
      int n = processes.size();
      mixture_components.reserve(n);
      for (int i = 0; i < n; ++i) {
        mixture_components.push_back(emits_[processes[i]]);
      }
    }
    process_info_.reset(new ProcessInfo(processes, mixture_components));
  }

}  // namespace BOOM
