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

#ifndef BOOM_MARKOV_MODULATED_POISSON_PROCESS_HPP_
#define BOOM_MARKOV_MODULATED_POISSON_PROCESS_HPP_

#include <vector>

#include <unordered_map>
#include <memory>

#include "Models/PointProcess/PointProcess.hpp"
#include "Models/PointProcess/PoissonProcess.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "cpputil/RefCounted.hpp"

namespace BOOM {

  namespace MmppHelper {
    typedef std::vector<std::vector<PoissonProcess *>> SourceVector;

    // A class to describe which processes are active in an HMM state,
    // and keep track of information needed during the forward-backward
    // algorithm, such as which other states it communicates with.
    class HmmState : private RefCounted {
     public:
      // The HmmState does not own the PoissonProcess * pointers that
      // it is passed.  Each should be owned by a Ptr external to this
      // class.
      explicit HmmState(const std::vector<PoissonProcess *> &processes);

      // The id_number is this state's position in the vector of
      // hmm_states_ in the managing MMPP.
      int id_number() const;
      void set_id_number(int id_number);

      // Two states compare equal if they have the same set of active
      // processes.  The comparison is fast because active_processes_
      // is kept sorted.
      bool operator==(const HmmState &rhs) const;

      // The number of processes constituting the state.
      int number_of_active_processes() const;

      // Accessor for the set of active processes.  Note that the
      // constructor sorts them, so the order here may be different
      // than the order the processes were in when they were passed to
      // the constructor.
      const std::vector<PoissonProcess *> &active_processes() const;

      // Add or remove processes to the set of active processes
      // defining the state.  Adding processes that are already part
      // of the state will not result in duplicates.  Removing
      // processes that are not part of the state is safe and results
      // in no action.
      void add_processes(std::vector<PoissonProcess *> additional_processes);
      void remove_processes(std::vector<PoissonProcess *> processes_to_remove);

      const std::vector<HmmState *> &potential_outgoing_transitions() const;
      const std::vector<HmmState *> &potential_incoming_transitions() const;

      // The set of processes that could be responsible for a transition
      // from *this to *destination.  The result could be a singleton,
      // or it could be empty.
      const std::vector<PoissonProcess *> &processes_transitioning_to(
          const HmmState *destination) const;

      // Inform the object that it is possible to transition to *this
      // from *previous_state.
      void add_transition_from(HmmState *previous_state);

      // Inform the object that it is possible to transition from *this
      // to *next_state.
      // Args:
      //   next_state: the state that *this transitions to.
      //   source_of_transition: The PoissonProcess whose event causes
      //     the transition to next_state.
      void add_transition_to(HmmState *next_state,
                             PoissonProcess *source_of_transition);

     private:
      // These will be kept sorted by the address of their pointers.
      std::vector<PoissonProcess *> active_processes_;
      int id_number_;

      std::vector<HmmState *> potential_outgoing_transitions_;
      std::vector<HmmState *> potential_incoming_transitions_;

      // Keep track of which processes are responsible for a transition
      // from this state to each state in can_transition_to_.
      typedef std::map<const HmmState *, std::vector<PoissonProcess *>>
          TransitionResponsibilityMap;
      TransitionResponsibilityMap responsible_for_transition_to_;

     public:
      friend void intrusive_ptr_add_ref(HmmState *s) { s->up_count(); }
      friend void intrusive_ptr_release(HmmState *s) {
        s->down_count();
        if (s->ref_count() == 0) delete s;
      }
    };

    //----------------------------------------------------------------------
    // A class for managing the probabilistic calculations for the
    // various component processes defining the HmmStates.  It
    // exists because each process is a part of multiple
    // HmmStates, so we want to avoid multiple calls to logp(),
    // cumulative_hazard(), and
    // event_rate().  This class will call the necessary members once
    // for each process, and then store the results so they can be
    // easily reused.
    class ProcessInfo {
     public:
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
      ProcessInfo(const std::vector<PoissonProcess *> &processes,
                  const std::vector<MixtureComponent *> &mixture_components);

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
      void evaluate(const PointProcess &data, const SourceVector &source);

      // If the call to 'evaluate' indicated that 'process' was not a
      // possible source of the event at time 't' then this function
      // returns -infinity.  Otherwise it returns the log of the event
      // rate for 'process' at the timestamp of event 't'.
      double log_event_rate(const PoissonProcess *process, int t) const;

      // Returns the log density for the mixture component associated
      // with 'process' evalutated on event t.  If that event contains
      // no marks or if no mixture components are supplied then 0 is
      // returned.
      double mixture_log_likelihood(const PoissonProcess *process, int t) const;

      // The sum of the cumulative hazards of the active processes in
      // state from time t-1 to t.  This return values is not affected
      // by the presence or absence of the 'source' argument to
      // 'evaluate'.
      double conditional_cumulative_hazard(const HmmState *state, int t) const;

     private:
      // Returns the position of 'process' in processes_;
      int process_id(const PoissonProcess *process) const;

      const double neginf_;
      std::vector<PoissonProcess *> processes_;

      // The vector of mixture components, sorted and with duplicates
      // removed.  This vector is not parallel to processes_, but
      // minimal_mixture_components_[mixture_component_id_[i]]
      // does correspond to processes_[i].
      std::vector<MixtureComponent *> minimal_mixture_components_;

      // The process id of a PoissonProcess * is its position in
      // processes_.  We need this to resolve calls with
      // PoissonProcess * arguments.
      std::unordered_map<const PoissonProcess *, int> process_id_;

      // The mixture component id of a PoissonProcess * is the index of
      // the corresponding pointer in minimal_mixture_components_
      std::vector<int> mixture_component_id_;

      //----------------------------------------------------------------------
      // Space below here is allocated and managed by calls to evaluate().

      // Cumlative hazard for the component Poisson processes.
      // Columns are time.  Rows correspond to process_id_.
      Matrix cumulative_hazard_;

      // Logs of the instantaneous event rates for the component Poisson
      // processes.  Columns are time.  Rows correspond to process_id_.
      Matrix log_event_rate_;

      // The log_density of the mixture components, if any are present.
      // Columns are time.  Rows correspond to mixture_component_id_.
      Matrix logp_;
    };

  }  // namespace MmppHelper

  //======================================================================
  // The Markov modulated Poisson Process, as defined in Scott and
  // Smyth (2003) "The Markov Modulated Poisson Process and Markov
  // Poisson Cascade with Applications to Web Traffic Modeling."
  //
  // The basic workflow is
  //  MarkovModulatedPoissonProcess mmpp;
  //  mmpp.add_component_process(
  //     first_process, processes_it_spawns, processes_it_kills, mark_model);
  //  mmpp.add_component_process(
  //     second_process, processes_it_spawns, processes_it_kills, mark_model);
  //
  //  std::vector<Ptr<PoissonProcess> > initial_state_members;
  //  initial_state_members.push_back(first_process);
  //  mmpp.make_hmm_states(initial_state_members);
  //
  //  mmpp.impute_latent_data();
  //  for (int i = 0; i < 1000; ++i) {
  //    mmpp.sample_posterior();
  //  }
  //
  // Priors are set for each process and each mixture component before
  // calling add_component_process (or pointers to these objects can
  // be held outside the class, in which case the priors should be set
  // before calling sample_posterior().  Each of the component objects
  // should be capable of calling sample_posterior().
  //
  // Calling sample_posterior
  class MarkovModulatedPoissonProcess : public CompositeParamPolicy,
                                        public IID_DataPolicy<PointProcess>,
                                        public PriorPolicy {
   public:
    typedef MmppHelper::HmmState HmmState;
    typedef MmppHelper::ProcessInfo ProcessInfo;
    typedef MmppHelper::SourceVector SourceVector;

    MarkovModulatedPoissonProcess();
    MarkovModulatedPoissonProcess(const MarkovModulatedPoissonProcess &rhs);
    MarkovModulatedPoissonProcess *clone() const override;

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
    void add_component_process(const Ptr<PoissonProcess> &process,
                               std::vector<Ptr<PoissonProcess>> spawns,
                               std::vector<Ptr<PoissonProcess>> kills,
                               const Ptr<MixtureComponent> &emits);

    // After all component processes have been added using
    // add_component_process() the user should call make_hmm_states()
    // to finalize construction.
    // Args:
    //   initial_state_elements: The collection of processes defining
    //      one of the states in the HMM.  This is not necessarily the
    //      'initial_state' in the sense of the state at time 0.  It
    //      is just a seed to use for determining which states of
    //      nature need to be considered.
    void make_hmm_states(
        const std::vector<Ptr<PoissonProcess>> &initial_state_elements);

    // Add data to the model.  This had to be over-ridden because the
    // matrices that keep track of the probability of activity and the
    // probability of responsibility get modified when a new process
    // is added.
    void add_data(const Ptr<Data> &dp) override;
    void add_data(const Ptr<PointProcess> &dp) override;

    // If a subset (possibly all) of the data are from a training set
    // where the original source is knwon then add the data using
    // add_supervised_data instead of add_data.  Each event
    // corresponds to a vector of PoissonProcess * pointers indicating
    // the set of possible models that could have generated the data.
    // If the source information is missing for a particular event
    // then you can use an empty vector to signal "don't know" for
    // that event.  The size of 'source' must match the number of
    // events in 'process'.  It cannot be empty.
    void add_supervised_data(const Ptr<PointProcess> &dp,
                             const SourceVector &source);

    // Clears the data managed by this model.  The over-ride is
    // necessary because of internal state not managed by the
    // DataPolicy which must be cleared as well.
    void clear_data() override;

    // Calls clear_data() on all component processes and mixture
    // components.  Does not clear the data managed by this class.
    void clear_client_data();

    // Impute values for the latent processes using the forward
    // backward simulation algorithm.  Returns the observed-data log
    // likelihood of the current set of model parameters.
    virtual double impute_latent_data(RNG &rng);

    // Returns the log likelihood value that was computed during the
    // most recent data imputation.
    double last_loglike() const { return last_loglike_; }

    // As the MCMC progresses, the model accumulates information about
    // which processes were active at various points in time and which
    // were responsible for producing the different events.  Calling
    // burn() will reset these estimates, effectively discarding all
    // preceding iterations as MCMC burn-in.
    void burn();

    // Call sample_posterior for each of the component processes and
    // mixture components.
    void sample_complete_data_posterior();

    // Compute the log of the prior distribution at the current values
    // of the model parameters.  logpri is implemented here so the
    // model does not have to expose the component models to the
    // posterior sampler.
    double logpri() const override;

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
    double filter(const PointProcess &process, const SourceVector &source);

    // Updates the state of the filter at time t to give the conditional
    // distribution of state[t-1] and state[t] given observed data to
    // time t.
    // Args:
    //   t:  The index of the event to use for updating.
    //   process_info: An ProcessInfo object containting all the
    //     relevant likelihood evalutations for interval t.
    // Returns:
    //   log p(events[t] | events[0, ..., t-1])
    double fwd_1(int t, const ProcessInfo &process_info);

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
    void backward_sampling(RNG &rng, const PointProcess &process,
                           Matrix &probability_of_activity,
                           Matrix &probability_of_responsibility);

    // In some MMPP's the hmm state space size grows exponentially
    // with the number of latent processes.
    int hmm_state_space_size() const;
    int number_of_processes() const;

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
    Matrix probability_of_activity(int data_series_number = 0) const;

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
    Matrix probability_of_responsibility(int data_series_number) const;

    // Return log(lambda_1(t) * p_1(y) + lambda_2(t) * p_2(y) + ...),
    // where
    //  * lambda_j(t) is the instantaneous event rate at the time
    //    of event t,
    //  * p_j(y) is the likelihood of the mark value y under mixture
    //    component j., and
    //  * the sum is over all possible processes that could have been
    //    responsible for the transition from first_state to
    //    second_state.
    double conditional_event_loglikelihood(
        int t, const HmmState *first_state, const HmmState *second_state,
        const ProcessInfo &process_info) const;

    // Add the exposure time between events t-1 and t from 'process'
    // to the active processes from the hmm_state indexed by
    // 'previous_state'.
    void update_exposure_time(const PointProcess &process, int t,
                              int previous_state);

    // Draw the previous HMM state given the index of the current
    // state.
    // Args:
    //   rng:  A uniform random number generator.
    //   t:  The time index corresponding to 'current_state'.
    //   current_state:  The index of the HMM state at time t.
    int draw_previous_state(RNG &rng, int t, int current_state);

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
    PoissonProcess *sample_responsible_process(RNG &rng, int previous_state,
                                               int current_state,
                                               const ProcessInfo &process_info,
                                               int t);

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
    void record_activity(VectorView probability_of_activity, int hmm_state_id);

    // The set of hmm states created by make_hmm_states().  Exposed
    // mainly for testing.
    const std::vector<Ptr<HmmState>> &hmm_states() const { return hmm_states_; }

   private:
    std::vector<Ptr<PoissonProcess>> component_processes_;

    // The minimal list of mixture components, with each element being
    // unique.
    std::vector<Ptr<MixtureComponent>> mixture_components_;

    // A flag for whether mixture components are present, which is
    // determined by the first call to add_component_process().
    bool have_mixture_components_;

    // TODO(stevescott): You could get rid of the unordered_maps by
    // introducing a class MmppComponentProcess that was a
    // PoissonProcess wrapper with extra fields for id_number and
    // MixtureComponent *.
    std::unordered_map<const PoissonProcess *, int> process_id_;

    //------------------------------------------------------------
    // This storage is used between calls to add_component_process()
    // and make_hmm_states().

    // spawns_[process] is the vector of processes that an event from
    // 'process' turns on.
    std::map<const PoissonProcess *, std::vector<PoissonProcess *>> spawns_;

    // The processes that an event from 'process' turns off.
    std::map<const PoissonProcess *, std::vector<PoissonProcess *>> kills_;
    std::map<const PoissonProcess *, MixtureComponent *> emits_;
    //------------------------------------------------------------

    // The set of HMM states used to define the forward-backward
    // algorithm.
    std::vector<Ptr<HmmState>> hmm_states_;

    void check_that_all_processes_have_been_registered();
    void check_first_entry(const Ptr<PoissonProcess> &process);
    void check_for_new_process(const Ptr<PoissonProcess> &process);
    void check_for_new_mixture_component(
        const Ptr<MixtureComponent> &component);
    Ptr<HmmState> check_for_new_hmm_state(const Ptr<HmmState> &potential_state);
    void generate_new_states(const Ptr<HmmState> &state);

    // Return the position of 'process' in the data member
    // component_processes_.
    int process_id(const PoissonProcess *process) const;
    double initialize_filter(const PointProcess &process);
    void create_process_info();

    // Storage needed for forward_backward filtering.  It is managed
    // during the call to initialize_filter, so it does not need
    // special attention in the constructor.
    Vector pi0_;
    Vector one_;
    std::vector<Matrix> filter_;
    double last_loglike_;
    mutable Vector mutable_workspace_;

    // Each vector element corresponds to the PointProcess for a
    // single data series.  Space for a new data series is allocated
    // when add_data is called.  Each matrix has a number of rows
    // equal to the number of latent processes, and a number of
    // columns equal to the number of events in that data series'
    // PointProcess data.
    std::vector<Matrix> probability_of_activity_;
    std::vector<Matrix> probability_of_responsibility_;

    // The process_info_ object is created during a call to
    // make_hmm_states().
    std::shared_ptr<ProcessInfo> process_info_;

    // Keeps track of the set of potential sources associated with
    // data from a supervised or semi-supervised training problem.
    // This is where the 'source' information is stored after a call
    // to add_supervised_data().
    typedef std::unordered_map<const PointProcess *, SourceVector> SourceMap;
    SourceMap known_source_store_;
  };

}  // namespace BOOM
#endif  // BOOM_MARKOV_MODULATED_POISSON_PROCESS_HPP_
