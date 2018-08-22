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

#ifndef BOOM_POISSON_CLUSTER_PROCESS_HPP_
#define BOOM_POISSON_CLUSTER_PROCESS_HPP_

#include "Models/PointProcess/PointProcess.hpp"
#include "Models/PointProcess/PoissonProcess.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

#include <functional>
#include <map>
#include <vector>
#include "LinAlg/Selector.hpp"

namespace BOOM {

  // A struct containing the component processes in a Poisson cluster
  // process.
  struct PoissonClusterComponentProcesses {
    Ptr<PoissonProcess> background;
    Ptr<PoissonProcess> primary_birth;
    Ptr<PoissonProcess> primary_traffic;
    Ptr<PoissonProcess> primary_death;
    Ptr<PoissonProcess> secondary_traffic;
    Ptr<PoissonProcess> secondary_death;
  };

  // A Poisson cluster process is a type of Markov modulated Poisson
  // process.  There is a baseline process that sweeps up stray
  // events.  When the primary Poisson process generates a top-level
  // event, it also activates a secondary process that generates
  // events subsequent events until it dies.
  class PoissonClusterProcess : public CompositeParamPolicy,
                                public IID_DataPolicy<PointProcess>,
                                public PriorPolicy {
   public:
    // Use this constructor if there are no marks in the process, or
    // if you don't want to model the marks.
    explicit PoissonClusterProcess(
        const PoissonClusterComponentProcesses &components);

    // Use this constructor if there are marks to be modeled.
    PoissonClusterProcess(const PoissonClusterComponentProcesses &components,
                          const Ptr<MixtureComponent> &primary_mark_model,
                          const Ptr<MixtureComponent> &secondary_mark_model);

    PoissonClusterProcess(const PoissonClusterProcess &rhs);
    PoissonClusterProcess *clone() const override;

    ~PoissonClusterProcess() override {}

    void set_mark_models(const Ptr<MixtureComponent> &primary,
                         const Ptr<MixtureComponent> &secondary);

    virtual void clear_client_data();
    void impute_latent_data(RNG &rng);

    // Sample the posterior distributions of the client models.  To be
    // called after impute_latent_data().
    virtual void sample_client_posterior();
    double logpri() const override;

    // The log-event rate at time t for the process (or superposition
    // of processes) responsible for a transition from hmm state r to
    // hmm state s.  If r==s then this is the sum of the event rates
    // for processes active in state r.  Otherwise it is the rate of
    // the birth or death process associated with the process
    // responsible for the change in the activity state.
    // Args:
    //   r, s:  The hmm states defining the transition (from r to s).
    //   event:  The event produced by the transition.
    //   logp_primary: The conditional density of the event's marks
    //     (if any) under the primary mark model.
    //   logp_secondary:  The conditional density of the event's marks
    //     (if any) under the secondary mark model.
    //   source: The mark model known to have produced 'event',
    //     primary (1), secondary or background (0), or unknown source
    //     (< 0).
    // Returns:
    //   The conditional log likelihood of the event given the r->s
    //   transition.
    virtual double conditional_event_loglikelihood(
        int r, int s, const PointProcessEvent &event, double logp_primary,
        double logp_secondary, int source) const;

    // The sum of cumulative hazard functions between times t0 and t1
    // for processes active at time t0.  The set of active processes
    // is determined by hmm state 'r', using the function
    // active_processes(r).
    double conditional_cumulative_hazard(const DateTime &t0, const DateTime &t1,
                                         int r) const;

    int number_of_hmm_states() const;

    // Filter the data using the forward filtering algorithm.  Fills
    // filter_[t] data structure with the conditional distribution of
    // the transition from t-1 to t.
    // Args:
    //   data:  The PointProcess to be filtered.
    //   source: A vector of integers indicating the source of the
    //     observation at time t.  0 indicates the background or
    //     secondary process family.  1 indicates the primary process
    //     family.  -1 means the source is unknown.  If all sources
    //     are unknown (a common situation) then an empty vector can
    //     be passed instead.
    double filter(const PointProcess &data,
                  const std::vector<int> &source = std::vector<int>());
    double initialize_filter(const PointProcess &data);

    // Fills position t in the filter_ member with the conditional
    // distribution of activity state (r, s) given observed data up to
    // time t.
    //
    // Args:
    //   data:  The PointProcess being filtered
    //   t:  The step (current time point) being filtered.
    //   source: Indicates the source of the data: primary process
    //     (1), secondary or background process (0), or unknown source
    //     (anything < 0).
    //
    // Returns:
    //   The conditional log likelihood of observation t given
    //   preceding observations.
    double fwd_1(const PointProcess &data, int t, int source);

    // Backward sampling simulates the activity state of the process
    // at each point in time.  Along the way it ascribes each event to
    // one of the latent processes, or the associated birth and death
    // processes.  The 'source' argument can be an empty vector, in
    // which case unsupervised learning takes place (the normal case
    // for hmm's).  Otherwise the backward sampling is conditional on
    // the knowledge of which latent process produced the event.
    // There is still sampling to do because (a) the source for some
    // events may be unknown (represented by a -1 element in
    // 'source'), and (b) it must be determined whether the event was
    // produced by the specified latent process or by the associated
    // birth or death process.
    // Args:
    //   rng:  The random number generator.
    //   data:  The process whose hidden state is to be sampled.
    //   probability_of_activity: A matrix with 3 rows and
    //     data.number_of_events() columns indicating the probability
    //     that each process is active at time t.  The rows correspond
    //     to the background (0), primary (1), and secondary (2)
    //     processes.
    //   probability_of_responsibility: A matrix with dimensions
    //     matching probability_of_activity, but giving the probablity
    //     that each process was responsible at time t.
    void backward_sampling(RNG &rng, const PointProcess &data,
                           const std::vector<int> &source,
                           Matrix &probability_of_activity,
                           Matrix &probability_of_responsibility);

    int draw_previous_state(RNG &rng, int time, int current_state);

    // Takes a vector of data that has just been filtered, and updates
    // the filter matrices so that they condition on all data.
    // Args:
    //   data:  The point process whose hidden state should be smoothed.
    //   source: Indicates the source of the data: primary process
    //     (1), secondary or background process (0), or unknown source
    //     (anything < 0).
    //   probability_of_activity: A matrix with 3 rows and
    //     data.number_of_events() columns indicating the probability
    //     that each process is active at time t.  The rows correspond
    //     to the background (0), primary (1), and secondary (2)
    //     processes.
    //   probability_of_responsibility: A matrix with dimensions
    //     matching probability_of_activity, but giving the probablity
    //     that each process was responsible at time t.
    void backward_smoothing(const PointProcess &data,
                            const std::vector<int> &source,
                            Matrix &probability_of_activity,
                            Matrix &probability_of_responsibility);

    // On input, 'transition_density' is the joint distribution of
    // (h[t-1], h[t]) given data up to time t, and 'marginal' is the
    // marginal density of h[t] given complete data.  On output,
    // transition_density is updated to condition on all data, and
    // 'marginal' is the marginal of h[t-1] given complete data.
    void backward_smoothing_step(Matrix &transition_density, Vector &marginal);

    // Determine the specific process responsible for the event at
    // time t, given that the state at time t-1 is prev_state and the
    // state at time t is current_state.  If the states are the same
    // then a further Monte Carlo draw is used to determine the
    // responsible process.
    //
    // Args:
    //   rng:  The random number generator.
    //   data:  The process being filtered (and sampled).
    //   t: The step in the filtering or sampling process (counting
    //     from 0 at the beginning).
    //   previous_state:  The hmm state at time t-1.
    //   current_state:  The hmm state at time t.
    //   source: The mark model known to have produced the event at
    //     time t.  Primary (1), secondary or background (0), or
    //     unknown source ( < 0 ).
    //
    // Returns:
    //   The process responsible for the transition at time t, given
    //   the value of the transition.  If source < 0 (the expected
    //   state in many cases, the source for this observation is
    //   missing.
    virtual PoissonProcess *assign_responsibility(RNG &rng,
                                                  const PointProcess &data,
                                                  int t, int previous_state,
                                                  int current_state,
                                                  int source);

    // Attribute the event at time 'current_time' to the responsible
    // process and update its sufficient statistics accordingly,
    // including the sufficient statistics of the associated mark
    // models.  Exposure time is not updated, because it has already
    // been updated with update_exposure_time.
    virtual void attribute_event(const PointProcessEvent &data,
                                 PoissonProcess *responsible_process);

    // Update the statistics for all the processes determined to be
    // running between current_time and current_time + 1, conditional
    // on the values of current_state and next_state.
    void update_exposure_time(const PointProcess &data, int current_time,
                              int previous_state, int current_state);

    double loglike() const { return last_loglike_; }

    void clear_data() override;
    void add_data(const Ptr<Data> &dp) override;  // *dp is a PointProcess
    void add_data(const Ptr<PointProcess> &dp) override;

    // Adds a point process to the model, along with "ground truth"
    // information about which processes generated each event.  The
    // length of 'source' must match the number of events in 'dp'.
    // Entries in 'source' must negative (source unkown), 0 (secondary
    // or background process), or 1 (primary process).
    void add_supervised_data(const Ptr<PointProcess> &dp,
                             const std::vector<int> &source);

    // Simulate a PoissonClusterProcess observed from t0 to t1.
    virtual PointProcess simulate(
        RNG &rng, const DateTime &t0, const DateTime &t1,
        std::function<Data *()> primary_mark_simulator = NullDataGenerator(),
        std::function<Data *()> secondary_mark_simulator =
            NullDataGenerator()) const;

    const std::vector<Matrix> &probability_of_activity() const;
    const std::vector<Matrix> &probability_of_responsibility() const;

    void record_activity(VectorView activity_probs, int state);
    void record_responsibility(VectorView activity_probs,
                               PoissonProcess *responsible_process);

    void record_activity_distribution(VectorView probs,
                                      const Matrix &transition_distribution);
    void record_responsibility_distribution(
        VectorView probs, const Matrix &transition_distribution,
        const PointProcessEvent &event, int source);
    void allocate_probability(int previous_state, int current_state,
                              VectorView process_probs,
                              double transition_probability,
                              double logp_primary, double logp_secondary,
                              const DateTime &timestamp, int source);

    // These functions can return 0/NULL if no mark_models have been
    // assigned.
    MixtureComponent *mark_model(const PoissonProcess *process);
    const MixtureComponent *mark_model(const PoissonProcess *process) const;

   private:
    void initialize();
    void fill_state_maps();  // make virtual
    void setup_filter();
    virtual void register_models_with_param_policy();

    // Returns true iff process is associated with a primary event.
    // I.e. primary_traffic, primary_birth, or primary_death.
    bool primary(const PoissonProcess *process) const;

    // Returns true iff process is background, secondary_traffic, or
    // secondary_death.
    bool secondary(const PoissonProcess *process) const;

    // Returns the set of component processes that might have produced
    // an r->s transition.
    // Args:
    //   r, s:  The hmm states defining the transition.
    //   source: Indicator of the mark model that the event associated
    //     with the transition to 's'. Primary (1), secondary or
    //     background (0), or unknown source ( < 0 ).
    std::vector<PoissonProcess *> get_responsible_processes(int r, int s,
                                                            int source);
    std::vector<const PoissonProcess *> get_responsible_processes(
        int r, int s, int source) const;

    // Given a vector of Poisson processes, return the subset
    // associated with the mark model specified in 'source'.
    // Args:
    //   candidates:  The processes to consider.
    //   source: Indicator of the mark model that must be matched.
    //     Primary process (1), secondary or background process (0),
    //     or unknown source ( < 0 ).
    // Returns:
    //   The subset of candidates matching source.  If source < 0 then
    //   no matching is done and 'candidates' is returned unaltered.
    std::vector<PoissonProcess *> subset_matching_source(
        std::vector<PoissonProcess *> &candidates, int source);
    std::vector<const PoissonProcess *> subset_matching_source(
        const std::vector<PoissonProcess *> &candidates, int source) const;

    // Returns true if process is associated with latent process
    // 'source', where source is 0 (secondary or background), 1
    // (primary), or negative (unknown).
    bool matches_source(const PoissonProcess *process, int source) const;

    // Returns true if the transition from state r to state s is
    // possible.
    bool legal_transition(int r, int s) const;

    // Throws an exception if positive probability was assigned to an
    // impossible state based on known information about an
    // observation's source.
    // Args:
    //   probability:  The probability assigned to the state being checked.
    //   source: The source of the observation: primary (1), secondary
    //     or background (0), or unknown (<0).
    //   primary: A flag indicating whether the process being checked
    //     is the primary process.
    void check_source(double probability, int source, bool primary);

    Ptr<PoissonProcess> background_;

    Ptr<PoissonProcess> primary_birth_;
    Ptr<PoissonProcess> primary_death_;
    Ptr<PoissonProcess> primary_traffic_;

    Ptr<PoissonProcess> secondary_traffic_;
    Ptr<PoissonProcess> secondary_death_;

    // Responsible for user births, traffic, and deaths
    Ptr<MixtureComponent> primary_mark_model_;

    // Responsible for machine deaths, and machine traffic, including
    // background traffic.
    Ptr<MixtureComponent> secondary_mark_model_;

    //  Indicates which processes are on/off in each hmm state
    std::vector<Selector> activity_state_;

    // Holds the set of legal given the current state, for use in
    // fwd_1.
    std::vector<std::vector<int> > legal_target_transitions_;

    // Holds the vector of processes active in each HMM
    // state, including birth and death processes.
    std::vector<std::vector<PoissonProcess *> > active_processes_;

    // Keeps track of which processes are potentially responsible for
    // an (r->s) transition.  If a transition is impossible then no
    // map entry will be present.
    typedef std::map<std::pair<int, int>, std::vector<PoissonProcess *> >
        ResponsibleProcessMap;
    ResponsibleProcessMap responsible_process_map_;

    std::vector<Matrix> filter_;
    Vector pi0_;
    mutable Vector wsp_;
    Vector one_;
    double last_loglike_;

    // Each vector element corresponds to the PointProcess for a
    // single subject.  Space for a new subject is allocated when
    // add_data is called.  Each matrix has a number of rows equal to
    // the number of latent processes, and a number of columns equal
    // to the number of events in that subjects PointProcess data.
    std::vector<Matrix> probability_of_activity_;
    std::vector<Matrix> probability_of_responsibility_;

    enum InitializationStrategy {
      UniformInitialState = 0,
      StationaryDistribution
    };
    InitializationStrategy initialization_strategy_;

    // The known_source_store_ keeps track of source information for
    // each PointProcess.  If some events are known to be
    typedef std::map<Ptr<PointProcess>, std::vector<int> > SourceMap;
    SourceMap known_source_store_;
  };

}  // namespace BOOM
#endif  // BOOM_POISSON_CLUSTER_PROCESS_HPP_
