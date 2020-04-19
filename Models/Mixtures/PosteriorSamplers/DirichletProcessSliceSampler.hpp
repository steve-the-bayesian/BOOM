// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#ifndef BOOM_DIRICHLET_PROCESS_SLICE_SAMPLER_HPP_
#define BOOM_DIRICHLET_PROCESS_SLICE_SAMPLER_HPP_

#include "Models/Mixtures/DirichletProcessMixture.hpp"
#include "Models/Mixtures/PosteriorSamplers/SplitMerge.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Samplers/MoveAccounting.hpp"

namespace BOOM {
  // This class implements the slice sampling algorithm from Kalli, Griffin, and
  // Walker, Statistics and Computing (2011), pp 93 -- 105.
  class DirichletProcessSliceSampler : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model to be managed.
    //   initial_clusters: The initial number of clusters to assume for the
    //     model.
    //   seeding_rng: The external RNG used to seed the RNG owned by this
    //     sampler.
    explicit DirichletProcessSliceSampler(
        DirichletProcessMixtureModel *model,
        int initial_clusters = 1,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;

    // The true Bayesian prior for this model is the prior on the base measure.
    double logpri() const override;

    //  Slice sampling code --------------------------------------------------
    // The Kalli, Griffin Walker (2011) slice sampler introduces a sequence
    // xi[j] used to decouple the mixing weights from the definition of the
    // slice.  This class refers to xi[i] as the mixing_weight_importance for
    // cluster i.
    //
    // The KGW paper claims xi[i] can be "any positive sequence", but xi values
    // should be < 1, and decreasing with i.
    //
    // TODO: Investigate sequences other than geometric,
    // e.g. Poisson or NB distributions.
    double mixing_weight_importance(int cluster) const;
    double log_mixing_weight_importance(int cluster) const;

    // The x[i] parameters mentioned above are assumed (by this class) to be a
    // geometric sequence xi[i] = xi0^i.  The value xi0 is referred to as the
    // mixing_weight_importance_ratio.
    void set_mixing_weight_importance_ratio(double value);

    void draw_parameters_given_mixture_indicators();
    void draw_stick_fractions_given_mixture_indicators();
    void draw_slice_variables_given_mixture_indicators();
    void draw_mixture_indicators();

    // Returns the smallest index k such that mixing_weight_importance_ratio(k)
    // <= slice_variable.
    int find_max_number_of_clusters(double slice_variable) const;

    //----------------------------------------------------------------------
    // A Metropolis-Hastings move to shuffle the order of the mixture
    // components.
    void shuffle_order();

    //----------------------------------------------------------------------
    // Code in this section is for the split-merge Metropolis-Hastings move.
    // Two data points are chosen at random.  If they fall into the same mixture
    // component then the call is passed to attempt_split_move.  If they fall in
    // different components then the call passes to attempt_merge_move.
    //
    // The logic for how the splits and merges are proposed is contained in a
    // SplitMerge::ProposalStrategy object.  The DirichletProcessSliceSampler
    // object will take ownership of the strategy and delete it when the sampler
    // is destroyed.
    //
    // If this function is not called, no split-merge move will be attempted.
    // NOTE: The code in PosteriorSamplers/SplitMerge.hpp seems to
    // asymmetrically prefer splits to merges in an unbalanced way.  This code
    // should be viewed as experimental and not used until the asymmetry issue
    // can be resolved.
    void set_split_merge_strategy(SplitMerge::ProposalStrategy *strategy);

    void split_merge_move();

    // Attempt, using Metropolis-Hastings, to merge the components containing
    // observations data_index_1 and data_index_2.  These must currently be
    // allocated to different mixture components.
    void attempt_merge_move(int data_index_1, int data_index_2);

    // Attempt, using Metropolis-Hastings, to split the component containing
    // observations 'data_index_1' and 'data_index_2'.  These must currently be
    // allocated to the same mixture component.
    void attempt_split_move(int data_index_1, int data_index_2);

    // Returns the log of the MH acceptance probability (the "alpha" in min(1,
    // alpha)) for a split-merge proposal.  This function evaluates the log
    // target density ratio.  The proposal already carries the log proposal
    // density ratio.
    //
    // The log target density ratio is the log of p(model | split) / p(model |
    // merged).  where
    //      p(model) = p(data | parameters, indicators, mixing weights)
    //         * p(indicators | mixing weights, --parameters)
    //         * p(parameters | --mixing weights)
    //         * p(mixing weights)
    //
    // Factors in the model unaffected by the split-merge move cancel in the
    // ratio, and are not computed.
    double log_MH_probability(const SplitMerge::Proposal &proposal) const;

    // Returns a labeled matrix giving the success rates of the split and merge
    // moves, as well as the total amount of time spent in each move type.
    LabeledMatrix MH_performance_report() const {
      return move_accounting_.to_matrix();
    }

   private:
    DirichletProcessMixtureModel *model_;

    double mixing_weight_importance_ratio_;
    double log_mixing_weight_importance_ratio_;

    // The maximum cluster size for each observation, conditional on slice
    // variables.
    std::vector<int> max_clusters_;

    // The (current) maximum number of clusters for the whole model, as
    // determined by the slice variables.  This is the maximal entry in
    // max_clusters_.
    int global_max_clusters_;

    // If first_time_ is true then the draw() method will randomly allocate data
    // to clusters (and then set first_time_ to false) before proceeding.
    bool first_time_;

    // Proposes split and merge moves.
    std::unique_ptr<SplitMerge::ProposalStrategy> split_merge_strategy_;

    // Keeps track of how often different MH moves are tried, and their success
    // rates.
    MoveAccounting move_accounting_;

    // Allocates data to clusters uniformly at random.  Used for initialization.
    void randomly_allocate_data_to_clusters();
  };

}  // namespace BOOM

#endif  //  BOOM_DIRICHLET_PROCESS_SLICE_SAMPLER_HPP_
