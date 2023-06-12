// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#ifndef BOOM_MULTINOMIAL_LOGIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_MULTINOMIAL_LOGIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_

#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/MLVS.hpp"
#include "Models/MvnBase.hpp"
#include "Samplers/MoveAccounting.hpp"

namespace BOOM {

  class MultinomialLogitCompositeSpikeSlabSampler : public MLVS {
   public:
    // Args:
    //   model:  The model to be posterior sampled.
    //   coefficient_prior: The conditional prior distribution on all
    //     the coefficients, given inclusion.
    //   inclusion_prior: The prior probability of inclusion for the
    //     coefficients.
    //   t_degrees_of_freedom: The tail thickness parameter for the
    //     proposal distribution when making Metropolis Hastings
    //     proposals.  If t_degrees_of_freedom is <= 0 then a Normal
    //     proposal will be used.
    //   rwm_variance_scale_factor: A constant amount (> 0) by which
    //     to scale the proposal variance in random walk Metropolis
    //     proposals.  Smaller values will yield more frequent
    //     acceptances but smaller moves conditional on acceptance.
    //   nthreads: The number of threads to use for data augmentation
    //     when making a data augmentation move.  If the sample size
    //     is small then a single thread is the fastest option.  Once
    //     you hit a hundred thousand observations or so then it makes
    //     sense to set nthreads to the maximum number of hardware
    //     threads.
    //   max_chunk_size: The largest "chunk" of coefficients that will
    //     be proposed by a Metropolis Hastings proposal.
    //   check_initial_condition: Passed to MLVS.  Throws an exception
    //     if the initial state of model has zero support under the
    //     prior.  This is mainly an issue when variables are forced
    //     in or out of the model with the inclusion_prior, but those
    //     variables are initially excluded (or included) by the
    //     model.
    MultinomialLogitCompositeSpikeSlabSampler(
        MultinomialLogitModel *model,
        const Ptr<MvnBase> &coefficient_prior,
        const Ptr<VariableSelectionPrior> &inclusion_prior,
        double t_degrees_of_freedom = -1,
        double rwm_variance_scale_factor = 1,
        uint nthreads = 1,
        int max_chunk_size = 10,
        bool check_initial_condition = true,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    void rwm_draw();
    void tim_draw();
    void rwm_draw_chunk(int chunk);

    LabeledMatrix timing_report() const;

    enum MoveType { DATA_AUGMENTATION_MOVE = 0, RWM_MOVE = 1, TIM_MOVE = 2 };

    // Set the probabilities of selecting each of the three move
    // types.  The three arguments must be non-negative numbers
    // summing to 1.
    void set_move_probabilities(double data_augmentation, double rwm,
                                double tim);

    int compute_chunk_size() const;
    int compute_number_of_chunks() const;

   private:
    MultinomialLogitModel *model_;
    Ptr<MvnBase> prior_;
    Ptr<VariableSelectionPrior> inclusion_prior_;
    MoveAccounting accounting_;
    int max_chunk_size_;
    double tdf_;
    double rwm_variance_scale_factor_;
    Vector move_probs_;
  };

}  // namespace BOOM

#endif  // BOOM_MULTINOMIAL_LOGIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_
