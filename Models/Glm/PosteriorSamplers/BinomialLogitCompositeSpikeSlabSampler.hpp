// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_BINOMIAL_LOGIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_BINOMIAL_LOGIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_
#include "Models/Glm/PosteriorSamplers/BinomialLogitSpikeSlabSampler.hpp"
#include "Models/MvnBase.hpp"
#include "Samplers/MoveAccounting.hpp"

namespace BOOM {
  //======================================================================
  // A functor that returns the log posterior and first two
  // derivatives for the specified chunk.
  class BinomialLogitLogPostChunk {
   public:
    BinomialLogitLogPostChunk(const BinomialLogitModel *model,
                              const MvnBase *prior, int chunk_size,
                              int chunk_number)
        : m_(model), pri_(prior), start_(chunk_size * chunk_number) {
      int nvars = m_->coef().nvars();
      int elements_remaining = nvars - start_;
      chunk_size_ = std::min(chunk_size, elements_remaining);
    }
    double operator()(const Vector &beta_chunk) const;
    double operator()(const Vector &beta_chunk, Vector &grad, Matrix &hess,
                      int nd) const;

   private:
    const BinomialLogitModel *m_;
    const MvnBase *pri_;
    int start_;
    int chunk_size_;
  };

  //======================================================================
  // The BinomialLogitSpikeSlabSampler can be slow in the presence of
  // a separating hyperplane, but it is good at making
  // trans-dimensional moves.  One solution is to bundle the
  // spike-and-slab sampler with one or more fixed dimensional samplers.
  // At each iteration one of the samplers is chosen at random and used
  // to generate a move.  This class combines the AuxiliaryMixture
  // sampler with RandomWalkMetropolis and
  // TailoredIndependenceMetropolis (TIM) samplers.
  class BinomialLogitCompositeSpikeSlabSampler
      : public BinomialLogitSpikeSlabSampler {
   public:
    BinomialLogitCompositeSpikeSlabSampler(
        BinomialLogitModel *model, const Ptr<MvnBase> &prior,
        const Ptr<VariableSelectionPrior> &vpri, int clt_threshold, double tdf,
        int max_tim_chunk_size, int max_rwm_chunk_size = 1,
        double rwm_variance_scale_factor = 1.0,
        RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    void rwm_draw();
    void tim_draw();

    // Draw the specified chunk using a random walk proposal.
    void rwm_draw_chunk(int chunk);

    BinomialLogitLogPostChunk log_posterior(int chunk_number,
                                            int max_chunk_size) const;

    // The three samplers will be used in proportion to the weights
    // supplied here.  Weights must be non-negative, and at least one
    // must be positive.
    void set_sampler_weights(double da_weight, double rwm_weight,
                             double tim_weight);

    std::ostream &time_report(std::ostream &out) const;

   private:
    BinomialLogitModel *m_;
    Ptr<MvnBase> pri_;
    double tdf_;
    int max_tim_chunk_size_;
    int max_rwm_chunk_size_;
    double rwm_variance_scale_factor_;

    MoveAccounting move_accounting_;
    Vector sampler_weights_;

    // Compute the size of the largest chunk
    int compute_chunk_size(int max_chunk_size) const;
    int compute_number_of_chunks(int max_chunk_size) const;
  };
}  // namespace BOOM
#endif  //  BOOM_BINOMIAL_LOGIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_
