// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#ifndef BOOM_MLVS_DATA_IMPUTER_HPP
#define BOOM_MLVS_DATA_IMPUTER_HPP

#include "Models/Glm/ChoiceData.hpp"
#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/Glm/PosteriorSamplers/MultinomialLogitCompleteDataSuf.hpp"
#include "Models/PosteriorSamplers/Imputer.hpp"

namespace BOOM {
  namespace ML = MultinomialLogit;
  class MlvsDataImputer
      : public SufstatImputeWorker<ChoiceData,
                                   ML::CompleteDataSufficientStatistics> {
   public:
    typedef ML::CompleteDataSufficientStatistics SufficientStatistics;

    // Args:
    //   model:  A pointer to the model being data-agumented.
    MlvsDataImputer(SufficientStatistics &global_suf,
                    std::mutex &global_suf_mutex, MultinomialLogitModel *model,
                    RNG *rng = nullptr, RNG &seeding_rng = GlobalRng::rng);

    // Impute latent data for a single observation, and add the
    // results to the complete data sufficient statistics.
    void impute_latent_data_point(const ChoiceData &observed_data,
                                  SufficientStatistics *suf, RNG &rng) override;

    // Used to decompose latent utilities into a mixture of Gaussians.
    // Args:
    //   rng: Random number genrerator.  Must not be shared by any
    //     other threads.
    //   u:  Latent utility.
    // Returns:
    //   The index of the imputed Gaussian mixture component
    //   responsible for u.
    uint unmix(RNG &rng, double u) const;

   private:
    MultinomialLogitModel *model_;
    Iterator observed_data_begin_;
    Iterator observed_data_end_;

    const Vector mu_;                  // mean for EV approx
    const Vector sigsq_inv_;           // inverse variance for EV approx
    const Vector sd_;                  // standard deviations for EV approx
    const Vector log_mixing_weights_;  // log of mixing weights for EV approx
    const Vector &log_sampling_probs_;
    const bool downsampling_;

    // Workspace for impute_latent_data, to avoid reallocating space
    // each time.
    mutable Vector post_prob_;
    mutable Vector u;
    mutable Vector eta;
    mutable Vector wgts;
  };

}  // namespace BOOM

#endif  // BOOM_MLVS_DATA_IMPUTER_HPP
