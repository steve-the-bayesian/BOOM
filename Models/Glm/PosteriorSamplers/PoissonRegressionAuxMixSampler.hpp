// Copyright 2018 Google LLC. All Rights Reserved.
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

#ifndef BOOM_POISSON_REGRESSION_AUXILIARY_MIXTURE_SAMPLER_HPP_
#define BOOM_POISSON_REGRESSION_AUXILIARY_MIXTURE_SAMPLER_HPP_

#include <memory>

#include "Models/Glm/PoissonRegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/NormalMixtureApproximation.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/Imputer.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class PoissonDataImputer;

  class PoissonRegressionDataImputer
      : public SufstatImputeWorker<PoissonRegressionData, WeightedRegSuf> {
   public:
    // Args:
    //   coefficients: The coefficients for the model managed by the
    //     sampler.  These are constant for the duration of the data
    //     augmentation step, and then change (for all workers) after
    //     the parameter sampling step.
    PoissonRegressionDataImputer(WeightedRegSuf &global_suf,
                                 std::mutex &global_suf_mutex,
                                 const GlmCoefs *coefficients,
                                 RNG *rng = nullptr,
                                 RNG &seeding_rng = GlobalRng::rng);

    void impute_latent_data_point(const PoissonRegressionData &data_point,
                                  WeightedRegSuf *complete_data_suf,
                                  RNG &rng) override;

   private:
    const GlmCoefs *coefficients_;
    std::unique_ptr<PoissonDataImputer> imputer_;
  };

  //----------------------------------------------------------------------

  class PoissonRegressionAuxMixSampler
      : public PosteriorSampler,
        public LatentDataSampler<PoissonRegressionDataImputer> {
   public:
    PoissonRegressionAuxMixSampler(PoissonRegressionModel *model,
                                   const Ptr<MvnBase> &prior,
                                   int number_of_threads = 1,
                                   RNG &seeding_rng = GlobalRng::rng);

    PoissonRegressionAuxMixSampler *clone_to_new_host(Model *new_host) const override;
    void draw() override;
    double logpri() const override;

    void clear_latent_data() override;
    Ptr<PoissonRegressionDataImputer> create_worker(std::mutex &m) override;
    void assign_data_to_workers() override;

    // The first trip through the data is single threaded, so that the
    // PoissonDataImputer object can be filled with required values
    // without causing a race condition.  Subsequent trips can use
    // multiple threads.  The overrides to set_number_of_workers() and
    // impute_latent_data_point() ensure that multi-threading is
    // delayed until after the first iteration.
    void set_number_of_workers(int n) override;
    void impute_latent_data() override;

    // Below this line are implementation details exposed for testing.
    double draw_final_event_time(int y);
    double draw_censored_event_time(double final_event_time, double rate);
    double draw_censored_event_time_zero_case(double rate);

    void draw_beta_given_complete_data();
    const WeightedRegSuf &complete_data_sufficient_statistics() const;

    // Clear the complete data sufficient statistics.  This is
    // normally unnecessary.  This function is primarily intended for
    // nonstandard situations where the complete data sufficient
    // statistics need to be manipulated by an outside actor.
    void clear_complete_data_sufficient_statistics();

    // Increment the complete data sufficient statistics with the
    // given quantities.  This is normally unnecessary.  This function
    // is primarily intended for nonstandard situations where the
    // complete data sufficient statistics need to be manipulated by
    // an outside actor.
    void update_complete_data_sufficient_statistics(
        double precision_weighted_sum, double total_precision, const Vector &x);

   private:
    PoissonRegressionModel *model_;
    Ptr<MvnBase> prior_;
    WeightedRegSuf complete_data_suf_;

    // The Poisson data imputer needs single threaded access during
    // the first MCMC iteration.  After that it is safe to access in a
    // multi-threaded environment.  This flag keeps track of whether
    // impute_data() has previously been called.
    bool first_pass_through_data_;

    // Once the first pass through the data is complete then the
    // multi-threaded environment can be set up.  This field keeps
    // track of the desired number of workers.
    int desired_number_of_workers_;
  };

}  // namespace BOOM

#endif  // BOOM_POISSON_REGRESSION_AUXILIARY_MIXTURE_SAMPLER_HPP_
