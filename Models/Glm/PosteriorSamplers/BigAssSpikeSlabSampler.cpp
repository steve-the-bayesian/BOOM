/*
  Copyright (C) 2005-2021 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include <thread>
#include "Models/Glm/PosteriorSamplers/BigAssSpikeSlabSampler.hpp"
#include "distributions.hpp"
#include "cpputil/ThreadTools.hpp"

namespace BOOM {

  BigAssSpikeSlabSampler::BigAssSpikeSlabSampler(
      BigRegresionModel *model,
      const Ptr<VariableSelectionPrior> &spike,
      const Ptr<RegressionSlabGivenXandSigma> &slab,
      const Ptr<GammaModelBase> &residual_precision_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        spike_(spike),
        slab_(slab),
        residual_precision_prior_(residual_precision_prior)
  {}

  void BigAssSpikeSlabSampler::initial_screen(int niter, double threshold) {
    assign_subordinate_samplers();
    run_parallel_initial_screen(niter);
    gather_candidate_predictors_from_workers(threshold);
  }

  void BigAssSpikeSlabSampler::assign_subordinate_samplers() {
    int number_of_workers = std::lround(std::ceil(
        double(model_->xdim()) / model_->worker_dim_upper_limit()));
    for (int w = 0; w < number_of_workers; ++w) {
      int this_worker_dim = -1;  //////////////
      if (w + 1 == number_of_workers) {
      } else {
        NEW(RegressionModel, worker)(model_->worker_dim_upper_limit());
      }
    }
  }

  void BigAssSpikeSlabSampler::run_parallel_initial_screen(int niter) {

    int num_threads = std::min<int>(std::thread::hardware_concurrency(),
                                    model_->number_of_subordinate_models());
    ThreadWorkerPool pool(num_threads);
    std::vector<std::future<void>> futures;
    std::vector<std::vector<Selector>> draws;

    for (int i = 0; i < model_->number_of_subordinate_models(); ++i) {
      draws.push_back(std::vector<Selector>());
      std::vector<Selector> &worker_model_draws(draws.back());
      RegressionModel *worker_model = model_->subordinate_model(i);
      futures.emplace_back(
          pool.submit(
              [worker_model, niter, &worker_model_draws]() {
                for (int i = 0; i < niter; ++i) {
                  worker_model->sample_posterior();
                  worker_model_draws.push_back(worker_model->inc());
                }
              }));
    }
    for (auto & future : futures) {
      future.get();
    }
  }

}  // namespace BOOM
