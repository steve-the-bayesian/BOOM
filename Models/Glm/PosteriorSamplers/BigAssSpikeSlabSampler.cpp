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
      BigRegressionModel *model,
      const Ptr<VariableSelectionPrior> &spike,
      const Ptr<RegressionSlabPrior> &slab_prototype,
      const Ptr<GammaModelBase> &residual_precision_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        spike_(spike),
        slab_prototype_(slab_prototype),
        residual_precision_prior_(residual_precision_prior)
  {}

  void BigAssSpikeSlabSampler::draw() {
    ensure_restricted_model_sampler();
    model_->restricted_model()->sample_posterior();
    model_->expand_restricted_model_parameters();
  }

  double BigAssSpikeSlabSampler::logpri() const {
    return negative_infinity();
  }

  void BigAssSpikeSlabSampler::initial_screen(int niter, double threshold, bool use_threads) {
    assign_subordinate_samplers();
    run_parallel_initial_screen(niter, threshold, use_threads);
    set_candidate_variables();
  }

  ConstVectorView BigAssSpikeSlabSampler::select_chunk(
      const Vector &v, int chunk) const {
    int start = 0;
    if (chunk < 0 || chunk >= model_->number_of_subordinate_models()) {
      report_error("Chunk out of bounds.");
    }
    for (int m = 0; m < chunk; ++m) {
      const RegressionModel *worker = model_->subordinate_model(m);
      int end = start + worker->xdim();
      if (m > 0 and model_->force_intercept()) {
        --end;
      }
      start = end;
    }
    const RegressionModel *worker = model_->subordinate_model(chunk);
    int dim = worker->xdim();
    if (chunk > 0 and model_->force_intercept()) {
      --dim;
    }
    return ConstVectorView(v, start, dim);
  }

  void BigAssSpikeSlabSampler::assign_subordinate_samplers() {
    int number_of_workers = model_->number_of_subordinate_models();
    double sigma_guess = 1.0 / std::sqrt(residual_precision_prior_->mean());
    double sigma_df = residual_precision_prior_->alpha() / 2.0;

    for (int w = 0; w < number_of_workers; ++w) {
      RegressionModel *worker = model_->subordinate_model(w);
      Vector prior_mean(worker->xdim(), 0.0);
      prior_mean[0] = worker->suf()->ybar();
      SpdMatrix xtx = slab_prototype_->scale_xtx(
          worker->suf()->xtx(),
          worker->suf()->n(),
          slab_prototype_->diagonal_shrinkage());

      Vector prior_inclusion_probabilities = select_chunk(
          spike_->prior_inclusion_probabilities(), w);
      if (w > 0 and model_->force_intercept()) {
        prior_inclusion_probabilities = concat(
            1.0, prior_inclusion_probabilities);
      }

      NEW(BregVsSampler, worker_sampler)(
          worker,
          prior_mean,
          xtx,
          sigma_guess,
          sigma_df,
          prior_inclusion_probabilities,
          rng());
      worker->set_method(worker_sampler);
    }
  }

  void BigAssSpikeSlabSampler::set_candidate_variables() {
    Selector inc(model_->xdim(), false);
    // The model cursor points to the position in inc corresponding to position
    // 0 for the current model.
    long model_cursor = 0;
    bool force_intercept = model_->force_intercept();
    for (int m = 0; m < model_->number_of_subordinate_models(); ++m) {
      const RegressionModel &sub(*model_->subordinate_model(m));
      const Selector &sub_inc(sub.coef().inc());
      bool intercept_adjustment = force_intercept && m > 0;
      for (auto local_pos : sub_inc.included_positions()) {
        long global_pos = model_cursor + local_pos - intercept_adjustment;
        if (local_pos == 0 && intercept_adjustment) {
          // do nothing
        } else {
          inc.add(global_pos);
        }
      }
      model_cursor += sub.xdim() - intercept_adjustment;
    }
    model_->set_candidates(inc);
  }

  // Given a sequence of Selector objects, compute the fraction of time element
  // j is active among the population of selectors.
  Vector compute_inclusion_probabilities(
      const std::vector<Selector> &draws) {
    if (draws.empty()) {
      return Vector(0);
    }

    Vector ans(draws[0].nvars_possible(), 0.0);
    for (const auto &draw : draws) {
      ans += draw.to_Vector();
    }
    ans /= draws.size();
    return ans;
  }

  void BigAssSpikeSlabSampler::run_parallel_initial_screen(
      int niter, double threshold, bool use_threads) {

    int num_models = model_->number_of_subordinate_models();
    std::vector<std::vector<Selector>> draws(num_models);

    if (use_threads) {
      int num_threads = std::min<int>(std::thread::hardware_concurrency(),
                                      num_models);
      ThreadWorkerPool pool(num_threads);
      std::vector<std::future<void>> futures;


      for (int i = 0; i < num_models; ++i) {
        std::vector<Selector> &worker_model_draws(draws[i]);
        RegressionModel *worker_model = model_->subordinate_model(i);
        futures.emplace_back(
            pool.submit(
                [worker_model, niter, &worker_model_draws]() {
                  for (int iter = 0; iter < niter; ++iter) {
                    worker_model->sample_posterior();
                    worker_model_draws.push_back(worker_model->inc());
                  }
                }));
      }
      for (auto & future : futures) {
        future.get();
      }
    } else {
      // The non-thread code path should match the code in the thread code path
      // as closely as possible.
      for (int i = 0; i < num_models; ++i) {
        std::vector<Selector> &worker_model_draws(draws[i]);
        RegressionModel *worker_model = model_->subordinate_model(i);
        auto callback = [worker_model, niter, &worker_model_draws]() {
                          for (int iter = 0; iter < niter; ++iter) {
                            worker_model->sample_posterior();
                            worker_model_draws.push_back(worker_model->inc());
                          }
                        };
        callback();
      }
    }

    for (int i = 0; i < num_models; ++i) {
      RegressionModel *worker = model_->subordinate_model(i);
      worker->coef().drop_all();
      Vector inclusion_probabilities = compute_inclusion_probabilities(
          draws[i]);
      for (int column = 0; column < inclusion_probabilities.size(); ++column) {
        if (inclusion_probabilities[column] > threshold) {
          worker->coef().add(column);
        }
      }
    }
  }

  void BigAssSpikeSlabSampler::ensure_restricted_model_sampler() {
    RegressionModel *restricted_model = model_->restricted_model();
    if (!restricted_model) {
      report_error("Restricted model was not set.");
    }
    if (restricted_model->number_of_sampling_methods() == 0) {
      Vector prior_mean(restricted_model->xdim(), 0.0);
      prior_mean[0] = restricted_model->suf()->ybar();

      SpdMatrix modified_xtx = slab_prototype_->scale_xtx(
          restricted_model->xtx(),
          restricted_model->suf()->n(),
          slab_prototype_->diagonal_shrinkage());
      double sigma_guess = 1.0 / std::sqrt(residual_precision_prior_->mean());
      double sigma_df = residual_precision_prior_->alpha() / 2.0;
      Vector prior_inclusion_probabilities = model_->candidate_selector().select(
          spike_->prior_inclusion_probabilities());

      NEW(BregVsSampler, restricted_sampler)(
          restricted_model,
          prior_mean,
          modified_xtx,
          sigma_guess,
          sigma_df,
          prior_inclusion_probabilities,
          rng());

      restricted_model->set_method(restricted_sampler);
    }
  }

}  // namespace BOOM
