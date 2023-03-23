/*
  Copyright (C) 2023 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/PosteriorSamplers/SharedSeasonalPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using SSPS = SharedSeasonalPosteriorSampler;

    void check_slabs(const std::vector<Ptr<MvnBase>> &slabs,
                     int nseries,
                     int state_dimension) {
      if (slabs.size() != nseries) {
        report_error("Number of slab priors does not match number of series.");
      }
      for (int i = 0; i < slabs.size(); ++i) {
        if (slabs[i]->dim() != state_dimension) {
          report_error("At least one slab prior expects the wrong state size.");
        }
      }
    }

    void check_spikes(const std::vector<Ptr<VariableSelectionPrior>> &spikes,
                      int nseries,
                      int state_dimension) {
      if (spikes.size() != nseries) {
        report_error("Number of spike priors does not match number of series.");
      }
      for (int i = 0; i < spikes.size(); ++i) {
        if (spikes[i]->potential_nvars() != state_dimension) {
          report_error("At least one spike prior expects the wrong state size.");
        }
      }
    }

    void build_samplers(std::vector<SpikeSlabSampler> &samplers,
                        const std::vector<Ptr<MvnBase>> &slabs,
                        const std::vector<Ptr<VariableSelectionPrior>> &spikes) {
      for (int i = 0; i < spikes.size(); ++i) {
        samplers.push_back(SpikeSlabSampler(nullptr, slabs[i], spikes[i]));
      }
    }

  }  // namespace

  //===========================================================================

  SSPS::SharedSeasonalPosteriorSampler(
      SharedSeasonalStateModel *model,
      const std::vector<Ptr<MvnBase>> &slabs,
      const std::vector<Ptr<VariableSelectionPrior>> &spikes,
      const std::vector<Ptr<UnivParams>> &sigsq,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slabs_(slabs),
        spikes_(spikes),
        sigsq_(sigsq)
  {
    check_spikes(spikes, model->nseries(), model->number_of_factors());
    check_slabs(slabs, model->nseries(), model->number_of_factors());
    build_samplers(samplers_, slabs_, spikes_);
  }

  void SSPS::draw() {
    for (int i = 0; i < model_->nseries(); ++i) {
      double sigsq = sigsq_[i]->value();
      Selector inc = model_->compressed_observation_coefficients(i)->inc();
      samplers_[i].draw_inclusion_indicators(
          rng(), inc, *model_->suf(i), sigsq);
      model_->compressed_observation_coefficients(i)->set_inc(inc);

      Vector full_beta = model_->compressed_observation_coefficients(i)->Beta();
      samplers_[i].draw_coefficients_given_inclusion(
          rng(), full_beta, inc, *model_->suf(i), sigsq);
      model_->compressed_observation_coefficients(i)->set_Beta(full_beta);
    }
  }

  double SSPS::logpri() const {
    double ans = 0;
    for (int i = 0; i < samplers_.size(); ++i) {
      ans += samplers_[i].log_prior(
          *model_->compressed_observation_coefficients(i));
    }
    return ans;
  }

}  // namespace BOOM
